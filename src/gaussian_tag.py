'''
    Xiang Li
    xli150@jhu.edu
'''

import numpy as np
from torch import nn
import torch
from allen_parser import BiaffineDependencyParser
import torch.optim as optim
import pickle

SMALL = 1e-08
class Continuous_Encoder(nn.Module):
    def __init__(self, options, word_dict, type_=False):
        super(Continuous_Encoder, self).__init__()

        self.device = options.device
        self.hidden_dim = options.hidden_dim
        self.tag_dim = options.tag_dim
        self.word_dict = word_dict
        self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu}
        self.activation = self.activations[options.activation]
        if type_: self.embedding_source = 'elmo_type'
        self.embedding_source = options.embedding_source
        
        # for type encoder
        if self.embedding_source == 'elmo_type':
            self.embedding_dim = 1024
        else:
            self.embedding_dim = options.embedding_dim

        # ============= Covariance matrix & Mean vector ================
        interm_layer_size = (self.embedding_dim + self.hidden_dim) // 2
        self.linear_layer = nn.Linear(self.embedding_dim, interm_layer_size )
        self.linear_layer3 = nn.Linear(interm_layer_size, self.hidden_dim)

        self.hidden2mean = nn.Linear(self.hidden_dim, self.tag_dim)
        self.hidden2std = nn.Linear(self.hidden_dim, self.tag_dim)

    def forward_sent(self, sent, elmo_embeds, index=None):

        ''' used for some evaluation scripts, not for training '''
        sent_len = len(sent)
        embeds = elmo_embeds[index]
        temps = self.activation(self.linear_layer(embeds))
        temps = self.activation(self.linear_layer3(temps))
        mean = self.hidden2mean(temps)
        std = self.hidden2std(temps)
        std = std.view(sent_len, 1, self.tag_dim)
        cov_lst = []
        cov_line = []
        cov_line = std.view(-1)
        cov_line = cov_line * cov_line + SMALL
        return mean, cov_lst, cov_line


    def forward_sent_batch(self, sent, embeds):

        temps = self.activation(self.linear_layer(embeds))
        temps = self.activation(self.linear_layer3(temps))
        mean = self.hidden2mean(temps) # bsz, seqlen, dim
        std = self.hidden2std(temps) # bsz, seqlen, dim
        cov = std * std + SMALL
        return mean, cov

    def get_sample_from_param_batch(self, mean, cov, sample_size):
        bsz, seqlen, tag_dim =  mean.shape
        z = torch.randn(bsz, sample_size, seqlen, tag_dim).to(self.device)
        z = z * torch.sqrt(cov).unsqueeze(1).expand(-1, sample_size, -1, -1) + \
            mean.unsqueeze(1).expand(-1, sample_size, -1, -1)
        return z.view(-1, seqlen, tag_dim)


    def get_statistics_batch(self, sent, elmo_embeds):
        mean, cov = self.forward_sent_batch(sent, elmo_embeds)
        return mean, cov


class Continuous_VIB(nn.Module):
    '''
        this is the primary class for this bottleneck model.
        enjoy and have fun !
    '''

    def __init__(self, options, word_dict):

        if options is None:
            return
        super(Continuous_VIB, self).__init__()

        # ===============Param setup===================
        self.beta = options.beta
        self.max_sent_len = options.max_sent_len
        self.tag_dim = options.tag_dim
        self.embedding_dim = options.embedding_dim
        self.hidden_dim = options.embedding_dim
        self.batch_size = options.batch_size
        self.sample_size = options.sample_size
        self.sample_method = options.sample_method

        self.device = options.device

        # Annealing parameters. currently set to FALSE
        self.anneal_rate = 0.0005
        self.temperature = 5
        self.min_temp = 0.5
        self.min_inv_gamma = 0.1
        self.min_inv_beta = 0.1
        self.beta_annealing = False
        self.gamma_annealing = False

        self.type_token_reg = (options.type_token_reg == 'yes')
        if self.type_token_reg:
            self.gamma = options.gamma
        else:
            self.gamma = -1

        # =============== Encoder Decoder setup ==================
        self.encoder = Continuous_Encoder(options, word_dict)
        self.decoder = BiaffineDependencyParser(options, self.tag_dim)
        ## TODO: could swap the decoder here.
        if self.type_token_reg:
            self.variational_encoder = Continuous_Encoder(options, word_dict, type_=True)

        self.r_var = self.tag_dim * self.max_sent_len
        self.r_mean = nn.Parameter(torch.randn(self.max_sent_len, self.tag_dim))
        self.r_std = nn.Parameter(torch.randn(self.max_sent_len, self.tag_dim))

        ## ================= Setup Optimizer ========================
        weight_decay = options.weight_decay
        self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=options.lr, weight_decay=weight_decay)
        opti_lst = list(self.decoder.parameters())
        opti_lst += [self.r_mean] + [self.r_std]
        if self.type_token_reg:
            self.optimizer_var = optim.Adam(self.variational_encoder.parameters(), lr=options.lr, weight_decay=weight_decay)
        self.optimizer_decoder = optim.Adam(opti_lst, lr=options.lr, weight_decay=weight_decay)


    def kl_div(self, param1, param2):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.
        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D)
        """
        mean1, cov1 = param1
        mean2, cov2 = param2
        bsz, seqlen, tag_dim = mean1.shape
        var_len = tag_dim * seqlen

        cov2_inv = 1 / cov2
        mean_diff = mean1 - mean2

        mean_diff = mean_diff.view(bsz, -1)
        cov1 = cov1.view(bsz, -1)
        cov2 = cov2.view(bsz, -1)
        cov2_inv = cov2_inv.view(bsz, -1)

        temp = (mean_diff * cov2_inv).view(bsz, 1, -1)
        KL = 0.5 * (torch.sum(torch.log(cov2), dim=1) - torch.sum(torch.log(cov1), dim=1) - var_len
                    + torch.sum(cov2_inv * cov1, dim=1) + torch.bmm(temp, mean_diff.view(bsz, -1, 1)).view(bsz))
        return KL

    ## TODO: could swap this function to be compatible with any decoders that return a log prob.
    def compute_y_t_batch(self, t, y, y_label, x):
        '''
            this is a variational bound. q(y|t)
            namely the decoder.
            This returns a log-probability, total_error_number, and the sentence length.
        '''
        out_dict = self.decoder.forward(x, t, head_indices=y, head_tags=y_label)
        heads = out_dict["heads"] # bsz, seqlen + 1
        head_tags = out_dict["head_tags"] # bsz, seqlen + 1

        corr = (heads[:, 1:] == y[:,:])
        corr_lab = (head_tags[:, 1:] == y_label)

        corr_L = (corr_lab & corr).sum(1).float().mean()
        corr = corr.sum(1).float().mean()
        err = (heads[:, 1:] - y[:, :] != 0).sum(1).float().mean()

        return out_dict["loss"], err, corr, corr_L


    def forward_batch(self, sample_sentence, sample_tag = None, type_embeds=None, non_context_embeds=None):
        if sample_tag is None:
            sample_tag = self.sample_method
        result = 0
        x, y, y_label = sample_sentence
        mean, cov = self.encoder.get_statistics_batch(x, type_embeds)
        bsz, seqlen = x.shape

        if sample_tag == "argmax":
            t = mean
            nlpy_t, err, corr, label_corr = self.compute_y_t_batch(t, y, y_label, x)

        elif sample_tag == 'iid':
            t = self.encoder.get_sample_from_param_batch(mean, cov, self.sample_size)
            y = y.unsqueeze(1).repeat(1, self.sample_size, 1).view(bsz * self.sample_size, seqlen)
            y_label = y_label.unsqueeze(1).repeat(1, self.sample_size, 1).view(bsz * self.sample_size, seqlen)
            nlpy_t, err, corr, label_corr = self.compute_y_t_batch(t, y, y_label, x)
        else:
            print('missing option for sample_tag, double check')


        mean_r = self.r_mean[:seqlen].unsqueeze(0).expand(bsz, -1, -1)
        std_r = self.r_std[:seqlen].unsqueeze(0).expand(bsz, -1, -1)
        cov_r = std_r * std_r + SMALL
        kl_div = self.kl_div((mean, cov), (mean_r, cov_r))
        result = nlpy_t + self.beta * kl_div.mean()

        if self.type_token_reg:
            mean2, cov2 = self.variational_encoder.get_statistics_batch(x, non_context_embeds)
            kl_div2 = self.kl_div((mean, cov), (mean2, cov2))
            result += self.gamma * kl_div2.mean()
        else:
            kl_div2 = torch.tensor([-1])

        return result, err/seqlen, nlpy_t.mean().item(), seqlen, self.sample_size, kl_div.mean().item(), \
               kl_div2.mean().item(), label_corr/seqlen


    def anneal_clustering(self, decrease_rate, tag='beta'):
        '''
        This function aims to do annealing and gradually do more compression, by tuning the gamma and beta
        to be larger. So, this is equivalent as annealing the inverse of the beta and gamma, and make them
        smaller, we decide that the lower limit of this annealing is when beta = 10, that is, inv_beta = 0.1

        :param decrease_rate:
        :param tag:
        :return:
        '''
        if tag == 'beta':
            inv_beta = 1/self.beta
            inv_beta = np.maximum(inv_beta * np.exp(-decrease_rate), self.min_inv_beta)
            self.beta = np.asscalar(1/inv_beta)
        elif tag == 'gamma':
            inv_gamma = 1 / self.gamma
            inv_gamma = np.maximum(inv_gamma * np.exp(-decrease_rate), self.min_inv_gamma)
            self.gamma = np.asscalar(1 / inv_gamma)

    def parse_dev_batch(self, corpus, elmo_embeds, non_context_embeds, out_path):
        self.encoder.training = False
        align_err_total, nlogp_total, word_total, sent_total, kl_total, kl_total2, label_LAS_total = 0, 0, 0, 0, 0, 0, 0
        batch_total, epoch_loss = 0, 0
        lst_las, lst_uas = [], []
        with torch.no_grad():
            for ind in range(len(corpus)):
                x, tag_, y, y_label = corpus[ind]
                bsz, seqlen = x.shape
                # GGG
                x = x.to(self.device)
                y = y.to(self.device)
                y_label = y_label.to(self.device)
                elmo_embeds_ = elmo_embeds[ind].to(self.device)
                non_context_embeds_ = non_context_embeds[ind].to(self.device)

                result, err_total, accuracy_loss, length_total, sample_total, kl_loss, kl_loss2, label_LAS = \
                    self.forward_batch((x, y, y_label), sample_tag='argmax', type_embeds=elmo_embeds_,
                                       non_context_embeds=non_context_embeds_)
                # average per batch, actually per token.
                align_err_total += err_total
                batch_total += 1
                nlogp_total += accuracy_loss
                label_LAS_total += label_LAS
                lst_las.append(label_LAS)
                lst_uas.append(1-err_total)
                kl_total += kl_loss

                # average per sentence.
                word_total += length_total * bsz
                sent_total += bsz
                kl_total2 += kl_loss2
                epoch_loss += result.item()

            avg_seqlen = word_total / sent_total
            align_err_w = align_err_total / batch_total
            nlogp_w = nlogp_total / batch_total
            align_err_s = align_err_w * avg_seqlen
            nlogp_s = nlogp_w * avg_seqlen
            kl_s = kl_total / batch_total
            kl_s2 = kl_total2 / batch_total
            kl_w = kl_s / avg_seqlen
            kl_w2 = kl_s2 / avg_seqlen
            LAS = label_LAS_total / batch_total

            print(
                'Total: totalLoss_per_sent=%f, NLL=%.3f, KL=%.3f, KL2=%.3f, UAS=%.3f LAS=%.3f, beta=%f, gamma=%f, temp=%f'
                % (epoch_loss / sent_total, nlogp_s, kl_s, kl_s2, 1 - align_err_w, LAS, self.beta, self.gamma,
                   self.temperature))

            result_dict = {}
            result_dict["align_err_w"] = align_err_w
            result_dict["nlogp_w"] = nlogp_w
            result_dict["align_err_s"] = align_err_s
            result_dict["nlogp_s"] = nlogp_s
            result_dict["kl_s"] = kl_s
            result_dict["kl_s2"] = kl_s2
            result_dict["kl_w"] = kl_w
            result_dict["kl_w2"] = kl_w2
            result_dict["LAS"] = LAS

            dict_dump = {"LAS": lst_las, "UAS": lst_uas}
            with open(out_path, 'wb') as f:
                pickle.dump(dict_dump, f)

        self.encoder.training = True
        return result_dict


    def train_batch(self, corpus, sent_per_epoch, elmo_embeds,
                        non_context_embeds, delta_temp=0.01 , tag=''):

        shuffledData = corpus
        shuffle_indices = np.random.choice(len(shuffledData), min(sent_per_epoch, len(shuffledData)), replace=False)
        epoch_loss = 0
        batch_total = 0

        align_err_total, nlogp_total, word_total, sent_total, kl_total, kl_total2, label_LAS_total = 0,0,0,0,0,0,0
        for iSentence, ind in enumerate(shuffle_indices):
            x, tag_, y, y_label = shuffledData[ind]

            bsz, seqlen = x.shape

            # GGG
            x = x.to(self.device)
            y = y.to(self.device)
            y_label = y_label.to(self.device)
            elmo_embeds_ = elmo_embeds[ind].to(self.device)
            non_context_embeds_ = non_context_embeds[ind].to(self.device)


            result, err_total, accuracy_loss, length_total, sample_total, kl_loss, kl_loss2, label_LAS = \
                self.forward_batch((x, y, y_label), type_embeds=elmo_embeds_, non_context_embeds=non_context_embeds_)
            # average per batch, actually per token.
            align_err_total += err_total
            batch_total += 1
            nlogp_total += accuracy_loss
            label_LAS_total += label_LAS
            kl_total += kl_loss

            # average per sentence.
            word_total += length_total * bsz
            sent_total += bsz
            kl_total2 += kl_loss2

            result.backward()
            self.optimizer_decoder.step()
            self.optimizer_encoder.step()
            self.optimizer_var.step()
            self.optimizer_var.zero_grad()
            self.optimizer_decoder.zero_grad()
            self.optimizer_encoder.zero_grad()
            epoch_loss += result.item()

            ''' min(args.kl_pen + args.delta_kl, 1) '''
            self.temperature = np.maximum(self.temperature - delta_temp, self.min_temp)
            # self.temperature = np.maximum(self.temperature * np.exp(-self.anneal_rate * self.batch_size),
            #                               self.min_temp)
            self.encoder.temperature = self.temperature

        avg_seqlen = word_total / sent_total
        align_err_w = align_err_total / batch_total
        nlogp_w = nlogp_total / batch_total
        align_err_s = align_err_w * avg_seqlen
        nlogp_s = nlogp_w * avg_seqlen
        kl_s = kl_total / batch_total
        kl_s2 = kl_total2 / batch_total
        kl_w = kl_s / avg_seqlen
        kl_w2 = kl_s2 / avg_seqlen
        LAS = label_LAS_total / batch_total

        print(
            'Total: totalLoss_per_sent=%f, NLL=%.3f, KL=%.3f, KL2=%.3f, UAS=%.3f LAS=%.3f, beta=%f, gamma=%f, temp=%f'
            % (epoch_loss / sent_total, nlogp_s, kl_s, kl_s2, 1 - align_err_w, LAS, self.beta, self.gamma,
               self.temperature))

        result_dict = {}
        result_dict["align_err_w"] = align_err_w
        result_dict["nlogp_w"] = nlogp_w
        result_dict["align_err_s"] = align_err_s
        result_dict["nlogp_s"] = nlogp_s
        result_dict["kl_s"] = kl_s
        result_dict["kl_s2"] = kl_s2
        result_dict["kl_w"] = kl_w
        result_dict["kl_w2"] = kl_w2

        result_dict["LAS"] = LAS

        return result_dict



    def save_model(self, path):
        self.cpu()
        torch.save(self.state_dict(), path)
        self.to(self.device)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.to(self.device)
        return self



