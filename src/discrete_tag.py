'''
    Xiang Li
    xli150@jhu.edu
'''

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from allen_parser import BiaffineDependencyParser
import torch.optim as optim
import torch.distributions.multivariate_normal as D
import pickle, random, time
from collections import defaultdict
import visual as vis
import sys
import math

'''
    This is a discrete set of tags. Analogous to the continuous case, we use a Gumble softmax 
    to sample and backpropagate. 
'''
EPS = 1e-12

class Discrete_Encoder(nn.Module):
    def __init__(self, options, word_dict, type_=False):
        super(Discrete_Encoder, self).__init__()

        self.hidden_dim = options.hidden_dim
        self.tag_dim = options.tag_dim
        self.tag_size = 1
        self.word_dict = word_dict
        self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu}
        self.activation = self.activations[options.activation]
        self.embedding_source = options.embedding_source
        if type_: self.embedding_source = 'elmo_type'
        self.device = options.device
        self.temperature = 5
        self.training = True
        # ============= Embeddings =======================
        
        if self.embedding_source == 'elmo_type':
            self.embedding_dim = 1024
        else:
            self.embedding_dim = options.embedding_dim

        # ============= Covariance matrix ================
        interm_layer_size = (self.embedding_dim + self.hidden_dim) // 2
        self.linear_layer = nn.Linear(self.embedding_dim, interm_layer_size )
        self.linear_layer3 = nn.Linear(interm_layer_size, self.hidden_dim)
        self.hidden2alpha =  nn.Linear(self.hidden_dim, self.tag_dim)

    def forward_sent_batch(self, sent, embeds):
        '''

        :param sent: the input sentence in string
        :param elmo_embeds: the elmo look up table.
        :param index: the index of the sentence in the elmo lookup table.
        :return: alphas -- the categorical distributions: all the entries sum up to 1.
        '''

        # ========== get the parameter of the softmax distribution ===================
        temps = self.activation(self.linear_layer(embeds))
        temps = self.activation(self.linear_layer3(temps))
        alphas = self.hidden2alpha(temps)
        alphas = nn.Softmax(dim=-1)(alphas)

        return alphas

    def get_statistics_batch(self, sent, elmo_embeds):
        # alpha is a discrete probability distribution.
        alphas = self.forward_sent_batch(sent, elmo_embeds)
        return alphas

    def get_elmo_embeds(self, sent, index=None):
        return self.elmo_embeds[index]

    def get_sample_from_param_batch(self, alpha, sample_size):

        if self.training:
            bsz, seqlen, tag_dim = alpha.shape 
            unif = torch.rand(bsz, sample_size, seqlen, tag_dim).to(self.device)
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)

            log_alpha = torch.log(alpha + EPS).unsqueeze(1).expand(-1, sample_size, -1, -1)
            logit = (log_alpha + gumbel) / self.temperature
            return nn.Softmax(dim=-1)(logit)
        else:
            # in reconstruction mode, pick the distribution over samples.
            if False:#self.distrib_eval: 
                ''' pick a distrib [for debugging] '''
                log_alpha = torch.log(alpha + EPS)
                return nn.Softmax(dim=-1) (log_alpha)
            else :
                ''' pick one best '''
                # In reconstruction mode, pick most likely sample
                _, max_alpha = torch.max(alpha, dim=-1)
                one_hot_samples = torch.zeros(alpha.size()).to(self.device)
                one_hot_samples.scatter_(-1, max_alpha.unsqueeze(-1).data, 1)
                return one_hot_samples



class Discrete_VIB(nn.Module):
    '''
        this is the primary class for this bottleneck model.
        enjoy and have fun !
    '''

    def __init__(self, options, word_dict):

        if options is None:
            return
        super(Discrete_VIB, self).__init__()

        # ===============Param setup===================
        self.beta = options.beta
        self.max_sent_len = options.max_sent_len
        self.tag_dim = options.tag_dim
        self.embedding_dim = options.embedding_dim
        self.hidden_dim = options.embedding_dim
        self.batch_size = options.batch_size
        self.sample_size = options.sample_size
        self.sample_method = options.sample_method
        self.tag_embedding_dim = self.tag_dim
        self.device = options.device
        # Annealing parameters.
        self.anneal_rate = 0.00005
        self.temperature = 5
        self.min_temp = 0.5
        self.min_inv_gamma = 0.1
        self.min_inv_beta = 0.1
        self.beta_annealing = False
        self.type_token_reg = (options.type_token_reg == 'yes')
        if self.type_token_reg:
            self.gamma = options.gamma
            self.gamma_annealing = False
        else:
            self.gamma=-1
            self.gamma_annealing = False

        # =============== Encoder Decoder setup ==================
        # Encoder_LinTrans
        self.encoder = Discrete_Encoder(options, word_dict)
        if self.type_token_reg: self.variational_encoder = Discrete_Encoder(options, word_dict, type_=True)
        self.decoder = BiaffineDependencyParser(options, self.tag_dim)

        self.r_var = self.tag_dim * self.max_sent_len

        # self.r_alphas =  nn.Parameter(torch.rand(self.r_var))
        self.r_alphas =  nn.Parameter(torch.rand(self.max_sent_len, self.tag_dim))
        self.tag_embeddings = nn.Linear(self.tag_dim, self.tag_embedding_dim)

        ## ================= Setup Optimizer ========================
        weight_decay = options.weight_decay
        self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=options.lr, weight_decay=weight_decay)

        opti_lst = list(self.decoder.parameters())
        opti_lst += [self.r_alphas]
        if self.type_token_reg:
            self.optimizer_var = optim.Adam(self.variational_encoder.parameters(), lr=options.lr, weight_decay=weight_decay)
        self.optimizer_decoder = optim.Adam(opti_lst, lr=options.lr, weight_decay=weight_decay)




    def kl_div(self, alpha1, alpha2):
        kl_loss = torch.sum(alpha1 * (torch.log(alpha1 + EPS) - torch.log(alpha2 + EPS)), dim=-1)
        return kl_loss



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



    def tag2embeddings(self, tags):
        embeds = self.tag_embeddings(tags)
        return embeds

    def forward_batch(self, sample_sentence, sample_tag = None, type_embeds=None, non_context_embeds=None):
        if sample_tag is None:
            sample_tag = self.sample_method
        result = 0
        x, y, y_label = sample_sentence
        alphas = self.encoder.get_statistics_batch(x, type_embeds)
        bsz, seqlen = x.shape
       
        if sample_tag == "identity":

            t = self.encoder.get_sample_from_param_batch(alphas, 1).reshape(bsz, seqlen, self.tag_dim)
            t = self.tag2embeddings(t)
            nlpy_t, err, corr, label_corr = self.compute_y_t_batch(t, y, y_label, x)

        elif sample_tag == 'iid':
            t = self.encoder.get_sample_from_param_batch(alphas, self.sample_size).reshape(bsz*self.sample_size, seqlen, self.tag_dim)
            t = self.tag2embeddings(t)

            # y = torch.repeat_interleave(y, self.sample_size, dim=0)
            # y_label = torch.repeat_interleave(y_label, self.sample_size, dim=0)
            y = y.unsqueeze(1).repeat(1, self.sample_size, 1).view(bsz * self.sample_size, seqlen)
            y_label = y_label.unsqueeze(1).repeat(1, self.sample_size, 1).view(bsz * self.sample_size, seqlen)
            nlpy_t, err, corr, label_corr = self.compute_y_t_batch(t, y, y_label, x)


        # kl_div = self.compute_diag_kl_batch(alphas)
        # r_alpha_temp = nn.LogSoftmax(dim=0)(self.r_alphas[start:end])
        r_alphas = nn.Softmax(dim=-1)(self.r_alphas[:seqlen, ]).unsqueeze(0).expand(bsz, -1, -1) # bsz, seqlen, tag_dim
        kl_div = self.kl_div(alphas, r_alphas)
        result = nlpy_t + self.beta * kl_div.mean()


        if self.type_token_reg:
            alphas2 = self.variational_encoder.get_statistics_batch(x, non_context_embeds)
            # kl_div2 = self._kl_discrete_loss_batch(alphas, alphas2)
            kl_div2 = self.kl_div(alphas, alphas2)
            result += self.gamma * kl_div2.mean()
        else:
            kl_div2 = torch.tensor([-1.])


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
                    self.forward_batch((x, y, y_label),sample_tag="identity", type_embeds=elmo_embeds_,
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

            if self.type_token_reg:
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



