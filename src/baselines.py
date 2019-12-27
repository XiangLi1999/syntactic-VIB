from time import time
import numpy as np
import matplotlib.pyplot as plt
import sys, torch
from collections import defaultdict

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
import matplotlib.pyplot as plt
import conllu_handler
from termcolor import colored
import math


import gaussian_tag
from conllu_handler import Data_Loader, Embedding_Weight
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import re
import sys

################################## CLASS --- baseline parsers #####################################

class finetune_info_parse(nn.Module):
    def __init__(self, options):
        super(finetune_info_parse, self).__init__()
        self.tag_dim = options.embedding_dim
        self.elmo = options.embed_loader
        self.parser = BiaffineDependencyParser(options, self.tag_dim)
        self.lang = options.lang
        if self.lang == 'en':
            full_lst = list(self.parser.parameters()) + list(self.elmo.elmo.parameters())
        else:
            self.elmo.e.model.train()
            full_lst = list(self.parser.parameters()) + list(self.elmo.e.model.parameters())
        self.optimizer = optim.Adam(full_lst, weight_decay=options.weight_decay)
        self.batch_size = options.batch_size
        self.device = options.device


    def train(self, x_lst, data, sent_per_epoch):
        # compute_y_t_batch
        shuffledData = [(x, data_) for x,  data_ in zip(x_lst, data)]
        shuffle_indices = np.random.choice(len(shuffledData), min(sent_per_epoch, len(shuffledData)), replace=False)
        epoch_loss = 0
        align_err_total, nlogp_total, word_total, sent_total, LAS_total, UAS_total = 0, 0, 0, 0, 0, 0
        batch_total = 0
        for iSentence, ind in enumerate(shuffle_indices):
            x, temp_data = shuffledData[ind]
            t = self.elmo.get_part_elmo(x)
            _, _, y, y_label = temp_data
            bsz, seqlen = y.shape

            y = y.to(self.device)
            y_label = y_label.to(self.device)


            result, err_total, corr, corr_L = self.compute_y_t_batch(t, y, y_label, x)

            batch_total += 1
            epoch_loss += result.item()
            align_err_total += err_total /seqlen
            word_total += seqlen * bsz
            sent_total += bsz
            LAS_total += corr_L /seqlen
            UAS_total += corr /seqlen
            nlogp_total += result.item()

            result.backward()
            # torch.nn.utils.clip_grad_value_(self.param_lst, 5.)
            self.optimizer.step()
            self.optimizer.zero_grad()


        align_err_w = align_err_total / batch_total
        nlogp_w = nlogp_total / batch_total
        align_err_s = align_err_total / batch_total
        nlogp_s = nlogp_total / sent_total
        LAS = LAS_total / batch_total

        print('Total: totalLoss_per_sent=%.3f, NLL=%.3f, Err=%.3f, LAS=%.3f' % (
            epoch_loss / sent_total, nlogp_s, align_err_w, LAS))

        result_dict = {}
        result_dict["align_err_w"] = align_err_w
        result_dict["nlogp_w"] = nlogp_w
        result_dict["align_err_s"] = align_err_s
        result_dict["nlogp_s"] = nlogp_s
        '''
            note that the mutual information of discrete case variable is inf = Entropy
        '''
        result_dict["kl_s"] = -1
        result_dict["LAS"] = LAS

        return result_dict

    def parse_dev(self, x_lst, data):
        with torch.no_grad():
            self.parser.eval()
            self.elmo.elmo.eval()

            shuffledData = [(x, data_) for x, data_ in zip(x_lst, data)]
            shuffle_indices = list(range(len(shuffledData)))

            epoch_loss = 0
            align_err_total, nlogp_total, word_total, sent_total, LAS_total, UAS_total = 0, 0, 0, 0, 0, 0
            batch_total = 0

            for iSentence, ind in enumerate(shuffle_indices):
                x, temp_data = shuffledData[ind]
                t = self.elmo.get_part_elmo(x)
                _, _, y, y_label = temp_data
                bsz, seqlen = y.shape


                y = y.to(self.device)
                y_label = y_label.to(self.device)

                # print(y)

                result, err_total, corr, corr_L = self.compute_y_t_batch(t, y, y_label, x)

                batch_total += 1
                epoch_loss += result.item()
                align_err_total += err_total / seqlen
                word_total += seqlen * bsz
                sent_total += bsz
                LAS_total += corr_L / seqlen
                UAS_total += corr / seqlen
                nlogp_total += result.item()

            align_err_w = align_err_total / batch_total
            nlogp_w = nlogp_total / batch_total
            align_err_s = align_err_total / batch_total
            nlogp_s = nlogp_total / sent_total
            LAS = LAS_total / batch_total

            result_dict = {}
            result_dict["align_err_w"] = align_err_w
            result_dict["nlogp_w"] = nlogp_w
            result_dict["align_err_s"] = align_err_s
            result_dict["nlogp_s"] = nlogp_s
            result_dict["kl_s"] = -1
            result_dict["LAS"] = LAS

        self.parser.train()
        self.elmo.elmo.train()

        return result_dict

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        path2 = path+'_elmo'
        if self.lang != 'en':
            torch.save(self.elmo.e.model.state_dict(), path2)
        else:
            torch.save(self.elmo.elmo.state_dict(), path2)
        return

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        path2 = path + '_elmo'
        if self.lang != 'en':
            self.elmo.e.model.load_state_dict(torch.load(path2))
        else:
            self.elmo.elmo.load_state_dict(torch.load(path2))
        return self

    def compute_y_t_batch(self, t, y, y_label, x):
        '''
            this is a variational bound. q(y|t)
            namely the decoder.
            This returns a log-probability, total_error_number, and the sentence length.
        '''
        out_dict = self.parser.forward(x, t, head_indices=y, head_tags=y_label)
        heads = out_dict["heads"].long().cpu() # bsz, seqlen + 1
        head_tags = out_dict["head_tags"].long().cpu()  # bsz, seqlen + 1

        # print(heads[:,1:], y[:,:])
        corr = (heads[:, 1:] == y[:, :].cpu())
        corr_lab = (head_tags[:, 1:] == y_label.cpu())

        corr_L = (corr_lab & corr).sum(1).float().mean()
        corr = corr.sum(1).float().mean()
        err = (heads[:, 1:] - y[:, :].cpu() != 0).sum(1).float().mean()

        return out_dict["loss"], err, corr, corr_L


class POS_Parser(nn.Module):

    def __init__(self, tag_dict, options):
        super(POS_Parser, self).__init__()
        self.tag_dict = tag_dict
        self.tag_size = len(self.tag_dict)
        self.tag_dim = options.tag_dim
        self.tag_embedding = nn.Embedding(self.tag_size, self.tag_dim)
        self.parser = BiaffineDependencyParser(options, self.tag_dim)
        optim_lst = list(self.parser.parameters()) + list(self.tag_embedding.parameters())
        self.optimizer = optim.Adam(optim_lst, weight_decay=options.weight_decay)
        self.batch_size = options.batch_size
        self.device = options.device

    def compute_y_t_batch(self, t, y, y_label, x):
        '''
            this is a variational bound. q(y|t)
            namely the decoder.
            This returns a log-probability, total_error_number, and the sentence length.
        '''
        out_dict = self.parser.forward(x, t, head_indices=y, head_tags=y_label)
        heads = out_dict["heads"] # bsz, seqlen + 1
        head_tags = out_dict["head_tags"] # bsz, seqlen + 1

        corr = (heads[:, 1:] == y[:,:])
        corr_lab = (head_tags[:, 1:] == y_label)

        corr_L = (corr_lab & corr).sum(1).float().mean()
        corr = corr.sum(1).float().mean()
        err = (heads[:, 1:] - y[:, :] != 0).sum(1).float().mean()

        return out_dict["loss"], err, corr, corr_L


    def forward_batch(self, sample_sentence, tags=None):
        x, y, y_label = sample_sentence
        bsz, seqlen = x.shape

        embeds = self.tag_embedding(tags)

        nlpy_t, err, corr, label_corr = self.compute_y_t_batch(embeds, y, y_label, x)

        return nlpy_t, err/seqlen, nlpy_t.mean().item(), seqlen, 1, label_corr/seqlen

    def parse_dev_batch(self, corpus, out_path):
        align_err_total, nlogp_total, word_total, sent_total, label_LAS_total = 0, 0, 0, 0, 0
        batch_total, epoch_loss = 0, 0
        lst_las, lst_uas = [], []
        with torch.no_grad():
            for ind in range(len(corpus)):
                x, tag_, y, y_label = corpus[ind]
                bsz, seqlen = x.shape
                # GGG
                x = x.to(self.device)
                y = y.to(self.device)
                tag_ = tag_.to(self.device)
                y_label = y_label.to(self.device)


                result, err_total, accuracy_loss, length_total, sample_total, label_LAS = \
                    self.forward_batch((x, y, y_label), tags=tag_)

                # average per batch, actually per token.
                align_err_total += err_total
                batch_total += 1
                nlogp_total += accuracy_loss
                label_LAS_total += label_LAS
                lst_las.append(label_LAS)
                lst_uas.append(1-err_total)

                # average per sentence.
                word_total += length_total * bsz
                sent_total += bsz
                epoch_loss += result.item()

            avg_seqlen = word_total / sent_total
            align_err_w = align_err_total / batch_total
            nlogp_w = nlogp_total / batch_total
            align_err_s = align_err_w * avg_seqlen
            nlogp_s = nlogp_w * avg_seqlen
            LAS = label_LAS_total / batch_total

            result_dict = {}
            result_dict["align_err_w"] = align_err_w
            result_dict["nlogp_w"] = nlogp_w
            result_dict["align_err_s"] = align_err_s
            result_dict["nlogp_s"] = nlogp_s
            result_dict["LAS"] = LAS
            result_dict["kl_w"] = -1
            result_dict["kl_w2"] = -1

            dict_dump = {"LAS": lst_las, "UAS": lst_uas}
            with open(out_path, 'wb') as f:
                pickle.dump(dict_dump, f)

        return result_dict


    def train_batch(self, corpus):

        shuffledData = corpus

        shuffle_indices = np.random.choice(len(shuffledData), len(shuffledData), replace=False)
        epoch_loss = 0
        batch_total = 0


        align_err_total, nlogp_total, word_total, sent_total, label_LAS_total = 0,0,0,0,0
        for iSentence, ind in enumerate(shuffle_indices):
            x, tag_, y, y_label = shuffledData[ind]

            bsz, seqlen = x.shape

            x = x.to(self.device)
            y = y.to(self.device)
            tag_ = tag_.to(self.device)
            y_label = y_label.to(self.device)


            result, err_total, accuracy_loss, length_total, sample_total, label_LAS = \
                self.forward_batch((x, y, y_label), tags=tag_)

            align_err_total += err_total
            batch_total += 1
            nlogp_total += accuracy_loss
            label_LAS_total += label_LAS
            word_total += length_total * bsz
            sent_total += bsz

            result.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += result.item()

        avg_seqlen = word_total / sent_total
        align_err_w = align_err_total / batch_total
        nlogp_w = nlogp_total / batch_total
        align_err_s = align_err_w * avg_seqlen
        nlogp_s = nlogp_w * avg_seqlen
        LAS = label_LAS_total / batch_total


        result_dict = {}
        result_dict["align_err_w"] = align_err_w # 1-uas
        result_dict["nlogp_w"] = nlogp_w
        result_dict["align_err_s"] = align_err_s 
        result_dict["nlogp_s"] = nlogp_s
        result_dict["LAS"] = LAS
        result_dict["kl_w"] = -1
        result_dict["kl_w2"] = -1
        return result_dict

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        return

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self



class baseline_cont_parse(nn.Module):
    def __init__(self, options):
        super(baseline_cont_parse, self).__init__()
        self.embedding_dim = options.embedding_dim
        self.tag_dim = options.tag_dim
        self.parser = BiaffineDependencyParser(options, self.tag_dim)
        self.batch_size = options.batch_size
        self.device = options.device
        if options.task == "CLEAN":
            print('clean up layer before LSTM')
            self.clean_ = True
        else:
            self.clean_ = False
        if self.clean_:
            middle_size = (self.embedding_dim + self.tag_dim) // 2 
            self.clean_layer = nn.Sequential(nn.Linear(self.embedding_dim, middle_size), nn.Tanh(), nn.Linear(middle_size, self.tag_dim))


        self.optimizer = optim.Adam(self.parameters(), weight_decay=options.weight_decay, lr=options.lr)

    def compute_y_t_batch(self, t, y, y_label, x):
        '''
            this is a variational bound. q(y|t)
            namely the decoder.
            This returns a log-probability, total_error_number, and the sentence length.
        '''
        out_dict = self.parser.forward(x, t, head_indices=y, head_tags=y_label)
        heads = out_dict["heads"] # bsz, seqlen + 1
        head_tags = out_dict["head_tags"] # bsz, seqlen + 1

        corr = (heads[:, 1:] == y[:,:])
        corr_lab = (head_tags[:, 1:] == y_label)

        corr_L = (corr_lab & corr).sum(1).float().mean()
        corr = corr.sum(1).float().mean()
        err = (heads[:, 1:] - y[:, :] != 0).sum(1).float().mean()

        return out_dict["loss"], err, corr, corr_L


    def forward_batch(self, sample_sentence, embeds=None):
        x, y, y_label = sample_sentence
        bsz, seqlen = x.shape

        if self.clean_:
            embeds = self.clean_layer(embeds)

        nlpy_t, err, corr, label_corr = self.compute_y_t_batch(embeds, y, y_label, x)

        return nlpy_t, err/seqlen, nlpy_t.mean().item(), seqlen, 1, label_corr/seqlen

    def parse_dev_batch(self, corpus, elmo_embeds, out_path):
        align_err_total, nlogp_total, word_total, sent_total, label_LAS_total = 0, 0, 0, 0, 0
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

                result, err_total, accuracy_loss, length_total, sample_total, label_LAS = \
                    self.forward_batch((x, y, y_label), embeds=elmo_embeds_)
                # average per batch, actually per token.
                align_err_total += err_total
                batch_total += 1
                nlogp_total += accuracy_loss
                label_LAS_total += label_LAS
                lst_las.append(label_LAS)
                lst_uas.append(1-err_total)

                # average per sentence.
                word_total += length_total * bsz
                sent_total += bsz
                epoch_loss += result.item()

            avg_seqlen = word_total / sent_total
            align_err_w = align_err_total / batch_total
            nlogp_w = nlogp_total / batch_total
            align_err_s = align_err_w * avg_seqlen
            nlogp_s = nlogp_w * avg_seqlen
            LAS = label_LAS_total / batch_total

            result_dict = {}
            result_dict["align_err_w"] = align_err_w
            result_dict["nlogp_w"] = nlogp_w
            result_dict["align_err_s"] = align_err_s
            result_dict["nlogp_s"] = nlogp_s
            result_dict["LAS"] = LAS
            result_dict["kl_w"] = -1
            result_dict["kl_w2"] = -1

            dict_dump = {"LAS": lst_las, "UAS": lst_uas}
            with open(out_path, 'wb') as f:
                pickle.dump(dict_dump, f)

        return result_dict


    def train_batch(self, corpus, sent_per_epoch, elmo_embeds):

        shuffledData = corpus

        if sent_per_epoch is None:
            sent_per_epoch = len(shuffledData)

        shuffle_indices = np.random.choice(len(shuffledData), min(sent_per_epoch, len(shuffledData)), replace=False)
        epoch_loss = 0
        batch_total = 0


        align_err_total, nlogp_total, word_total, sent_total, label_LAS_total = 0,0,0,0,0
        for iSentence, ind in enumerate(shuffle_indices):
            x, tag_, y, y_label = shuffledData[ind]

            bsz, seqlen = x.shape

            x = x.to(self.device)
            y = y.to(self.device)
            y_label = y_label.to(self.device)
            elmo_embeds_ = elmo_embeds[ind].to(self.device)


            result, err_total, accuracy_loss, length_total, sample_total, label_LAS = \
                self.forward_batch((x, y, y_label), embeds=elmo_embeds_)

            align_err_total += err_total
            batch_total += 1
            nlogp_total += accuracy_loss
            label_LAS_total += label_LAS
            word_total += length_total * bsz
            sent_total += bsz

            result.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += result.item()

        avg_seqlen = word_total / sent_total
        align_err_w = align_err_total / batch_total
        nlogp_w = nlogp_total / batch_total
        align_err_s = align_err_w * avg_seqlen
        nlogp_s = nlogp_w * avg_seqlen
        LAS = label_LAS_total / batch_total


        result_dict = {}
        result_dict["align_err_w"] = align_err_w # 1-uas
        result_dict["nlogp_w"] = nlogp_w
        result_dict["align_err_s"] = align_err_s 
        result_dict["nlogp_s"] = nlogp_s
        result_dict["LAS"] = LAS
        result_dict["kl_w"] = -1
        result_dict["kl_w2"] = -1
        return result_dict

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        return

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self




################################## CLASS --- evaluations #####################################
# Analysis 2
class Eval_predict_pos(nn.Module):
    def __init__(self, word_lst, pos_dict, model, embed_loader, options):
        super(Eval_predict_pos, self).__init__()
        print(len(list(self.parameters())))
        self.tag_dim = model.encoder.tag_dim
        self.model = model
        self.batch_size = 10
        self.pos_dim = len(pos_dict)
        self.hidden_size1 = self.tag_dim//2
        self.hidden_size2 = self.hidden_size1//2
        self.mode = 'nonlinear'
        print(len(list(self.parameters())))
        if self.mode == 'nonlinear':
            print('layer1dim={}, layer2dim={}'.format(self.hidden_size1, self.hidden_size2))
            self.linear1 = nn.Linear(self.tag_dim, self.hidden_size1)
            self.linear_mid = nn.Linear(self.hidden_size1, self.hidden_size2)
            self.linear2 = nn.Linear(self.hidden_size2, self.pos_dim)
            self.activation = torch.tanh
            self.forward = self.forward_nonlinear
        elif self.mode == 'linear':
            self.transition = nn.Linear(self.tag_dim, self.pos_dim)
            self.forward = self.forward_linear
            print(len(list(self.parameters())))
        else:
            self.transition_matrix = nn.Parameter(torch.ones(self.pos_dim, self.tag_dim))
            self.forward = self.forward_matrix
        self.weight_decay = options.weight_decay
        # self.get_model_output_for_words(model, word_lst, embed_loader)
        self.criterion = nn.CrossEntropyLoss()
        print(len(list(self.parameters())))
        print('end enumerating the params ')
        # self.optimizer = optim.Adam(self.parameters(), weight_decay=0)
        lst_param  = []
        lst_param += list(self.linear1.parameters())
        lst_param += list(self.linear_mid.parameters())
        lst_param += list(self.linear2.parameters())

        self.optimizer = optim.Adam(lst_param, weight_decay=options.weight_decay, lr=options.lr)

    def get_model_output_for_words(self, model, word_lst, embed_loader):
        dict_word2compress = {}
        lst_word = []
        for ind, word in enumerate(word_lst):
            lst_word += [x for x in word]
        word_set = list(set(lst_word))
        elmo_embeds = embed_loader.elmo_embeddings_first([word_set], 1)
        # LISA
        mean, _, _ = model.encoder.forward_sent(word_set, elmo_embeds, 0)
        print(len(mean))
        print(mean[0].shape)
        mean = elmo_embeds[0]
        print(len(mean))
        print(mean[0].shape)
        for index, elem in enumerate(word_set):
            dict_word2compress[elem] = mean[index].data
        return dict_word2compress

    def forward_nonlinear(self, sent_compress):
        '''
        This function batch processes the sentence together.
        :param sent: the compressed version of the sentence.
        :return:
        '''
        # sent_compress = torch.cat([self.dict_word2compress[ww] for ww in sent], dim=0).unsqueeze(1)
        out = self.linear_mid(self.activation(self.linear1(sent_compress)))
        out = self.linear2(self.activation(out))
        return nn.LogSoftmax(dim=-1)(out)

    def forward_linear(self, sent_compress):
        temp = self.transition(sent_compress)
        return nn.LogSoftmax(dim=-1)(temp)

    def forward_matrix(self, sent_compress):
        return torch.mm(self.transition_matrix, sent_compress)


    def train_discrete(self, word_lst, pos_lst, elmo_train):
        batch_loss = 0
        total_loss = 0
        total_tokens = 0
        total_err = 0
        total_cond_entr = 0
        tag_distrib = defaultdict(float)
        for index, cont in enumerate(zip(word_lst, pos_lst)):
            words, poss = cont
            sent_compress = self.model.encoder.forward_sent(words, elmo_train, index).data
            distrib = self.forward(sent_compress).squeeze(1)
            # print(distrib)
            # print(poss)
            loss = self.criterion(distrib, poss)
            # print(loss)
            _, pred = torch.max(distrib, dim=1)
            err = np.sum([1 if pp != tt else 0 for pp, tt in zip(pred, poss)])
            batch_loss += loss
            total_loss += loss.item()
            total_err += err
            # print(torch.exp(distrib))
            # print(torch.exp(distrib) * distrib)
            cond_entr = -torch.sum(torch.exp(distrib) * distrib).item()
            # print(cond_entr)
            total_cond_entr += cond_entr
            total_tokens += len(words)
            if (index + 1) % self.batch_size == 0:
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                batch_loss = 0

            for elem in poss:
                tag_distrib[elem.item()] += 1

        if (index + 1) % self.batch_size != 0:
            batch_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        H_tag = 0
        for elem in tag_distrib:
            prob = tag_distrib[elem] / total_tokens
            H_tag -= prob * np.log(prob)

        ''' compute H(POS | TAG) '''
        # this is about a conditional entropy quantity.
        # H(POS | TAG) = p(POS | TAG) * log p(POS | TAG)
        cond_entr = total_cond_entr/total_tokens
        ''' compute MI (POS, TAG) '''
        mi = H_tag - cond_entr
        # print('the total loss is %.5f' % total_loss)

        return 'avg_loss=%.5f, err_rate=%.5f, H(POS|Tag)=%.5f, H(POS)=%.5f, MI(POS,Tag)=%.5f '\
               %(total_loss/total_tokens, total_err/total_tokens,
                cond_entr, H_tag, mi)


    def fewshot_filtering_proportional(self, word_lst, pos_lst, elmo_embeds, k):
        '''
        In this function, we will pick k = 500 and see the result for thr further results.
        :param word_lst:
        :param pos_lst:
        :return:
        '''
        book = {}
        pos_locs = {}
        pos_count = defaultdict(int)
        total_count = 0
        for index, cont, in enumerate(zip(word_lst, pos_lst)):
            words, poss = cont
            for ind, (word, pos_) in enumerate(zip(words, poss)):
                book[total_count] = (index, ind)
                pos = pos_.item()
                pos_count[pos] += 1
                if pos not in pos_locs:
                    pos_locs[pos] = [total_count]
                else:
                    pos_locs[pos] += [total_count]
                total_count += 1

        ''' get the proportion distribution of the POS tags '''
        for pos, count in pos_count.items():
            pos_count[pos] = count / total_count

        word_lst_x = []
        pos_lst_x = []
        elmo_embeds_x = []
        for elem in pos_locs:
            k_cand = int(k * len(pos_count) * pos_count[elem])
            shuffle_indices = np.random.choice(len(pos_locs[elem]), k_cand, replace=True)
            true_index = [pos_locs[elem][x] for x in shuffle_indices]
            for elem in true_index:
                index, ind = book[elem]
                word_lst_x.append([word_lst[index][ind]])
                pos_lst_x.append(pos_lst[index][ind:ind+1])
                elmo_embeds_x.append(elmo_embeds[index][ind:ind+1])
        return word_lst_x, pos_lst_x, elmo_embeds_x


    def fewshot_filtering(self, word_lst, pos_lst, elmo_embeds, k):
        '''
        In this function, we will pick k = 50, number for each POS tags to train.
        :param word_lst:
        :param pos_lst:
        :return:
        '''
        book = {}
        pos_locs = {}
        total_count = 0
        for index, cont, in enumerate(zip(word_lst, pos_lst)):
            words, poss = cont
            for ind, (word, pos_) in enumerate(zip(words, poss)):
                book[total_count] = (index, ind)
                pos = pos_.item()
                if pos not in pos_locs:
                    pos_locs[pos] = [total_count]
                else:
                    pos_locs[pos] += [total_count]

                total_count += 1

        word_lst_x = []
        pos_lst_x = []
        elmo_embeds_x = []
        for elem in pos_locs:
            shuffle_indices = np.random.choice(len(pos_locs[elem]), k, replace=True)
            true_index = [pos_locs[elem][x] for x in shuffle_indices]
            for elem in true_index:
                index, ind = book[elem]
                word_lst_x.append([word_lst[index][ind]])
                pos_lst_x.append(pos_lst[index][ind:ind+1])
                elmo_embeds_x.append(elmo_embeds[index][ind:ind+1])
        return word_lst_x, pos_lst_x, elmo_embeds_x


    def eval_dev_discrete(self, word_lst, pos_lst, elmo_train):
        with torch.no_grad():
            total_loss = 0
            total_tokens = 0
            total_err = 0
            total_cond_entr = 0
            tag_distrib = defaultdict(float)
            for index, cont in enumerate(zip(word_lst, pos_lst)):
                words, poss = cont
                sent_compress = self.model.encoder.forward_sent(words, elmo_train, index).data
                distrib = self.forward(sent_compress).squeeze(1)
                _, pred = torch.max(distrib, dim=1)
                loss = self.criterion(distrib, poss)
                err = np.sum([1 if pp != tt else 0 for pp, tt in zip(pred, poss)])
                total_loss += loss.item()
                total_err += err
                total_tokens += len(words)
                cond_entr = -torch.sum(torch.exp(distrib) * distrib).item()
                total_cond_entr += cond_entr
                for elem in poss:
                    tag_distrib[elem.item()] += 1

        H_tag = 0
        for elem in tag_distrib:
            prob = tag_distrib[elem] / total_tokens
            H_tag -= prob * np.log(prob)

        ''' compute H(POS | TAG) '''
        # this is about a conditional entropy quantity.
        # H(POS | TAG) = p(POS | TAG) * log p(POS | TAG)
        cond_entr = total_cond_entr/total_tokens
        ''' compute MI (POS, TAG) '''
        mi = H_tag - cond_entr
        # print('the total loss is %.5f' % total_loss)


        dict_result = {"H(POS)":H_tag,
                       "MI(POS,Tag)":mi,
                       "H(POS|Tag)":cond_entr,
                       "error":total_err/total_tokens}

        # return 'avg_loss=%.5f, err_rate=%.5f, H(POS|Tag)=%.5f, H(POS)=%.5f, MI(POS,Tag)=%.5f '\
        #        %(total_loss/total_tokens, total_err/total_tokens,
        #         total_loss/total_tokens, H_tag, H_tag-total_loss/total_tokens), dict_result
        return 'avg_loss=%.5f, err_rate=%.5f, H(POS|Tag)=%.5f, H(POS)=%.5f, MI(POS,Tag)=%.5f '\
               %(total_loss/total_tokens, total_err/total_tokens,
                cond_entr, H_tag, mi), dict_result



    def save_model(self, path):
        torch.save(self.state_dict(), path)

        return

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self





# analysis 4
class Recon_Lemma(nn.Module):
    def __init__(self,  stem_dict, model, options):
        super(Recon_Lemma, self).__init__()

        if options.inp == 'elmo':
            self.prepare = self.prepare_elmo
        elif options.inp == 'mean':
            self.prepare = self.prepare_mean
        elif options.inp == 'sample':
            self.prepare = self.prepare_sample
            self.sample_size = options.sample_size
            print('sample size is ', self.sample_size)


        self.tag_dim = model.encoder.tag_dim_tsne_token5
        if options.inp == 'elmo':
            if options.embedding_source in ['elmo_2', 'elmo_3']:
                self.tag_dim = 1024
            else:
                self.tag_dim = 512
        self.model = model
        self.batch_size = 5
        self.stem_dim = len(stem_dict)
        self.hidden_size1 = self.tag_dim//2
        self.hidden_size2 = 1024
        self.mode = 'nonlinear'
        self.sample_size = options.sample_size
        print(len(list(self.parameters())))
        if self.mode == 'nonlinear':
            self.linear1 = nn.Linear(self.tag_dim, self.tag_dim)
            self.linear_mid = nn.Linear(self.tag_dim, self.hidden_size2)
            self.linear2 = nn.Linear(self.hidden_size2, self.stem_dim)
            self.activation = torch.tanh
            self.forward = self.forward_nonlinear
        elif self.mode == 'linear':
            self.transition = nn.Linear(self.tag_dim, self.pos_dim)
            self.forward = self.forward_linear
            print(len(list(self.parameters())))
        else:
            self.transition_matrix = nn.Parameter(torch.ones(self.pos_dim, self.tag_dim))
            self.forward = self.forward_matrix
        self.weight_decay = options.weight_decay
        # self.get_model_output_for_words(model, word_lst, embed_loader)
        self.criterion = nn.CrossEntropyLoss()

        if options.task == 'VIB_discrete':
            self.train = self.train_discrete
            self.eval_dev = self.eval_dev_discrete

        elif options.task == 'VIB':
            self.train = self.train_continuous
            self.eval_dev = self.eval_dev_continuous

        # self.optimizer = optim.Adam(self.parameters(), weight_decay=0)
        lst = list(self.linear1.parameters()) + list(self.linear_mid.parameters()) + list(self.linear2.parameters())
        self.optimizer = optim.Adam(lst, weight_decay=options.weight_decay, lr=0.001)


    def forward_nonlinear(self, sent_compress):
        '''
        This function batch processes the sentence together.
        :param sent: the compressed version of the sentence.
        :return:
        '''
        out = self.linear_mid(self.activation(self.linear1(sent_compress)))
        out = self.linear2(self.activation(out))
        return out

    def forward_linear(self, sent_compress):
        temp = self.transition(sent_compress)
        return nn.LogSoftmax(dim=-1)(temp)

    def forward_matrix(self, sent_compress):
        return torch.mm(self.transition_matrix, sent_compress)


    def train_discrete(self, word_lst, pos_lst, elmo_train,  sent_per_epoch=200):
        shuffle_indices = np.random.choice(len(word_lst), min(sent_per_epoch, len(word_lst)), replace=False)
        batch_loss = 0
        total_loss = 0
        total_tokens = 0
        total_err = 0
        for index, ind in enumerate(shuffle_indices):
            words = word_lst[ind]
            poss = pos_lst[ind]
            sent_compress = self.model.encoder.forward_sent(words, elmo_train, ind).data
            distrib = self.forward(sent_compress).squeeze(1)
            loss = self.criterion(distrib, poss)
            # print(loss)
            _, pred = torch.max(distrib, dim=1)
            err = np.sum([1 if pp != tt else 0 for pp, tt in zip(pred, poss)])
            batch_loss += loss
            total_loss += loss.item()
            total_err += err
            total_tokens += len(words)
            if (index + 1) % self.batch_size == 0:
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                batch_loss = 0

        if (index + 1) % self.batch_size != 0:
            batch_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


        return 'avg_loss=%.5f, err_rate=%.5f '\
               %(total_loss/total_tokens, total_err/total_tokens)

    def prepare_mean(self, words, elmo_train, ind):
        sent_compress, _, _ = self.model.encoder.forward_sent(words, elmo_train, ind)
        yield sent_compress

    def prepare_sample(self, x, type_embeds, index):
        mean, cov_lst, cho_lst = self.model.encoder.get_statistics(x, type_embeds, index)
        for i in range(self.sample_size):
            t = self.model.encoder.get_sample_from_param(mean, cov_lst, cho_lst)
            yield t

    def prepare_elmo(self, words, elmo_train, ind ):
        yield elmo_train[ind]


    def train_continuous(self, word_lst, pos_lst, elmo_train,  sent_per_epoch=200):
        '''

        :param word_lst: a list of tokens in a sentence
        :param pos_lst: a list of target stem tensor that's in the format of LongTensor
        :param model_output: a dict that maps word to its compression
        :return:
        '''
        shuffle_indices = np.random.choice(len(word_lst), min(sent_per_epoch, len(word_lst)), replace=False)
        batch_loss = 0
        total_loss = 0
        total_err = 0
        total_tokens = 0
        for index, ind in enumerate(shuffle_indices):
            words = word_lst[ind]
            poss = pos_lst[ind]

            for sent_compress in self.prepare( words, elmo_train, ind ):
                distrib = self.forward(sent_compress).squeeze(1)
                loss = self.criterion(distrib, poss)
                _, pred = torch.max(distrib, dim=1)
                err = np.sum([1 if pp != tt else 0 for pp, tt in zip(pred, poss)])
                batch_loss += loss
                total_loss += loss.item()
                total_err += err
                total_tokens += len(words)
            if (index + 1) % self.batch_size == 0:
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                batch_loss = 0
        if (index + 1) % self.batch_size != 0:
            batch_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return 'avg_loss=%.5f, err_rate=%.5f ' \
               % (total_loss / total_tokens, total_err / total_tokens)


    def eval_dev_discrete(self, word_lst, pos_lst, elmo_train):

        with torch.no_grad():
            total_loss = 0
            total_tokens = 0
            total_err = 0
            for index, cont in enumerate(zip(word_lst, pos_lst)):
                words, poss = cont
                sent_compress = self.model.encoder.forward_sent(words, elmo_train, index).data
                distrib = self.forward(sent_compress).squeeze(1)
                _, pred = torch.max(distrib, dim=1)
                loss = self.criterion(distrib, poss)
                err = np.sum([1 if pp != tt else 0 for pp, tt in zip(pred, poss)])
                total_loss += loss.item()
                total_err += err
                total_tokens += len(words)

        dict_result = {"NLL":total_loss/total_tokens,
                       "error":total_err/total_tokens}

        # return 'avg_loss=%.5f, err_rate=%.5f, H(POS|Tag)=%.5f, H(POS)=%.5f, MI(POS,Tag)=%.5f '\
        #        %(total_loss/total_tokens, total_err/total_tokens,
        #         total_loss/total_tokens, H_tag, H_tag-total_loss/total_tokens), dict_result
        return 'avg_loss=%.5f, err_rate=%.5f'\
               %(total_loss/total_tokens, total_err/total_tokens), dict_result


    def eval_dev_continuous(self, word_lst, pos_lst, elmo_train):
        with torch.no_grad():
            total_loss = 0
            total_tokens = 0
            total_err = 0
            for index, cont in enumerate(zip(word_lst, pos_lst)):
                words, poss = cont
                for sent_compress in self.prepare(words, elmo_train, index):
                    distrib = self.forward(sent_compress).squeeze(1)
                    _, pred = torch.max(distrib, dim=1)
                    loss = self.criterion(distrib, poss)
                    err = np.sum([1 if pp != tt else 0 for pp, tt in zip(pred, poss)])
                    total_loss += loss.item()
                    total_err += err
                    total_tokens += len(words)

        dict_result = {"NLL":total_loss/total_tokens,
                       "error":total_err/total_tokens}

        return 'avg_loss=%.5f, err_rate=%.5f '\
               %(total_loss/total_tokens, total_err/total_tokens), dict_result


    def save_model(self, path):
        torch.save(self.state_dict(), path)
        return

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self



#======================================= HELPER FUNCTIONS ============================================
def get_word_pos_dict(word_lst, pos_lst):
    total_tag_p = defaultdict(int)
    word_tag = defaultdict(lambda: defaultdict(int))
    word_pos_dict = {}
    for sent, tag_seq in zip(word_lst, pos_lst):
        for ww, pp in zip(sent, tag_seq):
            word_tag[ww][pp] += 1
            total_tag_p[pp] += 1

    for word in word_tag:
        max_pos = max(word_tag[word], key=lambda x:word_tag[word][x])
        word_pos_dict[word] = max_pos
    return word_pos_dict

def get_word_pos_dict_thre(word_lst, pos_lst, threshold):
    total_tag_p = defaultdict(float)
    word_tag = defaultdict(lambda: defaultdict(float))
    word_pos_dict = {}
    for sent, tag_seq in zip(word_lst, pos_lst):
        for ww, pp in zip(sent, tag_seq):
            word_tag[ww][pp] += 1
            total_tag_p[pp] += 1
    pp_set = {}
    for ww in word_tag:
        w_count = 0
        for pp in word_tag[ww]:
            w_count += word_tag[ww][pp]
            if pp not in pp_set:
                pp_set[pp] = len(pp_set)
        for pp in word_tag[ww]:
            word_tag[ww][pp] = word_tag[ww][pp] / w_count

    for word in word_tag:
        max_pos = max(word_tag[word], key=lambda x: word_tag[word][x])
        if word_tag[word][max_pos] >= threshold:
            word_pos_dict[word] = max_pos
    p_count = 0
    avg_tag_p = torch.zeros(len(pp_set))
    for pp in total_tag_p:
        p_count += total_tag_p[pp]

    for ind, pp in enumerate(pp_set):
        avg_tag_p[ind] = total_tag_p[pp] / p_count
    return word_pos_dict

def get_word_pos_dict_thre_1(word_lst, pos_lst, threshold):
    total_tag_p = defaultdict(float)
    word_tag = defaultdict(lambda: defaultdict(float))
    word_pos_dict = {}
    for ww, pp in zip(word_lst, pos_lst):
        word_tag[ww][pp] += 1
        total_tag_p[pp] += 1
    pp_set = {}
    for ww in word_tag:
        w_count = 0
        for pp in word_tag[ww]:
            w_count += word_tag[ww][pp]
            if pp not in pp_set:
                pp_set[pp] = len(pp_set)
        for pp in word_tag[ww]:
            word_tag[ww][pp] = word_tag[ww][pp] / w_count

    for word in word_tag:
        max_pos = max(word_tag[word], key=lambda x: word_tag[word][x])
        if word_tag[word][max_pos] >= threshold:
            word_pos_dict[word] = max_pos
    p_count = 0
    avg_tag_p = torch.zeros(len(pp_set))
    for pp in total_tag_p:
        p_count += total_tag_p[pp]

    for ind, pp in enumerate(pp_set):
        avg_tag_p[ind] = total_tag_p[pp] / p_count
    return word_pos_dict







