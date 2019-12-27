'''
    Xiang Li
    xli150@jhu.edu
'''

import sys
import re
from collections import Counter
from collections import defaultdict
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from allennlp.modules.elmo import Elmo, batch_to_ids
import os
from elmoformanylangs import Embedder


class Data_Loader:

    def __init__(self, option, bos_tag=True, path=None):

        if option is None:
            self.threshold_sent_len = 15
            self.threshold_word = 1
        else:
            path = option.dataset
            self.threshold_sent_len = option.max_sent_len
            self.threshold_word = option.word_threshold

        if option is None or option.embedding_source in ['elmo', 'elmo_2', 'elmo_3']:
            self.keep_char = True
        else:
            self.keep_char = False

        if option is None or option.projective == 'projective':
            self.check_projective = True
        else:
            self.check_projective = False
        print('check_projective is ', self.check_projective)
        self.bos_tag = bos_tag
        self.train_data = self.pre_process_data_(path)
        self.processed_tag, self.processed_sent, self.processed_sent_Long, self.processed_tag_Long = self.get_dict(
            self.train_data)
        self.processed_tree, self.processed_tree_lab, self.processed_tree_Long, self.processed_tree_lab_Long = self.get_tree(
            self.train_data)

    def load_dev(self, dataset_dev_path):
        self.dev_data = self.pre_process_data_(dataset_dev_path)
        processed_tag_dev, processed_sent_dev, processed_sent_Long_dev, processed_tag_Long_dev = self.apply_dict(
            self.dev_data)
        processed_tree_dev, processed_tree_lab_dev, processed_tree_Long_dev, processed_tree_lab_Long_dev = self.get_tree_dev(
            self.dev_data)
        return processed_tag_dev, processed_sent_dev, processed_tree_dev, \
               processed_tree_lab_dev, processed_tree_Long_dev, processed_tree_lab_Long_dev

    def load_dev_verbo(self, dataset_dev_path):
        self.dev_data = self.pre_process_data_(dataset_dev_path)
        processed_tag_dev, processed_sent_dev, processed_sent_Long_dev, processed_tag_Long_dev = self.apply_dict(
            self.dev_data)
        processed_tree_dev, processed_tree_lab_dev, processed_tree_Long_dev, processed_tree_lab_Long_dev = self.get_tree_dev(
            self.dev_data)
        return processed_tag_dev, processed_sent_dev, processed_tree_dev, \
               processed_tree_lab_dev, processed_tree_Long_dev, processed_tree_lab_Long_dev, processed_tag_Long_dev

    def load_dev_with_tags(self, dataset_dev_path):
        dev_data = self.pre_process_data_(dataset_dev_path)
        processed_tag_dev, processed_sent_dev, processed_sent_Long_dev, processed_tag_Long_dev = self.apply_dict(
            dev_data)
        processed_tree_dev, processed_tree_lab_dev, processed_tree_Long_dev, processed_tree_lab_Long_dev = self.get_tree_dev(
            dev_data)
        dev_data = self.pre_process_data_verbo(dataset_dev_path)
        processed_feature_dev = [x[-1] for x in dev_data]
        dict_ = {}
        dict_["processed_tag_dev"] = processed_tag_dev
        dict_["processed_sent_dev"] = processed_sent_dev
        dict_["processed_feature_dev"] = processed_feature_dev
        dict_["processed_tree_dev"] = processed_tree_dev
        dict_["processed_tree_lab_dev"] = processed_tree_lab_dev
        return dict_

    def pre_process_data_verbo(self, path):
        train_data = []
        tree_struct = []
        dep_tags = []
        tag = []
        words = []
        verbose = []
        threshold = self.threshold_sent_len

        with open(path, 'r') as f:
            for sent in f:
                if sent[:1] == '#':
                    if tag and tree_struct:
                        ###
                        if (threshold is None or (len(words) < threshold and len(words) >= 2)) and (
                                not self.check_projective or self._is_projective(tree_struct)):
                            train_data.append(
                                (sentence, tuple(words), tuple(tag), tuple(tree_struct), tuple(dep_tags), verbose))
                            tree_struct = []
                            tag = []
                            dep_tags = []
                            words = []
                            verbose = []
                        else:
                            tree_struct = []
                            tag = []
                            dep_tags = []
                            words = []
                            verbose = []
                    try:
                        sent = sent.split('# sentence-text:')[1][:-1]
                    except:
                        pass
                        # please note that in language like Arabic, we don't have very clean sent in the header.
                    sentence = sent

                elif len(sent) > 2:
                    sent = sent.split('\t')
                    try:
                        tree_struct.append(int(sent[6]))
                        dep_tags.append(sent[7])
                        tag.append(sent[3])
                        words.append(sent[1].lower())
                        verbose.append(sent[5])
                    except:
                        pass
        if tag and tree_struct:
            # print(self._is_projective(tree_struct))
            if (threshold is None or (len(words) < threshold and len(words) >= 2)) and (
                    not self.check_projective or self._is_projective(tree_struct)):
                train_data.append((sentence, tuple(words), tuple(tag), tuple(tree_struct), tuple(dep_tags), verbose))
            else:
                pass
            # train_data.append((sentence, tuple(words), tuple(tag), tuple(tree_struct), tuple(dep_tags)))
        return train_data

    def pre_process_data_(self, path):
        train_data = []
        tree_struct = []
        dep_tags = []
        tag = []
        words = []
        threshold = self.threshold_sent_len

        with open(path, 'r') as f:
            for sent in f:
                if sent[:1] == '#' or sent[:1] == '\n':
                    if tag and tree_struct:
                        ###
                        if (threshold is None or (len(words) < threshold and len(words) >= 2)) and (
                                not self.check_projective or self._is_projective(tree_struct)):
                            train_data.append((sentence, tuple(words), tuple(tag), tuple(tree_struct), tuple(dep_tags)))
                            tree_struct = []
                            tag = []
                            dep_tags = []
                            words = []
                        else:
                            tree_struct = []
                            tag = []
                            dep_tags = []
                            words = []
                    try:
                        sent = sent.split('# sentence-text:')[1][:-1]
                    except:
                        pass
                        # print(sent)
                    sentence = sent

                elif len(sent) > 2:
                    sent = sent.split('\t')
                    try:
                        tree_struct.append(int(sent[6]))
                        dep_tags.append(sent[7])
                        tag.append(sent[3])
                        words.append(sent[1].lower())
                    except:
                        pass

        if tag and tree_struct:
            # print(self._is_projective(tree_struct))
            if (threshold is None or (len(words) < threshold and len(words) >= 2)) and (
                    not self.check_projective or self._is_projective(tree_struct)):
                train_data.append((sentence, tuple(words), tuple(tag), tuple(tree_struct), tuple(dep_tags)))
            else:
                pass
            # train_data.append((sentence, tuple(words), tuple(tag), tuple(tree_struct), tuple(dep_tags)))
        return train_data

    def _is_projective(self, depgraph):
        """
        Checks if a dependency graph is projective
        """
        arc_list = set()

        for index, father in enumerate(depgraph):
            child = index + 1
            arc_list.add((father, child))

        for (parentIdx, childIdx) in arc_list:
            # Ensure that childIdx < parentIdx
            if childIdx > parentIdx:
                temp = childIdx
                childIdx = parentIdx
                parentIdx = temp
            for k in range(childIdx + 1, parentIdx):
                for m in range(len(depgraph) + 1):
                    if (m < childIdx) or (m > parentIdx):
                        if (k, m) in arc_list:
                            return False
                        if (m, k) in arc_list:
                            return False
        return True

    def print_conllu(self, predicted, start, end, path=None):
        # print(predicted)
        if path is None:
            for sample, pred in zip(self.train_data[start:end], predicted):
                # print('1', pred)
                sentence, words, tags, corr_tree, dep_tag = sample

                print('#sent={}'.format(sentence))
                index = 1
                pred_temp = pred[0] if len(pred) == 2 else pred
                for word, tag, corr_t, dep_tag, pred_t in zip(words, tags, corr_tree, dep_tag, pred_temp):
                    print('{}\t{}\t{}\t{}\t{}\t_\t{}\t{}'.format(index, word, word, tag, tag, corr_t, pred_t))
                    index += 1
                print('')
        else:
            with open(path, 'w') as f:
                for sample in self.train_data[start:end]:
                    sentence, words, tags, corr_tree, dep_tag = sample
                    print('#sent={}'.format(sentence), file=f)
                    index = 0
                    for word, tag, corr_t, dep_tag, pred_t in zip(words, tags, corr_tree, dep_tag, predicted):
                        print('{}\t{}\t{}\t{}\t{}\t_\t{}\t{}'.format(index, word, word, tag, tag, corr_t, pred_t),
                              file=f)
                        index += 1
                    print('', file=f)

    def get_tree(self, train_data):
        processed_tree = []
        processed_tree_lab = []
        processed_tree_Long = []
        processed_tree_lab_Long = []

        self.label_dict = {}
        for _, _, _, tree, tree_label in train_data:
            for lab in tree_label:
                if lab not in self.label_dict:
                    self.label_dict[lab] = len(self.label_dict)
            processed_tree.append(tree)
            processed_tree_lab.append(tree_label)
            processed_tree_Long.append(torch.LongTensor(list(tree)).view(1, -1))
            processed_tree_lab_Long.append(torch.LongTensor([self.label_dict[x] for x in tree_label]).view(1, -1))
        self.label_dict['UNK'] = len(self.label_dict)

        return processed_tree, processed_tree_lab, processed_tree_Long, processed_tree_lab_Long

    def get_tree_dev(self, train_data):
        processed_tree = []
        processed_tree_lab = []
        processed_tree_Long = []
        processed_tree_lab_Long = []

        unk_label = self.label_dict['UNK']
        for _, _, _, tree, tree_label in train_data:
            processed_tree.append(tree)
            processed_tree_lab.append(tree_label)
            processed_tree_Long.append(torch.LongTensor(list(tree)).view(1, -1))
            processed_tree_lab_Long.append(
                torch.LongTensor([self.label_dict[x] if x in self.label_dict else unk_label for x in tree_label]).view(
                    1, -1))

        return processed_tree, processed_tree_lab, processed_tree_Long, processed_tree_lab_Long

    def get_dict(self, train_data):
        tag_count = Counter()
        word_count = Counter()

        for sentence, words, tags, _, _ in train_data:
            for word in words:
                word_count[word] += 1
            for tag in tags:
                tag_count[tag] += 1

        self.tag_count = tag_count
        self.word_count = word_count
        self.tag_dict = {}
        self.word_dict = {}

        for elem in tag_count:
            if tag_count[elem] > 0:
                self.tag_dict[elem] = len(self.tag_dict)

        for elem in word_count:
            if word_count[elem] > self.threshold_word:
                self.word_dict[elem] = len(self.word_dict)

        unk_word = len(self.word_dict)
        self.word_dict['<unk>'] = unk_word
        unk_tag = len(self.tag_dict)
        self.tag_dict['UNK'] = unk_tag

        if self.bos_tag:
            self.tag_dict['<BOS>'] = len(self.tag_dict)
            self.word_dict['<BOS>'] = len(self.word_dict)

        processed_sent = []
        processed_tag = []
        processed_tag_Long = []
        processed_sent_Long = []
        if self.keep_char:
            for sentence, words, tags, _, _ in train_data:
                sent = list(words)
                tag_seq = list(tags)
                processed_tag.append(tag_seq)
                processed_sent.append(sent)
                sent_long = torch.LongTensor([self.word_dict[x] if x in self.word_dict else unk_word for x in words])
                tag_seq_long = torch.LongTensor([self.tag_dict[x] if x in self.tag_dict else unk_tag for x in tags])
                processed_tag_Long.append(tag_seq_long)
                processed_sent_Long.append(sent_long)
        else:
            for sentence, words, tags, _, _ in train_data:
                sent = torch.LongTensor([self.word_dict[x] if x in self.word_dict else unk_word for x in words])
                tag_seq = torch.LongTensor([self.tag_dict[x] if x in self.tag_dict else unk_tag for x in tags])
                processed_tag.append(tag_seq)
                processed_sent.append(sent)

        return processed_tag, processed_sent, processed_sent_Long, processed_tag_Long

    def apply_dict(self, dev_data):

        processed_sent = []
        processed_tag = []
        processed_tag_Long = []
        processed_sent_Long = []
        unk_word = self.word_dict['<unk>']
        unk_tag = self.tag_dict['UNK']

        if self.keep_char:
            for sentence, words, tags, _, _ in dev_data:
                sent = list(words)
                tag_seq = list(tags)
                processed_tag.append(tag_seq)
                processed_sent.append(sent)
                sent_long = torch.LongTensor([self.word_dict[x] if x in self.word_dict else unk_word for x in words])
                tag_seq_long = torch.LongTensor([self.tag_dict[x] if x in self.tag_dict else unk_tag for x in tags])
                processed_tag_Long.append(tag_seq_long)
                processed_sent_Long.append(sent_long)
        else:
            for sentence, words, tags, _, _ in dev_data:
                sent = torch.LongTensor([self.word_dict[x] if x in self.word_dict else unk_word for x in words])
                tag_seq = torch.LongTensor([self.tag_dict[x] if x in self.tag_dict else unk_tag for x in tags])
                processed_tag.append(tag_seq)
                processed_sent.append(sent)

        return processed_tag, processed_sent, processed_sent_Long, processed_tag_Long

    def print_sentence_length_stats(self):
        lst = [len(sent) for sent in self.processed_sent]
        plt.hist(lst, bins=10)
        plt.show()
        return (min(lst), max(lst))


class Foreign_Elmo():
    def __init__(self, dir_name, embedding_source, device=None):
        self.embedding_source = embedding_source
        self.device = device
        print('loading the embedder')
        self.e = Embedder(dir_name)
        print('finish loading the embedder')

    def _get_embeddings(self, sent_lst):
        # if self.embedding_source  == 'elmo_0':
        #     type = self.e.sents2elmo(sent_lst, 0)
        #     type = [torch.tensor(x).unsqueeze(1) for x in type]
        #     return type, type
        # else:
        full_type = []
        full_token = []
        for batch_ in sent_lst:
            # print(len(batch_), len(batch_[0]))
            result = torch.tensor(self.e.sents2elmo(batch_, -2))
            # print(result.shape)
            type_ = result[:, 0, :, :]
            if self.embedding_source == 'elmo_1':
                index = 1
            elif self.embedding_source == 'elmo_2':
                index = 2
            elif self.embedding_source == 'elmo_0':
                index = 0 
            token_ = result[:, index, :, :]
            full_type.append(type_)
            full_token.append(token_)
        return full_token, full_type

    def get_part_elmo(self, batch_):
        result = torch.tensor(self.e.sents2elmo(batch_, -2))
        # print(result.shape)
        if self.embedding_source == 'elmo_1':
            index = 1
        elif self.embedding_source == 'elmo_2':
            index = 2
        elif self.embedding_source == 'elmo_0':
            index = 0 
        token_ = result[:, index, :, :]
        return token_

      



class Embedding_Weight():
    def __init__(self, embedding_source, path=None, data_loader = None, num_sent = None, device=None, requires_grad=False):
        self.requires_grad = requires_grad
        if embedding_source == 'glove':
            self.word_map = {}
            self.file_reader(path)
        elif embedding_source == 'elmo':
            pass
        self.device = device
        # self.elmo = self.load_init_elmo().to(self.device)
        self.elmo = self.load_init_elmo().to(self.device)

    def load_init_elmo(self, lang = 'en'):
        if lang == 'en':
            system_config = '/home-4/xli150@jhu.edu/mutualinfo/others/elmo_2x4096_512_2048cnn_2xhighway_options.json'
            system_weight = '/home-4/xli150@jhu.edu/mutualinfo/others/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
            config_isfile = os.path.isfile(system_config)
            weight_isfile = os.path.isfile(system_weight)
            options_file = system_config if config_isfile else "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = system_weight if weight_isfile else "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            save_embeddings = []
            # Compute two different representation for each token.
            # Each representation is a linear weighted combination for the
            # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))

        elif lang == 'pt':
            pass
            # options_file = args.elmo_option_path
            # weight_file = args.elmo_model_path

        elmo = Elmo(options_file, weight_file, 1, dropout=0.0, requires_grad=self.requires_grad)

        return elmo



    def file_reader(self, path):
        with open(path, 'r') as f:
            for line in f:
                line = line.split()
                self.word_map[line[0]] = np.array(line[1:])
        self.embedding_size = len(self.word_map[line[0]])
        return 

    def gen_word_embed(self, data_loader):
        '''
            return a Numpy matrix. 
        '''
        result = np.zeros([len(data_loader.word_dict), self.embedding_size])
        for word, val in data_loader.word_dict.items():
            if word in self.word_map:
                result[val, :] = self.word_map[word]
            else:
                result[val,:] = np.zeros(self.embedding_size)
        return result


        embed = nn.Embedding(num_embeddings, embedding_dim)
        # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        embed.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def get_part_elmo(self, sentences):
        character_ids = batch_to_ids(sentences).to(self.device)
        bilm_output = self.elmo._elmo_lstm(character_ids)
        temp = bilm_output['activations']

        # first_layer = temp[1][:, 1:-1]  # shape = (bsz, seqlen, dim)
        second_layer = temp[1][:, 1:-1]
        # third_layer = temp[2][:, 1:-1]

        return second_layer

    def elmo_embeddings_first_batch(self, processed_sent, number_sent):
        # with open(temp_file, 'w')
        '''
            (batch, sequence_length, 50)
            returns dict 
        '''
        save_embeddings = []
        for index, sent in enumerate(processed_sent[:number_sent]):
            sentences = sent # create a list of size one of a list.
            character_ids = batch_to_ids(sentences).to(self.device)
            embeddings = self.elmo._elmo_lstm._token_embedder(character_ids)
            if index % 1000 == 0:
                sys.stdout.write('-')
                sys.stdout.flush()
            sys.stdout.flush()
            save_embeddings.append(embeddings["token_embedding"][:, 1:-1]) # shape = (bsz, seqlen, dim)
        sys.stdout.write('\n')
        return save_embeddings

    def elmo_embeddings(self, processed_sent, number_sent, lang='en', args=None):
        print('complicated_layers_extraction')


        '''
            (batch, sequence_length, 50)
            returns dict 
        '''

        first_layer_lst = []
        second_layer_lst = []
        third_layer_lst = []
        for index, sent in enumerate(processed_sent[:number_sent]):
            sentences = sent # create a list of size one of a list.
            # print(sentences)
            character_ids = batch_to_ids(sentences).to(self.device)
            # embeddings = elmo._elmo_lstm._token_embedder(character_ids)
            bilm_output = self.elmo._elmo_lstm(character_ids)
            temp = bilm_output['activations']
            first_layer = temp[0].cpu()[:, 1:-1] # shape = (bsz, seqlen, dim)
            second_layer = temp[1].cpu()[:, 1:-1]
            third_layer = temp[2].cpu()[:, 1:-1]

            first_layer_lst.append(first_layer)
            second_layer_lst.append(second_layer)
            third_layer_lst.append(third_layer)

            if index % 1000 == 0:
                sys.stdout.write('-')
                sys.stdout.flush()
        sys.stdout.write('\n')

        return first_layer_lst, second_layer_lst, third_layer_lst

    def elmo_embeddings_first(self, processed_sent, number_sent,
                              store_file=None):
        # with open(temp_file, 'w')
        try:
            with open(store_file, 'rb') as f:
                save_embeddings = pickle.load(f)

            return save_embeddings
        except:
            system_config = '/home-4/xli150@jhu.edu/mutualinfo/others/elmo_2x4096_512_2048cnn_2xhighway_options.json'
            system_weight = '/home-4/xli150@jhu.edu/mutualinfo/others/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
            config_isfile = os.path.isfile(system_config)
            weight_isfile = os.path.isfile(system_weight)
            options_file = system_config if config_isfile else "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = system_weight if weight_isfile else "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            save_embeddings = []
            # Compute two different representation for each token.
            # Each representation is a linear weighted combination for the
            # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
            elmo = Elmo(options_file, weight_file, 1, dropout=0)


            '''
                (batch, sequence_length, 50)
                returns dict 
            '''
            for index, sent in enumerate(processed_sent[:number_sent]):
                sentences = [list(sent)] # create a list of size one of a list.
                character_ids = batch_to_ids(sentences)
                embeddings = elmo._elmo_lstm._token_embedder(character_ids)
                if index % 1000 == 0:
                    sys.stdout.write('-')
                    sys.stdout.flush()
                save_embeddings.append(embeddings["token_embedding"].squeeze(0)[1:-1].unsqueeze(1))
            sys.stdout.write('\n')

            return save_embeddings




        
    def helper(self, ):
        options_file = 'ELMo/elmo_2x1024_128_2048cnn_1xhighway_weights.json'
        weight_file = 'ELMo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

        # Compute two different representation for each token.
        # Each representation is a linear weighted combination for the
        # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
        elmo = Elmo(options_file, weight_file, 2, dropout=0)

        # use batch_to_ids to convert sentences to character ids
        sentences = [['First', 'sentence', '.'], ['Another', '.']]
        character_ids = batch_to_ids(sentences)

        embeddings = elmo(character_ids)

        # embeddings['elmo_representations'] is length two list of tensors.
        # Each element contains one layer of ELMo representations with shape
        # (2, 3, 1024).
        #   2    - the batch size
        #   3    - the sequence length of the batch
        #   1024 - the length of each ELMo vector


class Data_Loader2:

    def __init__(self, option, bos_tag=True, path=None):
        
        if option is None:
            self.threshold_sent_len = 15
            self.threshold_word = 1
        else:
            path = option.dataset
            self.threshold_sent_len = option.max_sent_len
            self.threshold_word = option.word_threshold

        if option is None or option.embedding_source in ['elmo', 'elmo_0','elmo_1', 'elmo_2']:
            self.keep_char = True
        else:
            self.keep_char = False

        if option is None or option.projective == 'projective':
            self.check_projective = True
        else:
            self.check_projective = False
        print('check_projective is ', self.check_projective)
        self.bos_tag = bos_tag
        self.bsz = option.batch_size
        self.device = option.device
        self.train_data = self.pre_process_data_(path)
        processed_tag, processed_sent, processed_sent_Long, processed_tag_Long = self.get_dict(self.train_data)
        processed_tree, processed_tree_lab, processed_tree_Long, processed_tree_lab_Long = self.get_tree(self.train_data)
        self.corpus, self.batch_idx, self.en_batch, self.processed_tree = self.batchify((processed_tag_Long, processed_sent_Long, processed_tree_Long, processed_tree_lab_Long, processed_sent),
                                    self.bsz)

    def load_dev(self, dataset_dev_path):
        self.dev_data = self.pre_process_data_(dataset_dev_path)
        processed_tag_dev, processed_sent_dev, processed_sent_Long_dev, processed_tag_Long_dev = self.apply_dict(self.dev_data)
        processed_tree_dev, processed_tree_lab_dev, processed_tree_Long_dev, processed_tree_lab_Long_dev = self.get_tree_dev(self.dev_data)

        return self.batchify((processed_tag_Long_dev, processed_sent_Long_dev, processed_tree_Long_dev, \
               processed_tree_lab_Long_dev, processed_sent_dev), self.bsz)
        # return processed_tag_dev, processed_sent_dev, processed_tree_dev, \
        #        processed_tree_lab_dev, processed_tree_Long_dev, processed_tree_lab_Long_dev

    def batchify(self, corpus, bsz):
        processed_tag_, processed_sent_, processed_tree_, processed_tree_lab_, processed_sent_en = corpus

        sents, sorted_idxs = zip(*sorted(zip(processed_sent_, range(len(processed_sent_))), key=lambda x: len(x[0])))
        minibatches, mb2linenos, en_batch = [], [], []
        curr_batch, curr_tags, curr_sent, curr_tree, curr_treelab = [], [], [], [], []
        curr_linenos = []

        curr_inps = []
        curr_len = len(sents[0])
        for i in range(len(sents)):
            if len(sents[i]) != curr_len or len(curr_batch) == bsz:
                # print(torch.LongTensor(curr_batch).shape)
                # print(torch.LongTensor(curr_tags).shape, torch.LongTensor(curr_sent).shape,
                #                     torch.LongTensor(curr_tree).shape,
                #                     torch.LongTensor(curr_treelab).shape)
                minibatches.append((torch.LongTensor(curr_batch),
                                    torch.LongTensor(curr_tags),
                                    # torch.LongTensor(curr_sent),
                                    torch.LongTensor(curr_tree),
                                    torch.LongTensor(curr_treelab))
                                    )
                mb2linenos.append(curr_linenos)
                en_batch.append(curr_sent)
                curr_batch = [sents[i]]
                curr_len = len(sents[i])
                curr_tags = [processed_tag_[sorted_idxs[i]]]
                curr_sent = [processed_sent_en[sorted_idxs[i]]]
                curr_tree = [processed_tree_[sorted_idxs[i]]]
                curr_treelab = [processed_tree_lab_[sorted_idxs[i]]]
                curr_linenos = [sorted_idxs[i]]
            else:
                curr_batch.append(sents[i])
                curr_tags.append(processed_tag_[sorted_idxs[i]])
                curr_sent.append(processed_sent_en[sorted_idxs[i]])
                curr_tree.append(processed_tree_[sorted_idxs[i]])
                curr_treelab.append(processed_tree_lab_[sorted_idxs[i]])
                curr_linenos.append(sorted_idxs[i])
        # catch last
        if len(curr_batch) > 0:
            minibatches.append((torch.LongTensor(curr_batch),
                                torch.LongTensor(curr_tags),
                                # torch.LongTensor(curr_sent),
                                torch.LongTensor(curr_tree),
                                torch.LongTensor(curr_treelab))
                               )
            mb2linenos.append(curr_linenos)
            en_batch.append(curr_sent)
        return minibatches, mb2linenos, en_batch, processed_tree_


    def load_dev_verbo(self, dataset_dev_path):
        self.dev_data = self.pre_process_data_(dataset_dev_path)
        processed_tag_dev, processed_sent_dev, processed_sent_Long_dev, processed_tag_Long_dev = self.apply_dict(self.dev_data)
        processed_tree_dev, processed_tree_lab_dev, processed_tree_Long_dev, processed_tree_lab_Long_dev = self.get_tree_dev(self.dev_data)
        return processed_tag_dev, processed_sent_dev, processed_tree_dev, \
               processed_tree_lab_dev, processed_tree_Long_dev, processed_tree_lab_Long_dev, processed_tag_Long_dev

    def load_dev_with_tags(self, dataset_dev_path):
        dev_data = self.pre_process_data_(dataset_dev_path)
        processed_tag_dev, processed_sent_dev, processed_sent_Long_dev, processed_tag_Long_dev = self.apply_dict(
            dev_data)
        processed_tree_dev, processed_tree_lab_dev, processed_tree_Long_dev, processed_tree_lab_Long_dev = self.get_tree_dev(
            dev_data)
        dev_data = self.pre_process_data_verbo(dataset_dev_path)
        processed_feature_dev = [x[-1] for x in dev_data]
        dict_ = {}
        dict_["processed_tag_dev"] = processed_tag_dev
        dict_["processed_sent_dev"] = processed_sent_dev
        dict_["processed_feature_dev"] = processed_feature_dev
        dict_["processed_tree_dev"] = processed_tree_dev
        dict_["processed_tree_lab_dev"] = processed_tree_lab_dev
        return dict_

    def pre_process_data_verbo(self, path):
        train_data = []
        tree_struct = []
        dep_tags = []
        tag = []
        words = []
        verbose = []
        threshold = self.threshold_sent_len

        with open(path, 'r') as f:
            for sent in f:
                if sent[:1] == '#':
                    if tag and tree_struct:
                        ###
                        if (threshold is None or (len(words) < threshold  and len(words) >= 2) ) and (not self.check_projective or self._is_projective(tree_struct)):
                            train_data.append((sentence, tuple(words), tuple(tag), tuple(tree_struct), tuple(dep_tags), verbose))
                            tree_struct = []
                            tag = []
                            dep_tags=[]
                            words=[]
                            verbose=[]
                        else:
                            tree_struct = []
                            tag = []
                            dep_tags=[]
                            words=[]
                            verbose = []
                    try:
                        sent = sent.split('# sentence-text:')[1][:-1]
                    except:
                        pass
                        # please note that in language like Arabic, we don't have very clean sent in the header.
                    sentence = sent

                elif len(sent)>2:
                    sent = sent.split('\t')
                    tree_struct.append(int(sent[6]))
                    dep_tags.append(sent[7])
                    tag.append(sent[3])
                    words.append(sent[1].lower())
                    verbose.append(sent[5])

        if tag and tree_struct:
            # print(self._is_projective(tree_struct))
            if (threshold is None or (len(words) < threshold and len(words) >= 2 )) and (not self.check_projective or self._is_projective(tree_struct)):
                train_data.append((sentence, tuple(words), tuple(tag), tuple(tree_struct), tuple(dep_tags), verbose))
            else:
                pass
            # train_data.append((sentence, tuple(words), tuple(tag), tuple(tree_struct), tuple(dep_tags)))
        return train_data

    def pre_process_data_(self, path):
        train_data = []
        tree_struct = []
        dep_tags = []
        tag = []
        words = []
        threshold = self.threshold_sent_len

        with open(path, 'r') as f:
            for sent in f:
                if sent[:1] == '#' or sent[:1] == '\n':
                    if tag and tree_struct:
                        ### 
                        if (threshold is None or (len(words) < threshold  and len(words) >= 2) ) and (not self.check_projective or self._is_projective(tree_struct)):
                            train_data.append((sentence, tuple(words), tuple(tag), tuple(tree_struct), tuple(dep_tags)))
                            tree_struct = []
                            tag = []
                            dep_tags=[]
                            words=[]
                        else:
                            tree_struct = []
                            tag = []
                            dep_tags=[]
                            words=[]
                    try:
                        sent = sent.split('# sentence-text:')[1][:-1]
                    except:
                        pass
                        # print(sent)
                    sentence = sent

                elif len(sent)>2:
                    sent = sent.split('\t')
                    try:
                        tree_struct.append(int(sent[6]))
                        dep_tags.append(sent[7])
                        tag.append(sent[3])
                        words.append(sent[1].lower())
                    except:
                        pass

        if tag and tree_struct:
            # print(self._is_projective(tree_struct))
            if (threshold is None or (len(words) < threshold and len(words) >= 2 )) and (not self.check_projective or self._is_projective(tree_struct)):
                train_data.append((sentence, tuple(words), tuple(tag), tuple(tree_struct), tuple(dep_tags)))
            else:
                pass
            # train_data.append((sentence, tuple(words), tuple(tag), tuple(tree_struct), tuple(dep_tags)))
        return train_data



    def _is_projective(self, depgraph):
        """
        Checks if a dependency graph is projective
        """
        arc_list = set()

        for index, father in enumerate(depgraph):
            child = index + 1
            arc_list.add((father, child))

        for (parentIdx, childIdx) in arc_list:
            # Ensure that childIdx < parentIdx
            if childIdx > parentIdx:
                temp = childIdx
                childIdx = parentIdx
                parentIdx = temp
            for k in range(childIdx + 1, parentIdx):
                for m in range(len(depgraph)+1):
                    if (m < childIdx) or (m > parentIdx):
                        if (k, m) in arc_list:
                            return False
                        if (m, k) in arc_list:
                            return False
        return True


    def print_conllu(self, predicted, start, end, path=None):
        # print(predicted)
        if path is None:
            for sample, pred in zip(self.train_data[start:end], predicted):
                # print('1', pred)
                sentence, words, tags, corr_tree, dep_tag = sample

                print('#sent={}'.format(sentence))
                index = 1
                pred_temp = pred[0] if len(pred)==2 else pred
                for word, tag, corr_t, dep_tag, pred_t in zip(words, tags, corr_tree, dep_tag, pred_temp):
                    print('{}\t{}\t{}\t{}\t{}\t_\t{}\t{}'.format(index, word, word, tag, tag, corr_t, pred_t))
                    index += 1
                print('')
        else:
            with open(path, 'w') as f:
                for sample in self.train_data[start:end]:
                    sentence, words, tags, corr_tree, dep_tag = sample
                    print('#sent={}'.format(sentence), file=f)
                    index = 0
                    for word, tag, corr_t, dep_tag, pred_t in zip(words, tags, corr_tree, dep_tag, predicted):
                        print('{}\t{}\t{}\t{}\t{}\t_\t{}\t{}'.format(index, word, word, tag, tag, corr_t, pred_t), file=f)
                        index += 1
                    print('', file=f)

    def get_tree(self, train_data):
        processed_tree = []
        processed_tree_lab = []
        processed_tree_Long = []
        processed_tree_lab_Long = []

        self.label_dict = {}
        for _, _, _, tree, tree_label in train_data:
            for lab in tree_label:
                if lab not in self.label_dict:
                    self.label_dict[lab] = len(self.label_dict)
            processed_tree.append(tree)
            processed_tree_lab.append(tree_label)
            processed_tree_Long.append(list(tree))
            processed_tree_lab_Long.append([self.label_dict[x] for x in tree_label])
        self.label_dict['UNK'] = len(self.label_dict)

        return processed_tree, processed_tree_lab, processed_tree_Long, processed_tree_lab_Long

    def get_tree_dev(self, train_data):
        processed_tree = []
        processed_tree_lab = []
        processed_tree_Long = []
        processed_tree_lab_Long = []

        unk_label = self.label_dict['UNK']
        for _, _, _, tree, tree_label in train_data:

            processed_tree.append(tree)
            processed_tree_lab.append(tree_label)
            processed_tree_Long.append(list(tree))
            processed_tree_lab_Long.append(
                [self.label_dict[x] if x in self.label_dict else unk_label for x in tree_label])

        return processed_tree, processed_tree_lab, processed_tree_Long, processed_tree_lab_Long

    def get_dict(self, train_data):
        tag_count = Counter()
        word_count = Counter()
       
        for sentence, words, tags, _, _ in train_data:
            for word in words:
                word_count[word] += 1
            for tag in tags:
                tag_count[tag] += 1

        self.tag_count = tag_count
        self.word_count = word_count
        self.tag_dict = {}
        self.word_dict = {}

        for elem in tag_count:
            if tag_count[elem] > 0:
                self.tag_dict[elem] = len(self.tag_dict)

        for elem in word_count:
            if word_count[elem] > self.threshold_word:
                self.word_dict[elem] = len(self.word_dict)

        unk_word = len(self.word_dict)
        self.word_dict['<unk>'] = unk_word 
        unk_tag = len(self.tag_dict)
        self.tag_dict['UNK'] = unk_tag

        if self.bos_tag:
            self.tag_dict['<BOS>'] = len(self.tag_dict)
            self.word_dict['<BOS>'] = len(self.word_dict)

        processed_sent = []
        processed_tag = []
        processed_tag_Long = []
        processed_sent_Long = []
        if self.keep_char:
            for sentence, words, tags, _, _ in train_data:
                sent = list(words)
                tag_seq = list(tags) 
                processed_tag.append(tag_seq)
                processed_sent.append(sent)
                sent_long = [self.word_dict[x] if x in self.word_dict else unk_word for x in words]
                tag_seq_long = [self.tag_dict[x] if x in self.tag_dict else unk_tag for x in tags]
                processed_tag_Long.append(tag_seq_long)
                processed_sent_Long.append(sent_long)
        else:
            for sentence, words, tags, _, _ in train_data:
                sent = [self.word_dict[x] if x in self.word_dict else unk_word for x in words]
                tag_seq = [self.tag_dict[x] if x in self.tag_dict else unk_tag for x in tags]
                processed_tag.append(tag_seq)
                processed_sent.append(sent)

        return processed_tag, processed_sent, processed_sent_Long, processed_tag_Long

    def apply_dict(self, dev_data):

        processed_sent = []
        processed_tag = []
        processed_tag_Long = []
        processed_sent_Long = []
        unk_word = self.word_dict['<unk>']
        unk_tag = self.tag_dict['UNK']

        if self.keep_char: 
            for sentence, words, tags, _, _ in dev_data:
                sent = list(words)
                tag_seq = list(tags) 
                processed_tag.append(tag_seq)
                processed_sent.append(sent)
                sent_long = [self.word_dict[x] if x in self.word_dict else unk_word for x in words]
                tag_seq_long = [self.tag_dict[x] if x in self.tag_dict else unk_tag for x in tags]
                processed_tag_Long.append(tag_seq_long)
                processed_sent_Long.append(sent_long)
        else:
            for sentence, words, tags, _, _ in dev_data:
                sent = [self.word_dict[x] if x in self.word_dict else unk_word for x in words]
                tag_seq = [self.tag_dict[x] if x in self.tag_dict else unk_tag for x in tags]
                processed_tag.append(tag_seq)
                processed_sent.append(sent)

        return processed_tag, processed_sent, processed_sent_Long, processed_tag_Long


    def print_sentence_length_stats(self):
        lst = [len(sent) for sent in self.processed_sent]
        plt.hist(lst, bins=10)
        plt.show()
        return (min(lst), max(lst))

        