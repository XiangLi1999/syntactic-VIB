'''
04/26/2019
'''
import numpy as np
import torch
import random
from lang_select import fetch_paths
from conllu_handler import Data_Loader, Foreign_Elmo, Embedding_Weight
import sys, pickle
from collections import defaultdict
from others import *


def load_elmo_batch(data_loader, args, embed_loader, mod='train', processed_sent_dev=None, processed_sent_test=None):
    if mod == 'train':
        if args.embedding_source == 'elmo_1':
            elmo_embeds_train, elmo_embeds_train_type = train_token_loader_batch(data_loader, embed_loader, dim=1)
        elif args.embedding_source == 'elmo_2':
            elmo_embeds_train, elmo_embeds_train_type = train_token_loader_batch(data_loader, embed_loader, dim=2)
        elif args.embedding_source == 'elmo_0':  
            elmo_embeds_train, elmo_embeds_train_type = train_token_loader_batch(data_loader, embed_loader, dim=0)
        else:
            # elmo_embeds_train, elmo_embeds_train_type = train_token_loader_batch(args.elmo_train_data_path, data_loader,
            #                                                                      embed_loader, dim=0)
            elmo_embeds_train = embed_loader.elmo_embeddings_first_batch(data_loader.en_batch , len(data_loader.en_batch ))
            elmo_embeds_train_type = elmo_embeds_train
        return elmo_embeds_train, elmo_embeds_train_type

    elif mod == 'dev':
        assert processed_sent_dev is not None
        if args.embedding_source == 'elmo_1':
            elmo_embeds_dev, elmo_embeds_dev_type = dev_token_loader_batch(processed_sent_dev,
                                               embed_loader, dim=1)
        elif args.embedding_source == 'elmo_2':
            elmo_embeds_dev, elmo_embeds_dev_type = dev_token_loader_batch(processed_sent_dev,
                                               embed_loader, dim=2)
        elif args.embedding_source == 'elmo_0':
            elmo_embeds_dev, elmo_embeds_dev_type = dev_token_loader_batch(processed_sent_dev,
                                               embed_loader, dim=0)
        else:
            # elmo_embeds_dev, elmo_embeds_dev_type = dev_token_loader_batch(args.elmo_dev_data_path, processed_sent_dev,
            #                                                                embed_loader, dim=0)

            elmo_embeds_dev = embed_loader.elmo_embeddings_first_batch(processed_sent_dev,
                                                                 len(processed_sent_dev))
            elmo_embeds_dev_type = elmo_embeds_dev

        return elmo_embeds_dev, elmo_embeds_dev_type

    elif mod == 'test':
        assert processed_sent_test is not None
        if args.embedding_source == 'elmo_1':
            elmo_embeds_dev, elmo_embeds_dev_type = dev_token_loader_batch(processed_sent_test,
                                               embed_loader, dim=1)
        elif args.embedding_source == 'elmo_2':
            elmo_embeds_dev, elmo_embeds_dev_type = dev_token_loader_batch(processed_sent_test,
                                               embed_loader, dim=2)
        elif args.embedding_source == 'elmo_0':
            elmo_embeds_dev, elmo_embeds_dev_type = dev_token_loader_batch(processed_sent_test,
                                               embed_loader, dim=0)
        else:
            # elmo_embeds_dev, elmo_embeds_dev_type = dev_token_loader_batch(args.elmo_test_data_path,
            #                                                                processed_sent_test,
            #                                                                embed_loader, dim=0)
            elmo_embeds_dev = embed_loader.elmo_embeddings_first_batch(processed_sent_test,
                                                                 len(processed_sent_test))
            elmo_embeds_dev_type = elmo_embeds_dev

        return elmo_embeds_dev, elmo_embeds_dev_type



def dev_token_loader_batch(dev_sent, embed_loader, dim=1):
    print('computing pretrained embeddings ...')
    sys.stdout.flush()
    elmo_embeds_train = embed_loader.elmo_embeddings(dev_sent,
                                                     len(dev_sent))
    return elmo_embeds_train[dim], elmo_embeds_train[0]

def train_token_loader_batch(data_loader, embed_loader, dim=1):
    print('computing pretrained embeddings ...')
    sys.stdout.flush()
    elmo_embeds_train = embed_loader.elmo_embeddings(data_loader.en_batch,
                                                     len(data_loader.en_batch))
    big_lst = elmo_embeds_train[dim], elmo_embeds_train[0]
    return big_lst


# Analysis 2
# ==========================================================================================================
# ============================================== POS MI EVAL ===============================================
# ==========================================================================================================
def eval_pos(model, data_loader, args):
    for param in model.parameters():
        param.require_grad = False
    if args.foreign == 'no':
        embed_loader = Embedding_Weight(args.embedding_source, data_loader=data_loader, num_sent=args.epoch_sent)
        elmo_embeds_train, elmo_type_train = load_elmo(data_loader, args, embed_loader,
                                                    mod='train', processed_sent_dev=None,
                                                    processed_sent_test=None)
        ''' load the dev data '''
        processed_tag_dev, processed_sent_dev, processed_tree_dev, processed_tree_lab_dev, \
        processed_tree_Long_dev, processed_tree_lab_Long_dev, processed_tag_Long_dev = data_loader.load_dev_verbo(
            args.dataset_dev)

        token_embeds_dev, elmo_type_dev = load_elmo(data_loader, args, embed_loader,
                                                    mod='dev', processed_sent_dev=processed_sent_dev,
                                                    processed_sent_test=None)
        ''' load the test data '''
        if args.test == 'yes':
            processed_tag_test, processed_sent_test, processed_tree_test, processed_tree_lab_test, \
            processed_tree_Long_test, processed_tree_lab_Long_test, processed_tag_Long_test = data_loader.load_dev_verbo(
                args.dataset_test)

            token_embeds_test, elmo_type_test = load_elmo(data_loader, args, embed_loader,
                                                        mod='test', processed_sent_dev=None,
                                                        processed_sent_test=processed_sent_test)

        full=True
        mi_total_dev, err_total_dev, entr_total_dev, cond_entr_total_dev = 0, 0, 0, 0
        mi_total_test, err_total_test, entr_total_test, cond_entr_total_test = 0, 0, 0, 0

        predictor = baselines.Eval_predict_pos(data_loader.processed_sent, data_loader.tag_dict, model,
                                                embed_loader, args)
    elif args.foreign == 'yes':
        embed_loader = Foreign_Elmo(args.elmo_model_path, args.embedding_source)
        elmo_embeds_train, non_context_embeds_train = embed_loader._get_embeddings(data_loader.processed_sent)

        processed_tag_dev, processed_sent_dev, processed_tree_dev, processed_tree_lab_dev, \
        processed_tree_Long_dev, processed_tree_lab_Long_dev, processed_tag_Long_dev = data_loader.load_dev_verbo(
            args.dataset_dev)

        token_embeds_dev, elmo_type_dev = embed_loader._get_embeddings(processed_sent_dev)

        if args.test == 'yes':
            processed_tag_test, processed_sent_test, processed_tree_test, processed_tree_lab_test, \
            processed_tree_Long_test, processed_tree_lab_Long_test, processed_tag_Long_test = data_loader.load_dev_verbo(
                args.dataset_test)

            token_embeds_test, elmo_type_test = embed_loader._get_embeddings(processed_sent_test)
   
    

    for e in range(args.epoch):
        shuffle_indices = np.random.choice(len(data_loader.processed_sent), len(data_loader.processed_sent),
                                       replace=False)
        train_sent = [data_loader.processed_sent[x] for x in shuffle_indices]
        train_tag = [data_loader.processed_tag_Long[x] for x in shuffle_indices]
        train_elmo = [elmo_embeds_train[x] for x in shuffle_indices]
        outp = predictor.train_discrete(train_sent, train_tag, train_elmo)
        print(outp)
    print('\n\n')
    sys.stdout.flush()

    predictor.save_model(args.save_path + '_pos_pred' + 'seed={}-quant={}'.format(rand_seed, few_shot_quant))

    # ================ eval on dev ===================
    str1, dict_result_dev = predictor.eval_dev_discrete(processed_sent_dev, processed_tag_Long_dev,
                                                        token_embeds_dev)
    print('dev_summary:{}, beta={}'.format(str1, model.beta))
    sys.stdout.flush()
    mi_total_dev += dict_result_dev["MI(POS,Tag)"]
    err_total_dev += dict_result_dev["error"]
    entr_total_dev += dict_result_dev["H(POS)"]
    cond_entr_total_dev += dict_result_dev["H(POS|Tag)"]
    print('LOOK:summary_dev: few_shot={}, mi_avg={}, err_avg={}, '
              'entr_H={}, cond_entr={}'.format(few_shot_quant, mi_total_dev / trial_count,
                                                 err_total_dev / trial_count, entr_total_dev / trial_count,
                                                 cond_entr_total_dev / trial_count))

    # ================ eval on test ===================
    if args.test == 'yes':
        str2, dict_result_test = predictor.eval_dev_discrete(processed_sent_test, processed_tag_Long_test,
                                                             token_embeds_test)
        print('test_summary: ' + str2 + ', beta=%f' % model.beta)
        sys.stdout.flush()
        mi_total_test += dict_result_test["MI(POS,Tag)"]
        err_total_test += dict_result_test["error"]
        entr_total_test += dict_result_test["H(POS)"]
        cond_entr_total_test += dict_result_test["H(POS|Tag)"]
        print('LOOK:summary_test: few_shot={}, mi_avg={}, err_avg={}, '
              'entr_H={}, cond_entr={}'.format(few_shot_quant, mi_total_test / trial_count,
                                                 err_total_test / trial_count, entr_total_test / trial_count,
                                                 cond_entr_total_test / trial_count))
    sys.stdout.flush()
    return


# ==========================================================================================================
# ===================================      STEM RECONSTRUCTION    ==========================================
# ==========================================================================================================
from nltk.stem import PorterStemmer as NltkPorterStemmer
def get_stem(processed_sent, stemmer, stem_dict):
    UNK_num = stem_dict['UNK']
    train_stem = []
    for sent in processed_sent:
        sent_stem = []
        for word in sent:
            word_stem = stemmer.stem(word)
            if word_stem in stem_dict:
                temp = stem_dict[word_stem]
            else:
                temp = UNK_num
            sent_stem.append(temp)
        train_stem.append(torch.LongTensor(sent_stem))
    return train_stem

def reconstruct_stem(model, data_loader, args):
    stem_dict = {}
    num2stem = {}
    stemmer = NltkPorterStemmer()
    train_stem = []
    for sent in data_loader.processed_sent:
        sent_stem = []
        for word in sent:
            word_stem = stemmer.stem(word)
            if word_stem not in stem_dict:
                stem_dict[word_stem] = len(stem_dict)
                num2stem[stem_dict[word_stem]] = word_stem
            sent_stem.append(stem_dict[word_stem])
        train_stem.append(torch.LongTensor(sent_stem))
    stem_dict['UNK'] = len(stem_dict)
    print(len(stem_dict))

    ''' training '''
    for param in model.parameters():
        param.require_grad = False

    embed_loader = Embedding_Weight(args.embedding_source, data_loader=data_loader, num_sent=args.epoch_sent)
    args.weight_decay = args.weight_decay2
    elmo_embeds_train, elmo_type_train = load_elmo(data_loader, args, embed_loader,
                                                mod='train', processed_sent_dev=None,
                                                processed_sent_test=None)

    predictor = baselines.Recon_Lemma(stem_dict, model, args)
    for e in range(args.epoch):
        out1 = predictor.train(data_loader.processed_sent, train_stem, elmo_embeds_train,
                                        sent_per_epoch=args.sent_per_epoch)
        print(out1)
    #
    ''' load the dev data '''
    processed_tag_dev, processed_sent_dev, processed_tree_dev, processed_tree_lab_dev, \
    processed_tree_Long_dev, processed_tree_lab_Long_dev, processed_tag_Long_dev = data_loader.load_dev_verbo(
        args.dataset_dev)

    token_embeds_dev, elmo_type_dev = load_elmo(data_loader, args, embed_loader,
                                                mod='dev', processed_sent_dev=processed_sent_dev,
                                                processed_sent_test=None)

    stem_dev = get_stem(processed_sent_dev, stemmer, stem_dict)
    out1, result_dict = predictor.eval_dev(processed_sent_dev, stem_dev, token_embeds_dev)
    print('dev-final-summary: {}'.format(out1))

    ''' load the test data '''
    if args.test == 'yes':
        processed_tag_test, processed_sent_test, processed_tree_test, processed_tree_lab_test, \
        processed_tree_Long_test, processed_tree_lab_Long_test, processed_tag_Long_test = data_loader.load_dev_verbo(
            args.dataset_test)

        token_embeds_test, elmo_type_test = load_elmo(data_loader, args, embed_loader,
                                                    mod='test', processed_sent_dev=None,
                                                    processed_sent_test=processed_sent_test)

        stem_test = get_stem(processed_sent_test, stemmer, stem_dict)
        out1, result_dict = predictor.eval_dev(processed_sent_test, stem_test, token_embeds_test)
        print('test-final-summary: {}'.format(out1))


def reconstruct_stem_fore(model, data_loader, args):
    '''
    the foreign version of the stem reconstruction training.
    :param model:
    :param data_loader:
    :param args:
    :return:
    '''
    stem_dict = {}
    num2stem = {}
    stemmer = NltkPorterStemmer()
    train_stem = []
    for sent in data_loader.processed_sent:
        sent_stem = []
        for word in sent:
            word_stem = stemmer.stem(word)
            if word_stem not in stem_dict:
                stem_dict[word_stem] = len(stem_dict)
                num2stem[stem_dict[word_stem]] = word_stem
            sent_stem.append(stem_dict[word_stem])
        train_stem.append(torch.LongTensor(sent_stem))
    stem_dict['UNK'] = len(stem_dict)
    print(len(stem_dict))

    ''' training steps '''
    for param in model.parameters():
        param.require_grad = False

    embed_loader = Foreign_Elmo(args.elmo_model_path, args.embedding_source)
    elmo_embeds_train, elmo_type_train = embed_loader._get_embeddings(data_loader.processed_sent)

    args.weight_decay = args.weight_decay2

    predictor = baselines.Recon_Lemma(stem_dict, model, args)
    for e in range(args.epoch):
        out1 = predictor.train(data_loader.processed_sent, train_stem, elmo_embeds_train,
                                        sent_per_epoch=args.sent_per_epoch)
        print(out1)
    #
    ''' load the dev data '''
    processed_tag_dev, processed_sent_dev, processed_tree_dev, processed_tree_lab_dev, \
    processed_tree_Long_dev, processed_tree_lab_Long_dev, processed_tag_Long_dev = data_loader.load_dev_verbo(
        args.dataset_dev)

    token_embeds_dev, elmo_type_dev = embed_loader._get_embeddings(processed_sent_dev)

    stem_dev = get_stem(processed_sent_dev, stemmer, stem_dict)
    out1, result_dict = predictor.eval_dev(processed_sent_dev, stem_dev, token_embeds_dev)
    print('dev-final-summary: {}'.format(out1))

    ''' load the test data '''
    if args.test == 'yes':
        processed_tag_test, processed_sent_test, processed_tree_test, processed_tree_lab_test, \
        processed_tree_Long_test, processed_tree_lab_Long_test, processed_tag_Long_test = data_loader.load_dev_verbo(
            args.dataset_test)

        token_embeds_test, elmo_type_test = embed_loader._get_embeddings(processed_sent_test)
        stem_test = get_stem(processed_sent_test, stemmer, stem_dict)

        out1, result_dict = predictor.eval_dev(processed_sent_test, stem_test, token_embeds_test)
        print('test-final-summary: {}'.format(out1))




# ==========================================================================================================
# ===================================   VERB-NOUN-TENSE-ANALYSIS  ==========================================
# ==========================================================================================================
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import time

def svm_eval(gamma_lst, X, y):
    final_collection = []
    final_time = []
    for gamma_val in gamma_lst:
        start_time = time.time()
        clf = svm.SVC(gamma=gamma_val)
        a = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
        final_collection.append(np.mean(a))
        elapsed_time = time.time() - start_time
        final_time.append(elapsed_time)
        # print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        sys.stdout.flush()
    best_gamma = gamma_lst[np.argmax(final_collection)]
    final_acc = np.max(final_collection)
    final_time = np.mean(final_time)
    return final_time, final_acc, best_gamma

def eval_verbs(model, data_loader, args, usage='verb'):
   
    if usage == 'verb':
        function_usage = verb_tense_process
        func_name = 'verb_tense'
    else:
        function_usage = noun_number_process
        func_name = 'noun_numb'

    ''' Train set evaluation '''
    use_train_set_ = True

    if use_train_set_:
        dict_dev_data = data_loader.load_dev_with_tags(args.dataset)
        embed_loader = Embedding_Weight(args.embedding_source, args.embedding)
        word_lst, tense_lst, name_map_lst = function_usage(dict_dev_data["processed_sent_dev"],
                                                            dict_dev_data["processed_tag_dev"],
                                                            dict_dev_data["processed_feature_dev"])

        token_embeds_dev, elmo_type_dev = load_elmo(data_loader, args, embed_loader,
                                                    mod='train', processed_sent_dev=None,
                                                    processed_sent_test=None)

        embeds_dev = name_map2embed(name_map_lst, token_embeds_dev)
        embeds_dev = torch.cat(embeds_dev)

        mean, _, _ = model.encoder.forward_sent(word_lst, [embeds_dev], 0)
        X = mean.data.numpy()
        target_dict = {}
        for elem in set(tense_lst):
            target_dict[elem] = len(target_dict)
        y = np.array([target_dict[x] for x in tense_lst])
        yy_set = defaultdict(int)
        for yy in y:
            yy_set[yy] += 1
        for key, val in yy_set.items():
            print('key:{} has percentage {}'.format(key, val/len(y)))
            sys.stdout.flush()

        ''' to make training practical by randomly selecting 5000 items '''
        # LISA
        items = np.random.choice(len(y), min(5000, len(y)), replace=False)

        X = np.array([X[x] for x in items])
        y = np.array([y[x] for x in items])

        final_time, final_acc, best_gamma = svm_eval( [1, 0.1, 0.01, 0.001, 0.0005], X, y)
        print('final_train_vib_{}:acc={}, time={}, '
              'beta={}'.format(func_name, final_acc,
                               time.strftime("%H:%M:%S", time.gmtime(final_time)),
                               model.beta))

        clf_train = svm.SVC(gamma=best_gamma)
        clf_train.fit(X, y)


        X = embeds_dev.data.numpy()
        X = np.array([X[x] for x in items])

        final_time, final_acc, _ = svm_eval( [0.001, 0.0005], X, y)
        print('final_train_elmo_{}:acc={}, time={}, '
              'beta={}'.format(func_name, final_acc,
                               time.strftime("%H:%M:%S", time.gmtime(final_time)),
                               model.beta))


    ''' Dev set evaluation '''
    dict_dev_data = data_loader.load_dev_with_tags(args.dataset_dev)
    embed_loader = Embedding_Weight(args.embedding_source, args.embedding)
    word_lst, tense_lst, name_map_lst = function_usage(dict_dev_data["processed_sent_dev"],
                                                        dict_dev_data["processed_tag_dev"],
                                                        dict_dev_data["processed_feature_dev"])

    token_embeds_dev, elmo_type_dev = load_elmo(data_loader, args, embed_loader,
                                                mod='dev', processed_sent_dev=dict_dev_data["processed_sent_dev"],
                                                processed_sent_test=None)

    embeds_dev = name_map2embed(name_map_lst, token_embeds_dev)
    embeds_dev = torch.cat(embeds_dev)
    mean, _, _ = model.encoder.forward_sent(word_lst, [embeds_dev], 0)
    X_dev = mean.data.numpy()
    target_dict = {}
    for elem in set(tense_lst):
        target_dict[elem] = len(target_dict)
    y_dev = np.array([target_dict[x] for x in tense_lst])
    # yy_set = defaultdict(int)
    # for yy in y_dev:
    #     yy_set[yy] += 1
    # for key, val in yy_set.items():
    #     print('key:{} has percentage {}'.format(key, val / len(y_dev)))
    #     sys.stdout.flush()
    # word_tense_dict = evaluation.get_word_pos_dict_thre_1(word_lst, tense_lst, 0)

    if use_train_set_:
        dev_acc = accuracy_score(y_dev, clf_train.predict(X_dev))
        print('final_train_dev_eval_vib_{}: acc={}'.format(func_name, dev_acc))


    final_time, final_acc, best_gamma = svm_eval( [1, 0.1, 0.01, 0.001, 0.0005], X_dev, y_dev)
    print('final_dev_vib_{}:acc={}, time={}, '
          'beta={}'.format(func_name, final_acc,
                           time.strftime("%H:%M:%S", time.gmtime(final_time)),
                           model.beta))
    sys.stdout.flush()

    X_dev = embeds_dev.data.numpy()
    final_time, final_acc, best_gamma = svm_eval( [0.001, 0.0005], X_dev, y_dev)
    print('final_dev_elmo_{}:acc={}, time={}, '
          'beta={}'.format(func_name, final_acc,
                           time.strftime("%H:%M:%S", time.gmtime(final_time)),
                           model.beta))
    sys.stdout.flush()
    return





def verb_tense_process(x_lst, tag_lst, feature_lst):
    word_lst = []
    tense_lst = []
    name_map = []
    for sent_ind, (sent, tags, sent_feature) in enumerate(zip(x_lst, tag_lst, feature_lst)):
        for word_ind, (word, tag, feat) in enumerate(zip(sent, tags, sent_feature)):
            if tag == 'VERB':
                try:
                    feat = feat.split('Tense=')
                    tense = feat[1].split('|')[0]
                    word_lst.append(word)
                    tense_lst.append(tense)
                    name_map.append((sent_ind, word_ind))
                except:
                    pass
    # print(tense_lst)
    print(len(word_lst), len(tense_lst), len(name_map))
    return word_lst, tense_lst, name_map

def noun_number_process(x_lst, tag_lst, feature_lst):
    word_lst = []
    tense_lst = []
    name_map = []
    for sent_ind, (sent, tags, sent_feature) in enumerate(zip(x_lst, tag_lst, feature_lst)):
        for word_ind, (word, tag, feat) in enumerate(zip(sent, tags, sent_feature)):
            if tag == 'NOUN':
                try:
                    feat = feat.split('Number=')
                    tense = feat[1].split('|')[0]
                    word_lst.append(word)
                    tense_lst.append(tense)
                    name_map.append((sent_ind, word_ind))
                except:
                    pass
    print(len(word_lst), len(tense_lst), len(name_map))
    return word_lst, tense_lst, name_map

def name_map2embed(name_map, embeds):
    lst = []
    for sent_ind, word_ind in name_map:
        lst.append(embeds[sent_ind][word_ind])
    return lst

def transitive_data_process( x_lst, tag_lst, tree_lst, edge_lst):
    
    word_lst = []
    tag2_lst = []
    namemap_lst = []
    for sent_index, (words, poss, trees, edges) in enumerate(zip(x_lst, tag_lst, tree_lst, edge_lst)):
        child_lst = [[] for _ in range(len(words))]
        for index, info  in enumerate(zip(words, poss, trees, edges)):
            word, pos, tree, edge = info
            if tree == 0:
                continue
            child_lst[tree-1].append((index, word, pos, edge))
        for index, info  in enumerate(zip(words, poss, trees, edges)):
            word, pos, tree, edge = info
            flag = False
            if pos == 'VERB':
                child_edges = [child_info[3] for child_info in child_lst[index]]
                if 'dobj' in child_edges and 'iobj' in child_edges:
                    word_lst.append(word)
                    namemap_lst.append((sent_index, index))
                    tag2_lst.append('Ditrans')
                elif 'dobj' in child_edges:
                    word_lst.append(word)
                    namemap_lst.append((sent_index, index))
                    tag2_lst.append('Trans')
                else:
                    word_lst.append(word)
                    namemap_lst.append((sent_index, index))
                    tag2_lst.append('Intrans')
    # print(transitive_verbs)
    # print('*' * 10)
    # print(intransitive_verbs)
    return word_lst, tag2_lst, namemap_lst


def SVM_classifier_verb(model, data_loader, args):
    ''' Train set evaluation '''
    use_train_set_ = True
    func_name = 'verb_subcat'
    if use_train_set_:
        dict_dev_data = data_loader.load_dev_with_tags(args.dataset)
        embed_loader = Embedding_Weight(args.embedding_source, args.embedding)
        word_lst, tense_lst, name_map_lst = transitive_data_process(dict_dev_data["processed_sent_dev"],
                                                                     dict_dev_data["processed_tag_dev"],
                                                                     dict_dev_data["processed_tree_dev"],
                                                                     dict_dev_data["processed_tree_lab_dev"])

        print(len(word_lst), len(tense_lst), len(name_map_lst))

        token_embeds_dev, elmo_type_dev = load_elmo(data_loader, args, embed_loader,
                                                    mod='train', processed_sent_dev=None,
                                                    processed_sent_test=None)

        embeds_dev = name_map2embed(name_map_lst, token_embeds_dev)
        embeds_dev = torch.cat(embeds_dev)

        mean, _, _ = model.encoder.forward_sent(word_lst, [embeds_dev], 0)
        X = mean.data.numpy()
        target_dict = {}
        for elem in set(tense_lst):
            target_dict[elem] = len(target_dict)
        y = np.array([target_dict[x] for x in tense_lst])
        yy_set = defaultdict(int)
        for yy in y:
            yy_set[yy] += 1
        for key, val in yy_set.items():
            print('key:{} has percentage {}'.format(key, val / len(y)))
        sys.stdout.flush()

        ''' to make training practical by randomly selecting 5000 items '''
        items = np.random.choice(len(y), min(5000, len(y)), replace=False)

        X = np.array([X[x] for x in items])
        y = np.array([y[x] for x in items])

        final_time, final_acc, best_gamma = svm_eval( [1, 0.1, 0.01, 0.001, 0.0005], X, y)
        print('final_train_vib_{}:acc={}, time={}, '
              'beta={}'.format(func_name, final_acc,
                               time.strftime("%H:%M:%S", time.gmtime(final_time)),
                               model.beta))

        clf_train = svm.SVC(gamma=best_gamma)
        clf_train.fit(X, y)

        X = embeds_dev.data.numpy()
        X = np.array([X[x] for x in items])

        final_time, final_acc, _ = svm_eval([0.001, 0.0005], X, y)
        print('final_train_elmo_{}:acc={}, time={}, '
              'beta={}'.format(func_name, final_acc,
                               time.strftime("%H:%M:%S", time.gmtime(final_time)),
                               model.beta))


    ''' Dev set evaluation '''
    dict_dev_data = data_loader.load_dev_with_tags(args.dataset_dev)
    embed_loader = Embedding_Weight(args.embedding_source, args.embedding)
    word_lst, tense_lst, name_map_lst = transitive_data_process(dict_dev_data["processed_sent_dev"],
                                                                 dict_dev_data["processed_tag_dev"],
                                                                 dict_dev_data["processed_tree_dev"],
                                                                 dict_dev_data["processed_tree_lab_dev"])

    token_embeds_dev, elmo_type_dev = load_elmo(data_loader, args, embed_loader,
                                                mod='dev', processed_sent_dev=dict_dev_data["processed_sent_dev"],
                                                processed_sent_test=None)

    embeds_dev = name_map2embed(name_map_lst, token_embeds_dev)
    embeds_dev = torch.cat(embeds_dev)
    mean, _, _ = model.encoder.forward_sent(word_lst, [embeds_dev], 0)
    X_dev = mean.data.numpy()
    target_dict = {}
    for elem in set(tense_lst):
        target_dict[elem] = len(target_dict)
    y_dev = np.array([target_dict[x] for x in tense_lst])

    if use_train_set_:
        dev_acc = accuracy_score(y_dev, clf_train.predict(X_dev))
        print('final_train_dev_eval_vib_{}: acc={}'.format(func_name, dev_acc))




    final_time, final_acc, best_gamma = svm_eval( [1, 0.1, 0.01, 0.001, 0.0005], X_dev, y_dev)
    print('final_dev_vib_{}:acc={}, time={}, '
          'beta={}'.format('verb_subcat', final_acc,
                           time.strftime("%H:%M:%S", time.gmtime(final_time)),
                           model.beta))
    sys.stdout.flush()

    X_dev = embeds_dev.data.numpy()
    final_time, final_acc, best_gamma = svm_eval( [0.001, 0.0005], X_dev, y_dev)
    print('final_dev_elmo_{}:acc={}, time={}, '
          'beta={}'.format('verb_subcat', final_acc,
                           time.strftime("%H:%M:%S", time.gmtime(final_time)),
                           model.beta))
    sys.stdout.flush()

    return


def SVM_classifier_verb_fore(model, data_loader, args):
    ''' Train set evaluation '''
    use_train_set_ = True
    func_name = 'verb_subcat'
    if use_train_set_:
        dict_dev_data = data_loader.load_dev_with_tags(args.dataset)
        embed_loader = Foreign_Elmo(args.elmo_model_path, args.embedding_source)

        word_lst, tense_lst, name_map_lst = transitive_data_process(dict_dev_data["processed_sent_dev"],
                                                                     dict_dev_data["processed_tag_dev"],
                                                                     dict_dev_data["processed_tree_dev"],
                                                                     dict_dev_data["processed_tree_lab_dev"])

        print(len(word_lst), len(tense_lst), len(name_map_lst))

        token_embeds_dev, elmo_type_dev = embed_loader._get_embeddings(data_loader.processed_sent)

        embeds_dev = name_map2embed(name_map_lst, token_embeds_dev)
        embeds_dev = torch.cat(embeds_dev)

        mean, _, _ = model.encoder.forward_sent(word_lst, [embeds_dev], 0)
        X = mean.data.numpy()
        target_dict = {}
        for elem in set(tense_lst):
            target_dict[elem] = len(target_dict)
        y = np.array([target_dict[x] for x in tense_lst])
        yy_set = defaultdict(int)
        for yy in y:
            yy_set[yy] += 1
        for key, val in yy_set.items():
            print('key:{} has percentage {}'.format(key, val / len(y)))
        sys.stdout.flush()

        ''' to make training practical by randomly selecting 5000 items '''
        items = np.random.choice(len(y), min(8000, len(y)), replace=False)
        # this is bad, note that if we need to choose randomly, maybe very nesessary to choose specifically for both trans and intrans.

        X = np.array([X[x] for x in items])
        y = np.array([y[x] for x in items])

        final_time, final_acc, best_gamma = svm_eval( [1, 0.1, 0.01, 0.001, 0.0005], X, y)
        print('final_train_vib_{}:acc={}, time={}, '
              'beta={}'.format(func_name, final_acc,
                               time.strftime("%H:%M:%S", time.gmtime(final_time)),
                               model.beta))

        clf_train = svm.SVC(gamma=best_gamma)
        clf_train.fit(X, y)

        X = embeds_dev.data.numpy()
        X = np.array([X[x] for x in items])

        final_time, final_acc, _ = svm_eval([0.001, 0.0005], X, y)
        print('final_train_elmo_{}:acc={}, time={}, '
              'beta={}'.format(func_name, final_acc,
                               time.strftime("%H:%M:%S", time.gmtime(final_time)),
                               model.beta))


    ''' Dev set evaluation '''
    # processed_tag_dev, processed_sent_dev, processed_tree_dev, processed_tree_lab_dev, \
    # processed_tree_Long_dev, processed_tree_lab_Long_dev = data_loader.load_dev(args.dataset_dev)



    dict_dev_data = data_loader.load_dev_with_tags(args.dataset_dev)
    word_lst, tense_lst, name_map_lst = transitive_data_process(dict_dev_data["processed_sent_dev"],
                                                                 dict_dev_data["processed_tag_dev"],
                                                                 dict_dev_data["processed_tree_dev"],
                                                                 dict_dev_data["processed_tree_lab_dev"])

    token_embeds_dev, elmo_type_dev = embed_loader._get_embeddings(dict_dev_data["processed_sent_dev"])

    # token_embeds_dev, elmo_type_dev = load_elmo(data_loader, args, embed_loader,
    #                                             mod='dev', processed_sent_dev=dict_dev_data["processed_sent_dev"],
    #                                             processed_sent_test=None)

    embeds_dev = name_map2embed(name_map_lst, token_embeds_dev)
    embeds_dev = torch.cat(embeds_dev)
    mean, _, _ = model.encoder.forward_sent(word_lst, [embeds_dev], 0)
    X_dev = mean.data.numpy()
    target_dict = {}
    for elem in set(tense_lst):
        target_dict[elem] = len(target_dict)
    y_dev = np.array([target_dict[x] for x in tense_lst])

    if use_train_set_:
        dev_acc = accuracy_score(y_dev, clf_train.predict(X_dev))
        print('final_train_dev_eval_vib_{}: acc={}'.format(func_name, dev_acc))




    final_time, final_acc, best_gamma = svm_eval( [1, 0.1, 0.01, 0.001, 0.0005], X_dev, y_dev)
    print('final_dev_vib_{}:acc={}, time={}, '
          'beta={}'.format('verb_subcat', final_acc,
                           time.strftime("%H:%M:%S", time.gmtime(final_time)),
                           model.beta))
    sys.stdout.flush()

    X_dev = embeds_dev.data.numpy()
    final_time, final_acc, best_gamma = svm_eval( [0.001, 0.0005], X_dev, y_dev)
    print('final_dev_elmo_{}:acc={}, time={}, '
          'beta={}'.format('verb_subcat', final_acc,
                           time.strftime("%H:%M:%S", time.gmtime(final_time)),
                           model.beta))
    sys.stdout.flush()

    return


# ==========================================================================================================
# ===================================  TSNE - plots  =======================================================
# ==========================================================================================================

def get_tsne_plot(model, type_embeds, token_embeds, embed_loader, word_pos_dict, x_lst, pos_lst, exp_mode = '', out_path=''):
    # word type part.
    with torch.no_grad():
        '''
            plot data for word type level.  
        '''
        if exp_mode == 'dis_type_plot':
            lst = []
            pos0_lst = []
            for word, index in word_pos_dict.items():
                lst.append(word)
                pos0_lst.append(word_pos_dict[word])
            elmo_type = embed_loader.elmo_embeddings_first([lst], 1)
            if model.encoder.embedding_source == 'elmo_0':
                alphas = model.encoder.forward_sent(lst, elmo_type, 0)
            else:
                alphas = model.variational_encoder.forward_sent(lst, elmo_type, 0)
                embeds = model.tag_embeddings(alphas)
                _, max_alpha = torch.max(alphas, dim=-1)


            print(embeds.data.numpy().shape)
            vis.robost_tsne(embeds[:,0,:].data.numpy(), '../_tsne_type5_embed_elmo-.pdf', pos0_lst, dim=2, word_lst=lst)
            # vis.spectrum_tsne(embeds.data.numpy(), '../_tsne_type3_embed10.pdf', pos0_lst, dim=2, word_lst=lst)
            # vis.spectrum_tsne(embeds.data.numpy(), '../_tsne_type3_embed10.pdf', pos0_lst, dim=2, word_lst=lst)

        if exp_mode == 'dis_token_plot':
            alphas_tokens, alphas2_tokens, alpha_embed_lst = [], [], []
            book_lst, pos1_lst, word_lst, mi_c_lst, mi_s_lst = [], [], [], [], []
            for index, data in enumerate(zip(x_lst, pos_lst)):
                sent, pos = data
                alphas_ = model.encoder.forward_sent(sent, token_embeds, index)
                if hasattr(model, 'tag_embeddings'):
                    alphas_emb = model.tag_embeddings(alphas_)
                    alpha_embed_lst.append(alphas_emb)
                alphas_tokens.append(alphas_)
                # alphas2_tokens.append(alphas2_)
                book_lst += [(index, ind) for ind, x in enumerate(sent)]
                pos1_lst += pos
                word_lst += sent
                
            token_lst = torch.cat(alphas_tokens, dim=0).squeeze(1).data.numpy()
            token_lst_embeds = torch.cat(alpha_embed_lst, dim=0).squeeze(1).data.numpy()
            elmo_temp = torch.cat(token_embeds, dim=0).squeeze(1).data.numpy()
            # vis.robost_tsne(token_lst, '../_tsne_type3_embed_heavy-.pdf', pos1_lst, dim=2,
            #                 word_lst=word_lst)
            vis.robost_tsne(elmo_temp, '../_tsne_type3_embed_elmo-.pdf', pos1_lst, dim=2,word_lst=word_lst)
            vis.robost_tsne(token_lst_embeds, '../_tsne_type4_embed_heavy-.pdf', pos1_lst, dim=2,
                            word_lst=word_lst)

        '''
            continuous case
        '''

        if exp_mode == 'con_token_plot':
            alphas_tokens = []
            elmo_tokens = []
            alphas2_tokens = []
            alpha_embed_lst = []
            book_lst = []
            pos1_lst = []
            word_lst = []
            mi_c_lst = []
            mi_s_lst = []
            for index, data in enumerate(zip(x_lst, pos_lst)):
                sent, pos = data
                alphas_, _, _ = model.encoder.forward_sent(sent, token_embeds, index) # alpha here is actually mu
                for elmo_single in token_embeds[index]:
                    elmo_tokens.append(elmo_single)
                alphas_tokens.append(alphas_)
                book_lst += [(index, ind) for ind, x in enumerate(sent)]
                pos1_lst += pos
                word_lst += sent

            print(alphas_tokens)
            token_lst = torch.cat(alphas_tokens, dim=0).squeeze(1).data.numpy()

            np.random.seed(9)
            cand = np.random.choice(len(token_lst), 5000, replace=False)
            token_cand = token_lst[cand]
            elmo_tokens = torch.cat(elmo_tokens, dim=0).squeeze(1).data.numpy()
            pos_cand = [pos1_lst[x] for x in cand]
            word_cand = [word_lst[x] for x in cand]
            elmo_cand = [elmo_tokens[x] for x in cand]
            # vis.robost_tsne(elmo_cand, '../_tsne_token5.pdf', pos_cand, dim=2, word_lst=word_cand)
            vis.robost_tsne(token_cand, '../_tsne_token4.pdf', pos_cand, dim=2, word_lst=word_cand)


    return


