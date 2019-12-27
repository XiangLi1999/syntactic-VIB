'''
    Xiang Li
    xli150@jhu.edu
'''
from glo import get_logger, experiment, Option, Global, VarDict
import numpy as np
import torch
import argparse, random, time
import baselines
from lang_select import fetch_paths
from gaussian_tag import Continuous_VIB
from conllu_handler import Foreign_Elmo, Data_Loader2, Embedding_Weight
from discrete_tag import Discrete_VIB
from collections import Counter

from torch import autograd
import sys, pickle
from experiment_helper import *
from sklearn.decomposition import PCA


logger = get_logger()

#################################################################################

def _start_():
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    print()
    print('[ARGUMENTS]')
    print(args)
    print()

    return 

def _load_data():

    if args.lang !=  'en':
        dict_path = fetch_paths(args.lang)
        print('[language path {}]'.format(args.lang))
        print(dict_path)
        args.dataset = dict_path['train']
        args.dataset_dev = dict_path['dev']
        args.dataset_test = dict_path['test']
        args.elmo_model_path = dict_path['elmo']

        # '''for local testing -- del on marcc'''
        # args.dataset = '/Users/xiangli/Desktop/latstruct/workspace/data/UD_Arabic/ar-ud-dev.conllu'
        # args.dataset_dev = '/Users/xiangli/Desktop/latstruct/workspace/data/UD_Arabic/ar-ud-dev.conllu'
        # args.dataset_test = '/Users/xiangli/Desktop/latstruct/workspace/data/UD_Arabic/ar-ud-dev.conllu'
        # args.elmo_model_path = '/Users/xiangli/Downloads/136'

    else:
        args.dataset_dev = args.dataset_base  + 'en-ud-dev.conllu'
        args.dataset_test = args.dataset_base  + 'en-ud-test.conllu'
        args.dataset = args.dataset_base  + 'en-ud-train.conllu'

    data_loader = Data_Loader2(args)
    args.num_labels = len(data_loader.label_dict)
    args.epoch_sent = len(data_loader.corpus)
    return data_loader

def _load_pretrain_emb(data_loader, en_batch_dev = None, en_batch_test=None):
    if  args.embedding_source == 'elmo_1' or args.embedding_source == 'elmo_2' or  args.embedding_source == 'elmo_0':
        ext_dim = 1024 # elmo size is 1024
        args.embedding_dim = ext_dim
        args.hidden_dim = (ext_dim + args.tag_dim) // 2
    elif args.embedding_source[:4] == 'BERT':
        ext_dim = 768 # bert dim is 768 per layer
        args.embedding_dim = ext_dim
        args.hidden_dim = (ext_dim + args.tag_dim) // 2
    else:
        print('embedding name is not available, double check.')

    pre_compute_dict = {}
    if args.lang != 'en':
        embed_loader = Foreign_Elmo(args.elmo_model_path, args.embedding_source, device=args.device)
        elmo_embeds_train, non_context_embeds_train = embed_loader._get_embeddings(data_loader.en_batch)

        pre_compute_dict['elmo_embeds_train'] = elmo_embeds_train
        pre_compute_dict['non_context_embeds_train'] = non_context_embeds_train


        if en_batch_dev is not None:
            elmo_embeds_dev, non_context_embeds_dev = embed_loader._get_embeddings(en_batch_dev)
            pre_compute_dict['elmo_embeds_dev'] = elmo_embeds_dev
            pre_compute_dict['non_context_embeds_dev'] = non_context_embeds_dev

        if en_batch_test is not None:
            elmo_embeds_test, non_context_embeds_test = embed_loader._get_embeddings(en_batch_test)
            pre_compute_dict['elmo_embeds_test'] = elmo_embeds_test
            pre_compute_dict['non_context_embeds_test'] = non_context_embeds_test


    else:
        embed_loader = Embedding_Weight(args.embedding_source, data_loader=data_loader, num_sent=args.epoch_sent, device=args.device)

        elmo_embeds_train, non_context_embeds_train = load_elmo_batch(data_loader, args, embed_loader,
                                                                      mod='train', processed_sent_dev=None,
                                                                      processed_sent_test=None)
        pre_compute_dict['elmo_embeds_train'] = elmo_embeds_train
        pre_compute_dict['non_context_embeds_train'] = non_context_embeds_train

        if en_batch_dev is not None:
            elmo_embeds_dev, non_context_embeds_dev = load_elmo_batch(None, args, embed_loader,
                                                                      mod='dev', processed_sent_dev=en_batch_dev,
                                                                      processed_sent_test=None)
            pre_compute_dict['elmo_embeds_dev'] = elmo_embeds_dev
            pre_compute_dict['non_context_embeds_dev'] = non_context_embeds_dev

        if en_batch_test is not None:
            elmo_embeds_test, non_context_embeds_test = load_elmo_batch(None, args, embed_loader,
                                                                          mod='test', processed_sent_dev=None,
                                                                          processed_sent_test=en_batch_test)
            pre_compute_dict['elmo_embeds_test'] = elmo_embeds_test
            pre_compute_dict['non_context_embeds_test'] = non_context_embeds_test

    return pre_compute_dict, embed_loader

def fine_parse(args):
    _start_()

    # ==================== LOAD DATA ==========================
    data_loader = _load_data()

    corpus_dev, batch_idx_dev, en_batch_dev, processed_tree_dev = data_loader.load_dev(args.dataset_dev)

    if args.test == 'yes':
        corpus_test, batch_idx_test, en_batch_test, processed_tree_test = data_loader.load_dev(args.dataset_test)
    else:
        en_batch_test = None

    # # ==================== LOAD ELMo/BERT ==========================
    if  args.embedding_source == 'elmo_1' or args.embedding_source == 'elmo_2' or  args.embedding_source == 'elmo_0':
        ext_dim = 1024 # elmo size is 1024
        args.embedding_dim = ext_dim
        args.hidden_dim = (ext_dim + args.tag_dim) // 2
    elif args.embedding_source[:4] == 'BERT':
        ext_dim = 712 # bert dim is 712
        args.embedding_dim = ext_dim
        args.hidden_dim = (ext_dim + args.tag_dim) // 2
    else:
        print('embedding name is not available, double check.')

    pre_compute_dict = {}
    if args.lang != 'en':
        embed_loader = Foreign_Elmo(args.elmo_model_path, args.embedding_source, device=args.device)
    else:
        embed_loader = Embedding_Weight(args.embedding_source, data_loader=data_loader, num_sent=args.epoch_sent, 
            device=args.device, requires_grad=True)

    # ==================== MODEL =======================
    args.embed_loader = embed_loader
    model = baselines.finetune_info_parse(args)
    model.to(args.device)

    if args.mode == 'evaluate':
        model = model.load_model(args.checkpoint_path)
        # =================== evaluate on dev ==========================
        print('dev data total:', len(en_batch_dev))
        result_dict = model.parse_dev(
            en_batch_dev, corpus_dev)

        align_err_w = result_dict["align_err_w"]
        nlogp_w = result_dict["nlogp_w"]
        align_err_s = result_dict["align_err_s"]
        nlogp_s = result_dict["nlogp_s"]
        kl_s = result_dict["kl_s"]
        LAS = result_dict["LAS"]

        H_YT = nlogp_s
        print(
            "summary_dev_final_iden: tag_dim=%d, dev, elmo=%s,"
            " MI(X,T)=%.5f, H(Y|T)=%.5f, err=%.5f, "
            "LAS=%.3f, UAS=%.3f"
            % (args.tag_dim, args.embedding_source,
               kl_s, H_YT, align_err_w,
               LAS, 1 - align_err_w))

        sys.stdout.flush()

        # =================== evaluate on test ==========================
        if args.test == 'yes':
            print('test data total:', len(en_batch_test))
            result_dict = model.parse_dev(
                en_batch_test, corpus_test)

            align_err_w = result_dict["align_err_w"]
            nlogp_w = result_dict["nlogp_w"]
            align_err_s = result_dict["align_err_s"]
            nlogp_s = result_dict["nlogp_s"]
            kl_s = result_dict["kl_s"]
            LAS = result_dict["LAS"]

            H_YT = nlogp_s
            print(
                "summary_test_final_iden:tag_dim=%d, dev, elmo=%s,"
                " MI(X,T)=%.5f, H(Y|T)=%.5f, err=%.5f, "
                "LAS=%.3f, UAS=%.3f"
                % (args.tag_dim, args.embedding_source,
                   kl_s, H_YT, align_err_w,
                   LAS, 1 - align_err_w))
            sys.stdout.flush()
        return 0

    if args.mode == 'train':
        combined_epoch_sent = 0

        # print(elmo_embeds_train[0].shape)
        print('training data total:', len(data_loader.en_batch))
        for e in range(args.epoch):
            print('epoch %d/%d:' % (e, args.epoch))

            result_dict = model.train(
                data_loader.en_batch,
                data_loader.corpus,
                args.sent_per_epoch,
            )

            align_err_w = result_dict["align_err_w"]
            nlogp_w = result_dict["nlogp_w"]
            align_err_s = result_dict["align_err_s"]
            nlogp_s = result_dict["nlogp_s"]
            kl_s = result_dict["kl_s"]
            LAS = result_dict["LAS"]

            combined_epoch_sent += args.sent_per_epoch
            # =========== save model based on the amount of sentence already trained. =============
            if combined_epoch_sent >= 200:
                model.save_model(args.save_path + '_epoch_%d' % e)
                combined_epoch_sent = 0
                print("UAS=%.5f, NLL=%.5f, LAS=%.5f, KL=%.5f" % (1 - align_err_w, nlogp_w, LAS, kl_s))
                sys.stdout.flush()

        print("finished training [DONE]")
        model.save_model(args.save_path)


        # =================== evaluate on full dev ==========================
        print('dev data total:', len(en_batch_dev))
        result_dict = model.parse_dev(
            en_batch_dev, corpus_dev)

        align_err_w = result_dict["align_err_w"]
        nlogp_w = result_dict["nlogp_w"]
        align_err_s = result_dict["align_err_s"]
        nlogp_s = result_dict["nlogp_s"]
        kl_s = result_dict["kl_s"]
        LAS = result_dict["LAS"]

        H_YT = nlogp_s
        print(
            "summary_dev_final_iden: tag_dim=%d, dev, elmo=%s,"
            " MI(X,T)=%.5f, H(Y|T)=%.5f, err=%.5f, "
            "LAS=%.3f, UAS=%.3f"
            % (args.tag_dim, args.embedding_source,
               kl_s, H_YT, align_err_w,
               LAS, 1 - align_err_w))
        sys.stdout.flush()


        # =================== evaluate on full test ==========================
        if args.test == 'yes':
            print('test data total:', len(en_batch_test))
            result_dict = model.parse_dev(
                en_batch_test, corpus_test)

            align_err_w = result_dict["align_err_w"]
            nlogp_w = result_dict["nlogp_w"]
            align_err_s = result_dict["align_err_s"]
            nlogp_s = result_dict["nlogp_s"]
            kl_s = result_dict["kl_s"]
            LAS = result_dict["LAS"]
            
            H_YT = nlogp_s
            print(
                "summary_test_final_iden:tag_dim=%d, dev, elmo=%s,"
                " MI(X,T)=%.5f, H(Y|T)=%.5f, err=%.5f, "
                "LAS=%.3f, UAS=%.3f"
                % (args.tag_dim, args.embedding_source,
                   kl_s, H_YT, align_err_w,
                   LAS, 1 - align_err_w))
            sys.stdout.flush()

        return 0



def iden_parse(args):
    _start_()

    # ==================== LOAD DATA ==========================
    data_loader = _load_data()

    corpus_dev, batch_idx_dev, en_batch_dev, processed_tree_dev = data_loader.load_dev(args.dataset_dev)

    if args.test == 'yes':
        corpus_test, batch_idx_test, en_batch_test, processed_tree_test = data_loader.load_dev(args.dataset_test)
    else:
        en_batch_test = None

    # ==================== LOAD ELMo/BERT ==========================
    pre_compute_dict, embed_loader = _load_pretrain_emb(data_loader, en_batch_dev=en_batch_dev, en_batch_test=en_batch_test)
    elmo_embeds_train = pre_compute_dict['elmo_embeds_train']
    non_context_embeds_train = pre_compute_dict['non_context_embeds_train']
    elmo_embeds_dev, non_context_embeds_dev = pre_compute_dict['elmo_embeds_dev'], pre_compute_dict['non_context_embeds_dev']
    if args.test == 'yes':
        elmo_embeds_test, non_context_embeds_test = pre_compute_dict['elmo_embeds_test'], pre_compute_dict['non_context_embeds_test']

    # ==================== PRE-PROCESS =======================

    if args.task == "PCA":
        print('using PCA with dimension %d' %args.tag_dim)
        temp_elmo = [xx.contiguous().view(-1, args.embedding_dim) for xx in elmo_embeds_train]
        temp = torch.cat(temp_elmo, dim=0).squeeze(1)
        print(temp.shape)
        pca = PCA(n_components=args.tag_dim)
        pca.fit(temp.data)
        result = pca.transform(temp.squeeze(1).data)
        result = torch.tensor(result).unsqueeze(1).float()
        start = 0
        elmo_embeds_train_pca = []
        for elem in elmo_embeds_train:
            end = start + elem.size(0) * elem.size(1)
            elmo_embeds_train_pca.append(result[start:end].reshape(elem.size(0), elem.size(1), -1))
            start = end
        elmo_embeds_train = elmo_embeds_train_pca

        # dev data 
        temp_elmo = [xx.contiguous().view(-1, args.embedding_dim) for xx in elmo_embeds_dev]
        temp = torch.cat(temp_elmo, dim=0).squeeze(1)
        result = pca.transform(temp.squeeze(1).data)
        result = torch.tensor(result).unsqueeze(1).float()
        start = 0
        elmo_embeds_dev_pca = []
        for elem in elmo_embeds_dev:
            end = start + elem.size(0) * elem.size(1)
            elmo_embeds_dev_pca.append(result[start:end].reshape(elem.size(0), elem.size(1), -1))
            start = end
        elmo_embeds_dev = elmo_embeds_dev_pca

        # test data 
        if args.test == 'yes':
            temp_elmo = [xx.contiguous().view(-1, args.embedding_dim) for xx in elmo_embeds_test]
            temp = torch.cat(temp_elmo, dim=0).squeeze(1)
            result = pca.transform(temp.squeeze(1).data)
            result = torch.tensor(result).unsqueeze(1).float()
            start = 0
            elmo_embeds_test_pca = []
            for elem in elmo_embeds_test:
                end = start + elem.size(0) * elem.size(1)
                elmo_embeds_test_pca.append(result[start:end].reshape(elem.size(0), elem.size(1), -1))
                start = end
            elmo_embeds_test = elmo_embeds_test_pca



    elif args.task == 'CLEAN':
        args.embedding_dim = 1024

    else:
        args.tag_dim=1024


    # ==================== MODEL =======================

    model = baselines.baseline_cont_parse(args)
    model.to(args.device)

    if args.mode == 'train':

        training_total_data = 0
        for elem in data_loader.en_batch:
            training_total_data += len(elem)
        print('the total amount of training sentence is {}'.format(training_total_data))
        print('total training batch number:', len(data_loader.corpus))
        print('tag_dim=%d, beta=%f' %(args.tag_dim, args.beta))
        print('start_training:')

        delta_temp = 1 / len(data_loader.corpus) 
        for e in range(args.epoch):
            logger.info('epoch %d/%d:' % (e, args.epoch))
            result_dict = model.train_batch(
                data_loader.corpus,
                None, 
                elmo_embeds_train)

            # =========== gather stats =============



            align_err_w = result_dict["align_err_w"]
            nlogp_w = result_dict["nlogp_w"]
            align_err_s = result_dict["align_err_s"]
            nlogp_s = result_dict["nlogp_s"]
            kl_s = result_dict["kl_w"]
            LAS = result_dict["LAS"]
            kl_s2 = result_dict["kl_w2"]
            H_YT = nlogp_s

            # =========== save model =============
            if True:
                model.save_model(args.save_path + '_epoch_%d' %e)
                print(
                "train_epoch%i: tag_dim=%d, dev, elmo=%s,"
                " MI(X,T)=%.5f, H(Y|T)=%.5f,  MI(X,Ti|Xi)=%.5f, err=%.5f, "
                "LAS=%.3f, UAS=%.3f, KL1=%.3f, KL2=%.3f"
                % (e, args.tag_dim, args.embedding_source,
                   kl_s, H_YT, kl_s2, align_err_w,
                   LAS, 1 - align_err_w, kl_s, kl_s2))

                # print("UAS=%.5f, NLL=%.5f, LAS=%.5f, KL=%.5f" %(1-align_err_w, nlogp_w, LAS, kl_s))
                sys.stdout.flush()

            # ========== check dev set ===========
            result_dict = model.parse_dev_batch(
                corpus_dev,
                elmo_embeds_dev,
                args.out_path + '_middev')

            align_err_w = result_dict["align_err_w"]
            nlogp_w = result_dict["nlogp_w"]
            align_err_s = result_dict["align_err_s"]
            nlogp_s = result_dict["nlogp_s"]
            kl_s = result_dict["kl_w"]
            LAS = result_dict["LAS"]
            kl_s2 = result_dict["kl_w2"]

            # print("alignment error rate is %.5f, negative log-likelihood per word is %.5f" % (align_err_w, nlogp_w))
            # print("alignment error per sentence is %.5f, NLL per sentence is %.5f" % (align_err_s, nlogp_s))
            # print("KL1 is %.3f, KL2 is %.3f" % (kl_s, kl_s2))

            H_YT = nlogp_s
            print(
                "dev_epoch%i: tag_dim=%d, dev, elmo=%s,"
                " MI(X,T)=%.5f,H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
                "LAS=%.3f, UAS=%.3f, KL1=%.3f, KL2=%.3f"
                % (e, args.tag_dim, args.embedding_source,
                   kl_s, H_YT,kl_s2, align_err_w,
                   LAS, 1 - align_err_w, kl_s, kl_s2))
            sys.stdout.flush()

            if e == 20:
                print('mid-point evaluation:')
                print('train data total batches:', len(data_loader.corpus))
                result_dict = model.parse_dev_batch(
                    data_loader.corpus,
                    elmo_embeds_train,
                    args.out_path + '_midtrain')

                align_err_w = result_dict["align_err_w"]
                nlogp_w = result_dict["nlogp_w"]
                align_err_s = result_dict["align_err_s"]
                nlogp_s = result_dict["nlogp_s"]
                kl_s = result_dict["kl_w"]
                LAS = result_dict["LAS"]
                kl_s2 = result_dict["kl_w2"]

                # print("alignment error rate is %.5f, negative log-likelihood per word is %.5f" % (align_err_w, nlogp_w))
                # print("alignment error per sentence is %.5f, NLL per sentence is %.5f" % (align_err_s, nlogp_s))
                # print("KL1 is %.3f, KL2 is %.3f" % (kl_s, kl_s2))

                H_YT = nlogp_s
                print(
                    "summary_train_mid: tag_dim=%d, train, elmo=%s,"
                    " MI(X,T)=%.5f, H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
                    "LAS=%.3f, UAS=%.3f"
                    % (args.tag_dim, args.embedding_source,
                       kl_s, H_YT, kl_s2, align_err_w,
                       LAS, 1 - align_err_w))
                sys.stdout.flush()


        print("finished training [DONE]")
        model.save_model(args.save_path)

        # =================== evaluate on full train ==========================

        print('train data total batches:', len(data_loader.corpus))
        result_dict = model.parse_dev_batch(
            data_loader.corpus,
            elmo_embeds_train, args.out_path + '_train')

        align_err_w = result_dict["align_err_w"]
        nlogp_w = result_dict["nlogp_w"]
        align_err_s = result_dict["align_err_s"]
        nlogp_s = result_dict["nlogp_s"]
        kl_s = result_dict["kl_w"]
        LAS = result_dict["LAS"]
        kl_s2 = result_dict["kl_w2"]

        # print("alignment error rate is %.5f, negative log-likelihood per word is %.5f" % (align_err_w, nlogp_w))
        # print("alignment error per sentence is %.5f, NLL per sentence is %.5f" % (align_err_s, nlogp_s))
        # print("KL1 is %.3f, KL2 is %.3f" % (kl_s, kl_s2))

        H_YT = nlogp_s
        print(
            "summary_train_final: tag_dim=%d, train, elmo=%s,"
            " MI(X,T)=%.5f, H(Y|T)=%.5f,MI(X,Ti|Xi)=%.5f, err=%.5f, "
            "LAS=%.3f, UAS=%.3f"
            % (args.tag_dim, args.embedding_source,
               kl_s, H_YT, kl_s2, align_err_w,
               LAS, 1 - align_err_w))
        sys.stdout.flush()

        # ======================== eval on dev set ===============================
        print('dev data total batches:', len(corpus_dev))
        result_dict = model.parse_dev_batch(
            corpus_dev,
            elmo_embeds_dev,
            args.out_path + '_dev')

        align_err_w = result_dict["align_err_w"]
        nlogp_w = result_dict["nlogp_w"]
        align_err_s = result_dict["align_err_s"]
        nlogp_s = result_dict["nlogp_s"]
        kl_s = result_dict["kl_w"]
        LAS = result_dict["LAS"]
        kl_s2 = result_dict["kl_w2"]

        # print("alignment error rate is %.5f, negative log-likelihood per word is %.5f" % (align_err_w, nlogp_w))
        # print("alignment error per sentence is %.5f, NLL per sentence is %.5f" % (align_err_s, nlogp_s))
        # print("KL1 is %.3f, KL2 is %.3f" % (kl_s, kl_s2))

        H_YT = nlogp_s
        print(
            "summary_dev_final: tag_dim=%d, dev, elmo=%s,"
            " MI(X,T)=%.5f,H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
            "LAS=%.3f, UAS=%.3f"
            % (args.tag_dim, args.embedding_source,
               kl_s, H_YT, kl_s2, align_err_w,
               LAS, 1 - align_err_w))

        sys.stdout.flush()

        # ======================== eval on test set ===============================
        if args.test == 'yes':
            print('test data total batches:', len(corpus_test))

            result_dict = model.parse_dev_batch(
                    corpus_test,
                    elmo_embeds_test,
                    args.out_path + '_test')

            align_err_w = result_dict["align_err_w"]
            nlogp_w = result_dict["nlogp_w"]
            align_err_s = result_dict["align_err_s"]
            nlogp_s = result_dict["nlogp_s"]
            kl_s = result_dict["kl_w"]
            LAS = result_dict["LAS"]
            kl_s2 = result_dict["kl_w2"]

            print("alignment error rate is %.5f, negative log-likelihood per word is %.5f" % (align_err_w, nlogp_w))
            print("alignment error per sentence is %.5f, NLL per sentence is %.5f" % (align_err_s, nlogp_s))
            print("KL1 is %.3f, KL2 is %.3f" % (kl_s, kl_s2))

            H_YT = nlogp_s
            print(
                "summary_test_final: tag_dim=%d, test, elmo=%s,"
                " MI(X,T)=%.5f, H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
                "LAS=%.3f, UAS=%.3f"
                % (args.tag_dim, args.embedding_source,
                   kl_s, H_YT, kl_s2, align_err_w,
                   LAS, 1 - align_err_w))
            sys.stdout.flush()

        return 0
    elif args.mode == 'evaluate':
        # ======================== load trained model ===============================
        model = model.load_model(args.checkpoint_path)
        # ======================== eval on dev set ===============================
        print('dev data total batches:', len(corpus_dev))
        result_dict = model.parse_dev_batch(
            corpus_dev,
            elmo_embeds_dev,
            args.out_path + '_dev')

        align_err_w = result_dict["align_err_w"]
        nlogp_w = result_dict["nlogp_w"]
        align_err_s = result_dict["align_err_s"]
        nlogp_s = result_dict["nlogp_s"]
        kl_s = result_dict["kl_w"]
        LAS = result_dict["LAS"]
        kl_s2 = result_dict["kl_w2"]

        # print("alignment error rate is %.5f, negative log-likelihood per word is %.5f" % (align_err_w, nlogp_w))
        # print("alignment error per sentence is %.5f, NLL per sentence is %.5f" % (align_err_s, nlogp_s))
        # print("KL1 is %.3f, KL2 is %.3f" % (kl_s, kl_s2))

        H_YT = nlogp_s
        print(
            "summary_dev_final: tag_dim=%d, dev, elmo=%s,"
            " MI(X,T)=%.5f,H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
            "LAS=%.3f, UAS=%.3f"
            % (args.tag_dim, args.embedding_source,
               kl_s, H_YT, kl_s2, align_err_w,
               LAS, 1 - align_err_w))

        sys.stdout.flush()

        # ======================== eval on test set ===============================
        if args.test == 'yes':
            print('test data total batches:', len(corpus_test))

            result_dict = model.parse_dev_batch(
                    corpus_test,
                    elmo_embeds_test,
                    args.out_path + '_test')

            align_err_w = result_dict["align_err_w"]
            nlogp_w = result_dict["nlogp_w"]
            align_err_s = result_dict["align_err_s"]
            nlogp_s = result_dict["nlogp_s"]
            kl_s = result_dict["kl_w"]
            LAS = result_dict["LAS"]
            kl_s2 = result_dict["kl_w2"]

            H_YT = nlogp_s
            print(
                "summary_test_final: tag_dim=%d, test, elmo=%s,"
                " MI(X,T)=%.5f, H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
                "LAS=%.3f, UAS=%.3f"
                % (args.tag_dim, args.embedding_source,
                   kl_s, H_YT, kl_s2, align_err_w,
                   LAS, 1 - align_err_w))
            sys.stdout.flush()

        return 0

#################################################################################

def pos_parse(args):
    _start_()

    # ==================== LOAD DATA ==========================
    data_loader = _load_data()

    corpus_dev, batch_idx_dev, en_batch_dev, processed_tree_dev = data_loader.load_dev(args.dataset_dev)

    if args.test == 'yes':
        corpus_test, batch_idx_test, en_batch_test, processed_tree_test = data_loader.load_dev(args.dataset_test)
    else:
        en_batch_test = None
    

    model = baselines.POS_Parser(data_loader.tag_dict, args)


    if args.mode == 'train':

        training_total_data = 0
        for elem in data_loader.en_batch:
            training_total_data += len(elem)
        print('the total amount of training sentence is {}'.format(training_total_data))
        print('total training batch number:', len(data_loader.corpus))
        print('tag_dim=%d' %(args.tag_dim))
        print('start_training:')

        delta_temp = 1 / len(data_loader.corpus) 
        for e in range(args.epoch):
            logger.info('epoch %d/%d:' % (e, args.epoch))
            result_dict = model.train_batch(data_loader.corpus)

            # =========== gather stats =============
            align_err_w = result_dict["align_err_w"]
            nlogp_w = result_dict["nlogp_w"]
            align_err_s = result_dict["align_err_s"]
            nlogp_s = result_dict["nlogp_s"]
            kl_s = result_dict["kl_w"]
            LAS = result_dict["LAS"]
            kl_s2 = result_dict["kl_w2"]
            H_YT = nlogp_s

            # =========== save model =============
            if True:
                model.save_model(args.save_path + '_epoch_%d' %e)
                print(
                "train_epoch%i: tag_dim=%d, dev, elmo=%s,"
                " MI(X,T)=%.5f, H(Y|T)=%.5f,  MI(X,Ti|Xi)=%.5f, err=%.5f, "
                "LAS=%.3f, UAS=%.3f, KL1=%.3f, KL2=%.3f"
                % (e, args.tag_dim, args.embedding_source,
                   kl_s, H_YT, kl_s2, align_err_w,
                   LAS, 1 - align_err_w, kl_s, kl_s2))
                sys.stdout.flush()

            # ========== check dev set ===========
            result_dict = model.parse_dev_batch(corpus_dev, args.out_path + '_middev')

            align_err_w = result_dict["align_err_w"]
            nlogp_w = result_dict["nlogp_w"]
            align_err_s = result_dict["align_err_s"]
            nlogp_s = result_dict["nlogp_s"]
            kl_s = result_dict["kl_w"]
            LAS = result_dict["LAS"]
            kl_s2 = result_dict["kl_w2"]

            H_YT = nlogp_s
            print(
                "dev_epoch%i: tag_dim=%d, dev, elmo=%s,"
                " MI(X,T)=%.5f,H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
                "LAS=%.3f, UAS=%.3f, KL1=%.3f, KL2=%.3f"
                % (e, args.tag_dim, args.embedding_source,
                   kl_s, H_YT, kl_s2, align_err_w,
                   LAS, 1 - align_err_w, kl_s, kl_s2))
            sys.stdout.flush()

            if e == 20:
                print('mid-point evaluation:')
                print('train data total batches:', len(data_loader.corpus))
                result_dict = model.parse_dev_batch(data_loader.corpus, args.out_path + '_midtrain')

                align_err_w = result_dict["align_err_w"]
                nlogp_w = result_dict["nlogp_w"]
                align_err_s = result_dict["align_err_s"]
                nlogp_s = result_dict["nlogp_s"]
                kl_s = result_dict["kl_w"]
                LAS = result_dict["LAS"]
                kl_s2 = result_dict["kl_w2"]

                H_YT = nlogp_s
                print(
                    "summary_train_mid: tag_dim=%d, train, elmo=%s,"
                    " MI(X,T)=%.5f, H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
                    "LAS=%.3f, UAS=%.3f"
                    % (args.tag_dim, args.embedding_source,
                       kl_s, H_YT, kl_s2, align_err_w,
                       LAS, 1 - align_err_w))
                sys.stdout.flush()


        print("finished training [DONE]")
        model.save_model(args.save_path)

        # =================== evaluate on full train ==========================

        print('train data total batches:', len(data_loader.corpus))
        result_dict = model.parse_dev_batch(data_loader.corpus, args.out_path + '_train')

        align_err_w = result_dict["align_err_w"]
        nlogp_w = result_dict["nlogp_w"]
        align_err_s = result_dict["align_err_s"]
        nlogp_s = result_dict["nlogp_s"]
        kl_s = result_dict["kl_w"]
        LAS = result_dict["LAS"]
        kl_s2 = result_dict["kl_w2"]

        # print("alignment error rate is %.5f, negative log-likelihood per word is %.5f" % (align_err_w, nlogp_w))
        # print("alignment error per sentence is %.5f, NLL per sentence is %.5f" % (align_err_s, nlogp_s))
        # print("KL1 is %.3f, KL2 is %.3f" % (kl_s, kl_s2))

        H_YT = nlogp_s
        print(
            "summary_train_final: tag_dim=%d, train, elmo=%s,"
            " MI(X,T)=%.5f, H(Y|T)=%.5f,MI(X,Ti|Xi)=%.5f, err=%.5f, "
            "LAS=%.3f, UAS=%.3f"
            % (args.tag_dim, args.embedding_source,
               kl_s, H_YT, kl_s2, align_err_w,
               LAS, 1 - align_err_w))
        sys.stdout.flush()

        # ======================== eval on dev set ===============================
        print('dev data total batches:', len(corpus_dev))
        result_dict = model.parse_dev_batch(corpus_dev, args.out_path + '_dev')

        align_err_w = result_dict["align_err_w"]
        nlogp_w = result_dict["nlogp_w"]
        align_err_s = result_dict["align_err_s"]
        nlogp_s = result_dict["nlogp_s"]
        kl_s = result_dict["kl_w"]
        LAS = result_dict["LAS"]
        kl_s2 = result_dict["kl_w2"]

        H_YT = nlogp_s
        print(
            "summary_dev_final: tag_dim=%d, dev, elmo=%s,"
            " MI(X,T)=%.5f,H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
            "LAS=%.3f, UAS=%.3f"
            % (args.tag_dim, args.embedding_source,
               kl_s, H_YT, kl_s2, align_err_w,
               LAS, 1 - align_err_w))

        sys.stdout.flush()

        # ======================== eval on test set ===============================
        if args.test == 'yes':
            print('test data total batches:', len(corpus_test))

            result_dict = model.parse_dev_batch(corpus_test, args.out_path + '_test')

            align_err_w = result_dict["align_err_w"]
            nlogp_w = result_dict["nlogp_w"]
            align_err_s = result_dict["align_err_s"]
            nlogp_s = result_dict["nlogp_s"]
            kl_s = result_dict["kl_w"]
            LAS = result_dict["LAS"]
            kl_s2 = result_dict["kl_w2"]
            H_YT = nlogp_s
            print(
                "summary_test_final: tag_dim=%d, test, elmo=%s,"
                " MI(X,T)=%.5f, H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
                "LAS=%.3f, UAS=%.3f"
                % (args.tag_dim, args.embedding_source,
                   kl_s, H_YT, kl_s2, align_err_w,
                   LAS, 1 - align_err_w))
            sys.stdout.flush()

        return 0

    elif args.mode == 'evaluate':
        model.load_model(args.checkpoint_path)
        # ======================== eval on dev set ===============================
        print('dev data total batches:', len(corpus_dev))
        result_dict = model.parse_dev_batch(corpus_dev, args.out_path + '_dev')

        align_err_w = result_dict["align_err_w"]
        nlogp_w = result_dict["nlogp_w"]
        align_err_s = result_dict["align_err_s"]
        nlogp_s = result_dict["nlogp_s"]
        kl_s = result_dict["kl_w"]
        LAS = result_dict["LAS"]
        kl_s2 = result_dict["kl_w2"]

        H_YT = nlogp_s
        print(
            "summary_dev_final: tag_dim=%d, dev, elmo=%s,"
            " MI(X,T)=%.5f,H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
            "LAS=%.3f, UAS=%.3f"
            % (args.tag_dim, args.embedding_source,
               kl_s, H_YT, kl_s2, align_err_w,
               LAS, 1 - align_err_w))

        sys.stdout.flush()

        # ======================== eval on test set ===============================
        if args.test == 'yes':
            print('test data total batches:', len(corpus_test))

            result_dict = model.parse_dev_batch(corpus_test, args.out_path + '_test')

            align_err_w = result_dict["align_err_w"]
            nlogp_w = result_dict["nlogp_w"]
            align_err_s = result_dict["align_err_s"]
            nlogp_s = result_dict["nlogp_s"]
            kl_s = result_dict["kl_w"]
            LAS = result_dict["LAS"]
            kl_s2 = result_dict["kl_w2"]
            H_YT = nlogp_s
            print(
                "summary_test_final: tag_dim=%d, test, elmo=%s,"
                " MI(X,T)=%.5f, H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
                "LAS=%.3f, UAS=%.3f"
                % (args.tag_dim, args.embedding_source,
                   kl_s, H_YT, kl_s2, align_err_w,
                   LAS, 1 - align_err_w))
            sys.stdout.flush()

        return 0






#################################################################################

def main(args):
    _start_()

    # ==================== LOAD DATA ==========================
    data_loader = _load_data()

    corpus_dev, batch_idx_dev, en_batch_dev, processed_tree_dev = data_loader.load_dev(args.dataset_dev)

    # # DELETE
    # data_loader.corpus = data_loader.corpus[1:30]
    # data_loader.en_batch = data_loader.en_batch[1:30]
    # data_loader.batch_idx = data_loader.batch_idx[1:30]
    # corpus_dev, batch_idx_dev, en_batch_dev = corpus_dev[1:20], batch_idx_dev[1:20], en_batch_dev[1:20]

    if args.test == 'yes':
        corpus_test, batch_idx_test, en_batch_test, processed_tree_test = data_loader.load_dev(args.dataset_test)
    else:
        en_batch_test = None

    # ==================== LOAD ELMo/BERT ==========================
    pre_compute_dict, embed_loader = _load_pretrain_emb(data_loader, en_batch_dev=en_batch_dev, en_batch_test=en_batch_test)
    elmo_embeds_train = pre_compute_dict['elmo_embeds_train']
    non_context_embeds_train = pre_compute_dict['non_context_embeds_train']
    elmo_embeds_dev, non_context_embeds_dev = pre_compute_dict['elmo_embeds_dev'], pre_compute_dict['non_context_embeds_dev']
    if args.test == 'yes':
        elmo_embeds_test, non_context_embeds_test = pre_compute_dict['elmo_embeds_test'], pre_compute_dict['non_context_embeds_test']

    # ==================== INITIALIZING MODEL ==========================
    if args.task == 'VIB_continuous':
        model = Continuous_VIB(args, data_loader.word_dict)

    elif args.task == 'VIB_discrete':
        model = Discrete_VIB(args, data_loader.word_dict)
    else:
        print('invalid model type -- should be either VIB_continuous or VIB_discrete')
    model.to(args.device)


    if args.mode == 'train':
        training_total_data = 0
        for elem in data_loader.en_batch:
            training_total_data += len(elem)
        print('the total amount of training sentence is {}'.format(training_total_data))
        print('total training batch number:', len(data_loader.corpus))
        print('tag_dim=%d, beta=%f' %(args.tag_dim, args.beta))
        print('start_training:')

        delta_temp = 1 / len(data_loader.corpus) 
        for e in range(args.epoch):
            logger.info('epoch %d/%d:' % (e, args.epoch))
            result_dict = model.train_batch(
                data_loader.corpus,
                args.sent_per_epoch,
                elmo_embeds_train,
                non_context_embeds_train,
                delta_temp=delta_temp
                )

            # =========== gather stats =============

            align_err_w = result_dict["align_err_w"]
            nlogp_w = result_dict["nlogp_w"]
            align_err_s = result_dict["align_err_s"]
            nlogp_s = result_dict["nlogp_s"]
            kl_s = result_dict["kl_w"]
            LAS = result_dict["LAS"]
            kl_s2 = result_dict["kl_w2"]
            H_YT = nlogp_s

            # =========== save model =============
            if True:
                model.save_model(args.save_path + '_epoch_%d' %e)
                print(
                "train_epoch%i: beta=%f, gamma=%f, tag_dim=%d, dev, elmo=%s,"
                " MI(X,T)=%.5f, H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
                "LAS=%.3f, UAS=%.3f, KL1=%.3f, KL2=%.3f"
                % (e, model.beta, model.gamma, args.tag_dim, args.embedding_source,
                   kl_s, H_YT, kl_s2, align_err_w,
                   LAS, 1 - align_err_w, kl_s, kl_s2))

                # print("UAS=%.5f, NLL=%.5f, LAS=%.5f, KL=%.5f" %(1-align_err_w, nlogp_w, LAS, kl_s))
                sys.stdout.flush()

            # ========== check dev set ===========
            result_dict = model.parse_dev_batch(
                corpus_dev,
                elmo_embeds_dev,
                non_context_embeds_dev, args.out_path + '_middev')

            align_err_w = result_dict["align_err_w"]
            nlogp_w = result_dict["nlogp_w"]
            align_err_s = result_dict["align_err_s"]
            nlogp_s = result_dict["nlogp_s"]
            kl_s = result_dict["kl_w"]
            LAS = result_dict["LAS"]
            kl_s2 = result_dict["kl_w2"]
            H_YT = nlogp_s
            print(
                "dev_epoch%i: beta=%f, gamma=%f, tag_dim=%d, dev, elmo=%s,"
                " MI(X,T)=%.5f, H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
                "LAS=%.3f, UAS=%.3f, KL1=%.3f, KL2=%.3f"
                % (e, model.beta, model.gamma, args.tag_dim, args.embedding_source,
                   kl_s, H_YT, kl_s2, align_err_w,
                   LAS, 1 - align_err_w, kl_s, kl_s2))
            sys.stdout.flush()

            if e == 20:
                print('mid-point evaluation:')
                print('train data total batches:', len(data_loader.corpus))
                result_dict = model.parse_dev_batch(
                    data_loader.corpus,
                    elmo_embeds_train,
                    non_context_embeds_train, args.out_path + '_midtrain')

                align_err_w = result_dict["align_err_w"]
                nlogp_w = result_dict["nlogp_w"]
                align_err_s = result_dict["align_err_s"]
                nlogp_s = result_dict["nlogp_s"]
                kl_s = result_dict["kl_w"]
                LAS = result_dict["LAS"]
                kl_s2 = result_dict["kl_w2"]
                H_YT = nlogp_s
                print(
                    "summary_train_mid: beta=%f, gamma=%f, tag_dim=%d, train, elmo=%s,"
                    " MI(X,T)=%.5f, H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
                    "LAS=%.3f, UAS=%.3f"
                    % (model.beta, model.gamma, args.tag_dim, args.embedding_source,
                       kl_s, H_YT, kl_s2, align_err_w,
                       LAS, 1 - align_err_w))
                sys.stdout.flush()


        print("finished training [DONE]")
        model.save_model(args.save_path)

        # =================== evaluate on full train ==========================

        print('train data total batches:', len(data_loader.corpus))
        result_dict = model.parse_dev_batch(
            data_loader.corpus,
            elmo_embeds_train, non_context_embeds_train, args.out_path + '_train')

        align_err_w = result_dict["align_err_w"]
        nlogp_w = result_dict["nlogp_w"]
        align_err_s = result_dict["align_err_s"]
        nlogp_s = result_dict["nlogp_s"]
        kl_s = result_dict["kl_w"]
        LAS = result_dict["LAS"]
        kl_s2 = result_dict["kl_w2"]
        H_YT = nlogp_s
        print(
            "summary_train_final: beta=%f, gamma=%f, tag_dim=%d, train, elmo=%s,"
            " MI(X,T)=%.5f, H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
            "LAS=%.3f, UAS=%.3f"
            % (model.beta, model.gamma, args.tag_dim, args.embedding_source,
               kl_s, H_YT, kl_s2, align_err_w,
               LAS, 1 - align_err_w))
        sys.stdout.flush()

        # ======================== eval on dev set ===============================
        print('dev data total batches:', len(corpus_dev))
        result_dict = model.parse_dev_batch(
            corpus_dev,
            elmo_embeds_dev,
            non_context_embeds_dev, args.out_path + '_dev')

        align_err_w = result_dict["align_err_w"]
        nlogp_w = result_dict["nlogp_w"]
        align_err_s = result_dict["align_err_s"]
        nlogp_s = result_dict["nlogp_s"]
        kl_s = result_dict["kl_w"]
        LAS = result_dict["LAS"]
        kl_s2 = result_dict["kl_w2"]
        H_YT = nlogp_s
        print(
            "summary_dev_final: beta=%f, gamma=%f, tag_dim=%d, dev, elmo=%s,"
            " MI(X,T)=%.5f, H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
            "LAS=%.3f, UAS=%.3f"
            % (model.beta, model.gamma, args.tag_dim, args.embedding_source,
               kl_s, H_YT, kl_s2, align_err_w,
               LAS, 1 - align_err_w))

        sys.stdout.flush()

        # ======================== eval on test set ===============================
        if args.test == 'yes':
            print('test data total batches:', len(corpus_test))

            result_dict = model.parse_dev_batch(
                    corpus_test,
                    elmo_embeds_test,
                    non_context_embeds_test, args.out_path + '_test')

            align_err_w = result_dict["align_err_w"]
            nlogp_w = result_dict["nlogp_w"]
            align_err_s = result_dict["align_err_s"]
            nlogp_s = result_dict["nlogp_s"]
            kl_s = result_dict["kl_w"]
            LAS = result_dict["LAS"]
            kl_s2 = result_dict["kl_w2"]
            H_YT = nlogp_s
            print(
                "summary_test_final: beta=%f, gamma=%f, tag_dim=%d, test, elmo=%s,"
                " MI(X,T)=%.5f, H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
                "LAS=%.3f, UAS=%.3f"
                % (model.beta, model.gamma, args.tag_dim, args.embedding_source,
                   kl_s, H_YT, kl_s2, align_err_w,
                   LAS, 1 - align_err_w))
            sys.stdout.flush()

        return 0

    elif args.mode == 'evaluate':
        # ======================== load trained model ===============================
        model = model.load_model(args.checkpoint_path)

        # ======================== eval on dev set ===============================
        print('dev data total batches:', len(corpus_dev))
        result_dict = model.parse_dev_batch(
            corpus_dev,
            elmo_embeds_dev,
            non_context_embeds_dev, args.out_path + '_dev')

        align_err_w = result_dict["align_err_w"]
        nlogp_w = result_dict["nlogp_w"]
        align_err_s = result_dict["align_err_s"]
        nlogp_s = result_dict["nlogp_s"]
        kl_s = result_dict["kl_w"]
        LAS = result_dict["LAS"]
        kl_s2 = result_dict["kl_w2"]
        H_YT = nlogp_s
        print(
            "summary_dev_final: beta=%f, gamma=%f, tag_dim=%d, dev, elmo=%s,"
            " MI(X,T)=%.5f, H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
            "LAS=%.3f, UAS=%.3f"
            % (model.beta, model.gamma, args.tag_dim, args.embedding_source,
               kl_s, H_YT, kl_s2, align_err_w,
               LAS, 1 - align_err_w))

        sys.stdout.flush()

        # ======================== eval on test set ===============================
        if args.test == 'yes':
            print('test data total batches:', len(corpus_test))

            result_dict = model.parse_dev_batch(
                    corpus_test,
                    elmo_embeds_test,
                    non_context_embeds_test, args.out_path + '_test')

            align_err_w = result_dict["align_err_w"]
            nlogp_w = result_dict["nlogp_w"]
            align_err_s = result_dict["align_err_s"]
            nlogp_s = result_dict["nlogp_s"]
            kl_s = result_dict["kl_w"]
            LAS = result_dict["LAS"]
            kl_s2 = result_dict["kl_w2"]
            H_YT = nlogp_s
            print(
                "summary_test_final: beta=%f, gamma=%f, tag_dim=%d, test, elmo=%s,"
                " MI(X,T)=%.5f, H(Y|T)=%.5f, MI(X,Ti|Xi)=%.5f, err=%.5f, "
                "LAS=%.3f, UAS=%.3f"
                % (model.beta, model.gamma, args.tag_dim, args.embedding_source,
                   kl_s, H_YT, kl_s2, align_err_w,
                   LAS, 1 - align_err_w))
            sys.stdout.flush()

    else: 
        print('invalid mode')
        return 0







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Syntactic-VIB')
    parser.add_argument('--epoch', default = 30, type=int, help='the number of epoch to train')
    parser.add_argument('--lr', default = 1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--beta', default = 1e-3, type=float, help='beta value (tradeoff parameter)')
    parser.add_argument("--type_token_reg", type=str, dest="type_token_reg", default='yes', help='whether to enable to type encoder \
        and thereby control the contextsensitivity')
    parser.add_argument('--gamma', default=1e-3, type=float, help='gamma value (tradeoff parameter)')

    parser.add_argument('--seed', default = 1, type=int, help='random seed')
    parser.add_argument('--batch_size', default = 36, type=int, help='batch size')
    parser.add_argument('--test', default='no', type=str, help='whether to load the test set')
    parser.add_argument('--dataset_base', default='/Users/xiangli/documents/UD_English/',
                        type=str, help='path to the dataset')
    parser.add_argument('--save_path', default='model_checkpoint', type=str, help='path to save the model')
    parser.add_argument('--out_path', default='model_checkpoint', type=str, help='path to save the output')
    parser.add_argument('--checkpoint_path', default='model_checkpoint', type=str, help='path to load the trained model')
    parser.add_argument('--mode',default='train', type=str, help='train or evaluate')
    parser.add_argument("--task", type=str, dest="task", default="VIB_continuous", help='options are VIB_discrete, VIB_continuous, \
        IDEN, PCA, CLEAN(MLP in the paper baseline), POS, FINETUNE.')
    parser.add_argument("--sample_method", type=str, dest="sample_method", default="iid",
        help="how to draw samples from the ptheta distribution")
    parser.add_argument("--embedding_source", type=str, dest="embedding_source", default='elmo_1', help='other options are elmo_0, \
        elmo_2')
    parser.add_argument("--tag_dim", type=int, dest="tag_dim", default=128, help='d (dimensionalities) for the continuous tags, k \
        (cardinalities) for the discrete tagset')
    parser.add_argument("--sample_size", type=int, dest="sample_size", default=5, help='the number of samples to take when \
        estimating the stochastic gradient')

    parser.add_argument("--lang", type=str, dest="lang", default="en", help='the language choice')
    parser.add_argument("--cuda", type=int, dest="cuda", default=1, help='whether to use CUDA or not')
    parser.add_argument("--elmo_model_path", type=str, dest="elmo_model_path", default='/Users/xiangli/Downloads/136', help='path \
        to the saved pre-trained model')
    parser.add_argument("--elmo_option_path", type=str, dest="elmo_option_path", default='/Users/xiangli/Downloads/136', help='path \
        to the options for the saved pretrained model.')
    parser.add_argument("--activation", type=str, dest="activation", default="tanh", help='the choice of \
        non-linearity transfer function')
    parser.add_argument("--projective", type=str, dest="projective", default="no", help='whether to filter by projective trees')

# ========================================================================================================
    parser.add_argument("--hidden_units", type=int, dest="hidden_units", default=100)
    parser.add_argument("--hidden2", type=int, dest="hidden2_units", default=0)
    parser.add_argument("--lstm_dims", type=int, dest="lstm_dims", default=125)
    parser.add_argument("--max_sent_len", type=int, dest="max_sent_len", default=30, help='max sentence length')
    parser.add_argument("--word_threshold", type=int, dest="word_threshold", default=1, help='word threshold')
    parser.add_argument("--tag_representation_dim", type=int, dest="tag_representation_dim", default=100)
    parser.add_argument("--arc_representation_dim", type=int, dest="arc_representation_dim", default=50)
    parser.add_argument("--num_labels", type=int, dest="num_labels", default=0)
    parser.add_argument("--sent_per_epoch", type=int, dest="sent_per_epoch", default=10000)
    parser.add_argument("--data_type", type=str, dest="data_type", default='UD')
    parser.add_argument("--ablation", type=bool, dest="ablation", default=False)
    parser.add_argument("--submode", type=str, dest="submode", default='')
    parser.add_argument("--inp", type=str, dest="inp", default='sample')
    parser.add_argument("--temperature_anneal", type=str, dest="temperature_anneal", default="yes")
    parser.add_argument("--jitter", type=str, dest="jitter", default="yes")

    args = parser.parse_args()

    print(args.ablation)
    print(torch.cuda.is_available())

    # ================================= CUDA =====================================
    args.device = None
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda 1")
            args.device = torch.device('cpu')
        else:
            print('using cuda successfully. ')
            torch.cuda.manual_seed(args.seed)
            args.device = torch.device('cuda')
    else:
        print('there is no cuda available')
        args.device = torch.device('cpu')

    # ================================= TASK =====================================
    if args.task == 'VIB_continuous':
        if args.gamma == -1:
            args.gamma = args.beta
        print('beta is {}, gamma is {}'.format(args.beta, args.gamma))
        main(args)

    elif args.task == 'VIB_discrete':
        if args.gamma == -1:
            args.gamma = args.beta
        main(args)

    # ================================= BASELINES =====================================

    elif args.task == 'FINETUNE':
        fine_parse(args)

    elif args.task == 'POS':
        pos_parse(args)

    elif args.task == 'IDEN' or args.task == 'CLEAN' or args.task == 'PCA':
        iden_parse(args)

    else:
        print('task is not available !!! double check spelling')
