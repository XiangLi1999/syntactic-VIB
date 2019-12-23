#!/usr/bin/python
# --------------------------------------- 
# File Name : experiment.py
# Creation Date : 21-11-2017
# Last Modified : Tue Nov 21 11:04:35 2017
# Created By : wdd 
# ---------------------------------------
from shepherd import shepherd, init, post, USR, ALL, SYS, SPD, grid_search, basic_func, load_job_info, _args
import shutil

import os, re

try:
    import cPickle as pickle
except:
    import pickle

udv14_train = ['cs', 'ru_syntagrus', 'es_ancora', 'ca', 'es', 'fr', 'hi', 'la_ittb', 'it', 'de', 'zh', 'ar']
udv14_dev = ['en_cesl', 'en_esl', 'en_lines', 'en']
udv14 = ['ar', 'zh', 'en', 'hi', 'es_ancora']


def _itr_file_list(input, pattern):
    print('Search Patterm:', pattern)
    ptn = re.compile(pattern)
    for root, dir, files in os.walk(input):
        for fn in files:
            abs_fn = os.path.normpath(os.path.join(root, fn))
            m = ptn.match(abs_fn)
            if m:
                lang = m.groups()
                yield lang, abs_fn


def _get_data(config):
    ret = []
    for l in config.split('+'):
        if l == 'train':
            ret += udv14_train
        elif l == 'dev':
            ret += udv14_dev
    ret = config.split('|') if ret == [] else ret
    return ret


@shepherd(before=[init], after=[post])
def build_perm_feature():
    USR.set('tb', 'ud-treebanks-v1.4')
    pattern = '.*/UD_.*/({lang})-ud-train.conllu$' % USR
    header_fmt = 'python %(S_python_dir)s/permutation/main.py' \
                 ' --task build_feature' \
                 ' --src {src_fn}' \
                 ' --output %(S_output)s/{config}.pkl' % ALL()
    lgs = udv14
    for (lg,), lg_fn in _itr_file_list('%(S_data)s/%(U_tb)s/' % ALL(), pattern.format(lang='|'.join(lgs))):
        config = lg
        command = [header_fmt.format(src_fn=lg_fn, config=config)]
        SPD().submit(command, config)


@shepherd(before=[init], after=[post])
def train_permute():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('feature', 'TRA')
    USR.set('batch_size', '100')
    USR.set('mx_dep_train', '6')
    pattern = '.*/UD_.*/({lang})-ud-train.conllu$' % USR
    command = 'python %(S_python_dir)s/permutation/main.py' \
              ' --task train' % ALL() + \
              f' --feature {USR.feature}' \
              f' --batch_size {USR.batch_size}' \
              f' --mx_dep_train {USR.mx_dep_train}'
    USR.set('feature_vocab', 'lat-build_perm_feature-ud14')
    lgs = udv14
    for (src,), src_fn in _itr_file_list('%(S_data)s/%(U_tb)s/' % ALL(), pattern.format(lang='|'.join(lgs))):
        config, vocab = src, src
        src_out = os.path.join(SYS.tmp, config + '.conllu')
        src_spec = f' --feature_vocab {SYS.jobs}/{USR.feature_vocab}/output/{vocab}.pkl' \
                   f' --model {SYS.model}/{config}' \
                   f' --src {src_fn}' \
                   f' --src_out {src_out}'
        SPD().submit([command + src_spec], config)



@shepherd(before=[init], after=[post])
def test_permute():
    USR.set('tb', 'UD_English_punct' % SYS)
    command_tmpl = '%(S_python_itrptr)s %(S_python_dir)s/permutation/main.py' \
                   ' --task test' \
                   ' --test {test}' \
                   ' --test_out %(S_output)s/{config}' \
                   ' --model {model}' \
                   % ALL()
    USR.set('model', '')
    USR.set('param', '({lang})\.train_status.json'.format(lang='|'.join(udv14)))
    pattern = '.*/{lang}-ud-.*.conllu$'
    for (sup,), md_path in _itr_file('%(S_jobs)s/%(U_model)s/model' % ALL(), f'.*/{USR.param}'):
        for _, sub_fn in _itr_file_list('%(S_data)s/%(U_tb)s' % ALL(), pattern.format(lang=sup)):
            config = os.path.basename(sub_fn)
            command = command_tmpl.format(config=config, test=sub_fn, model=md_path)
            SPD().submit(command=[command], config=config)


@shepherd(before=[init], after=[post])
def depunct():
    USR.set('tb', 'ud-treebanks-v1.4')
    command_tmpl = '%(S_python_itrptr)s %(S_python_dir)s/permutation/main.py' \
                   ' --task depunct' \
                   ' --test {test}' \
                   ' --test_out {test_out}' \
                   % ALL()
    pattern = '.*/.*?-ud-.*?.conllu$' % USR
    for _, fn in _itr_file_list('%(S_data)s/%(U_tb)s/' % ALL(), pattern):
        config = os.path.basename(fn)
        command = command_tmpl.format(test=fn, test_out=os.path.join(SYS.output, config))
        SPD().submit(command=[command], config=config)


@shepherd(before=[init], after=[post])
def mi_train1():
    command = ''
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    command = '%(S_python_itrptr)s %(S_python_dir)s/supervised_parser.py' \
                   ' --mode train' \
                   ' --dataset %(U_dataset)s'\
                   % ALL()
    config = 'example'
    print(command)
    SPD().submit(command=[command], config=config)
    # pattern = '.*/.*?-ud-.*?.conllu$' % USR
    # for _, fn in _itr_file_list('%(S_data)s/%(U_tb)s/' % ALL(), pattern):
    #     config = os.path.basename(fn)
    #     command = command_tmpl.format(test=fn, test_out=os.path.join(SYS.output, config))
    #     SPD().submit(command=[command], config=config)

@shepherd(before=[init], after=[post])
def con_train():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/scratch/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/scratch/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/scratch/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')
    USR.set('ablated', 'False')

    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task VIB' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg %(U_type_token_reg)s' \
                   ' --cuda 1' \
              % ALL()

#                    ' --ablation %(U_ablated)s' \

    # search_list = [
    #     ('batch_size', '5'),
    #     ('beta', '0|0.000001|0.00001|0.0001|0.0005|0.001|0.005|0.01|0.05|0.1|1|10'),
    #     ('seed', '1'),
    #     ('sample_size', '8'),
    #     ('sample_method', 'iid'),
    #     ('r_cov_mode', 'diagonal'),
    #     ('rr_mode', 'diagonal'),
    #     ('max_sent_len', '30'),
    #     ('optim', 'alternate'),
    #     ('encotype', 'linear3'),
    #     ('sent_per_epoch', '5000'), # 200|500|1000
    #     ('tag_dim', '512'),
    #     ('embedding_source', 'elmo|elmo_2|elmo_3')
    # ]

    # search_list = [
    #     ('batch_size', '36'),
    #     ('beta', '0.00001|0.0001|0.001'),
    #     ('gamma', '0|0.00001|0.0001|0.001|0.01|0.1|1'), # -1|-5|-0.01|-0.001|
    #     ('seed', '1'),
    #     ('sample_size', '5'),
    #     ('weight_decay', '0.0001|0.001'),
    #     ('max_sent_len', '30'),
    #     ('sent_per_epoch', '5000'),
    #     ('tag_dim', '256'),
    #     ('embedding_source', 'elmo|elmo_2|elmo_3')
    # ]

    search_list = [
        ('batch_size', '36'),
        ('beta', '0.00001|0.0001|0.001'),
        ('gamma', '-1|-5|-0.01|-0.001'),
        ('seed', '1'),
        ('sample_size', '5'),
        ('weight_decay', '0.0001|0.001'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'),
        ('tag_dim', '256'),
        ('embedding_source', 'elmo|elmo_2|elmo_3')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

# %(S_data)s
@shepherd(before=[init], after=[post])
def dis_train_local():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', 'UD_English/en-ud-train.conllu')
    USR.set('dataset_d', 'UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', 'UD_English/en-ud-test.conllu')
    USR.set('elmo_test', 'ELMo_token/elmo_test')
    USR.set('elmo_dev', 'ELMo_token/elmo_dev')
    USR.set('elmo_train', 'ELMo_token/elmo_train')
    USR.set('elmo_first', 'ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')
    USR.set('ablated', 'False')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(S_data)s/%(U_dataset_t)s' \
                   ' --dataset_dev %(S_data)s/%(U_dataset_d)s' \
                   ' --dataset_test %(S_data)s/%(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task VIB_discrete' \
                   ' --elmo_test_data_path %(S_data)s/%(U_elmo_test)s' \
                   ' --elmo_train_data_path %(S_data)s/%(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(S_data)s/%(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg %(U_type_token_reg)s' \
              % ALL()

    # ' --ablation %(U_ablated)s' \

    search_list = [
        ('batch_size', '5'),
        ('beta', '0.0|0.01|0.001|0.0001|0.00001|0.1|1|10'),
        ('gamma', '-1'),
        ('seed', '1'),
        ('weight_decay', '0.001|0.0001|0.00001'),
        ('sample_size', '4'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'),
        ('tag_dim', '128|64'),
        ('embedding_source', 'elmo|elmo_2|elmo_3')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def dis_train_marcc():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')
    USR.set('ablated', 'False')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task VIB_discrete' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg %(U_type_token_reg)s' \
                   ' --cuda 1' \
              % ALL()

    # ' --ablation %(U_ablated)s' \

    search_list = [
        ('batch_size', '30'),
        ('beta', '0.0|0.01|0.001|0.0001|0.00001|0.1|1'),
        ('gamma', '-1'),
        ('seed', '1'),
        ('weight_decay', '0.001|0.0001|0.00001|0'),
        ('sample_size', '5'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'),
        ('tag_dim', '128|64'),
        ('embedding_source', 'elmo_0|elmo_1|elmo_2')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def dis_train_marcc_fore():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')
    USR.set('ablated', 'False')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task VIB_discrete' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg %(U_type_token_reg)s' \
                   ' --cuda 1' \
                   ' --test yes' \
                   ' --foreign yes' \
              % ALL()

    # ' --ablation %(U_ablated)s' \

    search_list = [
        ('batch_size', '30'),
        ('beta', '0.0|0.01|0.001|0.0001|0.00001|0.1|1'),
        ('gamma', '-1'),
        ('seed', '1'),
        ('weight_decay', '0.001|0.0001|0.00001|0'),
        ('sample_size', '5'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'),
        ('tag_dim', '128'),
        ('embedding_source', 'elmo_0|elmo_1|elmo_2'),
        ('lang', 'ar|es') #zh|hi|fr|it|pt|ru
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def dis_eval_marcc():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('prefix', '/home-4/xli150@jhu.edu/mutualinfo/jobs/dis_train_marcc-2/model')
    USR.set('type_token_reg', 'yes')
    USR.set('ablated', 'False')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --checkpoint_path %(U_prefix)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode evaluate' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task VIB_discrete' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg %(U_type_token_reg)s' \
                   ' --cuda 1' \
                   ' --test yes' \
              % ALL()

    # ' --ablation %(U_ablated)s' \

    search_list = [
        ('batch_size', '30'),
        ('beta', '0.0|0.01|0.001|0.0001|0.00001|0.1|1'),
        ('gamma', '-1'),
        ('seed', '1'),
        ('weight_decay', '0.001|0.0001|0.00001|0'),
        ('sample_size', '5'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'),
        ('tag_dim', '128|64'),
        ('embedding_source', 'elmo_1|elmo_2')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def hier_train():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')
    USR.set('ablated', 'False')
    USR.set('jittered', 'yes')
    USR.set('temperature_anneal', 'yes')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 10' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task VIB_discrete' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg no' \
                   ' --foreign 3' \
                   ' --jitter %(U_jittered)s' \
                   ' --temperature_anneal %(U_temperature_anneal)s' \
              % ALL()

    # ' --ablation %(U_ablated)s' \

    search_list = [
        ('batch_size', '5'),
        ('beta', '4'),
        ('seed', '1|22|55'),
        ('weight_decay', '0.001|0.0001|0.00001'),
        ('sample_size', '4'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'),
        ('embedding_source', 'elmo|elmo_2|elmo_3')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return


@shepherd(before=[init], after=[post])
def dis_train_pt():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_option_path','/home-4/xli150@jhu.edu/data/lisa/pt_elmo/elmo_pt_options.json')
    USR.set('elmo_model_path', '/home-4/xli150@jhu.edu/data/lisa/pt_elmo/elmo_pt_weights.hdf5')
    USR.set('type_token_reg', 'yes')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task VIB_discrete' \
                   ' --elmo_model_path %(U_elmo_model_path)s'\
                   ' --elmo_option_path %(U_elmo_option_path)s'\
                   ' --type_token_reg %(U_type_token_reg)s' \
                   ' --foreign yes'\
                   ' --lang pt' \
              % ALL()

    # ' --ablation %(U_ablated)s' \

    search_list = [
        ('batch_size', '5'),
        ('beta', '0.0|0.01|0.001|0.0001|0.00001|0.1'),
        ('gamma', '0|-1'),
        ('seed', '1'),
        ('weight_decay', '0.001|0.0001|0.00001'),
        ('sample_size', '4'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'),
        ('tag_dim', '128|64'),
        ('embedding_source', 'elmo|elmo_2|elmo_3')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def con_test_sem():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('prefix', '/home-4/xli150@jhu.edu/mutualinfo/jobs/mi_train-b/model')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')
    USR.set('weight_decay2', '0.001')
    USR.set('mode', 'sem_recon')
    USR.set('inp', 'sample')

    # USR.set('embedding', '/home-4/xli150@jhu.edu/data/lisa/Glove/glove.6B.50d.txt') # other options like 100d, 200d, 300d also available.
    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
              ' --dataset %(U_dataset_t)s' \
              ' --dataset_dev %(U_dataset_d)s' \
              ' --dataset_test %(U_dataset_test)s' \
              ' --save_path %(S_model)s/{config}' \
              ' --checkpoint_path %(U_prefix)s/{config}' \
              ' --out_path %(S_output)s/{config}' \
              ' --weight_decay2 %(U_weight_decay2)s' \
              ' --epoch 50' \
              ' --mode %(U_mode)s' \
              ' --word_threshold 1' \
              ' --projective non-projective' \
              ' --task VIB' \
              ' --elmo_test_data_path %(U_elmo_test)s' \
              ' --elmo_train_data_path %(U_elmo_train)s' \
              ' --elmo_dev_data_path %(U_elmo_dev)s' \
              ' --elmo_first %(U_elmo_first)s' \
              ' --type_token_reg %(U_type_token_reg)s' \
              ' --inp %(U_inp)s' \
              % ALL()

    search_list = [
        ('batch_size', '5'),
        ('beta', '0|0.00001|0.0001|0.01|0.1|1'),
        ('gamma', '-1'),
        ('seed', '1'),
        ('sample_size', '4'),
        ('weight_decay', '0.0|0.001|0.0001|0.00001'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'),
        ('tag_dim', '256'),
        ('embedding_source', 'elmo|elmo_2|elmo_3')
    ]


    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def dis_test_sem():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('prefix', '/home-4/xli150@jhu.edu/mutualinfo/jobs/mi_train-b/model')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')
    USR.set('weight_decay2', '0.001')

    # USR.set('embedding', '/home-4/xli150@jhu.edu/data/lisa/Glove/glove.6B.50d.txt') # other options like 100d, 200d, 300d also available.
    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
              ' --dataset %(U_dataset_t)s' \
              ' --dataset_dev %(U_dataset_d)s' \
              ' --dataset_test %(U_dataset_test)s' \
              ' --save_path %(S_model)s/{config}' \
              ' --checkpoint_path %(U_prefix)s/{config}' \
              ' --out_path %(S_output)s/{config}' \
              ' --weight_decay2 %(U_weight_decay2)s' \
              ' --epoch 50' \
              ' --mode sem_recon' \
              ' --word_threshold 1' \
              ' --projective non-projective' \
              ' --task VIB_discrete' \
              ' --elmo_test_data_path %(U_elmo_test)s' \
              ' --elmo_train_data_path %(U_elmo_train)s' \
              ' --elmo_dev_data_path %(U_elmo_dev)s' \
              ' --elmo_first %(U_elmo_first)s' \
              ' --type_token_reg %(U_type_token_reg)s' \
              % ALL()

    search_list = [
        ('batch_size', '5'),
        ('beta', '0.0|0.01|0.001|0.0001|0.00001|0.1|1|10'),
        ('gamma', '-1'),
        ('seed', '1'),
        ('weight_decay', '0.001|0.0001|0.00001'),
        ('sample_size', '4'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'),
        ('tag_dim', '128|64'),
        ('embedding_source', 'elmo|elmo_2|elmo_3')
    ]


    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def dis_test_pos():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('prefix', '/home-4/xli150@jhu.edu/mutualinfo/jobs/mi_train-b/model')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')
    USR.set('weight_decay2', '0.001')

    # USR.set('embedding', '/home-4/xli150@jhu.edu/data/lisa/Glove/glove.6B.50d.txt') # other options like 100d, 200d, 300d also available.
    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
              ' --dataset %(U_dataset_t)s' \
              ' --dataset_dev %(U_dataset_d)s' \
              ' --dataset_test %(U_dataset_test)s' \
              ' --save_path %(S_model)s/{config}' \
              ' --checkpoint_path %(U_prefix)s/{config}' \
              ' --out_path %(S_output)s/{config}' \
              ' --weight_decay2 %(U_weight_decay2)s' \
              ' --epoch 150' \
              ' --mode few_shot_pos' \
              ' --word_threshold 1' \
              ' --projective non-projective' \
              ' --task VIB_discrete' \
              ' --elmo_test_data_path %(U_elmo_test)s' \
              ' --elmo_train_data_path %(U_elmo_train)s' \
              ' --elmo_dev_data_path %(U_elmo_dev)s' \
              ' --elmo_first %(U_elmo_first)s' \
              ' --type_token_reg %(U_type_token_reg)s' \
              ' --lr 0.001'\
              % ALL()

    search_list = [
        ('batch_size', '5'),
        ('beta', '0.0|0.01|0.001|0.0001|0.00001|0.1|1|10'),
        ('gamma', '-1'),
        ('seed', '1'),
        ('weight_decay', '0.001|0.0001|0.00001'),
        ('sample_size', '4'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'),
        ('tag_dim', '128|64'),
        ('embedding_source', 'elmo|elmo_2|elmo_3')
    ]


    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def fore_tr_disc():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')
    USR.set('prefix', '/home-4/xli150@jhu.edu/mutualinfo/jobs/mi_train-8/model')
    USR.set('ablated', 'False')
    USR.set('mode', 'train')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --checkpoint_path %(U_prefix)s/{config}' \
                   ' --mode %(U_mode)s' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task VIB_discrete' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg yes' \
                   ' --foreign yes' \
              % ALL()

    if USR.ablated == 'True':
        command += ' --ablation True'



    search_list = [
        ('batch_size', '5'),
        ('beta', '0.0|0.01|0.001|0.0001|0.00001|0.1|1|10'),
        ('gamma', '-1'),
        ('seed', '1'),
        ('weight_decay', '0.001|0.0001|0.00001'),
        ('sample_size', '4'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'),
        ('tag_dim', '64'),
        ('embedding_source', 'elmo|elmo_2|elmo_3'),
        ('lang', 'ar|es|zh|hi|fr|de|it|pt|ru')
    ]


    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return




@shepherd(before=[init], after=[post])
def fore_de_disc():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')
    USR.set('prefix', '/home-4/xli150@jhu.edu/mutualinfo/jobs/mi_train-8/model')
    USR.set('ablated', 'False')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --checkpoint_path %(U_prefix)s/{config}' \
                   ' --epoch 150' \
                   ' --mode few-shot' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task VIB_discrete' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg yes' \
                   ' --foreign yes' \
                   ' --ablation %(U_ablated)s' \
                   ' --lr 0.001' \
              % ALL()

    search_list = [
        ('batch_size', '5'),
        ('beta', '0.0|0.01|0.001|0.0001|0.00001|0.1|1|10'),
        ('gamma', '-1'),
        ('seed', '1'),
        ('weight_decay', '0.001|0.0001|0.00001'),
        ('sample_size', '4'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'),
        ('tag_dim', '64'),
        ('embedding_source', 'elmo|elmo_2|elmo_3'),
        ('lang', 'ar|es|zh|hi|fr|fi|de|it|pt|ru')
    ]

    # search_list = [
    #     ('batch_size', '5'),
    #     ('beta', '0.00001'),
    #     ('gamma', '0.00001'),
    #     ('seed', '1'),
    #     ('sample_size', '4'),
    #     ('sample_method', 'iid'),
    #     ('r_cov_mode', 'diagonal'),  # sphereical|diagonal|
    #     ('rr_mode', 'diagonal'),  # diagonal|
    #     ('max_sent_len', '30'),
    #     ('optim', 'alternate'),
    #     ('encotype', 'linear3'),
    #     ('sent_per_epoch', '5000'),  # 200|500|1000
    #     ('tag_dim', '64'),
    #     ('embedding_source', 'elmo_2'),
    #     ('lang', 'ar|es|hi|fr|fi|de|ru')
  # ar|es|zh|hi|fr|fi|cs|de|it|pt|ru
    #]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return


@shepherd(before=[init], after=[post])
def fore_tr_cont():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')
    USR.set('ablated', 'False')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task VIB' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg yes' \
                   ' --foreign yes' \
                   ' --ablation %(U_ablated)s' \
              % ALL()

    search_list = [
        ('batch_size', '5'),
        ('beta', '0.01'), #0.00001|0.01|10
        ('gamma', '0.01'), #|0.01|10
        ('seed', '1'),
        ('sample_size', '4'),
        ('sample_method', 'iid'),
        ('r_cov_mode', 'diagonal'), #sphereical|diagonal|
        ('rr_mode', 'diagonal'), #diagonal|
        ('max_sent_len', '30'),
        ('optim', 'alternate'),
        ('encotype', 'linear3'),
        ('sent_per_epoch', '5000'), # 200|500|1000
        ('tag_dim', '256'),
        ('embedding_source', 'elmo_2|elmo|elmo_3'),
        ('lang', 'ar|es|zh|hi|fr|fi|cs|de|it|pt|ru')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def fore_de_cont_verb():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('prefix', '/home-4/xli150@jhu.edu/mutualinfo/jobs/mi_train-8/model')
    USR.set('type_token_reg', 'yes')
    USR.set('ablated', 'False')
    USR.set('mode', 'sem_recon')

 # can use prefix=/home-4/xli150@jhu.edu/mutualinfo/jobs/fore_tr_cont-3/model  # 5000
    # prefix=/home-4/xli150@jhu.edu/mutualinfo/jobs/fore_tr_cont-2/model   # 3000
    # and /home-4/xli150@jhu.edu/mutualinfo/jobs/fore_tr_cont-2/std

    # verb_subcat

    command = "%(S_python_itrptr)s %(S_python_dir)s/main.py" \
              " --dataset %(U_dataset_t)s" \
              " --dataset_dev %(U_dataset_d)s" \
              " --dataset_test %(U_dataset_test)s" \
              " --save_path %(S_model)s/{config}" \
              " --out_path %(S_output)s/{config}" \
              " --checkpoint_path %(U_prefix)s/{config}" \
              " --epoch 50" \
              " --mode %(U_mode)s" \
              " --word_threshold 1" \
              " --projective non-projective" \
              " --task VIB" \
              " --elmo_test_data_path %(U_elmo_test)s" \
              " --elmo_train_data_path %(U_elmo_train)s" \
              " --elmo_dev_data_path %(U_elmo_dev)s" \
              " --elmo_first %(U_elmo_first)s" \
              " --type_token_reg yes" \
              " --foreign yes" \
              " --ablation %(U_ablated)s" \
              " --lr 0.001" \
              % ALL()


    search_list = [
        ('batch_size', '5'),
        ('beta', '0.00001|0.01'), #0.00001|0.01|10
        ('gamma', '0.00001|0.01'), #|0.01|10
        ('seed', '1'),
        ('sample_size', '4'),
        ('sample_method', 'iid'),
        ('r_cov_mode', 'diagonal'), #sphereical|diagonal|
        ('rr_mode', 'diagonal'), #diagonal|
        ('max_sent_len', '30'),
        ('optim', 'alternate'),
        ('encotype', 'linear3'),
        ('sent_per_epoch', '5000'), # 200|500|1000
        ('tag_dim', '256'),
        ('embedding_source', 'elmo_2|elmo|elmo_3'),
        ('lang', 'fr|it|pt|ru|hi') #|fr|it|pt|ru') # fr | it | pt | ru | ar|es|zh|hi
    ]

    # search_list = [
    #     ('batch_size', '5'),
    #     ('beta', '0.00001|0.01'), #0.00001|0.01|10
    #     ('gamma', '0.00001|0.01'), #|0.01|10
    #     ('seed', '1'),
    #     ('sample_size', '4'),
    #     ('sample_method', 'iid'),
    #     ('r_cov_mode', 'diagonal'), #sphereical|diagonal|
    #     ('rr_mode', 'diagonal'), #diagonal|
    #     ('max_sent_len', '30'),
    #     ('optim', 'alternate'),
    #     ('encotype', 'linear3'),
    #     ('sent_per_epoch', '3000'), # 200|500|1000
    #     ('tag_dim', '256'),
    #     ('embedding_source', 'elmo_2|elmo|elmo_3'),
    #     ('lang', 'ar|es|zh') #|fr|it|pt|ru') # fr | it | pt | ru | ar|es|zh|hi
    # ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def fore_de_cont():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('prefix', '/home-4/xli150@jhu.edu/mutualinfo/jobs/mi_train-8/model')
    USR.set('type_token_reg', 'yes')
    USR.set('ablated', 'False')

 # can use prefix=/home-4/xli150@jhu.edu/mutualinfo/jobs/fore_tr_cont-3/model  # 5000
    # prefix=/home-4/xli150@jhu.edu/mutualinfo/jobs/fore_tr_cont-2/model   # 3000
    # and /home-4/xli150@jhu.edu/mutualinfo/jobs/fore_tr_cont-2/std

    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --checkpoint_path %(U_prefix)s/{config}' \
                   ' --epoch 150' \
                   ' --mode few_shot_cont' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task VIB' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg yes' \
                   ' --foreign yes' \
                   ' --ablation %(U_ablated)s' \
                   ' --lr 0.001' \
              % ALL()

    search_list = [
        ('batch_size', '5'),
        ('beta', '0.00001|0.01'), #0.00001|0.01|10
        ('gamma', '0.00001|0.01'), #|0.01|10
        ('seed', '1'),
        ('sample_size', '4'),
        ('sample_method', 'iid'),
        ('r_cov_mode', 'diagonal'), #sphereical|diagonal|
        ('rr_mode', 'diagonal'), #diagonal|
        ('max_sent_len', '30'),
        ('optim', 'alternate'),
        ('encotype', 'linear3'),
        ('sent_per_epoch', '5000'), # 200|500|1000
        ('tag_dim', '256'),
        ('embedding_source', 'elmo_2|elmo|elmo_3'),
        ('lang', 'fr|it|pt|ru') #|fr|it|pt|ru') # fr | it | pt | ru | ar|es|zh|hi
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return


@shepherd(before=[init], after=[post])
def fore_tr_iden():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')
    USR.set('ablated', 'False')
    USR.set('PCA', 'other')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task IDEN_Foreign' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg yes' \
                   ' --foreign yes' \
                   ' --PCA %(U_PCA)s' \
                   ' --tag_dim 256' \
              % ALL()

    if USR.ablated == 'True':
        command += ' --ablation True'

    search_list = [
        ('batch_size', '5'),
        ('seed', '1'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'), # 200|500|1000
        ('weight_decay', '0.0001|0.001|0.00001'),
        ('embedding_source', 'elmo_2|elmo|elmo_3'),
        ('lang', 'ar|es|zh|hi|fr|fi|cs|de|it|pt|ru')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return


@shepherd(before=[init], after=[post])
def mi_train_discrete_anneal():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task VIB_discrete' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg %(U_type_token_reg)s' \
              % ALL()

    search_list = [
        ('batch_size', '5'),
        ('beta', '0.00001'),
        ('gamma', '0.00001'),
        ('seed', '1'),
        ('sample_size', '8'),
        ('sample_method', 'iid'),
        ('r_cov_mode', 'diagonal'), #sphereical|diagonal|
        ('rr_mode', 'diagonal'), #diagonal|
        ('max_sent_len', '30'),
        ('optim', 'alternate'),
        ('encotype', 'linear3'),
        ('sent_per_epoch', '5000|2000'), # 200|500|1000
        ('tag_dim', '64'),
        ('embedding_source', 'elmo_2|elmo_3|elmo')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def mi_test_discrete_anneal():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('prefix', '/home-4/xli150@jhu.edu/mutualinfo/jobs/mi_train_discrete_anneal-1/model')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --checkpoint_path %(U_prefix)s/{config}' \
                   ' --epoch 50' \
                   ' --mode rate_distortion' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task VIB_discrete' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg %(U_type_token_reg)s' \
              % ALL()

    search_list = [
        ('batch_size', '5'),
        ('beta', '0.00001'),
        ('gamma', '0.00001'),
        ('seed', '1'),
        ('sample_size', '8'),
        ('sample_method', 'iid'),
        ('r_cov_mode', 'diagonal'), #sphereical|diagonal|
        ('rr_mode', 'diagonal'), #diagonal|
        ('max_sent_len', '30'),
        ('optim', 'alternate'),
        ('encotype', 'linear3'),
        ('sent_per_epoch', '2000'), # 200|500|1000
        ('tag_dim', '64'),
        ('embedding_source', 'elmo_2|elmo_3|elmo')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def mi_train_cont_anneal():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task VIB' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg %(U_type_token_reg)s' \
              % ALL()

    search_list = [
        ('batch_size', '5'),
        ('beta', '0.00001'),
        ('gamma', '0.00001'),
        ('seed', '1'),
        ('sample_size', '8'),
        ('sample_method', 'iid'),
        ('r_cov_mode', 'diagonal'), #sphereical|diagonal|
        ('rr_mode', 'diagonal'), #diagonal|
        ('max_sent_len', '30'),
        ('optim', 'alternate'),
        ('encotype', 'linear3'),
        ('sent_per_epoch', '5000|2000'), # 200|500|1000
        ('tag_dim', '256'),
        ('embedding_source', 'elmo_2|elmo_3')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def mi_train_discrete_no():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'no')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task VIB_discrete' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg %(U_type_token_reg)s' \
              % ALL()

    search_list = [
        ('batch_size', '5'),
        ('beta', '0|0.000001|0.0001|0.01|0.1|10'),
        ('seed', '1'),
        ('sample_size', '8'),
        ('sample_method', 'iid'),
        ('r_cov_mode', 'diagonal'), #sphereical|diagonal|
        ('rr_mode', 'diagonal'), #diagonal|
        ('max_sent_len', '30'),
        ('optim', 'alternate'),
        ('encotype', 'linear3'),
        ('sent_per_epoch', '5000'), # 200|500|1000
        ('tag_dim', '256|64'),
        ('embedding_source', 'elmo')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def mi_train_back():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    # USR.set('embedding', '/home-4/xli150@jhu.edu/data/lisa/Glove/glove.6B.50d.txt') # other options like 100d, 200d, 300d also available.
    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task VIB' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   % ALL()

    search_list = [
        ('batch_size', '5'),
        ('beta', '0.001|0.01|0'),
        ('seed', '1'),
        ('sample_size', '8'),
        ('sample_method', 'iid'),
        ('r_cov_mode', 'diagonal'), #sphereical|diagonal|
        ('rr_mode', 'diagonal'), #diagonal|
        ('max_sent_len', '30'),
        ('optim', 'alternate'),
        ('encotype', 'linear3'),
        ('sent_per_epoch', '5000'), # 200|500|1000
        ('tag_dim', '256'),
        ('embedding_source', 'elmo_2|elmo_3')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return


@shepherd(before=[init], after=[post])
def glove_train():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('embedding', '/home-4/xli150@jhu.edu/data/lisa/Glove/glove.6B.300d.txt')
    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --embedding %(U_embedding)s' \
                   ' --epoch 25' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --embedding_source glove' \
                   % ALL()

    search_list = [
        ('batch_size', '5'),
        ('beta', '0.000001|0.00001|0.0001|0.001|0.01|0.1|1|10'),
        ('seed', '1'),
        ('sample_size', '8'),
        ('sample_method', 'iid'),
        ('r_cov_mode', 'diagonal'), #sphereical|diagonal|
        ('rr_mode', 'diagonal'), #diagonal|
        ('max_sent_len', '30'),
        ('optim', 'alternate'),
        ('encotype', 'linear3'),
        ('sent_per_epoch', '5000'), # 200|500|1000
        ('tag_dim', '5|36|256|300')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def mi_test():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('prefix', '/home-4/xli150@jhu.edu/mutualinfo/jobs/mi_train-8/model')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    # USR.set('embedding', '/home-4/xli150@jhu.edu/data/lisa/Glove/glove.6B.50d.txt') # other options like 100d, 200d, 300d also available.
    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --checkpoint_path %(U_prefix)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode rate_distortion' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --embedding_source elmo' \
                   ' --task VIB' \
                   % ALL()

    search_list = [
        ('batch_size', '5'),
        ('beta', '0.000001|0.00001|0.0001|0.001|0.01|0.1|1|10'),
        ('seed', '1'),
        ('sample_size', '8'),
        ('sample_method', 'iid'),
        ('r_cov_mode', 'diagonal'), #sphereical|diagonal|
        ('rr_mode', 'diagonal'), #diagonal|
        ('max_sent_len', '30'),
        ('optim', 'alternate'),
        ('encotype', 'linear3'),
        ('sent_per_epoch', '2000'), # 200|500|1000
        ('tag_dim', '5')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return


@shepherd(before=[init], after=[post])
def pca_train():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    # USR.set('embedding', '/home-4/xli150@jhu.edu/data/lisa/Glove/glove.6B.50d.txt') # other options like 100d, 200d, 300d also available.
    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task IDEN' \
                   ' --PCA pca' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   % ALL()

    search_list = [
        ('batch_size', '5'),
        ('seed', '1'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'), # 200|500|1000
        ('tag_dim', '5|36|256|512'),
        ('embedding_source', 'elmo|elmo_2|elmo_3')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def iden_train():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')

    # USR.set('embedding', '/home-4/xli150@jhu.edu/data/lisa/Glove/glove.6B.50d.txt') # other options like 100d, 200d, 300d also available.
    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task IDEN' \
                   ' --PCA no' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   % ALL()

    search_list = [
        ('batch_size', '5'),
        ('seed', '1'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'), # 200|500|1000
        ('weight_decay', '0.0001|0.001|0.00001'),
        ('tag_dim', '5'),
        ('embedding_source', 'elmo|elmo_2|elmo_3')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def fine_train():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')

    # USR.set('embedding', '/home-4/xli150@jhu.edu/data/lisa/Glove/glove.6B.50d.txt') # other options like 100d, 200d, 300d also available.
    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task finetune' \
                   ' --PCA no' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --cuda 1' \
              % ALL()

    search_list = [
        ('batch_size', '36'),
        ('seed', '1|101'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'), # 200|500|1000
        ('weight_decay', '0.0001|0.001|0.00001'),
        ('embedding_source', 'elmo_2')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def clean_train():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')

    # USR.set('embedding', '/home-4/xli150@jhu.edu/data/lisa/Glove/glove.6B.50d.txt') # other options like 100d, 200d, 300d also available.
    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task CLEAN' \
                   ' --PCA no' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   % ALL()

    search_list = [
        ('batch_size', '5'),
        ('seed', '1'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'), # 200|500|1000
        ('weight_decay', '0.0001|0.001|0.00001'),
        ('tag_dim', '5'),
        ('embedding_source', 'elmo|elmo_2|elmo_3')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def fore_tr_clean():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')
    USR.set('ablated', 'False')
    USR.set('PCA', 'other')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task CLEAN_FOREIGN' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg yes' \
                   ' --foreign yes' \
                   ' --PCA %(U_PCA)s' \
                   ' --tag_dim 256' \
              % ALL()

    if USR.ablated == 'True':
        command += ' --ablation True'

    search_list = [
        ('batch_size', '5'),
        ('seed', '1'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'),
        ('weight_decay', '0.0001|0.001|0.00001'),
        ('embedding_source', 'elmo_2|elmo|elmo_3'),
        ('lang', 'ar|es|zh|hi|fr|cs|de|it|pt|ru')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return


@shepherd(before=[init], after=[post])
def fore_tr_pos():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')
    USR.set('ablated', 'False')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task POS_Foreign' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg yes' \
                   ' --foreign yes' \
                   ' --tag_dim 15' \
              % ALL()

    if USR.ablated == 'True':
        command += ' --ablation True'

    search_list = [
        ('batch_size', '5'),
        ('seed', '1'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '0|5000'), # 200|500|1000
        ('weight_decay', '0.0001|0.001|0.01|0'),
        ('embedding_source', 'elmo'),
        ('lang', 'ar|es|zh|hi|fr|de|it|pt|ru')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def fore_tr_stats():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    USR.set('elmo_test', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_test')
    USR.set('elmo_dev', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_dev')
    USR.set('elmo_train', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/elmo_train')
    USR.set('elmo_first', '/home-4/xli150@jhu.edu/data/lisa/ELMo_token/first_elmo_weights_')
    USR.set('type_token_reg', 'yes')
    USR.set('ablated', 'False')


    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --task stats' \
                   ' --elmo_test_data_path %(U_elmo_test)s' \
                   ' --elmo_train_data_path %(U_elmo_train)s' \
                   ' --elmo_dev_data_path %(U_elmo_dev)s' \
                   ' --elmo_first %(U_elmo_first)s' \
                   ' --type_token_reg yes' \
                   ' --foreign yes' \
                   ' --tag_dim 256' \
              % ALL()

    if USR.ablated == 'True':
        command += ' --ablation True'

    search_list = [
        ('batch_size', '5'),
        ('seed', '1'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '5000'), # 200|500|1000
        ('weight_decay', '0.0001'),
        ('embedding_source', 'elmo'),
        ('lang', 'fr')
        # ('lang', 'ar|es|zh|hi|fr|fi|cs|de|it|pt|ru')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return


@shepherd(before=[init], after=[post])
def pos_train():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    USR.set('dataset_test', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-test.conllu')
    # USR.set('embedding', '/home-4/xli150@jhu.edu/data/lisa/Glove/glove.6B.50d.txt') # other options like 100d, 200d, 300d also available.
    command = '%(S_python_itrptr)s %(S_python_dir)s/main.py' \
                   ' --dataset %(U_dataset_t)s' \
                   ' --dataset_dev %(U_dataset_d)s' \
                   ' --dataset_test %(U_dataset_test)s' \
                   ' --save_path %(S_model)s/{config}' \
                   ' --out_path %(S_output)s/{config}' \
                   ' --epoch 50' \
                   ' --mode train' \
                   ' --word_threshold 1' \
                   ' --projective non-projective' \
                   ' --embedding_source elmo' \
                   ' --task POS' \
                   % ALL()

    search_list = [
        ('batch_size', '5'),
        ('seed', '1'),
        ('weight_decay', '0.01|0.001|0.0001'),
        ('max_sent_len', '30'),
        ('sent_per_epoch', '0|5000') # 200|500|1000
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return
# python main.py --mode train --save_path ../others/debug_model15 --checkpoint_path ../others/debug_model15 --r_cov_mode diagonal --sample_method iid --sample_size 8 --epoch_sent 10 --epoch 3 --beta 0.005 --rr_mode diagonal --encotype linear3 --optim alternate --embedding /Users/xiangli/documents/glove/glove.6B.200d.txt --out_path 6elmo_beta=0.005-sent=100-non-proj-30 --tag_dim 5 --word_threshold 1 --batch_size 5 --projective non-proj --max_sent_len 30 --data_type CCG --sent_per_epoch 10 --task POS

'''
parser.py --outdir result --train data/en-universal-train.conll 
--dev data/en-universal-dev.conll --epochs 30 --lstmdims 125 --bibi-lstm
'''
@shepherd(before=[init], after=[post])
def bist_train():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset_t', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-train.conllu')
    USR.set('dataset_d', '/home-4/xli150@jhu.edu/workspace/data/UD_English/en-ud-dev.conllu')
    command = '%(S_python_itrptr)s %(S_python_dir)s/bist-parser-master/bmstparser/src/parser.py' \
                   ' --outdir %(S_model)s' \
                   ' --train %(U_dataset_t)s' \
                   ' --dev %(U_dataset_d)s' \
                   ' --epochs 30' \
                   ' --lstmdims 125' \
                   ' --bibi-lstm' \
                   % ALL()
    config = 'example_bist'
    print(command)
    SPD().submit(command=[command], config=config)

@shepherd(before=[init], after=[post])
def train_parser():
    USR.set('tb', 'lat-depunct-ud14')
    USR.set('train_split', 'train')
    SYS.before_exe = ''
    SYS.after_exe = ''
    USR.set('nepoch', '30')
    USR.set('beam', '8')
    USR.train_parser_proc = 'java -cp %(workspace)s/tools/yara.jar -Xmx20g YaraParser.Parser.YaraParser train' % SYS
    train_parser_fmt = '%(U_train_parser_proc)s' \
                       ' -train-file {train}' \
                       ' -feature ext' \
                       ' -train-prsv {rate}' \
                       ' -model %(S_model)s/{config}' \
                       ' iter:%(U_nepoch)s nt:%(S_cpus)s beam:%(U_beam)s' \
                       ' -punc-file %(S_workspace)s/tools/universal.puncs' \
                       ' -eval-file %(S_model)s/{config}.train_status.csv' \
                       % ALL()
    pattern = '.*/({lang})-ud-%s.conllu$' % USR.train_split
    lgs = _get_data(USR.set('lgs', 'train+dev'))
    USR.set('rate', '0.1|0.2|0.3|0.4|0.5|0.6|0.7|0.8|0.9|1.00')
    for rate in USR.rate.split('|'):
        rate = float(rate)
        for (lg,), train_fn in _itr_file_list('%(S_jobs)s/%(U_tb)s/output' % ALL(), pattern.format(lang='|'.join(lgs))):
            config = lg + '-r=%.2f' % rate
            dev_fn = train_fn.replace('-%s.' % USR.train_split, '-dev.')
            sub_train_cmd = train_parser_fmt.format(train=train_fn, config=config, rate=str(rate)) + \
                            f' -dev-file {dev_fn}' \
                            f' -out %(output)s/{os.path.basename(dev_fn)}-r={rate:.2f}' % SYS
            SPD().submit([sub_train_cmd], config)


@shepherd(before=[init], after=[post])
def test_parser():
    USR.set('md', 'lat-train_parser-ud14')
    USR.set('tb', 'lat-depunct-ud14')
    USR.set('beam', '8')
    SYS.before_exe = ''
    SYS.after_exe = ''
    USR.test_parser_proc = 'java -cp %(workspace)s/tools/yara.jar -Xmx20g YaraParser.Parser.YaraParser ' % SYS
    USR.eval_proc = 'python %(python_dir)s/preproc.py --task eval_parse' % SYS
    test_parser_fmt = '%(U_test_parser_proc)s' \
                      ' -parse_conll conll' \
                      ' -model {model}' \
                      ' -input {tgt}' \
                      ' nt:%(S_cpus)s beam:%(U_beam)s' \
                      ' -gold-file {tgt}' \
                      ' -out %(S_output)s/{config}' \
                      ' -eval-file %(S_evaluation)s/{config}' \
                      ' -punc-file %(S_workspace)s/tools/universal.puncs;' \
                      '%(U_eval_proc)s' \
                      ' --gold {tgt}' \
                      ' --pred %(S_output)s/{config}' \
                      ' --output %(S_evaluation)s/{config}.json' \
                      % ALL()
    md_pattern = '.*/(.*?).train_status.csv$'
    pattern = '.*/(?:{lang})-ud-.*.conllu$'
    for (lg,), md_fn in _itr_file_list('%(S_jobs)s/%(U_md)s/model' % ALL(), md_pattern):
        with open(md_fn, 'r') as pretrain:
            for l in pretrain:
                if 'bestLASMd' % USR in l:
                    best_model = l.strip().split(',')[1]
                    break
        for _, fn in _itr_file_list('%(S_jobs)s/%(U_tb)s/output' % ALL(), pattern.format(lang=lg.split('-')[0])):
            config = lg + '-f=' + os.path.basename(fn)
            cmd = test_parser_fmt.format(model=best_model, tgt=fn, config=config)
            SPD().submit([cmd], config)


@shepherd(before=[init], after=[post])
def pick_best():
    USR.set('prd', 'lat-test_parser-ud14_b')
    pattern = '.*-r=(.*)-f=({lang})-ud-dev.conllu$'.format(lang='|'.join(udv14))
    res = {}
    for prd_dir in filter(lambda x: x.startswith(USR.prd), os.listdir('%(S_jobs)s' % ALL())):
        USR.prd_dir = prd_dir
        for (rate, lg), fn in _itr_file_list('%(S_jobs)s/%(U_prd_dir)s/eval' % ALL(), pattern):
            with open(fn, 'r') as pretrain:
                for l in pretrain:
                    if 'LAS' % USR in l:
                        las = float(l.strip().split(',')[1])
                        break
            lg2rt = res.setdefault(lg, {})
            score = lg2rt.setdefault(float(rate), [0, None])
            if las > score[0]:
                score[0] = las
                score[1] = fn
    for lg, lg2rt in res.items():
        print(lg, end='')
        for rate, score in sorted(lg2rt.items()):
            print(',%.2f:%.2f' % (rate, score[0]), end='')
            if not _args.dry:
                fn, fn_base = score[1], os.path.basename(score[1])
                fn_prd = fn.replace('/eval/', '/output/')
                shutil.copy(fn, os.path.join(SYS.evaluation, fn_base))
                shutil.copy(fn + '.json', os.path.join(SYS.evaluation, fn_base + '.json'))
                shutil.copy(fn_prd, os.path.join(SYS.output, fn_base))
                shutil.copy(fn_prd.replace('ud-dev.conllu', 'ud-test.conllu'),
                            os.path.join(SYS.output, fn_base.replace('ud-dev.conllu', 'ud-test.conllu')))
        print('')

@shepherd(before=[init], after=[post])
def lmng():
    USR.set('train', 'ud-treebanks-v1.4|GD|half|gdv2')
    command = 'python %(S_python_dir)s/lmng/lm.py' \
              ' --src {src}' \
              ' --tgt {tgt}' % ALL()
    lgs = udv14
    pattern = '.*/({lang})-ud-train..*$' % USR
    search_list = [
        ('smoother', 'add1|add0.1|add0.01|add0.001'),
        ('punct', '0|1'),
        ('thr', '1|2|4|8'),
    ]
    for (tgt,), tgt_fn in _itr_file_list('%(S_data)s/ud-treebanks-v1.4/' % ALL(), pattern.format(lang='|'.join(lgs))):
        src_lst = []
        for sr in USR.train.split('|'):
            USR.sr = sr
            for _, src_fn in _itr_file_list('%(S_data)s/%(U_sr)s/' % ALL(), pattern.format(lang=tgt)):
                src_lst += [src_fn]
        for tgt_spl in ['train', 'dev', 'test']:
            config = tgt + '-' + tgt_spl
            tgt_file = tgt_fn.replace('-ud-train.', '-ud-%s.' % tgt_spl)
            cmd = command.format(src=' '.join(src_lst), tgt=tgt_file)
            grid_search(lambda map: basic_func(cmd, map,
                                               param_func=lambda x: config + '-' + x,
                                               jobid_func=lambda x: config + '-' + x),
                        search_list, seed=1)


@shepherd(before=[init], after=[post])
def lm():
    USR.set('train', 'ud-treebanks-v1.4')
    command = 'python %(S_python_dir)s/lm/main.py' \
              ' --save %(S_model)s/{config}' % ALL() + \
              ' --output %(S_output)s/{config}' % ALL() + \
              ' --train_path {train}' \
              ' --dev_path {dev}' \
              ' --test_path {test}'
    if SYS.device == 'gpu':
        command += ' --cuda 1'
    lgs = udv14
    pattern = '.*/({lang})-ud-train..*$' % USR
    for (src,), train_fn in _itr_file_list('%(S_data)s/%(U_train)s/' % ALL(), pattern.format(lang='|'.join(lgs))):
        for _, tgt_fn in _itr_file_list('%(S_data)s/ud-treebanks-v1.4/' % ALL(), pattern.format(lang=src)):
            config = src
            train = tgt_fn
            dev = tgt_fn.replace('-ud-train.', '-ud-dev.')
            test = tgt_fn.replace('-ud-train.', '-ud-test.')
            SPD().submit([command.format(config=config,
                                         train=train_fn,
                                         dev=dev,
                                         test=' '.join([train, dev, test])
                                         )], config)


@shepherd(before=[init], after=[post])
def collect_tb_info():
    USR.set('tb', 'ud-treebanks-v1.4')
    pattern = '.*/(.*?)-ud-train.conllu$' % USR
    command = 'python %(S_python_dir)s/permutation/main.py --task collect_info' % ALL()
    for (src,), src_fn in _itr_file_list('%(S_data)s/%(U_tb)s/' % ALL(), pattern):
        config = src
        src_spec = f' --test {src_fn}'
        SPD().submit([command + src_spec], config)



def _itr_file(input, pattern):
    print('Search Patterm:', pattern)
    ptn = re.compile(pattern)
    for root, dir, files in os.walk(input):
        for fn in files:
            abs_fn = os.path.normpath(os.path.join(root, fn))
            m = ptn.match(abs_fn)
            if m:
                lang = m.groups()
                yield lang, abs_fn

