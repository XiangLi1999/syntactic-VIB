lang_dict = {
    'ar': 'UD_Arabic-PADT',
    'hi': 'UD_Hindi-HDTB',
    'es': 'UD_Spanish-AnCora',
    'zh': 'UD_Chinese-GSD',
    # 'fr': 'UD_French-GSD',
    'fr': 'UD_French-ParTUT',
    'fi': 'UD_Finnish-FTB',
    'cs': 'UD_Czech-PDT',
    'de': 'UD_German-GSD',
    'it': 'UD_Italian-ISDT',
    'pt': 'UD_Portuguese-Bosque',
    'ru': 'UD_Russian-GSD',
}

lang_suff = {}
for key, val in lang_dict.items():
    suff = val.split('-')[1]
    lang_suff[key] = suff.lower()

def fetch_paths(lang):
    # base_path = '/home-4/xli150@jhu.edu/data/lisa/'
    base_path = '/home-4/xli150@jhu.edu/scratch/'
    # base_path_elmo = base_path + 'ELMo_many_langs'
    base_path_elmo = base_path + 'ELMo_models'
    # base_path_ud = base_path + 'UD_langs/ud-treebanks-v2.3'
    base_path_ud = base_path + 'ud-treebanks-v2.3'
    path_dict = {}
    bank_name = lang_dict[lang]
    suff_name = lang_suff[lang]
    path_dict['train'] = '{}/{}/{}_{}-ud-{}.conllu'.format(base_path_ud, bank_name, lang, suff_name, 'train')
    path_dict['dev'] = '{}/{}/{}_{}-ud-{}.conllu'.format(base_path_ud, bank_name, lang, suff_name, 'dev')
    path_dict['test'] = '{}/{}/{}_{}-ud-{}.conllu'.format(base_path_ud, bank_name, lang, suff_name, 'test')
    path_dict['elmo'] = '{}/{}_elmo'.format(base_path_elmo, lang)
    return path_dict


batch_size = 30
gamma = -1 
sample_size = 5
tag_dim=128
embedding_source = 'elmo_1'
hyper_param_dict1 = {
    'ar': {'weight_decay':0.00001, 'beta':0.01, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
           'tag_dim':tag_dim, 'embedding_source':embedding_source},
    'hi': {'weight_decay':0.00001, 'beta':0.01, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
           'tag_dim':tag_dim, 'embedding_source':embedding_source},
    'fr': {'weight_decay':0.0, 'beta':0.1, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
           'tag_dim':tag_dim, 'embedding_source':embedding_source},
    'es': {'weight_decay':0.0001, 'beta':0.1, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
           'tag_dim':tag_dim, 'embedding_source':embedding_source},
    'pt': {'weight_decay':0.0001, 'beta':0.1, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
           'tag_dim':tag_dim, 'embedding_source':embedding_source},
    'ru': {'weight_decay':0.00001, 'beta':0.1, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
           'tag_dim':tag_dim, 'embedding_source':embedding_source},
    'zh': {'weight_decay':0.00001, 'beta':0.1, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
           'tag_dim':tag_dim, 'embedding_source':embedding_source},
    'it': {'weight_decay':0.0001, 'beta':0.01, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
       'tag_dim':tag_dim, 'embedding_source':embedding_source},
}


# embedding_source = 'elmo_2'
# hyper_param_dict2 = {
#     'ar': {'weight_decay':0.00001, 'beta':0.01, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
#            'tag_dim':tag_dim, 'embedding_source':embedding_source},
#     'hi': {'weight_decay':0.00001, 'beta':0.01, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
#            'tag_dim':tag_dim, 'embedding_source':embedding_source},
#     'fr': {'weight_decay':0.0, 'beta':0.1, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
#            'tag_dim':tag_dim, 'embedding_source':embedding_source},
#     'es': {'weight_decay':0.0001, 'beta':0.1, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
#            'tag_dim':tag_dim, 'embedding_source':embedding_source}, 
#     'pt': {'weight_decay':0.0001, 'beta':0.1, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
#            'tag_dim':tag_dim, 'embedding_source':embedding_source},
#     'ru': {'weight_decay':0.00001, 'beta':0.1, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
#            'tag_dim':tag_dim, 'embedding_source':embedding_source},
#     'zh': {'weight_decay':0.00001, 'beta':0.1, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
#            'tag_dim':tag_dim, 'embedding_source':embedding_source},
#     'it': {'weight_decay':0.0001, 'beta':0.01, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
#        'tag_dim':tag_dim, 'embedding_source':embedding_source},
# }

# embedding_source = 'elmo_0'
# hyper_param_dict0 = {
#     'ar': {'weight_decay':0.00001, 'beta':0.01, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
#            'tag_dim':tag_dim, 'embedding_source':embedding_source},
#     'hi': {'weight_decay':0.00001, 'beta':0.01, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
#            'tag_dim':tag_dim, 'embedding_source':embedding_source},
#     'fr': {'weight_decay':0.0, 'beta':0.1, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
#            'tag_dim':tag_dim, 'embedding_source':embedding_source},
#     'es': {'weight_decay':0.0001, 'beta':0.1, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
#            'tag_dim':tag_dim, 'embedding_source':embedding_source}, 
#     'pt': {'weight_decay':0.0001, 'beta':0.1, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
#            'tag_dim':tag_dim, 'embedding_source':embedding_source},
#     'ru': {'weight_decay':0.00001, 'beta':0.1, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
#            'tag_dim':tag_dim, 'embedding_source':embedding_source},
#     'zh': {'weight_decay':0.00001, 'beta':0.1, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
#            'tag_dim':tag_dim, 'embedding_source':embedding_source},
#     'it': {'weight_decay':0.0001, 'beta':0.01, 'batch_size':batch_size, 'gamma':gamma, 'sample_size':sample_size, 
#        'tag_dim':tag_dim, 'embedding_source':embedding_source},
# }

if __name__ == '__main__':
    fetch_paths('ar')



