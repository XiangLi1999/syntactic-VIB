# the following elmo related function are for CPU, please use load_elmo_batch in experiment_helper.py for GPU code. 
def load_elmo(data_loader, args, embed_loader, mod='train', processed_sent_dev=None, processed_sent_test=None):
    if mod == 'train':
        if args.embedding_source == 'elmo_2':
            elmo_embeds_train = train_token_loader(args.elmo_train_data_path, data_loader, embed_loader, dim=1)
        elif args.embedding_source == 'elmo_3':
            elmo_embeds_train = train_token_loader(args.elmo_train_data_path, data_loader, embed_loader, dim=2)
        else:
            elmo_embeds_train = embed_loader.elmo_embeddings_first(data_loader.processed_sent, args.epoch_sent,
                                                                   args.elmo_first + mod)
        elmo_embeds_train_type = embed_loader.elmo_embeddings_first(data_loader.processed_sent,
                                                               len(data_loader.processed_sent),
                                                               args.elmo_first + mod)

        return elmo_embeds_train, elmo_embeds_train_type
    elif mod == 'dev':
        assert processed_sent_dev is not None
        if args.embedding_source == 'elmo_2':
            elmo_embeds_dev = dev_token_loader(args.elmo_dev_data_path, processed_sent_dev,
                                               embed_loader, dim=1)
        elif args.embedding_source == 'elmo_3':
            elmo_embeds_dev = dev_token_loader(args.elmo_dev_data_path, processed_sent_dev,
                                               embed_loader, dim=2)
        else:
            elmo_embeds_dev = embed_loader.elmo_embeddings_first(processed_sent_dev,
                                                                 len(processed_sent_dev),
                                                                 args.elmo_first + 'dev')

        elmo_embeds_dev_type = embed_loader.elmo_embeddings_first(processed_sent_dev,
                                                             len(processed_sent_dev),
                                                             args.elmo_first + 'dev')

        return elmo_embeds_dev, elmo_embeds_dev_type
    elif mod == 'test':
        assert processed_sent_test is not None
        if args.embedding_source == 'elmo_2':
            elmo_embeds_dev = dev_token_loader(args.elmo_test_data_path, processed_sent_test,
                                               embed_loader, dim=1)
        elif args.embedding_source == 'elmo_3':
            elmo_embeds_dev = dev_token_loader(args.elmo_test_data_path, processed_sent_test,
                                               embed_loader, dim=2)
        else:
            elmo_embeds_dev = embed_loader.elmo_embeddings_first(processed_sent_test,
                                                                 len(processed_sent_test),
                                                                 args.elmo_first + 'test')
        elmo_embeds_dev_type = embed_loader.elmo_embeddings_first(processed_sent_test,
                                                             len(processed_sent_test),
                                                             args.elmo_first + 'test')
        return elmo_embeds_dev, elmo_embeds_dev_type


def train_token_loader2(train_data_path, dim=1):
    big_lst = []
    for index in range(11):
        try:
            with open(train_data_path + str(index), 'rb') as f:
                elmo_embeds_train = pickle.load(f)
            big_lst += elmo_embeds_train[dim]
        except:
            print('error with loading files, retry:')
            sys.stdout.flush()
            with open(train_data_path + str(index) + '_np.pkl', 'rb') as f:
                elmo_embeds_train = pickle.load(f)
            temp = elmo_embeds_train[dim]
            lst_temp = []
            for elem in temp:
                lst_temp.append(torch.tensor(elem))
            big_lst += lst_temp
    return big_lst

def train_token_loader(train_data_path, data_loader, embed_loader,  dim=1):
    big_lst = []
    for index in range(11):
        try:
            with open(train_data_path + str(index), 'rb') as f:
                elmo_embeds_train = pickle.load(f)
            big_lst += elmo_embeds_train[dim]
        except:
            print('error with loading files, retry from computing:')
            sys.stdout.flush()
            elmo_embeds_train = embed_loader.elmo_embeddings(data_loader.en_batch,
                                                             len(data_loader.en_batch))
            big_lst = elmo_embeds_train[dim]
            return big_lst
    return big_lst


def dev_token_loader(dev_path, dev_sent, embed_loader, dim=1):
    try:
        with open(dev_path, 'rb') as f:
            elmo_embeds_train = pickle.load(f)
        return elmo_embeds_train[dim]
    except:
        print('error with loading (dev/test) files, retry from computing:')
        sys.stdout.flush()
        elmo_embeds_train = embed_loader.elmo_embeddings(dev_sent,
                                                         len(dev_sent))
        return elmo_embeds_train[dim]

def dev_token_loader2(dev_path, dim=1):
    try:
        with open(dev_path, 'rb') as f:
            elmo_embeds_train = pickle.load(f)
        return elmo_embeds_train[dim]
    except:
        print('error with loading (dev/test) files, retry:')
        sys.stdout.flush()
        with open(dev_path + '_np.pkl', 'rb') as f:
            elmo_embeds_train = pickle.load(f)
        temp = elmo_embeds_train[dim]
        lst_temp = []
        for elem in temp:
            lst_temp.append(torch.tensor(elem))
        return lst_temp


def cluster_sub_func(lst_tag, lst_cand):
    tree_path_dict = defaultdict(list)
    word2path_dict = {}
    print([np.unique(tag_layer) for tag_layer in lst_tag])
    for ind, elem in  enumerate(lst_cand):
        enhanced_tag = tuple([tag_layer[ind][0] for tag_layer in lst_tag])
        tree_path_dict[enhanced_tag].append(elem)
        word2path_dict[elem] = enhanced_tag

    embedded_dict = defaultdict(lambda: defaultdict( lambda: defaultdict( lambda: defaultdict (dict))))
    for path, elem in tree_path_dict.items():
        embedded_dict[path[0]][path[1]][path[2]][path[3]][path[4]] = elem
    print(embedded_dict)
    return word2path_dict, tree_path_dict