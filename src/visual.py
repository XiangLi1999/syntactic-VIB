import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from matplotlib.pyplot import cm
from matplotlib.backends.backend_pdf import PdfPages
# import tsne
import pickle

color_lst = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']



from sklearn.manifold import TSNE
def robost_tsne(mean_lst, visual_out_path, tag_lst, word_lst=None, dim=2):
    if visual_out_path:
        pp = PdfPages(visual_out_path)

    X_new = TSNE(n_components=2).fit_transform(mean_lst)

    if tag_lst is not None:
        fig, ax = plt.subplots()
        # filter the mean_lst into the POS groups.
        tag_set = set(tag_lst)
        colors=cm.rainbow(np.linspace(0,1,len(tag_set)))
        for pos, color in zip(tag_set, colors):
            indices = np.array([i for i, x in enumerate(tag_lst) if x == pos])
            temp_tag = X_new[indices]
            ax.scatter(temp_tag[:,0], temp_tag[:,1], color=color, label=pos)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})


    if word_lst is not None:
        for i, txt in enumerate(word_lst):
            ax.annotate(txt, (X_new[i, 0], X_new[i, 1]), fontsize=3)

    if visual_out_path:
        pp.savefig()
    else:
        plt.show()

    # also want to dump the data for plot for better visualization
    data = {}
    data["tag_lst"] = tag_lst
    data["X_new"] = X_new

    with open(visual_out_path+'save', 'wb') as f:
        pickle.dump(data, f)

    if visual_out_path:
        pp.close()
    return X_new

