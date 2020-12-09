from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import numpy as np

def get_class_dict(class_label_path):
    """
    Function to return dictionary that maps [image name] to class (int)
    :param class_label_path: the path to the class label file.
    :return: a dictionary of the file names mapped to classes.
    """
    class_dict = {}
    with open(class_label_path, 'r') as f:
        for ln in f:
            splts = ln.split(' ')
            class_dict[splts[0]] = int(splts[1])

    return class_dict

def tsne_visualization(feats, labels, show=True, **params):
    plt.figure()
    labels = labels.astype(np.int32)
    tsne = TSNE(**params)
    embedded = tsne.fit_transform(feats)
    for i in range(max(labels)+1):
        mask = (labels == i)
        if np.sum(mask) == 0:
            continue
        plt.scatter(embedded[mask, 0], embedded[mask, 1], cmap='tab10', label=f'Cluster {i}')
    plt.legend()
    if show:
        plt.show()


