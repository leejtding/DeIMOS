from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import numpy as np
import time

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

def save_cluster_images(n, feats, labels, orig_ds):
    id_1 = str(int((time.time() - int(time.time())) * 1000))
    num_clusters = int(max(labels)+1)
    for i in range(num_clusters):
        label_map = {}
        ct = 0
        for j, lab in enumerate(labels):
            if lab == i:
                label_map[ct] = j
                ct += 1
        if not ct:
            continue
        cluster_data = feats[labels == i, :]
        cluster_mean = np.mean(cluster_data, axis=0)
        delta_dists = np.sum((cluster_data - cluster_mean)**2, axis=1)
        n_smallest_inds = np.argpartition(delta_dists, n)[:n]
        n_smallest_orig_inds = {label_map[ind] for ind in n_smallest_inds}
        for k, img in enumerate(orig_ds):
            if k in n_smallest_orig_inds:
                save_path = f'img/run_{id_1}_cluster_{i}_id_{k}.png'
                plt.imsave(save_path, img[:, :, 0].numpy(), cmap='Greys')


