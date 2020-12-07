from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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

def tsne_visualization(feats, **params):
    tsne = TSNE(**params)
    embedded = tsne.fit_transform(feats)
    plt.scatter(embedded[:, 0], embedded[:, 1])
    plt.show()


