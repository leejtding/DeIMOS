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


'''
Function that uses pre-trained model to extract features from image
Edit: idk if I am actually going to write this anymore
'''

