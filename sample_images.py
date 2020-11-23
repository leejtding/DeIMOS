import os, sys, zipfile, random
import numpy as np

def sample_images(base_dir):
    # if len(sys.argv) != 2: print('USAGE: \'python3 sample_images.py [
    # directory containing zip file]\'') exit()

    zip_name = 'deimos.zip'
    img_lst_name = 'hirise-map-proj-v3_2/labels-map-proj_v3_2.txt'
    class_info_name = 'hirise-map-proj-v3_2/landmarks_map-proj-v3_2_classmap.csv'
    img_dir = 'hirise-map-proj-v3_2/map-proj-v3_2'
    unlabeled_dir = 'hirise-map-proj-v3_2/map-proj-v3_2/unlabeled'
    labeled_dir = 'hirise-map-proj-v3_2/map-proj-v3_2/labeled'

    # base_dir = sys.argv[1]
    zip_dir = os.path.join(base_dir, zip_name)

    SAMPLE_NUM = 100
    RANDOM_SEED = 123
    random.seed(RANDOM_SEED)

    img_names = []
    img_labels = []

    with zipfile.ZipFile(zip_dir, 'r') as z:
        with z.open(img_lst_name) as f:
            for ln in f:
                img = ln.decode('utf-8').split(' ')
                img_names.append(img[0])
                img_labels.append(img[1].strip())

        img_names_arr = np.array(img_names)
        img_labels_arr = np.array(img_labels)

        index_list = np.indices(np.shape(img_names_arr)).flatten().tolist()
        sampled_indices = random.sample(index_list, SAMPLE_NUM)
        img_names_samples = img_names_arr[sampled_indices]
        img_labels_samples = img_labels_arr[sampled_indices]

        unlabeled_inds = np.where(img_labels_samples == '0')
        labeled_inds = np.where(img_labels_samples != '0')

        unlabeled_sampled = img_names_samples[unlabeled_inds].tolist()
        labeled_sampled = img_names_samples[labeled_inds].tolist()

        z.extract(img_lst_name, path=base_dir)
        z.extract(class_info_name, path=base_dir)

        if not os.path.exists(os.path.join(base_dir, unlabeled_dir)):
            os.makedirs(os.path.join(base_dir, unlabeled_dir))
        for img_nm in unlabeled_sampled:
            z.extract(img_dir + '/' + img_nm, path=base_dir)

        if not os.path.exists(os.path.join(base_dir, labeled_dir)):
            os.makedirs(os.path.join(base_dir, labeled_dir))
        for img_nm in labeled_sampled:
            z.extract(img_dir + '/' + img_nm, path=base_dir)


sample_images('data')
