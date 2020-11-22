import os, sys, zipfile, random

if len(sys.argv) != 2:
    print('USAGE: \'python3 sample_images.py [directory containing zip file]\'')
    exit()

zip_name = 'hirise-map-proj-v3.zip'
img_lst_name = 'labels-map-proj-v3.txt'
class_info_name = 'landmarks_map-proj-v3_classmap.csv'
img_dir = 'map-proj-v3'


base_dir = sys.argv[1]
zip_dir = os.path.join(base_dir, zip_name)
SAMPLE_NUM = 100
RANDOM_SEED = 123

random.seed(RANDOM_SEED)
img_names = []
with zipfile.ZipFile(zip_dir, 'r') as z:
    z.extract(img_lst_name, path=base_dir)
    z.extract(class_info_name, path=base_dir)
    with z.open(img_lst_name) as f:
        for ln in f:
            img_names.append(ln.decode('utf-8').split(' ')[0])
    sampled = random.sample(img_names, SAMPLE_NUM)
    if not os.path.exists(os.path.join(base_dir, img_dir)):
        os.makedirs(os.path.join(base_dir, img_dir))
    for img_nm in sampled:
        z.extract(img_dir + '/' + img_nm, path=base_dir)





