#Code to generate a .txt file with the list of training, validation and test samples for a given dataset

import os, glob
import random
import sys
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--state", default="train")
parser.add_argument("-d", "--dataset", default="ecp")
parser.add_argument("-n", "--name", default="full-occ60")
args = parser.parse_args()

state = args.state
dataset = args.dataset
name_file = args.name

# set source path of custom dataset 
if dataset == 'cmp':
    src_path = r'/home/cero_ma/MCV/window_benchmarks/originals/resized/cmp-occ60/'
elif dataset == 'artdeco':
    src_path =  r'/home/cero_ma/MCV/window_benchmarks/originals/resized/artdeco-2class-1class-mcv/'
elif dataset == 'ecp':
    src_path = r'/home/cero_ma/MCV/window_benchmarks/originals/resized/'+dataset+'-2class-1class-mcv-occ60/'  
elif dataset == 'modern': 
    src_path =  r'/home/cero_ma/MCV/window_benchmarks/originals/modern_dataset/resized/m-occ100/' 
elif dataset == 'full': 
    src_path =  r'/home/cero_ma/MCV/window_benchmarks/originals/split_data/full_occ100/' 
elif dataset == 'full-80': 
    src_path =  r'/home/cero_ma/MCV/window_benchmarks/originals/split_data/full_occ80/' 
elif dataset == 'inference': 
    src_path =  r'/home/cero_ma/MCV/window_benchmarks/originals/data_inference/'
elif dataset == 'modern-80': 
    src_path =  r'/home/cero_ma/MCV/window_benchmarks/originals/modern_dataset/modern_occ80/'
elif dataset == 'new_inference': 
    src_path =  r"/home/cero_ma/MCV/window_benchmarks/originals/new_data_inference_facades/images"
elif dataset == 'full-modern-occ-80': 
    src_path =  r"/home/cero_ma/MCV/window_benchmarks/originals/split_data/full_modern_occ80/"

# set path of each element of the dataset
if 'inference' not in dataset:
    clean_image = os.path.join('/home/cero_ma/MCV/window_benchmarks/originals/resized/',dataset,state, 'images')
    image_path = os.path.join(src_path, state, 'images')
    label_path = os.path.join(src_path, state, 'labels')
    occ_path = os.path.join(src_path, state, 'occ_masks')
    visible_path = os.path.join(src_path, state, 'occ_labels')
else:
    image_path = src_path
    label_path = src_path

# Set of real datasets without simulated occlusions
real = {'artdeco', 'inference', 'new_inference'}

# write .txt file with names of all samples in the dataset
images = os.listdir(image_path)
random.shuffle(images)

with open('{}_list_{}.txt'.format(state, name_file), 'a') as ff: 
    for name in images:
        if name.split('.')[1] == 'jpg' :
            name_label = name.replace('jpg', 'png')
            name_modal = name.replace('jpg', 'png')
        else:
            name_label = name
            name_modal = name
        
        if os.path.exists(os.path.join(image_path, name)) is True and os.path.exists(os.path.join(label_path, name_label)) is True:
        
            # Real occluded dataset
            if dataset in real:
                cur_info = '{}  {}\n'.format(os.path.join(image_path, name), os.path.join(label_path, name_label))
            # Simulated dataset
            else:                
                cur_info = '{}  {}  {}  {}  {}\n'.format(os.path.join(image_path, name), os.path.join(label_path, name_label), os.path.join(occ_path, name_modal), 
                os.path.join(visible_path, name_label), os.path.join(clean_image, name)) 
        

        ff.writelines(cur_info)

