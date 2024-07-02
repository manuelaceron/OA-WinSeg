from __future__ import print_function, division
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tools import utils
import pdb, json


class AmodalSegmentation(Dataset):
    """
    Simulated occlusion dataset
    """

    def __init__(self,
                 txt_path=None,
                 transform=None,
                 occSegFormer = False,
                 feature_extractor = None,
                 processor = None,
                 jsonAnnotation = None,
                 simulated_dataset = False,
                 img_path = None
                 ):
        if img_path is None:
            self.list_sample = open(txt_path, 'r').readlines()
        else:
            self.list_sample = img_path

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        
        print('# samples: {}'.format(self.num_sample))
        self.transform = transform
        self.feature_extractor = feature_extractor
        self.occSegFormer = occSegFormer               
        self.processor = processor
        self.simulated_dataset = simulated_dataset
                
        

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        
        # Load dataset items
        _img, _target, img_path, gt_path, _occ, clean_image, occ_path, _visible, visible_path = self._make_img_gt_point_pair(index) 
        
        # Create dictionary with elements of the sample
        sample = {'image': _img, 'gt': _target, 'img_sf': _img}

        pro_target = None
        
        # Add additional elements to the sample dictionary
        if self.simulated_dataset:      
            sample = {'image': _img, 'gt': _target, 'img_sf': _img, 'occ': _occ, 'occ_sf': _occ, 'visible_mask': _visible}                                    
        
        # Perform data augmentation to the sample
        if self.transform is not None:
            sample = self.transform(sample)
        
        if not self.simulated_dataset:
            sample['occ'] = None
            sample['occ_sf'] = None
            sample['visible_mask'] = None
            sample['clean_image'] = None
            sample['pro_target'] = None
        else:
            sample['pro_target'] = pro_target
            sample['clean_image'] = None
        
        sample['img_path'] = img_path
        sample['gt_path'] = gt_path
        sample['occ_path'] = occ_path
        sample['visible_path'] = visible_path
        
        # Extract features for occlusion detection mode
        if self.occSegFormer:
            if self.simulated_dataset:
                encoded_inputs = self.feature_extractor(sample['img_sf'], sample['occ_sf'], return_tensors="pt")
                for k,v in encoded_inputs.items():
                    encoded_inputs[k].squeeze_()
            
                sample['df_fimage'] = encoded_inputs['pixel_values']
                sample['df_fooc'] = encoded_inputs['labels']            
            else:
                encoded_inputs = self.feature_extractor(sample['img_sf'], return_tensors="pt")
                encoded_inputs['pixel_values'].squeeze_()
                sample['df_fimage'] = encoded_inputs['pixel_values'] 
                sample['df_fooc'] = None                               
            
        return sample

    def _make_img_gt_point_pair(self, index):
        
        file = self.list_sample[index].strip()

        img_path = file.split('  ')[0] #RGB image
        gt_path = file.split('  ')[1] #GT WSeg
        
        _img = utils.read_image(os.path.join(img_path))
        _target = utils.read_image(os.path.join(gt_path), 'gt').astype(np.int32)
        
        # By default, only processing RGB image and label
        _occ = None
        _visible = None
        _clean_img = None
        occ_path = None
        visible_path = None
        clean_img_path = None

        # If working with simulated dataset, load more dataset elements
        if self.simulated_dataset:   
            occ_path = file.split('  ')[2] #GT occluder 
            visible_path = file.split('  ')[3] #GT visible windows 
            clean_img_path = file.split('  ')[4] #GT clean image    
                        
            _occ = utils.read_image(os.path.join(occ_path), 'gt').astype(np.int32)        
            _visible = utils.read_image(os.path.join(visible_path), 'gt').astype(np.int32)
            #_clean_img = utils.read_image(os.path.join(clean_img_path))
        
        return _img, _target, img_path, gt_path, _occ, _clean_img, occ_path, _visible, visible_path
    

def resize_annotation(annotation, orig_size, target_size):

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size))
    ratio_height, ratio_width = ratios
    
    for key, value in annotation.items():
        
        if key == 'annotations':            
            for val in value:                
                
                boxes = val['bbox']
                scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
                val['bbox'] =  scaled_boxes

                
                area = val['area']
                scaled_area = area * (ratio_width * ratio_height)
                val['area'] = scaled_area

                val['size'] = target_size
                                
        if key == 'images':
            for val in value:
                val['height'] =  int(orig_size[0]* ratio_height)
                val['width'] =  int(orig_size[1]* ratio_width)
    
             
    return annotation           