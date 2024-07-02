from __future__ import division
import sys
import os, math, time
import torchvision
from tools.utils import read_image
from tools.metrics import get_acc_v2
import numpy as np
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader, Dataset
import tqdm
from collections import OrderedDict
import tools.transform as tr
from tools.dataloader import AmodalSegmentation
import tools
import torch
from networks.get_model import get_net
from tools.parse_config_yaml import parse_yaml
import torch.onnx
import pdb, cv2
import torch.nn as nn
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, DetrImageProcessor
from transformers.image_transforms import center_to_corners_format
import albumentations as aug
from timm.optim import create_optimizer_v2
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from networks.CompletionNet.networks_BK0 import Generator as CN_Generator
from PIL import Image
from networks.CompletionNet.utils import draw_grd, calculate_sym, corner_detector_algorithm, corner_reg_inference
from scipy import ndimage as ndi
from skimage.color import label2rgb
from scipy.ndimage import label
from skimage.segmentation import active_contour
import timm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_weight(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


def directVectorization(pred, img_size):
    # Contour-based Boundary detection

    contours, _ = cv2.findContours(image=np.uint8(pred), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    vect_img = np.zeros((img_size,img_size), dtype='uint8')
        
    for component in contours:
        x,y,w,h = cv2.boundingRect(component)            
        cv2.rectangle(vect_img, (x, y), (x+w, y+h), 1, -1) 
    
    """ cv2.imshow('vect_image', ((vect_img*255)).astype(float))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pdb.set_trace()  """ 

    return vect_img 

def activeContour(pred):
    # Contour-based Active contours
    
    labeled_array, num_features = label(pred) 
    img = np.zeros((512, 512), dtype=np.uint8)
    
    for idx in range(1, num_features+1):                        
        new_mask = np.where(labeled_array == idx, 1, 0)
        
        singl_cnt, _ = cv2.findContours(image=np.uint8(new_mask),mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        if len(singl_cnt) > 1:
            print('1')
            pdb.set_trace()
        x,y,w,h = cv2.boundingRect(singl_cnt[0])
        
        #------
        margin = 1

        """ x -= margin
        y -= margin
        w += 2 * margin
        h += 2 * margin """

        x += margin
        y += margin
        w -= 2 * margin
        h -= 2 * margin

        x = max(0, x)
        y = max(0, y)

        #------

        x1 = x+w
        y1 = y+h

        s = np.linspace(0, 1, 2)                
        top_side = np.column_stack((np.full_like(s, y), x + s * (x1 - x)))
        right_side = np.column_stack((y + s * (y1 - y), np.full_like(s, x1)))
        bottom_side =  np.column_stack((np.full_like(s, y1), x1 - s * (x1 - x)))
        left_side = np.column_stack((y1 - s * (y1 - y), np.full_like(s, x)))
        
        init = np.concatenate((top_side, right_side, bottom_side, left_side))

        snake = active_contour(new_mask.astype(float), init, alpha=0.015, beta=0.1, gamma=0.0001, boundary_condition='periodic', w_line=0.5, w_edge=0.9)

        #img = cv2.fillPoly(img, np.int32([np.column_stack((init[:, 1], init[:, 0]))]), color=(255,0,0))
        img = cv2.fillPoly(img, np.int32([np.column_stack((snake[:, 1], snake[:, 0]))]), color=1)
                            
    return img

def collate_fn(batch):
    # 'image' 'gt' 'img_sf' 'occ' 'occ_sf' 'visible_mask' 'occ_clean_image' 'pro_target' 'img_path' 'gt_path' 'occ_path' 'visible_path' 'df_fimage' 'df_fooc'
    
    tmp = []
    n_batch = {}
    for item in batch: 
        tmp.append(item.values())
    
    inputs = list(zip(*tmp))         

    n_batch["image"] = inputs[0]
    n_batch["gt"] = inputs[1]    
    n_batch["img_path"] = inputs[8]
    n_batch["gt_path"] = inputs[9]
    n_batch["occ"] = inputs[3]
    #n_batch["gt_path"] = inputs[10]
        
    if param_dict['adversarial']:
        n_batch["occ"] = inputs[3]
        n_batch["visible_mask"] = inputs[5]
        n_batch["pro_target"] = inputs[7]
        n_batch['df_fimage'] = inputs[12]
        n_batch['df_fooc'] = inputs[13]
    
    return n_batch

def test(testloader, model, epoch):
    # Occlusion network
    if param_dict['use-occ-model']:

        if 'ecp' in param_dict['dataset'] :#or 'inference' in param_dict['dataset']:
            checkpoint_path = param_dict['ecp_occ_model']
        elif 'full-occ60' in param_dict['dataset']: 
            checkpoint_path = param_dict['full_occ60_occ_model'] 
        elif 'full_modern' in param_dict['dataset']: 
            checkpoint_path = param_dict['full_occ60_occ_model'] #param_dict['full_modern_occ80_occ_model'] 
        if 'inference' in param_dict['dataset'] :
            checkpoint_path = param_dict['full_occ60_occ_model']
        if 'artdeco' in param_dict['dataset'] :
            checkpoint_path = param_dict['ecp_occ_model']
        
        state_dict = torch.load(checkpoint_path)['net']    

        pre_model = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024" 
        id2label = {0: 'background', 1: 'occlusion'}
        label2id = {'background': 0, 'occlusion': 1}
        
        occ_model = SegformerForSemanticSegmentation.from_pretrained(pre_model, ignore_mismatched_sizes=True,
                                                            num_labels=len(id2label), id2label=id2label, label2id=label2id,
                                                            reshape_last_stage=True)

        occ_model = torch.nn.DataParallel(occ_model, device_ids=[0])
        occ_model.load_state_dict(state_dict)
        print('epoch: ', torch.load(checkpoint_path)['epoch'], checkpoint_path)

        occ_model.eval()

        for param in occ_model.parameters():
            param.requires_grad = False
        
        occ_model.cuda()
    
    # Coarse netwoek
    if param_dict['use-coarse-model']:
        
        if 'ecp' in param_dict['dataset']:                
            checkpoint_path = param_dict['ecp_coarse_model'] 
        elif 'artdeco_ori' in param_dict['dataset']:
            checkpoint_path = param_dict['artdeco_ori_coarse_model']
        elif 'artdeco_ref' in param_dict['dataset']:
            checkpoint_path = param_dict['artdeco_ref_coarse_model']
        elif 'full-occ60' in param_dict['dataset']:                
            checkpoint_path = param_dict['full_occ60_coarse_model']
        elif 'full_modern_occ80' in param_dict['dataset']:            
            checkpoint_path = param_dict['full_modern_occ80_coarse_model'] 
        elif 'modern' in param_dict['dataset']: 
            checkpoint_path = param_dict['modern_coarse_model']
        elif 'full-100' in param_dict['dataset']:            
            checkpoint_path = param_dict['full_occ100_coarse_model']
        elif 'full-occ80' in param_dict['dataset']:            
            checkpoint_path = param_dict['full_occ80_coarse_model']

        elif 'inference' in param_dict['dataset']:            
            checkpoint_path = param_dict['full_modern_occ80_coarse_model'] 

        print('Using coarse complete model')
        
        state_dict = torch.load(checkpoint_path)['net']
        visi_model = get_net('Res_UNet_101', 3, 1, 512, None) 
        #visi_model = get_net('UNet', 3, 1, 512, None)        
        #visi_model = get_net('DeepLabV3Plus', 3, 1, 512, param_dict['pretrained_model'])
        visi_model = torch.nn.DataParallel(visi_model, device_ids=[0])
        visi_model.load_state_dict(state_dict)
        visi_model.eval()

        for param in visi_model.parameters():
            param.requires_grad = False
        
        visi_model.cuda()

        print('epoch: ', torch.load(checkpoint_path)['epoch'], checkpoint_path)        

    if param_dict['adversarial']:

        checkpoint_path = os.path.join(param_dict['model_dir'], param_dict['main_model_inference'])
        state_dict = torch.load(checkpoint_path)['net']

        im_channel = 1
        im_channel_mid = 1
        im_channel_out = 1
        G = CN_Generator(cnum_in=im_channel+3, cnum_mid = im_channel_mid, cnum_out=im_channel_out, cnum=48, return_flow=True, k= param_dict['1D_kernel'])

        model = torch.nn.DataParallel(G, device_ids=[0])
        model.load_state_dict(state_dict)
        epoch = torch.load(checkpoint_path)['epoch']            
        model = model.cuda()
        model = model.eval()
        print('Main model epoch: ', epoch)

    if param_dict['corner_reg']:
        checkpoint_path = param_dict['corner_reg_model']
        state_dict = torch.load(checkpoint_path)['net']
        
        model_cr = timm.create_model('seresnet50', pretrained=True, in_chans=4)                                
        model_cr.fc = nn.Linear(model_cr.fc.in_features, 4)

        model_cr = torch.nn.DataParallel(model_cr, device_ids=[0])
        model_cr.load_state_dict(state_dict)
        epoch = torch.load(checkpoint_path)['epoch'] 
        model_cr = model_cr.cuda()
        model_cr = model_cr.eval()

        print('Corner regressor model epoch: ', epoch)

    test_num = len(testloader.dataset)

    # To calculate metric of final window segmentation (complete)
    label_all = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    # To calculate metrics with result of vectorization    
    vect_predict_all = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    cornerReg_predict_all = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    activeC_predict_all = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all_Avcorners = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    label_all_mask = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all_mask = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    label_all_vis = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all_vis = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    # To calculate metric of segmentation of hidden windows
    label_all_hidden = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all_hidden = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    if os.path.exists(param_dict['pred_path']) is False:
        os.mkdir(param_dict['pred_path'])
    
    if os.path.exists(os.path.join(param_dict['pred_path'], 'overL')) is False: 
        os.makedirs(os.path.join(param_dict['pred_path'], 'overL'))     
    
    if os.path.exists(os.path.join(param_dict['pred_path'], 'overLvect')) is False:
        os.makedirs(os.path.join(param_dict['pred_path'], 'overLvect'))  

    if os.path.exists(os.path.join(param_dict['pred_path'], 'overLactiveC')) is False:
        os.makedirs(os.path.join(param_dict['pred_path'], 'overLactiveC'))
    
    if os.path.exists(os.path.join(param_dict['pred_path'], 'overLcornerReg')) is False:
        os.makedirs(os.path.join(param_dict['pred_path'], 'overLcornerReg'))
    
    if os.path.exists(os.path.join(param_dict['pred_path'], 'overLRGB')) is False:
        os.makedirs(os.path.join(param_dict['pred_path'], 'overLRGB')) 
    
    if os.path.exists(os.path.join(param_dict['pred_path'], 'overLRGB_occ')) is False:
        os.makedirs(os.path.join(param_dict['pred_path'], 'overLRGB_occ'))
    

    with torch.no_grad():
        print('Parameters: ',count_parameters(model))
        model_weight(model)
    
        batch_num = 0
        n = 0        
        inf_time = []
        time_vect = []
        time_act = []
        time_corner = []
        for i, data in tqdm.tqdm(enumerate(testloader), ascii=True, desc="test step"):
            
            output_list = []

            if param_dict['adversarial']:
                
                images = torch.stack(data["image"], dim=0)
                labels = torch.stack(data["gt"], dim=0)
                img_path = np.stack(data["img_path"])
                gt_path = np.stack(data["gt_path"])
                sf_fimages = torch.stack(data["df_fimage"], dim=0)                
                
                if data["occ"][0] is not None:
                    occ_mask = torch.stack(data["occ"], dim=0)
                    visible_mask = torch.stack(data["visible_mask"], dim=0)
                    
               
                pro_target = data["pro_target"]
                    
                if pro_target[0] is not None:
                    for v in pro_target:                        
                        v["boxes"] = v["boxes"].cuda() 
                       
                    for v in pro_target:                        
                        v["class_labels"] = v["class_labels"].cuda() 

                    
                sf_fimages = sf_fimages.cuda()
                           
            else:    
                images = torch.stack(data["image"], dim=0)
                labels = torch.stack(data["gt"], dim=0) #torch.stack(data["occ"], dim=0)
                img_path = np.stack(data["img_path"])
                gt_path = np.stack(data["gt_path"])
                #occ_mask = torch.stack(data["occ"], dim=0)
                #visible_mask = torch.stack(data["visible_mask"], dim=0)                

            i += images.size()[0]            
            images = images.cuda()
            #labels = labels.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long()
            

            if param_dict['adversarial']:
                # Coarse network
                
                visible = visi_model(images)
                
                # Occlusion network
                occ = occ_model(sf_fimages)
                occ = nn.functional.interpolate(occ.logits, size=param_dict['img_size'], mode="bilinear", align_corners=False) 
                occ = occ.argmax(dim=1)  
                occ = torch.unsqueeze(occ, 1)
                mask = occ.float()   
                
                start_time = time.time() 
                # Completion network
                image_masked = visible * (1.-mask) 
                ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
                
                grid = True
                grid_all = []
                if grid:                                                
                    
                    for i in range(visible.shape[0]):                            
                        gr = draw_grd(visible[i][0], occ[i][0])/255
                        grid_all.append(gr)
                    
                    grid = np.stack(grid_all, 0)
                    grid = torch.tensor(grid).cuda()
                    grid = torch.unsqueeze(grid,1).float()                                           

                    x = torch.cat([image_masked, ones_x, ones_x*mask, grid*mask],dim=1)                                        
                else:                         
                    x = torch.cat([image_masked, ones_x, ones_x*mask],dim=1)  
                
                                                                       
                stg1, x_stage2, offset_flow = model(x, mask)
                
                end_time = time.time()

                inference_time = end_time - start_time
                inf_time.append(inference_time)
                outputs = x_stage2
                                    
                #sym_loss = calculate_sym(outputs) 

            else:
                outputs = model(images)

            inference_PFLNet = False                
            if not inference_PFLNet:                  
                pred = tools.utils.out2pred(outputs, param_dict['num_class'], param_dict['thread'])                                 
            batch_num += images.size()[0]

            
            
            #---------------
            pred_occ = tools.utils.out2pred(occ, param_dict['num_class'], param_dict['thread'])
            pred_visi = tools.utils.out2pred(visible, param_dict['num_class'], param_dict['thread'])
            #---------------

            for kk in range(len(img_path)):
                cur_name = os.path.basename(img_path[kk]) 
                 

                #---------EVAL PFLNet (baseline) --------
                if inference_PFLNet:
                    root = '/home/cero_ma/MCV/baselines/PFLNet/saves/full-modern-occ80-results/fac_sig/output_multi_eval/'
                    pfl_pred_path = os.path.join(root,cur_name.split('.')[0]+'.png')                    
                    pfl_pred = read_image(pfl_pred_path, 'gt') 
                    pred = np.expand_dims(pfl_pred, axis=0)                  

                #---------EVAL PFLNet (baseline) --------  


                pred_sub = pred[kk, :, :]
                #---------------
                pred_sub_occ = pred_occ[kk, :, :]
                pred_sub_visi = pred_visi[kk, :, :]
                #---------------
                                                
                #predict_all_Avcorners[n] = corner_detector_algorithm(param_dict['batch_size'], param_dict['img_size'], pred_sub, kk)
                label_all[n] = read_image(gt_path[kk], 'gt')                                            
                predict_all[n]= pred_sub

                def overlappingGT(prediction, label, path = 'overL', rgb_image = False, imgPath = None, pred= None):                                                
                        
                        # Save binary masks and GT
                        if rgb_image == False:
                            new_pred = np.stack([prediction] * 3, axis=-1)   
                            new_pred = new_pred * 255    

                            #Ground truth
                            contours, _ = cv2.findContours(image=np.uint8(label), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
                            image_new = np.zeros((512,512,3), dtype='uint8')
                            
                            for component in contours:                            
                                x,y,w,h = cv2.boundingRect(component) 
                                box = np.array([[x,y+h],[x,y],[x+w,y],[x+w,y+h]], dtype='float32') 
                                box = np.int0(box)
                                #print(gt_path[kk], new_pred.shape, new_pred.dtype)
                                cv2.drawContours(new_pred, [box], -1, (0,0,255) , 2)
                        
                        # Save RGB image and colored segments
                        else:
                            ori_rgb_img = cv2.imread(imgPath)  
                            new_pred_sub = cv2.resize(pred.astype('uint8'), (ori_rgb_img.shape[1], ori_rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                            labeled_windows, _ = ndi.label(new_pred_sub) 
                            new_pred = label2rgb(labeled_windows, image=ori_rgb_img)*255

                            new_label = cv2.resize(label.astype('uint8'), (ori_rgb_img.shape[1], ori_rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                            contours, _ = cv2.findContours(image=np.uint8(new_label), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
                            
                            for component in contours:                            
                                x,y,w,h = cv2.boundingRect(component) 
                                box = np.array([[x,y+h],[x,y],[x+w,y],[x+w,y+h]], dtype='float32') 
                                box = np.int0(box)                                
                                cv2.drawContours(new_pred, [box], -1, (0,0,255) , 1)
                                #cv2.drawContours(new_pred, contours, -1, (0,0,255) , 2)
                                                    
                        cv2.imwrite(os.path.join(param_dict['pred_path'],path, cur_name.replace('.jpg','.png')), (new_pred).astype(float))                    
                
                # Draw segmentation results and GT                                
                overlappingGT(predict_all[n], label_all[n])

                # Draw GT on RGB images
                #overlappingGT(None,label_all[n], 'overLRGB-GT', rgb_image = True, imgPath=img_path[kk], pred = label_all[n])

                # Draw segments and RGB image                             
                overlappingGT(None,label_all[n], 'overLRGB', rgb_image = True, imgPath=img_path[kk], pred = pred_sub)
                #overlappingGT(None,None, 'overLRGB_occ', rgb_image = True, imgPath=img_path[kk], pred = pred_sub_occ) 

                # Vectorization: Contour-based: bondary detection
                time_vect_s = time.time() 
                vect_predict_all[n] = directVectorization(pred_sub, param_dict['img_size'])
                time_vect_e = time.time() 
                #overlappingGT(vect_predict_all[n], label_all[n], 'overLvect')
                overlappingGT(None,label_all[n], 'overLvect', rgb_image = True, imgPath=img_path[kk], pred = vect_predict_all[n],)
                
                # Vectorization: Contour-based: active contour
                time_act_s = time.time() 
                activeC_predict_all[n] = activeContour(pred_sub)
                time_act_e = time.time() 
                #overlappingGT(activeC_predict_all[n], label_all[n], 'overLactiveC')
                overlappingGT(None,label_all[n], 'overLactiveC', rgb_image = True, imgPath=img_path[kk], pred = activeC_predict_all[n],)

                # Vectorization: Corner regressor
                if param_dict['corner_reg']:
                    time_corner_s = time.time() 
                    cornerReg_predict_all[n] = corner_reg_inference(torch.unsqueeze(torch.tensor(pred_sub.astype('float')),0), images, model_cr, None, None, True, 9)[0][0,:,:,0]
                    time_corner_e = time.time() 
                    #overlappingGT(cornerReg_predict_all[n], label_all[n], 'overLcornerReg')
                    overlappingGT(None,label_all[n], 'overLcornerReg', rgb_image = True, imgPath=img_path[kk], pred = cornerReg_predict_all[n],)
                
                time_vect.append(time_vect_e- time_vect_s)
                time_act.append(time_act_e- time_act_s)
                #time_corner.append(time_corner_e- time_corner_s)
                print('time_vect: ', time_vect_e- time_vect_s)
                print('time_act: ', time_act_e- time_act_s)
                #print('time_corner: ', time_corner_e- time_corner_s)

                
                ###########################################
                
                #Visible prediction: complete - occluder mask                    
                            
                if 'occ_mask' not in locals():
                    print('Compute hidden IoU with predicted occlusion masks...')
                    sim_data = False 
                    occ_mask = np.expand_dims(tools.utils.out2pred(mask, param_dict['num_class'], param_dict['thread']),1)
                    inv_occ = (1-occ_mask[kk][0])
                else:#simulated data
                    sim_data = True 
                    inv_occ = (1-occ_mask[kk][0].numpy())
                
                pred_visible = pred_sub * inv_occ
                predict_all_vis[n] = pred_visible

                if 'visible_mask' in locals():
                    label_all_vis[n] = visible_mask[kk][0].numpy()
                    

                #Hidden prediction: complete - visible
                                
                pred_hidden = np.where(pred_sub - pred_visible > 0, 1, 0)  
                #pred_hidden = np.asarray(pred_sub * (occ_mask[kk][0]))                              
                label_all_hidden[n] = np.asarray(labels[kk][0] * (occ_mask[kk][0])) 
                predict_all_hidden[n] = pred_hidden 
                if not sim_data:
                    del occ_mask
                ###########################################
                                                 
                img = (pred_sub*255).astype(float)
                cv2.imwrite( os.path.join(param_dict['pred_path'], cur_name.replace('.jpg','.png')),img)
                #if mask.max() ==1:
                #    cv2.imwrite(os.path.join(param_dict['pred_path'], 'attn_maps', cur_name.replace('.jpg','.png')), (np.transpose((np.asarray(offset_flow.detach().cpu())*255)[0],(1,2,0))).astype(float))
                        

                n += 1
        
        print('Contour-based: ', np.sum(time_vect)/len(time_vect))
        print('Active contour: ', np.sum(time_act)/len(time_act))
        print('Corner regressor: ', np.sum(time_corner)/len(time_corner))

        print('Average inference time: ', np.sum(inf_time)/len(inf_time))                                            
        print('Segmentation')
        precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
            label_all, predict_all,
            param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
            param_dict['save_dir_model'],
            file_name =str(epoch)+'_accuracy.txt')        
                  

        print('Hidden windows')
        results_hidden = get_acc_v2(
                label_all_hidden, predict_all_hidden,
                param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
                os.path.join(param_dict['save_dir_model']),
                file_name =str(epoch)+'_hidden.txt')

        print('Direct contour results - complete windows')
        precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
            label_all, vect_predict_all,
            param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
            param_dict['save_dir_model'],
            file_name =str(epoch)+'_accuracy_vect.txt') 
        
        print('Active contour results - complete windows')
        precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
            label_all, activeC_predict_all,
            param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
            param_dict['save_dir_model'],
            file_name =str(epoch)+'_accuracy_activeC.txt')
        
        print('Corner regressor results - complete windows')
        precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
            label_all, cornerReg_predict_all,
            param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
            param_dict['save_dir_model'],
            file_name =str(epoch)+'_accuracy_cornerReg.txt')  
        

def load_model(model_path):

    input_bands = param_dict['input_bands']

    model = get_net(param_dict['model_name'], input_bands, param_dict['num_class'], param_dict['img_size'], param_dict['pretrained_model'])
    model = torch.nn.DataParallel(model, device_ids=[0])
    
    state_dict = torch.load(model_path)['net']
    new_state_dict = OrderedDict()
    model.load_state_dict(state_dict)
    
    epoch = torch.load(model_path)['epoch']    
    model.cuda()
    model.eval()
    print('epoch: ', epoch)
    return model, epoch


if __name__ == '__main__':
    if len(sys.argv) == 1:
        yaml_file = 'config.yaml'
    else:
        yaml_file = sys.argv[1]
    param_dict = parse_yaml(yaml_file)

    for kv in param_dict.items():
        print(kv)
        
    os.environ["CUDA_VISIBLE_DEVICES"] = param_dict['gpu_id']
    gpu_list = [i for i in range(len(param_dict['gpu_id'].split(',')))]
    gx = torch.cuda.device_count()
    print('useful gpu count is {}'.format(gx))

    model_path =  os.path.join(param_dict['model_dir'], param_dict['main_model_inference'])
    
    composed_transforms_val = standard_transforms.Compose([
        tr.FixedResize(param_dict['img_size']),
        tr.Normalize(mean=param_dict['mean'], std=param_dict['std']),
        tr.ToTensor(do_not = {'img_sf', 'occ_sf'})]) 

    simulated_dataset = False
    if 'occ' in param_dict['dataset']:
        simulated_dataset = True

    if param_dict['adversarial']:        
        feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
        processor = DetrImageProcessor(do_resize =False, do_rescale = True)                
        
        road_test = AmodalSegmentation(txt_path=param_dict['test_list'], transform=composed_transforms_val, occSegFormer = True, feature_extractor= feature_extractor,  processor=processor, jsonAnnotation = param_dict['json_list'], simulated_dataset=simulated_dataset) 
    else:
        road_test = AmodalSegmentation(txt_path=param_dict['test_list'], transform=composed_transforms_val, jsonAnnotation = param_dict['json_list'], simulated_dataset=simulated_dataset)
    

    testloader = DataLoader(road_test, batch_size=param_dict['batch_size'], shuffle=False, num_workers=param_dict['num_workers'], drop_last=False, collate_fn=collate_fn)  
    
    if param_dict['adversarial'] or param_dict['corner_reg']:
        test(testloader, None, None)        
    else:
        model, epoch = load_model(model_path)
        test(testloader, model, epoch)