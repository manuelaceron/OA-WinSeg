from __future__ import division
from tools.utils import read_image
import sys
import numpy as np
import os, time, pdb, math
from tools.metrics import get_acc_v2
from timm.optim import create_optimizer_v2
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import tqdm
import tools.transform as tr
from tools.dataloader import AmodalSegmentation
import tools
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from networks.get_model import get_net
from tools.losses import get_loss 
from tools.parse_config_yaml import parse_yaml
import cv2
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, PretrainedConfig, DetrImageProcessor
import albumentations as aug
from networks.CompletionNet.networks_BK0 import Generator as CN_Generator
import networks.CompletionNet.misc as misc
from networks.discriminator import NLayerDiscriminator
from tools.newLosses import AdversarialLoss
from transformers.image_transforms import center_to_corners_format
from networks.CompletionNet.utils import draw_grd, calculate_gloabl_sym, calculate_sym, regularizationLoss, compute_corner_reg, corner_reg_inference
import timm

np.seterr(divide='ignore', invalid='ignore')

   
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_iou = -1000

    def early_stop(self, validation_iou):
        if validation_iou > self.min_validation_iou:
            self.min_validation_iou = validation_iou
            self.counter = 0
        elif validation_iou < (self.min_validation_iou + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def additional_mask():
    # Creation of an additional occlusion mask combining regular and irregular polygons at random locations to introduce diversity

    img_size= param_dict['img_size']
    bbox = misc.random_bbox(img_size,img_size)
    regular_mask = misc.bbox2mask(img_size,img_size, bbox).cuda()
    irregular_mask = misc.brush_stroke_mask(img_size,img_size).cuda()
    mask = torch.logical_or(irregular_mask, regular_mask).to(torch.float32)
    return mask

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collate_fn(batch):
    # processes the batch to return from dataloader: takes batch of data items and returns a new dictionary n_batch
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
    
    #n_batch["occ"] = inputs[3]
    #n_batch["occ_path"] = inputs[10]
        
    if param_dict['adversarial']:
        n_batch["occ"] = inputs[3]
        n_batch["pro_target"] = inputs[7]
        n_batch['df_fimage'] = inputs[12]
        n_batch['df_fooc'] = inputs[13]
    
    return n_batch


def main(frame_work):
    
    # Early stop strategy
    early_stopper = EarlyStopper(patience=param_dict['stop_pat'], min_delta=param_dict['stop_delta'])
    
    # Data augmentation for training and validation
    composed_transforms_train = standard_transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomVerticalFlip(),        
        tr.FixedResize(param_dict['img_size']),
        tr.Normalize(mean=param_dict['mean'], std=param_dict['std']),
        tr.ToTensor(do_not = {'img_sf', 'occ_sf'})])
    composed_transforms_val = standard_transforms.Compose([
        tr.FixedResize(param_dict['img_size']),
        tr.Normalize(mean=param_dict['mean'], std=param_dict['std']),
        tr.ToTensor(do_not = {'img_sf', 'occ_sf'})]) 

    
    # Flags to run OA-WinSeg model and simulated dataset 
    occSegFormer = False
    feature_extractor = None
    processor = None
    simulated_dataset = False
    
    if param_dict['adversarial']:
        occSegFormer = True
        feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
        processor = DetrImageProcessor(do_resize =False, do_rescale = True)
    
    if 'occ' in param_dict['dataset']:
        simulated_dataset = False 
    
    # Create dataset
    train_dataset = AmodalSegmentation(txt_path=param_dict['train_list'], transform=composed_transforms_train, occSegFormer = occSegFormer, feature_extractor= feature_extractor, processor=processor, jsonAnnotation = param_dict['json_list'], simulated_dataset= simulated_dataset )         
    val_dataset = AmodalSegmentation(txt_path=param_dict['val_list'], transform=composed_transforms_val, occSegFormer = occSegFormer, feature_extractor= feature_extractor,  processor=processor, jsonAnnotation = param_dict['json_list'], simulated_dataset= simulated_dataset)  
    
    # Create dataloader     
    trainloader = DataLoader(train_dataset, batch_size=param_dict['batch_size'], shuffle=True,num_workers=param_dict['num_workers'], drop_last=False, collate_fn=collate_fn)
    valloader = DataLoader(val_dataset, batch_size=param_dict['batch_size'], shuffle=False, num_workers=param_dict['num_workers'], drop_last=False, collate_fn=collate_fn)  
    
    # Resume training in checkpoint
    def resume_training(model, resume_ckpt, optimizer, lr_schedule):        
        checkpoint = torch.load(resume_ckpt)  
        #model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(checkpoint['net']) 
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        start_epoch = checkpoint['epoch'] 
        lr_schedule.load_state_dict(checkpoint['lr_schedule']) 
        print('load the model %s' % param_dict['resume_ckpt'])

        return model, optimizer, lr_schedule, start_epoch
    
    # Default initial epoch
    start_epoch = 0
    
    # Load model according to params
    
    # Vectorization model: Corner regressor
    if param_dict['corner_reg']:
        frame_work = timm.create_model('seresnet50', pretrained=True, in_chans=4)                                
        frame_work.fc = nn.Linear(frame_work.fc.in_features, 4)
    
    # Simple model without adversarial training    
    if not param_dict['adversarial']:
        if len(gpu_list) > 1:
            print('gpu>1')  
            model = torch.nn.DataParallel(frame_work, device_ids=gpu_list)
        else:
            model = frame_work
        model.cuda()

        optimizer = create_optimizer_v2(model, 'adam', lr=param_dict['base_lr'])        
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

        if param_dict['resume_ckpt']:
            model, optimizer, lr_schedule, epoch = resume_training(model, param_dict['resume_ckpt'], optimizer, lr_schedule)
    
    # Finetune DeepLabV+ model
    if param_dict['fine-tune-DL']:
        print('Fine tuning DL...')
        num_classes = param_dict['num_class']
        checkpoint = torch.load(param_dict['bk_pretrained'], map_location=torch.device('cpu'))
        checkpoint['model_state']['classifier.classifier.3.bias'] = checkpoint['model_state']['classifier.classifier.3.bias'][:num_classes]
        checkpoint['model_state']['classifier.classifier.3.weight'] = checkpoint['model_state']['classifier.classifier.3.weight'][:num_classes]
        model.load_state_dict(checkpoint["model_state"],False)
        print("Model restored from %s" % param_dict['bk_pretrained'])


    # Occlusion classification model
    if param_dict['use-occ-model']:
        
        if 'ecp' in param_dict['dataset'] or 'artdeco' in param_dict['dataset']:
            checkpoint_path = param_dict['ecp_occ_model']
        else: 
            checkpoint_path = param_dict['full_occ60_occ_model']

        state_dict = torch.load(checkpoint_path)['net']
   
        pre_model = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024" # "nvidia/mit-b2"

        configuration_b2 = PretrainedConfig( 
            architectures = ["SegformerForSemanticSegmentation"],
            attention_probs_dropout_prob = 0.0,
            classifier_dropout_prob = 0.1,
            decoder_hidden_size = 768,  
            depths = [3, 4, 6, 3], 
            downsampling_rates = [1, 4, 8, 16],
            drop_path_rate = 0.1,
            hidden_act = 'gelu',
            hidden_dropout_prob = 0.0,
            hidden_sizes = [64, 128, 320, 512], 
            id2label = { "0": "background", "1": "occlusion"},
            image_size = 224, 
            initializer_range = 0.02,
            label2id = { "background": 0, "occlusion": 1 },
            layer_norm_eps = 1e-06,
            mlp_ratios = [4, 4, 4, 4], 
            model_type = "segformer",
            num_attention_heads = [1, 2, 5, 8], 
            num_channels = 3, 
            num_encoder_blocks = 4, 
            patch_sizes = [7, 3, 3, 3], 
            reshape_last_stage = True,
            semantic_loss_ignore_index = 255,
            sr_ratios = [8, 4, 2, 1], 
            strides = [4, 2, 2, 2], 
            ignore_mismatched_sizes=True,
            torch_dtype = "float32",
            transformers_version = "4.18.0"
            ) 
        
        occ_model = SegformerForSemanticSegmentation.from_pretrained(pretrained_model_name_or_path = pre_model, ignore_mismatched_sizes=True, 
                                                            config = configuration_b2)

        occ_model = torch.nn.DataParallel(occ_model, device_ids=[0])
        occ_model.load_state_dict(state_dict)
        print('Occlusion model epoch: ', torch.load(checkpoint_path)['epoch'])

        occ_model.eval()

        for param in occ_model.parameters():
            param.requires_grad = False
        occ_model.cuda()
        
    # Coarse segmentation model
    if param_dict['use-coarse-model']:
        
        if 'ecp' in param_dict['dataset']:
            checkpoint_path = param_dict['ecp_coarse_model']               
        elif 'artdeco_ori' in param_dict['dataset']:
            checkpoint_path = param_dict['artdeco_ori_coarse_model']
        elif 'artdeco_ref' in param_dict['dataset']:
            checkpoint_path = param_dict['artdeco_ref_coarse_model']
        elif 'full-occ60' in param_dict['dataset']:            
            checkpoint_path = param_dict['full_occ60_coarse_model']
        
        elif 'full-100' in param_dict['dataset']:            
            checkpoint_path = param_dict['full_occ100_coarse_model']
        elif 'full-occ80' in param_dict['dataset']:            
            checkpoint_path = param_dict['full_occ80_coarse_model'] 
        elif 'full_modern_occ80' in param_dict['dataset']:            
            checkpoint_path = param_dict['full_modern_occ80_coarse_model'] 
        elif 'modern' in param_dict['dataset']: 
            checkpoint_path = param_dict['modern_coarse_model']

            print('Using coarse complete model')
        
        state_dict = torch.load(checkpoint_path)['net']

        visi_model = get_net('Res_UNet_101', 3, 1, 512, None)        
        visi_model = torch.nn.DataParallel(visi_model, device_ids=[0])
        visi_model.load_state_dict(state_dict)
        print('Coarse model epoch: ', torch.load(checkpoint_path)['epoch'])
        print(checkpoint_path)

        visi_model.eval()

        for param in visi_model.parameters():
            param.requires_grad = False
        
        visi_model.cuda()
    
    # Completion model
    if param_dict['adversarial']:
                                    
        im_channel = 1
        im_channel_mid = 1
        im_channel_out = 1 

        cnum= 48
        G = CN_Generator(cnum_in=im_channel+3, cnum_mid = im_channel_mid, cnum_out=im_channel_out, cnum=cnum, return_flow=False, k= param_dict['1D_kernel']) 
        D = NLayerDiscriminator(input_nc=im_channel+1)        
        
        optimizerG = torch.optim.Adam(G.parameters(), lr=param_dict['base_lr'], betas=[0.5, 0.99]) 
        optimizerD = torch.optim.Adam(D.parameters(), lr=param_dict['base_lr'], betas=[0.5, 0.99])
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.8)
        lr_scheduleD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.8)
        

        if len(gpu_list) > 1:
            print('gpu>1')  
            G = torch.nn.DataParallel(G, device_ids=gpu_list) 
            D = torch.nn.DataParallel(D, device_ids=gpu_list)
        G = G.cuda()
        D = D.cuda()
        
        
        if param_dict['resume_ckpt']:
            G, optimizerG, lr_schedule, epoch = resume_training(G, param_dict['resume_ckpt'], optimizerG, lr_schedule)
            D, optimizerD, lr_scheduleD, epoch = resume_training(D, param_dict['resume_ckpt_Discr'], optimizerD, lr_scheduleD)
            print('Completion model epoch ', epoch)            
          
        

    """ Weights for loss function
    ecp-occ60
        occlusions: [0.57487292, 3.83899089]
        visible parts: [0.60757092, 2.8240482 ]
        hidden windoww: [ 0.5160267  16.09896997]
        complete windows: [0.63147511 2.40150057]
    modern-occ100:
        complete windows: [0.7031397  1.73068019]
    full-occ6
        complete windows: [0.60048239 2.98799814] """
        
    # Set loss function according to model to train
    
    if param_dict['corner_reg']:
        criterion = nn.SmoothL1Loss(beta=1).cuda()
    else: 
        loss_weight = [float(i) for i in param_dict['loss_weight'].split(',')]
        criterion = get_loss(param_dict['loss_type'], torch.tensor(loss_weight))    

    if param_dict['adversarial']:
        # Adversarial loss
        adv_loss = AdversarialLoss('nsgan').cuda()
        # L1 loss
        L1_loss = nn.L1Loss().cuda()      
        
    writer = SummaryWriter(os.path.join(param_dict['save_dir_model'], 'runs'))
    best_val_acc = 0.0
        
    log_name= 'log.txt'
    if param_dict['resume_ckpt']:
        log_name = 'resuming_'+log_name

    # Start training 
    
    with open(os.path.join(param_dict['save_dir_model'], log_name), 'w') as ff:
        
        start_time = time.time()        
        for epoch in range(start_epoch, param_dict['epoches']):
            
            # Set models to training mode
            if param_dict['adversarial']:
                G.train()
                D.train()
                losses = {}                                                
            else:
                model.train()            
            
            # Init loss
            running_loss = 0.0
            running_segm_loss = 0.0
            batch_num = 0

            epoch_start_time = time.time()
            
            for i, data in tqdm.tqdm(enumerate(trainloader)): 
                
                output_list = []

                # Load all data                             
                images = torch.stack(data["image"], dim=0)
                labels = torch.stack(data["gt"], dim=0)
                
                # Only for bounding box detection
                if "pro_target" in data.keys():
                    pro_target = data["pro_target"]
                    
                    if pro_target[0] is not None:
                        
                        for v in pro_target:                        
                            v["boxes"] = v["boxes"].cuda() 
                        
                        for v in pro_target:                        
                            v["class_labels"] = v["class_labels"].cuda()  
 
                path = data['img_path']                
                i += images.size()[0] 
                                
                labels = labels.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 
                images = images.cuda()
                labels = labels.cuda()
                
                # Reset gradients
                if param_dict['adversarial']:
                    optimizerG.zero_grad()
                    optimizerD.zero_grad()                                                                     
                else:
                    optimizer.zero_grad()
                                                                                                 
                if param_dict['adversarial']:
                    
                    # Coarse mask to complete
                    out_visible = visi_model(images)
                    visible = out_visible

                    # Occlusion mask 
                    sf_fimages = torch.stack(data["df_fimage"], dim=0)                    
                    sf_fimages = sf_fimages.cuda()
                    
                    occ = occ_model(sf_fimages)
                    occ = nn.functional.interpolate(occ.logits, size=param_dict['img_size'], mode="bilinear", align_corners=False) 
                    occ = occ.argmax(dim=1)  
                    occ = torch.unsqueeze(occ, 1)
                    occ_mask = occ.float()
                    
                    # Additional occlusion masks for augmentation
                    mask_2 = additional_mask()
                    mask = torch.logical_or(occ_mask, mask_2).to(torch.float32)
                                        
                    # Completion network                                       
                    batch_incomplete = visible*(1.-mask)
                    ones_x = torch.ones_like(batch_incomplete)[:, 0:1].cuda() 

                    # Recangular boundary guidance
                    grid = True
                    grid_all = []
                    if grid:
                        for i in range(visible.shape[0]):
                            gr = draw_grd(visible[i][0], occ[i][0])/255
                            grid_all.append(gr)
                        
                        grid = np.stack(grid_all, 0)
                        grid = torch.tensor(grid).cuda()
                        grid = torch.unsqueeze(grid,1).float()
                        
                        x = torch.cat([batch_incomplete, ones_x, ones_x*mask, grid*mask], axis=1)                                            
                    else:                                                               
                        x = torch.cat([batch_incomplete, ones_x, ones_x*mask], axis=1)                                                                                                                                                                                                                                 

                    labels = torch.unsqueeze(labels,1).float()
                                        
                    batch_real = batch_incomplete 

                    dis_real = D(torch.cat([batch_real, labels],1)) #visible
                    dis_real_loss = adv_loss(dis_real, True, True)  
                    
                    # Generator coarse and fine stages                    
                    img_stg1, img = G(x, mask)
                    img_2_loss = img
                    
                    # Optional addition loss functions 
                    # Symmetric loss
                    #sym_loss, _ = calculate_sym(img_2_loss, symmetric = True, rectangular = False)
                    # Global symmetric loss
                    #global_sym_loss = calculate_gloabl_sym(img_2_loss) 
                    # Regularization loss
                    #regLoss = regularizationLoss(img_2_loss, pro_target, 0.65, scale_bbox = True)
                    #test_loss_dict = criterionDetr(img, pro_target)
                                                                                                    
                
                    # Adversarial training
                                        
                    do_stg1 = True
                    # Run coarse stage
                    if do_stg1:
                        # Stg1:
                        # -Dis-
                        dis_input_fake_stg1 = img_stg1.detach()
                        dis_fake_stg1 = D(torch.cat([batch_real, dis_input_fake_stg1],1))
                        dis_fake_loss_stg1 = adv_loss(dis_fake_stg1, False, True)

                        # -Gen-
                        gen_input_fake_stg1 = img_stg1
                        gen_fake_stg1 = D(torch.cat([batch_real, gen_input_fake_stg1],1))

                        # -Loss-
                        adv_loss_stg1 = adv_loss(gen_fake_stg1, True, False)
                        bce_loss_stg1 = criterion(gen_input_fake_stg1, torch.squeeze(labels,1))
                    else:
                        dis_fake_loss_stg1 = 0
                        adv_loss_stg1 = 0
                        bce_loss_stg1 = 0


                    # Stg2:
                    # -Dis-
                    dis_input_fake = img.detach()
                    dis_fake = D(torch.cat([batch_real, dis_input_fake],1))
                    dis_fake_loss = adv_loss(dis_fake, False, True)
                    
                    dis_loss = (dis_real_loss + dis_fake_loss + dis_fake_loss_stg1).mean()

                    # -Gen-
                    gen_loss = 0     
                    gen_input_fake = img                            
                    gen_fake = D(torch.cat([batch_real, gen_input_fake],1))
                    
                    # -Loss-
                    gen_gan_loss = adv_loss(gen_fake, True, False) + criterion(gen_input_fake, torch.squeeze(labels,1)) + \
                        adv_loss_stg1 +  bce_loss_stg1  + L1_loss(gen_input_fake, labels)    #+ sym_loss                                           
                    
                                                                                            
                    running_segm_loss += gen_gan_loss.mean()
                    gen_loss += gen_gan_loss.mean()

                    
                    # Backpropagation
                    gen_loss.backward()                              
                    optimizerG.step()
                    running_loss += gen_loss                     
                    dis_loss.backward()
                    optimizerD.step()                                                                                                                                                                        
                    
                else:   
                    # Compute loss for Vectorization: corner regressor
                    if param_dict['corner_reg']:
                        losses = compute_corner_reg(labels, images, model, criterion, True, None, 9)
                    
                    # Compute loss
                    else: 
                        outputs = model(images)
                        losses = criterion(outputs, labels)
                
                if not param_dict['adversarial']:
                    losses.backward()  
                    optimizer.step()
                    running_loss += losses
                batch_num += images.size()[0]
            
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
                        
            print('epoch is {}, train loss is {}, Per epoch time {}'.format(epoch, running_loss.item() / batch_num, per_epoch_ptime ))
                
            if param_dict['adversarial']:
                cur_lr = optimizerG.param_groups[0]['lr']
                                
            else:
                cur_lr = optimizer.param_groups[0]['lr']
            
            writer.add_scalar('learning_rate', cur_lr, epoch)
            writer.add_scalar('train_loss', running_loss / batch_num, epoch)
            
            
            lr_schedule.step()   
            
            # Validation step
            if epoch % param_dict['save_iter'] == 0:
                                                
                if param_dict['adversarial']: 
                    val_miou, val_acc, val_f1, val_loss = eval(valloader=valloader, model=G, model2 = occ_model, model3 = visi_model, criterion = criterion, criterion2=adv_loss, criterion3= L1_loss, epoch= epoch)
                
                else:                    
                    val_miou, val_acc, val_f1, val_loss = eval(valloader=valloader, model=model, criterion = criterion, epoch= epoch, model2 = None, model3 = None, criterion2 = None, criterion3 = None)

                writer.add_scalar('val_miou', val_miou, epoch)
                writer.add_scalar('val_acc', val_acc, epoch)
                writer.add_scalar('val_f1', val_f1, epoch)
                writer.add_scalar('val_loss', val_loss, epoch)
                
                # Log training info
                cur_log = 'epoch:{}, learning_rate:{}, train_loss:{}, val_loss:{}, val_f1:{}, val_acc:{}, val_miou:{}\n'.format(
                    str(epoch), str(cur_lr), str(running_loss.item() / batch_num), str(val_loss), str(val_f1),
                    str(val_acc),
                    str(val_miou)
                )
                
                print('Final Seg: ', cur_log)
                ff.writelines(str(cur_log))
                
                # Save model
                if epoch >= 10:#val_miou > best_val_acc:
                    if param_dict['adversarial']:
                        checkpoint = {
                                "net": G.state_dict(),
                                'optimizer': optimizerG.state_dict(),
                                "epoch": epoch,
                                'lr_schedule': lr_schedule.state_dict()}
                        checkpointD = {
                                "net": D.state_dict(),
                                'optimizer': optimizerD.state_dict(),
                                "epoch": epoch,
                                'lr_schedule': lr_scheduleD.state_dict()}                                                                   
                    else: 
                        checkpoint = {
                                "net": model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                "epoch": epoch,
                                'lr_schedule': lr_schedule.state_dict()}
                    
                    if val_miou > best_val_acc:
                        name= str(epoch)+'valiou_best.pth'
                    else:
                        name = str(epoch)+'model.pth'
                    
                    model_dir= param_dict['model_dir']
                    
                    if param_dict['resume_ckpt']:
                        name = 'res_'+ name

                    torch.save(checkpoint, os.path.join(model_dir, name))
                       
                    if param_dict['adversarial']:
                        torch.save(checkpointD, os.path.join(model_dir, 'D_'+name))

                    best_val_acc = val_miou
                    if early_stopper.early_stop(best_val_acc):
                        print('Early stop break')             
                        break

        end_time = time.time()
        total_ptime = end_time - start_time
        print('Total time: ',total_ptime) 


def eval(valloader, model, model2, model3, criterion, criterion2, criterion3, epoch): 

    val_num = valloader.dataset.num_sample
    
    label_all = np.zeros((val_num,) + (3, param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all = np.zeros((val_num,) + (3, param_dict['img_size'], param_dict['img_size']), np.uint8)
        
    model.eval()
    if param_dict['val_visual']:
        
        name_folder = 'val_visual'
        if param_dict['resume_ckpt']:
            name_folder = 'resuming_'+name_folder

        if os.path.exists(os.path.join(param_dict['save_dir_model'], name_folder)) is False:
            os.mkdir(os.path.join(param_dict['save_dir_model'], name_folder))
        if os.path.exists(os.path.join(param_dict['save_dir_model'], name_folder, str(epoch))) is False:
            os.mkdir(os.path.join(param_dict['save_dir_model'], name_folder, str(epoch)))
            os.mkdir(os.path.join(param_dict['save_dir_model'], name_folder, str(epoch), 'slice'))                     
    
    
    L1_loss = nn.L1Loss().cuda()        
    with torch.no_grad():
        
        batch_num = 0
        val_loss = 0.0
        val_loss_mask = 0.0
        val_loss_vis = 0.0
        n = 0
        for i, data in tqdm.tqdm(enumerate(valloader), ascii=True, desc="validate step"):
            output_list = []
                                       
            images = torch.stack(data["image"], dim=0)
            labels = torch.stack(data["gt"], dim=0)
            
            img_path = np.stack(data["img_path"])
            gt_path = np.stack(data["gt_path"])
            
            # Only for bounding box detection
            if "pro_target" in data.keys():                        
                pro_target = data["pro_target"]
                
                if pro_target[0] is not None:
                    for v in pro_target:                        
                        v["boxes"] = v["boxes"].cuda() 
                    
                    for v in pro_target:                        
                        v["class_labels"] = v["class_labels"].cuda()

            
            
            i += images.size()[0]
                        
            labels = labels.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long()
            images = images.cuda()
            labels = labels.cuda()
            
            if param_dict['adversarial']:                
                
                # Coarse network
                visible = model3(images)
                
                # Occlusion network
                sf_fimages = torch.stack(data["df_fimage"], dim=0)
                    
                sf_fimages = sf_fimages.cuda()
                    
                occ = model2(sf_fimages)#
                occ = nn.functional.interpolate(occ.logits, size=param_dict['img_size'], mode="bilinear", align_corners=False) 
                occ = occ.argmax(dim=1)  
                occ = torch.unsqueeze(occ, 1)
                
                # Completion network
                batch_real = visible                
                mask = occ.float()   
                
                image_masked = batch_real * (1.-mask) 
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
                
                x_stage1, x_stage2 = model(x, mask)                                                            
                outputs = x_stage2
                                    
                mask_losses = criterion(outputs,labels) + criterion(x_stage1,labels) + criterion3(outputs,torch.unsqueeze(labels,1).float())
                vallosses = mask_losses.mean()                                                                                               
                
            else:     
                # Compute loss for Vectorization: corner regressor
                if param_dict['corner_reg']:
                    
                    outputs, vallosses = corner_reg_inference(labels, images, model, criterion, None, True, 9)
                    pred = outputs[:,:,:,0]
                
                # Compute loss
                else:
                    outputs = model(images)                
                    vallosses = criterion(outputs, labels)                                  
            
            if not param_dict['corner_reg']:                
                pred = tools.utils.out2pred(outputs, param_dict['num_class'], param_dict['thread'])    

            val_loss += vallosses.item()            
            
            batch_num += images.size()[0]
            
            if param_dict['val_visual']:
                for kk in range(len(img_path)):
                    cur_name = os.path.basename(img_path[kk])
                    
                    pred_sub = pred[kk, :, :]                      
                    label_all[n] = read_image(gt_path[kk], 'gt') 
                    predict_all[n] = pred_sub

                    # Uncomment to save validation images                                        
                    #cv2.imwrite( os.path.join(param_dict['save_dir_model'], name_folder, str(epoch), 'slice', cur_name.split('.')[0]+'.png'), (pred_sub*255).astype(float))
                    n += 1
        
        print('W. Segmentation')            
        
        # Log accuracy 
        precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
                label_all, predict_all,
                param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
                os.path.join(param_dict['save_dir_model'], name_folder, str(epoch)))
        val_loss = val_loss / batch_num

    return IoU[1], OA, f1ccore[1], val_loss
    


if __name__ == '__main__':
    
    # Read config file
    if len(sys.argv) == 1:
        yaml_file = 'config.yaml'
    else:
        yaml_file = sys.argv[1]
    param_dict = parse_yaml(yaml_file)

    for kv in param_dict.items():
        print(kv)
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = param_dict['gpu_id']
    gpu_list = [i for i in range(len(param_dict['gpu_id'].split(',')))]
    gx = torch.cuda.device_count()
    print('useful gpu count is {}'.format(gx))

    input_bands = param_dict['input_bands']
    
    # Create network structure
    if not param_dict['adversarial'] and not param_dict['corner_reg']:        
        frame_work = get_net(param_dict['model_name'], input_bands, param_dict['num_class'],
                            param_dict['img_size'], param_dict['pretrained_model'])      
            
    if os.path.exists(param_dict['model_dir']) is False:
        print('Create dir')
        os.mkdir(param_dict['model_dir'])
    position_channel = torch.tensor(np.mgrid[0:param_dict['img_size'], 0:param_dict['img_size']]).cuda()
    
    if param_dict['adversarial'] or param_dict['corner_reg']:
        main(None)        
    else:
        main(frame_work)