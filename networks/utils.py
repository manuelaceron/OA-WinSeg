import cv2
import numpy as np
import torch
import tools
from scipy.ndimage import label
import math
from transformers.image_transforms import center_to_corners_format
import matplotlib.pyplot as plt
import pdb
import torch.nn as nn
from skimage.color import label2rgb

def plot_multi(prediction, path):


    color_map = {
    0: (255, 255, 255),   
    1: (255, 0, 0),   
    2: (0, 255, 0),    
    3: (0, 0, 255),
    4: (255, 128, 0),
    5: (0, 128, 255),
    6: (255, 153, 255)
    }

    # Create an RGB image from the prediction
    rgb_image = np.zeros((512, 512, 3), dtype=np.uint8)

    for i in range(512):
        for j in range(512):
            class_id = prediction[i, j]
            rgb_image[i, j] = color_map[class_id]

    # Save the image
    cv2.imwrite(path, rgb_image)

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for data in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data['image'], dim=[0,2,3])
        channels_squared_sum += torch.mean(data['image']**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

def reduce_lines(binary_image):
    gray_image = binary_image.astype('uint8')
    
    kernel_d = np.ones((3,3), np.uint8)
    kernel_e = np.ones((3,3), np.uint8)

    dilated_image = cv2.dilate(gray_image, kernel_d, iterations=1)
    merged_image = cv2.erode(dilated_image, kernel_e, iterations=1)

    return merged_image

def calculate_line(x1, x2, y1, y2, up, low, raw):
    if x2 != x1:
        slope = (y2 - y1) / (x2 - x1)
        if not raw:
            if (slope > up) or (slope < low):
                return None          
        intercept = y1 - (slope * x1)                                                
        x_start = 0                            
        y_start = int((slope * x_start) + intercept)
        x_end = 512 - 1
        y_end = int((slope * x_end) + intercept)
    else:
        x_common = x1
        y_start = 0
        y_end = 512 - 1
        x_start = x_end = x_common
    
    return x_start, x_end, y_start, y_end

# function to find minimum area of Rectangle
def draw_grd(visible, mask):
    
    from_visible = False
    any_contour  = True #False: filter contour size
    any_width = True #False: merge close lines
    any_slope = True #False: filter lines with weird angles     
    
    #visible = visible[0][0]
    #mask = mask[0][0]

    vis_out = np.asarray(visible.detach().cpu())
    mask_out = np.asarray(mask.detach().cpu())

    (thresh, vis_out_bw) = cv2.threshold(vis_out.astype('uint8'), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    (thresh_mask, mask_out_bw) = cv2.threshold(mask_out.astype('uint8'), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    vis_out_bw = 255 - vis_out_bw
        
    if from_visible:
        incomplete = (vis_out_bw*(255-mask_out_bw))*255    
    else:
        incomplete = vis_out_bw
                    
    contours, hierarchy = cv2.findContours(image=incomplete, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)#cv2.CHAIN_APPROX_NONE) 
        
    
    #-------------Cleaning restrictions-----------------------------------------------------
    up_1 = up_2 = low_1 = low_2 = 0
    if not any_contour: #False: #
        area = []
        angle1 = []
        angle2 = []
        
        for cnt in contours:
            area.append(cv2.contourArea(cnt))

            rect = cv2.minAreaRect(cnt)                            
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            seq_box_1 = [(box[0], box[1]), (box[2], box[3])]
            seq_box_2 = [(box[1], box[2]), (box[3], box[0])]
            
            # Angle 1
            for coor in seq_box_1:
                x1, y1 = coor[0]
                x2, y2 = coor[1]

                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    angle1.append(slope)
            
            # Angle 2
            for coor in seq_box_2:
                x1, y1 = coor[0]
                x2, y2 = coor[1]

                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    angle2.append(slope)
            
        std_a1 = np.std(angle1, ddof=1)
        std_a2 = np.std(angle2, ddof=1)
        mean_a1= sum(angle1)/len(angle1)
        mean_a2= sum(angle2)/len(angle2)
        
        tol = 0.5
        up_1 = mean_a1 + (tol*std_a1)
        low_1 = mean_a1 - (tol*std_a1)
        up_2 = mean_a2 + (tol*std_a2)
        low_2 = mean_a2 - (tol*std_a2)
            
        area_avg = sum(area)/len(area)
    #-------------------------------------------------------------------------

    contours_to_keep = np.zeros_like(incomplete, dtype=np.uint8)              
    
    image_copy = np.zeros((512,512,3), dtype='uint8')
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
                 
    grid_color = 1
    H,W = visible.shape
    grid1 = np.zeros((H,W))
    grid2 = np.zeros((H,W))

    for component in contours:
        if not any_contour:
            # Ignore small contours, probably noise
            if area_avg > 1.8 * cv2.contourArea(component): #1.8
                continue

        # Generate contours 
        image_new = np.zeros((512,512), dtype='uint8')
                       
        #Not rotated rectangle 
        x,y,w,h = cv2.boundingRect(component) 
        box = np.array([[x,y+h],[x,y],[x+w,y],[x+w,y+h]], dtype='float32')
        
        # Rotated rectangle                                        
        #rect = cv2.minAreaRect(component)
        #box = cv2.boxPoints(rect)

        box = np.int0(box)
        cv2.drawContours(image_new, [box], -1, 255, 1)         
        
        if from_visible:
        # Disregard segments ovelapping with occlusions
            intersection_mask = np.logical_and(mask_out_bw, image_new)
            if np.any(intersection_mask):            
                continue
        
        cv2.drawContours(contours_to_keep, [box], -1, 255, 1)                           
                        
        
        seq_box_1 = [(box[0], box[1]), (box[2], box[3])] #Vertical boundaries
        seq_box_2 = [(box[1], box[2]), (box[3], box[0])] #Horizontal boundaries

        # Angle 1
        for coor in seq_box_1:
            x1, y1 = coor[0]
            x2, y2 = coor[1]  
            out = calculate_line(x1, x2, y1, y2, up_1, low_1, any_slope)
            if out == None:
                continue
            else: 
                x_start, x_end, y_start, y_end = out                              
            cv2.line(grid1, (x_start, y_start), (x_end, y_end), 255, 1)
                
        for coor in seq_box_2:
            x1, y1 = coor[0]
            x2, y2 = coor[1]
            out =  calculate_line(x1, x2, y1, y2, up_2, low_2, any_slope)  
            if out == None:
                continue
            else: 
                x_start, x_end, y_start, y_end = out   
            cv2.line(grid2, (x_start, y_start), (x_end, y_end), 255, 1)
    
    if not any_width:
        new_grid1 = reduce_lines(grid1)
        new_grid2 = reduce_lines(grid2)
        out = new_grid1+new_grid2
    else:
        out = grid1+grid2

    """ #cv2.imshow('image_copy', image_copy)
    cv2.imshow('mask_out_bw', mask_out_bw)
    cv2.imshow('incomplete', incomplete)
    cv2.imshow('rect2', contours_to_keep)
    cv2.imshow('grid', contours_to_keep+grid1+grid2)    
    if not any_width:
        cv2.imshow('new_grid', new_grid1+new_grid2+contours_to_keep)

    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    pdb.set_trace() """ 
      
    return out

def calculate_sym(outputs, symmetric = True, rectangular = False):
    #pred_input = tools.utils.out2pred(outputs, param_dict['num_class'], param_dict['thread'], prob= False)
    pred_input = tools.utils.out2pred(outputs, 1, 0.5, prob= False)
    s_score = 0
    rect_loss = 0
    

    for i in range(pred_input.shape[0]):
        pred = pred_input[i]
        
        ###################### RECTANGULAR LOSS ######################################################
        if rectangular:
            labeled_array, num_features = label(pred)
            for i in range(1, num_features):
                        
                new_mask = np.where(labeled_array == i, 1, 0)
                singl_cnt, _ = cv2.findContours(image=np.uint8(new_mask), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
                x,y,w,h = cv2.boundingRect(singl_cnt[0])
                center_hull = np.array([x+w//2, y+h//2])

                centroid = np.mean(np.argwhere(new_mask),axis=0)
                center_mass = np.array([int(centroid[1]), int(centroid[0])])

                area_mass = np.sum(new_mask == 1)
                area_hull = w*h
                
                term1 = np.sum(np.square(np.abs(center_mass - center_hull))) 
                term2 = np.square(np.abs((area_mass / area_hull) - 1))

                rect_loss += term1 + term2            

        ###################################################################################


        ###################### Symmetric LOSS ######################################################

        if symmetric:
            contours, hierarchy = cv2.findContours(image=np.uint8(pred), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
            #image_new = np.zeros((512,512), dtype='uint8')
            
            for component in contours:
                                        
                x,y,w,h = cv2.boundingRect(component)
                area = w*h
                                        
                half_w = math.ceil(w/2)
                half_h = math.ceil(h/2)
                                        
                fix_w = 0
                if (w % 2) != 0: #odd
                    fix_w = 1
                fix_h = 0
                if (h % 2) != 0: #odd
                    fix_h = 1

                left_half = pred[y:y + h, x:x + half_w]
                right_half = pred[y:y + h, x + half_w-fix_w:x + w]    

                top_half = pred[y:y + half_h, x:x + w]
                bottom_half = pred[y + half_h-fix_h: y + h, x:x + w]                        
                                        
                diff_v = cv2.absdiff(np.uint8(left_half), np.uint8(np.flip(right_half,1)))   #Higher difference, less symmetric
                diff_score_v = np.sum(diff_v)                                                       
                sv_score = diff_score_v / area

                                        
                diff_h = cv2.absdiff(np.uint8(top_half), np.uint8(np.flip(bottom_half,0)))   
                diff_score_h = np.sum(diff_h)                                                       
                sh_score = diff_score_h / area                            
                                        
                                        
                s_score += sv_score + sh_score
     

    """ cv2.rectangle(image_new, (x, y), (x+w, y+h), 255, 1)                                                        
    print('s_score: ', s_score)
    cv2.imshow('4', ((pred[0]*255)+image_new).astype(float))
    cv2.waitKey(0)
    cv2.destroyAllWindows() """

    
    return s_score, rect_loss

def calculate_gloabl_sym(outputs):
    pred_input = tools.utils.out2pred(outputs, 1, 0.5, prob= False)
    s_score = 0

    for i in range(pred_input.shape[0]):
        pred = pred_input[i]
        
        
        w,h = pred.shape
        midpoint = w // 2  
        area = w*h

        # Split the image into left and right halves
        left_half = pred[:, :midpoint]
        right_half = pred[:, midpoint:]

        diff_v = cv2.absdiff(np.uint8(left_half), np.uint8(np.flip(right_half,1)))   #Higher difference, less symmetric
        diff_score_v = np.sum(diff_v)                                                       
        s_score += diff_score_v / area
    
    return s_score

def regularizationLoss(input_segm_mask, pred_box, iou_threshold = 0.65, pen_non_matching= False, scale_bbox = False):
    
    if scale_bbox:
        
        for smpl in range(len(pred_box)):
            gt_boxes = center_to_corners_format(pred_box[smpl].boxes)
            target_sizes = [torch.tensor((image.shape[1], image.shape[2])) for image in input_segm_mask]  
            img_h = torch.Tensor([i[0] for i in target_sizes])
            img_w = torch.Tensor([i[1] for i in target_sizes])
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).cuda()            
            gt_un_boxes = gt_boxes * scale_fct[:, None, :][smpl]    
                   
            pred_box[smpl].update({'boxes': gt_un_boxes})
        

        """ image = np.zeros((param_dict['img_size'],param_dict['img_size'],1), dtype='uint8')
        
        for box in gt_un_boxes:
            x0, y0, x1, y1 = box.tolist() 
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                
            cv2.rectangle(image (x0, y0), (x1, y1), 1, 1)
                                                                       
        
        cv2.imshow('4', ((image*255)).astype(float))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pdb.set_trace() """

    
    # Binarize segmentation mask first:    
    thread = 0.5                    
    input_mask = tools.utils.out2pred(input_segm_mask, 1, thread, prob= False)                                                                        

    det_loss = 0

    for sample in range(input_mask.shape[0]):
        pred_mask = input_mask[sample]

        labeled_array, num_features = label(pred_mask)
        if num_features > 1:
            
            """ cv2.imshow('4', ((input_segm_mask[0][0].cpu().data.numpy()*255)).astype(float))
            cv2.imshow('5', ((pred_mask*255)).astype(float))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            pdb.set_trace() """
            
        for i in range(1, num_features):
            new_mask = np.where(labeled_array == i, 1, 0)            
            singl_cnt, _ = cv2.findContours(image=np.uint8(new_mask), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
            x,y,w,h = cv2.boundingRect(singl_cnt[0])
            m_x0 = x
            m_y0 = y
            m_x1 = x+w
            m_y1 = y+h
        
            center_hull = np.array([x+w//2, y+h//2])
            
            max_iou = -1                                            
            matching_bbox_idx = -1
            iou_threshold = iou_threshold

            cur_predbox = pred_box[sample]['boxes']
            
            # For the current segment, fing matching BBox
            for idx, box in enumerate(cur_predbox):
                                                                    
                b_x0, b_y0, b_x1, b_y1 = box.tolist()

                xA = max(m_x0, b_x0)
                yA = max(m_y0, b_y0)
                xB = min(m_x1, b_x1)
                yB = min(m_y1, b_y1)

                interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))                                                                                                
                
                m_boxArea = abs(w*h)                                                
                b_boxArea = abs((b_x1 - b_x0) * (b_y1 - b_y0))

                iou = interArea / float(m_boxArea + b_boxArea - interArea)
                
                if iou >= iou_threshold and iou > max_iou:
                    max_iou = iou
                    matching_bbox_idx = idx
                                
            
            if pen_non_matching:   
                if max_iou == -1: # No matching BBox for the current segment
                    det_loss += 0.25

            if max_iou != -1: # Matching BBox -> compute difference with segment hull                                                        
                c_x0, c_y0, c_x1, c_y1 = cur_predbox[matching_bbox_idx].cpu().data.numpy()
                

                # Compute Loss                                            
                c_w = c_x1 - c_x0
                c_h = c_y1 - c_y0
                center_det = np.array([(c_x0 + c_x1) / 2, (c_y0 + c_y1) / 2])

                term1 = np.sum(np.square(np.abs((center_hull - center_det))))
                term2 = np.square(np.abs((c_w - w)))
                term3 = np.square(np.abs((c_h - h)))

                det_loss += (term1 + term2 + term3) #/ cur_predbox.shape[0]    

                if True: #type_loss == 'L1-IoU':
                    target_bbox =  cur_predbox[matching_bbox_idx]  
                    src_bbox = torch.tensor([m_x0, m_y0, m_x1, m_y1]).cuda()
                    L1_loss_bbox = nn.functional.l1_loss(src_bbox, target_bbox, reduction='none').sum()   
                    det_loss += L1_loss_bbox                                  
                    
            
        if num_features != 0:
            det_loss = det_loss /  num_features      
              
    return det_loss/input_mask.shape[0]

def compute_corner_segmentation(labels, images,reg_model, loss_fcn, pro_target=None):    

    #Prepare GT
    corner_gt = np.zeros((images.shape[0],2,images.shape[2],images.shape[3]), dtype=np.uint8)

    for smpl in range(len(pro_target)):            
        gt_boxes = center_to_corners_format(pro_target[smpl].boxes)
        target_sizes = [torch.tensor((image.shape[1], image.shape[2])) for image in images]  
        img_h = torch.Tensor([i[0] for i in target_sizes])
        img_w = torch.Tensor([i[1] for i in target_sizes])
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).cuda()
            
        gt_un_boxes = gt_boxes * scale_fct[:, None, :][smpl]
        
        # Ground truth  
        for idx, box in enumerate(gt_un_boxes):
            x0, y0, x1, y1 = box.tolist() 
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(images.shape[2]-1, x1), min(images.shape[3]-1, y1)            

            corner_gt[smpl,0,y0, x0] = 1 
            #corner_gt[smpl,1,y0, x1] = idx+1
            corner_gt[smpl,1,y1, x1] = 1 
            #corner_gt[smpl,3,y1, x0] = idx+1                                    
    
    #Feed model    
    input_img = torch.cat((labels, images),dim=1)        
    out = reg_model(input_img)
    
    #Compute loss        
    #loss = loss_fcn(out,torch.tensor(corner_gt).float().cuda())

    loss1 = loss_fcn(out[:,0,:,:],torch.tensor(corner_gt[:,0,:,:]).float().cuda())
    loss2 = loss_fcn(out[:,1,:,:],torch.tensor(corner_gt[:,1,:,:]).float().cuda())

    loss = loss1+loss2
    
    return loss, out

def prepare_bbox_data(pred_box, size=None, images=None, scale_bbox=True):
    new_pred_box = []
    if scale_bbox:        
        for smpl in range(len(pred_box)):            
            gt_boxes = center_to_corners_format(pred_box[smpl].boxes)
            if size is not None:
                target_sizes = [torch.tensor((size, size)) for image in images] 
            else:
                target_sizes = [torch.tensor((image.shape[1], image.shape[2])) for image in images] 

            img_h = torch.Tensor([i[0] for i in target_sizes])
            img_w = torch.Tensor([i[1] for i in target_sizes])
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).cuda()            
            gt_un_boxes = gt_boxes * scale_fct[:, None, :][smpl]    
            
            new_pred_box.append({'boxes': gt_un_boxes})     
            #pred_box[smpl].update({'boxes': gt_un_boxes})    
    
    return new_pred_box

def matching_bbox(cur_predbox, m_x0,m_y0, m_x1, m_y1, iou_threshold):
    max_iou = -1                                            
    matching_bbox_idx = -1
    iou_threshold = iou_threshold

    for idx, box in enumerate(cur_predbox):
                                                                    
        b_x0, b_y0, b_x1, b_y1 = box.tolist()

        xA = max(m_x0, b_x0)
        yA = max(m_y0, b_y0)
        xB = min(m_x1, b_x1)
        yB = min(m_y1, b_y1)

        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))                                                                                                
        
        m_boxArea = abs((m_x1 - m_x0) * (m_y1 - m_y0))                                                
        b_boxArea = abs((b_x1 - b_x0) * (b_y1 - b_y0))

        iou = interArea / float(m_boxArea + b_boxArea - interArea)
        if iou >= iou_threshold and iou > max_iou:
            max_iou = iou
            matching_bbox_idx = idx
    return matching_bbox_idx

def corner_reg_inference(labels,images,reg_model, loss_fcn, pro_target=None, use_patches = False, batch_size = 4):
    loss = 0
    size = 128 #Scale GT BBox to size
    if use_patches:
        outputs = np.zeros((labels.shape[0],labels.shape[1],labels.shape[2],1), dtype='uint8')
    else:
        outputs = np.zeros((labels.shape[0],size,size,1), dtype='uint8')
    
    pred_box = None
    if pro_target is not None:
        pred_box = prepare_bbox_data(pro_target, size, images) 
        pred_box = pred_box[sample]

    coords_to_draw = []
    all_preds = all_targets = [] 
    for sample in range(labels.shape[0]):
        gt = []
        in_patch = []
        local_gt = []
        scale_factor = []
        corner_pred = []       

        # For test the corner regression model:                
        in_patch, gt, local_gt, scale_factor = pre_process_regression(labels[sample],images[sample], in_patch, gt, local_gt, scale_factor,use_patches, pred_box, size, False)
        
        if len(in_patch) > 0:
            batch = create_variable_size_batches(in_patch, batch_size)
            for bt in batch:                                                                     
                bt_result = reg_model(torch.squeeze(bt,1))            
                corner_pred.append(bt_result)
                                
            all_preds = torch.cat(corner_pred, dim=0)

            if use_patches:
                all_targets = torch.tensor(local_gt).cuda() #Not GT but the input local coords of window segment
                
                # Post-process:
                global_targets = torch.tensor(gt).cuda() #Not GT but the input global coords of window segment
                all_scale_factor = torch.tensor(scale_factor).cuda()                    
                
                global_coords = torch.zeros_like(all_preds)
                
                #global_coords[:, 0] = global_targets[:,0] + (all_preds[:, 0] - all_targets[:, 0])*all_scale_factor[:, 0]
                #global_coords[:, 1] = global_targets[:,1] + (all_preds[:, 1] - all_targets[:, 1])*all_scale_factor[:, 1]
                #global_coords[:, 2] = global_targets[:,2] + (all_preds[:, 2] - all_targets[:, 2])*all_scale_factor[:, 0]
                #global_coords[:, 3] = global_targets[:,3] + (all_preds[:, 3] - all_targets[:, 3])*all_scale_factor[:, 1]  

                widths= global_targets[:,2] - global_targets[:,0]
                heights= global_targets[:,3] - global_targets[:,1]
                center_x = global_targets[:,0] + widths/2
                center_y = global_targets[:,1] + heights/2
                
                global_coords[:, 0] = center_x - (((all_preds[:,2] - all_preds[:,0])/2) *all_scale_factor[:, 0])
                global_coords[:, 1] = center_y - (((all_preds[:,3] - all_preds[:,1])/2) *all_scale_factor[:, 1])
                global_coords[:, 2] = center_x + (((all_preds[:,2] - all_preds[:,0])/2) *all_scale_factor[:, 0])
                global_coords[:, 3] = center_y + (((all_preds[:,3] - all_preds[:,1])/2) *all_scale_factor[:, 1])

                coords_to_draw =  global_coords   

            else:                
                all_targets = torch.tensor(gt)
                coords_to_draw = all_preds
        else:
            continue
    
    for box in coords_to_draw:                                                                                         
        x0, y0, x1, y1 = box.tolist() 
        x0, y0, x1, y1 = round(x0), round(y0), round(x1), round(y1)
                
        cv2.rectangle(outputs[sample], (x0, y0), (x1, y1), 1, -1)  
    
    if loss_fcn is not None:
        loss += loss_fcn(all_preds, all_targets)/all_targets.shape[0] 

    return outputs, loss

def crop_expanded_patch(mask, img, x0, y0, width, height, ratio=0.1, size=128):
    x1 = x0 + width
    y1 = y0 + height

    new_x0 = x0 - (width * ratio)
    new_y0 = y0 - (height * ratio)
    new_x1 = x1 + (width * ratio)
    new_y1 = y1 + (height * ratio)

    # Ensure the coordinates are within bounds
    new_x0 = math.ceil(max(new_x0, 0))
    new_y0 = math.ceil(max(new_y0, 0))
    new_x1 = math.ceil(min(new_x1, mask.shape[1]))
    new_y1 = math.ceil(min(new_y1, mask.shape[0]))
    
    patch_mask = torch.unsqueeze(mask[new_y0:new_y1 + 1, new_x0:new_x1 + 1],0)
    patch_img = img[:, new_y0:new_y1+1, new_x0:new_x1+1]
    #patch_pos = position_channel[:, int(new_y0):int(new_y1), int(new_x0):int(new_x1)]
    
    # Get the local GT coordinated of the windows
    #indices = torch.argwhere(patch_mask[0] == 1).cpu()
    #top_left= indices.min(axis=0)    
    #local_or_gt = torch.tensor([(top_left[0][1].item(),top_left[0][0].item()),(top_left[0][1].item()+width,top_left[0][0].item()+height) ])
        
    new_patch_mask = nn.functional.interpolate(torch.unsqueeze(patch_mask,0).float(), size=size, mode="bilinear", align_corners=False) 
    patch_img = nn.functional.interpolate(torch.unsqueeze(patch_img,0), size=size, mode="bilinear", align_corners=False)
    #patch_pos_n = nn.functional.interpolate(torch.unsqueeze(patch_pos,0).float(), size=size, mode="bilinear", align_corners=False) 

    # Get local GT in reshaped mask    
    singl_cnt, _ = cv2.findContours(image=np.uint8(new_patch_mask[0][0].cpu()), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    if len(singl_cnt) > 1: 
        print('2')       
        pdb.set_trace()
        

    x,y,w,h = cv2.boundingRect(singl_cnt[0])
    global_gt = [x0,y0,x1,y1]
    local_gt = [x,y,x+w,y+h]

    # Global / local
    scale_x = (x1 - x0) / w 
    scale_y = (y1 - y0) / h
    scale_factor = [scale_x,scale_y]

    #or_patch_shape = [patch_mask.shape[1],patch_mask.shape[2]]

    out = torch.cat((new_patch_mask, patch_img),dim=1)
    
    """ mean = np.array([0.472455, 0.320782, 0.318403])
    std = np.array([0.215084, 0.408135, 0.409993])
    rgb_img = (np.array(patch_img.cpu()).transpose(1,2,0)* std)+mean

    cv2.imshow('4', ((np.array(patch_mask[0].cpu())*255)).astype(float)) 
    cv2.imshow('5', (image).astype(float)) 
    cv2.imshow('6', ((np.array(new_patch_mask[0][0].cpu())*255)).astype(float)) 
    cv2.imshow('7', (image2.astype(float)) )
    cv2.imshow('8', (image3.astype(float)) )
    #cv2.imshow('5', (rgb_img).astype(float))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pdb.set_trace() """
    
    return out, local_gt, scale_factor

def pre_process_regression (labels,images, in_patch, global_gt, local_gt, scale_factor,use_patch=True, pred_box=None, size=128, do_match = True):

    # Extract bbox from segments
    pred_mask = labels
    labeled_array, num_features = label(pred_mask.cpu())            
    
    for idx in range(num_features):
        new_mask = np.where(labeled_array == idx+1, 1, 0)                                                            
                
        if use_patch: #Using patches and training with GT labels (independet training)
            singl_cnt, _ = cv2.findContours(image=np.uint8(new_mask), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            # Global coords/GT of each window
            
            if len(singl_cnt) > 1:
                print('1') 
                mask = np.zeros((new_mask.shape[1],new_mask.shape[0]), dtype='uint8')
                for cnt in singl_cnt:
                    mask = cv2.drawContours(image=mask, contours=[cnt], contourIdx=-1, color=1, thickness=2)
                
                cv2.imshow('1', ((new_mask*255)).astype(float))
                cv2.imshow('2', (mask).astype(float))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                pdb.set_trace()
                
                    

            x,y,w,h = cv2.boundingRect(singl_cnt[0])
            m_x0 = x
            m_y0 = y
            m_x1 = x+w
            m_y1 = y+h

            mask = np.zeros((new_mask.shape[1],new_mask.shape[0]), dtype='uint8')
            mask = cv2.rectangle(mask, (m_x0, m_y0), (m_x1, m_y1), 1, -1)
            patch, l_gt, scale = crop_expanded_patch(torch.tensor(mask).cuda(), images, x,y,w,h, size=size)
            
            # Append all the crops and GT coordinates for all windows in an image
            in_patch.append(patch)
            #loca_or_gt.append(l_or_gt)
            local_gt.append(l_gt)
            scale_factor.append(scale)
            global_gt.append([m_x0, m_y0, m_x1, m_y1]) #scaled corners: global coords only when training regCorner with GT
            
        else: # Not using patches, end-2-end training, matching GT          
            
            tmp = torch.unsqueeze(torch.tensor(new_mask),0).float()
            new_rmask = nn.functional.interpolate(torch.unsqueeze(tmp,0), size=size, mode="bilinear", align_corners=False)
            new_img = nn.functional.interpolate(torch.unsqueeze(images,0), size=size, mode="bilinear", align_corners=False)
            
            singl_cnt, _ = cv2.findContours(image=np.uint8(np.array(new_rmask[0][0])), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
            # Global coords/GT of each window
            
            if len(singl_cnt) > 0:
                out = torch.cat((new_rmask.cuda(), new_img),dim=1)
                
                if do_match:
                    x,y,w,h = cv2.boundingRect(singl_cnt[0])
                    m_x0 = x
                    m_y0 = y
                    m_x1 = x+w
                    m_y1 = y+h

                    # Global GT must  be given by real GT labels, not extracted from the segmentation mask, that only works when training regCorner with GT
                    # Find matching of x,y,w,h with any of the GT BBox
                    
                    cur_predbox = pred_box['boxes']
                    match_idx = matching_bbox(cur_predbox, m_x0,m_y0, m_x1, m_y1, 0.65)
                    
                    # Supervise the predictions with matching GT
                    # As long as there are predictions that match the real GT, I'm supervising those predictions
                    if match_idx != -1:                
                        coord=[cur_predbox[match_idx][0].item(), cur_predbox[match_idx][1].item(),cur_predbox[match_idx][2].item(),cur_predbox[match_idx][3].item()]
                        global_gt.append(coord)  
                        in_patch.append(out)   
                else:
                    in_patch.append(out)

            else:                
                continue       
                  
    return in_patch, global_gt, local_gt, scale_factor

def create_variable_size_batches(tensor_list, batch_size):
    batches = []
    num_elements = len(tensor_list)

    for start in range(0, num_elements, batch_size):
        end = min(start + batch_size, num_elements)
        batch = tensor_list[start:end]
        batches.append(torch.stack(batch))    
    return batches

def compute_corner_reg(labels, images,reg_model, loss_fcn, use_patches = False, pro_target=None, batch_size = 4):
    loss = 0
    size = 128 #Scale GT BBox to size
    
    pred_box = None
    if pro_target is not None:
        pred_box = prepare_bbox_data(pro_target, size, images)

    for sample in range(labels.shape[0]):
        gt = []
        in_patch = []
        corner_pred = []
        local_gt = []
        scale_factor = []

        if pro_target is not None:
            pred_box =  pred_box[sample]
        
        in_patch, gt, local_gt, scale_factor = pre_process_regression(labels[sample],images[sample], in_patch, gt, local_gt, scale_factor, use_patches, pred_box, size)
        
        if len(gt) > 0:
            batch = create_variable_size_batches(in_patch, batch_size)            
            for bt in batch:                                   
                bt_result = reg_model(torch.squeeze(bt,1))            
                corner_pred.append(bt_result)
            
            all_preds = torch.cat(corner_pred, dim=0) 

            if use_patches:
                all_targets = torch.tensor(local_gt).cuda() # Only when training with GT labels...
                global_targets = torch.tensor(gt)
                all_scale_factor = torch.tensor(scale_factor)  
            else:
                all_targets = torch.tensor(gt).cuda()
                   
            #scale_fct = torch.tensor([size, size, size, size], device=all_targets.device)
            #norm_targets = all_targets / scale_fct[None, :]
                        
            loss += loss_fcn(all_preds, all_targets)/all_targets.shape[0]   
        else:
            continue        

        #loss += loss_fcn(all_preds.sigmoid(), norm_targets)        
    return loss #/ labels.shape[0]

def vect_stage(input_mask, pred_box, rgb_images, vect_model, iou_threshold = 0.65, pen_non_matching= False, scale_bbox = False, do_vectorize = True, do_reg_loss=False, smooth_l1_loss = None):
    
    if scale_bbox: # scale if pred_box is ground truth
        
        for smpl in range(len(pred_box)):
            gt_boxes = center_to_corners_format(pred_box[smpl].boxes)
            target_sizes = [torch.tensor((image.shape[1], image.shape[2])) for image in input_mask]  
            img_h = torch.Tensor([i[0] for i in target_sizes])
            img_w = torch.Tensor([i[1] for i in target_sizes])
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).cuda()            
            gt_un_boxes = gt_boxes * scale_fct[:, None, :][smpl]    

            pred_box[smpl].update({'boxes_corner': gt_boxes}) #corners normalized
            pred_box[smpl].update({'scaled_boxes_corner': gt_un_boxes}) #corners scaled
                    

        """ image = np.zeros((param_dict['img_size'],param_dict['img_size'],1), dtype='uint8')
        
        for box in gt_un_boxes:
            x0, y0, x1, y1 = box.tolist() 
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                
            cv2.rectangle(image (x0, y0), (x1, y1), 1, 1)
                                                                       
        
        cv2.imshow('4', ((image*255)).astype(float))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pdb.set_trace() """

    # Binarize segmentation mask first:                        
    input_mask = tools.utils.out2pred(input_mask, 1, 0.5, prob= False)
    
    # No binarization
    #input_mask = input_mask.cpu().data.numpy()
    
    #thread = 0.5
    #input_mask[input_mask >= thread] = 1
    #input_mask[input_mask < thread] = 0                                                                        

    vect_loss = 0
    
    

    for sample in range(input_mask.shape[0]):
        
        # Extract bbox from segments
        pred_mask = input_mask[sample]
        labeled_array, num_features = label(pred_mask)            
        pred_segm = []
        inst_segm = []
        target_bbox = []
        src_bbox = []
        for idx in range(num_features):
            new_mask = np.where(labeled_array == idx+1, 1, 0)                                               

            singl_cnt, _ = cv2.findContours(image=np.uint8(new_mask), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
            x,y,w,h = cv2.boundingRect(singl_cnt[0])
            m_x0 = x
            m_y0 = y
            m_x1 = x+w
            m_y1 = y+h

            inst_segm.append(new_mask)
            pred_segm.append([m_x0, m_y0, m_x1, m_y1]) #corners scaled         
        
        sca_cur_predbox = pred_box[sample]['scaled_boxes_corner']
        norm_cur_predbox = pred_box[sample]['boxes']#['boxes_corner']
        
        # Iterate for all GT boxes        
        for box, norm_box in zip(sca_cur_predbox, norm_cur_predbox):  
            #norm_b_x0, norm_b_y0, norm_b_x1, norm_b_y1 = norm_box.tolist()   
            norm_b_x, norm_b_y, norm_b_w, norm_b_h = norm_box.tolist()                   
            b_x0, b_y0, b_x1, b_y1 = box.tolist()
            
            center_bbox = np.array([(b_x0 + b_x1) / 2 , (b_y0 + b_y1) / 2])
            b_w = b_x1 - b_x0
            b_h = b_y1 - b_y0

            max_iou = -1                                            
            matching_bbox_idx = -1
            iou_threshold = iou_threshold
            

            # For the current GT box, find matching segment
            for idx in range(len(pred_segm)):                  
                m_x0, m_y0, m_x1, m_y1 = pred_segm[idx] 
                m_w = m_x1 - m_x0
                m_h = m_y1 - m_y0           

                xA = max(m_x0, b_x0)
                yA = max(m_y0, b_y0)
                xB = min(m_x1, b_x1)
                yB = min(m_y1, b_y1)

                interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))                                                                                                
                
                m_boxArea = abs(m_w*m_h)                                                
                b_boxArea = abs((b_x1 - b_x0) * (b_y1 - b_y0))

                iou = interArea / float(m_boxArea + b_boxArea - interArea)
                
                if iou >= iou_threshold and iou > max_iou:
                    max_iou = iou
                    matching_bbox_idx = idx                                
            
            if pen_non_matching:   
                if max_iou == -1: # No matching BBox for the current segment
                    vect_loss += 0.25
            
            # If there is a match for the current GT box:
            if max_iou != -1:                                                                      
                c_x0, c_y0, c_x1, c_y1 = pred_segm[matching_bbox_idx]#.cpu().data.numpy()
                cur_segm_mask = inst_segm[matching_bbox_idx]
                
                img_n_segm = torch.cat((rgb_images[sample], torch.unsqueeze(torch.tensor(cur_segm_mask, dtype=torch.float),0).cuda()), dim=0)  
                
                if do_vectorize:
                    # Use normalized corners coordinates to compute loss
                    pred_corners = vect_model(torch.unsqueeze(img_n_segm,0)) #returns x0,y0, x1,y1 normalized corners
                    #print(pred_corners.data)
                    src_bbox.append(pred_corners.data) 
                    #target_bbox.append([norm_b_x0, norm_b_y0, norm_b_x1, norm_b_y1])
                    target_bbox.append([norm_b_x, norm_b_y, norm_b_w, norm_b_h])
                    
                
                elif do_reg_loss:                
                    # Compute Loss                                            
                    c_w = c_x1 - c_x0
                    c_h = c_y1 - c_y0
                    center_seg = np.array([(c_x0 + c_x1) / 2, (c_y0 + c_y1) / 2])

                    term1 = np.sum(np.square(np.abs((center_bbox - center_seg))))
                    term2 = np.square(np.abs((c_w - b_w)))
                    term3 = np.square(np.abs((c_h - b_h)))

                    vect_loss += (term1 + term2 + term3) #/ cur_predbox.shape[0]    

                    if True: #type_loss == 'L1-IoU':
                        src_bbox =  torch.tensor(pred_segm[matching_bbox_idx]).cuda()
                        target_bbox = torch.tensor([b_x0, b_y0, b_x1, b_y1]).cuda()
                        L1_loss_bbox = nn.functional.l1_loss(src_bbox, target_bbox, reduction='none').sum()   
                        vect_loss += L1_loss_bbox     
                                       
        if len(src_bbox) > 0: #max_iou != -1:
            # Corners of all windows of one image
            all_preds = torch.sigmoid(torch.cat(src_bbox, dim=0))
            all_targets = torch.tensor(target_bbox).cuda()                      
            vect_loss += smooth_l1_loss(all_preds, all_targets)                                  
            #vect_loss = vect_loss / len(sca_cur_predbox)          
        
    #out = vect_loss/input_mask.shape[0]        
    return vect_loss

def detrLosses(outputs, targets, matcher, classes):
    
    # Retrieve the matching between the outputs of the last layer and the targets
    indices = matcher(outputs, targets)
    
    # Compute the average number of target boxes across all nodes, for normalization purposes
    num_boxes = sum(len(t["class_labels"]) for t in targets)
    num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
    num_boxes = torch.clamp(num_boxes, min=1).item()

    l_labels = loss_labels(outputs, targets, indices, num_boxes, classes)
    l_boxes = loss_boxes(outputs, targets, indices, num_boxes)
                    
    loss = l_labels + l_boxes
    return loss

def pos_process_boxes(add_out,img_size):

    tg= torch.tensor([img_size,img_size])
    tg = tg.repeat(img_size,1)


    out_logits, out_bbox = add_out['logits'], add_out['pred_boxes']

    prob = nn.functional.softmax(out_logits, -1)
    
    scores, labels = prob[..., :].max(-1)

    # Convert to [x0, y0, x1, y1] format
    boxes = center_to_corners_format(out_bbox) 
    
    img_h, img_w = tg.unbind(1)
    
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
    
    boxes = boxes * scale_fct
    return boxes

def visualize_attention(all_attentions, k, name, save):
    
    attention_weights = all_attentions[k]
    L = attention_weights.shape[0]
    num_tokens = attention_weights.shape[1]
    
    height = int(math.sqrt(num_tokens))
    width = int(math.sqrt(num_tokens))

    # Normalize attention weights across the source sequence length
    attention_weights = nn.functional.softmax(attention_weights, dim=-1)

    # Iterate over each sample in the batch
    
    # Iterate over each target query (L) and its corresponding attention weights
    for j in range(L):
        # Get the attention weights for the j-th target query
        query_attention_weights = attention_weights[j]
            
        # Resize attention weights to match the spatial dimensions of the input images
        query_attention_weights = query_attention_weights.view(height, width)
            
        # Visualize the attention weights as a heatmap
        imgplot =  plt.imshow(query_attention_weights.cpu().numpy(), cmap='hot', interpolation='nearest')
        #plt.colorbar()
        plt.title(f'Attention weights for query {j+1}')
        #plt.show()
        plt.savefig(os.path.join(save, 'test',str(j)+name), bbox_inches='tight')
    pdb.set_trace()
    
    
    
    #attentions = all_attentions[k] #first image
    #num_tokens = attentions.shape[1]
    #depth = attentions.shape[0]
    
    #height = int(math.sqrt(num_tokens))
    #width = int(math.sqrt(num_tokens))

    #attentions = attentions.reshape(height, width, depth)

    #gray_att = torch.sum(attentions,2)
    #gray_att = gray_att / attentions.shape[2]

    #fig = plt.figure(figsize=(30, 50))
    #for i in range(1):#num_heads):
    #        a = fig.add_subplot(5, 4, i+1)
    #        imgplot = plt.imshow(gray_att.data.cpu().numpy())
    #        a.axis("off")
            #a.set_title(names[i].split('(')[0], fontsize=30)
        
     
    #plt.savefig(os.path.join(save, name), bbox_inches='tight')

def corner_detector_algorithm(batch_size, img_size, pred_sub, kk):
    #img = np.uint8(np.repeat(pred_sub[:, :, np.newaxis], 3, axis=2))  *255
    img = np.zeros((batch_size, img_size,img_size), dtype='uint8')
    
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image=np.uint8(pred_sub))                
    ratio = 0.1
    
    for segm in range(1, n_labels):
        x0, y0, w, h, area = stats[segm] 
        cx, cy = centroids[segm]
        x1 = x0 + w
        y1 = y0 + h

        new_x0 = x0 - (w * ratio)
        new_y0 = y0 - (h * ratio)
        new_x1 = x1 + (w * ratio)
        new_y1 = y1 + (h * ratio)

        new_x0 = round(max(new_x0, 0))
        new_y0 = round(max(new_y0, 0))
        new_x1 = round(min(new_x1, pred_sub.shape[1]))
        new_y1 = round(min(new_y1, pred_sub.shape[0]))

        window = pred_sub[new_y0:new_y1+1, new_x0:new_x1+1]                     

        # Find corners within this bounding box using corner detection algorithms
        corners = cv2.goodFeaturesToTrack(np.uint8(window), maxCorners=4, qualityLevel=0.01, minDistance=20)
        
        if len(corners) < 4:
            contours, _ = cv2.findContours(image=np.uint8(window), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)                
            for component in contours:                            
                x,y,w,h = cv2.boundingRect(component) 
                g_x0 = cx - (w//2) 
                g_y0 = cy - (h//2)       
                g_x1 = g_x0 + w 
                g_y1 = g_y0 + h 
                
                g_x0, g_y0, g_x1, g_y1 = round(g_x0), round(g_y0), round(g_x1), round(g_y1)
                cv2.rectangle(img[kk], (g_x0, g_y0), (g_x1, g_y1), 1, -1)
            continue
        
        left = sorted(corners, key=lambda p: p[0][0])[:2]
        top_left = min(left, key=lambda p: p[0][1])
        bot_left = max(left, key=lambda p: p[0][1])
        right = sorted(corners, key=lambda p: p[0][0])[2:]
        top_right = min(right, key=lambda p: p[0][1])
        bot_right = max(right, key=lambda p: p[0][1])
        
        bb_x0, bb_y0 = top_left[0][0], top_left[0][1]
        bb_x1, bb_y1 = top_right[0][0], top_right[0][1]
        bb_x2, bb_y2 = bot_right[0][0], bot_right[0][1]
        bb_x3, bb_y3 = bot_left[0][0], bot_left[0][1]
        
        #Now compute new (x0,y0) (x1,y1) coordinates averaging 4 coorners                    
        n_x0 = np.mean((bb_x0,bb_x3))
        n_y0 = np.mean((bb_y0,bb_y1))
        n_x1 = np.mean((bb_x1,bb_x2))
        n_y1 = np.mean((bb_y2,bb_y3))  

        n_w = n_x1 - n_x0
        n_h = n_y1 - n_y0

        #Global averaged corners
        g_x0 = cx - (n_w/2) 
        g_y0 = cy - (n_h/2)       
        g_x1 = g_x0 + n_w 
        g_y1 = g_y0 + n_h  

        g_x0, g_y0, g_x1, g_y1 = round(g_x0), round(g_y0), round(g_x1), round(g_y1)
        cv2.rectangle(img[kk], (g_x0, g_y0), (g_x1, g_y1), 1, -1)
        
        #corners = np.int0(corners) 
        #for coord in corners: 
        #    bb_x, bb_y = coord.ravel()
        #    cv2.circle(img, (bb_x, bb_y), 1, (255, 0, 0), -1) 
    
        #contours, _ = cv2.findContours(image=np.uint8(window), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)                
        #for component in contours:                            
        #    x,y,w,h = cv2.boundingRect(component) 
        #    box = np.array([[x,y+h],[x,y],[x+w,y],[x+w,y+h]], dtype='float32') 
        #    box = np.int0(box)
        #    cv2.drawContours(img, [box], -1, (0,0,255) , 1)
        
    #cv2.imshow('outputs', (img).astype(float))        
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()  
    return img[kk]