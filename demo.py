import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from scipy import ndimage as ndi
from test_dataloader import directVectorization
from networks.CompletionNet.utils import draw_grd
from torch.utils.data import DataLoader, Dataset
import tools.transform as tr
from tools.dataloader import AmodalSegmentation
import tools, os
import torch
import pdb, cv2
import torch.nn as nn
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, DetrImageProcessor
from skimage.color import label2rgb
from scipy.ndimage import label
import torchvision.transforms as standard_transforms
from networks.get_model import get_net
from networks.CompletionNet.networks_BK0 import Generator as CN_Generator
import tqdm
import numpy as np
from tools.utils import read_image

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
    n_batch['df_fimage'] = inputs[12]
    return n_batch

def count_windows(segmented_image):
    # Your logic to count segmented windows
    return 42  

def load_occlusion_model():
    checkpoint_path = "/home/cero_ma/MCV/code220419_windows/0401_files/SegFormer_full-occ60-noNorm/50valiou_best.pth"
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

    return occ_model

def load_visi_model():
    checkpoint_path = "/home/cero_ma/MCV/code220419_windows/0401_files/Res_UNet_101_full-modern-occ80/pth_Res_UNet_101/110valiou_best.pth"
    
    state_dict = torch.load(checkpoint_path)['net']
    visi_model = get_net('Res_UNet_101', 3, 1, 512, None)     
    visi_model = torch.nn.DataParallel(visi_model, device_ids=[0])
    visi_model.load_state_dict(state_dict)
    visi_model.eval()

    for param in visi_model.parameters():
        param.requires_grad = False
    
    visi_model.cuda()

    return visi_model

def load_comp_network():
    checkpoint_path = '/home/cero_ma/MCV/code220419_windows/0401_files/DFV2_full-modern-occ80/./pth_DFV2/40valiou_best.pth'
    state_dict = torch.load(checkpoint_path)['net']

    im_channel = 1
    im_channel_mid = 1
    im_channel_out = 1
    D1_kernel = 7
    G = CN_Generator(cnum_in=im_channel+3, cnum_mid = im_channel_mid, cnum_out=im_channel_out, cnum=48, return_flow=True, k= D1_kernel)

    model = torch.nn.DataParallel(G, device_ids=[0])
    model.load_state_dict(state_dict)
    epoch = torch.load(checkpoint_path)['epoch']            
    model = model.cuda()
    model = model.eval()
    return model


def run_segmentation(path, occ_model, visi_model, model):
    img_size = 512
    num_class = 1
    thread = 0.5

    simulated_dataset = False
    raw_img_path = [path+'  '+path+'\n']

    composed_transforms_val = standard_transforms.Compose([
        tr.FixedResize(img_size),
        tr.Normalize(mean=(0.472455, 0.320782, 0.318403), std=(0.215084, 0.408135, 0.409993)),
        tr.ToTensor(do_not = {'img_sf', 'occ_sf'})]) 

    feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
    processor = DetrImageProcessor(do_resize =False, do_rescale = True)                    
    road_test = AmodalSegmentation(txt_path=None, img_path=raw_img_path, transform=composed_transforms_val, occSegFormer = True, feature_extractor= feature_extractor,  processor=processor, jsonAnnotation = "json_coco", simulated_dataset=simulated_dataset) 

    testloader = DataLoader(road_test, batch_size=1, shuffle=False, num_workers=0, drop_last=False, collate_fn=collate_fn)  

    test_num = len(testloader.dataset)

    label_all = np.zeros((test_num,) + (img_size, img_size), np.uint8)
    predict_all = np.zeros((test_num,) + (img_size, img_size), np.uint8)
    vect_predict_all = np.zeros((test_num,) + (img_size, img_size), np.uint8)    

    with torch.no_grad():
        batch_num = 0
        n = 0
        for i, data in tqdm.tqdm(enumerate(testloader), ascii=True, desc="test step"):

            images = torch.stack(data["image"], dim=0)
            labels = torch.stack(data["gt"], dim=0)
            img_path = np.stack(data["img_path"])
            gt_path = np.stack(data["gt_path"])
            sf_fimages = torch.stack(data["df_fimage"], dim=0)                
                
            sf_fimages = sf_fimages.cuda()

            i += images.size()[0]            
            images = images.cuda()

            #Visible network
            visible = visi_model(images)
                    
            # Occlusion network
            occ = occ_model(sf_fimages)
            occ = nn.functional.interpolate(occ.logits, size=img_size, mode="bilinear", align_corners=False) 
            occ = occ.argmax(dim=1)  
            occ = torch.unsqueeze(occ, 1)
            mask = occ.float()   
                    
            # Completion network
            image_masked = visible * (1.-mask) 
            ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
            
            
            grid_all = []                                                                  
            for i in range(visible.shape[0]):                            
                gr = draw_grd(visible[i][0], occ[i][0])/255
                grid_all.append(gr)
            
            grid = np.stack(grid_all, 0)
            grid = torch.tensor(grid).cuda()
            grid = torch.unsqueeze(grid,1).float()                                           

            x = torch.cat([image_masked, ones_x, ones_x*mask, grid*mask],dim=1)                                                                                                                          
            stg1, x_stage2, offset_flow = model(x, mask)
                    
            outputs = x_stage2
            pred = tools.utils.out2pred(outputs, num_class, thread)      
            pred_occ = tools.utils.out2pred(occ, num_class, thread)
            pred_visi = tools.utils.out2pred(visible, num_class, thread)       

            batch_num += images.size()[0]        

            for kk in range(len(img_path)):
                cur_name = os.path.basename(img_path[kk])                 
                pred_sub = pred[kk, :, :]            
                pred_sub_occ = pred_occ[kk, :, :]
                pred_sub_visi = pred_visi[kk, :, :]
                
                label_all[n] = read_image(gt_path[kk], 'gt')                                            
                predict_all[n]= pred_sub

                def overlappingGT(prediction, label, path = 'overL', rgb_image = False, imgPath = None, pred= None):                                                
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
                                                
                    return new_pred                    

                # Draw segments and RGB image                             
                segment_img = overlappingGT(None,label_all[n], 'overLRGB', rgb_image = True, imgPath=img_path[kk], pred = pred_sub)            

                # Vectorization: Direct contour            
                vect_predict_all[n] = directVectorization(pred_sub,img_size)            
                vector_img = overlappingGT(None,label_all[n], 'overLvect', rgb_image = True, imgPath=img_path[kk], pred = vect_predict_all[n],)             
    
    segment_img=Image.fromarray((segment_img).astype(np.uint8))
    vector_img=Image.fromarray((vector_img).astype(np.uint8))
    return segment_img, vector_img

def postprocess_image(segmented_image):
    
    return segmented_image

class global_var():
    def __init__(self):
        self.var = []

    def set_var(self, var):
        self.var = var

    def get_var(self):
        return self.var


class ImageSegmentationDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("A Deep learning-based framework for window information extraction from façade images with occlusions - Manuela Cerón")

        # Title label
        self.title_label = tk.Label(root, text="OA-WinSeg Demo", font=("Helvetica", 16))
        self.title_label.grid(row=0, column=1, pady=10)

        self.root.configure(bg="lightgray") 

        self.small_image_path = "/home/cero_ma/MCV/code220419_windows/thseg_clean/logo.png"         
        self.small_image = ImageTk.PhotoImage(self.resize_image(Image.open(self.small_image_path), (200, 70)))
        self.small_image_label = tk.Label(root, image=self.small_image, bg="lightgray")
        self.small_image_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")  # Align to the left

        self.g_vars = global_var()

        self.occ_model = load_occlusion_model()  
        self.visi_model = load_visi_model()
        self.model = load_comp_network() 

        self.cava_size= 400

        # UI Components        
       
        self.load_button = tk.Button(root, text="Load Image", font=("Helvetica", 11), command=self.load_image)
        self.load_button.grid(row=1, column=0, padx=20, pady=5)

        self.segment_button = tk.Button(root, text="Run Segmentation", font=("Helvetica", 11), command=self.segment_image)
        self.segment_button.grid(row=1, column=2, pady=10)

        # Create canvases to display images
        self.canvas_original = tk.Canvas(root, width=self.cava_size, height=self.cava_size, bg="lightgray", bd=2, relief=tk.GROOVE)
        self.canvas_original.grid(row=2, column=0, padx=10)
        self.label_original = tk.Label(root, text="Original Image", font=("Helvetica", 11))
        self.label_original.grid(row=3, column=0)

        self.canvas_segmented = tk.Canvas(root, width=self.cava_size, height=self.cava_size, bg="lightgray", bd=2, relief=tk.GROOVE)
        self.canvas_segmented.grid(row=2, column=1, padx=10)
        self.label_segmented = tk.Label(root, text="Segmented Image", font=("Helvetica", 11))
        self.label_segmented.grid(row=3, column=1)

        self.canvas_vectorized = tk.Canvas(root, width=self.cava_size, height=self.cava_size, bg="lightgray", bd=2, relief=tk.GROOVE)
        self.canvas_vectorized.grid(row=2, column=2, padx=10)
        self.label_vectorized = tk.Label(root, text="Vectorized Image", font=("Helvetica", 11))
        self.label_vectorized.grid(row=3, column=2)

        # Variable to store loaded image
        self.image = None

    def resize_image(self, image, size):
        return image.resize(size, Image.ANTIALIAS)

    def load_image(self):
        file_path = filedialog.askopenfilename()        
        self.g_vars.set_var(file_path)
        if file_path:
            self.image = Image.open(file_path)
            self.display_image(self.image, self.canvas_original)      
               
            

    def segment_image(self):
        if self.image:
            
            path = self.g_vars.get_var() 
            
            segm, vect = run_segmentation(path, self.occ_model, self.visi_model, self.model)           

            self.display_image(self.image, self.canvas_original)
            self.display_image(segm, self.canvas_segmented)
            self.display_image(vect, self.canvas_vectorized)

            # Count segmented windows and update the label
            window_count = count_windows(segm)
            #self.label.config(text=f"Number of Segmented Windows: {window_count}")

    def display_image(self, img, canvas):
        # Resize image to fit in the canvas
        img.thumbnail((self.cava_size, self.cava_size))
        img = ImageTk.PhotoImage(img)

        x_center = (self.cava_size - img.width()) // 2
        y_center = (self.cava_size - img.height()) // 2
        
        canvas.create_image(x_center, y_center, anchor=tk.NW, image=img)
        canvas.image = img

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationDemo(root)
    root.mainloop()
