from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import pdb
from ..deeplab.deeplabv3_plus import DeepLabV3Plus
from ..deeplab.deeplabv3_plus import _ASPP


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_c, num_class):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, num_class, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_class),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_class, num_class, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_class), 
            )
         

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_c, num_class):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_c, num_class, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_class),
            nn.ReLU(inplace=True),     
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=None): 
        super(UNet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = conv_block(in_c, filters[0]) #3,16 - 256
        self.Conv2 = conv_block(filters[0], filters[1]) #16,32 - 128
        self.Conv3 = conv_block(filters[1], filters[2]) #32,64 -64
        self.Conv4 = conv_block(filters[2], filters[3]) #64,128 -32
        self.Conv5 = conv_block(filters[3], filters[4]) #128,256 -16

        self.Up5 = up_conv(filters[4], filters[3]) #256 -> 128
        self.Up_conv5 = conv_block(filters[4], filters[3]) #256 -> 128

        self.Up4 = up_conv(filters[3], filters[2]) #128 -> 64
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], num_class, kernel_size=3, stride=1, padding=1)

        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4) 

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5) 

        d5 = self.Up5(e5) 
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        prob = out

        #out = self.activate(out)
        
        return out

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

#https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py

class AttU_Net(nn.Module):
    #def __init__(self,img_ch=3,output_ch=1, pretrained_path=None):
    def __init__(self,in_c=3,num_class=1, pretrained_path=None):
        super(AttU_Net,self).__init__()
        
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = conv_block(in_c, filters[0]) #3,16
        self.Conv2 = conv_block(filters[0], filters[1]) #16,32
        self.Conv3 = conv_block(filters[1], filters[2]) #32,64
        self.Conv4 = conv_block(filters[2], filters[3]) #64,128
        self.Conv5 = conv_block(filters[3], filters[4]) #128,256

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3],F_l=filters[3],F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2],F_l=filters[2],F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1],F_l=filters[1],F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0],F_l=filters[0],F_int=8)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], num_class, kernel_size=3, stride=1, padding=1)

        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv(d2)

        out = self.activate(d1)

        return out


class AlkModule(nn.Module):
    def __init__(self,in_c,out_c,k,r):
        super(AlkModule,self).__init__()

        # Vertical -> horizontal convolution
        self.leftStripConv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size = [1, k],  dilation =r, padding='same'),
            nn.Conv2d(out_c, out_c, kernel_size = [k, 1],  dilation =r, padding='same')
        )

        self.rightStripConv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size = [k, 1],  dilation =r, padding='same'),
            nn.Conv2d(out_c, out_c, kernel_size = [1, k],  dilation =r, padding='same')
        )
        
    def forward(self,x):

        l = self.leftStripConv(x)
        r = self.rightStripConv(x)

        return l+r

class ALKU_Net(nn.Module):
    def __init__(self, in_c=3, num_class=4, pretrained_path=None): 
        super(ALKU_Net, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = conv_block(in_c, filters[0]) #3,16
        self.Conv2 = conv_block(filters[0], filters[1]) #16,32
        self.Conv3 = conv_block(filters[1], filters[2]) #32,64
        self.Conv4 = conv_block(filters[2], filters[3]) #64,128
        self.Conv5 = conv_block(filters[3], filters[4]) #128,256


        k = 10
        d= 1
        self.AlkModule1 = AlkModule(filters[0], filters[0], k, d)
        self.AlkModule2 = AlkModule(filters[1], filters[1], k, d)
        self.AlkModule3 = AlkModule(filters[2], filters[2], k, d)
        self.AlkModule4 = AlkModule(filters[3], filters[3], k, d)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], num_class, kernel_size=3, stride=1, padding=1)

        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

    def forward(self, x):
        #encoding 
        e1 = self.Conv1(x) #16

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2) #32

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3) #64

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4) #128

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #decoding
        d5 = self.Up5(e5)
        e4 = self.AlkModule4(e4) #128
        
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        e3 = self.AlkModule3(e3) #64
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        e2 = self.AlkModule2(e2) #32
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        e1 = self.AlkModule1(e1) #16
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
               

        out = self.activate(out)
        
        return out

class ASPP_UNet(nn.Module):
    def __init__(self, in_c=3, num_class=4, pretrained_path=None): 
        super(ASPP_UNet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = conv_block(in_c, filters[0]) #3,16
        self.Conv2 = conv_block(filters[0], filters[1]) #16,32
        self.Conv3 = conv_block(filters[1], filters[2]) #32,64
        self.Conv4 = conv_block(filters[2], filters[3]) #64,128
        self.Conv5 = conv_block(filters[3], filters[4]) #128,256

        self.aspp1 = _ASPP(filters[0], [12, 24, 36], num_classannels = 16, norm_layer=nn.BatchNorm2d) 
        self.aspp2 = _ASPP(filters[1], [12, 24, 36], num_classannels = 32, norm_layer=nn.BatchNorm2d) 
        self.aspp3 = _ASPP(filters[2], [12, 24, 36], num_classannels = 64, norm_layer=nn.BatchNorm2d) 
        self.aspp4 = _ASPP(filters[3], [12, 24, 36], num_classannels = 128, norm_layer=nn.BatchNorm2d) 

        self.Up5 = up_conv(filters[4], filters[3]) #256,128
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], num_class, kernel_size=3, stride=1, padding=1)

        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

    def forward(self, x):
        #encoding 
        e1 = self.Conv1(x) #16

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2) #32

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3) #64

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4) #128

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5) #256

        #decoding
        d5 = self.Up5(e5) #128
        e4 = self.aspp4(e4) #out 128 
        
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5) # 256, 128

        d4 = self.Up4(d5)
        e3 = self.aspp3(e3) #64
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4) # 128, 64

        d3 = self.Up3(d4)
        e2 = self.aspp2(e2) #32
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3) #64, 34

        d2 = self.Up2(d3)
        e1 = self.aspp1(e1) #16
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2) #32, 16

        out = self.Conv(d2) #16, 1
               

        #out = self.activate(out)
        
        return out

class singleASPP_UNet(nn.Module):
    def __init__(self, in_c=3, num_class=4, pretrained_path=None): 
        super(singleASPP_UNet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = conv_block(in_c, filters[0]) #3,16
        self.Conv2 = conv_block(filters[0], filters[1]) #16,32
        self.Conv3 = conv_block(filters[1], filters[2]) #32,64
        self.Conv4 = conv_block(filters[2], filters[3]) #64,128
        self.Conv5 = conv_block(filters[3], filters[4]) #128,256

        rate=  [6,12,18] #[12,24,36]
        self.aspp = _ASPP(filters[4], rate, norm_layer=nn.BatchNorm2d) 

        self.Up5 = up_conv(filters[4], filters[3]) #256,128
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], num_class, kernel_size=3, stride=1, padding=1)

        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

    def forward(self, x):
        #encoding 
        e1 = self.Conv1(x) #16

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2) #32

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3) #64

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4) #128

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5) #256

        e5 = self.aspp(e5) #out 

        #decoding
        d5 = self.Up5(e5) #128
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5) # 256, 128

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4) # 128, 64

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3) #64, 34

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2) #32, 16

        out = self.Conv(d2) #16, 1
               

        #out = self.activate(out)
        
        return out


class BayesianUNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=None): 
        super(BayesianUNet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = conv_block(in_c, filters[0]) #3,16
        self.Conv2 = conv_block(filters[0], filters[1]) #16,32
        self.Conv3 = conv_block(filters[1], filters[2]) #32,64
        self.Conv4 = conv_block(filters[2], filters[3]) #64,128
        self.Conv5 = conv_block(filters[3], filters[4]) #128,256

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], num_class, kernel_size=3, stride=1, padding=1)

        self.Drop = nn.Dropout(0.5)

        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)

        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)

        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)

        e4 = self.Conv4(e4)
        e5 = self.Maxpool4(e4)

        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
              

        out = self.activate(out)
        
        return out