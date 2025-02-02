import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import class_weight
from tools.lovasz_losses import lovasz_softmax
import pdb


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    pdb.set_trace()
    weights = np.ones(2)
    weights[classes] = cls_w
    out = torch.from_numpy(weights).float().cuda()
    pdb.set_trace()
    return out


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        
        target = torch.argmax(target, dim=1)
        loss = self.CE(output, target)
        return loss

class BCELoss(nn.Module):
    def __init__(self, smooth=1., weight=None, ignore_index=255):
        super(BCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.BCE = nn.BCELoss() #(reduction='none')Reduction none for BCNN

    def forward(self, output, target):
        output = output.squeeze(1)
        loss = self.BCE(output, target.float())
        
        return loss

class BCELogLoss(nn.Module):
    def __init__(self, smooth=1., weight=None, ignore_index=255):
        super(BCELogLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.BCELog = nn.BCEWithLogitsLoss(pos_weight = weight)
        

    def forward(self, output, target):
        
        output = output.squeeze(1)
        loss = self.BCELog(output, target.float())
        
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        
        #if self.ignore_index not in range(target.min(), target.max()):
        #    if (target == self.ignore_index).sum() > 0:
        #        target[target == self.ignore_index] = target.min()
        #target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        
        #output = output.sigmoid()
        #output_flat = output.contiguous().view(-1)
        #target_flat = target.contiguous().view(-1)
        #intersection = (output_flat * target_flat).sum()
        #loss = 1 - ((2. * intersection + self.smooth) /
        #            (output_flat.sum() + target_flat.sum() + self.smooth))

        
        output = output.sigmoid()
        output = output.flatten(1)
        target = target.unsqueeze(1)
        target = target.flatten(1)
        numerator = 2 * (output * target).sum(1)
        denominator = output.sum(-1) + target.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum()

        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = 2
        self.alpha = 0.5
        self.size_average = size_average
        #self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)
        #self.BCE_loss = nn.BCELoss(reduce=False)
        self.BCE_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        output = output.squeeze(1)
        logpt = self.BCE_loss(output, target.float())
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt * self.alpha
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss


def get_loss(loss_type, class_weights=None):
    print(loss_type, class_weights)
    

    if loss_type == 'ce':
        if class_weights is not None:
            return CrossEntropyLoss2d()
        else:
            return CrossEntropyLoss2d(weight=class_weights)
    elif loss_type == 'DiceLoss':
        return DiceLoss()
    elif loss_type == 'FocalLoss':
        return FocalLoss()
    elif loss_type == 'CE_DiceLoss':
        return CE_DiceLoss()
    elif loss_type == 'LovaszSoftmax':
        return LovaszSoftmax()
    elif loss_type == 'bce':
        return BCELoss()
    elif loss_type == 'bce_log':        
        return BCELogLoss(weight = class_weights[1])
    else:
        return None
