import numpy as np
import cv2
import os
from PIL import Image
from tools.utils import read_image
import pdb
from sklearn.metrics import confusion_matrix


def color_dict(labelFolder, classNum):
    colorDict = []
    
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        
        img_new = img[:, :, 0] * 1000000 + img[:, :, 1] * 1000 + img[:, :, 2]
        unique = np.unique(img_new)
       
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        
        colorDict = sorted(set(colorDict))
        
        if (len(colorDict) == classNum):
            break
    
    colorDict_BGR = []
    for k in range(len(colorDict)):
       
        color = str(colorDict[k]).rjust(9, '0')
        
        color_BGR = [int(color[0: 3]), int(color[3: 6]), int(color[6: 9])]
        colorDict_BGR.append(color_BGR)
    
    colorDict_BGR = np.array(colorDict_BGR)
    
    colorDict_GRAY = colorDict_BGR.reshape((colorDict_BGR.shape[0], 1, colorDict_BGR.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_BGR, colorDict_GRAY


def ConfusionMatrix(numClass, imgPredict, Label):

    #confusion_matrix(Label, imgPredict)

    mask = (Label >= 0) & (Label < numClass)
    label = numClass * Label[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=numClass ** 2)
    #pdb.set_trace()
    confusionMatrix = count.reshape(numClass, numClass)

    

    return confusionMatrix


def OverallAccuracy(confusionMatrix):
    # OA
    # acc = (TP + TN) / (TP + TN + FP + TN)
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return OA


def Precision(confusionMatrix):
    # precision
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    return precision


def Recall(confusionMatrix):
    # recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    return recall


def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score


def IntersectionOverUnion(confusionMatrix):
    # IoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix) #axis 1 is horizontal
    IoU = intersection / union
    return IoU


def MeanIntersectionOverUnion(confusionMatrix):
    #  mIoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    
    return mIoU


def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis=1) +
            np.sum(confusionMatrix, axis=0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU


def get_acc_v2(label_all, predict_all, classNum=2, save_path='./', file_name = 'accuracy.txt'):
    label_all = label_all.flatten()
    predict_all = predict_all.flatten()
    confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
    precision = Precision(confusionMatrix)
    recall = Recall(confusionMatrix)
    OA = OverallAccuracy(confusionMatrix)
    IoU = IntersectionOverUnion(confusionMatrix)
    FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
    mIOU = MeanIntersectionOverUnion(confusionMatrix)
    f1ccore = F1Score(confusionMatrix)
    print("")
    print("confusion_matrix:")
    print(confusionMatrix)
    print("precision:")
    print(precision)
    print("recall:")
    print(recall)
    print("F1-Score:")
    print(f1ccore)
    print("overall_accuracy:")
    print(OA)
    print("IoU:")
    print(IoU)
    print("mIoU:")
    print(mIOU)
    print("FWIoU:")
    print(FWIOU)
    with open('{}/{}'.format(save_path, file_name), 'w') as ff:
        ff.writelines("confusion_matrix:\n")
        ff.writelines(str(confusionMatrix)+"\n")
        ff.writelines("precision:\n")
        ff.writelines(str(precision)+"\n")
        ff.writelines("recall:\n")
        ff.writelines(str(recall)+"\n")
        ff.writelines("F1-Score:\n")
        ff.writelines(str(f1ccore)+"\n")
        ff.writelines("overall_accuracy:\n")
        ff.writelines(str(OA)+"\n")
        ff.writelines("IoU:\n")
        ff.writelines(str(IoU)+"\n")
        ff.writelines("mIoU:\n")
        ff.writelines(str(mIOU)+"\n")
        ff.writelines("FWIoU:\n")
        ff.writelines(str(FWIOU)+"\n")
    return precision, recall, f1ccore, OA, IoU, mIOU



def get_acc_info(PredictPath, LabelPath, classNum=2, save_path='./'):
    
    colorDict_BGR, colorDict_GRAY = color_dict(LabelPath, classNum)

    
    labelList = os.listdir(LabelPath)
    PredictList = os.listdir(PredictPath)

    
    Label0 = cv2.imread(PredictPath + "//" + PredictList[0], 0)

    
    label_num = len(PredictList)

    
    label_all = np.zeros((label_num,) + Label0.shape, np.uint8)
    predict_all = np.zeros((label_num,) + Label0.shape, np.uint8)
    for i in range(label_num):
        Label = read_image(LabelPath + "//" + PredictList[i], 'gt')
        label_all[i] = Label
        Predict = read_image(PredictPath + "//" + PredictList[i])
        predict_all[i] = Predict

    
    for i in range(colorDict_GRAY.shape[0]):
        label_all[label_all == colorDict_GRAY[i][0]] = i
        predict_all[predict_all == colorDict_GRAY[i][0]] = i

    
    label_all = label_all.flatten()
    predict_all = predict_all.flatten()

    
    confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
    precision = Precision(confusionMatrix)
    recall = Recall(confusionMatrix)
    OA = OverallAccuracy(confusionMatrix)
    IoU = IntersectionOverUnion(confusionMatrix)
    FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
    mIOU = MeanIntersectionOverUnion(confusionMatrix)
    f1ccore = F1Score(confusionMatrix)

    for i in range(colorDict_BGR.shape[0]):
        # pip install webcolors
        try:
            import webcolors

            rgb = colorDict_BGR[i]
            rgb[0], rgb[2] = rgb[2], rgb[0]
            print(webcolors.rgb_to_name(rgb), end="  ")
        
        except:
            print(colorDict_GRAY[i][0], end="  ")
    print("")
    print("confusion_matrix:")
    print(confusionMatrix)
    print("precision:")
    print(precision)
    print("recall:")
    print(recall)
    print("F1-Score:")
    print(f1ccore)
    print("overall_accuracy:")
    print(OA)
    print("IoU:")
    print(IoU)
    print("mIoU:")
    print(mIOU)
    print("FWIoU:")
    print(FWIOU)
    with open('{}/accuracy.txt'.format(save_path), 'w') as ff:
        ff.writelines("confusion_matrix:\n")
        ff.writelines(str(confusionMatrix)+"\n")
        ff.writelines("precision:\n")
        ff.writelines(str(precision)+"\n")
        ff.writelines("recall:\n")
        ff.writelines(str(recall)+"\n")
        ff.writelines("F1-Score:\n")
        ff.writelines(str(f1ccore)+"\n")
        ff.writelines("overall_accuracy:\n")
        ff.writelines(str(OA)+"\n")
        ff.writelines("IoU:\n")
        ff.writelines(str(IoU)+"\n")
        ff.writelines("mIoU:\n")
        ff.writelines(str(mIOU)+"\n")
        ff.writelines("FWIoU:\n")
        ff.writelines(str(FWIOU)+"\n")
    return precision, recall, f1ccore, OA, IoU, mIOU


if __name__ == '__main__':
    #################################################################
    LabelPath = r'path_to_trining_gt'
    PredictPath = r'path_to_prediction'
    classNum = 2
    #################################################################
    precision, recall, f1ccore, OA, IoU, mIOU = get_acc_info(PredictPath, LabelPath, classNum)
