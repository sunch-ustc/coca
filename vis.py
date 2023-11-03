 

import imageio
import os
import itertools
import pdb  #pdb.set_trace()
import torch
import logging
import torch.nn.functional as F
import torchvision
import numpy as np
#import umap
# draw confusion_matrix
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Tuple, Text, BinaryIO
from filecmp import cmp
from torch.utils.tensorboard import SummaryWriter

from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,roc_curve, auc,roc_auc_score,classification_report,accuracy_score    # 生成混淆矩阵的函数
from sklearn.preprocessing import label_binarize 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from datetime import datetime

from scipy import interp
from itertools import cycle

# draw failure case
import cv2
from utils import mkdir_if_missing 

# cam
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

_logger = logging.getLogger('vis')

def compute_metrics(confusion_matrix):
    """
    计算混淆矩阵中的各项性能指标，并返回一个包含这些指标的字典。

    参数：
    confusion_matrix (list): 4x4的混淆矩阵（四分类问题）

    返回：
    metrics (dict): 包含AUC、F1分数、Sensitivity、Specificity和Accuracy的字典
    """
    # 定义类别数（这里是四分类问题）
    num_classes = len(confusion_matrix)

    # 初始化四分类问题的AUC字典
    auc_dict = {}

    # # 计算AUC
    # for i in range(num_classes):
    #     y_true = [1 if j == i else 0 for j in range(num_classes)]
    #     y_score = [confusion_matrix[j][i] for j in range(num_classes)]
    #     auc_dict[f'Class {i}'] = roc_auc_score(y_true, y_score)

    # 计算F1分数并求平均值
    f1_scores =  calculate_f1_score(confusion_matrix)

    # 计算Sensitivity和Specificity
    sensitivity_list = []
    specificity_list = [] 
    confusion_matrix_np = np.array(confusion_matrix)
    for class_label in range(num_classes): 
        # 提取类别为class_label的索引
        idx = class_label
        
        # 计算类别为class_label的真正例、假正例、真负例、假负例数量
        TP = confusion_matrix_np[idx, idx]
        FP = confusion_matrix_np[:, idx].sum() - TP
        FN = confusion_matrix_np[idx, :].sum() - TP
        TN = confusion_matrix_np.sum() - (TP + FP + FN)
        
        # 计算敏感性（Recall）和特异性
        sensitivity = TP / (TP + FN + 1e-8)  # 加上一个小的常数，避免分母为零
        specificity = TN / (TN + FP + 1e-8)
        
        sensitivity_list.append(round(sensitivity, 4))
        specificity_list.append(round(specificity, 4))
        
        mean_sensitivity = np.mean(sensitivity_list)
        mean_specificity = np.mean(specificity_list)

    # 计算Accuracy
    correct_classifications = sum(confusion_matrix[i][i] for i in range(num_classes))
    total_samples = sum(sum(confusion_matrix[i]) for i in range(num_classes))
    accuracy = correct_classifications / total_samples

    # 汇总所有指标到一个字典
    metrics = {
        # 'AUC': auc_dict,
        'F1 Score': f1_scores,
        'Sensitivity': mean_sensitivity,
        'Specificity': mean_specificity,
        'Accuracy': accuracy,
        'sensitivity_list': sensitivity_list,
        'specificity_list': specificity_list,
    }

    return metrics
def calculate_f1_score(confusion_matrix):
    # Calculate precision, recall, and F1 score for each class
    f1_scores = []
    for i in range(len(confusion_matrix)):
        TP = confusion_matrix[i][i]
        FP = sum(confusion_matrix[j][i] for j in range(len(confusion_matrix)) if j != i)
        FN = sum(confusion_matrix[i][j] for j in range(len(confusion_matrix)) if j != i)

        # Avoid division by zero in case of precision and recall being 0
        if TP + FP == 0 or TP + FN == 0:
            f1_score = 0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1_score = 2 * (precision * recall) / (precision + recall)

        f1_scores.append(f1_score)

    # Calculate the average F1 score
    avg_f1_score = sum(f1_scores) / len(f1_scores)

    return avg_f1_score
def sensitivity_specificity_multiclass(y_true, y_pred):
    """
    计算四分类问题中每个类别的敏感性和特异性。
    
    参数：
        y_true: 真实标签数组，形状为 (n_samples,)
        y_pred: 预测标签数组，形状为 (n_samples,)
        
    返回：
        sensitivity_list: 每个类别的敏感性（召回率）列表
        specificity_list: 每个类别的特异性列表
    """
    # 确定类别的数量
    num_classes = len(np.unique(y_true))
    
    sensitivity_list = []
    specificity_list = []
    
    for class_label in range(num_classes):
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 提取类别为class_label的索引
        idx = class_label
        
        # 计算类别为class_label的真正例、假正例、真负例、假负例数量
        TP = cm[idx, idx]
        FP = cm[:, idx].sum() - TP
        FN = cm[idx, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        
        # 计算敏感性（Recall）和特异性
        sensitivity = TP / (TP + FN + 1e-8)  # 加上一个小的常数，避免分母为零
        specificity = TN / (TN + FP + 1e-8)
        
        sensitivity_list.append(round(sensitivity, 4))
        specificity_list.append(round(specificity, 4))
        
        mean_sensitivity = np.mean(sensitivity_list)
        mean_specificity = np.mean(specificity_list)
    return sensitivity_list, specificity_list, mean_sensitivity, mean_specificity
def plot_multi_model_avg_roc_4class(cfg, test_loader, label_name, title="confusion_matrix", save_path=None, category = None):
    """
    绘制多个模型的四分类问题的平均ROC曲线

    Args:
    models_list (list): 包含多个PyTorch模型的列表。
    test_loader: 测试数据加载器，可以遍历数据批次。
    label_name (list): 每个模型对应的标签名称列表。
    title (str): 图表的标题，默认为 "confusion_matrix"。
    save_path (str): 图像保存的路径，如果不提供则不保存图像。

    Returns:
    None
    """
    # 创建一个新的图
    plt.figure()
    model_list = ["CLIP","COOP","COCOOP","Baseline","ACL","MLS","Ours"]
    # 设置颜色循环 
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'purple', 'green', 'yellow', 'red']

    # 存储每个类别的TPR和FPR
    all_tpr = []
    all_fpr = []
    from timm.models import load_checkpoint
    # 遍历每个模型并计算每个类别的TPR和FPR
    for i, (  color) in enumerate(zip( colors)):
        if i == 0:
            import   clip
            text = clip.tokenize(["This the MRI of choroid", "This the MRI of ependymoma", 
                        "This the MRI of glioma", "This the MRI of medulloblastoma"])
            model, preprocess = clip.load("RN50", text = text, device=cfg.DEVICE)
            for name,para in model.named_parameters():
                para.data=para.data.to(torch.float32)
            model.cuda()  
            load_checkpoint(model, "/data/sunch/output/Experiment/clip-5e-5/seed1/seed-1-fold-5/model_best.pth.tar", use_ema=False,strict=False)
        elif i ==1:
            from coop import creat_coop
            model = creat_coop(cfg,  classnames =  ["choroid", "ependymoma", "glioma", "medulloblastoma"] ).cuda()
            load_checkpoint(model, "/data/sunch/output/Experiment/coop-1e-4/seed1/seed-1-fold-5/model_best.pth.tar", use_ema=False,strict=False)
        elif i ==2 :
            from cocoop import creat_cocoop
            model = creat_cocoop(cfg,  classnames =  ["choroid", "ependymoma", "glioma", "medulloblastoma"] ).cuda()
            model.float()
            load_checkpoint(model, "/data/sunch/output/Experiment/cocoop-2.5e-5/seed1/seed-1-fold-5/model_best.pth.tar", use_ema=False,strict=False)
        elif i ==3 :
            from models import create_model
            model = create_model("resnet18", cfg.MODEL).cuda()
            load_checkpoint(model, "/data/sunch/output/Experiment/pre-train Resnet18_base_aug_transfer-1e-4/seed1/seed-1-fold-5/model_best.pth.tar", use_ema=False,strict=False)
        elif i ==5 :
            from models import create_model
            model = create_model("resnet50", cfg.MODEL).cuda()
            load_checkpoint(model, "/data/sunch/output/Experiment/Resnet50-1e-4/seed1/seed-1-fold-5/model_best.pth.tar", use_ema=False,strict=False)
        elif i ==4 :
            from models import create_model
            model = create_model("resnet18", cfg.MODEL).cuda()
            load_checkpoint(model, "/data/sunch/output/KBS/adv_training_b_0_1_repeat2_e0025-1e-4_T2_repeat/seed1/seed-1-fold-5/model_best.pth.tar", use_ema=False,strict=False)
                
        elif i ==6 :
            from models import create_model
            model = create_model("resnet18", cfg.MODEL).cuda()
            load_checkpoint(model, "/data/sunch/output/KBS/adv_training_b_0_1_repeat2_e0025-1e-4_T4/seed1/seed-1-fold-5/model_best.pth.tar", use_ema=False,strict=False)
        model.eval()  # 设置模型为评估模式
        all_probs = []
        all_true_labels = []

        # 遍历测试数据加载器
        for batch_idx, (input, _, target, xlsx) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()

            # 获取模型输出的预测概率
            with torch.no_grad():
                output,feature = model(input)
                probs = torch.softmax(output, dim=1)  # 获取四个类别的概率分布

            all_probs.append(probs.cpu().numpy())
            all_true_labels.append(target.cpu().numpy())

        # 将预测概率和真实标签转换为数组
        probs_array = np.concatenate(all_probs)
        true_labels_array = np.concatenate(all_true_labels)

        # 计算每个类别的ROC曲线
        fpr_list = []
        tpr_list = []
        # pdb.set_trace()
        for class_idx in range(4):
            fpr, tpr, _ = roc_curve(true_labels_array == class_idx, probs_array[:, class_idx])
            # 插值以确保TPR和FPR具有相同的长度
            interpolated_tpr = np.linspace(0, 1, 100)
            interpolated_fpr = np.interp(interpolated_tpr, tpr, fpr)
            fpr_list.append(interpolated_fpr)
            tpr_list.append(interpolated_tpr)

        # 将每个类别的TPR和FPR平均
        if category == None:
            avg_tpr = np.mean(tpr_list, axis=0)
            avg_fpr = np.mean(fpr_list, axis=0)
        else:
            avg_tpr =  tpr_list[category] 
            avg_fpr =  fpr_list[category]  
            title = title + label_name[category]
        all_tpr.append(avg_tpr)
        all_fpr.append(avg_fpr)

    # 绘制每个模型的平均ROC曲线
    for i, (avg_tpr, avg_fpr, color) in enumerate(zip(all_tpr, all_fpr, colors)):
        roc_auc = auc(avg_fpr, avg_tpr)
        plt.plot(avg_fpr, avg_tpr, color=color, lw=2, label=f'{model_list[i]} ') # (area = {roc_auc:.2f})

    # 绘制随机猜测的ROC曲线（对比用）
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.title(title)
    plt.legend(loc='lower right')

    # 保存图像
    if save_path:
        plt.savefig(save_path)

    # 显示图像
    plt.show()
def plot_confusion_matrix(model,test_loader,title="confusion_matrix",labels_name=['epe','med'],save_path=None,normalize=False,auroc=False,cfg =None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - labels_name : 混淆矩阵中的类别
    - normalize : True:显示百分比, False:显示个数
    """
    #-----  get confusion matrix from model and data
    model.eval()
    tumor_index = torch.arange(0,len(labels_name)).cpu().numpy()
    conf_matrix  = torch.zeros(len(labels_name), len(labels_name))
    target_all = []
    pred_all = []
    patient_all = []
    with torch.no_grad():
        for batch_idx, (input, patient, target,xlsx) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()
            
            if cfg.multi_classification == []:
                try:
                    output = model(input)  
                except:
                    num = int(int(input.shape[0]/4)*4)
                    print("num = (input.shape[0]/4)*4, num = ",num)
                    input = input[:num]
                    target = target[:num]
                    output = model(input) 
            else:
                output_multi = model(input)
                output = output_multi['class']
            
            if isinstance(output, (tuple, list)):
                output = output[0]
            mm = torch.nn.Softmax(dim=1)
            
            output_softmax = mm(output)
            if not batch_idx:
                output_softmax_all = output_softmax
            else:
                output_softmax_all = torch.cat((output_softmax_all, output_softmax), 0)
            pred = torch.argmax(output,1)
            pred_all.extend(pred.cpu().numpy())
            target_all.extend(target.cpu().numpy())
            patient_all.extend(patient)
            pred=pred.t()
    
    wrong_sample = []
    true_sample = []
    for i in range(len(target_all)):
        if target_all[i] != pred_all[i]: 
            #wrong_sample.append(labels_name[target_all[i]]+"_"+patient_all[i]+"_to_"+labels_name[pred_all[i]])
            wrong_sample.append(labels_name[target_all[i]]+"_"+patient_all[i] )
        else:
            # true_sample.append(labels_name[target_all[i]]+"_"+patient_all[i]+"_to_"+labels_name[pred_all[i]])
            true_sample.append(labels_name[target_all[i]]+"_"+patient_all[i] )
 
    cm = confusion_matrix(target_all,pred_all)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #----draw confusion matrix
    cm_n = cm 
    np.set_printoptions(precision=2)
    plt.figure(dpi=144)
    plt.imshow(cm_n, interpolation='nearest',cmap=plt.cm.Blues)    # 在特定的窗口上显示图像
    plt.title("Confusion_Matrix_"+title, fontsize=8)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, fontsize=8)    # , rotation=90 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name, fontsize=8)    # 将标签印在y轴坐标上
    # add numbers to the picture
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i,cm[i, j] , fontsize=20, #format(cm[i, j], fmt)
                 horizontalalignment="center",
                 color= "black")
    # show confusion matrix
    plt.ylabel('True label',fontsize=8)    
    plt.xlabel('Predicted label',fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        save_path_confusion = os.path.join(save_path,'confusion_matrix')
        if not os.path.exists(save_path_confusion):
            os.mkdir(save_path_confusion)
        plt.savefig(save_path_confusion +"/confusion_matrix_"+title+'.png', format='png') 
    else:
        plt.savefig('./fig/'+"confusion_matrix_"+title+'.png', format='png') 

    #----draw ROC
    if len(labels_name) == 2:
        fpr, tpr, threshold = roc_curve(target_all,output_softmax_all[:,1].cpu())
        auroc_macro = auc(fpr,tpr)
        auroc_micro = auroc_macro
        plt.figure()
        lw = 2
        plt.figure(dpi=144)
        plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % auroc_macro) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
    else:
        target_all_binarize = label_binarize(target_all, classes=tumor_index)

        n_classes = target_all_binarize.shape[1]

        # 计算每一类的ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(target_all_binarize[:, i], output_softmax_all[:, i].cpu())
            roc_auc[i] = auc(fpr[i], tpr[i])

        # micro（方法二）
        if auroc:
            fpr["micro"], tpr["micro"], _ = roc_curve(target_all_binarize.ravel(), output_softmax_all.cpu().numpy().ravel() )
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # macro（方法一）
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            # Finally average it and compute AUC
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Plot all ROC curves
            lw=2
            plt.figure(dpi = 144)
            # plt.plot(fpr["micro"], tpr["micro"],
            #         label='micro-average ROC curve (area = {0:0.2f})'
            #             ''.format(roc_auc["micro"]),
            #         color='deeppink', linestyle=':', linewidth=4)

            plt.plot(fpr["macro"], tpr["macro"],
                    label='macro-average ROC curve (area = {0:.3f})'
                        ''.format(roc_auc["macro"]),
                    color='navy', linestyle=':', linewidth=4)

            colors = cycle(['aqua', 'darkorange', 'cornflowerblue','mediumspringgreen'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                        label='ROC curve of class {0} (area = {1:.3f})'
                        ''.format(labels_name[i], roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('multi-calss ROC')

            plt.legend(loc="lower right",fontsize=7, markerscale=2.)
            auroc_macro = roc_auc_score(target_all_binarize,output_softmax_all.cpu().numpy(),average="macro")
            auroc_micro = roc_auc_score(target_all_binarize,output_softmax_all.cpu().numpy(),average="micro")
        else:
            auroc_macro =0
            auroc_micro =0
        if save_path is not None:
            save_path_roc = os.path.join(save_path,'roc')
            if not os.path.exists(save_path_roc):
                os.mkdir(save_path_roc)
            plt.savefig(save_path_roc +"/roc_"+title+'.png', format='png') 
        else:
            plt.savefig('./fig/'+"roc_"+title+'.png', format='png') 

        print('finish drawing confusion matrix and roc')
    
    return auroc_macro, auroc_micro,cm_n

def plot_all_confusion_matrix(confusion_matrix_array,title = "confusion_matrix",labels_name=['epe','med'], save_path=None,normalize=False,cmap=plt.cm.Blues):
 
    np.set_printoptions(precision=2)
    plt.figure(dpi=144)
    plt.imshow(confusion_matrix_array, interpolation='nearest',cmap=cmap )    # 在特定的窗口上显示图像
    plt.title("Confusion_Matrix_"+title, fontsize=8)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, fontsize=8)    # , rotation=90 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name, fontsize=8)    # 将标签印在y轴坐标上
    # add numbers to the picture
    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix_array.max() / 2.
    for i, j in itertools.product(range(confusion_matrix_array.shape[0]), range(confusion_matrix_array.shape[1])):
        plt.text(j, i,confusion_matrix_array[i, j] , fontsize=20,fontname='DejaVu Sans', #format(cm[i, j], fmt)
                 horizontalalignment="center",
                 color= "black" if confusion_matrix_array[i, j] < thresh else "white")
    # show confusion matrix
    plt.ylabel('True label',fontsize=8)    
    plt.xlabel('Predicted label',fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        save_path_confusion = os.path.join(save_path,'confusion_matrix')
        if not os.path.exists(save_path_confusion):
            os.mkdir(save_path_confusion)
        plt.savefig(save_path_confusion +"/confusion_matrix_"+title+'.png', format='png') 
    else:
        plt.savefig('./fig/'+"confusion_matrix_"+title+'.png', format='png') 


def Visualize_failure_case(model,test_loader,title="failure_case",labels_name=['epe','med'],save_path="./fig/failure_case"):
    """ 
    The funtion can Visualize failure case in different numbers of  modality

    title: the save_dir_name of one experiment
    """
    model.eval()
    save_path=os.path.join(save_path,title)
    mkdir_if_missing(save_path)
    with torch.no_grad():
        patient_num=0
        for batch_idx, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            _, pred = output.topk(1, 1, True, True)
            pred=pred.t()
            target=target.cpu().numpy() 
            pred=pred.cpu().numpy().reshape(target.shape)
            #find the false case 
            for index,judge in enumerate(pred!=target):
                if judge:
                    
                    save_path_temp=os.path.join(save_path,str(patient_num))
                    mkdir_if_missing(save_path_temp)
                    fcase=input.cpu().numpy()[index]
                    fcase = (fcase - fcase.min())/fcase.max()*255
                    true_label=target[index]
                    false_label=pred[index]
                    num=fcase.shape[0] # different modilaty 
                    while num:
                        num-=1
                        image_name="modality"+str(num)+"_"+labels_name[true_label]+"->"+labels_name[false_label]+'.jpeg'
                        for j in range(fcase[num].shape[0]):
                            
                            if j%6==0:
                                image_name="modality"+str(num)+"_"+labels_name[true_label]+"->"+labels_name[false_label]+str(j)+'.jpeg'
                                cv2.imwrite(os.path.join(save_path_temp,image_name),  fcase[num][j])
                    patient_num+=1

def Visualize_success_case(model,test_loader,title="success_case",labels_name=['epe','med'],save_path="./fig/success_case"):
    """ 
    The funtion can Visualize failure case in different numbers of  modality

    title: the save_dir_name of one experiment
    """
    model.eval()
    save_path=os.path.join(save_path,title)
    mkdir_if_missing(save_path)
    with torch.no_grad():
        patient_num=0
        for batch_idx, (input, num, target,xlsx) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()
            output, feature_vector = model(input)
            _, pred = output.topk(1, 1, True, True)
            pred=pred.t()
            target=target.cpu().numpy() 
            pred=pred.cpu().numpy().reshape(target.shape)
            #find the false case 
            for index,judge in enumerate(pred!=target):
                # if judge:
                #     pass
                # else:
                save_path_temp=os.path.join(save_path,num+str(patient_num))
                mkdir_if_missing(save_path_temp)
                fcase=input.cpu().numpy()[index]
                fcase = (fcase - fcase.min())/fcase.max()*255
                true_label=target[index]
                false_label=pred[index]
                num=fcase.shape[0] # different modilaty 
                while num:
                    num-=1
                    image_name="modality"+str(num)+"_"+labels_name[true_label]+"->"+labels_name[false_label]+'.jpeg'
                    for j in range(fcase[num].shape[0]):
                         
                        image_name="modality"+str(num)+"_"+labels_name[true_label]+"->"+labels_name[false_label]+str(j)+'.jpeg'
                        cv2.imwrite(os.path.join(save_path_temp,image_name),  fcase[num][j])
                patient_num+=1

def Visualize_all_case(test_loader,title="all_case",labels_name=['epe','med'],save_path="./fig/failure_case"):
    """ 
    The funtion can Visualize failure case in different numbers of  modality

    title: the save_dir_name of one experiment
    """

    save_path=os.path.join(save_path,title)
    mkdir_if_missing(save_path)
    with torch.no_grad():
        patient_num=0
        for batch_idx, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()
            target=target.cpu().numpy() 
 
            #find the false case 
            for index,true_label in enumerate(target):

                save_path_temp=os.path.join(save_path,str(patient_num))
                mkdir_if_missing(save_path_temp)
                fcase=input.cpu().numpy()[index]
                fcase = (fcase - fcase.min())/fcase.max()*255
            
                num=fcase.shape[0] # different modilaty 
                while num:
                    num-=1
                    for j in range(fcase[num].shape[0]):
                        
                        # if j%6==0:
                        image_name="modality"+str(num)+"_"+labels_name[true_label]+str(j)+'.jpeg'
                        cv2.imwrite(os.path.join(save_path_temp,image_name),  fcase[num][j])
                patient_num+=1
                      
class Visualizer(object):
    def __init__(self, model=None, train_loader=None, test_loader=None, use_tb=True, root='./vis', device='cuda'):
        if model is not None and getattr(model, 'module') is not None:
            model = model.module
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.use_tb = use_tb
        if use_tb:
            self.tb_writer = SummaryWriter(log_dir=root)
        self.root = root
        self.device = device


    def visualize(self, epoch, train_metrics={}, eval_metrics={}):
        if self.use_tb:
            self.update_tensorboard(epoch, train_metrics, eval_metrics)

    def update_tensorboard(self, epoch, train_metrics={}, eval_metrics={}):
        for k, v in train_metrics.items():
            self.tb_writer.add_scalar('train/{}'.format(k), v, epoch)
        for k, v in eval_metrics.items():
            self.tb_writer.add_scalar('eval/{}'.format(k), v, epoch)
    def adv_loss_tensorboard(self, epoch , loss_all={}):
        # if epoch % 1 == 0 : 
        for name,list in loss_all.items():
            for i, value in enumerate(list):
                self.tb_writer.add_scalar('adv/'+'ep'+str(epoch)+name , value, i)
                 
                 
# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    #print(grad_out[0].size())
    grad_block.append(grad_out[0].detach())

# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)

def grad_cam_show_img(inputs, mean, std, feature_maps, grads, mod, out_dir):
    '''
    feature_map [6,512,7,6]
    grad        [6,512,7,6]
    '''
    B, C, T, H, W= inputs.shape
    inputs = inputs.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W) #[6,3,218,182]
    for slice in range(inputs.shape[0]):
        input = inputs[slice,:,:,:]
        feature_map = feature_maps[slice,:,:,:] #[512,7,6]
        grad = grads[slice,:,:,:]     #[512,7,6]
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  #[7,6]
        grad = grad.reshape([grad.shape[0],-1]) #[512,42] 
        weights = np.mean(grad, axis=1) #[512,] 
        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :] 
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        cam = cv2.resize(cam, (W, H))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        out_slice_dir = os.path.join(out_dir,str(slice))
        if not os.path.exists(out_slice_dir):
            os.mkdir(out_slice_dir)

        for j in range(len(mean)):
            img = input[j,:,:] #[218,182]
            tmp = torch.zeros(H,W).cuda()
            if torch.equal(tmp,img):
                continue
            img *= std[j]
            img += mean[j]
            img = np.array(img.cpu())
            img = np.array([img,img,img])
            img = img.transpose(1, 2, 0)
            
            cam_img = 0.3 * heatmap + 0.7 * img
            path_cam_img = out_slice_dir + '/' + mod[j] + '.png'
            cv2.imwrite(path_cam_img, cam_img)

# 存放梯度和特征图
fmap_block = list()
grad_block = list()

def plot_grad_cam(model,test_loader,title="cam",labels_name=['epe','med'],mean = None,std = None,mod = None,save_path=None):
    model.eval()
    target_all = []
    pred_all = []
    
    # 注册hook
    hook1 = model.layer4[1].conv2.register_forward_hook(farward_hook)	# 9
    hook2 = model.layer4[1].conv2.register_backward_hook(backward_hook)
    save_path = os.path.join(save_path, 'grad_cam')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    num_right=0
    num_wrong=0
    for batch_idx, (input,patient, target,xlsx) in enumerate(test_loader):
        if num_wrong >1 and  num_right>1:
            break
        input = input.cuda()
        target = target.cuda()
        target_all.extend(target.cpu().numpy())
        output = model(input)
  
        pred = torch.argmax(output[0],1)
        pred_all.extend(pred.cpu().numpy())
        pred=pred.t()
        pdb.set_trace()
        class_loss = output[0][0,pred]
        class_loss.backward()
        patient_num = str(patient)[str(patient).find('\'') + 1:str(patient).find(',') - 1]
        # print(grad_block[batch_idx].size())
        # print(fmap_block[batch_idx].size())
        grads_val = grad_block[batch_idx].cpu().data.numpy().squeeze()
        fmap = fmap_block[batch_idx].cpu().data.numpy().squeeze()
        if target == pred:
            num_right += 1
            cam_path = os.path.join(save_path,'prdict_right')
            if not os.path.exists(cam_path):
                os.mkdir(cam_path)
            cam_tumor_path = os.path.join(cam_path,labels_name[pred])
            if not os.path.exists(cam_tumor_path):
                os.mkdir(cam_tumor_path)
            cam_patient_path = os.path.join(cam_tumor_path,str(patient_num))
            if not os.path.exists(cam_patient_path):
                os.mkdir(cam_patient_path)
        else:
            num_wrong += 1
            cam_path = os.path.join(save_path,'prdict_wrong')
            if not os.path.exists(cam_path):
                os.mkdir(cam_path)
            cam_true_tumor_path = os.path.join(cam_path,'true_' + labels_name[target])
            if not os.path.exists(cam_true_tumor_path):
                os.mkdir(cam_true_tumor_path)
            cam_pred_tumor_path = os.path.join(cam_true_tumor_path, 'pred_' + labels_name[pred])
            if not os.path.exists(cam_pred_tumor_path):
                os.mkdir(cam_pred_tumor_path)
            cam_patient_path = os.path.join(cam_pred_tumor_path,str(patient_num))
            if not os.path.exists(cam_patient_path):
                os.mkdir(cam_patient_path) 
        grad_cam_show_img(input, mean, std, fmap, grads_val, mod, cam_patient_path)
    hook1.remove()
    hook2.remove()

def grad_cam_diaobao_show_img(inputs, mean, std, grayscale_cam, mod, out_dir):
    B, C, T, H, W= inputs.shape
    inputs = inputs.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W) #[6,3,218,182]
    for slice in range(inputs.shape[0]):
        input = inputs[slice,:,:,:]
        grayscale_cam_slice = grayscale_cam[slice, :]
        out_slice_dir = os.path.join(out_dir,str(slice))
        if not os.path.exists(out_slice_dir):
            os.mkdir(out_slice_dir)
        for j in range(len(mean)):
            img = input[j,:,:] #[218,182]
            tmp = torch.zeros(H,W).cuda()
            if torch.equal(tmp,img):
                continue
            img *= std[j]
            img += mean[j]
            img = np.array(img.cpu())
            img = np.array([img,img,img])
            img = img.transpose(1, 2, 0)

            heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam_slice), cv2.COLORMAP_JET)
            cam_img = 0.3 * heatmap + 0.7 * img
            
            path_cam_img = out_slice_dir + '/' + mod[j] + '.png'
            cv2.imwrite(path_cam_img, cam_img)
def grad_cam_diaobao_show_img1(inputs, mean, std, grayscale_cam, mod, out_dir):
    B, C, T, H, W= inputs.shape
    inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()# .view(B * T, C, H, W) #[6,3,218,182]
    for slice in range(inputs.shape[0]):
        input = inputs[slice,:,:,:,:]
        grayscale_cam_slice = grayscale_cam[slice, :]
        out_slice_dir = os.path.join(out_dir,str(slice))
        if not os.path.exists(out_slice_dir):
            os.mkdir(out_slice_dir)
        for j in range(len(mean)):
            img = input[j,:,:] #[218,182]
            tmp = torch.zeros(H,W).cuda()
            if torch.equal(tmp,img):
                continue
            img *= std[j]
            img += mean[j]
            img = np.array(img.cpu())
            img = np.array([img,img,img])
            img = img.transpose(1, 2, 0)

            heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam_slice), cv2.COLORMAP_JET)
            cam_img = 0.3 * heatmap + 0.7 * img
            
            path_cam_img = out_slice_dir + '/' + mod[j] + '.png'
            cv2.imwrite(path_cam_img, cam_img)
def plot_grad_cam_diaobao(model,test_loader,title="cam",labels_name=['epe','med'],mean = None,std = None,mod = None,save_path=None):
    model.eval()
    target_all = []
    pred_all = []
    target_layers = [model.layer4[1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # save_path = os.path.join(save_path, 'grad_cam_diaobao')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    num_right=0
    num_wrong=0
    for batch_idx, (input,patient, target,xlsx) in enumerate(test_loader):
        # if num_wrong >1 and  num_right>1:
        #     break
        input = input.cuda()
        target = target.cuda()
        target_all.extend(target.cpu().numpy())
        output = model(input)
  
        pred = torch.argmax(output[0],1)
        pred_all.extend(pred.cpu().numpy())
        pred=pred.t()

        grayscale_cam = cam(input_tensor=input, targets=[ClassifierOutputTarget(pred)])
        print(grayscale_cam.shape)
        patient_num = str(patient)[str(patient).find('\'') + 1:str(patient).find(',') - 1]
        if target == pred:
            num_right += 1
            cam_path = os.path.join(save_path,'prdict_right')
            if not os.path.exists(cam_path):
                os.mkdir(cam_path)
            cam_tumor_path = os.path.join(cam_path,labels_name[pred])
            if not os.path.exists(cam_tumor_path):
                os.mkdir(cam_tumor_path)
            cam_patient_path = os.path.join(cam_tumor_path,str(patient_num))
            if not os.path.exists(cam_patient_path):
                os.mkdir(cam_patient_path)
        else:
            num_wrong += 1
            cam_path = os.path.join(save_path,'prdict_wrong')
            if not os.path.exists(cam_path):
                os.mkdir(cam_path)
            cam_true_tumor_path = os.path.join(cam_path,'true_' + labels_name[target])
            if not os.path.exists(cam_true_tumor_path):
                os.mkdir(cam_true_tumor_path)
            cam_pred_tumor_path = os.path.join(cam_true_tumor_path, 'pred_' + labels_name[pred])
            if not os.path.exists(cam_pred_tumor_path):
                os.mkdir(cam_pred_tumor_path)
            cam_patient_path = os.path.join(cam_pred_tumor_path,str(patient_num))
            if not os.path.exists(cam_patient_path):
                os.mkdir(cam_patient_path) 
        grad_cam_diaobao_show_img(input, mean, std, grayscale_cam, mod, cam_patient_path)
def joint_picture(path):
    for tumor_dir in os.listdir(path): 
        tumor_path  = os.path.join( path , tumor_dir ) 
        # 读取每一个病例的 cam图像文件
        for patient_dir in os.listdir(tumor_path):
            patient_path  = os.path.join( tumor_path , patient_dir ) 
            # 读取当前病例下面的 8 帧切片
            MRI_T1  = [ ]
            MRI_T1E = [ ]
            MRI_T2  = [ ]
            for patient_frame_dir in os.listdir(patient_path):
                if "png" in patient_frame_dir:
                    continue
                patient_frame_path = os.path.join( patient_path , patient_frame_dir )
                patient_frame_path_T1 = os.path.join( patient_frame_path , "T1_Ax.png" )
                patient_frame_path_T1E = os.path.join( patient_frame_path , "T1_E_Ax.png" )
                patient_frame_path_T2 = os.path.join( patient_frame_path , "T2_Ax.png" )
                if not os.path.exists(patient_frame_path_T1):
                    continue
                if not os.path.exists(patient_frame_path_T1E):
                    continue
                if not os.path.exists(patient_frame_path_T2):
                    continue
                MRI_T1.append(imageio.imread(patient_frame_path_T1))
                MRI_T1E.append(imageio.imread(patient_frame_path_T1E))
                MRI_T2.append(imageio.imread(patient_frame_path_T2))
            plot_multi_mri(MRI_T1,save_path=os.path.join( patient_path , patient_dir+"_T1_Ax.png" ),title=patient_dir+"_T1_Ax")
            plot_multi_mri(MRI_T1E,save_path=os.path.join( patient_path , patient_dir+"_T1_E_Ax.png" ),title=patient_dir+"_T1_E_Ax")
            plot_multi_mri(MRI_T2,save_path=os.path.join( patient_path , patient_dir+"_T2_Ax.png" ),title=patient_dir+"_T2_Ax")
def joint_wrong_picture(path):
    for orig_tumor_dir in os.listdir(path): 
        orig_tumor_path  = os.path.join( path , orig_tumor_dir ) 
        for tumor_dir in os.listdir(orig_tumor_path): 
            tumor_path  = os.path.join( orig_tumor_path , tumor_dir ) 
            # 读取每一个病例的 cam图像文件
            for patient_dir in os.listdir(tumor_path):
                patient_path  = os.path.join( tumor_path , patient_dir ) 
                # 读取当前病例下面的 8 帧切片
                MRI_T1  = [ ]
                MRI_T1E = [ ]
                MRI_T2  = [ ]
                if 'png' in patient_path:
                    os.system("rm "+patient_path)
                    continue
                for patient_frame_dir in os.listdir(patient_path):
                    if "png" in patient_frame_dir:
                        continue
                    patient_frame_path = os.path.join( patient_path , patient_frame_dir )
                    patient_frame_path_T1 = os.path.join( patient_frame_path , "T1_Ax.png" )
                    patient_frame_path_T1E = os.path.join( patient_frame_path , "T1_E_Ax.png" )
                    patient_frame_path_T2 = os.path.join( patient_frame_path , "T2_Ax.png" )
                    if not os.path.exists(patient_frame_path_T1):
                        continue
                    if not os.path.exists(patient_frame_path_T1E):
                        continue
                    if not os.path.exists(patient_frame_path_T2):
                        continue
                    MRI_T1.append(imageio.imread(patient_frame_path_T1))
                    MRI_T1E.append(imageio.imread(patient_frame_path_T1E))
                    MRI_T2.append(imageio.imread(patient_frame_path_T2))
                plot_multi_mri(MRI_T1,save_path=os.path.join( patient_path , patient_dir+"_T1_Ax.png" ),title=patient_dir+"_T1_Ax")
                plot_multi_mri(MRI_T1E,save_path=os.path.join( patient_path , patient_dir+"_T1_E_Ax.png" ),title=patient_dir+"_T1_E_Ax")
                plot_multi_mri(MRI_T2,save_path=os.path.join( patient_path , patient_dir+"_T2_Ax.png" ),title=patient_dir+"_T2_Ax")
def plot_multi_mri(images,save_path, title = 'Tiled MRI Images'):
    # 计算图像的行数和列数
    num_rows = 2 # int(np.ceil(len(images) ** 0.5))
    num_cols = 4 # int(np.ceil(len(images) / num_rows))

    # 创建一个大画布，用于绘制平铺图
    fig, axs = plt.subplots(num_rows, num_cols)

    # 遍历图像列表，并在画布上绘制图像
    for i, image in enumerate(images):
        row_idx = i // num_cols
        col_idx = i % num_cols
        axs[row_idx, col_idx].imshow(image, cmap='gray')
        axs[row_idx, col_idx].axis('off')
        # 添加子标题
        axs[row_idx, col_idx].set_title(f'Frame {i+4}')
          # 显式关闭当前图形窗口，以释放内存
    # 可选：如果图像数量不足，填充空白位置
    for i in range(len(images), num_rows * num_cols):
        row_idx = i // num_cols
        col_idx = i % num_cols
        axs[row_idx, col_idx].axis('off')
    # 添加总标题
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    # 保存绘制的平铺图为PNG格式
    plt.savefig(save_path)
    plt.close()
    # plt.show()
     
# 下面提供三种数据 降维的方法，将高维的数据降维到指定的维度 
def Feature_reduce_visial(X, labels, notion= ['choroid', 'ependymoma', 'glioma', 'mb'], method = "tsne", n_components = 2 , save_path = "/code/tumor/fig/feature_reduce",name = None ):# n_components 目标维度
    if method == "tsne": # 非线性降维算法 
        tsne_model = TSNE(n_components=n_components, random_state=42) 
        X_out = tsne_model.fit_transform(X) 
    elif method == "pca": 
        pca_model = PCA(n_components=n_components)
        X_out = pca_model.fit_transform(X) 
    elif method == "lda": 
        lda_model = LinearDiscriminantAnalysis(n_components=n_components)
        X_out = lda_model.fit_transform(X ,labels) 
        
    if n_components==3:
        # 创建3D图像
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 添加每个颜色的点相对应的类别
        # 为每个类别指定不同的颜色 mediumpurple
        colors = ['darkblue','darkslateblue',  'green', 'orange']
        for i, label in enumerate(np.unique(labels)):
            indices = labels == label
            print(labels[indices])
            ax.scatter(X_out[indices, 0], X_out[indices, 1], X_out[indices, 2], c=colors[i], label=notion[i], marker='o')
        # labels[indices]
        #   marker='o' 表示使用圆圈标记来绘制散点图  '+' : 加号标记
        # # 将数据可视化在三维空间中
        # ax.scatter(X_out[:, 0], X_out[:, 1], X_out[:, 2], c=labels, marker='o')
        # 添加图例 
        # 可选：添加其他绘图属性，例如标题、坐标轴标签等
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # 添加图例
        ax.legend()
        ax.set_title(name)
        plt.savefig(save_path)
        return X_out
    plt.clf()
    scatter = plt.scatter(X_out[:, 0], X_out[:, 1], c=labels  ) 
    # 添加标题
    if name == None:
        name = "The method of "+method
    plt.title(name)
    # 添加图例
    handles, _ = scatter.legend_elements()
    plt.legend(handles, notion) #['choroid', 'ependymoma', 'glioma', 'mb']
    #plt.legend() 
    #plt.imshow() 
    
    
    plt.savefig(save_path)
    return X_out
def visualize_tsne(model, loader , name="pretrain_pca" , n_components=3,
                   method = "pca",save_path = "/data/sunch/output/KBS_fig/pretrain/pretrain_pca_3D.png") :
    
    model.eval() 
    last_idx = len(loader) - 1
    target_all = []
    pred_all = []   
    output_all = [] 
    with torch.no_grad():
        for batch_idx, (input, num, target,xlsx) in enumerate(loader):
            
            last_batch = batch_idx == last_idx

            input = input.cuda()
            target = target.cuda() 
            output = model(input,adv = True)  
            output_all.append(output[1].cpu())
       
            target_all.extend(target.cpu().numpy())
               
        output = torch.cat(output_all, axis=0)
        target = np.array(target_all )
        Feature_reduce_visial(output.cpu().numpy(), target  , name="pretrain_pca", n_components=3,
                              method = "pca",save_path = "/data/sunch/output/KBS_fig/pretrain/pretrain_pca_3D.png") 
        # Feature_reduce_visial(output.cpu().numpy(), target  , name="pretrain_tsne",n_components=3,
        #                       method = "tsne",save_path = "/data/sunch/output/KBS_fig/pretrain/pretrain_tsne_3D.png") 
        # Feature_reduce_visial(output.cpu().numpy(), target  , name="adv_pca_3D", n_components=3,
        #                       method = "pca",save_path = "/data/sunch/output/KBS_fig/adv/adv_pca_3D.png") 
        # Feature_reduce_visial(output.cpu().numpy(), target  , name="adv_tsne_3D",n_components=3,
        #                       method = "tsne",save_path = "/data/sunch/output/KBS_fig/adv/adv_tsne_3D.png") 
if __name__ == '__main__':
 
    # joint_picture("/data/sunch/output/KBS_fig/adv/prdict_right")
    # joint_picture("/data/sunch/output/KBS_fig/pretrain/prdict_right")
    # joint_wrong_picture("/data/sunch/output/KBS_fig/adv/prdict_wrong")
    # joint_wrong_picture("/data/sunch/output/KBS_fig/pretrain/prdict_wrong")
    
    # 通过混淆矩阵计算 acc f1 sensitivity 和 specificity
    confusion_matrix = np.array([[18,  3,  0,  0],
       [ 0, 28,  2,  7],
       [ 0,  3, 25,  0],
       [ 0,  1,  3, 37]])
    confusion_matrix_doctor = [[64,23,2,10], 
                               [4,237,41,63],
                               [0,9,234,11], 
                               [1,95,24,600]]
    confusion_matrix_doctor_np = np.array(confusion_matrix_doctor)
    #pdb.set_trace()
    confusion_matrix_doctor_np = np.around(confusion_matrix_doctor_np / confusion_matrix_doctor_np.sum(axis=1,keepdims=True),  decimals=3)
 
    plot_all_confusion_matrix(confusion_matrix_doctor_np,title = "human_Reds",   labels_name=['Choroid','Ependymoma','Glioma','MB'], 
                              save_path="/data/sunch/output/KBS_fig",  cmap=plt.cm.Reds) 
    
    print(compute_metrics(confusion_matrix_doctor_np))
    # train_set, test_set = get_brain_hdf5_dataset_v1(
    #     './data/med_epe-Ax-T1_T1_E_T2.hdf5',
    #     modality=['T1_Ax_reg', 'T1_E_Ax_reg'],
    #     test_ratio=0.1,
    #     num_frames=6 ,
    #     sample_step=5,
    #     seed=1)

    # trainloader = torch.utils.data.DataLoader(dataset=train_set,
    #                                           batch_size=16,
    #                                           shuffle=True,
    #                                           num_workers=8)
    # testloader = torch.utils.data.DataLoader(dataset=test_set,
    #                                          batch_size=16,
    #                                          shuffle=False,
    #                                          num_workers=8)
    # Visualize_all_case(testloader)                                         
