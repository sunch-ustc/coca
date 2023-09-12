#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : ZhenHuang (hz13@mail.ustc.edu.cn)
Date               : 2021-12-20 22:26
Last Modified By   : ZhenHuang (hz13@mail.ustc.edu.cn)
Last Modified Date : 2021-12-26 18:04
Description        : DataSet
-------- 
Copyright (c) 2021 Multimedia Group USTC. 
'''
from operator import truediv 
import logging
import torch  
from torchvision.utils import save_image
from torch.utils.data import Dataset

from typing import List, Tuple, Dict
import numpy as np
import h5py 
import pdb 
from collections import defaultdict 
from monai.transforms import Resize,Compose, Resized
from monai.transforms import  CropForegroundd,RandSpatialCropd,RandFlipd,RandScaleIntensityd,RandShiftIntensityd,CenterSpatialCropd,RandGaussianSharpend,RandGaussianSmoothd,Orientationd,Spacingd
_logger = logging.getLogger('vis') 
 
class BrainDatasetV1(Dataset):
    """ Dataset for brain data for HDF5 file, created by `create_hdf5_file_v1` function

    Init Args
    ---------
    data_list : list
        list of the arrays and labels
    train : bool
        whether for training
    num_frames : int, optional
        number of frames in the dataset, by default 6
    sample_step : int, optional
        step between consecutive frames, by default 5
    rand_crop : bool, optional
        whether spatially random crop
    """
    def __init__(
        self,
        data_list: List[Tuple[np.ndarray, int]],
        cfg = None,
        train: bool = False, 
        rand_crop: bool = False, 
        mean: list = [83.4, 65.6, 66.94],
        std: list = [73.06, 65.06, 64.06]
    ): 
        super().__init__()
        self.cfg = cfg
        self.data_list = data_list
        self.train = train  
        self.transform =    transformation( )  
        self.mean = mean
        self.std = std  

    def __len__(self):
        return len(self.data_list)
    def len(self):
        return len(self.data_list)
    def __getitem__(self, index): 
        array,num, label,xlsx = self.data_list[index]   
        
        array = torch.from_numpy(array).cuda().float()/255.0  
  
        if self.train:
            image = self.transform["train"]({'image':array })['image'] 
        else:
            image = self.transform["valid"]({'image':array })['image']
  
        image = Normalize_layer(image, self.mean , self.std )
     
        orig_image= array.cpu()
        xlsx["orig_image"] = self.transform["adv_orig"]({'image':orig_image })['image']
 
        return image, num,label,xlsx
    
def Normalize_layer(image, mean, std):
    dtype = image.dtype
    amplitude = 255 if image.max() < 200 else 1
    mean = torch.as_tensor(mean, dtype=dtype, device=image.device)/amplitude
    std = torch.as_tensor(std, dtype=dtype, device=image.device)/amplitude
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1, 1) 
    if std.ndim == 1:
        std = std.view(-1, 1, 1, 1) 
    image = (image-mean)/std 
    #image.sub_(mean).div_(std)
    return image 


def get_dataloader_from_hdf5(hdf5_path: str ,
                              modality: List[str] = [
                                  'T1_Ax'  , 'T1_E_Ax', 'T2_Ax'
                              ],
                              tumor: List[str] = ['choroid','ependymoma','glioma','mb'], 
                              interval=[0,0,0,0, 4, 12],  
                              just_test=False,
                              cfg = None
                              )  :
    """ 
    Get list for brain hdf5 file

    Parameters
    ----------
    hdf5_path : str
        path of the hdf5 path
    modality : List[str]
        list of modalities you want to use
      
    """
    
    ''' load hdf5 file ''' 
    assert hdf5_path.endswith('.hdf5'), '{} does not ends with .hdf5'.format(
        hdf5_path)
    _logger.info('Load file {}'.format(hdf5_path))
    hdf5_file = h5py.File(hdf5_path, 'r') 
    available_modalities = hdf5_file.attrs['modalities'].split('+')
    for m in modality:
        assert m in available_modalities, 'modality {} is not supported, maybe {}'.format(
            m, available_modalities) 
    
    ''' create info list ''' 
    mean = [34.66, 36.28, 26.8]
    std =  [73.44, 75.56, 57.0] 
    tumor_label = -1
    patient_num = defaultdict(int)
    tumor_idx = []
    info_list = defaultdict(list) 
    patient_num_label=0
    resize_monai=Resize(spatial_size=(384,384)) 
    # Tumor path
    for tumor_type, tumor_grp in hdf5_file.items(): 
        tumor_label += 1
        tumor_idx.append(tumor_type) 

        # Patient path
        for patient, patient_grp in tumor_grp.items():

            patient = tumor_type + "_" + patient  
            
            if just_test :
                if tumor_type=="mb" and patient_num[tumor_type] == 64: 
                    break
                elif tumor_type !="mb" and patient_num[tumor_type]==20:
                    break
            patient_num[tumor_type] += 1 

            array_list = [] 
            
            # Modility path
            for mod in modality:
                if mod in patient_grp: 
                    mod_array = np.array(patient_grp[mod])  
                else: 
                    _logger.warn(
                        'Tumor({}) - Patient({}) does not have modality {}'.
                        format(tumor_type, patient, mod)) 
                    mod_array = np.zeros((24,384,384))

                mod_array=resize_monai(mod_array)
                    
                if mod_array[interval[4]:interval[5]].sum() == 0:  
                    print(
                        'Tumor({}) - Patient({}) does not have modality {}, which equels 0'.
                        format(tumor_type, patient, mod))
                    mod_array[interval[4]:interval[5]] += 1  
                array_list.append(mod_array[interval[4]:interval[5]]) 
            stacked_array = np.stack(array_list, axis=0)    
            try:
                info_list[np.array(patient_grp["fold"]).item()].append((stacked_array, patient,tumor_label, {"patient_num_label"  : patient_num_label })) 
            except:
                pdb.set_trace()
            patient_num_label += 1  
    print("The number of all patients is "+str(patient_num_label))
  
    for tumor_type, tumor_grp in hdf5_file.items(): 
        print('Tumor({})----num({})'.format(tumor_type, patient_num[tumor_type]))
    train_list = info_list[1] + info_list[2] + info_list[3] 
    val_list = info_list[4]
    test_list = info_list[5]
    train_set = BrainDatasetV1( train_list, cfg = cfg, train=True ,mean = mean, std = std ) 
    test_set = BrainDatasetV1( test_list, cfg = cfg, train=False,mean = mean, std = std ) 
    val_set = BrainDatasetV1( val_list, cfg = cfg, train=False,mean = mean, std = std )
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                              batch_size=cfg.BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=cfg.NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                             batch_size=cfg.TEST_BATCH_SIZE,
                                             shuffle=True,
                                             num_workers=cfg.NUM_WORKERS,)   
    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                             batch_size=cfg.VAL_BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=cfg.NUM_WORKERS,)  
    return train_loader, val_loader, test_loader

def get_external_dataloader_from_hdf5(hdf5_path: str,
                              modality: List[str] = [
                                  'T1_Ax'  , 'T1_E_Ax', 'T2_Ax'
                              ],
                              tumor: List[str] = ['choroid','ependymoma','glioma','mb'],  
                              just_test=False,
                              cfg = None
                              )  :
    """ 
    Get list for brain hdf5 file

    Parameters
    ----------
    hdf5_path : str
        path of the hdf5 path
    modality : List[str]
        list of modalities you want to use
      
    """
    
    ''' load hdf5 file ''' 
    assert hdf5_path.endswith('.hdf5'), '{} does not ends with .hdf5'.format(
        hdf5_path)
    _logger.info('Load file {}'.format(hdf5_path))
    hdf5_file = h5py.File(hdf5_path, 'r') 
    available_modalities = hdf5_file.attrs['modalities'].split('+')
    for m in modality:
        assert m in available_modalities, 'modality {} is not supported, maybe {}'.format(
            m, available_modalities)
        
    ''' load mean and std '''
    mean_per_modality = {
        m: hdf5_file.attrs['mean_{}'.format(m)].astype(np.float16)
        for m in modality
    }
    std_per_modality = {
        m: hdf5_file.attrs['std_{}'.format(m)].astype(np.float16)
        for m in modality
    }  
 
    
    ''' create info list ''' 
    tumor_label = -1
    tumor_label_dict = {"choroid":0,"ependymoma":1,"glioma":2,"mb":3}
    patient_num = defaultdict(int)
    tumor_idx = []
    info_list = [] 
    store = []
    # Tumor path
    resize_monai=Resize(spatial_size=(384,384)) 
    for tumor_type, tumor_grp in hdf5_file.items(): 
        tumor_label += 1
        tumor_idx.append(tumor_type) 

        # Patient path
        for patient, patient_grp in tumor_grp.items(): 
            store.append(patient)
            patient =   patient  
            
            patient_num[tumor_type] += 1 
            array_list = [] 
            
            # Modility path
            for mod in modality:
                if mod in patient_grp: 
                    mod_array = np.array(patient_grp[mod])  
                else: 
                    _logger.warn(
                        'Tumor({}) - Patient({}) does not have modality {}'.
                        format(tumor_type, patient, mod)) 
                    mod_array = np.zeros(patient_grp[mod].shape)
                # print(patient)
                # print(mod_array[0].sum())
                # pdb.set_trace()
                mod_array=resize_monai(mod_array)
                array_list.append( mod_array ) 

            stacked_array = np.stack(array_list, axis=0)   
            info_list.append((stacked_array, patient, tumor_label_dict[tumor_type], { } ))   
    print(store)
    for tumor_type, tumor_grp in hdf5_file.items(): 
        print('Tumor({})----num({})'.format(tumor_type, patient_num[tumor_type]))
 
    test_list = info_list
    test_set = BrainDatasetV1( test_list, cfg = cfg, train=False )   
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                             batch_size=cfg.TEST_BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=cfg.NUM_WORKERS,)    
    return  test_loader

def transformation( ): 
  
    return { 'train': Compose([ 
    CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=1,allow_missing_keys=True), # 裁剪出image有像素信息的区域
    RandSpatialCropd(keys=["image", "label"], roi_size=(8,224,224), random_size=False,allow_missing_keys=True), # D, H, W  train_crop_size
    Resized(keys=["image", "label"],spatial_size=(8,224,224),allow_missing_keys=True),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=-1,allow_missing_keys=True),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=-2,allow_missing_keys=True),
    # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),  
    RandScaleIntensityd(keys="image", factors=0.1, prob=0.5), # 通过 v = v * (1 + 因子) 随机缩放输入图像的强度，其中因子是随机选取的。
    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5), # 使用随机选择的偏移量随机改变强度  
    # ToTensord(keys=["image", "label"]),
    ]),
    'valid': Compose([ 
    CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=1,allow_missing_keys=True),
    CenterSpatialCropd(keys=["image", "label"], roi_size=(8,224,224),allow_missing_keys=True),
    Resized(keys=["image", "label"],spatial_size=(8,224,224),allow_missing_keys=True), 
    # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),  
    ]), 
    'adv_orig':Compose([ 
    CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=1,allow_missing_keys=True),
    CenterSpatialCropd(keys=["image", "label"], roi_size=(8,224,224),allow_missing_keys=True),
    Resized(keys=["image", "label"],spatial_size=(8,224,224),allow_missing_keys=True),
    # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),  
    ])}    
  