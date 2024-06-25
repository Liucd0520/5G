import radiomics
from radiomics import featureextractor
import os 
import numpy as np 
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import glob
import six
from utils import *

def feature_extractor(image_dir, label_dir, raw_data_dir, save_path, logger):

    image_path = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
    label_path = sorted(glob.glob(os.path.join(label_dir, '*.nii.gz')))
#     print(image_path[-1], label_path[-1])
    
    extractor = featureextractor.RadiomicsFeatureExtractor('./radiomics_feature/Params.yaml')

    feature_name = []
    feature_values = []
    feature_values_tot_subjects = []
    
    prefixs = ('original', 'log', 'wavelet')  # 和 Param.yaml想匹配
    for idx, (image_name, label_name) in enumerate(zip(image_path, label_path)):

        subject_name = label_name.split('/')[-1]
        logger.info('subject name: {}'.format(subject_name))
        subject_path = filename2rawpath(subject_name)
        patient_id = determ_id(os.path.join(raw_data_dir, subject_path))
        try:
            features = extractor.execute(image_name, label_name)
            # 对于第一个数据，把所有的特征名称保存到数组里
            if idx == 0:  
                feature_name = [key for key, value in six.iteritems(features) if key.startswith(prefixs)] 
                feature_name.insert(0, 'patient_id')
            feature_values = [value for key, value in six.iteritems(features)  if key.startswith(prefixs)]    
            feature_values.insert(0, patient_id)
            # feature_values = renormalized_feature(feature_values)
            feature_values_tot_subjects.append(feature_values)
            logger.info('subject name: {} Done'.format(subject_name))
        except Exception as e:
            logger.info('{} Error: {}'.format(subject_name, e))
        
    df = pd.DataFrame(feature_values_tot_subjects, columns=feature_name)
    df.to_csv(save_path, index=False)
