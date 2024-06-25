import radiomics
from radiomics import featureextractor
import os 
import numpy as np 
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import glob
import six


def feature_extractor(image_dir, label_dir,  save_path):

    image_path = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
    label_path = sorted(glob.glob(os.path.join(label_dir, '*.nii.gz')))
#     print(image_path[-1], label_path[-1])

    extractor = featureextractor.RadiomicsFeatureExtractor('./Params.yaml')

    feature_name = []
    feature_values = []
    feature_values_tot_subjects = []

    prefixs = ('original', 'log', 'wavelet')  # 和 Param.yaml想匹配
    # origin_list = ['2021_08_16_1732688_DWI.nii.gz']
    for idx, (image_name, label_name) in enumerate(zip(image_path, label_path)):

        subject_name = label_name.split('/')[-1]
        print(idx, subject_name)
#         if subject_name in origin_list:
#             continue

        patient_id = subject_name.split('_')[-3]

        features = extractor.execute(image_name, label_name)
        # 对于第一个数据，把所有的特征名称保存到数组里
        if idx == 0:
            feature_name = [key for key, value in six.iteritems(features) if key.startswith(prefixs)]
            feature_name.insert(0, 'patient_id')
        feature_values = [value for key, value in six.iteritems(features)  if key.startswith(prefixs)]
        feature_values.insert(0, patient_id)
        # feature_values = renormalized_feature(feature_values)
        feature_values_tot_subjects.append(feature_values)

    df = pd.DataFrame(feature_values_tot_subjects, columns=feature_name)
    print(df)
    df.to_csv(save_path, index=False)


base_dir = '/app/liucd/deeplearn_dec/data/syf/merge/'

image2_dir = os.path.join(base_dir, 'DCE2')
label2_dir = os.path.join(base_dir, 'DCE_labelReg')

feature_extractor(image2_dir, label2_dir,  'syf.csv')
