import os
import SimpleITK as sitk
import glob
import monai
from monai.transforms import (

    AsDiscrete,
    RandAdjustContrastd,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    AddChanneld,
    SpatialPadd,
    RandRotate90d,
    RandShiftIntensityd,
    EnsureTyped,
    EnsureType,
    MapTransform,
    Resized,
    Invertd,
    ToTensord,
    NormalizeIntensityd,
    RandFlipd,
    Lambdad,
    Activations,
    AsDiscrete,
)
from monai.metrics import ROCAUCMetric
from monai.data import CacheDataset, ThreadDataLoader,DataLoader, Dataset, decollate_batch,load_decathlon_datalist
import torch
import torch.nn as  nn
from torch.nn import Linear,  Softmax
import torch.nn.functional as F
from monai.utils import first, set_determinism
from random import shuffle, seed
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./log/tensorboard')

torch.multiprocessing.set_sharing_strategy('file_system')
set_determinism(seed=1)

import pandas as pd
df_raw = pd.read_csv('/app/liucd/判定_fill_df.csv')
df_cli = df_raw[['patient_ID', 'T_stage', 'HER2_status', 'NAC_classification', 'ER_percentage', 'PR_percentage', 'Ki_67']]


syf_adcdir = '/app/liucd/deeplearn_dec/DL_dec/data_adc/syf/Mixed'
zy_adcdir = '/app/liucd/deeplearn_dec/DL_dec/data_adc/zunyi/Mixed'
sd_adcdir = '/app/liucd/deeplearn_dec/DL_dec/data_adc/shandong/Mixed'
yizhong_adcdir = '/app/liucd/deeplearn_dec/DL_dec/data_adc/yizhong/Mixed'
xian_adcdir = '/app/liucd/deeplearn_dec/DL_dec/data_adc/xian/Mixed'

syf_dcedir = '/app/liucd/deeplearn_dec/DL_dec/data/syf/Mixed'
zy_dcedir = '/app/liucd/deeplearn_dec/DL_dec/data/zunyi/Mixed'
sd_dcedir = '/app/liucd/deeplearn_dec/DL_dec/data/shandong/Mixed'
yizhong_dcedir = '/app/liucd/deeplearn_dec/DL_dec/data/yizhong/Mixed'
xian_dcedir = '/app/liucd/deeplearn_dec/DL_dec/data/xian/Mixed'

# 基线的路径
syf_bsadcdir = '/app/liucd/deeplearn_dec/DL_dec/data_adc/syf/NonPCR'
zy_bsadcdir = '/app/liucd/deeplearn_dec/DL_dec/data_adc/zunyi/NonPCR'
sd_bsadcdir = '/app/liucd/deeplearn_dec/DL_dec/data_adc/shandong/NonPCR'
yizhong_bsadcdir = '/app/liucd/deeplearn_dec/DL_dec/data_adc/yizhong/NonPCR'
xian_bsadcdir = '/app/liucd/deeplearn_dec/DL_dec/data_adc/xian/NonPCR'

syf_bsdcedir = '/app/liucd/deeplearn_dec/DL_dec/data/syf/NonPCR'
zy_bsdcedir = '/app/liucd/deeplearn_dec/DL_dec/data/zunyi/NonPCR'
sd_bsdcedir = '/app/liucd/deeplearn_dec/DL_dec/data/shandong/NonPCR'
yizhong_bsdcedir = '/app/liucd/deeplearn_dec/DL_dec/data/yizhong/NonPCR'
xian_bsdcedir = '/app/liucd/deeplearn_dec/DL_dec/data/xian/NonPCR'


train_adcimages = sorted(glob.glob(os.path.join(syf_adcdir,  '*.nii.gz'))) + \
                 sorted(glob.glob(os.path.join(zy_adcdir,  '*.nii.gz')))


train_dceimages = sorted(glob.glob(os.path.join(syf_dcedir,  '*.nii.gz'))) + \
                 sorted(glob.glob(os.path.join(zy_dcedir,  '*.nii.gz')))

val_adcimages =  sorted(glob.glob(os.path.join(sd_adcdir,  '*.nii.gz'))) + \
                 sorted(glob.glob(os.path.join(yizhong_adcdir,  '*.nii.gz'))) + \
                 sorted(glob.glob(os.path.join(xian_adcdir,  '*.nii.gz'))) 

val_dceimages =  sorted(glob.glob(os.path.join(sd_dcedir,  '*.nii.gz'))) + \
                 sorted(glob.glob(os.path.join(yizhong_dcedir,  '*.nii.gz'))) + \
                 sorted(glob.glob(os.path.join(xian_dcedir,  '*.nii.gz'))) 


# 基线的文件

train_bsadcimages = sorted(glob.glob(os.path.join(syf_bsadcdir,  '*.nii.gz'))) + \
                 sorted(glob.glob(os.path.join(zy_bsadcdir,  '*.nii.gz')))


train_bsdceimages = sorted(glob.glob(os.path.join(syf_bsdcedir,  '*.nii.gz'))) + \
                 sorted(glob.glob(os.path.join(zy_bsdcedir,  '*.nii.gz')))

val_bsadcimages =  sorted(glob.glob(os.path.join(sd_bsadcdir,  '*.nii.gz'))) + \
                 sorted(glob.glob(os.path.join(yizhong_bsadcdir,  '*.nii.gz'))) + \
                 sorted(glob.glob(os.path.join(xian_bsadcdir,  '*.nii.gz'))) 

val_bsdceimages =  sorted(glob.glob(os.path.join(sd_bsdcedir,  '*.nii.gz'))) + \
                 sorted(glob.glob(os.path.join(yizhong_bsdcedir,  '*.nii.gz'))) + \
                 sorted(glob.glob(os.path.join(xian_bsdcedir,  '*.nii.gz'))) 




train_clinical = []
for file_path in train_adcimages:
    p_id = file_path.split('_')[-4]
    clinical_data = df_cli[df_cli['patient_ID'] == int(p_id)].values.tolist()[0][1:]
    train_clinical.append(clinical_data)

val_clinical = []
for file_path in val_adcimages:
    p_id = file_path.split('_')[-4]
    clinical_data = df_cli[df_cli['patient_ID'] == int(p_id)].values.tolist()[0][1:]
    val_clinical.append(clinical_data)


train_dict = [{'image_adc': image_adc, 'image_dce': image_dce,
               'image_bsadc': image_bsadc, 'image_bsdce': image_bsdce,
               'clinical': clinical,  'label': int(image_adc.split('_')[-1].replace('.nii.gz', ''))}
                  for image_adc, image_dce, image_bsadc, image_bsdce, clinical in zip(
                      train_adcimages,  train_dceimages, train_bsadcimages,  train_bsdceimages, train_clinical)]

val_dict = [{'image_adc': image_adc, 'image_dce': image_dce, 
             'image_bsadc': image_bsadc, 'image_bsdce': image_bsdce, 
             'clinical': clinical,  'label': int(image_adc.split('_')[-1].replace('.nii.gz', ''))}
                  for image_adc, image_dce,image_bsadc, image_bsdce, clinical in zip(val_adcimages, val_dceimages,val_bsadcimages, val_bsdceimages, val_clinical)]

print(train_dict[-1])
print(len(train_dict), len(val_dict), len(train_dict + val_dict))

train_transforms = Compose(
        [
            LoadImaged(keys=["image_adc", "image_dce", "image_bsadc", "image_bsdce"]),
            EnsureChannelFirstd(keys=["image_adc", "image_dce", "image_bsadc", "image_bsdce"]),
            Orientationd(keys=["image_adc", 'image_dce', "image_bsadc", "image_bsdce"], axcodes="RAS"),
            Resized(keys=["image_adc", "image_bsadc"], spatial_size=(64, 64, 16)),
            Resized(keys=["image_dce", "image_bsdce"], spatial_size=(96, 96, 32)),
            
            NormalizeIntensityd(keys=["image_adc", "image_dce", "image_bsadc", "image_bsdce"], nonzero=True, channel_wise=True),
            
            RandFlipd( keys=["image_adc", "image_bsadc"], spatial_axis=[0], prob=0.50),
            RandFlipd( keys=["image_adc", "image_bsadc"], spatial_axis=[1], prob=0.50),
            RandFlipd( keys=["image_adc", "image_bsadc"], spatial_axis=[2], prob=0.50),
            
            RandFlipd( keys=["image_dce", "image_bsdce"], spatial_axis=[0], prob=0.50),
            RandFlipd( keys=["image_dce", "image_bsdce"], spatial_axis=[1], prob=0.50),
            RandFlipd( keys=["image_dce", "image_bsdce"], spatial_axis=[2], prob=0.50),
            
            RandRotate90d(keys=["image_adc", 'image_dce', "image_bsadc", 'image_bsdce'], prob=0.50, max_k=3 ),
            RandShiftIntensityd( keys=["image_adc", 'image_dce', "image_bsadc", 'image_bsdce'], offsets=0.10, prob=0.50),
            
            ToTensord(keys=['image_adc', 'image_dce','image_bsadc', 'image_bsdce', 'clinical',  'label'])
        ]
    )

val_transforms = Compose(
        [
            LoadImaged(keys=["image_adc",'image_dce','image_bsadc', 'image_bsdce' ]),
            EnsureChannelFirstd(keys=["image_adc", 'image_dce', 'image_bsadc', 'image_bsdce', ]),
            Orientationd(keys=["image_adc",'image_dce', 'image_bsadc', 'image_bsdce', ], axcodes="RAS"),
            Resized(keys=["image_adc", "image_bsadc"], spatial_size=(64, 64, 16)),
            Resized(keys=["image_dce", "image_bsdce"], spatial_size=(96, 96, 32)),
            
            NormalizeIntensityd(keys=["image_adc", 'image_dce', 'image_bsadc', 'image_bsdce', ], nonzero=True, channel_wise=True),
            ToTensord(keys=['image_adc', 'image_dce', 'image_bsadc', 'image_bsdce', 'clinical', 'label'])
        ]
    )


train_ds = CacheDataset(data=train_dict, transform=train_transforms, cache_rate=1, num_workers=12)
val_ds = CacheDataset(data=val_dict, transform=val_transforms, cache_rate=1, num_workers=12)

# create a training data loader
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=16, pin_memory=True)

# create a validation data loader
val_loader = DataLoader(val_ds, batch_size=8, num_workers=8, pin_memory=True)


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DoubleTower(nn.Module):
    def __init__(self,
                 pretrained_dce='',
                 pretrained_adc='',
                 device = torch.device("cuda"),
                 num_classes=2,
                 fc_hidden_size = 256
                ):
        super().__init__()
        self.pretrained_dce = pretrained_dce
        self.pretrained_adc = pretrained_adc
        self.fc_hidden_size = fc_hidden_size
        self.num_classes = num_classes
        self.device = device

        self.model_dce = monai.networks.nets.resnet34(spatial_dims=3, n_input_channels=1, num_classes=2, feed_forward=False).to(self.device)
        self.model_adc = monai.networks.nets.resnet34(spatial_dims=3, n_input_channels=1, num_classes=2, feed_forward=False).to(self.device)

        if  pretrained_dce != '':
            dce_dict = self.model_dce.state_dict()
            dce_pretrain = torch.load(self.pretrained_dce, map_location=self.device)
            dce_pretrain_dict = {k:v for k, v in dce_pretrain.items() if  k in  dce_dict.keys()}
            dce_dict.update(dce_pretrain_dict)
            self.model_dce.load_state_dict(dce_dict)

        if  pretrained_adc !='':
            adc_dict = self.model_adc.state_dict()
            adc_pretrain = torch.load(self.pretrained_adc, map_location=self.device)
            adc_pretrain_dict = {k:v for k, v in adc_pretrain.items() if  k in  adc_dict.keys()}
            adc_dict.update(adc_pretrain_dict)
            self.model_adc.load_state_dict(adc_dict)

        self.attn = nn.MultiheadAttention(512, num_heads=8, batch_first=True, device=self.device)

        # self.Linear1 = Linear(1024 + 6, self.num_classes, device=self.device)
        self.Linear1 = Linear(512, self.fc_hidden_size, device=self.device)  # 1024 是 所有下采样特征图globalpool之后拼接的结果
        self.Linear2 = Linear(self.fc_hidden_size + 6, self.num_classes, device=self.device)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x1, x2, bsx1, bsx2, structured_data):  # x 是SegResNet的输入影像矩阵

        encode_output1 = self.model_dce(x1)
        encode_output2 = self.model_dce(x2)
    
        encode_bsoutput1 = self.model_dce(bsx1)
        encode_bsoutput2 = self.model_dce(bsx2)

        concatenated = encode_output1 * encode_output2
        concatenatedbs = encode_bsoutput1 * encode_bsoutput2
        
        concat = torch.cat((concatenatedbs.unsqueeze(1), concatenated.unsqueeze(1)), dim=1)
        
        attn_output, _ = self.attn(concat, concat, concat)
        print(attn_output.shape)
        attn_output = attn_output[:, 1, :]

        fc1 = F.relu(self.Linear1(attn_output))
        fc1 = self.dropout(fc1)

        fc2 = self.Linear2( torch.concat([fc1, structured_data], dim=-1))
        return F.log_softmax(fc2, dim=-1)


# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

my_model = DoubleTower()

x1 = torch.randn(8, 1, 96, 96, 32)  # batch, channel, x, y, z
x2 = torch.randn(8, 1, 64, 64, 16)
bsx1 = torch.randn(8, 1, 96, 96, 32)  # batch, channel, x, y, z
bsx2 = torch.randn(8, 1, 64, 64, 16)

cli = torch.randn(8, 6)
output = my_model(x1.cuda(), x2.cuda(),bsx1.cuda(), bsx2.cuda(), cli.cuda())
print('output: ', output.shape)

# dce_pretrain_path = '/app/liucd/deeplearn_dec/DL_dec/pretrain/resnet_34.pth'
dce_pretrain_path = ''
adc_pretrain_path = ''

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = DoubleTower(dce_pretrain_path, adc_pretrain_path, device = device)
# model = DoubleTower(device = device)

post_pred = Compose([Activations(softmax=True)])
post_label = Compose([AsDiscrete(to_onehot=2)])

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
auc_metric = ROCAUCMetric()

# start a typical PyTorch training
val_interval = 1
best_metric = -1
best_metric_epoch = -1
max_epochs = 80
for epoch in range(max_epochs):

    model.train()
    epoch_loss = 0
    val_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        input_dce, input_adc,input_bsdce, input_bsadc, input_clinical, labels = batch_data["image_dce"].to(device), batch_data['image_adc'].to(device),  batch_data["image_bsdce"].to(device), batch_data['image_bsadc'].to(device), batch_data["clinical"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(input_dce, input_adc, input_bsdce, input_bsadc, input_clinical)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size

    epoch_loss /= step
    print(f"epoch {epoch + 1} average  train loss: {epoch_loss:.4f}")
    writer.add_scalar('loss/train_loss', epoch_loss, epoch)

    if (epoch + 1) % val_interval == 0:

        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)

            step2 = 0
            for val_data in val_loader:
                step2 += 1
                val_dce, val_adc,val_bsdce, val_bsadc, val_clinical, val_labels = val_data["image_dce"].to(device),val_data["image_adc"].to(device),val_data["image_bsdce"].to(device),val_data["image_bsadc"].to(device), val_data["clinical"].to(device), val_data["label"].to(device)
                val_output = model(val_dce, val_adc, val_bsdce, val_bsadc, val_clinical)
                y_pred = torch.cat([y_pred, val_output], dim=0)
                y = torch.cat([y, val_labels], dim=0)
                val_loss += loss_function(val_output, val_labels).item()

            val_loss /= step2
            writer.add_scalar('loss/val_loss', val_loss, epoch)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            auc_result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            if auc_result > best_metric:
                best_metric = auc_result
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_classification3d_dict.pth")
                print("saved new best metric model")
            print(
                "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                    epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch
                )
            )
            writer.add_scalar('metric/val_Acc', acc_metric, epoch)
            writer.add_scalar('metric/val_AUC', auc_result, epoch)

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")



