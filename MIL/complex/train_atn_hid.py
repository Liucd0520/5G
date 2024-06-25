import os
import SimpleITK as sitk
import glob
import monai
import torchvision
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
from torchvision.models import resnet50, densenet121
from monai.networks.nets import milmodel

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


train_dict = [{'image_adc': image_adc, 'image_dce': image_dce, 'clinical': clinical,  'label': int(image_adc.split('_')[-1].replace('.nii.gz', ''))}
                  for image_adc, image_dce, clinical in zip(train_adcimages,  train_dceimages, train_clinical)]
val_dict = [{'image_adc': image_adc, 'image_dce': image_dce, 'clinical': clinical,  'label': int(image_adc.split('_')[-1].replace('.nii.gz', ''))}
                  for image_adc, image_dce, clinical in zip(val_adcimages, val_dceimages, val_clinical)]

print(train_dict[-1])
print(len(train_dict), len(val_dict), len(train_dict + val_dict))

train_transforms = Compose(
        [
            LoadImaged(keys=["image_adc", "image_dce"]),
            EnsureChannelFirstd(keys=["image_adc", "image_dce"]),
            Orientationd(keys=["image_adc", 'image_dce'], axcodes="RAS"),
            Resized(keys=["image_adc"], spatial_size=(64, 64, 16)),
            Resized(keys=["image_dce"], spatial_size=(96, 96, 32)),
            
            NormalizeIntensityd(keys=["image_adc", "image_dce"], nonzero=True, channel_wise=True),
            
            RandFlipd( keys=["image_adc", ], spatial_axis=[0], prob=0.50),
            RandFlipd( keys=["image_adc", ], spatial_axis=[1], prob=0.50),
            RandFlipd( keys=["image_adc", ], spatial_axis=[2], prob=0.50),
            
            RandFlipd( keys=["image_dce", ], spatial_axis=[0], prob=0.50),
            RandFlipd( keys=["image_dce", ], spatial_axis=[1], prob=0.50),
            RandFlipd( keys=["image_dce", ], spatial_axis=[2], prob=0.50),
            
            RandRotate90d(keys=["image_adc", 'image_dce'], prob=0.50, max_k=3 ),
            RandShiftIntensityd( keys=["image_adc", 'image_dce'], offsets=0.10, prob=0.50),
            
            ToTensord(keys=['image_adc', 'image_dce','clinical',  'label'])
        ]
    )

val_transforms = Compose(
        [
            LoadImaged(keys=["image_adc",'image_dce' ]),
            EnsureChannelFirstd(keys=["image_adc", 'image_dce']),
            Orientationd(keys=["image_adc",'image_dce'], axcodes="RAS"),
            Resized(keys=["image_adc"], spatial_size=(64, 64, 16)),
            Resized(keys=["image_dce"], spatial_size=(96, 96, 32)),
            
            NormalizeIntensityd(keys=["image_adc", 'image_dce'], nonzero=True, channel_wise=True),
            ToTensord(keys=['image_adc', 'image_dce','clinical', 'label'])
        ]
    )

cache_rate = 0.1
train_ds = CacheDataset(data=train_dict, transform=train_transforms, cache_rate=cache_rate, num_workers=12)
val_ds = CacheDataset(data=val_dict, transform=val_transforms, cache_rate=cache_rate, num_workers=12)
trainval_ds = CacheDataset(data=train_dict, transform=val_transforms, cache_rate=cache_rate, num_workers=12)
# create a training data loader
train_loader = DataLoader(train_ds, batch_size=24, shuffle=True, num_workers=16, pin_memory=True)

# create a validation data loader
val_loader = DataLoader(val_ds, batch_size=24, num_workers=8, pin_memory=True)
trainval_loader = DataLoader(trainval_ds, batch_size=24, num_workers=8, pin_memory=True)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = densenet121(pretrained=False)
        encoder_layers = list(base_model.children())

        self.backbone = nn.Sequential(*encoder_layers[:1])  #[:9] for resnet50
                        
    def forward(self, x):
        return self.backbone(x)

class MIL_Atn_Network(nn.Module):
    def __init__(self, num_classes):
        super(MIL_Atn_Network, self).__init__()
        # import timm
        # self.base_model = timm.create_model('resnet18', pretrained=True,  in_chans=1)
        # imagenet pretrain
        # base_model = torchvision.models.resnet34(weights=False)
        # state_dict = torch.load('resnet34-b627a593.pth')
        # base_model.load_state_dict(state_dict)
        # self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
#         self.fc = torch.nn.Linear(base_model.fc.in_features, num_classes)
#         self.atn_fc = nn.Linear(base_model.fc.in_features, 1)  # 全连接层用于计算注意力权重
        
        # radimagenet pretrain
        backbone = Backbone()
        backbone.load_state_dict(torch.load("RadImageNet_pytorch/DenseNet121.pt"))
        self.feature_extractor = backbone
        self.fc = torch.nn.Linear(1024, num_classes)  # 2048 for resnet
        self.atn_fc = nn.Linear(1024, 1)  # 全连接层用于计算注意力权重
        

    def  forward(self, x, x1, cln):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)  # 变成3通道
        
        x = x.permute(0, 4, 1, 2, 3)  
        batch_size, patch_z, channel, patch_x, patch_y  = x.size()
        
        x = x.reshape(batch_size*patch_z, channel, patch_x, patch_y) # [batch*patch_z*channel, patch_x, patch_y] # channel = 1
        x = self.feature_extractor(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)   # *！！仅仅对densenet121有效
        
        x = x.view(batch_size, patch_z, -1)
        attention_scores = torch.sigmoid(self.atn_fc(x)) # [batch_size, num_small_images, 1]
        x = torch.sum(x * attention_scores, dim=1)
       
        x = self.fc(x)

        return x
     
    
class MIL_LSTM_Network(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_classes):
        super(MIL_LSTM_Network, self).__init__()

        backbone = Backbone()
        backbone.load_state_dict(torch.load("RadImageNet_pytorch/DenseNet121.pt"))
        self.feature_extractor = backbone
        
        self.lstm = nn.LSTM(1024, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def  forward(self, x, x1, cln):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)  # 变成3通道
        
        x = x.permute(0, 4, 1, 2, 3)  
        batch_size, patch_z, channel, patch_x, patch_y  = x.size()
        
        x = x.reshape(-1, channel, patch_x, patch_y)
        x = self.feature_extractor(x)
                  
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  
        
        x = x.view(batch_size, patch_z, -1)        # [batch_size, num_small_images, channels]
        x, _ = self.lstm(x)                 # [batch, num_small_images, hidden_dim]
        x = x[:, -1, :]                                     # 取LSTM的最后一层
        x = self.fc(x)

        return x

    
    
class MIL_monai_Network(nn.Module):
    def __init__(self, num_classes):
        super(MIL_monai_Network, self).__init__()
        self.model = milmodel.MILModel(num_classes=num_classes, pretrained=True, mil_mode='att') # att_trans_pyramid
        
    def forward(self, x, x1, cln):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)
        x = x.permute(0, 4, 1, 2, 3)
        return self.model(x)
   

    



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# model = MIL_Atn_Network(num_classes=2).to(device)
# model = MIL_LSTM_Network(hidden_dim=64, num_layers=4, num_classes=2).to(device)
model = MIL_monai_Network(num_classes=2).to(device)

post_pred = Compose([Activations(softmax=True)])
post_label = Compose([AsDiscrete(to_onehot=2)])

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 6e-4)
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
        input_dce, input_adc, input_clinical, labels = batch_data["image_dce"].to(device), batch_data['image_adc'].to(device), batch_data["clinical"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(input_dce, input_adc, input_clinical)
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
            # For validation
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)

            step2 = 0
            for val_data in val_loader:
                step2 += 1
                val_dce, val_adc, val_clinical, val_labels = val_data["image_dce"].to(device),val_data["image_adc"].to(device), val_data["clinical"].to(device), val_data["label"].to(device)
                val_output = model(val_dce, val_adc, val_clinical)
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
            
            # For Train
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)

            step2 = 0
            for val_data in trainval_loader:
                step2 += 1
                val_dce, val_adc, val_clinical, val_labels = val_data["image_dce"].to(device),val_data["image_adc"].to(device), val_data["clinical"].to(device), val_data["label"].to(device)
                val_output = model(val_dce, val_adc, val_clinical)
                y_pred = torch.cat([y_pred, val_output], dim=0)
                y = torch.cat([y, val_labels], dim=0)
                val_loss += loss_function(val_output, val_labels).item()

            val_loss /= step2
            writer.add_scalar('loss/val_loss', val_loss, epoch)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            trainval_acc_metric = acc_value.sum().item() / len(acc_value)
            y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            trainval_auc_result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot

            tot_metric = 2*auc_result+acc_metric+trainval_acc_metric+trainval_auc_result

            if tot_metric > best_metric:
                best_metric = tot_metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "model_{}_{:.4f}_{:.4f}.pth".format(epoch, acc_metric, auc_result))
                print("saved new best metric model")
            print(
                    "current epoch: {} train_acc:{:.4f}, train_AUC: {:.4f}, val_acc: {:.4f} val_AUC: {:.4f} best val_acc: {:.4f} at epoch {}".format(
                    epoch + 1,trainval_acc_metric, trainval_auc_result,  acc_metric, auc_result, best_metric, best_metric_epoch
                )
            )
            writer.add_scalar('metric/val_Acc', acc_metric, epoch)
            writer.add_scalar('metric/val_AUC', auc_result, epoch)

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")



