import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchsummary import summary

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha 
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    
class SNET(nn.Module):
    def __init__(self,channels,reduction = 16):
        super(SNET,self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels,channels//reduction),
            nn.ReLU(),
            nn.Linear(channels//reduction,channels),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_ = x.size()
        y = self.pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x*y
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True, downsample=False, bottleneck=True):
        super().__init__()
        stride = 2 if downsample else 1
        self.use_se = use_se
        self.bottleneck = bottleneck
        mid_channels = out_channels // 4 if bottleneck else out_channels
        if bottleneck:
            self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
            self.bn1 = nn.BatchNorm2d(mid_channels)
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(mid_channels)
            self.conv3 = nn.Conv2d(mid_channels,out_channels,kernel_size = 1,stride = 1,bias = False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if downsample or in_channels != out_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.identity = nn.Identity()

        if use_se:
            self.se = SNET(out_channels)

    def forward(self, x):
        identity = self.identity(x)
        if self.bottleneck:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
        else:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        if self.use_se:
            out = self.se(out)
        out += identity
        return self.relu(out)


class CNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        )
        self.res1 = nn.Sequential(
                    ResidualBlock(64,128,downsample = True),
                    ResidualBlock(128,128))
        self.res2 = nn.Sequential(
                    ResidualBlock(128,256,downsample = True),
                    ResidualBlock(256,256))
        self.res3 = nn.Sequential(
                    ResidualBlock(256,512,downsample = True),
                    ResidualBlock(512,512)) 
        self.res4 = nn.Sequential(
                    ResidualBlock(512,1024,downsample = True),
                    ResidualBlock(1024,1024))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(1024,256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,num_classes)
        )
    def get_summary(model,in_channels,img_size):
        return summary(model, input_size=(in_channels,img_size,img_size))
    def forward(self,x):
        x = self.initial(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
    