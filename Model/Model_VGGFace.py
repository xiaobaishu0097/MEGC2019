import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet18_pt_mcn', 'Flow_Part_npic', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

res18_model_name = r'resnet18-5c106cde.pth'

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet_Removed(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet_Removed, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        top = x[:, :, :3]
        bottom = x[:, :, 3:]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        top = self.avgpool(top)
        bottom = self.avgpool(bottom)

        top = top.view(top.size(0), -1)
        bottom = bottom.view(bottom.size(0), -1)

        return x, top, bottom

class Resnet18_pt_mcn(nn.Module):

    def __init__(self):
        super(Resnet18_pt_mcn, self).__init__()
        self.meta = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225],
                     'imageSize': [224, 224]}
        self.features_0 = nn.Conv2d(3, 64, kernel_size=[7, 7], stride=(2, 2), padding=(3, 3), bias=False)
        self.features_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.features_2 = nn.ReLU()
        self.features_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), ceil_mode=False)
        self.features_4_0_conv1 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_4_0_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.features_4_0_relu = nn.ReLU()
        self.features_4_0_conv2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_4_0_bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.features_4_0_id_relu = nn.ReLU()
        self.features_4_1_conv1 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_4_1_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.features_4_1_relu = nn.ReLU()
        self.features_4_1_conv2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_4_1_bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.features_4_1_id_relu = nn.ReLU()
        self.features_5_0_conv1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(2, 2), padding=(1, 1), bias=False)
        self.features_5_0_bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.features_5_0_relu = nn.ReLU()
        self.features_5_0_conv2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_5_0_bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.features_5_0_downsample_0 = nn.Conv2d(64, 128, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.features_5_0_downsample_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.features_5_0_id_relu = nn.ReLU()
        self.features_5_1_conv1 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_5_1_bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.features_5_1_relu = nn.ReLU()
        self.features_5_1_conv2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_5_1_bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.features_5_1_id_relu = nn.ReLU()
        self.features_6_0_conv1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(2, 2), padding=(1, 1), bias=False)
        self.features_6_0_bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.features_6_0_relu = nn.ReLU()
        self.features_6_0_conv2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_6_0_bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.features_6_0_downsample_0 = nn.Conv2d(128, 256, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.features_6_0_downsample_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.features_6_0_id_relu = nn.ReLU()
        self.features_6_1_conv1 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_6_1_bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.features_6_1_relu = nn.ReLU()
        self.features_6_1_conv2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_6_1_bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.features_6_1_id_relu = nn.ReLU()
        self.features_7_0_conv1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(2, 2), padding=(1, 1), bias=False)
        self.features_7_0_bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.features_7_0_relu = nn.ReLU()
        self.features_7_0_conv2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_7_0_bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.features_7_0_downsample_0 = nn.Conv2d(256, 512, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.features_7_0_downsample_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.features_7_0_id_relu = nn.ReLU()
        self.features_7_1_conv1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_7_1_bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.features_7_1_relu = nn.ReLU()
        self.features_7_1_conv2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_7_1_bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.features_7_1_id_relu = nn.ReLU()
        self.features_8 = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0, ceil_mode=False, count_include_pad=False)
        self.classifier_0 = nn.Linear(in_features=512, out_features=1000, bias=True)

    def forward(self, data):
        features_0 = self.features_0(data)
        features_1 = self.features_1(features_0)
        features_2 = self.features_2(features_1)
        features_3 = self.features_3(features_2)
        features_4_0_conv1 = self.features_4_0_conv1(features_3)
        features_4_0_bn1 = self.features_4_0_bn1(features_4_0_conv1)
        features_4_0_relu = self.features_4_0_relu(features_4_0_bn1)
        features_4_0_conv2 = self.features_4_0_conv2(features_4_0_relu)
        features_4_0_bn2 = self.features_4_0_bn2(features_4_0_conv2)
        features_4_0_merge = torch.add(features_3, 1, features_4_0_bn2)
        features_4_0_id_relu = self.features_4_0_id_relu(features_4_0_merge)
        features_4_1_conv1 = self.features_4_1_conv1(features_4_0_id_relu)
        features_4_1_bn1 = self.features_4_1_bn1(features_4_1_conv1)
        features_4_1_relu = self.features_4_1_relu(features_4_1_bn1)
        features_4_1_conv2 = self.features_4_1_conv2(features_4_1_relu)
        features_4_1_bn2 = self.features_4_1_bn2(features_4_1_conv2)
        features_4_1_merge = torch.add(features_4_0_id_relu, 1, features_4_1_bn2)
        features_4_1_id_relu = self.features_4_1_id_relu(features_4_1_merge)
        features_5_0_conv1 = self.features_5_0_conv1(features_4_1_id_relu)
        features_5_0_bn1 = self.features_5_0_bn1(features_5_0_conv1)
        features_5_0_relu = self.features_5_0_relu(features_5_0_bn1)
        features_5_0_conv2 = self.features_5_0_conv2(features_5_0_relu)
        features_5_0_bn2 = self.features_5_0_bn2(features_5_0_conv2)
        features_5_0_downsample_0 = self.features_5_0_downsample_0(features_4_1_id_relu)
        features_5_0_downsample_1 = self.features_5_0_downsample_1(features_5_0_downsample_0)
        features_5_0_merge = torch.add(features_5_0_downsample_1, 1, features_5_0_bn2)
        features_5_0_id_relu = self.features_5_0_id_relu(features_5_0_merge)
        features_5_1_conv1 = self.features_5_1_conv1(features_5_0_id_relu)
        features_5_1_bn1 = self.features_5_1_bn1(features_5_1_conv1)
        features_5_1_relu = self.features_5_1_relu(features_5_1_bn1)
        features_5_1_conv2 = self.features_5_1_conv2(features_5_1_relu)
        features_5_1_bn2 = self.features_5_1_bn2(features_5_1_conv2)
        features_5_1_merge = torch.add(features_5_0_id_relu, 1, features_5_1_bn2)
        features_5_1_id_relu = self.features_5_1_id_relu(features_5_1_merge)
        features_6_0_conv1 = self.features_6_0_conv1(features_5_1_id_relu)
        features_6_0_bn1 = self.features_6_0_bn1(features_6_0_conv1)
        features_6_0_relu = self.features_6_0_relu(features_6_0_bn1)
        features_6_0_conv2 = self.features_6_0_conv2(features_6_0_relu)
        features_6_0_bn2 = self.features_6_0_bn2(features_6_0_conv2)
        features_6_0_downsample_0 = self.features_6_0_downsample_0(features_5_1_id_relu)
        features_6_0_downsample_1 = self.features_6_0_downsample_1(features_6_0_downsample_0)
        features_6_0_merge = torch.add(features_6_0_downsample_1, 1, features_6_0_bn2)
        features_6_0_id_relu = self.features_6_0_id_relu(features_6_0_merge)
        features_6_1_conv1 = self.features_6_1_conv1(features_6_0_id_relu)
        features_6_1_bn1 = self.features_6_1_bn1(features_6_1_conv1)
        features_6_1_relu = self.features_6_1_relu(features_6_1_bn1)
        features_6_1_conv2 = self.features_6_1_conv2(features_6_1_relu)
        features_6_1_bn2 = self.features_6_1_bn2(features_6_1_conv2)
        features_6_1_merge = torch.add(features_6_0_id_relu, 1, features_6_1_bn2)
        features_6_1_id_relu = self.features_6_1_id_relu(features_6_1_merge)
        features_7_0_conv1 = self.features_7_0_conv1(features_6_1_id_relu)
        features_7_0_bn1 = self.features_7_0_bn1(features_7_0_conv1)
        features_7_0_relu = self.features_7_0_relu(features_7_0_bn1)
        features_7_0_conv2 = self.features_7_0_conv2(features_7_0_relu)
        features_7_0_bn2 = self.features_7_0_bn2(features_7_0_conv2)
        features_7_0_downsample_0 = self.features_7_0_downsample_0(features_6_1_id_relu)
        features_7_0_downsample_1 = self.features_7_0_downsample_1(features_7_0_downsample_0)
        features_7_0_merge = torch.add(features_7_0_downsample_1, 1, features_7_0_bn2)
        features_7_0_id_relu = self.features_7_0_id_relu(features_7_0_merge)
        features_7_1_conv1 = self.features_7_1_conv1(features_7_0_id_relu)
        features_7_1_bn1 = self.features_7_1_bn1(features_7_1_conv1)
        features_7_1_relu = self.features_7_1_relu(features_7_1_bn1)
        features_7_1_conv2 = self.features_7_1_conv2(features_7_1_relu)
        features_7_1_bn2 = self.features_7_1_bn2(features_7_1_conv2)
        features_7_1_merge = torch.add(features_7_0_id_relu, 1, features_7_1_bn2)
        features_7_1_id_relu = self.features_7_1_id_relu(features_7_1_merge)
        features_8 = self.features_8(features_7_1_id_relu)
        classifier_flatten = features_8.view(features_8.size(0), -1)
        classifier_0 = self.classifier_0(classifier_flatten)
        return classifier_0

class Resnet18_pt_mcn_Removed(nn.Module):

    def __init__(self):
        super(Resnet18_pt_mcn_Removed, self).__init__()
        self.meta = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225],
                     'imageSize': [224, 224]}
        self.features_0 = nn.Conv2d(3, 64, kernel_size=[7, 7], stride=(2, 2), padding=(3, 3), bias=False)
        self.features_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.features_2 = nn.ReLU()
        self.features_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), ceil_mode=False)
        self.features_4_0_conv1 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_4_0_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.features_4_0_relu = nn.ReLU()
        self.features_4_0_conv2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_4_0_bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.features_4_0_id_relu = nn.ReLU()
        self.features_4_1_conv1 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_4_1_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.features_4_1_relu = nn.ReLU()
        self.features_4_1_conv2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_4_1_bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.features_4_1_id_relu = nn.ReLU()
        self.features_5_0_conv1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(2, 2), padding=(1, 1), bias=False)
        self.features_5_0_bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.features_5_0_relu = nn.ReLU()
        self.features_5_0_conv2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_5_0_bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.features_5_0_downsample_0 = nn.Conv2d(64, 128, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.features_5_0_downsample_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.features_5_0_id_relu = nn.ReLU()
        self.features_5_1_conv1 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_5_1_bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.features_5_1_relu = nn.ReLU()
        self.features_5_1_conv2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_5_1_bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.features_5_1_id_relu = nn.ReLU()
        self.features_6_0_conv1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(2, 2), padding=(1, 1), bias=False)
        self.features_6_0_bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.features_6_0_relu = nn.ReLU()
        self.features_6_0_conv2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_6_0_bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.features_6_0_downsample_0 = nn.Conv2d(128, 256, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.features_6_0_downsample_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.features_6_0_id_relu = nn.ReLU()
        self.features_6_1_conv1 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_6_1_bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.features_6_1_relu = nn.ReLU()
        self.features_6_1_conv2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_6_1_bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.features_6_1_id_relu = nn.ReLU()
        self.features_7_0_conv1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(2, 2), padding=(1, 1), bias=False)
        self.features_7_0_bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.features_7_0_relu = nn.ReLU()
        self.features_7_0_conv2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_7_0_bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.features_7_0_downsample_0 = nn.Conv2d(256, 512, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.features_7_0_downsample_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.features_7_0_id_relu = nn.ReLU()
        self.features_7_1_conv1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_7_1_bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.features_7_1_relu = nn.ReLU()
        self.features_7_1_conv2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_7_1_bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.features_7_1_id_relu = nn.ReLU()
        self.features_8 = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0, ceil_mode=False, count_include_pad=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, data):
        features_0 = self.features_0(data)
        features_1 = self.features_1(features_0)
        features_2 = self.features_2(features_1)
        features_3 = self.features_3(features_2)
        features_4_0_conv1 = self.features_4_0_conv1(features_3)
        features_4_0_bn1 = self.features_4_0_bn1(features_4_0_conv1)
        features_4_0_relu = self.features_4_0_relu(features_4_0_bn1)
        features_4_0_conv2 = self.features_4_0_conv2(features_4_0_relu)
        features_4_0_bn2 = self.features_4_0_bn2(features_4_0_conv2)
        features_4_0_merge = torch.add(features_3, 1, features_4_0_bn2)
        features_4_0_id_relu = self.features_4_0_id_relu(features_4_0_merge)
        features_4_1_conv1 = self.features_4_1_conv1(features_4_0_id_relu)
        features_4_1_bn1 = self.features_4_1_bn1(features_4_1_conv1)
        features_4_1_relu = self.features_4_1_relu(features_4_1_bn1)
        features_4_1_conv2 = self.features_4_1_conv2(features_4_1_relu)
        features_4_1_bn2 = self.features_4_1_bn2(features_4_1_conv2)
        features_4_1_merge = torch.add(features_4_0_id_relu, 1, features_4_1_bn2)
        features_4_1_id_relu = self.features_4_1_id_relu(features_4_1_merge)
        features_5_0_conv1 = self.features_5_0_conv1(features_4_1_id_relu)
        features_5_0_bn1 = self.features_5_0_bn1(features_5_0_conv1)
        features_5_0_relu = self.features_5_0_relu(features_5_0_bn1)
        features_5_0_conv2 = self.features_5_0_conv2(features_5_0_relu)
        features_5_0_bn2 = self.features_5_0_bn2(features_5_0_conv2)
        features_5_0_downsample_0 = self.features_5_0_downsample_0(features_4_1_id_relu)
        features_5_0_downsample_1 = self.features_5_0_downsample_1(features_5_0_downsample_0)
        features_5_0_merge = torch.add(features_5_0_downsample_1, 1, features_5_0_bn2)
        features_5_0_id_relu = self.features_5_0_id_relu(features_5_0_merge)
        features_5_1_conv1 = self.features_5_1_conv1(features_5_0_id_relu)
        features_5_1_bn1 = self.features_5_1_bn1(features_5_1_conv1)
        features_5_1_relu = self.features_5_1_relu(features_5_1_bn1)
        features_5_1_conv2 = self.features_5_1_conv2(features_5_1_relu)
        features_5_1_bn2 = self.features_5_1_bn2(features_5_1_conv2)
        features_5_1_merge = torch.add(features_5_0_id_relu, 1, features_5_1_bn2)
        features_5_1_id_relu = self.features_5_1_id_relu(features_5_1_merge)
        features_6_0_conv1 = self.features_6_0_conv1(features_5_1_id_relu)
        features_6_0_bn1 = self.features_6_0_bn1(features_6_0_conv1)
        features_6_0_relu = self.features_6_0_relu(features_6_0_bn1)
        features_6_0_conv2 = self.features_6_0_conv2(features_6_0_relu)
        features_6_0_bn2 = self.features_6_0_bn2(features_6_0_conv2)
        features_6_0_downsample_0 = self.features_6_0_downsample_0(features_5_1_id_relu)
        features_6_0_downsample_1 = self.features_6_0_downsample_1(features_6_0_downsample_0)
        features_6_0_merge = torch.add(features_6_0_downsample_1, 1, features_6_0_bn2)
        features_6_0_id_relu = self.features_6_0_id_relu(features_6_0_merge)
        features_6_1_conv1 = self.features_6_1_conv1(features_6_0_id_relu)
        features_6_1_bn1 = self.features_6_1_bn1(features_6_1_conv1)
        features_6_1_relu = self.features_6_1_relu(features_6_1_bn1)
        features_6_1_conv2 = self.features_6_1_conv2(features_6_1_relu)
        features_6_1_bn2 = self.features_6_1_bn2(features_6_1_conv2)
        features_6_1_merge = torch.add(features_6_0_id_relu, 1, features_6_1_bn2)
        features_6_1_id_relu = self.features_6_1_id_relu(features_6_1_merge)
        features_7_0_conv1 = self.features_7_0_conv1(features_6_1_id_relu)
        features_7_0_bn1 = self.features_7_0_bn1(features_7_0_conv1)
        features_7_0_relu = self.features_7_0_relu(features_7_0_bn1)
        features_7_0_conv2 = self.features_7_0_conv2(features_7_0_relu)
        features_7_0_bn2 = self.features_7_0_bn2(features_7_0_conv2)
        features_7_0_downsample_0 = self.features_7_0_downsample_0(features_6_1_id_relu)
        features_7_0_downsample_1 = self.features_7_0_downsample_1(features_7_0_downsample_0)
        features_7_0_merge = torch.add(features_7_0_downsample_1, 1, features_7_0_bn2)
        features_7_0_id_relu = self.features_7_0_id_relu(features_7_0_merge)
        features_7_1_conv1 = self.features_7_1_conv1(features_7_0_id_relu)
        features_7_1_bn1 = self.features_7_1_bn1(features_7_1_conv1)
        features_7_1_relu = self.features_7_1_relu(features_7_1_bn1)
        features_7_1_conv2 = self.features_7_1_conv2(features_7_1_relu)
        features_7_1_bn2 = self.features_7_1_bn2(features_7_1_conv2)
        features_7_1_merge = torch.add(features_7_0_id_relu, 1, features_7_1_bn2)
        features_7_1_id_relu = self.features_7_1_id_relu(features_7_1_merge)

        top = features_7_1_id_relu[:, :, :3]
        bottom = features_7_1_id_relu[:, :, 3:]

        x = self.features_8(features_7_1_id_relu)
        x = x.view(x.size(0), -1)

        top = self.avgpool(top)
        bottom = self.avgpool(bottom)

        top = top.view(top.size(0), -1)
        bottom = bottom.view(bottom.size(0), -1)

        return x, top, bottom

class FC_3(nn.Module):
    def __init__(self, block, num_classes=3):
        super(FC_3, self).__init__()
        self.fc_3 = nn.Sequential(
            nn.Linear(512 * block.expansion, 64),
            nn.Tanh(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.fc_3(x)

        return x

class Classifier(nn.Module):
    def __init__(self, input_classes=3, num_classes=3):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(64 * input_classes, num_classes)

    def forward(self, x):
        x = self.fc(x)

        return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet18_pt_mcn(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Resnet18_pt_mcn()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model

def Flow_Part_npic(num_pic=3, num_classes=3, **kwargs):

    model = {}

    model['resnet'] = Resnet18_pt_mcn_Removed()
    model['fc_top'] = FC_3(BasicBlock)
    model['fc_bottom'] = FC_3(BasicBlock)
    model['classifier'] = Classifier(input_classes=num_pic, num_classes=num_classes)
    model['classifier_top'] = Classifier(input_classes=1, num_classes=num_classes)
    model['classifier_bottom'] = Classifier(input_classes=1, num_classes=num_classes)

    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model