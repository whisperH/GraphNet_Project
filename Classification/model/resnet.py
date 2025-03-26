import torch
import torch.nn as nn
import logging
import math
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from Classification.model.layers import CRF


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
from torchvision import models
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

    def __init__(self, block, layers, num_classes=1, num_nodes=16,
                 use_crf=True, use_levels=[0], level_crf=False, concat_level_feats=False, cfg=None):
        """Constructs a ResNet model.

        Args:
            num_classes: int, since we are doing binary classification
                (tumor vs normal), num_classes is set to 1 and sigmoid instead
                of softmax is used later
            num_nodes: int, number of nodes/patches within the fully CRF
            use_crf: bool, use the CRF component or not
        """
        self.inplanes = 64
        self.cfg = cfg
        self.num_classes = num_classes
        self.use_levels = use_levels
        self.level_crf = level_crf
        self.concat_level_feats = concat_level_feats
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        if self.concat_level_feats:
            self.fc = nn.Linear(512 * block.expansion * len(self.use_levels), num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.crf = CRF(num_nodes, num_classes=num_classes) if use_crf else None
        self.level_crf_model = CRF(num_nodes*len(use_levels), num_classes=num_classes) if level_crf else None
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: 5D tensor with shape of
            [batch_size, grid_size, 3, crop_size, crop_size],
            where grid_size is the number of patches within a grid (e.g. 9 for
            a 3x3 grid); crop_size is 224 by default for ResNet input;

        Returns:
            logits, 2D tensor with shape of [batch_size, grid_size], the logit
            of each patch within the grid being tumor
        """
        # x---->b,level,grid,c,h,w---->3,2,9,3,224,224

        # logging.info("00000x.shape---->",x.shape)
        batch_size, num_levels, grid_size, _, crop_size = x.shape[0:5]

        # flatten grid_size dimension and combine it into batch dimension
        x = x.view(-1, 3, crop_size, crop_size) # 54,3,224,224
        # logging.info("x.shape---->",x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) # torch.Size([54, 512, 1, 1])

        if self.concat_level_feats: # concat特征后多个level合并为一个特征
            feats = x.view(batch_size*grid_size, -1)# torch.Size([27, 512])
        else: # 得到所有level的所有grid的特征
            feats = x.view(x.size(0), -1)# torch.Size([54, 512])
        logits = self.fc(feats) # torch.Size([54, 1]) 得到所有grid的logit
        # restore grid_size dimension for CRF
        if self.level_crf:
            feats = feats.view((batch_size, grid_size*len(self.use_levels), -1)) # torch.Size([3, 18, 512])
            logits = logits.view((batch_size, grid_size*len(self.use_levels), -1))# torch.Size([3, 18, 1])
        else:
            if self.concat_level_feats:
                feats = feats.view((batch_size, grid_size, -1)) # torch.Size([3, 9, 512])
                logits = logits.view((batch_size, grid_size, -1))# torch.Size([3, 9, 1])
            else:
                feats = feats.view((batch_size*num_levels, grid_size, -1)) # torch.Size([6, 9, 512])
                logits = logits.view((batch_size*num_levels, grid_size, -1))# torch.Size([6, 9, 1])

        if self.crf:
            logits = self.crf(feats, logits)

        if self.level_crf:
            logits = self.level_crf_model(feats, logits)

        return logits, feats


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        print("Loading PreTrain resnet18 ckpt......")
        state_dict = load_state_dict_from_url(model_urls['resnet18'],
                                              progress=True)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and "fc" not in k}
        model.load_state_dict(state_dict, False)
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print("Loading PreTrain resnet34 ckpt......")
        state_dict = load_state_dict_from_url(model_urls['resnet34'],
                                              progress=True)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and "fc" not in k}
        model.load_state_dict(state_dict, False)
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print("Loading PreTrain resnet50 ckpt......")
        state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=True)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and "fc" not in k}
        model.load_state_dict(state_dict, False)
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        print("Loading PreTrain resnet101 ckpt......")
        state_dict = load_state_dict_from_url(model_urls['resnet101'],
                                              progress=True)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and "fc" not in k}
        model.load_state_dict(state_dict, False)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model