from collections import OrderedDict, Mapping
import inspect
import itertools
import copy
import typing
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as tm
import torchvision.models.detection as tmdet
import torchvision.models.segmentation as tmseg 

# try:
#     from .timm import create_model

# from torch.jit.annotations import Tuple, List, Dict

## typing has variables like Tuple, List, Dict, etc. It's conflict with torch.jit.annotations.
## So don't use typing or explicitily use typing.Tuple
# from typing import *

def deep_update(d, s):
    """ Deep copy/replace dictionary s into dictionary d. """
    for k in s:
        if k in d and isinstance(d[k], Mapping) and isinstance(s[k], Mapping):
            deep_update(d[k], s[k])
        else:
            d[k] = copy.deepcopy(s[k])


def trace_layer(x, path):
    """ Trace layer in a module from given path. 
        x: nn.Module
        path: list of key or index: ['conv', 1, -1,]
    """
    for _ in path:
        if isinstance(_, str):
            x = getattr(x, _)
        elif isinstance(_, int):
            x = x[_]
    return x


def unlist_module(x): 
    res = []
    for k, v in x.named_children():
        for k0, _ in unlist_module(v):
            res.append([k+'.'+k0, _])
    if len(res) == 0:
        res.append(['', x])
    
    return res


def get_norm_layer_and_bias(norm_layer='batch', use_bias=None):
    """ Return a normalization layer and set up use_bias for convoluation layers.
    
    Parameters:
        norm_layer: (str) -- the name of the normalization layer: [batch, instance]
                    None -- no batch norm
                    other module: nn.BatchNorm2d, nn.InstanceNorm2d

    For BatchNorm: use learnable affine parameters. (affine=True)
                   track running statistics (mean/stddev). (track_running_stats=True)
                   do not use bias in previous convolution layer. (use_bias=False)
    For InstanceNorm: do not use learnable affine parameters. (affine=False)
                      do not track running statistics. (track_running_stats=False)
                      use bias in previous convolution layer. (use_bias=True)
    Test commands:
        get_norm_layer_and_bias('batch', None) -> affine=True, track_running_stats=True, False
        get_norm_layer_and_bias('batch', True) -> affine=True, track_running_stats=True, True
        get_norm_layer_and_bias('instance', None) -> affine=False, track_running_stats=False, True
        get_norm_layer_and_bias('instance', False) -> affine=False, track_running_stats=False, False
        get_norm_layer_and_bias(None, None) -> None, True
        get_norm_layer_and_bias(None, False) -> None, False
        get_norm_layer_and_bias(nn.BatchNorm2d, None) -> BatchNorm2d, False
        get_norm_layer_and_bias(nn.BatchNorm2d, True) -> BatchNorm2d, True
        get_norm_layer_and_bias(nn.InstanceNorm2d, None) -> InstanceNorm2d, True
        get_norm_layer_and_bias(nn.InstanceNorm2d, False) -> InstanceNorm2d, False
    """
    if isinstance(norm_layer, str):
        if norm_layer == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_layer == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError('normalization layer {} is not found'.format(norm_layer))
    
    if use_bias is None:
        use_bias = norm_layer == nn.InstanceNorm2d
    
    return norm_layer, use_bias


class SharedLinear(nn.Linear):
    def __init__(self, in_features, out_features, share_weight=False):
        super(SharedLinear, self).__init__(in_features, out_features, bias=True)
        if share_weight:
            self.weight = nn.Parameter(torch.Tensor(1, in_features))
        self.reset_parameters()
    
    def forward(self, x):
        return F.linear(x, self.weight) + self.bias
        # return F.linear(x, self.weight, self.bias)


class Conv2d(nn.Conv2d):
    """ Make nn.Conv2d support tf padding="same" and padding="valid" option. """
    def conv2d_forward(self, input, weight):
        """ pytorch v1.4. """
        if self.padding == 'same':
            padding = [max(0, k-s) for k, s in zip(self.kernel_size, self.stride)]
            padding = (padding[1]//2, (padding[1] + 1)//2,
                       padding[0]//2, (padding[0] + 1)//2)
            return F.conv2d(F.pad(input, padding, mode='constant'),
                            weight, self.bias, self.stride,
                            0, self.dilation, self.groups)
        elif self.padding == 'valid':
            return F.conv2d(input, weight, self.bias, self.stride,
                            0, self.dilation, self.groups)
        elif self.padding == 'default':
            padding = tuple((k-1)//2 for k in self.kernel_size)
            return F.conv2d(input, weight, self.bias, self.stride,
                            padding, self.dilation, self.groups)

        return super(Conv2d, self).conv2d_forward(input, weight)

    def _conv_forward(self, input, weight):
        """ pytorch 1.5 """
        return self.conv2d_forward(input, weight)


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, size=None): # size:typing.Optional[int]=None
        "Output will be 2*size or 2 if size is None"
        super(AdaptiveConcatPool2d, self).__init__()
        self.output_size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)
    
    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], dim=1)


class ConvBNReLU(nn.Sequential):
    def __init__(self, conv, norm_layer=None, activation=None, dropout_rate=0.0):
        ## get norm layer:
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)
        
        layers = [conv]
        if norm_layer is not None:
            layers.append(norm_layer(conv.out_channels))
        if activation is not None:
            layers.append(activation)
        if dropout_rate:
            layers.append(nn.Dropout2d(dropout_rate))
        
        super(ConvBNReLU, self).__init__(*layers)


class Conv2dBNReLU(ConvBNReLU):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding='default', dilation=1, groups=1, bias=None, 
                 norm_layer='batch', activation=nn.ReLU(inplace=True), 
                 dropout_rate=0.0, 
                ):
        """ Create a Conv2d->BN->ReLU layer. 
            norm_layer: batch, instance, None
            activation: a nn layer.
            padding: 
                'default' (default): torch standard symmetric padding with (kernel_size - 1) // 2.
                int: symmetric padding to pass to nn.Conv2d(padding=padding)
                "same": tf padding="same", asymmetric for even kernel (l_0, r_1), etc)
                "valid": tf padding="valid", same as padding=0
        """
        ## get norm layer:
        norm_layer, bias = get_norm_layer_and_bias(norm_layer, bias)
        ## use Conv2d (extended nn.Conv2d) to support padding options
        conv = Conv2d(in_channels, out_channels, kernel_size, stride, 
                      padding, dilation, groups, bias=bias)
        super(Conv2dBNReLU, self).__init__(conv, norm_layer, activation, dropout_rate)


class DepthwiseConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding='default', dilation=1, groups=None, bias=None, expand_ratio=1, 
                 norm_layer='batch', activation=nn.ReLU(inplace=True)):
        """ Depthwise/Group convolution. 
            directly set bias based on norm_layer
        """
        inner_channels = int(expand_ratio * in_channels)
        groups = groups or int(np.gcd(in_channels, inner_channels))
        norm_layer, bias = get_norm_layer_and_bias(norm_layer, bias)
        
        super(DepthwiseConv2d, self).__init__(
            Conv2dBNReLU(in_channels, inner_channels, kernel_size, stride, 
                         padding, dilation, groups=groups, bias=bias, 
                         norm_layer=norm_layer, activation=activation),
            nn.Conv2d(inner_channels, out_channels, kernel_size=1, stride=1, 
                      padding=0, dilation=1, groups=1, bias=bias),
        )
        
        ## register values
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups


class InvertedResidual(nn.Module):
    """ https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding='default', dilation=1, groups=None, bias=None, expand_ratio=1,
                 norm_layer='batch', activation=nn.ReLU(inplace=True)):
        super(InvertedResidual, self).__init__()
        inner_channels = int(round(expand_ratio * in_channels))
        norm_layer, bias = get_norm_layer_and_bias(norm_layer, bias)
        assert stride in [1, 2]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups or inner_channels
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2dBNReLU(
                in_channels, inner_channels, kernel_size=1, # stride=1, padding='default' (0 for ks=1), 
                bias=bias, norm_layer=norm_layer, activation=activation))
        layers.extend([
            # dw
            Conv2dBNReLU(inner_channels, inner_channels, kernel_size=kernel_size, 
                         stride=stride, padding=padding, groups=self.groups, 
                         bias=bias, norm_layer=norm_layer, activation=activation),
            # pw-linear
            nn.Conv2d(inner_channels, out_channels, kernel_size=1, bias=bias), # stride=1, padding='default' (0 for ks=1), 
            norm_layer(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ResizeConv2d(nn.Sequential):
    """ Upsampling layer with resize convolution. """
    def __init__(self, in_channels, out_channels, scale_factor=2, 
                 conv=(Conv2d, {'kernel_size': 3, 'padding': 'default'}), mode='bilinear'):
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        super(ResizeConv2d, self).__init__(
            nn.Upsample(scale_factor=scale_factor, mode=mode),            
            conv[0](in_channels, out_channels, **conv[1]),
        )


class SubPixelConv2d(nn.Sequential):
    """ Upsampling layer with better modelling power. 
        Sub-pixel convolution usually gives better result than resize convolution.
        Use incr init (and weight_norm) to avoid checkboard artifact. 
        https://arxiv.org/pdf/1707.02937.pdf
        May combine with (https://arxiv.org/pdf/1806.02658.pdf):
            nn.Sequential(
                nn.LeakyReLU(inplace=True),
                nn.ReplicationPad2d((1,0,1,0)),
                nn.AvgPool2d(2, stride=1),
            )
        to generate non-checkboard artifact image.
    """
    def __init__(self, in_channels, out_channels, scale_factor=2, 
                 conv=(Conv2d, {'kernel_size': 3, 'padding': 'default'})):
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # torch.nn.utils.weight_norm()
        super(SubPixelConv2d, self).__init__(
            conv[0](in_channels, out_channels * scale_factor ** 2, **conv[1]),
            nn.PixelShuffle(scale_factor),
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        self.icnr_(self[0].weight)
    
    def icnr_(self, x):
        """ ICNR init of conv weight. """
        ni, nf, h, w = x.shape
        ni2 = int(ni / (self.scale_factor**2))
        k = nn.init.kaiming_normal_(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
        k = k.contiguous().view(ni2, nf, -1)
        k = k.repeat(1, 1, self.scale_factor**2)
        k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
        x.data.copy_(k)


##************************* Backbones *************************
## https://github.com/pytorch/pytorch/issues/21064 about extract intermediate layers

## return_layers = {'maxpool': '0', 'layer1': '1', 'layer2': '2', 'layer3': '3', 'layer4': '4'}
## tm._utils.IntermediateLayerGetter(backbone, return_layers)
class ResNetFeatures(nn.Module):
    """ torchvision ResNet feature layers.
        pytorch resnet structures:
            conv1: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            bn1: BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            relu: ReLU(inplace=True)
            maxpool: MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            layer1: Sequential(Bottleneck/Basic) # 256
            layer2: Sequential(Bottleneck/Basic) # 512
            layer3: Sequential(Bottleneck/Basic) # 1024
            layer4: Sequential(Bottleneck/Basic) # 2048
            avgpool: AdaptiveAvgPool2d(output_size=(1, 1))
            fc: Linear(in_features=2048, out_features=1000, bias=True)
    """
    def __init__(self, architecture, pretrained=False, progress=False, 
                 in_channels=None, maxpool=None, **kwargs):
        """ Get feature layers on different scale from resnet backbone.
            maxpool: current resnet. maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                     can use nn.Identity().
            **kwargs parameters:
                replace_stride_with_dilation [False, False, Flase]: 
                    replace stride with dilation in feature layers.
                zero_init_residual: False, 
                    This improves the model by 0.2~0.3% according to 
                    https://arxiv.org/abs/1706.02677
                norm_layer: None, nn.BatchNorm2d, use FrozenBatchNorm2d for FasterRCNN
                num_classes: num_classes in final FC, not used for Backbone.
        """
        # __RESNET__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
        #               'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
        #               'wide_resnet50_2', 'wide_resnet101_2']
        # assert architecture in __RESNET__
        
        super(ResNetFeatures, self).__init__()
        self.architecture = architecture
        self.backbone = getattr(tm, architecture)(pretrained, progress, **kwargs)
        ## remove last 2 layers
        self.backbone = nn.Sequential(OrderedDict(list(self.backbone.named_children())[:-2]))
        self.init_layer = None
        self.feature_block = [tm.resnet.BasicBlock, tm.resnet.Bottleneck]
        self.out_channels = [64, 256, 512, 1024, 2048] # 64-2048 for resnet, self.backbone.inplanes
        
        ## define first conv layer
        if in_channels is not None:
            self.in_channels = in_channels
            if pretrained is True:
                print("pretrained conv1 layer is replaced with random weights. ")
            
            attrs = ['in_channels', 'out_channels', 'kernel_size', 'stride', 
                     'padding', 'dilation', 'groups', 'bias', 'padding_mode']
            args = dict([(_, getattr(self.backbone.conv1, _)) for _ in attrs])
            args.update({'in_channels': self.in_channels})
            self.backbone.conv1 = nn.Conv2d(**args)
        else:
            self.in_channels = self.backbone.conv1.in_channels
        
        ## define the pooling layer
        # default: nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # can use nn.Identity()
        if maxpool is not None:
            if maxpool == False:
                self.backbone.maxpool = nn.Identity()
            else:
                self.backbone.maxpool = maxpool
    
    def feature_channels(self, idx=None):
        if isinstance(idx, int):
            return self.out_channels[idx]
        if idx is None:
            idx = range(len(self.out_channels))
        return [self.out_channels[_] for _ in idx]
    
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x0 = self.backbone.relu(x)
        x = self.backbone.maxpool(x0)

        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        
        return [x0, x1, x2, x3, x4]


class UNetFeatures(nn.Module):
    def __init__(self, in_channels=3, n_channels=32, n_downsampling=4):
        super(UNetFeatures, self).__init__()
        self.conv, self.out_channels = [], []
        
        for i in range(n_downsampling):
            out_channels = 2**i * n_channels
            self.conv.append(Conv2dBNReLU(in_channels, out_channels, kernel_size=3))
            self.conv.append(Conv2dBNReLU(out_channels, out_channels, kernel_size=3))
            self.conv.append(nn.MaxPool2d(2))
            self.out_channels.append(out_channels)
            in_channels = out_channels
        
        out_channels = 2**n_downsampling * n_channels
        self.conv.append(Conv2dBNReLU(in_channels, out_channels, kernel_size=3))
        self.conv.append(Conv2dBNReLU(out_channels, out_channels, kernel_size=3))
        self.out_channels.append(out_channels)
        
        self.conv = nn.Sequential(*self.conv)
        self.return_layers = [3*i+2 for i in range(n_downsampling+1)]
    
    def feature_channels(self, idx=None):
        if isinstance(idx, int):
            return self.out_channels[idx]
        if idx is None:
            idx = range(len(self.out_channels))
        return [self.out_channels[_] for _ in idx]
    
    def forward(self, x):
        res = []
        for s, t in zip([0] + self.return_layers, self.return_layers):
            for _ in range(s, t):
                x = self.conv[_](x)
            res.append(x)
        
        return res


class DepthwiseConv2dBNReLU6(ConvBNReLU):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, 
                 expand_ratio=1, padding='default', groups=None, norm_layer='batch', 
                 activation=nn.ReLU6(inplace=True), dropout_rate=0.0):
        """ Depthwise convolution BN Relu6. 
            (put kernel_size after stride to make it callable for 
             mobilenet_v2 constructor).
        """
        norm_layer, bias = get_norm_layer_and_bias(norm_layer)
        conv = DepthwiseConv2d(in_channels, out_channels, kernel_size,
                               stride, padding, groups=groups, bias=bias,
                               norm_layer=norm_layer, activation=activation)
        
        super(DepthwiseConv2dBNReLU6, self).__init__(conv, norm_layer, activation, dropout_rate)

        
class MobileNetFeatures(nn.Module):
    def __init__(self, architecture='mobilenet_v2', pretrained=False, progress=False, 
                 in_channels=None, **kwargs):
        """ Get features on different scale from Mobilenet (v1, v2) backbone.
            **kwargs parameters:
                width_mult=1.0: adjusts number of channels in each layer by this amount
                setting=None: Network structure. (inverted_residual_setting)
                round_nearest=8: Round the number of channels in each layer to be a multiple of this number.
                block=None: InvertedResidual
                num_classes=1000: num_classes in final FC, not used for Backbone.
        """
        super(MobileNetFeatures, self).__init__()
        if architecture in ['mobilenet_v1', 'mobilenet_v2']:
            self.architecture = self.default_settings(architecture)
        elif architecture is None:
            self.architecture = {}
        else:
            self.architecture = architecture
        
        ## Update architecture with kwargs/defaults
        assert isinstance(self.architecture, dict)
        self.architecture.update(kwargs)
        self.architecture.setdefault('init_block', tm.mobilenet.ConvBNReLU)
        self.architecture.setdefault('width_mult', 1.0)
        self.architecture.setdefault('round_nearest', 8)
        self.architecture.setdefault('return_layers', None)
        assert 'setting' in self.architecture
        assert 'block' in self.architecture
        
        if self.architecture['return_layers'] is None:
            all_layers = np.cumsum([0] + [_[2] for _ in self.architecture['setting']]) + 1
            self.architecture['return_layers'] = all_layers.tolist()
        
        ## build mobilnet classifier
        self.init_block = self.architecture['init_block']
        self.feature_block = self.architecture['block']
        self.backbone = getattr(tm, 'mobilenet_v2')(
            pretrained, progress, 
            inverted_residual_setting=self.architecture['setting'], 
            block=self.architecture['block'],
            width_mult=self.architecture['width_mult'],
            round_nearest=self.architecture['round_nearest'],
        ).features
        
        ## switch channel for the first conv layer
        if in_channels is not None:
            self.in_channels = in_channels
            if pretrained is True:
                print("pretrained conv1 layer is replaced with random weights. ")
            attrs = ['in_channels', 'out_channels', 'kernel_size', 'stride', 
                     'padding', 'dilation', 'groups', 'bias', 'padding_mode']
            args = dict([(_, getattr(self.backbone[0][0], _)) for _ in attrs])
            args.update({'in_channels': self.in_channels})
            self.backbone[0][0] = nn.Conv2d(**args)
        else:
            self.in_channels = self.backbone[0][0].in_channels
        
        ## remove last ConvBNReLU from feature layers
        feature_layers = list(self.backbone.named_children())[:-1]
        ## Replace the init_block if non-default init_block is provided
        # if init_block is not a class (then should be a function)
        # or init_block is a class, and feature_layers [0][1] is not an instance of this class.
        
        if not (inspect.isclass(self.init_block) and isinstance(feature_layers[0][1], self.init_block)):
            init_name, init_conv = feature_layers[0][0], feature_layers[0][1][0]
            attrs = ['in_channels', 'out_channels', 'kernel_size', 'stride']
            args = dict([(_, getattr(init_conv, _)) for _ in attrs])
            args.update({'in_channels': self.in_channels})
            feature_layers[0] = (init_name, self.init_block(**args))
        self.backbone = nn.Sequential(OrderedDict(feature_layers))
        self.block_channels = [
            trace_layer(feature_layers[0][1], self.architecture['trace']['init_block']).out_channels,
            *[trace_layer(v, self.architecture['trace']['block']).out_channels for k, v in feature_layers[1:]],
        ]
        
        ## Retrive out_channels for all returned feature layers
        self.return_layers = self.architecture['return_layers']
        self.out_channels = [self.block_channels[_-1] for _ in self.return_layers]
#         self.out_channels = [
#             trace_layer(self.backbone, [_-1] + self.architecture['trace']).out_channels 
#             for _ in self.return_layers
#         ]
    
    @staticmethod
    def default_settings(architecture):
        init_block = tm.mobilenet.ConvBNReLU
        ## exact same as tm.mobilenet.InvertedResidual
        v2_block = (lambda in_c, out_c, stride=1, kernel_size=3, expand_ratio=1: 
                    InvertedResidual(
                        in_c, out_c, kernel_size, stride, padding='default',
                        expand_ratio=expand_ratio, norm_layer='batch', 
                        activation=nn.ReLU6(inplace=True))
                   )
        v1_block = DepthwiseConv2dBNReLU6

        settings = {
            'mobilenet_v2': {
                'setting': [
                    # t, c, n, s
                    [1, 16, 1, 1],
                    [6, 24, 2, 2],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1],
                ], 
                'block': v2_block,
                'init_block': init_block, 
                'return_layers': [2, 4, 7, 14, 18],
                'trace': {'init_block': [0], 'block': ['conv', -2]},
            }, 
            'mobilenet_v1': {
                'setting': [
                    # t, c, n, s
                    [1, 64, 1, 1],
                    [1, 128, 2, 2],
                    [1, 256, 2, 2],
                    [1, 512, 6, 2],
                    [1, 1024, 2, 2],
                ],
                'block': v1_block, 
                'init_block': init_block, 
                'return_layers': [2, 4, 6, 12, 14],
                'trace': {'init_block': [0], 'block': [0, -1]},
            }
        }
        
        return settings[architecture]
    
    def feature_channels(self, idx=None):
        if isinstance(idx, int):
            return self.out_channels[idx]
        if idx is None:
            idx = range(len(self.out_channels))
        return [self.out_channels[_] for _ in idx]
    
    def forward(self, x):
        res = []
        for s, t in zip([0] + self.return_layers, self.return_layers):
            for _ in range(s, t):
                x = self.backbone[_](x)
            res.append(x)
        
        return res


# https://github.com/pytorch/vision/blob/master/torchvision/models/detection/backbone_utils.py
class BackboneWithFPN(nn.Module):
    """ Adds a FPN on top of features. 
    Arguments:
        backbone (nn.Module): should return a list/dict of tensors, don't only return a tensor.
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """
    def __init__(self, backbone, out_channels=None, return_layers=None, in_channels=None):
        super(BackboneWithFPN, self).__init__()
        self.backbone = backbone
        self.return_layers = return_layers or OrderedDict({-1: '0'})
        self.featmap_names = list(self.return_layers.values())
        
        if in_channels is None:
            in_channels = self.backbone.feature_channels()
            in_channels = [in_channels[k] for k in self.return_layers]
        self.in_channels = in_channels
        # self.backbone = tm._utils.IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        if out_channels is None:
            self.out_channels = self.in_channels
            self.fpn = nn.Identity()
        else:
            self.out_channels = out_channels
            self.fpn = torchvision.ops.FeaturePyramidNetwork(
                in_channels_list=self.in_channels,
                out_channels=self.out_channels,
                extra_blocks=torchvision.ops.feature_pyramid_network.LastLevelMaxPool(),
                # LastLevelP6P7 for retina network.
            )
    
    def forward(self, x):
        # https://www.codeleading.com/article/1897780211/
        # write with for loop not constructor, "GeneratorExp aren't supported"
        features = self.backbone(x)
        r = OrderedDict()
        for k, v in self.return_layers.items():
            r[v] = features[k]
        
        x = self.fpn(r)
        return x

################################################################
################ Semantic Segmentation Models ##################
################################################################
class UNet(nn.Module):
    def __init__(self, num_classes, scale_factor=1, resize_mode='bilinear',
                 encoder={}, decoder={}, out_channels=None, **kwargs):
        """ Abstract UNet class. 
            num_classes: num_classes in output layer.
            scale_factor: resize to original size.
            encoder: nn.Module or dictionary. (call UNetfeatures(**encoder)).
            decoder: nn.Module or dictionary. (call self.default_decoder(**decoder)).
            out_channels: call encoder.feature_channels() if None
        """
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.resize_mode = resize_mode
        
        ## encoder
        if isinstance(encoder, dict):
            self.encoder = self.get_encoder(**encoder)
        else:
            self.encoder = encoder
        assert isinstance(self.encoder, nn.Module)
        
        ## out_channels
        self.out_channels = out_channels or self.encoder.feature_channels()
        
        ## decoder
        if isinstance(decoder, dict):
            self.decoder = self.get_decoder(**decoder)
        else:
            self.decoder = decoder
        assert isinstance(self.decoder, nn.Module)

        ## final classification and resize layer
        classifier = [
            nn.Conv2d(self.out_channels[0], num_classes, kernel_size=1),
            (nn.Softmax2d() if num_classes > 1 else nn.Sigmoid()),
        ]
        ## pytorch interpolate == tf interpolate != keras.Upsample/tf.js.Upsample.
        ## Will see differences on resize in keras and pytorch.
        if scale_factor is not None and scale_factor != 1:
            classifier = [
                nn.Upsample(scale_factor=scale_factor, mode=self.resize_mode)
            ] + classifier
        
        self.classifier = nn.Sequential(*classifier)
    
    def get_encoder(self, **kwargs):
        return UNetFeatures(**kwargs)
    
    def get_decoder(self, **kwargs):
        up = kwargs.setdefault('up', ResizeConv2d)
        
        decoder = nn.ModuleList()
        for in_c, out_c in zip(self.out_channels[-1:0:-1], self.out_channels[-2::-1]):
            modules = nn.ModuleDict([
                ('up', up(in_c, out_c, scale_factor=2)),
                ('conv', nn.Sequential(Conv2dBNReLU(out_c*2, out_c), Conv2dBNReLU(out_c, out_c))),
            ])
            decoder.append(modules)
        
        return decoder
    
    def forward(self, x):
        features = self.encoder(x)
        x = features.pop()
        for layers in self.decoder:
            up = layers['up'](x)
            up = torch.cat([up, features.pop()], dim=1)
            x = layers['conv'](up)
        
        return self.classifier(x)


class SubPixelDepthwiseConv2d(SubPixelConv2d):
    def __init__(self, in_channels, out_channels, scale_factor=2, 
                 kernel_size=3, padding='default', expand_ratio=1, norm_layer='batch', 
                 activation=nn.ReLU6(inplace=True)):
        args = {'kernel_size': kernel_size, 'expand_ratio': expand_ratio, 'padding': padding, 
                'norm_layer': norm_layer, 'activation': activation}
        super(SubPixelDepthwiseConv2d, self).__init__(
            in_channels, out_channels, scale_factor, (DepthwiseConv2d, args))
    
    def reset_parameters(self):
        self.icnr_(self[0][1].weight)


class SubPixelInvertedResidual(SubPixelConv2d):
    def __init__(self, in_channels, out_channels, scale_factor=2, 
                 kernel_size=3, padding='default', expand_ratio=1, norm_layer='batch', 
                 activation=nn.ReLU6(inplace=True)):
        ## tm.mobilenet.InvertedResidual(inp, oup, stride, expand_ratio)
        args = {'kernel_size': kernel_size, 'expand_ratio': expand_ratio, 'padding': padding, 
                'norm_layer': norm_layer, 'activation': activation}
        super(SubPixelInvertedResidual, self).__init__(
            in_channels, out_channels, scale_factor, (InvertedResidual, args))
    
    def reset_parameters(self):
        self.icnr_(self[0].conv[-2].weight)


class ResizeDepthwiseConv2d(ResizeConv2d):
    def __init__(self, in_channels, out_channels, scale_factor=2, 
                 kernel_size=3, padding='default', expand_ratio=1, norm_layer='batch', 
                 activation=nn.ReLU6(inplace=True), mode='bilinear'):
        args = {'kernel_size': kernel_size, 'expand_ratio': expand_ratio, 'padding': padding, 
                'norm_layer': norm_layer, 'activation': activation}
        super(ResizeDepthwiseConv2d, self).__init__(
            in_channels, out_channels, scale_factor, (DepthwiseConv2d, args), mode=mode)


class ResizeInvertedResidual(ResizeConv2d):
    def __init__(self, in_channels, out_channels, scale_factor=2, 
                 kernel_size=3, padding='default', expand_ratio=1, norm_layer='batch', 
                 activation=nn.ReLU6(inplace=True), mode='bilinear'):
        args = {'kernel_size': kernel_size, 'expand_ratio': expand_ratio, 'padding': padding, 
                'norm_layer': norm_layer, 'activation': activation}
        super(ResizeInvertedResidual, self).__init__(
            in_channels, out_channels, scale_factor, (InvertedResidual, args), mode=mode)


class MobileUNet(UNet):
    def __init__(self, num_classes, scale_factor=2, resize_mode='bilinear',
                 encoder={'architecture': 'mobilenet_v1', 'width_mult': 1.0, 'return_layers': None}, 
                 decoder={'mode': 'bilinear', 'n_blocks': None}, 
                 out_channels=None, **kwargs):
        super(MobileUNet, self).__init__(
            num_classes, scale_factor, resize_mode,
            encoder, decoder, out_channels, **kwargs)
    
    def get_encoder(self, **kwargs):
        ## by default use all possible layers in settings
        kwargs.setdefault('return_layers', None)
        return MobileNetFeatures(**kwargs)
    
    def _default_uplayer(self, block, mode='bilinear'):
        default_args = {'kernel_size': 3, 'norm_layer': 'batch', 'padding': 'default', 
                        'activation': nn.ReLU6(inplace=True)}
        
        def f(in_c, out_c, scale_factor, **kwargs):
            args = {k: v for k, v in {**default_args, **kwargs}.items()
                    if k in inspect.getargspec(block)[0]}
            return ResizeConv2d(in_c, out_c, scale_factor, conv=(block, args), mode=mode)
            # return SubPixelConv2d(in_c, out_c, scale_factor, conv=(block, args))
        return f
    
    def get_decoder_config(self, **kwargs):
        config = {
            'setting': kwargs.get('setting') or 'default', 
            'init_block': kwargs.get('init_block', self.encoder.architecture['init_block']),
            'block': kwargs.get('block', self.encoder.architecture['block']),
            'up': kwargs.get('up', self._default_uplayer(Conv2d, kwargs.get('mode', 'bilinear'))),
        }
        # config.update(kwargs)
        if config['setting'] == 'default':
            n_blocks = kwargs.get('n_blocks', None)
            setting = [[1, 32, 1, 2]]
            for t, c, n, s in self.encoder.architecture['setting']:
                setting[-1][-1] = s
                setting.append([t, c, n_blocks or n, s])
            config['setting'] = setting[:-1]
        
        # expand_ratio = kwargs.setdefault('expand_ratio', 'original')
        # assert expand_ratio in ['symmetric', 'original'] # symmetric gives a little bit poor performance
        # n_blocks = kwargs.setdefault('n_blocks', 'symmetric')
        
        return config
    
    def get_decoder(self, **kwargs):
        N = len(self.out_channels) - 1
        config = self.get_decoder_config(**kwargs)
        assert len(config['setting']) == N, "decoder setting should match encoder layers. "
        # in_channels = self.out_channels[-1:1:-1]
        # out_channels = self.out_channels[-2:0:-1]
        # setting = config['setting'][-1::-1]
        
        ## bug in mobilenet v2: setting has s=1 part. So there is a mismatch
        decoder = nn.ModuleList()
        for i in range(N-1, -1, -1):
            in_c = self.out_channels[i+1]
            out_c = self.out_channels[i]
            t, c, n, s = config['setting'][i]
            
            if s == 1 and in_c == out_c:
                up_layer = nn.Identity()
            else:
                up_layer = config['up'](in_c, out_c, scale_factor=s, expand_ratio=t)
            
            block = config['block'] if i > 0 else config['init_block']
            if i > 0:
                block = config['block']
                conv_layer = nn.Sequential(
                    block(out_c*2, out_c, stride=1, expand_ratio=t), 
                    *[block(out_c, out_c, stride=1, expand_ratio=t) for _ in range(n-1)]
                )
            else:
                block = config['init_block']
                conv_layer = nn.Sequential(
                    block(out_c*2, out_c, stride=1), 
                    *[block(out_c, out_c, stride=1) for _ in range(n-1)]
                )
            
            decoder.append(nn.ModuleDict([('up', up_layer), ('conv', conv_layer)]))
        ## register config
        decoder.architecture = config
        
        return decoder

################################################################
################### Object Detection Models ####################
################################################################
class BoxPredictor(nn.Sequential):
    def __init__(self, in_channels, featmap_names, num_classes, 
                 roi_output_size=7, roi_sampling_ratio=2, layers=[1024, 1024]):
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=featmap_names, 
            output_size=roi_output_size, 
            sampling_ratio=roi_sampling_ratio)
        header = tmdet.faster_rcnn.TwoMLPHead(
            in_channels=in_channels * roi_pooler.output_size[0] ** 2,
            representation_size=layers[0],
        )
        predictor = tmdet.faster_rcnn.FastRCNNPredictor(
            in_channels=layers[-1], num_classes=num_classes)
        
        super(BoxPredictor, self).__init__(
            OrderedDict([
                ('box_roi_pool', roi_pooler),
                ('box_head', header),
                ('box_predictor', predictor),
            ])
        )

class MaskPredictor(nn.Sequential):
    def __init__(self, in_channels, featmap_names, num_classes, 
                 roi_output_size=14, roi_sampling_ratio=2,
                 layers=[256, 256, 256, 256], dim_reduced=256, dilation=1):
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=featmap_names, 
            output_size=roi_output_size, 
            sampling_ratio=roi_sampling_ratio)
        header = tmdet.mask_rcnn.MaskRCNNHeads(
            in_channels=in_channels, layers=layers, dilation=dilation)
        predictor = tmdet.mask_rcnn.MaskRCNNPredictor(
            in_channels=layers[-1], dim_reduced=dim_reduced, num_classes=num_classes)
        
        super(MaskPredictor, self).__init__(
            OrderedDict([
                ('mask_roi_pool', roi_pooler),
                ('mask_head', header),
                ('mask_predictor', predictor),
            ])
        )

class KeypointPredictor(nn.Sequential):
    def __init__(self, in_channels, featmap_names, num_keypoints,
                 roi_output_size=14, roi_sampling_ratio=2,
                 layers=[512] * 8):
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=featmap_names, 
            output_size=roi_output_size, 
            sampling_ratio=roi_sampling_ratio)
        header = tmdet.keypoint_rcnn.KeypointRCNNHeads(
            in_channels=in_channels, layers=layers)        
        predictor = tmdet.keypoint_rcnn.KeypointRCNNPredictor(
            in_channels=layers[-1], num_keypoints=num_keypoints)
        
        super(KeypointPredictor, self).__init__(
            OrderedDict([
                ('keypoint_roi_pool', roi_pooler),
                ('keypoint_head', header),
                ('keypoint_predictor', predictor),
            ])
        )

                 
class RCNN(tmdet.generalized_rcnn.GeneralizedRCNN):
    """ Build the FasterRCNN/MaskRCNN detection model.
        We try use pretrained model/layers as much as possible.
        If the given config match default config, we will keep corresponding layers 
        and only replace layers that need to be retrained.
        
        GeneralizedRCNN(backbone, rpn, roi_heads, transform)
            backbone: BackboneWithFPN
            rpn: RegionProposalNetwork (soft nms)
                anchor: AnchorGenerator, RPNHead
            roi_heads: RoIHeads (cascade)
                box: MultiScaleRoIAlign, TwoMLPHead, FastRCNNPredictor
                mask: MultiScaleRoIAlign, MaskRCNNHeads, MaskRCNNPredictor
                keypoint: MultiScaleRoIAlign, KeypointRCNNHeads, KeypointRCNNPredictor
            transform: GeneralizedRCNNTransform
    """
    def __init__(self, backbone, num_classes, masks=False, keypoints=None, 
                 config={}, pretrained=False):
        self.config = self.default_config(num_classes, masks, keypoints)
        deep_update(self.config, config)
        self.config['featmap_names'] = self.config['featmap_names'] or backbone.featmap_names
        self.config['in_channels'] = self.config['in_channels'] or backbone.out_channels
        
        rpn = self.get_rpn()
        roi_heads = self.get_roi_heads()
        transform = self.get_transform()
        super(RCNN, self).__init__(backbone, rpn, roi_heads, transform)
        
        if pretrained:
            self.load_pretrain(pretrained)
    
    def get_transform(self):
        return tmdet.transform.GeneralizedRCNNTransform(**self.config['transform'])
    
    def get_rpn(self):
        in_channels = self.config['in_channels']
        rpn_params = self.config['rpn_params']
        
        rpn_anchor = tmdet.rpn.AnchorGenerator(**rpn_params['anchor'])
        rpn_header = tmdet.rpn.RPNHead(in_channels, rpn_anchor.num_anchors_per_location()[0])
        rpn = tmdet.rpn.RegionProposalNetwork(rpn_anchor, rpn_header, **rpn_params['rpn'])
        
        return rpn
    
    def get_roi_heads(self):
        featmap_names = self.config['featmap_names']
        in_channels = self.config['in_channels']
        roi_params = self.config['roi_params']
        
        ## box header
        box_header = BoxPredictor(in_channels, featmap_names, **roi_params['box'])
        
        ## roi heads
        roi_heads = tmdet.roi_heads.RoIHeads(
            box_roi_pool=box_header.box_roi_pool,
            box_head=box_header.box_head,
            box_predictor=box_header.box_predictor,
            **roi_params['roi']
        )
        
        ## add mask header
        if 'mask' in roi_params:
            mask_header = MaskPredictor(in_channels, featmap_names, **roi_params['mask'])
            roi_heads.mask_roi_pool = mask_header.mask_roi_pool
            roi_heads.mask_head = mask_header.mask_head
            roi_heads.mask_predictor = mask_header.mask_predictor
        
        ## add keypoint header
        if 'keypoint' in roi_params:
            keypoint_header = KeypointPredictor(in_channels, featmap_names, **roi_params['keypoint'])
            roi_heads.keypoint_roi_pool = keypoint_header.keypoint_roi_pool
            roi_heads.keypoint_head = keypoint_header.keypoint_head
            roi_heads.keypoint_predictor = keypoint_header.keypoint_predictor
        
        return roi_heads
    
    def load_pretrain(self, pretrained):
        if isinstance(pretrained, str):
            weights = torch.load(pretrained)
        else:
            if self.roi_heads.has_mask():
                m = tmdet.maskrcnn_resnet50_fpn(
                    pretrained=True, progress=False, pretrained_backbone=False)
            elif self.roi_heads.has_keypoint():
                m = tmdet.keypointrcnn_resnet50_fpn(
                    pretrained=True, progress=False, pretrained_backbone=False)
            else:
                m = tmdet.fasterrcnn_resnet50_fpn(
                    pretrained=True, progress=False, pretrained_backbone=False)
            weights = m.state_dict()
        
        try:
            # remove backbone from state_dict
            w = {k: v for k, v in weights.items() 
                 if not k.startswith('backbone.body')}
            self.load_state_dict(w, strict=False)
        except RuntimeError as e:
            print(e)

    def default_config(self, num_classes, masks, keypoints):
        config = {
            ## backbone
            'featmap_names': None,
            'in_channels': None,
            ## rpn
            'rpn_params': {
                'anchor': {
                    'sizes': [[32], [64], [128], [256], [512]],
                    'aspect_ratios': [[0.5, 1.0, 2.0]] * 5,
                }, 
                'rpn': {
                    'fg_iou_thresh': 0.7, 
                    'bg_iou_thresh': 0.3,
                    'batch_size_per_image': 256, 
                    'positive_fraction': 0.5,

                    'pre_nms_top_n': {'training': 2000, 'testing': 1000},
                    'post_nms_top_n': {'training': 2000, 'testing': 1000},
                    'nms_thresh': 0.7,
                },
            },
            ## roi
            'roi_params': {
                ## roi predictor
                'roi': {
                    # Faster R-CNN training
                    'fg_iou_thresh': 0.5, 
                    'bg_iou_thresh': 0.5,
                    'batch_size_per_image': 512, 
                    'positive_fraction': 0.25,
                    'bbox_reg_weights': None,
                    # Faster R-CNN inference
                    'score_thresh': 0.05, 
                    'nms_thresh': 0.5, 
                    'detections_per_img': 100,
                },
                ## box predictor
                'box': {
                    'num_classes': num_classes,
                    'roi_output_size': 7, 
                    'roi_sampling_ratio': 2,
                    'layers': [1024, 1024], 
                },
            },
            'transform': {
                'min_size': 800, 'max_size': 1333, 
                'image_mean': [0.485, 0.456, 0.406], 'image_std': [0.229, 0.224, 0.225],
            }
        }
        if masks:
            ## mask predictor
            config['roi_params']['mask'] = {
                'num_classes': num_classes,
                'roi_output_size': 14, 
                'roi_sampling_ratio': 2,
                'layers': [256, 256, 256, 256],
                'dilation': 1,
                'dim_reduced': 256,
            }
        if keypoints:
            ## keypoint predictor
            config['roi_params']['keypoint'] = {
                'num_keypoints': keypoints,
                'roi_output_size': 14, 
                'roi_sampling_ratio': 2,
                'layers': [512] * 8,
            }

        return config

################################################################
###################### Losses and Metrics ######################
################################################################
class SoftCrossEntropyLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(SoftCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        loss = F.kl_div(F.log_softmax(input), target, reduction='none')
        if self.weight is not None:
            loss = loss * self.weight
        
        idx = [_ != self.ignore_index for _ in range(loss.shape[-1])]
        loss = loss[..., idx]
        loss = loss.sum(-1)
        
        if self.reduction == 'none':
            ret = loss
        elif self.reduction == 'mean':
            ret = loss.mean()
        elif self.reduction == 'sum':
            ret = loss.sum()
        else:
            ret = input
            raise ValueError(reduction + " is not valid")
        
        return ret


def match_pred_true(y_pred, y_true, binary=False, axis=1):
    """ Transform (sparse) y_true to match y_pred. """
    dtype = y_pred.dtype
    num_classes = y_pred.shape[axis]
    o = list(range(y_pred.ndim))
    o = o[0:axis] + [o[-1]] + o[axis:-1]
    
    ## squeeze if channel dimension is 1
    if y_true.ndim == y_pred.ndim and y_true.shape[axis] == 1:
        y_true = y_true.squeeze(axis)
    if y_true.ndim != y_pred.ndim:
        y_true = F.one_hot(y_true.type(torch.long), num_classes).permute(*o)
    
    if binary:
        y_true = F.one_hot(y_true.argmax(axis), num_classes).permute(*o)
        y_pred = F.one_hot(y_pred.argmax(axis), num_classes).permute(*o)
    
    return y_pred.to(dtype), y_true.to(dtype)


class IoU(nn.Module):
    def __init__(self, mode='iou', axis=1, eps=0.):
        """ Return a matrix of [batch * num_classes]. 
            Note: In order to separate from iou=0, function WILL return NaN if both 
            y_true and y_pred are 0. Need further treatment to remove nan in either 
            loss function or matrix.
        """
        super(IoU, self).__init__()
        assert mode in ['iou', 'dice']
        self.factor = {'iou': -1.0, 'dice': 0.0}[mode]
        self.eps = eps
        self.axis = axis
    
    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        sum_axis = list(range(1, self.axis)) + list(range(self.axis+1, y_pred.ndim))
        
        prod = (y_true * y_pred).sum(sum_axis)
        plus = (y_true + y_pred).sum(sum_axis)
        
        ## We keep nan for 0/0 in order to correctly apply weight
        iou = (2 + self.factor) * prod / (plus + self.factor * prod + self.eps)
        # print([y_true.shape, y_pred.shape, prod.shape, plus.shape])
        # print([prod, plus, iou])
        
        return iou


class SoftDiceLoss(IoU):
    def __init__(self, weight=None, ignore_index=[], reduction='mean',
                 mode='dice', axis=1, eps=0.):
        super(SoftDiceLoss, self).__init__(mode, axis, eps)
        self.ignore_index = ignore_index
        self.register_buffer('weight', weight)
        self.reduction = {
            'none': lambda x: x,
            'mean': torch.mean,
            'sum': torch.sum,
        }[reduction]
    
    def _apply_weight(self, x):
        """ Apply class_weights to calculate loss, ignore nan. """        
        if self.weight is None:
            weight = torch.ones(x.shape[-1], device=x.device)
        else:
            weight = self.weight
        
        ## remove ignore_index
        idx = np.ones(x.shape[-1], dtype=bool)
        idx[self.ignore_index] = False
        x, weight = x[:,idx], weight[idx]
        
        ## apply weight
        weight = ~torch.isnan(x) * weight
        return x * weight / weight.sum(-1, keepdim=True)
    
    def forward(self, y_pred, y_true):
        ## y_pred is softmax cannot be 0, so no nan in res
        iou = super(SoftDiceLoss, self).forward(y_pred, y_true)
        # iou = torch.where(torch.isnan(iou), torch.zeros_like(iou), iou)
        iou = self._apply_weight(iou)
        # print(["apply_weights", res])
        
        return -self.reduction(iou.sum(-1))


class Accuracy(nn.Module):
    def __init__(self, weight=None):
        super(Accuracy, self).__init__()
        self.register_buffer('weight', weight)
        # self.weight = torch.tensor(weight, dtype=torch.float)
    
    def forward(self, y_pred, y_true):
        if self.weight is not None:
            w = self.weight[y_true]
            c = (w * (y_true == y_pred)).sum() * 1.0
            t = w.sum()
        else:
            c = (y_true == y_pred).sum() * 1.0
            t = y_pred.shape[0]
        
        return c/t, t


class BinaryIndicators(nn.Module):
    def forward(self, y_pred, y_true):
        tp = (y_pred * y_true).sum().item()
        fp = (y_pred * (1-y_true)).sum().item()
        fn = (y_true * (1-y_pred)).sum().item()
        tn = ((1-y_pred) * (1-y_true)).sum().item()
        
        ## some common indicators
        fpr = (fp/(fp+tn+1e-8), fp+tn)
        tnr = specificity = (tn/(fp+tn+1e-8), fp+tn)
        
        fnr = (fn/(fn+tp+1e-8), fn+tp)
        tpr = sensitivity = recall = (tp/(fn+tp+1e-8), fn+tp)
        
        fdr = (fp/(fp+tp+1e-8), fp+tp)
        precision = (tp/(fp+tp+1e-8), fp+tp)
        
        acc = ((tp+tn)/(tp+tn+fp+fn+1e-8), tp+tn+fp+fn)
        # f_measure = 2 * precision * recall / (precision + recall)
        # from sklearn.metrics import roc_auc_score
        # roc_auc_score(y_true, y_pred)
        
        return {'precision': precision, 'recall': recall}


def pearson_coefficient(output, target, dim=-1):
    target = target.to(output.dtype)
    x = output - torch.mean(output, dim=dim, keepdim=True)
    y = target - torch.mean(target, dim=dim, keepdim=True)
    return torch.sum(x * y, dim=dim) * torch.rsqrt(torch.sum(x ** 2, dim=dim)) * torch.rsqrt(torch.sum(y ** 2, dim=dim))
