import os, sys, time
import typing as tp
import numpy as np
import torch, torchvision
from torchvision.models._utils import IntermediateLayerGetter


class UNet(torch.nn.Module):
    '''Backboned U-Net. (Not the original architecture.)'''

    class UpBlock(torch.nn.Module):
        def __init__(self, in_c, out_c, inter_c=None):
            #super().__init__()
            torch.nn.Module.__init__(self)
            inter_c        = inter_c or out_c
            self.conv1x1   = torch.nn.Conv2d(in_c, inter_c, 1)
            self.convblock = torch.nn.Sequential(
                torch.nn.Conv2d(inter_c, out_c, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(out_c),
                torch.nn.ReLU(),
            )
        def forward(self, x:torch.Tensor, skip_x:torch.Tensor, relu=True) -> torch.Tensor:
            x = resize_tensor(x, skip_x.shape[-2:], mode='nearest')   #TODO? mode='bilinear
            # x = resize_tensor(x, skip_x.shape[-2:], mode='bilinear')   #TODO? mode='bilinear
            x = torch.cat([x, skip_x], dim=1)
            x = self.conv1x1(x)
            x = self.convblock(x)
            return x
    
    def __init__(
        self, 
        backbone                  = 'mobilenet3l', 
        input_channels:int        = 3, 
        output_channels:int       = 1, 
        backbone_weights:str|None = 'DEFAULT'
    ):
        torch.nn.Module.__init__(self)
        factory_func = BACKBONES.get(backbone, None)
        if factory_func is None:
            raise NotImplementedError(backbone)
        self.backbone, C = factory_func(backbone_weights, input_channels)
        self.backbone_name = backbone
        
        self.up0 = self.UpBlock(C[-1]    + C[-2],  C[-2])
        self.up1 = self.UpBlock(C[-2]    + C[-3],  C[-3])
        self.up2 = self.UpBlock(C[-3]    + C[-4],  C[-4])
        self.up3 = self.UpBlock(C[-4]    + C[-5],  C[-5])
        self.up4 = self.UpBlock(C[-5]    + input_channels, 32)
        self.cls = torch.nn.Conv2d(32, output_channels, 3, padding=1)
    
    def forward(self, x:torch.Tensor, sigmoid=False, return_features=False) -> torch.Tensor:
        device = list(self.parameters())[0].device
        x      = x.to(device)
        
        X = self.backbone(x)
        X = ([x] + [X[f'out{i}'] for i in range(5)])[::-1]
        x = X.pop(0)
        x = self.up0(x, X[0])
        x = self.up1(x, X[1])
        x = self.up2(x, X[2])
        x = self.up3(x, X[3])
        x = self.up4(x, X[4])
        if return_features:
            return x
        x = self.cls(x)
        if sigmoid:
            x = torch.sigmoid(x)
        return x


def mobilenet3l_backbone(
    weights:str|None, 
    input_channels:int = 3
) -> tp.Tuple[torch.nn.Module, tp.List[int]]:
    base = torchvision.models.mobilenet_v3_large(weights='DEFAULT')
    return_layers = {'1':'out0', '3':'out1', '6':'out2', '10':'out3', '16':'out4'}
    backbone = IntermediateLayerGetter(base, return_layers)
    channels = [16, 24, 40, 80, 960]
    # if input_channels != 3:
    #     backbone['0'][0] = _clone_conv2d_with_new_input_channels(backbone['0'][0], input_channels)
    return backbone, channels

BACKBONES = {
    #'resnet18':    resnet18_backbone,
    #'resnet50':    resnet50_backbone,
    'mobilenet3l': mobilenet3l_backbone,
}

# torchvision.models.mobilenet_v3_large(weights='DEFAULT')

def resize_tensor(
    x:    torch.Tensor, 
    size: int|tp.Tuple[int,int]|torch.Size, 
    mode: tp.Literal['nearest', 'bilinear'],
    align_corners: bool|None = None,
) -> torch.Tensor:
    assert torch.is_tensor(x)
    assert len(x.shape) in [3,4]
    x0 = x
    if len(x0.shape) == 3:
        x = x[np.newaxis]
    y = torch.nn.functional.interpolate(x, size, mode=mode, align_corners=align_corners)
    if len(x0.shape) == 3:
        y = y[0]
    return y


