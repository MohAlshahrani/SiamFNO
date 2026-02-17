import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet
from neuralop.models import FNO
from utils import create_gaussian_response

class SiamFCBackbone(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 96, 11, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
        )

    def forward(self, x):
        return self.features(x)

# class SiamFC(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     self.backbone = SiamFCBackbone()

    # def xcorr_depthwise(self, z, x):
    #     """
    #     z: template feature, [B, C, Hk, Wk]
    #     x: search feature,   [B, C, Hx, Wx]
    #     """
    #     B, C, Hk, Wk = z.shape
    #     _, _, Hx, Wx = x.shape

    #     # reshape for grouped convolution
    #     x = x.view(1, B * C, Hx, Wx)
    #     z = z.view(B * C, 1, Hk, Wk)

    #     # grouped convolution with groups = B*C
    #     out = F.conv2d(x, z, groups=B*C)

    #     # reshape back
    #     out = out.view(B, C, out.size(-2), out.size(-1))
    #     return out.sum(dim=1, keepdim=True)  # sum over channels

    # def forward(self, z, x):
    #     f_z = self.backbone(z)
    #     f_x = self.backbone(x)
    #     out = self.xcorr_depthwise(f_z, f_x)
    #     out = torch.sigmoid(out)
    #     return out

class SiamFCAlexNet(nn.Module):
    def __init__(self, mode="valid"):
        super().__init__()

        # Load pretrained AlexNet
        alex = alexnet(weights="IMAGENET1K_V1")
        feats = list(alex.features)
        orig_conv1 = feats[0]   # Conv2d(3, 64, kernel_size=11, stride=4, padding=2)

        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv1.out_channels,
            kernel_size=orig_conv1.kernel_size,
            stride=(2, 2),     # SiamFC: change stride 4 to 2
            padding=orig_conv1.padding,
            bias=True
        )

        # Copy pretrained weights: average RGB filters to grayscale filter
        with torch.no_grad():
            new_conv1.weight[:] = orig_conv1.weight.mean(dim=1, keepdim=True)
            new_conv1.bias[:]   = orig_conv1.bias

        self.backbone = nn.Sequential(
            new_conv1,
            feats[1],  # relu
            feats[2],  # pool1
            feats[3], feats[4], feats[5],  # conv2, relu2, pool2
            feats[6], feats[7],  # conv3, relu3
            feats[8], feats[9],  # conv4, relu4
            feats[10], feats[11] # conv5, relu5
        )

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # This is the 1x1 conv used in SiamFC
        self.adjust = nn.Conv2d(256, 256, kernel_size=1)

        self.mode = mode  # 'valid': 17x17 / 'full': 31x31

    def xcorr_valid(self, z, x):
        # z: BxCxkxk, x: BxCxHxW
        B, C, k, _ = z.shape
        _, _, H, W = x.shape
        x = x.view(1, B*C, H, W)
        z = z.view(B*C, 1, k, k)
        out = F.conv2d(x, z, groups=B*C)
        out = out.view(B, C, out.shape[-2], out.shape[-1])
        # Sum over channels --> final single-channel response
        return out.sum(dim=1, keepdim=True)

    def xcorr_full(self, z, x):
        pad = z.shape[-1] // 2
        return nn.functional.conv2d(x, z, padding=pad, groups=z.shape[0])

    def forward(self, template, search):
        z = self.backbone(template)
        x = self.backbone(search)
        z = self.adjust(z)

        if self.mode == "valid":
            return self.xcorr_valid(z, x)
        else:
            return self.xcorr_full(z, x)

# ------------------------------- #
#         SiamFC for FNO          #
# ------------------------------- #
class SiamFC(nn.Module):
    def __init__(self, mode="valid"):
        super().__init__()

        # Load pretrained AlexNet
        alex = alexnet(weights="IMAGENET1K_V1")
        feats = list(alex.features)
        orig_conv1 = feats[0]   # Conv2d(3, 64, kernel_size=11, stride=4, padding=2)

        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv1.out_channels,
            kernel_size=orig_conv1.kernel_size,
            stride=(2, 2),     # SiamFC: change stride 4 to 2
            padding=orig_conv1.padding,
            bias=True
        )

        # Copy pretrained weights: average RGB filters to grayscale filter
        with torch.no_grad():
            new_conv1.weight[:] = orig_conv1.weight.mean(dim=1, keepdim=True)
            new_conv1.bias[:]   = orig_conv1.bias

        self.backbone = nn.Sequential(
            new_conv1,
            feats[1],  # relu
            feats[2],  # pool1
            feats[3], feats[4], feats[5],  # conv2, relu2, pool2
            feats[6], feats[7],  # conv3, relu3
            feats[8], feats[9],  # conv4, relu4
            feats[10], feats[11] # conv5, relu5
        )

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # This is the 1x1 conv used in SiamFC
        self.adjust = nn.Conv2d(256, 256, kernel_size=1)

        self.mode = mode  # 'valid': 17x17 / 'full': 31x31

    def xcorr_valid(self, z, x):
        # z: BxCxkxk, x: BxCxHxW
        B, C, k, _ = z.shape
        _, _, H, W = x.shape
        x = x.view(1, B*C, H, W)
        z = z.view(B*C, 1, k, k)
        out = F.conv2d(x, z, groups=B*C)
        out = out.view(B, C, out.shape[-2], out.shape[-1])
        # Sum over channels --> final single-channel response
        return out.sum(dim=1, keepdim=True)

    def xcorr_full(self, z, x):
        pad = z.shape[-1] // 2
        return nn.functional.conv2d(x, z, padding=pad, groups=z.shape[0])
    
    def forward(self, template, search):
        z = self.backbone(template)
        x = self.backbone(search)
        return x,z
    

class SiamFNO(nn.Module):
    """ This is the main model that combines SiamFC and FNO.
 The output of the SiamFC is fed into the FNO to learn the
 deformation operator.    """
    def __init__(self):
        super().__init__()
        self.siam = SiamFC()
        self.backbone = FNO(
            n_modes=(15,15),            # Number of Fourier modes to keep in each dimension
            hidden_channels=64,         # Hidden layer width:
            in_channels=513,            # Input channels: depth of feature maps (2)(256) + mask (1)
            out_channels=1,             # Output channels: 
            n_layers=4                   
        )

    def forward(self, template, search, object_location):
        f_t,f_s = self.siam(template, search)
        mask = create_gaussian_response(object_location, f_s.shape[-2], f_s.shape[-1], device=f_s.device)
        fno_input = torch.cat([f_s,f_t, mask], dim=1) # make sure the dim=1 is correct.
        phi = self.backbone(fno_input)
        #TO DO: insert warpping module here to warp the search feature map using the learned deformation operator phi.
        return phi
        
 
    
    