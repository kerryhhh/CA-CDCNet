import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange
from einops.layers.torch import Rearrange

#30 SRM filtes
from models.modules.srm_filter_kernel import all_normalized_hpf_list
from models.modules.MPNCOV import *


def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


#Truncation operation
class TLU(nn.Module):
  def __init__(self, threshold):
    super(TLU, self).__init__()

    self.threshold = threshold

  def forward(self, input):
    output = torch.clamp(input, min=-self.threshold, max=self.threshold)

    return output


class HPF_srm(nn.Module):
    def __init__(self, trainable=False) -> None:
        super().__init__()
        hpf_list = self.build_filters()

        hpf_weight = nn.Parameter(torch.Tensor(hpf_list).view(30, 1, 5, 5), requires_grad=trainable)
        self.hpf = nn.Conv2d(1, 30, kernel_size=(5, 5), stride=(1, 1), padding=2, bias=False)
        self.hpf.weight = hpf_weight
        
        self.tlu = TLU(5.0)
    
    def forward(self, x):
        output = []
        for i in range(x.shape[1]):
            output.append(self.hpf(x[:, i, :, :].unsqueeze(dim=1)))
        output = torch.cat(output, dim=1)
        output = self.tlu(output)
        return output
    
    def build_filters(self):
        hpf_list = []
        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
            hpf_list.append(hpf_item)
        return hpf_list


class HPF_gabor(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()

        hpf_list = self.build_filters()

        hpf_weight = nn.Parameter(torch.Tensor(hpf_list).view(32, 1, 5, 5), requires_grad=trainable)
        self.hpf = nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=2, bias=False)
        self.hpf.weight = hpf_weight

        self.tlu = TLU(5.0)

    def forward(self, x):
        output = []
        for i in range(x.shape[1]):
            output.append(self.hpf(x[:, i, :, :].unsqueeze(dim=1)))
        output = torch.cat(output, dim=1)
        output = self.tlu(output)

        return output
    
    def build_filters(self):
        filters = []
        ksize = [5]     
        lamda = np.pi/2.0 
        sigma = [0.5,1.0]
        phi = [0,np.pi/2]
        for theta in np.arange(0,np.pi,np.pi/8): #gabor 0 22.5 45 67.5 90 112.5 135 157.5
            for k in range(2):
                for j in range(2):
                    kern = cv2.getGaborKernel((ksize[0],ksize[0]),sigma[k],theta,sigma[k]/0.56,0.5,phi[j],ktype=cv2.CV_32F)
                    #print(1.5*kern.sum())
                    #kern /= 1.5*kern.sum()
                    filters.append(kern)
        return filters
    
    
class Pre(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv2 = nn.Conv2d(6, 6, 3, 1, 1)
        self.conv3 = nn.Conv2d(6, 6, 3, 1, 1)

    def forward(self, x):
        f1 = self.conv1(x)
        f1 = torch.cat((f1, x), dim=1)
        f2 = self.conv2(f1)
        f = f1-f2
        f3 = self.conv3(f)
        f = torch.cat((f, f3), dim=1)
        return f


class Block2(nn.Module):
    def __init__(self, in_channels, out_channels, theta=0.7):
        super().__init__()
        self.process = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            Conv2d_cd(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            Conv2d_cd(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
            # nn.AvgPool2d(3,2,1)
        )

    def forward(self, x):
        out = self.process(x)
        return out


class Residual1(nn.Module):
    def __init__(self, in_channels, out_channels, theta=0.7):
        super().__init__()
        self.blk = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            Conv2d_cd(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
            # nn.AvgPool2d(3,2,1)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.blk(x)
        x2 = self.shortcut(x)
        x = self.relu(x1 + x2)
        return x
    

class Residual2(nn.Module):
    def __init__(self, in_channels, out_channels, theta=0.7):
        super().__init__()
        self.blk = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            Conv2d_cd(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            Conv2d_cd(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
            # nn.AvgPool2d(3,2,1)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.blk(x)
        x2 = self.shortcut(x)
        x = self.relu(x1 + x2)
        return x
    

class Plain1(nn.Module):
    def __init__(self, in_channels, out_channels, theta=0.7):
        super().__init__()
        self.blk = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            Conv2d_cd(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
            # nn.AvgPool2d(3,2,1)
        )

    def forward(self, x):
        x = self.blk(x)
        return x


class Plain2(nn.Module):
    def __init__(self, in_channels, out_channels, theta=0.7):
        super().__init__()
        self.blk = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            Conv2d_cd(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            Conv2d_cd(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
            # nn.AvgPool2d(3,2,1)
        )

    def forward(self, x):
        x = self.blk(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0, 
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i]//2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)
            
    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


# class Mlp(nn.Module):  ### MS-FFN
#     """
#     Mlp implemented by with 1x1 convolutions.

#     Input: Tensor with shape [B, C, H, W].
#     Output: Tensor with shape [B, C, H, W].
#     Args:
#         in_features (int): Dimension of input features.
#         hidden_features (int): Dimension of hidden features.
#         out_features (int): Dimension of output features.
#         drop (float): Dropout rate. Defaults to 0.0.
#     """

#     def __init__(self,
#                  in_features,
#                  hidden_features=None,
#                  out_features=None,
#                  drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Sequential(
#             nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
#             nn.GELU(),
#             nn.BatchNorm2d(hidden_features),
#         )
#         self.dwconv = MultiScaleDWConv(hidden_features)
#         self.act = nn.GELU()
#         self.norm = nn.BatchNorm2d(hidden_features)
#         self.fc2 = nn.Sequential(
#             nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
#             nn.BatchNorm2d(in_features),
#         )
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
        
#         x = self.fc1(x)

#         x = self.dwconv(x) + x
#         x = self.norm(self.act(x))
        
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)

#         return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class CrossAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.) -> None:
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = out_dim // num_heads
        self.scale = head_dim ** -0.5
        # project_out = not(num_heads == 1 and in_dim == out_dim)
        # project_out = not(in_dim == out_dim)

        self.x_proj_qkv = nn.Linear(in_dim, out_dim * 3)

        self.y_proj_qkv = nn.Linear(in_dim, out_dim * 3)

        # self.dropout = nn.Dropout(dropout)

        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))

        self.softmax = nn.Softmax(dim=1)

        # self.proj_out1 = nn.Linear(out_dim, out_dim, 1) if project_out else nn.Identity()
        # self.proj_out2 = nn.Linear(out_dim, out_dim, 1) if project_out else nn.Identity()

    def forward(self, x, y):
        # b, c, h, w = x.shape
        
        x_qkv = self.x_proj_qkv(x).chunk(3, dim=-1)
        x_q, x_k, x_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), x_qkv)

        y_qkv = self.y_proj_qkv(y).chunk(3, dim=-1)
        y_q, y_k, y_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), y_qkv)

        # x_qk = self.x_proj_qk(x).chunk(2, dim=-1)
        # x_q, x_k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), x_qk)
        # x_v = rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads)

        # y_qk = self.y_proj_qk(y).chunk(2, dim=-1)
        # y_q, y_k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), y_qk)
        # y_v = rearrange(y, 'b n (h d) -> b h n d', h=self.num_heads)

        dot_x = (x_q @ x_k.transpose(-1, -2)) * self.scale
        dot_y = (y_q @ y_k.transpose(-1, -2)) * self.scale
        # attn = self.softmax(self.gamma1 * dot_x + self.gamma2 * dot_y)
        # attn = self.dropout(attn)

        attn_x = self.softmax(dot_x + self.gamma1 * dot_y)
        # attn_x = self.dropout(attn_x)
        attn_y = self.softmax(dot_y + self.gamma2 * dot_x)
        # attn_y = self.dropout(attn_y)

        x = attn_x @ x_v
        y = attn_y @ y_v

        x = rearrange(x, "b h n d -> b n (h d)")
        # x = self.dropout(self.proj_out1(x))
        y = rearrange(y, "b h n d -> b n (h d)")
        # y = self.dropout(self.proj_out2(y))

        return x, y


class CrossBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, mlp_ratio=4, dropout=0., drop_path=0.) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CrossAttention(in_dim, out_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp1 = Mlp(out_dim, out_dim * mlp_ratio, out_dim, dropout)
        self.mlp2 = Mlp(out_dim, out_dim * mlp_ratio, out_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x, y):
        x_out = self.norm1(x)
        y_out = self.norm1(y)
        x_out, y_out = self.attn(x_out, y_out)
        x_out = self.drop_path(x_out) + x
        y_out = self.drop_path(y_out) + y
        x_out = self.drop_path(self.mlp1(self.norm2(x_out))) + x_out
        y_out = self.drop_path(self.mlp2(self.norm2(y_out))) + y_out
        return x_out, y_out
        

class CrossTransformer(nn.Module):
    def __init__(self, depth, dim, num_heads, mlp_ratio=4, dropout=0., drop_path=0.) -> None:
        super().__init__()
        # self.patch_embed1 = PatchEmbed(patch_size=3, stride=1, padding=1, in_chans=dim, embed_dim=dim)
        # self.patch_embed2 = PatchEmbed(patch_size=3, stride=1, padding=1, in_chans=dim, embed_dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([CrossBlock(dim, dim, num_heads, mlp_ratio, dropout, dpr[i]) for i in range(depth)])
        # self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, y):
        # x = self.patch_embed1(x)
        # y = self.patch_embed2(y)
        b, c, h, w = x.size()
        x = rearrange(x, "b c h w -> b (h w) c")
        y = rearrange(y, "b c h w -> b (h w) c")
        for block in self.blocks:
            x, y = block(x, y)
        # x = self.norm(x)
        # y = self.norm(y)
        x = x.transpose(-1, -2).view(b, c, h, w).contiguous()
        y = y.transpose(-1, -2).view(b, c, h, w).contiguous()
        return x, y


class Classifier(nn.Module):
    def __init__(self, in_channels, n_class=2) -> None:
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_channels, n_class)

    def forward(self, x):
        x = self.aap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class CovClassifier(nn.Module):
    def __init__(self, in_channels, n_class=2) -> None:
        super().__init__()
        self.fc = nn.Linear(int(in_channels * (in_channels + 1) / 2), n_class)

    def forward(self, x):
        #Global covariance pooling
        x = CovpoolLayer(x)
        x = SqrtmLayer(x, 5)
        x = TriuvecLayer(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = [128, 128, 256]
        self.rates = [1, 2, 4, 8]
        self.block_num = [2, 2]
        self.img_size = 256
        self.theta1 = 0.7
        self.theta2 = 0.7

        # encoder
        self.rgb_pre = Pre()
        self.rgb_stream = nn.Sequential(
            Block2(12, 64, theta=self.theta1),
            Plain1(64, 128, theta=self.theta1),
            Plain1(128, 256, theta=self.theta1),
            # CBAM(256)
        )
        
        # self.freq = SRMConv2d_simple(3)
        # self.freq = BayarConv2d(3, self.channels, 7, padding=3)
        # self.freq = FreqFilter(3, im_size=self.img_size, type='dct')
        self.freq_srm = HPF_srm()
        # self.freq_gabor = HPF_gabor()
        self.freq_stream = nn.Sequential(
            Block2(90, 128, theta=self.theta2),
            Residual1(128, 128, theta=self.theta2),
            Residual1(128, 256, theta=self.theta2),
            # CBAM(256)
        )
            

        # self.attn = CrossBranchAttention(self.channels)

        # middle
        # self.middle = nn.Sequential(*[AOTBlock(self.channels, self.rates) for _ in range(self.block_num)])
        # self.freq_middle = nn.Sequential(*[AOTBlock(self.channels, self.rates) for _ in range(self.block_num)])

        # self.crosstran = Crosstrans(depth=2, dim=256, hidden_dim=1024, heads=8, head_dim=32, dropout=0.1)
        self.crosstran = CrossTransformer(depth=2, dim=256, num_heads=8, mlp_ratio=4, dropout=0.1, drop_path=0.)
        # self.crossattn = DualCrossModalAttention(256)

        self.rgb_final = nn.Sequential(
            Plain1(256, 256, theta=self.theta1),
            Plain1(256, 256, theta=self.theta1),
        )
        self.freq_final = nn.Sequential(
            Residual1(256, 256, theta=self.theta2),
            Residual1(256, 256, theta=self.theta2),
        )

        # self.fusion = DualAttention(512, 512)
        # self.fusion = FusionModule(512, 256)
        self.fusion = nn.Identity()

        # classifier
        self.classifier = Classifier(512, 2)
        # self.classifier = CovClassifier(512, 2)

        # init_weights
        # self.init_weights()


    def forward(self, image):
        x = self.rgb_pre(image)
        x = self.rgb_stream(x)

        # freq_1 = self.freq_srm(image)
        # freq_2 = self.freq_gabor(image)
        # freq_x = torch.cat([freq_1, freq_2], dim=1)
        freq_x = self.freq_srm(image)
        freq_x = self.freq_stream(freq_x)
        
        # middle
        # x = self.middle(xs[-1])
        # freq_x = self.freq_middle(freq_xs[-1])

        # crosstran
        x, freq_x = self.crosstran(x, freq_x)
        # crossattn
        # x, freq_x = self.crossattn(x, freq_x)
        
        x = self.rgb_final(x)
        freq_x = self.freq_final(freq_x)

        # fusion
        x = torch.cat([x, freq_x], dim=1)
        # x = self.fusion(torch.concat((x, freq_x), dim=1))
        # x = self.fusion(x, freq_x)
        x = self.fusion(x)

        # classifier
        x = self.classifier(x)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
