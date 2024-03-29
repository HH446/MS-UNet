import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


def Norm_layer(norm_cfg, inplanes):
    norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
    if norm_cfg == 'BN':
        out = nn.BatchNorm2d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm2d(inplanes, **norm_op_kwargs)

    return out


def Activation_layer(activation_cfg, inplace=True):
    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class ConvNormNonlin(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1,
                 norm_cfg='IN', activation_cfg='LeakyReLU'):
        super(ConvNormNonlin, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg)

    def forward(self, x):
        x = self.nonlin(self.norm(self.conv(x)))
        return x

class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.bias = nn.Parameter(torch.zeros(size=(channels,), dtype=torch.float32), requires_grad=True)
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.gaussian_weight = nn.Parameter(data=kernel, requires_grad=False)
        self.gaussian_factor = nn.Parameter(torch.ones(size=(3, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
 
    def forward(self, x):
        if torch.cuda.is_available():
            self.gaussian_factor = self.gaussian_factor.cuda()
            if isinstance(self.bias, nn.Parameter):
                self.bias = self.bias.cuda()

        gaussian_weight = self.gaussian_weight * self.gaussian_factor

        if torch.cuda.is_available():
            gaussian_weight = gaussian_weight.cuda()
        # print("gaussian_shape:", gaussian_weight.shape)
        x = F.conv2d(x, gaussian_weight, self.bias, padding=2, groups=self.channels)
        return x


class AnisoDiff2D(nn.Module):
 
    def __init__(self, num_iter=10, delta_t=1/7, kappa=30, option=2, channels=3):
 
        super(AnisoDiff2D, self).__init__()
 
        self.num_iter = num_iter
        self.delta_t = delta_t
        self.kappa = kappa
        self.option = option
        self.channels = channels

        self.hN_bias = nn.Parameter(torch.zeros(size=(channels,), dtype=torch.float32), requires_grad=True)
        self.hS_bias = nn.Parameter(torch.zeros(size=(channels,), dtype=torch.float32), requires_grad=True)
        self.hE_bias = nn.Parameter(torch.zeros(size=(channels,), dtype=torch.float32), requires_grad=True)
        self.hW_bias = nn.Parameter(torch.zeros(size=(channels,), dtype=torch.float32), requires_grad=True)
        self.hNE_bias = nn.Parameter(torch.zeros(size=(channels,), dtype=torch.float32), requires_grad=True)
        self.hSE_bias = nn.Parameter(torch.zeros(size=(channels,), dtype=torch.float32), requires_grad=True)
        self.hSW_bias = nn.Parameter(torch.zeros(size=(channels,), dtype=torch.float32), requires_grad=True)
        self.hNW_bias = nn.Parameter(torch.zeros(size=(channels,), dtype=torch.float32), requires_grad=True)
 
        self.hN_weight = nn.Parameter(data=np.repeat(torch.FloatTensor(
                                    [[0, 1, 0], [0, -1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0),
                                     self.channels, axis=0), 
                                     requires_grad=False)  
        self.hN_factor = nn.Parameter(torch.ones(size=(3, 1, 1, 1), dtype=torch.float32),requires_grad=True)
        self.hS_weight = nn.Parameter(data=np.repeat(torch.FloatTensor(
                                    [[0, 0, 0], [0, -1, 0], [0, 1, 0]]).unsqueeze(0).unsqueeze(0),
                                     self.channels, axis=0), 
                                     requires_grad=False) 
        self.hS_factor = nn.Parameter(torch.ones(size=(3, 1, 1, 1), dtype=torch.float32),requires_grad=True)
        self.hE_weight = nn.Parameter(data=np.repeat(torch.FloatTensor(
                                    [[0, 0, 0], [0, -1, 1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0),
                                     self.channels, axis=0), 
                                     requires_grad=False)
        self.hE_factor = nn.Parameter(torch.ones(size=(3, 1, 1, 1), dtype=torch.float32),requires_grad=True)
        self.hW_weight = nn.Parameter(data=np.repeat(torch.FloatTensor(
                                    [[0, 0, 0], [1, -1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0),
                                     self.channels, axis=0), 
                                     requires_grad=False)
        self.hW_factor = nn.Parameter(torch.ones(size=(3, 1, 1, 1), dtype=torch.float32),requires_grad=True)
        self.hNE_weight = nn.Parameter(data=np.repeat(torch.FloatTensor(
                                    [[0, 0, 1], [0, -1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0),
                                     self.channels, axis=0), 
                                     requires_grad=False)
        self.hNE_factor = nn.Parameter(torch.ones(size=(3, 1, 1, 1), dtype=torch.float32),requires_grad=True)
        self.hSE_weight = nn.Parameter(data=np.repeat(torch.FloatTensor(
                                    [[0, 0, 0], [0, -1, 0], [0, 0, 1]]).unsqueeze(0).unsqueeze(0),
                                     self.channels, axis=0), 
                                     requires_grad=False) 
        self.hSE_factor = nn.Parameter(torch.ones(size=(3, 1, 1, 1), dtype=torch.float32),requires_grad=True)
        self.hSW_weight = nn.Parameter(data=np.repeat(torch.FloatTensor(
                                    [[0, 0, 0], [0, -1, 0], [1, 0, 0]]).unsqueeze(0).unsqueeze(0),
                                     self.channels, axis=0), 
                                     requires_grad=False) 
        self.hSW_factor = nn.Parameter(torch.ones(size=(3, 1, 1, 1), dtype=torch.float32),requires_grad=True)
        self.hNW_weight = nn.Parameter(data=np.repeat(torch.FloatTensor(
                                    [[1, 0, 0], [0, -1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0),
                                     self.channels, axis=0), 
                                     requires_grad=False) 
        self.hNW_factor = nn.Parameter(torch.ones(size=(3, 1, 1, 1), dtype=torch.float32),requires_grad=True)
 
    def forward(self, diff_im):

        if torch.cuda.is_available():
            self.hN_factor = self.hN_factor.cuda()
            self.hS_factor = self.hS_factor.cuda()
            self.hW_factor = self.hW_factor.cuda()
            self.hE_factor = self.hE_factor.cuda()
            self.hNE_factor = self.hNE_factor.cuda()
            self.hSE_factor = self.hSE_factor.cuda()
            self.hSW_factor = self.hSW_factor.cuda()
            self.hNW_factor = self.hNW_factor.cuda()
            if isinstance(self.hN_bias, nn.Parameter):
                self.hN_bias = self.hN_bias.cuda()
                self.hS_bias = self.hS_bias.cuda()
                self.hW_bias = self.hW_bias.cuda()
                self.hE_bias = self.hE_bias.cuda()
                self.hNE_bias = self.hNE_bias.cuda()
                self.hSE_bias = self.hSE_bias.cuda()
                self.hSW_bias = self.hSW_bias.cuda()
                self.hNW_bias = self.hNW_bias.cuda()

        # diff_im = x
 
        dx = 1; dy = 1; dd = math.sqrt(2)

        hN = self.hN_weight * self.hN_factor
        hS = self.hS_weight * self.hS_factor
        hW = self.hW_weight * self.hW_factor
        hE = self.hE_weight * self.hE_factor
        hNE = self.hNE_weight * self.hNE_factor
        hSE = self.hSE_weight * self.hSE_factor
        hSW = self.hSW_weight * self.hSW_factor
        hNW = self.hNW_weight * self.hNW_factor

        if torch.cuda.is_available():
            hN = hN.cuda()
            hS = hS.cuda()
            hW = hW.cuda()
            hE = hE.cuda()
            hNE = hNE.cuda()
            hSE = hSE.cuda()
            hSW = hSW.cuda()
            hNW = hNW.cuda()

        for i in range(self.num_iter):
            nablaN = F.conv2d(diff_im, hN, self.hN_bias, padding=1, groups=self.channels)
            nablaS = F.conv2d(diff_im, hS, self.hS_bias, padding=1, groups=self.channels)
            nablaW = F.conv2d(diff_im, hW, self.hW_bias, padding=1, groups=self.channels)
            nablaE = F.conv2d(diff_im, hE, self.hE_bias, padding=1, groups=self.channels)
            nablaNE = F.conv2d(diff_im, hNE, self.hNE_bias, padding=1, groups=self.channels)
            nablaSE = F.conv2d(diff_im, hSE, self.hSE_bias, padding=1, groups=self.channels)
            nablaSW = F.conv2d(diff_im, hSW, self.hSW_bias, padding=1, groups=self.channels)
            nablaNW = F.conv2d(diff_im, hNW, self.hNW_bias, padding=1, groups=self.channels)
 
            cN = 0; cS = 0; cW = 0; cE = 0; cNE = 0; cSE = 0; cSW = 0; cNW = 0
 
            if self.option == 1:
                cN = torch.exp(-(nablaN/self.kappa)**2)
                cS = torch.exp(-(nablaS/self.kappa)**2)
                cW = torch.exp(-(nablaW/self.kappa)**2)
                cE = torch.exp(-(nablaE/self.kappa)**2)
                cNE = torch.exp(-(nablaNE/self.kappa)**2)
                cSE = torch.exp(-(nablaSE/self.kappa)**2)
                cSW = torch.exp(-(nablaSW/self.kappa)**2)
                cNW = torch.exp(-(nablaNW/self.kappa)**2)
            elif self.option == 2:
                cN = 1/(1+(nablaN/self.kappa)**2)
                cS = 1/(1+(nablaS/self.kappa)**2)
                cW = 1/(1+(nablaW/self.kappa)**2)
                cE = 1/(1+(nablaE/self.kappa)**2)
                cNE = 1/(1+(nablaNE/self.kappa)**2)
                cSE = 1/(1+(nablaSE/self.kappa)**2)
                cSW = 1/(1+(nablaSW/self.kappa)**2)
                cNW = 1/(1+(nablaNW/self.kappa)**2)
 
            diff_im = diff_im + self.delta_t * (
 
                (1/dy**2)*cN*nablaN +
                (1/dy**2)*cS*nablaS +
                (1/dx**2)*cW*nablaW +
                (1/dx**2)*cE*nablaE +
 
                (1/dd**2)*cNE*nablaNE +
                (1/dd**2)*cSE*nablaSE +
                (1/dd**2)*cSW*nablaSW +
                (1/dd**2)*cNW*nablaNW
            )
 
        return diff_im


class ImgDenoising(nn.Module):
    def __init__(self, channels=3, norm_cfg='IN', activation_cfg='LeakyReLU'):
        super(ImgDenoising, self).__init__()

        self.gaussian = GaussianBlurConv()
        self.anisodiff = AnisoDiff2D()
        self.flow = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.norm = Norm_layer(norm_cfg, channels)

    def forward(self, x):
        
        x_gauss = self.gaussian(x)
        x_anisodiff = self.anisodiff(x)
        flow = torch.cat([x, x_gauss, x_anisodiff], dim=1)
        flow = self.norm(self.flow(flow))
        attn = torch.sigmoid(flow)
        x = x * attn

        return x


class ModelSys(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print("ModelSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths,
        depths_decoder,drop_path_rate,num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.ImgDenoising = ImgDenoising()

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        
        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))), dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                depth=depths[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
        

        # build middle decoder layers1
        self.layers_up1 = nn.ModuleList()
        for i_layer1 in range(self.num_layers-1):
            if i_layer1 ==0 :
                layer_up1 = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-2-i_layer1)),
                patches_resolution[1] // (2 ** (self.num_layers-2-i_layer1))), dim=int(embed_dim * 2 ** (self.num_layers-2-i_layer1)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up1 = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-2-i_layer1)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-2-i_layer1)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-2-i_layer1))),
                                depth=depths[(self.num_layers-2-i_layer1)],
                                num_heads=num_heads[(self.num_layers-2-i_layer1)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers-2-i_layer1)]):sum(depths[:(self.num_layers-2-i_layer1) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpand if (i_layer1 < self.num_layers - 2) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_up1.append(layer_up1)

        # build middle decoder layers2
        self.layers_up2 = nn.ModuleList()
        for i_layer2 in range(self.num_layers-2):
            if i_layer2 ==0 :
                layer_up2 = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-3-i_layer2)),
                patches_resolution[1] // (2 ** (self.num_layers-3-i_layer2))), dim=int(embed_dim * 2 ** (self.num_layers-3-i_layer2)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up2 = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-3-i_layer2)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-3-i_layer2)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-3-i_layer2))),
                                depth=depths[(self.num_layers-3-i_layer2)],
                                num_heads=num_heads[(self.num_layers-3-i_layer2)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers-3-i_layer2)]):sum(depths[:(self.num_layers-3-i_layer2) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpand if (i_layer2 < self.num_layers - 3) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_up2.append(layer_up2)


        self.norm = norm_layer(self.num_features)
        self.norm_up= norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #Encoder and Bottleneck and middle layer
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for inx, layer in enumerate(self.layers):
            if inx == 1:
                x_num2 = x
                for i, layer_up2 in enumerate(self.layers_up2):
                    if i ==0:
                        x_num2 = layer_up2(x_num2)
                    else:
                        x_num2 = torch.cat([x_num2, x_downsample[inx-i]], -1)
                        x_num2 = self.concat_back_dim[i+2](x_num2)
                        x_downsample[inx-i] = x_num2
                        x_num2 = layer_up2(x_num2)

            if inx == 2:
                x_num1 = x
                for j, layer_up1 in enumerate(self.layers_up1):
                    if j ==0:
                        x_num1 = layer_up1(x_num1)
                    else:
                        x_num1 = torch.cat([x_num1, x_downsample[inx-j]], -1)
                        x_num1 = self.concat_back_dim[j+1](x_num1)
                        x_downsample[inx-j] = x_num1
                        x_num1 = layer_up1(x_num1)

            x_downsample.append(x)
            x = layer(x)    

        x = self.norm(x)  # B L C
  
        return x, x_downsample

    #Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x,x_downsample[3-inx]],-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C
  
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        if self.final_upsample=="expand_first":
            x = self.up(x)
            x = x.view(B,4*H,4*W,-1)
            x = x.permute(0,3,1,2) #B,C,H,W
            x = self.output(x)
            
        return x

    def forward(self, x):
        x = self.ImgDenoising(x)
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x,x_downsample)
        x = self.up_x4(x)

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
