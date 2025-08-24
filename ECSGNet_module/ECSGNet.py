import torch
from torch import nn
import torch.nn.functional as F
import math
from ECSGNet_module.sgformer_branch import Block, Head, PatchMerging
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .pos_embed import get_2d_sincos_pos_embed
from functools import partial
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from .EMSAFusion import EMSAFusion
from .CNNnet_branch import CNNnet_branch

class ECSGformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
        super().__init__()
        # sgformer branch
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.num_patches = img_size // 4
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.embed_dim = embed_dims
        for i in range(num_stages):
            if i == 0:
                patch_embed = Head(embed_dims[0])
            else:
                patch_embed = PatchMerging(dim=embed_dims[i - 1],
                                           out_dim=embed_dims[i])
            block = nn.ModuleList([Block(
                dim=embed_dims[i], mask=True if (j % 2 == 1 and i < num_stages - 1) else False, num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches * self.num_patches, embed_dims[0]))  # fixed sin-cos embedding

        # cnn branch
        self.cnn_branch = CNNnet_branch()

        # fusion module
        self.fusion = EMSAFusion(dim=embed_dims[3])

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches,
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, y):
        B = x.shape[0]
        mask = None
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            if i == 0:
                x += self.pos_embed
            for blk in block:
                x, mask = blk(x, H, W, mask)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        y = self.cnn_branch(y)
        x_out = self.fusion(x, y)
        C = x_out.size(1)
        x_out = x_out.reshape(B, C, -1).permute(0, 2, 1).contiguous()
        x_out = x_out.mean(dim=1)
        return x_out

    def forward(self, x, y):
        x = self.forward_features(x, y)
        feat = x
        x = self.head(x)

        return feat, x

@register_model
def ECSGNet_s(pretrained=False, **kwargs):
    model = ECSGformer(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 4, 16, 1], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model