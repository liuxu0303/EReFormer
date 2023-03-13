import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from torch.nn import init
from .vit import trunc_normal_
from .swin import SwinTransformer, SwinBlock, PatchSplitting, CrossSwinBlock
from .submodules import GRViT

from .model_util import *


class TransformerRecurrent(nn.Module):
    """
    
    """
    def __init__(self, EReFormer_kwargs):
        super(TransformerRecurrent, self).__init__()
        
        self.embed_dim = EReFormer_kwargs['embed_dim']
        self.window_size = EReFormer_kwargs['window_size']
        self.img_size = EReFormer_kwargs['img_size']
        self.patch_size = EReFormer_kwargs['patch_size']
        self.in_chans = EReFormer_kwargs['in_chans']
        self.depth = EReFormer_kwargs['depth']
        self.num_heads = EReFormer_kwargs['num_heads']
        self.upsampling_method = EReFormer_kwargs['upsampling_method']
        self.pretrained = EReFormer_kwargs['pretrained'] 
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = self.norm_layer(self.embed_dim)
        self.num_output_channels = 1
        # self.features=[192, 384, 768, 1536]
        self.features=[96, 192, 384, 768]
        decoder_drop = 0.1
      
        #encoder
        self.SwinTransformer = SwinTransformer(embed_dims=self.embed_dim, depths=self.depth, \
                                               num_heads=self.num_heads, window_size=self.window_size, \
                                                   pretrained=self.pretrained)
        
        #GRViT
        self.GRViT_0 = GRViT(dim=self.features[0], num_heads=self.num_heads[0], mlp_ratio=2., qkv_bias=False, \
                           drop=decoder_drop, attn_drop=0., proj_drop=0.)
        self.GRViT_1 = GRViT(dim=self.features[1], num_heads=self.num_heads[1], mlp_ratio=2., qkv_bias=False, \
                           drop=decoder_drop, attn_drop=0., proj_drop=0.)
        self.GRViT_2 = GRViT(dim=self.features[2], num_heads=self.num_heads[2], mlp_ratio=2., qkv_bias=False, \
                           drop=decoder_drop, attn_drop=0., proj_drop=0.)
        self.GRViT_3 = GRViT(dim=self.features[3], num_heads=self.num_heads[3], mlp_ratio=2., qkv_bias=False, \
                           drop=decoder_drop, attn_drop=0., proj_drop=0.)
            
        self.pos_embed_0 = nn.Parameter(torch.zeros(
            1, (224 // 4)*(224 // 4), self.features[0]))
        self.pos_embed_1 = nn.Parameter(torch.zeros(
            1, (224 // 8)*(224 // 8), self.features[1]))
        self.pos_embed_2 = nn.Parameter(torch.zeros(
            1, (224 // 16)*(224 // 16), self.features[2]))
        self.pos_embed_3 = nn.Parameter(torch.zeros(
            1, (224 // 32)*(224 // 32), self.features[3]))
        
        trunc_normal_(self.pos_embed_0, std=.02)
        trunc_normal_(self.pos_embed_1, std=.02)
        trunc_normal_(self.pos_embed_2, std=.02)
        trunc_normal_(self.pos_embed_3, std=.02)
        
        #res    
        self.resblock_0 = SwinBlock(embed_dims=self.features[3], num_heads=self.num_heads[3], \
                                         feedforward_channels=2 * self.features[3], window_size=self.window_size,\
                                         shift=False, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)  
        self.resblock_1 = SwinBlock(embed_dims=self.features[3], num_heads=self.num_heads[3], \
                                         feedforward_channels=2 * self.features[3], window_size=self.window_size, \
                                         shift=True, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)
        
        #decoder1
        self.decoder1_block0 = SwinBlock(embed_dims=self.features[3], num_heads=self.num_heads[3], \
                                         feedforward_channels=2 * self.features[3], window_size=self.window_size,\
                                         shift=False, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.) 
        self.decoder1_block1 = SwinBlock(embed_dims=self.features[3], num_heads=self.num_heads[3], \
                                         feedforward_channels=2 * self.features[3], window_size=self.window_size, \
                                         shift=True, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)
        
        self.PatchSplitting1 = PatchSplitting(in_channels=self.features[3], out_channels=self.features[2],\
                                             output_size=(self.img_size[0]//16, self.img_size[1]//16), stride=2)
        
        #STF1
        self.decoder2_query0 = CrossSwinBlock(embed_dims=self.features[2], num_heads=self.num_heads[2], \
                                             feedforward_channels=4 * self.features[2], window_size=self.window_size,\
                                             shift=False, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)
        self.decoder2_query1 = CrossSwinBlock(embed_dims=self.features[2], num_heads=self.num_heads[2], \
                                             feedforward_channels=4 * self.features[2], window_size=self.window_size,\
                                             shift=True, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)
        
        #decoder2
        self.decoder2_block0 = SwinBlock(embed_dims=self.features[2], num_heads=self.num_heads[2], \
                                         feedforward_channels=4 * self.features[2], window_size=self.window_size,\
                                         shift=False, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.) 
        self.decoder2_block1 = SwinBlock(embed_dims=self.features[2], num_heads=self.num_heads[2], \
                                         feedforward_channels=4 * self.features[2], window_size=self.window_size,\
                                         shift=True, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.) 
            
        self.PatchSplitting2 = PatchSplitting(in_channels=self.features[2], out_channels=self.features[1],\
                                             output_size=(self.img_size[0]//8, self.img_size[1]//8), stride=2)
        
        #STF2
        self.decoder3_query0 = CrossSwinBlock(embed_dims=self.features[1], num_heads=self.num_heads[1], \
                                             feedforward_channels=4 * self.features[1], window_size=self.window_size,\
                                             shift=False, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)
        self.decoder3_query1 = CrossSwinBlock(embed_dims=self.features[1], num_heads=self.num_heads[1], \
                                             feedforward_channels=4 * self.features[1], window_size=self.window_size,\
                                             shift=True, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)
         
        #decoder3
        self.decoder3_block0 = SwinBlock(embed_dims=self.features[1], num_heads=self.num_heads[1], \
                                         feedforward_channels=4 * self.features[1], window_size=self.window_size,\
                                         shift=False, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.) 
        self.decoder3_block1 = SwinBlock(embed_dims=self.features[1], num_heads=self.num_heads[1], \
                                         feedforward_channels=4 * self.features[1], window_size=self.window_size,\
                                         shift=True, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.) 
            
        self.PatchSplitting3 = PatchSplitting(in_channels=self.features[1], out_channels=self.features[0],\
                            output_size=(int(math.ceil(self.img_size[0] / 4)), int(math.ceil(self.img_size[1] / 4))), stride=2)
            
        #STF3
        self.decoder4_query0 = CrossSwinBlock(embed_dims=self.features[0], num_heads=self.num_heads[0], \
                                             feedforward_channels=4 * self.features[0], window_size=self.window_size,\
                                             shift=False, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)
        self.decoder4_query1 = CrossSwinBlock(embed_dims=self.features[0], num_heads=self.num_heads[0], \
                                             feedforward_channels=4 * self.features[0], window_size=self.window_size,\
                                             shift=True, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)
         
        #decoder4
        self.decoder4_block0 = SwinBlock(embed_dims=self.features[0], num_heads=self.num_heads[0], \
                                         feedforward_channels=4 * self.features[0], window_size=self.window_size,\
                                         shift=False, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.) 
        self.decoder4_block1 = SwinBlock(embed_dims=self.features[0], num_heads=self.num_heads[0], \
                                         feedforward_channels=4 * self.features[0], window_size=self.window_size,\
                                         shift=True, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.) 
            
        self.Unflatten = nn.Unflatten(2, torch.Size([int(math.ceil(self.img_size[0] / self.patch_size)),\
                                                     int(math.ceil(self.img_size[1] / self.patch_size))]))  
        
        
        #head
        self.conv2d0 = nn.Conv2d(self.features[0], 96, kernel_size=3, stride=1, padding=1)
        self.head = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
              
        self.num_decoders = EReFormer_kwargs['num_decoders']
        self.states = [None] * self.num_decoders
    
    def _resize_pos_embed(self, posemb, gs_h, gs_w):
        # posemb_tok, posemb_grid = (
        #     posemb[:, : 1],
        #     posemb[0, 1 :],
        # )
        posemb_grid = posemb

        # gs_old = int(math.sqrt(len(posemb_grid)))
        gs_old = int(math.sqrt(posemb_grid.shape[1]))

        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        
        posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bicubic")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

        # posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        posemb = posemb_grid

        return posemb
        

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """
        # vit encoder
        B, _, H, W = x.shape
        stage_feature = self.SwinTransformer(x)
        
        stage1 = stage_feature[0]
        stage2 = stage_feature[1]
        stage3 = stage_feature[2]
        stage4 = stage_feature[-1] #B, H/32*W/32, 8C
        
        pos_embed_0 = self._resize_pos_embed(
            self.pos_embed_0, int(math.ceil(H / 4)), int(math.ceil(W / 4))
        )
        pos_embed_1 = self._resize_pos_embed(
            self.pos_embed_1, int(math.ceil(H / 8)), int(math.ceil(W / 8))
        )
        pos_embed_2 = self._resize_pos_embed(
            self.pos_embed_2, int(math.ceil(H / 16)), int(math.ceil(W / 16))
        )
        pos_embed_3 = self._resize_pos_embed(
            self.pos_embed_3, int(math.ceil(H / 32)), int(math.ceil(W / 32))
        )
        
        #GRViT
        stage1, states_0 = self.GRViT_0(stage1, self.states[0], pos_embed_0)
        self.states[0] = states_0
        stage2, states_1 = self.GRViT_1(stage2, self.states[1], pos_embed_1)
        self.states[1] = states_1
        stage3, states_2 = self.GRViT_2(stage3, self.states[2], pos_embed_2)
        self.states[2] = states_2
        stage4, states_3 = self.GRViT_3(stage4, self.states[3], pos_embed_3)
        self.states[3] = states_3
        
        #res
        hw_shape = (int(math.ceil(H / 32)), int(math.ceil(W / 32)))
        res = self.resblock_0(stage4, hw_shape)
        res = self.resblock_1(res, hw_shape) #B, H/32*W/32, 8C
        
        #decoder1 
        fuse_stage4 = res + stage4
        hw_shape = (int(math.ceil(H / 32)), int(math.ceil(W / 32)))
        decoder_stage4 = self.decoder1_block0(fuse_stage4, hw_shape)
        decoder_stage4 = self.decoder1_block1(decoder_stage4, hw_shape) #B, H/32*W/32, 8C
        up_decoder_stage4 = self.PatchSplitting1(decoder_stage4) #B, H/16*W/16, 4C
        
        #STF1
        hw_shape = (int(math.ceil(H / 16)), int(math.ceil(W / 16)))
        query_stage3 = self.decoder2_query0(up_decoder_stage4, stage3, hw_shape)
        query_stage3 = self.decoder2_query1(query_stage3, stage3, hw_shape)
        fuse_stage3 = up_decoder_stage4 + query_stage3
        
        #decoder2
        decoder_stage3 = self.decoder2_block0(fuse_stage3, hw_shape)
        decoder_stage3 = self.decoder2_block1(decoder_stage3, hw_shape) #B, H/16*W/16, 4C
        up_decoder_stage3 = self.PatchSplitting2(decoder_stage3) #B, H/8*W/8, 2C
        
        #STF2 
        hw_shape = (int(math.ceil(H / 8)), int(math.ceil(W / 8)))
        query_stage2 = self.decoder3_query0(up_decoder_stage3, stage2, hw_shape)
        query_stage2 = self.decoder3_query1(query_stage2, stage2, hw_shape)
        fuse_stage2 = up_decoder_stage3 + query_stage2
        
        #decoder3
        hw_shape = (int(math.ceil(H / 8)), int(math.ceil(W / 8)))
        decoder_stage2 = self.decoder3_block0(fuse_stage2, hw_shape)
        decoder_stage2 = self.decoder3_block1(decoder_stage2, hw_shape) #B, H/8*W/8, 2C
        up_decoder_stage2 = self.PatchSplitting3(decoder_stage2) #B, H/4*W/4, C
        
        #STF3 
        hw_shape = (int(math.ceil(H / 4)), int(math.ceil(W / 4)))
        query_stage1 = self.decoder4_query0(up_decoder_stage2, stage1, hw_shape)
        query_stage1 = self.decoder4_query1(query_stage1, stage1, hw_shape)
        fuse_stage1 = up_decoder_stage2 + query_stage1
        
        #decoder4
        hw_shape = (int(math.ceil(H / 4)), int(math.ceil(W / 4)))
        decoder_stage1 = self.decoder4_block0(fuse_stage1, hw_shape)
        decoder_stage1 = self.decoder4_block1(decoder_stage1, hw_shape) #B, H/4*W/4, C
        
        decoder = self.norm(decoder_stage1)
        decoder = decoder.transpose(1, 2)
        decoder = self.Unflatten(decoder) #B, C, H/4, W/4
            
        out = self.conv2d0(decoder)
        out = F.interpolate(out, size=(self.img_size[0], self.img_size[1]), mode='bicubic', align_corners=True)
        # out = F.interpolate(out, size=(256, 336), mode='bicubic', align_corners=True)
        depth = self.head(out)
        # print(depth.shape)
                    
        return { 'pred_depth':depth}
