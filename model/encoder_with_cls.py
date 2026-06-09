import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
# from utils.layers import EnEmbedding,DataEmbedding_inverted,FullAttention
from torch.nn import TransformerEncoder,TransformerEncoderLayer
import math
from back_bone_cls.model import clsmodel_encoder,clsmodel_projection
from torch.autograd import Function
from model.moe import AMS
from model.downsample import DownsampleAndFuse
from model.freq_trans_enc import freq_trans_enc
from model.CC import ContrastiveVAECell
class encoder_with_cls(nn.Module):
    def __init__(self, args):
        self.test=False
        super(encoder_with_cls, self).__init__()
        self.window_len = args.window_len
        self.n_vars = args.dec_in
        self.n_layers = args.n_layers
        self.use_moe=args.use_moe
        self.use_cls_to_moe=args.use_cls_to_moe
        self.use_downsample=args.use_downsample
        self.without_freq=args.without_freq
        self.trans_freq=args.trans_freq
        self.use_mix_enc_moe=args.use_mix_enc_moe
        self.use_MSE=args.use_MSE
        
        print(f"Model Configurations:")
        print(f"  use_moe:            {self.use_moe}")
        print(f"  use_cls_to_moe:     {self.use_cls_to_moe}")
        print(f"  use_downsample:     {self.use_downsample}")
        print(f"  without_freq:       {self.without_freq}")
        print(f"  trans_freq:         {self.trans_freq}")
        print(f"  use_mix_enc_moe:         {self.use_mix_enc_moe}")
        print(f"  use_MSE:         {self.use_MSE}")
        self.encoder = ContrastiveVAECell(args)
        if self.use_mix_enc_moe is True:
        # 加载原始权重
            # state_dict =torch.load('/home/zhangzuoyuan/NCN/pretrained_models/1000-contrastive-mix_mixcls-15-30')
            state_dict =torch.load(f'/home/zhangzuoyuan/NCN/pretrained_models/mix_encoder/mix_mixcls-' + f'use_moe={str(self.use_moe)}'+ f'-use_cls_to_moe={str(False)}'+ f'use_downsample={str(self.use_downsample)}'+ f'trans_freq={str(self.trans_freq)}'+ f'without_freq={str(self.without_freq)}')
            # strict=False 表示允许部分参数缺失）
            self.encoder.load_state_dict(state_dict)
        else:
            # state_dict =torch.load('/home/zhangzuoyuan/NCN/pretrained_models/1000-contrastive-mix_mixcls-15-30')
            state_dict =torch.load(f'/home/zhangzuoyuan/NCN/pretrained_models/mix_encoder/mix_mixcls-' + f'use_moe={str(self.use_moe)}'+ f'-use_cls_to_moe={str(False)}'+ f'use_downsample={str(self.use_downsample)}'+ f'trans_freq={str(self.trans_freq)}'+ f'without_freq={str(self.without_freq)}')
            # strict=False 表示允许部分参数缺失）
            self.encoder.load_state_dict(state_dict, strict=False)
        self.cls_encoder = clsmodel_encoder(args)
        self.projection = clsmodel_projection(args)
        self.cls_encoder.load_state_dict(torch.load("/home/zhangzuoyuan/NCN/pretrained_models/1000-contrastive-mix_mixcls/feature_encoder"))
        self.projection.load_state_dict(torch.load("/home/zhangzuoyuan/NCN/pretrained_models/1000-contrastive-mix_mixcls/projection"))

        
        #针对原模型得到的分类最终结果的特征维度映射
        if self.use_mix_enc_moe is True:
            self.projecter_cls2= nn.Sequential(
            nn.Linear(4,64)
            )
            self.projecter_cls_embedding= nn.Sequential(
                nn.Linear(4,64),
                nn.ReLU(),
                nn.Linear(64,8)
            )
            self.cls_embedding_softmax=nn.Softmax(dim=-1)
            self.mix_enc_moe_share = AMS(origin_len=128 * 2,input_shape=[128 * 2,1],pred_len=128*2,num_experts=4,top_k=2,use_cls_to_moe=self.use_cls_to_moe)
            self.mix_enc_moe_sparse = AMS(origin_len=128 * 2,input_shape=[128 * 2,1],pred_len=128*2,num_experts=8,top_k=4,use_cls_to_moe=self.use_cls_to_moe,Sparse=True)
            self.regressor = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(128*2+64, 64)
                )
        else:
            self.projecter_cls2= nn.Sequential(
            nn.Linear(4,64)
            )
            # if self.use_cls_to_moe is True or self.use_mix_enc_moe is True:
            #     self.projecter_cls_embedding= nn.Sequential(
            #         nn.ReLU(),
            #         nn.Linear(64,4)
            #     )
            if self.without_freq is True:
                self.regressor = nn.Sequential(
                    nn.Linear(128+64, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64)
                )
            else:
                self.regressor = nn.Sequential(
                    nn.Linear(128 * 2+64, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64)
                )
        if  self.use_MSE is True:
            self.regressor_final = nn.Sequential(
                    nn.Linear(64,148)
                )
        

    def forward(self, x_aug1, x_aug2, x_augt_1, x_augt_2):
        x_temp=x_aug1
        x_cls=self.cls_encoder(x_temp)
        x_cls_temp=self.projection(x_cls)
        #！X_cls_temp是分类结果
        x_cls=self.projecter_cls2(x_cls_temp)
        if self.use_mix_enc_moe is True:
            cls_embedding=x_cls
            x_aug1,x_aug2=self.encoder(x_aug1,x_aug2,x_augt_1, x_augt_2)
            mix_feature=torch.cat((x_aug1, x_aug2), dim=1)
            #512*256 512*4
            mix_feature = mix_feature.unsqueeze(1)
            cls_embedding=cls_embedding.unsqueeze(1)
            cls_embedding=self.cls_embedding_softmax(cls_embedding)
            #512*1*256 512*1*4
            mix_feature1,moe_loss1,gates_share=self.mix_enc_moe_share(x=mix_feature,time_embedding=mix_feature,cls_embedding=cls_embedding)
            mix_feature2,moe_loss2,gates_sparse=self.mix_enc_moe_sparse(x=mix_feature,time_embedding=mix_feature,cls_embedding=cls_embedding)
            #512*1*128
            #在最后一个维度相加
            mix_feature=mix_feature2+mix_feature1
            moe_loss=moe_loss2+moe_loss1
            mix_feature = mix_feature.squeeze(1)
            #512*128
            raw_x=torch.cat((mix_feature,x_cls), dim=1)
            features=self.regressor(raw_x)
            if self.use_MSE is True:
                output=self.regressor_final(features)
                if self.test is True:
                    return features,output,moe_loss,gates_share,gates_sparse
                return features,output,moe_loss
            return features,moe_loss
        else:
            if self.use_cls_to_moe is True:
                cls_embedding=self.projecter_cls_embedding(x_cls)
                x_aug1,x_aug2,moe_loss=self.encoder(x_aug1,x_aug2,x_augt_1, x_augt_2,cls_embedding)
            else:
                x_aug1,x_aug2=self.encoder(x_aug1,x_aug2,x_augt_1, x_augt_2)
            if self.without_freq is True:
                raw_x = torch.cat((x_aug1,x_cls), dim=1)
                features = self.regressor(raw_x)
            else:
                raw_x = torch.cat((x_aug1, x_aug2,x_cls), dim=1)
                features = self.regressor(raw_x)
            if self.use_MSE is True:
                output=self.regressor_final(features)
                return features,output
        return features
