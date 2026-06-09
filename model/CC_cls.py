import torch
import torch.nn as nn
from model.encoder_with_cls import encoder_with_cls
from model.encoder_with_moe import encoder_with_moe
from model.encoder_with_cls_moe import encoder_with_cls_moe
class CellModel_cls_multi_enc(nn.Module):
    def __init__(self, args):
        super(CellModel_cls_multi_enc, self).__init__()
        self.args=args
        self.pretrained_path = args.pretrained_path
        self.window_len = args.window_len
        self.pred_len = args.pred_len
        self.n_vars = args.dec_in
        self.n_layers = args.n_layers
        self.use_MSE=args.use_MSE
        args.use_MSE=False
        self.encoder=encoder_with_cls(args)
        args.use_MSE=self.use_MSE
        self.use_moe=args.use_moe
        self.use_cls_to_moe=args.use_cls_to_moe
        self.use_downsample=args.use_downsample
        self.without_freq=args.without_freq
        self.trans_freq=args.trans_freq
        self.use_mix_enc_moe=args.use_mix_enc_moe
        print(f"all_Model Configurations:")
        print(f"  use_moe:            {self.use_moe}")
        print(f"  use_cls_to_moe:     {self.use_cls_to_moe}")
        print(f"  use_downsample:     {self.use_downsample}")
        print(f"  without_freq:       {self.without_freq}")
        print(f"  trans_freq:         {self.trans_freq}")
        print(f"  use_mix_enc_moe:         {self.use_mix_enc_moe}")
        print(f"  use_MSE:         {self.use_MSE}")
        self.encoder.load_state_dict(torch.load(f'./pretrained_models/encoder_with_cls/best_model'
                   + f'use_moe={str(self.args.use_moe)}'
                   + f'-use_cls_to_moe={str(self.args.use_cls_to_moe)}'
                   + f'use_downsample={str(self.args.use_downsample)}'
                   + f'trans_freq={str(self.args.trans_freq)}'
                   + f'without_freq={str(self.args.without_freq)}'
                   + f'use_mix_enc_moe={str(self.args.use_mix_enc_moe)}'
                   + f'use_MSE={str(self.args.use_MSE)}'
                   +'.pth'),strict=False)
        # 加载预训练权重
        pretrained_dict = torch.load(f'./pretrained_models/encoder_with_cls/best_model'
                   + f'use_moe={str(self.args.use_moe)}'
                   + f'-use_cls_to_moe={str(self.args.use_cls_to_moe)}'
                   + f'use_downsample={str(self.args.use_downsample)}'
                   + f'trans_freq={str(self.args.trans_freq)}'
                   + f'without_freq={str(self.args.without_freq)}'
                   + f'use_mix_enc_moe={str(self.args.use_mix_enc_moe)}'
                   + f'use_MSE={str(self.args.use_MSE)}'
                   +'.pth')

        # 获取当前模型的状态字典
        model_dict = self.encoder.state_dict()
        ignored_keys = set(pretrained_dict.keys()) - set(model_dict.keys())
        if ignored_keys:
            print("\n⚠️ Ignored keys (not loaded):")
            for key in ignored_keys:
                print(f"  - {key}")
        self.regressor = nn.Sequential(
            nn.Linear(64, self.pred_len)
        )

    def forward(self, x_aug1, x_aug2=None, x_augt_1=None, x_augt_2=None):
        if self.args.use_mix_enc_moe is True:
            features,moe_loss=self.encoder( x_aug1, x_aug2, x_augt_1, x_augt_2)
        else:    
            features=self.encoder( x_aug1, x_aug2, x_augt_1, x_augt_2)
        pred = self.regressor(features)
        if self.args.use_mix_enc_moe is True:
            return x_aug1, x_aug2, pred,moe_loss
        else:
            return x_aug1, x_aug2, pred
        
class CellModel_moe_multi_enc(nn.Module):
    def __init__(self, args):
        super(CellModel_moe_multi_enc, self).__init__()
        self.args=args
        self.pretrained_path = args.pretrained_path
        self.window_len = args.window_len
        self.pred_len = args.pred_len
        self.n_vars = args.dec_in
        self.n_layers = args.n_layers
        self.use_MSE=args.use_MSE
        args.use_MSE=False
        self.encoder=encoder_with_moe(args)
        args.use_MSE=self.use_MSE
        self.use_moe=args.use_moe
        self.use_cls_to_moe=args.use_cls_to_moe
        self.use_downsample=args.use_downsample
        self.without_freq=args.without_freq
        self.trans_freq=args.trans_freq
        self.use_mix_enc_moe=args.use_mix_enc_moe
        print(f"all_Model Configurations:")
        print(f"  use_moe:            {self.use_moe}")
        print(f"  use_cls_to_moe:     {self.use_cls_to_moe}")
        print(f"  use_downsample:     {self.use_downsample}")
        print(f"  without_freq:       {self.without_freq}")
        print(f"  trans_freq:         {self.trans_freq}")
        print(f"  use_mix_enc_moe:         {self.use_mix_enc_moe}")
        print(f"  use_MSE:         {self.use_MSE}")
        self.encoder.load_state_dict(torch.load(f'./pretrained_models/encoder_with_cls/best_model'
                   + f'use_moe={str(self.args.use_moe)}'
                   + f'-use_cls_to_moe={str(self.args.use_cls_to_moe)}'
                   + f'use_downsample={str(self.args.use_downsample)}'
                   + f'trans_freq={str(self.args.trans_freq)}'
                   + f'without_freq={str(self.args.without_freq)}'
                   + f'use_mix_enc_moe={str(self.args.use_mix_enc_moe)}'
                   + f'use_MSE={str(self.args.use_MSE)}'+'only_moe'
                   +'.pth'),strict=False)
        # 加载预训练权重
        pretrained_dict = torch.load(f'./pretrained_models/encoder_with_cls/best_model'
                   + f'use_moe={str(self.args.use_moe)}'
                   + f'-use_cls_to_moe={str(self.args.use_cls_to_moe)}'
                   + f'use_downsample={str(self.args.use_downsample)}'
                   + f'trans_freq={str(self.args.trans_freq)}'
                   + f'without_freq={str(self.args.without_freq)}'
                   + f'use_mix_enc_moe={str(self.args.use_mix_enc_moe)}'
                   + f'use_MSE={str(self.args.use_MSE)}'+'only_moe'
                   +'.pth')

        # 获取当前模型的状态字典
        model_dict = self.encoder.state_dict()
        ignored_keys = set(pretrained_dict.keys()) - set(model_dict.keys())
        if ignored_keys:
            print("\n⚠️ Ignored keys (not loaded):")
            for key in ignored_keys:
                print(f"  - {key}")
        self.regressor = nn.Sequential(
            nn.Linear(64, self.pred_len)
        )

    def forward(self, x_aug1, x_aug2, x_augt_1, x_augt_2):
        if self.args.use_mix_enc_moe is True:
            features,moe_loss=self.encoder( x_aug1, x_aug2, x_augt_1, x_augt_2)
        else:    
            features=self.encoder( x_aug1, x_aug2, x_augt_1, x_augt_2)
        pred = self.regressor(features)
        if self.args.use_mix_enc_moe is True:
            return x_aug1, x_aug2, pred,moe_loss
        else:
            return x_aug1, x_aug2, pred
class CellModel_cls_change_moe_multi_enc(nn.Module):
    def __init__(self, args):
        super(CellModel_cls_change_moe_multi_enc, self).__init__()
        self.args=args
        self.pretrained_path = args.pretrained_path
        self.window_len = args.window_len
        self.pred_len = args.pred_len
        self.n_vars = args.dec_in
        self.n_layers = args.n_layers
        self.use_MSE=args.use_MSE
        args.use_MSE=False
        self.encoder=encoder_with_cls_moe(args)
        args.use_MSE=self.use_MSE
        self.use_moe=args.use_moe
        self.use_cls_to_moe=args.use_cls_to_moe
        self.use_downsample=args.use_downsample
        self.without_freq=args.without_freq
        self.trans_freq=args.trans_freq
        self.use_mix_enc_moe=args.use_mix_enc_moe
        print(f"all_Model Configurations:")
        print(f"  use_moe:            {self.use_moe}")
        print(f"  use_cls_to_moe:     {self.use_cls_to_moe}")
        print(f"  use_downsample:     {self.use_downsample}")
        print(f"  without_freq:       {self.without_freq}")
        print(f"  trans_freq:         {self.trans_freq}")
        print(f"  use_mix_enc_moe:         {self.use_mix_enc_moe}")
        print(f"  use_MSE:         {self.use_MSE}")
        self.encoder.load_state_dict(torch.load(f'./pretrained_models/encoder_with_cls/best_model'
                   + f'use_moe={str(self.args.use_moe)}'
                   + f'-use_cls_to_moe={str(self.args.use_cls_to_moe)}'
                   + f'use_downsample={str(self.args.use_downsample)}'
                   + f'trans_freq={str(self.args.trans_freq)}'
                   + f'without_freq={str(self.args.without_freq)}'
                   + f'use_mix_enc_moe={str(self.args.use_mix_enc_moe)}'
                   + f'use_MSE={str(self.args.use_MSE)}'
                   +f'share={str(self.args.share)}'
                    +f'topk_share={str(self.args.topk_share)}'
                    +f'sparse={str(self.args.sparse)}'
                    +f'topk_sparse={str(self.args.topk_sparse)}'
                   +'.pth'),strict=False)
        # 加载预训练权重
        pretrained_dict = torch.load(f'./pretrained_models/encoder_with_cls/best_model'
                   + f'use_moe={str(self.args.use_moe)}'
                   + f'-use_cls_to_moe={str(self.args.use_cls_to_moe)}'
                   + f'use_downsample={str(self.args.use_downsample)}'
                   + f'trans_freq={str(self.args.trans_freq)}'
                   + f'without_freq={str(self.args.without_freq)}'
                   + f'use_mix_enc_moe={str(self.args.use_mix_enc_moe)}'
                   + f'use_MSE={str(self.args.use_MSE)}'
                   +f'share={str(self.args.share)}'
                    +f'topk_share={str(self.args.topk_share)}'
                    +f'sparse={str(self.args.sparse)}'
                    +f'topk_sparse={str(self.args.topk_sparse)}'
                   +'.pth')

        # 获取当前模型的状态字典
        model_dict = self.encoder.state_dict()
        ignored_keys = set(pretrained_dict.keys()) - set(model_dict.keys())
        if ignored_keys:
            print("\n⚠️ Ignored keys (not loaded):")
            for key in ignored_keys:
                print(f"  - {key}")
        self.regressor = nn.Sequential(
            nn.Linear(64, self.pred_len)
        )

    def forward(self, x_aug1, x_aug2, x_augt_1, x_augt_2):
        if self.args.use_mix_enc_moe is True:
            features,moe_loss=self.encoder( x_aug1, x_aug2, x_augt_1, x_augt_2)
        else:    
            features=self.encoder( x_aug1, x_aug2, x_augt_1, x_augt_2)
        pred = self.regressor(features)
        if self.args.use_mix_enc_moe is True:
            return x_aug1, x_aug2, pred,moe_loss
        else:
            return x_aug1, x_aug2, pred