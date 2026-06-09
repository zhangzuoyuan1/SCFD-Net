import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from utils.layers import EnEmbedding,DataEmbedding_inverted,FullAttention
from torch.nn import TransformerEncoder,TransformerEncoderLayer
import math
from back_bone_cls.model import clsmodel_encoder,clsmodel_projection
from torch.autograd import Function
from model.moe import AMS
from model.downsample import DownsampleAndFuse
from model.freq_trans_enc import freq_trans_enc
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
  
class EncoderLayer(nn.Module):
    def __init__(self,window_len,dropout=0.1,activation='relu'):
        super(EncoderLayer, self).__init__()
        self.window_len = window_len
        self.conv_encoder_t = nn.Sequential(
            #SepConv1d(8, 64, 3, 1, 1),
            SepConv1d(2, 64, 3, 1, 1),
            SE_Connect(64),
            SepConv1d(64, 32, 3, 1, 1),
            SE_Connect(32),
            #nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3),
            SepConv1d(32, 16, 3, 1, 1),
            SE_Connect(16),
            SepConv1d(16, 8, 3, 1, 1),
            SE_Connect(8)
            # B*C'*l
        )
        # self.CrossAttention = FullAttention(attention_dropout=0.1)
        # self.conv1 = nn.Conv1d(in_channels=self.window_len+1, out_channels=128, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=128, out_channels=self.window_len+1, kernel_size=1)
        # self.dropout = dropout
        # self.dropout = nn.Dropout(self.dropout)
        # self.activation = activation
        # self.activation = F.relu if self.activation == "relu" else F.gelu

    def forward(self, x, cross):
        #print(x.shape)
        x = self.conv_encoder_t(x)
        # x_glb_ori = x[:,:,-1].unsqueeze(-1)
        # x_glb_att = self.CrossAttention(x_glb_ori, cross, cross)[0]
        # #print(x_glb_att.shape)
        # x_glb = x_glb_att + x_glb_ori
        # x = torch.concat((x[:,:,:-1], x_glb), dim=-1)
        # x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        # y = self.dropout(self.conv2(x).transpose(-1, 1))
        return x

class Encoder(nn.Module):
    def __init__(self,window_len,n_layers=2,norm_layer=None):
        super(Encoder, self).__init__()
        self.window_len = window_len
        self.n_layers = n_layers
        self.layers = nn.ModuleList([EncoderLayer(window_len) for _ in range(n_layers)])
        self.norm = norm_layer

    def forward(self, x, cross):
        # print(self.norm)
        for layer in self.layers:
            x = layer(x, cross)

        if self.norm is not None:
            x = self.norm(x)
        return x

class TSFreqCNNEncoder(nn.Module): 
    def __init__(self,window_len,n_vars,n_layers=2,norm_layer=None,use_moe=False,use_cls_to_moe=False):
        super(TSFreqCNNEncoder, self).__init__()
        self.window_len = window_len
        self.n_layers = n_layers
        self.norm = norm_layer
        self.use_moe=use_moe
        self.use_cls_to_moe=use_cls_to_moe
        self.encoder = Encoder(self.window_len,self.n_layers,self.norm)
        # print("use_moe=",use_moe)
        if use_moe is True:
            self.moe=AMS(origin_len=window_len,input_shape=[4*(self.window_len//3),2],pred_len=4*(self.window_len//3),use_cls_to_moe=self.use_cls_to_moe)
        self.projector_t = nn.Sequential(
                Flatten(),
                # nn.Linear(8*(self.window_len+1), 512),
                nn.Linear(8 * (self.window_len//3), 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                # nn.ReLU(),
            )
    def forward(self, x_aug1, x_aug1_t,cls_embedding=None):
        # x_aug1 512*30*2
        # cls_embedding [batch_size, num_experts]
        # x_emb = self.EnEmbedding(x_aug1)
        # x_aug1 = self.encoder(x_emb, x_aug1_t)
        time_embedding=x_aug1.transpose(1,2)
        #time_embedding 512*2*30
        batch_size=x_aug1.shape[0]
        x_aug1 = self.encoder(x_aug1.transpose(1,2), x_aug1_t)
        # [512, 8, 10]
        if self.use_moe is True:
            x_aug1= x_aug1.reshape(batch_size, 2, -1)
            if self.use_cls_to_moe is True and cls_embedding is not None:
                x_aug1,moe_loss=self.moe(x_aug1,time_embedding=time_embedding,cls_embedding=cls_embedding)
            else:
                x_aug1,moe_loss=self.moe(x_aug1,time_embedding=time_embedding)
            x_aug1.transpose(1,2)
        x_aug1_proj = self.projector_t(x_aug1)
        if self.use_moe is True:
            return x_aug1_proj,moe_loss
        return x_aug1_proj 

class MuSigma(nn.Module):
    def __init__(self, input_dim):
        super(MuSigma, self).__init__()
        
        self.layer1 = nn.Sequential(
            # nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            # nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            # nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
        )
    def forward(self,  embedding):
        mu = self.layer1(embedding)
        sigma = self.layer2(embedding)     + 1e-6    # elu() + 1 ensures the covariance matrix is positive definite.
        # sigma = torch.nn.Softplus()( sigma  ) + 1e-6
        return mu, sigma    


class DecoderNet(nn.Module):
    def __init__(self, init_dim, num_filters, k_size,size):
        super(DecoderNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_filters , 1   * (init_dim - 3 * (k_size - 1))),
            # nn.Linear(num_filters ,  1  * (init_dim)),
            nn.ReLU()
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose1d(1, num_filters, k_size, 1, 0),
            nn.ReLU(),
            SE_Connect(num_filters, 8),
            nn.ConvTranspose1d(num_filters, num_filters, k_size, 1, 0 ),
            SE_Connect(num_filters, 8),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters, num_filters, k_size, 1, 0 ),
            SE_Connect(num_filters, 8),
            nn.ReLU(),
        )
        # self.layer2 = nn.Linear(256, size)
        self.gru = nn.GRU(num_filters, num_filters, batch_first=True, num_layers=2)
        self.layer2 =  nn.Sequential(
            nn.Linear(num_filters,size)
        )
        self.init_dim = init_dim
        self.num_filters = num_filters
        self.k_size = k_size 
        self.act = nn.ReLU()


    def forward(self, x):
        x = self.layer(x)
        # x = x.view(-1, 1, self.init_dim )
        x = x.view(-1, 1, self.init_dim - 3 * (self.k_size - 1))
        x = self.convt(x)

        x = x.permute(0,2,1)
        x, _ = self.gru(x)
        x = self.act(x)
        x = self.layer2(x)
        return x


class ContrastiveVAECell(nn.Module): 
    def __init__(self,args):
        super(ContrastiveVAECell, self).__init__()
        self.batch_size = args.batch_size
        self.window_len = args.window_len
        self.n_vars = args.dec_in
        self.n_layers = args.n_layers
        
        self.use_moe=args.use_moe
        self.use_cls_to_moe=args.use_cls_to_moe
        self.use_downsample=args.use_downsample
        self.without_freq=args.without_freq
        self.trans_freq=args.trans_freq
        print(f"Model Configurations:")
        print(f"  use_moe:            {self.use_moe}")
        print(f"  use_cls_to_moe:     {self.use_cls_to_moe}")
        print(f"  use_downsample:     {self.use_downsample}")
        print(f"  without_freq:       {self.without_freq}")
        print(f"  trans_freq:         {self.trans_freq}")
        self.encoder = TSFreqCNNEncoder(window_len=self.window_len,n_vars=self.n_vars,n_layers=self.n_layers,use_moe=self.use_moe,use_cls_to_moe=self.use_cls_to_moe)
        # self.musigma  = MuSigma(128)
        # self.musigma2 = MuSigma(128)
        # self.decoder = DecoderNet( init_dim=30, num_filters=128, k_size=3,size =2)
        # self.temperature = 0.07
        # self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        if self.use_downsample is True:
            self.downsample=DownsampleAndFuse(in_channels=self.n_vars, out_channels=self.n_vars)
        if self.trans_freq is True:
            self.freq_enc=freq_trans_enc(seq_len=self.window_len,c_in=self.n_vars)

    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_(0,0.1).to(mean.get_device())
        eps = Variable(eps)
        return eps.mul(std).add_(mean)
    
    def forward(self, x1, x2, x1_t, x2_t,cls_embedding=None):
        batch_size = x1.size(0)
        # x1 512*30*2
        if self.use_downsample is True:
            x1= self.downsample(x1)
        if self.without_freq is True:
            if self.use_moe is True:
                x_aug1,moe_loss1= self.encoder(x1, x1_t,cls_embedding)
                moe_loss=moe_loss1
                x_aug2=x_aug1
                return x_aug1, x_aug2,moe_loss
            else:
                x_aug1 = self.encoder(x1, x1_t)
                x_aug2 = x_aug1
        elif self.trans_freq is True:
            if self.use_moe is True:
                x_aug1,moe_loss1= self.encoder(x1, x1_t,cls_embedding)
                moe_loss=moe_loss1
                x_aug2=self.freq_enc(x1)
                return x_aug1, x_aug2,moe_loss
            else:
                x_aug1 = self.encoder(x1, x1_t)
                x_aug2=self.freq_enc(x1)
        else: 
            if self.use_moe is True:
                x_aug1,moe_loss1= self.encoder(x1, x1_t,cls_embedding)
                x_aug2,moe_loss2 = self.encoder(x2, x2_t,cls_embedding)
                moe_loss=moe_loss1+moe_loss2
                return x_aug1, x_aug2,moe_loss
            else:
                x_aug1 = self.encoder(x1, x1_t,cls_embedding)
                x_aug2 = self.encoder(x2, x2_t,cls_embedding) 
        return x_aug1, x_aug2

    # def forward(self, x1, x2 ):
    #     x_aug1 = self.encoder(x1  ) 
    #     x_aug2 = self.encoder(x2  )



    #     mu1, logvar1 = self.musigma(x_aug1)
    #     mu2, logvar2 = self.musigma2(x_aug2)

    #     sample1  = self.reparametrize(mu1, logvar1)
    #     sample2  = self.reparametrize(mu2, logvar2 )

    #     recon1 = self.decoder(sample1)
    #     recon2 = self.decoder(sample2)

    #     loss_1 = self.loss_f(recon1, x1, mu1, logvar1, lamda = 1)
    #     loss_2 = self.loss_f(recon2, x2, mu2, logvar2, lamda = 1)
    #     lamda = -1
    #     # print('VAE loss', loss_1.item(), loss_2.item(), (  loss_1 +  0.5 * loss_2).item())
    #     return x_aug1, x_aug2,  10 ** lamda *    ( loss_2  +  loss_1 )

     


    def  loss_f(self, recon_x, x, mu, logvar, lamda= 1):
        cit = nn.MSELoss(reduction='none')
        cr_loss1 = torch.sum(cit(recon_x[:,:,0], x[:,:,0]), 1)
        cr_loss2 = torch.sum(cit(recon_x[:,:,1], x[:,:,1]), 1)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        # print('kld-loss', torch.mean( 10 **lamda * KLD).item(), torch.mean(cr_loss1).item(), torch.mean(cr_loss2).item())
        return torch.mean(  (cr_loss1 + cr_loss2) * 0.5  +     KLD)
         

# conv_encoder_t.0.layers.0.depthwise.weight
# conv_encoder_t.0.layers.0.depthwise.bias
# conv_encoder_t.0.layers.0.pointwise.weight
# conv_encoder_t.0.layers.0.pointwise.bias
# conv_encoder_t.1.linear1.weight
# conv_encoder_t.1.linear1.bias
# conv_encoder_t.1.linear2.weight
# conv_encoder_t.1.linear2.bias
# conv_encoder_t.2.layers.0.depthwise.weight
# conv_encoder_t.2.layers.0.depthwise.bias
# conv_encoder_t.2.layers.0.pointwise.weight
# conv_encoder_t.2.layers.0.pointwise.bias
# conv_encoder_t.3.linear1.weight
# conv_encoder_t.3.linear1.bias
# conv_encoder_t.3.linear2.weight
# conv_encoder_t.3.linear2.bias
# conv_encoder_t.5.layers.0.depthwise.weight
# conv_encoder_t.5.layers.0.depthwise.bias
# conv_encoder_t.5.layers.0.pointwise.weight
# conv_encoder_t.5.layers.0.pointwise.bias
# conv_encoder_t.6.linear1.weight
# conv_encoder_t.6.linear1.bias
# conv_encoder_t.6.linear2.weight
# conv_encoder_t.6.linear2.bias
# conv_encoder_t.7.layers.0.depthwise.weight
# conv_encoder_t.7.layers.0.depthwise.weight
# conv_encoder_t.7.layers.0.depthwise.bias
# conv_encoder_t.7.layers.0.depthwise.bias
# conv_encoder_t.7.layers.0.pointwise.weight
# conv_encoder_t.7.layers.0.pointwise.weight
# conv_encoder_t.7.layers.0.pointwise.bias
# conv_encoder_t.7.layers.0.pointwise.bias
# conv_encoder_t.8.linear1.weight
# conv_encoder_t.8.linear1.weight
# conv_encoder_t.8.linear1.bias
# conv_encoder_t.8.linear1.bias
# conv_encoder_t.8.linear2.weight
# conv_encoder_t.8.linear2.weight
# conv_encoder_t.8.linear2.bias
# conv_encoder_t.8.linear2.bias
# projector_t.0.weight
# projector_t.0.weight
# projector_t.0.bias
# projector_t.0.bias
# projector_t.2.weight
# projector_t.2.weight
# projector_t.2.bias
# projector_t.2.bias
class DownstreamFromCVAE(nn.Module):
    def __init__(self, pretrained_path, output_size=140):
        super(DownstreamFromCVAE, self).__init__()
        self.pretrained_path = pretrained_path
        if pretrained_path is not None:
            self.encoder = TSFreqCNNEncoder()
            self.encoder.load_state_dict(torch.load(pretrained_path))
            # for param in self.encoder.parameters():
            #     param.requires_grad = False
            # for k, v in self.encoder.named_parameters():
            #     # print(k)
            #     if 'conv_encoder_t.8' in k  or 'conv_encoder_t.7' in k   or 'conv_encoder_t.6' in k  or 'conv_encoder_t.5' in k  or  'projector_t' in k:
            #         print(k)
            #         v.requires_grad = True
            # print(self.encoder)
        else:
            self.encoder = TSFreqCNNEncoder()
 
         
        self.regressor = nn.Sequential(
                nn.Linear(128*2, 128),
                nn.ReLU(),
                 
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_size),
            )
         

    def forward(self, x_aug1, x_aug2 ):
        x_aug1 = self.encoder(x_aug1  ) 
        x_aug2 = self.encoder(x_aug2  ) 
        raw_x = torch.cat((x_aug1, x_aug2 ), 1)
        pred = self.regressor(raw_x)
        return  x_aug1, x_aug2, pred   
  


class DANNCell(nn.Module):
    def __init__(self, pretrained_path):
        super(DANNCell, self).__init__()
        self.pretrained_path = pretrained_path
         
        self.encoder = TSFreqCNNEncoder()
        self.encoder.load_state_dict(torch.load(pretrained_path))
        self.domain_cls = nn.Sequential(
                nn.Linear(128*2, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )
         

    def forward(self, x_aug1, x_aug2, alpha ):
        x_aug1 = self.encoder(x_aug1  ) 
        x_aug2 = self.encoder(x_aug2  ) 
        raw_x = torch.cat((x_aug1, x_aug2 ), 1)
        reverse_feature = ReverseLayerF.apply(raw_x, alpha)
        pred = self.domain_cls(reverse_feature)
        return  x_aug1, x_aug2, pred   
 

''' Conv1d + BatchNorm1d + ReLU
'''

class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.
    
    The separable convlution is a method to reduce number of the parameters 
    in the deep learning network for slight decrease in predictions quality.
    """
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
    
    
    
class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad,  
                 activ=lambda: nn.ReLU()):
    
        super().__init__()
       # assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
         
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
       # print('bp_conv1')
        return self.layers(x)
    
    
    
class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.contiguous().view(x.size(0), -1)
        return x.view(-1)

class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels) 
    def forward(self, x):
        return   (F.relu(self.conv(x)))


''' The SE connection of 1D case.
'''


class SE_Connect(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        assert channels % s == 0, "{} % {} != 0".format(channels, s)
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


class CellModel(nn.Module):
    def __init__(self, args):
        super(CellModel, self).__init__()
        self.pretrained_path = args.pretrained_path
        self.window_len = args.window_len
        self.pred_len = args.pred_len
        self.n_vars = args.dec_in
        self.n_layers = args.n_layers

        if self.pretrained_path is not None:
            self.encoder = TSFreqCNNEncoder(self.window_len,self.n_vars,self.n_layers)
            state_dict = torch.load(self.pretrained_path, map_location='cuda:0')
            # 提取 backbone 的参数
            Encoder_state_dict = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
            self.encoder.load_state_dict(Encoder_state_dict)

        else:
            self.encoder = TSFreqCNNEncoder(self.window_len,self.n_vars,self.n_layers)

        self.regressor = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.pred_len),
        )

    def forward(self, x_aug1, x_aug2, x_augt_1, x_augt_2):
        x_aug1 = self.encoder(x_aug1, x_augt_1)
        x_aug2 = self.encoder(x_aug2, x_augt_2)
        raw_x = torch.cat((x_aug1, x_aug2), 1)
        pred = self.regressor(raw_x)
        return x_aug1, x_aug2, pred
from model.multi_enc.mutli import TSAD
from utils.decomp import decomp
class CellModel_multi_enc(nn.Module):
    def __init__(self, args):
        super(CellModel_multi_enc, self).__init__()
        self.args=args
        self.pretrained_path = args.pretrained_path
        self.window_len = args.window_len
        self.pred_len = args.pred_len
        self.n_vars = args.dec_in
        self.n_layers = args.n_layers


        self.encoder = TSAD(self.args)
        state_dict = torch.load('/home/zhangzuoyuan/NCN/pretrained_models/mix_encoder/mix_mixcls-use_moe=False-use_cls_to_moe=Falseuse_downsample=Falsetrans_freq=Falsewithout_freq=Falseuse_multi_enc=True')
        self.encoder.load_state_dict(state_dict)
        self.regressor = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.pred_len),
        )

    def forward(self, x_aug1, x_aug2, x_augt_1, x_augt_2):
        aug1_ts,aug1_fft,aug1_resid,aug1_wavelet=decomp(x_aug1)
        raw_repr,raw_x=self.encoder(aug1_ts,aug1_fft,aug1_resid,aug1_wavelet)
        pred = self.regressor(raw_x)
        return x_aug1, x_aug2, pred
 
 