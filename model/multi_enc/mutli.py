import torch
import torch.nn.functional as F
from torch import nn
from model.multi_enc.encoder import Encoder

class TSAD(nn.Module):
    '''The TSAD model: 用于时间序列异常检测的模型，融合多个领域的特征。'''
    
    def __init__(self, args,input_dims=2, output_dims=8, hidden_dims=64, depth=10):
        """
        初始化 TSAD 模型

        参数:
        input_dims (int): 不是输入数据的维度。当成值为1的固定参数即可
        output_dims (int): 编码器的输出的维度，默认为32
        hidden_dims (int): 编码器的隐藏层的维度，默认为64
        depth (int): 模型的深度，默认为10
        """
        super().__init__()
        
        # 定义三个 Encoder 模块，分别用于不同领域的输入数据处理
        self.net_general = Encoder(input_dims, output_dims, hidden_dims, depth)  # 一般的时间序列输入(B, T, 1)->(B, T, 8)
        self.net_freq = Encoder(input_dims*3, output_dims, hidden_dims, depth)  # 频域的输入(B, T, 3)->(B, T, 8)
        self.net_resid = Encoder(input_dims, output_dims, hidden_dims, depth)  # 残差输入(B, T, 1)->(B, T, 32)
        self.net_wavelet = Encoder(input_dims*(3 + 1), output_dims, hidden_dims, depth)  # 小波输入(B, T, 5)->(B, T, 32)

        # 定义全连接层用于处理各领域的特征
        self.projector_t = nn.Sequential(
                # nn.Linear(8*(self.window_len+1), 512),
                nn.Linear(8 *10, 256),
                nn.LeakyReLU(),
                nn.Linear(256,64),
                # nn.ReLU(),
            )

    def forward(self, ts_repr, fft_repr, resid_repr,wavelet_repr):
        """
        前向传播

        参数:
        ts (Tensor): 输入的一般时间序列数据，形状为 (B, T, 1)
        fft (Tensor): 输入的频域数据，形状为 (B, T, 3)
        resid (Tensor): 输入的残差数据，形状为 (B, T, 1)
        wavelet (Tensor): 输入的小波数据，形状为 (B, T, 4)

        返回:
        Tensor: 模型的输出，形状为 (D, B, T)，其中 D 为领域数
        """
        repr_list = []
        # 通过各自的 Encoder 处理输入数据
        if hasattr(self, 'net_general'):
            ts_repr = self.net_general(ts_repr)  # 处理一般时间序列数据(B, T, 1)->(B, T, 8)
            repr_list.append(ts_repr.unsqueeze(0))
        if hasattr(self, 'net_freq'):
            fft_repr = self.net_freq(fft_repr)  # 处理频域数据(B, T, 3)->(B, T, 8)
            repr_list.append(fft_repr.unsqueeze(0))
        if hasattr(self, 'net_resid'):
            resid_repr = self.net_resid(resid_repr)  # 处理残差数据(B, T, 1)->(B, T, 8)
            repr_list.append(resid_repr.unsqueeze(0))
        if hasattr(self, 'net_wavelet'):
            wavelet_repr = self.net_wavelet(wavelet_repr)  # 处理残差数据(B, T, 5)->(B, T, 8)
            repr_list.append(wavelet_repr.unsqueeze(0))

        # 将各领域的表示拼接起来，形状为 (D, B, T, 32),D为启用的域的个数
        repr = torch.cat(repr_list, dim=0)  # D * B * T * 8
        repr = repr.permute(0, 1, 3, 2)  # 将 (D, B, T, C) → (D, B, C, T)
        D=repr.shape[0]
        B=repr.shape[1]
        repr=repr.reshape(D, B, -1)
        # 通过全连接层处理拼接后的表示
        repr=self.projector_t(repr) #(D,B,64)
        output= torch.cat((repr[0], repr[1], repr[2], repr[3]), dim=-1)#(B,256)
        return repr,output
