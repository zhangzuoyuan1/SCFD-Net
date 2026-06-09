from torch import nn
from model.multi_enc.dilated_conv import DilatedConvEncoder

class Encoder(nn.Module):
    '''Encoder模块: 用于处理输入时间序列数据的编码器。'''

    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10):
        """
        初始化 Encoder 模块

        参数:
        input_dims (int): 输入数据的维度（每个时间步的特征数）
        output_dims (int): 编码后输出的维度
        hidden_dims (int): 隐藏层的维度，默认为64
        depth (int): 卷积层的深度，默认为10
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        
        # 输入层：通过全连接层将输入的维度映射到隐藏维度
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        
        # 批归一化层：对隐藏层输出进行归一化
        self.bn = nn.BatchNorm1d(hidden_dims)
        
        # 特征提取器：使用膨胀卷积（Dilated Convolution）进行特征提取
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,  # 输入的通道数
            [hidden_dims] * depth + [output_dims],  # 每层的输出维度
            kernel_size=3  # 卷积核的大小
        )
        
        # Dropout层：防止过拟合
        self.repr_dropout = nn.Dropout(p=0.1)
        self.maxpool=nn.MaxPool1d(kernel_size=3)
        
    def forward(self, x):
        """
        前向传播

        参数:
        x (Tensor): 输入数据，形状为 (B, T, input_dims)，其中 B 为批量大小，T 为时间步数，input_dims 为输入特征维度

        返回:
        Tensor: 编码后的表示，形状为 (B, T, output_dims)
        """
        # 输入通过全连接层进行转换
        x = self.input_fc(x)  # B x T x Ch，其中 Ch = hidden_dims
        
        # 转置操作，将 T 和 Ch 维度交换以适应卷积操作
        x = x.transpose(1, 2)  # B x Ch x T
        x=self.maxpool(x)
        # 批归一化处理
        x = self.bn(x)
        
        # 通过膨胀卷积特征提取器处理数据
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T，其中 Co = output_dims
        
        # 再次转置，使得输出形状为 (B, T, Co)
        x = x.transpose(1, 2)  # B x 10 x 8
        
        return x
