import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsampleAndFuse(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(DownsampleAndFuse, self).__init__()
        
        # 第一个卷积层，用于提取特征并进行第一次下采样
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=4, kernel_size=3, stride=2, padding=1)
        
        # 第二个卷积层，进一步提取特征并进行第二次下采样
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1)
        
        # 融合卷积层，用于融合不同尺度的特征
        self.fuse_conv = nn.Conv1d(in_channels=12, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 输入x的形状是(512, 30, 2)，首先需要调整形状以适应1D卷积的要求，变为(512, 2, 30)。
        x = x.permute(0, 2, 1)
        
        # 第一次卷积和下采样
        x1 = F.relu(self.conv1(x))
        
        # 第二次卷积和下采样
        x2 = F.relu(self.conv2(x1))
        
        # 在通道维度上拼接x1和x2，注意需要先通过插值或池化调整x1和x2到相同的尺寸。
        # 这里简单起见，假设x2的尺寸适合直接与x1拼接，实际情况可能需要额外处理。
        x_concat = torch.cat([F.interpolate(x1, size=x2.shape[-1]), x2], dim=1)
        
        # 融合特征
        fused = F.relu(self.fuse_conv(x_concat))
        
        # 使用插值恢复到原始的时间维度大小
        fused_upsampled = F.interpolate(fused, size=x.shape[-1])
        
        # 残差连接，将融合后的特征与原始输入相加
        output = fused_upsampled + x
        
        # 返回最终结果，如果需要，可以再调整回原始的时间维度顺序。
        return output.permute(0, 2, 1)