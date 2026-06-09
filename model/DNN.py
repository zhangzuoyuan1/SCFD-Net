import torch
import torch.nn as nn
import torch.nn.functional as F

class CasualConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(CasualConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        x = F.pad(x,(self.kernel_size-1,0),'constant',0)
        x = self.conv(x)
        return x

class GlobalPooling1d(nn.Module):
    def __init__(self):
        super(GlobalPooling1d, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.pool(x)
        return x.view(x.size(0), -1)

class Encoder(nn.Module):
    def __init__(self,window_len,pred_len):
        super(Encoder, self).__init__()
        self.window_len = window_len
        self.pred_len = pred_len
        self.model = nn.Sequential(
            CasualConv1d(2,16,3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            CasualConv1d(16, 8, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            CasualConv1d(8, 8, 3),
            nn.ReLU(),
            GlobalPooling1d(),
            nn.Linear(8, 140),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(140, self.pred_len)
            # model = Sequential()
            # model.add(layers.Conv1D(16, 3, activation='relu', padding='causal',
            #                         input_shape=(None, Input_train.shape[-1])))
            # model.add(layers.MaxPooling1D(3))
            # model.add(layers.Conv1D(8, 3, activation='relu', padding='causal'))
            # model.add(layers.MaxPooling1D(3))
            # model.add(layers.Conv1D(8, 3, activation='relu', padding='causal'))
            # model.add(layers.GlobalMaxPooling1D())
            # model.add(layers.Dense(140, activation='relu'))
            # model.add(layers.Dropout(rate=0.2))
            # model.add(layers.Dense(len(voltage)))
            # model.summary()
            )

    def forward(self, x):
        x = self.model(x)
        return x

class CellModel(nn.Module):
    def __init__(self, args):
        super(CellModel, self).__init__()
        self.window_len = args.window_len
        self.pred_len = args.pred_len
        self.encoder = Encoder(self.window_len, self.pred_len)

    def forward(self, x):
        x = self.encoder(x.transpose(1, 2))
        return x