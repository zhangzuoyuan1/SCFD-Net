import torch
import torch.nn as nn

import torch.nn.functional as F

class TemperatureScaling(nn.Module):
    def __init__(self, temperature=0.01):
        super(TemperatureScaling, self).__init__()
        # 将温度系数声明为可学习参数
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, logits):
        # 使用温度缩放后的 logits 应用 softmax
        return F.softmax(logits / self.temperature, dim=-1)

class TopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, noise_epsilon=1e-5,use_cls_to_moe=False,Spare=False):
        super(TopKGating, self).__init__()
        #采用线性层学习门控权重
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k
        self.Spare=Spare
        self.noise_epsilon = noise_epsilon
        self.num_experts = num_experts
        self.w_noise = nn.Parameter(torch.zeros(num_experts, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)
        self.use_cls_to_moe=use_cls_to_moe
        self.softmax_cls=TemperatureScaling(temperature=0.01)

    def decompostion_tp(self, x, alpha=10):
        # x [batch_size, num_expert]
        output = torch.zeros_like(x)
        # [batch_size]
        kth_largest_val, _ = torch.kthvalue(x, self.num_experts - self.top_k + 1)
        #找到top_K的门控分数边界
        # [batch_size, num_expert]
        kth_largest_mat = kth_largest_val.unsqueeze(1).expand(-1, self.num_experts)
        mask = x < kth_largest_mat
        x = self.softmax(x)
        # output[mask] = alpha * torch.log(x[mask] + 1)
        output[~mask] = alpha * (torch.exp(x[~mask]) - 1)
        # Ablation Spare MoE
        if self.Spare is True:
            output[mask] = 0
        else:
            output[mask] = alpha * torch.log(x[mask] + 1)
        # [batch_size, num_expert]
        return output
    def is_module_frozen(self, module):
        """检测指定模块是否解冻"""
        for param in module.parameters():
            if param.requires_grad:
                return True
        return False
    def forward(self, x,cls_embedding=None):
        # [batch_size, seq_len]

        x = self.gate(x)
        x=self.softmax(x)
        clean_logits = x
        # [batch_size, num_experts]
        if self.use_cls_to_moe is True or cls_embedding is not None:
            clean_logits=self.softmax_cls(x*cls_embedding)
        # if self.is_module_frozen(self.gate):
        #     raw_noise_stddev = x @ self.w_noise
        #     noise_stddev = ((self.softplus(raw_noise_stddev) + self.noise_epsilon))
        #     noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        #     logits = noisy_logits
        # else:
        logits = clean_logits


        logits = self.decompostion_tp(logits)
        gates = self.softmax(logits)

        # random order
        # indices = torch.randperm(gates.size(0))
        # shuffled_gates = gates[indices]

        # average
        # value = 1.0 / x.shape[1]
        # gates = torch.full(x.shape, value, device=x.device)
            
        return gates


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.2):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class AMS(nn.Module):
    def __init__(self, origin_len,input_shape, pred_len, ff_dim=128, dropout=0.2, loss_coef=10000.0, num_experts=8, top_k=4,use_cls_to_moe=False,Sparse=False):
        super(AMS, self).__init__()
        # input_shape[0] = seq_len    input_shape[1] = feature_num
        self.num_experts = num_experts
        self.top_k = top_k
        self.pred_len = pred_len
        self.use_cls_to_moe=use_cls_to_moe
        self.gating = TopKGating(input_dim=origin_len, num_experts=num_experts, top_k=top_k, use_cls_to_moe=use_cls_to_moe,Spare=Sparse)
        self.Sparse=Sparse
        self.experts = nn.ModuleList(
            [Expert(input_shape[0], pred_len, hidden_dim=ff_dim, dropout=dropout) for _ in range(num_experts)])
        self.loss_coef = loss_coef
        assert (self.top_k <= self.num_experts)
        self.projecter_cls_embedding= nn.Sequential(
                nn.ReLU(),
                nn.Linear(64,self.num_experts)
            )

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)
    # Step 1: 定义均匀性损失（以 KL 为例）
    def kl_uniformity_loss(self,router_logits, k):
        probs = torch.softmax(router_logits, dim=-1)
        topk_vals, _ = torch.topk(probs, k, dim=-1)
        uniform = torch.ones_like(topk_vals) / k
        kl = (topk_vals * (torch.log(topk_vals + 1e-10) - torch.log(uniform + 1e-10))).sum(-1)
        return kl.mean()
    def forward(self, x, time_embedding,cls_embedding=None):
        # [batch_size, feature_num, seq_len]
        batch_size = x.shape[0]
        feature_num = x.shape[1]
        if cls_embedding is not None:
            cls_embedding=self.projecter_cls_embedding(cls_embedding)
        
        # [feature_num, batch_size, seq_len]
        x = torch.transpose(x, 0, 1)
        time_embedding = torch.transpose(time_embedding, 0, 1)
        if cls_embedding is not None:
            cls_embedding=torch.transpose(cls_embedding,0,1)
        output = torch.zeros(feature_num, batch_size, self.pred_len).to(x.device)
        #[feature_num,batch_size,pred_len]
        loss = 0
        
        for i in range(feature_num):
            input = x[i]
            time_info = time_embedding[i]
            # x[i]  [batch_size, seq_len]
            #门控机制
            if self.use_cls_to_moe is True or cls_embedding is not None:
                cls_info=cls_embedding[i]
                gates = self.gating(time_info,cls_embedding=cls_info)
            else:
                gates = self.gating(time_info)
            
            # expert_outputs [batch_size, num_experts, pred_len]
            expert_outputs = torch.zeros(self.num_experts, batch_size, self.pred_len).to(x.device)
            #对每个特征，都有一堆专家进行处理，然后再融合
            for j in range(self.num_experts):
                expert_outputs[j, :, :] = self.experts[j](input)
            # expert_outputs [num_experts，batch_size, pred_len]
            expert_outputs = torch.transpose(expert_outputs, 0, 1)
            
            
            # gates [batch_size, num_experts, pred_len]
            gates = gates.unsqueeze(-1).expand(-1, -1, self.pred_len)
            # batch_output [batch_size, pred_len]
            batch_output = (gates * expert_outputs).sum(1)
            output[i, :, :] = batch_output
            gates_final=gates
            gates_final=gates_final.mean(dim=-1)
            importance = gates.sum(0)
        
            if self.Sparse is True:
                loss += self.loss_coef * self.cv_squared(importance)+0.5*self.loss_coef*self.kl_uniformity_loss(router_logits=gates,k=self.top_k)
            else:
                loss += self.loss_coef * self.cv_squared(importance)+0.5*self.loss_coef*self.kl_uniformity_loss(router_logits=gates,k=self.num_experts)

        # [feature_num, batch_size, seq_len]
        output = torch.transpose(output, 0, 1)
        # [batch_size, feature_num, seq_len]
        # print('moe loss',loss)
        return output, loss,gates_final