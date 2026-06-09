
import torch
import torch.nn.functional as F

def cls_loss(features, labels=None, mask=None):
    """
    features: hidden vector of shape [bsz, feat_dim]. e.g., [512, 128]
    labels: ground truth of shape [bsz]. e.g., [512]
    mask: contrastive mask of shape [bsz, bsz], mask[i][j]=1 if sample j is a positive for sample i.
    
    Returns:
        A loss scalar.
    """
    device = features.device
    
    # 如果 features 维度大于 2，则展平
    if len(features.shape) > 2:
        features = features.view(features.shape[0], -1)

    batch_size = features.shape[0]

    # 防止同时传入 labels 和 mask
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        # 默认 mask 是单位矩阵（只有自己是正例）
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        # 构造基于标签的 mask
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    # 计算相似度矩阵
    anchor_dot_contrast = torch.matmul(features, features.T) / 0.07  # 温度系数 0.07

    # 数值稳定处理（减去最大值）
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # 排除 self-contrast（自己和自己的对比）
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # 计算 exp(logits) 并加 eps 防止除零
    exp_logits = torch.exp(logits) * logits_mask
    eps = 1e-30
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eps)

    # 只计算正例的平均 log prob
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + eps)

    # 最终损失
    loss = -mean_log_prob_pos
    loss = loss.mean()  # 对 batch 求均值

    return loss 