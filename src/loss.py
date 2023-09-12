import torch
import torch.nn as nn
import torch.nn.functional as F

def one_LPN_output(outputs, labels, criterion, block=4):

    sm = nn.Softmax(dim=1)
    num_part = block
    score = 0
    loss = 0
    for i in range(num_part):
        part = outputs[i]
        score += sm(part)
        loss += criterion(part, labels)

    _, preds = torch.max(score.data, 1)

    return preds, loss

def nceloss(feature_d, feature_ds, feature_s, feature_sd, labels_d, labels_s):
    feature_d = F.normalize(feature_d, dim=1)
    feature_ds = F.normalize(feature_ds, dim=1)
    feature_sd = F.normalize(feature_sd, dim=1)
    feature_s = F.normalize(feature_s, dim=1)

    features1 = torch.cat([feature_d.unsqueeze(1), feature_ds.unsqueeze(1)], dim=1)
    features2 = torch.cat([feature_sd.unsqueeze(1), feature_s.unsqueeze(1)], dim=1)

    infonce = SupConLoss(temperature=0.1)

    nceloss1 = infonce(features1, labels_d)
    nceloss2 = infonce(features2, labels_s)
    
    return nceloss1 + nceloss2

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels_column = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            #mask = torch.eq(labels, labels.T).float().to(device)
            mask = (labels_column == labels).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # print(contrast_count,"contrast_count")
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print(contrast_feature.shape,"contrast_feature")

        # print(f"contrast_feature {contrast_feature.size()}")
        # print(f"mask {mask.size()}")
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print(f"mask repeat {mask.size()}")
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # print(f"logits {logits.size()}")
        # assert(0)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def decouple_loss(y1, y2, scale_loss=1/32, lambd=0.0013, e1=0, e2=0):
    y1 = y1.squeeze() if len(y1.shape) > 2 else y1
    y2 = y2.squeeze() if len(y2.shape) > 2 else y2

    batch_size = y1.size(0)
    c = y1.T @ y2
    c.div_(batch_size)
    on_diag = torch.diagonal(c)
    p_on = (1 - on_diag) / 2
    on_diag = torch.pow(p_on, e1) * torch.pow(torch.diagonal(c).add_(-1), 2)
    on_diag = on_diag.sum().mul(scale_loss)

    off_diag = off_diagonal(c)
    p_off = torch.abs(off_diag)
    off_diag = torch.pow(p_off, e2) * torch.pow(off_diagonal(c), 2)
    off_diag = off_diag.sum().mul(scale_loss)
    loss = on_diag + off_diag * lambd
    return loss, on_diag, off_diag * lambd