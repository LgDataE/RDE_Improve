import math
import torch
import torch.nn as nn
import torch.nn.functional as F

 
def compute_sdm_per(scores, pid, logit_scale, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    t2i_cosine_theta = scores
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.sum(i2t_loss, dim=1) + torch.sum(t2i_loss, dim=1)

    return loss

def compute_TRL_per(scores, pid, margin = 0.2, tau=0.02):       
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_1 =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_2 = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    pos_1 = (alpha_1 * scores).sum(1)
    pos_2 = (alpha_2 * scores.t()).sum(1)

    neg_1 = (mask*scores).max(1)[0]
    neg_2 = (mask*scores.t()).max(1)[0]

    cost_1 = (margin + neg_1 - pos_1).clamp(min=0)
    cost_2 = (margin + neg_2 - pos_2).clamp(min=0)
    return cost_1 + cost_2

 
def compute_InfoNCE_per(scores, logit_scale):
    
    # cosine similarity as logits
    logits_per_image = logit_scale * scores
    logits_per_text = logits_per_image.t()

    p1 = F.softmax(logits_per_image, dim=1)
    p2 = F.softmax(logits_per_text, dim=1)

    loss = (- p1.diag().log() - p2.diag().log())/2    
    return loss

def compute_TAL_per(scores, pid, tau, margin):
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    loss = (-  (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  \
        +  (-  (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    
    return loss 

def compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, pid, label_hat=None, tau=0.02, margin=0.1, loss_type='TAL', logit_scale=50):

    loss_bgm, _ = compute_per_loss(i_feats, t_feats, pid, tau, margin, loss_type, logit_scale)
    loss_tse, _ = compute_per_loss(i_tse_f, t_tse_f, pid, tau, margin, loss_type, logit_scale)

    loss_bgm = (label_hat*loss_bgm).sum()
    loss_tse = (label_hat*loss_tse).sum()
    
    if loss_type in ['TAL','TRL']:
        return loss_bgm, loss_tse
    else:
        return loss_bgm/label_hat.sum(), loss_tse/label_hat.sum() # mean

def compute_per_loss(image_features, text_features, pid, tau=0.02, margin=0.2, loss_type='TAL', logit_scale=50):
    
    # # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()

    if 'TAL' in loss_type:
        per_loss = compute_TAL_per(scores, pid, tau, margin=margin)
    elif 'TRL' in loss_type:
        per_loss = compute_TRL_per(scores, pid, tau=tau, margin=margin)
    elif 'InfoNCE' in loss_type:
        per_loss = compute_InfoNCE_per(scores, logit_scale)
    elif 'SDM' in loss_type:
        per_loss = compute_sdm_per(scores, pid, logit_scale)
    else:
        exit()

    return per_loss, scores.diag()


def compute_cfam_local_sdm_loss(v_local: torch.Tensor, w_local: torch.Tensor, pid: torch.Tensor, logit_scale, epsilon: float = 1e-8):
    if v_local.dim() != 3 or w_local.dim() != 3:
        raise ValueError("v_local and w_local must be 3D tensors of shape [B, K, D].")
    if v_local.shape[:2] != w_local.shape[:2] or v_local.shape[-1] != w_local.shape[-1]:
        raise ValueError("v_local and w_local must have the same shape [B, K, D].")

    v = F.normalize(v_local, p=2, dim=-1)
    w = F.normalize(w_local, p=2, dim=-1)
    scores = (w.unsqueeze(1) * v.unsqueeze(0)).sum(dim=-1).mean(dim=-1).float()
    per = compute_sdm_per(scores, pid, logit_scale=logit_scale, epsilon=epsilon)
    return per.mean()


def compute_cfam_local_sdm_per(v_local: torch.Tensor, w_local: torch.Tensor, pid: torch.Tensor, logit_scale, epsilon: float = 1e-8):
    if v_local.dim() != 3 or w_local.dim() != 3:
        raise ValueError("v_local and w_local must be 3D tensors of shape [B, K, D].")
    if v_local.shape != w_local.shape:
        raise ValueError("v_local and w_local must have the same shape [B, K, D].")

    v = F.normalize(v_local, p=2, dim=-1)
    w = F.normalize(w_local, p=2, dim=-1)
    scores = (w.unsqueeze(1) * v.unsqueeze(0)).sum(dim=-1).mean(dim=-1).float()
    per = compute_sdm_per(scores, pid, logit_scale=logit_scale, epsilon=epsilon)
    return per, scores.diag()


def compute_cfam_match_loss(v_local: torch.Tensor, w_local: torch.Tensor, pid: torch.Tensor, matcher: nn.Module, pos_labels: torch.Tensor = None):
    if v_local.dim() != 3 or w_local.dim() != 3:
        raise ValueError("v_local and w_local must be 3D tensors of shape [B, K, D].")
    B, K, D = v_local.shape
    if w_local.shape != (B, K, D):
        raise ValueError("v_local and w_local must have the same shape [B, K, D].")

    v = F.normalize(v_local, p=2, dim=-1)
    w = F.normalize(w_local, p=2, dim=-1)
    sim = (w.unsqueeze(1) * v.unsqueeze(0)).sum(dim=-1).mean(dim=-1).float()

    pid = pid.view(B, 1)
    same_id = (pid == pid.t())
    sim_i2t = sim.masked_fill(same_id, -1e9)
    sim_t2i = sim.t().masked_fill(same_id, -1e9)

    hard_txt = sim_i2t.argmax(dim=1)
    hard_img = sim_t2i.argmax(dim=1)

    pos_v = v_local
    pos_w = w_local
    neg_v1 = v_local
    neg_w1 = w_local[hard_txt]
    neg_v2 = v_local[hard_img]
    neg_w2 = w_local

    pair_v = torch.cat([pos_v, neg_v1, neg_v2], dim=0)
    pair_w = torch.cat([pos_w, neg_w1, neg_w2], dim=0)
    if pos_labels is None:
        pos_labels = torch.ones(B, device=v_local.device)
    else:
        pos_labels = pos_labels.to(device=v_local.device, dtype=torch.float)
        if pos_labels.numel() != B:
            raise ValueError("pos_labels must have shape [B].")

    labels = torch.cat([pos_labels, torch.zeros(2 * B, device=v_local.device)], dim=0)

    logits = matcher(pair_v, pair_w).view(-1)
    return F.binary_cross_entropy_with_logits(logits, labels)



