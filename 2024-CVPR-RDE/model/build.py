from model import objectives

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
from .bamg_modules import BottleneckFusionEncoder, MaskedGraphModelingHead
import torch
import torch.nn as nn 
import torch.nn.functional as F

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
 
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)
        self.use_bamg = getattr(self.args, "use_bamg", False)
        self.bamg_weight = getattr(self.args, "bamg_weight", 1.0)
        self.mgm_weight = getattr(self.args, "mgm_weight", 0.0)
        if self.use_bamg:
            bamg_num_tokens = getattr(self.args, "bamg_num_tokens", 4)
            bamg_depth = getattr(self.args, "bamg_depth", 2)
            bamg_num_heads = getattr(self.args, "bamg_num_heads", 8)
            bamg_mlp_ratio = getattr(self.args, "bamg_mlp_ratio", 4.0)
            bamg_dropout = getattr(self.args, "bamg_dropout", 0.1)
            self.bamg_encoder = BottleneckFusionEncoder(
                dim=self.embed_dim,
                num_bottlenecks=bamg_num_tokens,
                depth=bamg_depth,
                nhead=bamg_num_heads,
                mlp_ratio=bamg_mlp_ratio,
                dropout=bamg_dropout,
            )
            # ensure LayerNorm inside BAMG encoder uses fp16 to match CLIP features after convert_weights
            for m in self.bamg_encoder.modules():
                if isinstance(m, nn.LayerNorm):
                    m.weight.data = m.weight.data.half()
                    if m.bias is not None:
                        m.bias.data = m.bias.data.half()
            if self.mgm_weight > 0:
                mgm_hidden_dim = getattr(self.args, "mgm_hidden_dim", 0)
                if mgm_hidden_dim <= 0:
                    mgm_hidden_dim = self.embed_dim
                mgm_num_layers = getattr(self.args, "mgm_num_layers", 2)
                mgm_mask_ratio = getattr(self.args, "mgm_mask_ratio", 0.15)
                self.mgm_head = MaskedGraphModelingHead(
                    dim=self.embed_dim,
                    hidden_dim=mgm_hidden_dim,
                    num_layers=mgm_num_layers,
                    mask_ratio=mgm_mask_ratio,
                )
            else:
                self.mgm_head = None
        else:
            self.bamg_encoder = None
            self.mgm_head = None

        # masked language modeling on text (optional)
        self.mlm_weight = getattr(self.args, "mlm_weight", 0.0)
        if self.mlm_weight > 0:
            self.mlm_head = nn.Linear(self.embed_dim, self.args.vocab_size)
        else:
            self.mlm_head = None

        self.use_mgcc = getattr(self.args, "use_mgcc", False)
        self.mgcc_weight = getattr(self.args, "mgcc_weight", 1.0)
        self.mgcc_softmax_t = getattr(self.args, "mgcc_softmax_t", 1e-2)
        self.mgcc_max_words = getattr(self.args, "mgcc_max_words", 25)
        self.mgcc_rt = getattr(self.args, "mgcc_rt", 0.4)
        self.mgcc_rv = getattr(self.args, "mgcc_rv", 0.2)
        self.mgcc_aggregation_type = getattr(self.args, "mgcc_aggregation_type", "Attention")
        if self.use_mgcc:
            max_length_batch = max(1, min(self.mgcc_max_words, round(self.mgcc_rt * self.mgcc_max_words)))
            if hasattr(self.base_model.visual, "num_x") and hasattr(self.base_model.visual, "num_y"):
                max_patch = int(self.base_model.visual.num_x * self.base_model.visual.num_y)
            elif hasattr(self.base_model.visual, "attnpool") and hasattr(self.base_model.visual.attnpool, "positional_embedding"):
                max_patch = int(self.base_model.visual.attnpool.positional_embedding.shape[0] - 1)
            else:
                max_patch = 0
            num_patch = max(1, min(max_patch, round(self.mgcc_rv * max_patch))) if max_patch > 0 else 0

            self.mgcc_global_mat_weight = nn.parameter.Parameter(torch.eye(self.embed_dim), requires_grad=True)
            self.mgcc_word_logit_weight = nn.parameter.Parameter(torch.eye(max_length_batch), requires_grad=True)
            if num_patch > 0:
                self.mgcc_patch_logit_weight = nn.parameter.Parameter(torch.eye(num_patch), requires_grad=True)
            else:
                self.mgcc_patch_logit_weight = None

            if self.mgcc_aggregation_type == 'Attention':
                self.mgcc_local_mat_weight = nn.parameter.Parameter(torch.eye(self.embed_dim), requires_grad=True)
                self.mgcc_word_mat_weight = nn.parameter.Parameter(torch.eye(max_length_batch), requires_grad=True)
                self.mgcc_word_mat_weight2 = nn.parameter.Parameter(torch.eye(max_length_batch), requires_grad=True)
                if num_patch > 0:
                    self.mgcc_patch_mat_weight = nn.parameter.Parameter(torch.eye(num_patch), requires_grad=True)
                    self.mgcc_patch_mat_weight2 = nn.parameter.Parameter(torch.eye(num_patch), requires_grad=True)
                else:
                    self.mgcc_patch_mat_weight = None
                    self.mgcc_patch_mat_weight2 = None
            else:
                self.mgcc_local_mat_weight = None
                self.mgcc_patch_mat_weight = None
                self.mgcc_word_mat_weight = None
                self.mgcc_patch_mat_weight2 = None
                self.mgcc_word_mat_weight2 = None
 
        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type
 
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tse(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)   
        return i_tse_f.float()
 
    def encode_text_tse(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

    def encode_image_bamg(self, image):
        """Image encoder with BAMG bottleneck for evaluation.

        If BAMG is disabled, this falls back to the standard CLS embedding.
        """
        x, _ = self.base_model.encode_image(image)
        if not self.use_bamg or self.bamg_encoder is None:
            return x[:, 0, :].float()

        img_tokens, _ = self.bamg_encoder.forward_image(x)
        return img_tokens[:, 0, :].float()

    def encode_text_bamg(self, text):
        """Text encoder with BAMG bottleneck for evaluation.

        If BAMG is disabled, this falls back to the standard EOT embedding.
        """
        x, _ = self.base_model.encode_text(text.long())
        if not self.use_bamg or self.bamg_encoder is None:
            return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

        txt_tokens, _ = self.bamg_encoder.forward_text(x)
        idx = text.argmax(dim=-1)
        return txt_tokens[torch.arange(txt_tokens.shape[0]), idx].float()

    def mgcc_img_embedding(self, image_feats, atten_i):
        image_feats = image_feats.float()
        atten_i = atten_i.float()
        img_global = F.normalize(image_feats[:, 0, :], dim=-1)
        if not hasattr(self, 'mgcc_patch_logit_weight') or self.mgcc_patch_logit_weight is None:
            return img_global, None
        all_patch = image_feats[:, 1:, :]
        select_patch = int(self.mgcc_patch_logit_weight.size(0))
        max_patch = all_patch.size(1)
        attention_cls_part = atten_i[:, 0, 1:]
        _, indices = torch.sort(attention_cls_part, descending=True)
        k = min(select_patch, max_patch)
        selected_indices = indices[:, 0:k]
        selected_patch_embedding = []
        for i in range(selected_indices.size(0)):
            top_k_embedding = torch.index_select(all_patch[i], 0, selected_indices[i])
            if k < select_patch:
                pad = torch.zeros(select_patch - k, all_patch.size(-1), device=all_patch.device, dtype=all_patch.dtype)
                top_k_embedding = torch.cat([top_k_embedding, pad], dim=0)
            selected_patch_embedding.append(top_k_embedding.unsqueeze(0))
        patch_part = torch.cat(selected_patch_embedding, 0)
        patch_part = F.normalize(patch_part, dim=-1)
        return img_global, patch_part

    def mgcc_txt_embedding(self, text_feats, caption_ids, atten_t):
        text_feats = text_feats.float()
        caption_ids = caption_ids.long()
        atten_t = atten_t.float()

        bs, L, dim = text_feats.shape
        eot_idx = caption_ids.argmax(dim=-1)
        txt_global = text_feats[torch.arange(bs), eot_idx]
        txt_global = F.normalize(txt_global, dim=-1)

        select_token = int(self.mgcc_word_logit_weight.size(0))
        selected_word_embedding = []
        for i in range(bs):
            e = int(eot_idx[i].item())
            e = max(1, min(e, L - 1))

            all_word_embeddings_i = text_feats[i, 1:e, :]
            attention_cls_part = atten_t[i, e, 1:e]

            if attention_cls_part.numel() > self.mgcc_max_words:
                _, indices = torch.sort(attention_cls_part, descending=True)
                selected_indices = indices[: self.mgcc_max_words]
                attention_cls_part = torch.index_select(attention_cls_part, 0, selected_indices)
                all_word_embeddings_i = torch.index_select(all_word_embeddings_i, 0, selected_indices)
            elif attention_cls_part.numel() < self.mgcc_max_words:
                yhi = self.mgcc_max_words - attention_cls_part.numel()
                attention_cls_part = torch.cat((attention_cls_part, torch.zeros(yhi, device=attention_cls_part.device)), dim=0)
                all_word_embeddings_i = torch.cat((all_word_embeddings_i, torch.zeros(yhi, dim, device=all_word_embeddings_i.device)), dim=0)

            _, indices2 = torch.sort(attention_cls_part, descending=True)
            selected_indices2 = indices2[:select_token]
            top_k_embedding = torch.index_select(all_word_embeddings_i, 0, selected_indices2)
            selected_word_embedding.append(top_k_embedding.unsqueeze(0))

        word_part = torch.cat(selected_word_embedding, 0)
        word_part = F.normalize(word_part, dim=-1)
        return txt_global, word_part

    def mgcc_aggregation_fine_grained_similarity(self, patch_part, word_part):
        bs_img, num_patch, dim = patch_part.shape
        bs_text, max_length_batch, _ = word_part.shape
        fine_grained_sim_scores = torch.matmul(patch_part.view(-1, dim), word_part.view(-1, dim).t()).view(bs_img, num_patch, bs_text, max_length_batch)

        if self.mgcc_aggregation_type == 'Attention':
            softmax_t = self.mgcc_softmax_t
            fine_grained_sim_scores = torch.matmul(torch.matmul(patch_part.view(-1, dim), self.mgcc_local_mat_weight), word_part.view(-1, dim).t()).view(bs_img, num_patch, bs_text, max_length_batch)
            word_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores / softmax_t, dim=1).permute(0, 2, 3, 1), self.mgcc_patch_mat_weight).permute(0, 3, 1, 2) * fine_grained_sim_scores, dim=1)
            patch_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores / softmax_t, dim=-1), self.mgcc_word_mat_weight) * fine_grained_sim_scores, dim=-1)
            word_level_logit2 = torch.sum(torch.matmul(torch.softmax(word_level_logit / softmax_t, dim=-1), self.mgcc_word_mat_weight2) * word_level_logit, dim=-1)
            patch_level_logit2 = torch.sum(torch.matmul(torch.softmax(patch_level_logit / softmax_t, dim=1).permute(0, 2, 1), self.mgcc_patch_mat_weight2).permute(0, 2, 1) * patch_level_logit, dim=1)

        elif self.mgcc_aggregation_type == 'Max_Mean':
            word_level_logit, _ = torch.max(fine_grained_sim_scores, dim=1)
            patch_level_logit, _ = torch.max(fine_grained_sim_scores, dim=-1)
            word_level_logit2 = torch.mean(word_level_logit, dim=-1)
            patch_level_logit2 = torch.mean(patch_level_logit, dim=1)

        elif self.mgcc_aggregation_type == 'Max_Max':
            word_level_logit, _ = torch.max(fine_grained_sim_scores, dim=1)
            patch_level_logit, _ = torch.max(fine_grained_sim_scores, dim=-1)
            word_level_logit2, _ = torch.max(word_level_logit, dim=-1)
            patch_level_logit2, _ = torch.max(patch_level_logit, dim=1)

        elif self.mgcc_aggregation_type == 'Mean_Mean':
            word_level_logit = torch.mean(fine_grained_sim_scores, dim=1)
            patch_level_logit = torch.mean(fine_grained_sim_scores, dim=-1)
            word_level_logit2 = torch.mean(word_level_logit, dim=-1)
            patch_level_logit2 = torch.mean(patch_level_logit, dim=1)

        elif self.mgcc_aggregation_type == 'Mean_Max':
            word_level_logit = torch.mean(fine_grained_sim_scores, dim=1)
            patch_level_logit = torch.mean(fine_grained_sim_scores, dim=-1)
            word_level_logit2, _ = torch.max(word_level_logit, dim=-1)
            patch_level_logit2, _ = torch.max(patch_level_logit, dim=1)
        else:
            raise ValueError(f"Unsupported MGCC aggregation type: {self.mgcc_aggregation_type}")

        return (word_level_logit2 + patch_level_logit2) / 2

    def mgcc_get_similarity(self, image_feats, atten_i, text_feats, atten_t, caption_ids):
        img_global, patch_part = self.mgcc_img_embedding(image_feats, atten_i)
        txt_global, word_part = self.mgcc_txt_embedding(text_feats, caption_ids, atten_t)

        logit_scale = self.logit_scale
        softmax_t = self.mgcc_softmax_t

        total_logits = []

        img_text_logits = logit_scale * torch.matmul(torch.matmul(img_global, self.mgcc_global_mat_weight), torch.t(txt_global))
        total_logits.append(img_text_logits)

        img_word_sim = torch.einsum('id,jkd->ijk', img_global, word_part)
        img_word_att = torch.softmax(img_word_sim / softmax_t, dim=-1)
        img_word_logits = logit_scale * torch.sum(img_word_sim * torch.matmul(img_word_att, self.mgcc_word_logit_weight), dim=-1)
        total_logits.append(img_word_logits)

        if patch_part is not None:
            patch_text_sim = torch.matmul(patch_part, txt_global.t())
            patch_text_att = torch.softmax(patch_text_sim / softmax_t, dim=1).permute(0, 2, 1)
            patch_text_logits = logit_scale * torch.sum(patch_text_sim * torch.matmul(patch_text_att, self.mgcc_patch_logit_weight).permute(0, 2, 1), dim=1)
            total_logits.append(patch_text_logits)

            patch_word_logits = logit_scale * self.mgcc_aggregation_fine_grained_similarity(patch_part, word_part)
            total_logits.append(patch_word_logits)

        sim_i_2_t = sum(total_logits) / len(total_logits)
        return sim_i_2_t

    def mgcc_loss_from_sim(self, sim_matrix, label_hat):
        logpt_i2t = F.log_softmax(sim_matrix, dim=-1)
        logpt_t2i = F.log_softmax(sim_matrix.t(), dim=-1)
        per_loss = (-logpt_i2t.diag() - logpt_t2i.diag()) / 2
        if self.loss_type in ['TAL', 'TRL']:
            loss = (label_hat * per_loss).sum()
        else:
            denom = label_hat.sum().clamp(min=1.0)
            loss = (label_hat * per_loss).sum() / denom
        return loss

    def compute_per_loss(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        lossB, simsB = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        
        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB

    def forward(self, batch):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            
        label_hat = batch['label_hat'].to(i_feats.device) 
     
        loss1, loss2 = objectives.compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, batch['pids'], \
                                              label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
                                                loss_type=self.loss_type,logit_scale=self.logit_scale)
        ret.update({'bge_loss':loss1})
        ret.update({'tse_loss':loss2})

        if self.use_mgcc and self.mgcc_weight > 0:
            sim_i_2_t = self.mgcc_get_similarity(image_feats, atten_i, text_feats, atten_t, caption_ids)
            mgcc_loss = self.mgcc_loss_from_sim(sim_i_2_t, label_hat)
            mgcc_loss = self.mgcc_weight * mgcc_loss
            ret.update({'mgcc_loss': mgcc_loss})

        # masked language modeling loss on text tokens (optional)
        if self.mlm_head is not None and self.mlm_weight > 0:
            text = caption_ids
            bs, L = text.shape
            device = text.device

            with torch.no_grad():
                # candidate positions: non-padding, excluding first token and EOT token
                valid_mask = (text != 0)
                first_pos = torch.zeros_like(text, dtype=torch.bool, device=device)
                first_pos[:, 0] = True
                eot_indices = text.argmax(dim=-1)
                eot_pos = torch.zeros_like(text, dtype=torch.bool, device=device)
                eot_pos[torch.arange(bs), eot_indices] = True
                candidate = valid_mask & (~first_pos) & (~eot_pos)

                rand = torch.rand_like(text.float())
                mlm_mask = (rand < self.args.masked_token_rate) & candidate

            if mlm_mask.any():
                mlm_feats, _ = self.base_model.encode_text_mlm(text.long(), text_mask=mlm_mask)
                mlm_logits = self.mlm_head(mlm_feats).float()

                mlm_labels = text.clone()
                mlm_labels[~mlm_mask] = -100

                mlm_loss = F.cross_entropy(
                    mlm_logits.view(-1, mlm_logits.size(-1)),
                    mlm_labels.view(-1),
                    ignore_index=-100,
                )
                mlm_loss = self.mlm_weight * mlm_loss
                ret.update({'mlm_loss': mlm_loss})
  
        if self.use_bamg and self.bamg_encoder is not None:
            img_tokens = image_feats
            txt_tokens = text_feats
            img_tokens, txt_tokens, fsn_tokens = self.bamg_encoder(img_tokens, txt_tokens)
            i_bamg = img_tokens[:, 0, :].float()
            txt_indices = caption_ids.argmax(dim=-1)
            t_bamg = txt_tokens[torch.arange(txt_tokens.shape[0]), txt_indices].float()
            bamg_per_loss, _ = objectives.compute_per_loss(
                i_bamg,
                t_bamg,
                batch['pids'],
                tau=self.args.tau,
                margin=self.args.margin,
                loss_type=self.loss_type,
                logit_scale=self.logit_scale,
            )
            if self.loss_type in ['TAL', 'TRL']:
                bamg_loss = (label_hat * bamg_per_loss).sum()
            else:
                denom = label_hat.sum().clamp(min=1.0)
                bamg_loss = (label_hat * bamg_per_loss).sum() / denom
            bamg_loss = self.bamg_weight * bamg_loss
            ret.update({'bamg_loss': bamg_loss})

            if self.mgm_head is not None and self.mgm_weight > 0:
                patch_tokens = image_feats[:, 1:, :]
                mgm_loss = self.mgm_head(patch_tokens)
                mgm_loss = self.mgm_weight * mgm_loss
                ret.update({'mgm_loss': mgm_loss})

        return ret


def build_model(args, num_classes=11003):
    model = RDE(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
