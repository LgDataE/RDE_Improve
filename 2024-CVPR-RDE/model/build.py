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

        img_tokens, _, _ = self.bamg_encoder(x, x)
        return img_tokens[:, 0, :].float()

    def encode_text_bamg(self, text):
        """Text encoder with BAMG bottleneck for evaluation.

        If BAMG is disabled, this falls back to the standard EOT embedding.
        """
        x, _ = self.base_model.encode_text(text.long())
        if not self.use_bamg or self.bamg_encoder is None:
            return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

        img_tokens, txt_tokens, _ = self.bamg_encoder(x, x)
        idx = text.argmax(dim=-1)
        return txt_tokens[torch.arange(txt_tokens.shape[0]), idx].float()

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
