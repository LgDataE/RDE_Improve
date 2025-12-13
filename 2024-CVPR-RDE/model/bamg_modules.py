import torch
import torch.nn as nn


class FusionBottleneckLayer(nn.Module):
    def __init__(self, dim, nhead=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.img_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.txt_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )

    def forward(self, img_tokens, text_tokens, fsn_tokens):
        b, ni, _ = img_tokens.size()
        _, nt, _ = text_tokens.size()
        img_input = torch.cat([img_tokens, fsn_tokens], dim=1)
        txt_input = torch.cat([text_tokens, fsn_tokens], dim=1)
        img_out = self.img_layer(img_input)
        txt_out = self.txt_layer(txt_input)
        img_tokens_new = img_out[:, :ni, :]
        fsn1 = img_out[:, ni:, :]
        text_tokens_new = txt_out[:, :nt, :]
        fsn2 = txt_out[:, nt:, :]
        fsn_tokens_new = (fsn1 + fsn2) * 0.5
        return img_tokens_new, text_tokens_new, fsn_tokens_new


class BottleneckFusionEncoder(nn.Module):
    def __init__(self, dim, num_bottlenecks=4, depth=2, nhead=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.num_bottlenecks = num_bottlenecks
        self.fsn_tokens = nn.Parameter(torch.zeros(1, num_bottlenecks, dim))
        nn.init.normal_(self.fsn_tokens, std=0.02)
        self.layers = nn.ModuleList(
            [
                FusionBottleneckLayer(
                    dim=dim,
                    nhead=nhead,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, img_tokens, text_tokens):
        b = img_tokens.size(0)
        fsn_tokens = self.fsn_tokens.expand(b, -1, -1).to(img_tokens.dtype).to(img_tokens.device)
        for layer in self.layers:
            img_tokens, text_tokens, fsn_tokens = layer(img_tokens, text_tokens, fsn_tokens)
        return img_tokens, text_tokens, fsn_tokens

    def forward_image(self, img_tokens):
        b, ni, _ = img_tokens.size()
        fsn_tokens = self.fsn_tokens.expand(b, -1, -1).to(img_tokens.dtype).to(img_tokens.device)
        for layer in self.layers:
            img_input = torch.cat([img_tokens, fsn_tokens], dim=1)
            img_out = layer.img_layer(img_input)
            img_tokens = img_out[:, :ni, :]
            fsn_tokens = img_out[:, ni:, :]
        return img_tokens, fsn_tokens

    def forward_text(self, text_tokens):
        b, nt, _ = text_tokens.size()
        fsn_tokens = self.fsn_tokens.expand(b, -1, -1).to(text_tokens.dtype).to(text_tokens.device)
        for layer in self.layers:
            txt_input = torch.cat([text_tokens, fsn_tokens], dim=1)
            txt_out = layer.txt_layer(txt_input)
            text_tokens = txt_out[:, :nt, :]
            fsn_tokens = txt_out[:, nt:, :]
        return text_tokens, fsn_tokens


class MaskedGraphModelingHead(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers=2, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio
        if hidden_dim <= 0:
            hidden_dim = dim
        layers = []
        in_dim = dim
        for _ in range(max(num_layers - 1, 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        b, n, d = x.size()
        device = x.device
        num_mask = max(1, int(n * self.mask_ratio))
        mask = torch.zeros(b, n, dtype=torch.bool, device=device)
        for i in range(b):
            idx = torch.randperm(n, device=device)[:num_mask]
            mask[i, idx] = True
        x_masked = x.clone()
        x_masked[mask] = 0.0
        x_rec = self.mlp(x_masked)
        diff = (x_rec[mask] - x[mask]) ** 2
        loss = diff.mean()
        return loss
