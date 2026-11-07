# -*- coding:utf-8 -*-
"""
ArcMDTI Model Architecture (code_new/model.py)
==============================================
Structure-Informed Angular-Margin Learning with Bidirectional Cross-Attention
for Drug-Target Interaction Prediction.

Key Modules:
1. Hyperspherical Margin Classifiers:
   - ArcMarginProduct (ArcFace: additive angular margin cos(theta + m))
   - CosMarginProduct (CosFace: additive cosine margin cos(theta) - m)
   - SphereMarginProduct (SphereFace: multiplicative angular margin cos(m * theta))
   - DirectMarginProduct (Linear logit margin)
2. MultiScaleEncoder & SEBlock:
   - Parallel standard (k=3, d=1) and dilated (k=5, d=2) 1D convolutions + Squeeze-and-Excitation
3. HybridEncoder:
   - Multi-scale CNN + BiLSTM + Transformer Encoder with residual LayerNorm
   - Supports encoder-level ablations ('cnn_only', 'no_bilstm', 'no_transformer')
4. AMCADTI_V2 (ArcMDTI):
   - Secondary Structure (SS) conformation prior integration
   - Bidirectional Cross-Attention + Dynamic Gated Fusion
   - Supports full ablation suite and attention/angle extraction for interpretability
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from x_transformers import Encoder as XTransformerEncoder
    HAS_X_TRANSFORMERS = True
except ImportError:
    HAS_X_TRANSFORMERS = False


class FallbackTransformerEncoder(nn.Module):
    """Fallback Transformer encoder using native PyTorch nn.TransformerEncoder if x_transformers is absent."""

    def __init__(self, dim, depth, heads):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)


def build_transformer_encoder(dim, depth, heads):
    if HAS_X_TRANSFORMERS:
        return XTransformerEncoder(dim=dim, depth=depth, heads=heads)
    return FallbackTransformerEncoder(dim=dim, depth=depth, heads=heads)


# ==========================================
# 1. Hyperspherical Angular-Margin Modules
# ==========================================
class ArcMarginProduct(nn.Module):
    """
    Additive Angular Margin (ArcFace) Classifier on the Unit Hypersphere.
    Projects L2-normalized features and class prototypes onto S^(d-1) and enforces
    an additive angular margin penalty m on the ground-truth class angle theta_y.
    """

    def __init__(self, in_features, out_features=2, s=15.0, m=0.15, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self._update_margin(self.m)

    def _update_margin(self, m):
        self.m = float(m)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def get_cosine(self, input):
        x_norm = F.normalize(input, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        return cosine

    def forward(self, input, label=None):
        cosine = self.get_cosine(input)

        # Inference / validation / feature extraction mode (no margin penalty)
        if label is None or self.m == 0.0:
            return cosine * self.s

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-12, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class CosMarginProduct(nn.Module):
    """
    Large Margin Cosine (CosFace) Classifier: cos(theta_y) - m.
    """

    def __init__(self, in_features, out_features=2, s=15.0, m=0.15):
        super(CosMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def get_cosine(self, input):
        return F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    def forward(self, input, label=None):
        cosine = self.get_cosine(input)
        if label is None:
            return cosine * self.s

        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class SphereMarginProduct(nn.Module):
    """
    Multiplicative Angular Margin (SphereFace) Classifier: cos(m * theta_y).
    """

    def __init__(self, in_features, out_features=2, s=15.0, m=1.35):
        super(SphereMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def get_cosine(self, input):
        return F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    def forward(self, input, label=None):
        cosine = self.get_cosine(input)
        if label is None:
            return cosine * self.s

        theta = torch.acos(cosine)
        m_theta = (self.m * theta).clamp(0.0, math.pi)
        phi = torch.cos(m_theta)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class DirectMarginProduct(nn.Module):
    """
    Direct Linear Logit Margin Classifier (without hyperspherical normalization).
    """

    def __init__(self, in_features, out_features=2, m=0.15):
        super(DirectMarginProduct, self).__init__()
        self.m = float(m)
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, input, label=None):
        output = self.fc(input)
        if label is None:
            return output

        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        return output - one_hot * self.m


# ==========================================
# 2. Hybrid Local-to-Global Sequence Encoder
# ==========================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) channel recalibration block."""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MultiScaleEncoder(nn.Module):
    """Multi-scale 1D convolutional pyramid with standard (k=3, d=1) and dilated (k=5, d=2) kernels."""

    def __init__(self, dim):
        super().__init__()
        self.branch1 = nn.Conv1d(dim, dim // 2, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(dim, dim // 2, kernel_size=5, padding=4, dilation=2)
        self.se = SEBlock(dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return self.act(self.bn(self.se(out)))


class HybridEncoder(nn.Module):
    """
    Sequential Local-to-Global Hybrid Encoder:
    Stage A: Multi-Scale 1D-CNN + SE Block (local biochemical motifs)
    Stage B: Bidirectional LSTM (directional sequence dependencies)
    Stage C: Transformer Self-Attention Encoder (long-range global context)
    """

    def __init__(self, dim, depth, heads, encoder_mode="full"):
        super().__init__()
        self.encoder_mode = encoder_mode
        self.ms_cnn = MultiScaleEncoder(dim)
        self.lstm = nn.LSTM(dim, dim // 2, bidirectional=True, batch_first=True)
        self.transformer = build_transformer_encoder(dim=dim, depth=depth, heads=heads)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        c = self.ms_cnn(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x + c)

        if self.encoder_mode == "cnn_only":
            return x

        if self.encoder_mode != "no_bilstm":
            self.lstm.flatten_parameters()
            l_out, _ = self.lstm(x)
            x = self.norm(x + l_out)

        if self.encoder_mode != "no_transformer":
            x = self.transformer(x)

        return x


# ==========================================
# 3. Main ArcMDTI Architecture (AMCADTI_V2)
# ==========================================
class AMCADTI_V2(nn.Module):
    """
    ArcMDTI (AMCADTI_V2): Structure-Informed Angular-Margin Learning with
    Bidirectional Cross-Attention and Dynamic Gated Fusion.

    Supported ablation modes:
      - 'full'           : Complete ArcMDTI model
      - 'no_ss'          : Remove NetSurfP-3.0 secondary structure branch
      - 'no_crossattn'   : Remove bidirectional cross-attention (direct pooling)
      - 'no_gate'        : Remove dynamic sigmoid gated fusion
      - 'no_arcface'     : Replace ArcFace head with standard nn.Linear(256, 2)
      - 'no_focalloss'   : Full ArcMDTI trained with standard CrossEntropyLoss
      - 'ce'             : Linear head trained with standard CrossEntropyLoss
      - 'cosface'        : CosMarginProduct head
      - 'sphereface'     : SphereMarginProduct head
      - 'direct_margin'  : DirectMarginProduct head
      - 'cnn_only'       : Encoder uses only MultiScaleCNN + SE
      - 'no_bilstm'      : Encoder removes BiLSTM
      - 'no_transformer' : Encoder removes Transformer
    """

    def __init__(self, hp, ablation="full", arcface_s=None, arcface_m=None):
        super(AMCADTI_V2, self).__init__()
        self.ablation = ablation
        self.s = float(arcface_s if arcface_s is not None else getattr(hp, "arcface_s", 15.0))
        self.m = float(arcface_m if arcface_m is not None else getattr(hp, "arcface_m", 0.15))

        self.drug_embed = nn.Embedding(hp.drug_vocab_size, hp.char_dim)
        self.prot_embed = nn.Embedding(hp.prot_vocab_size, hp.char_dim)
        self.ss_embed = nn.Embedding(getattr(hp, "ss_vocab_size", 5), hp.char_dim, padding_idx=0)

        encoder_mode = ablation if ablation in ("cnn_only", "no_bilstm", "no_transformer") else "full"
        self.d_encoder = HybridEncoder(hp.char_dim, hp.depth, hp.heads, encoder_mode=encoder_mode)
        self.p_encoder = HybridEncoder(hp.char_dim, hp.depth, hp.heads, encoder_mode=encoder_mode)

        self.cross_attn = nn.MultiheadAttention(
            hp.char_dim,
            hp.heads,
            batch_first=True,
            dropout=getattr(hp, "attn_dropout", 0.3),
        )
        self.attn_norm = nn.LayerNorm(hp.char_dim)

        self.fusion_gate = nn.Sequential(
            nn.Linear(hp.char_dim * 4, hp.char_dim * 4),
            nn.Sigmoid(),
        )

        self.feat_extractor = nn.Sequential(
            nn.Linear(hp.char_dim * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Dropout(getattr(hp, "mlp_dropout", 0.4)),
            nn.Linear(1024, 256),
        )

        if self.ablation in ("no_arcface", "ce"):
            self.arcface = nn.Linear(256, 2)
        elif self.ablation == "cosface":
            self.arcface = CosMarginProduct(in_features=256, out_features=2, s=self.s, m=self.m)
        elif self.ablation == "sphereface":
            sphere_m = 1.35 if abs(self.m - 0.15) < 1e-6 else (1.0 + self.m * 2.0)
            self.arcface = SphereMarginProduct(in_features=256, out_features=2, s=self.s, m=sphere_m)
        elif self.ablation == "direct_margin":
            self.arcface = DirectMarginProduct(in_features=256, out_features=2, m=self.m)
        else:
            self.arcface = ArcMarginProduct(in_features=256, out_features=2, s=self.s, m=self.m)

    def get_hyperspherical_angles(self, feat_256):
        """
        Compute cosine similarities and angles (in degrees) between 256-D latent
        features and the class weight prototypes [W_0, W_1].
        Works for both hyperspherical heads and linear heads.
        """
        x_norm = F.normalize(feat_256, p=2, dim=1)
        if hasattr(self.arcface, "weight"):
            w_norm = F.normalize(self.arcface.weight, p=2, dim=1)
        elif hasattr(self.arcface, "fc"):
            w_norm = F.normalize(self.arcface.fc.weight, p=2, dim=1)
        else:
            return None, None
        cosines = F.linear(x_norm, w_norm).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        angles_deg = torch.acos(cosines) * (180.0 / math.pi)
        return cosines, angles_deg

    def forward(self, drug_seq, prot_seq, prot_ss, labels=None, return_cl=False, return_attn=False):
        # Stage 1: Input & Secondary Structure Embedding
        d_e = self.drug_embed(drug_seq)

        if self.ablation == "no_ss":
            p_e = self.prot_embed(prot_seq)
        else:
            p_e = self.prot_embed(prot_seq) + self.ss_embed(prot_ss)

        # Stage 2: Multi-Scale Hybrid Sequence Encoding
        d_enc = self.d_encoder(d_e)
        p_enc = self.p_encoder(p_e)

        # Stage 3: Bidirectional Cross-Attention & Gated Fusion
        d2p_weights, p2d_weights = None, None
        if self.ablation == "no_crossattn":
            d_vec = torch.cat([d_enc.mean(1), d_enc.max(1)[0]], dim=-1)
            p_vec = torch.cat([p_enc.mean(1), p_enc.max(1)[0]], dim=-1)
        else:
            d_ctx, d2p_weights = self.cross_attn(
                d_enc, p_enc, p_enc, need_weights=return_attn, average_attn_weights=True
            )
            p_ctx, p2d_weights = self.cross_attn(
                p_enc, d_enc, d_enc, need_weights=return_attn, average_attn_weights=True
            )
            d_norm = self.attn_norm(d_enc + d_ctx)
            p_norm = self.attn_norm(p_enc + p_ctx)
            d_vec = torch.cat([d_norm.mean(1), d_norm.max(1)[0]], dim=-1)
            p_vec = torch.cat([p_norm.mean(1), p_norm.max(1)[0]], dim=-1)

        raw_feat = torch.cat([d_vec, p_vec], dim=-1)
        gate_vals = self.fusion_gate(raw_feat)
        if self.ablation == "no_gate":
            gated_feat = raw_feat
        else:
            gated_feat = raw_feat * gate_vals

        feat_256 = self.feat_extractor(gated_feat)

        # Stage 4: Hyperspherical Angular-Margin Decision Layer
        if self.ablation in ("no_arcface", "ce"):
            logits = self.arcface(feat_256)
        else:
            logits = self.arcface(feat_256, labels)

        if return_attn:
            cosines, angles_deg = self.get_hyperspherical_angles(feat_256)
            attn_info = {
                "d2p_attn": d2p_weights,
                "p2d_attn": p2d_weights,
                "gate_values": gate_vals,
                "cosines": cosines,
                "angles_deg": angles_deg,
            }
            return logits, feat_256, attn_info

        if return_cl:
            return logits, feat_256

        return logits

