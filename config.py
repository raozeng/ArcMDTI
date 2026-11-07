# -*- coding:utf-8 -*-
"""
Unified Hyperparameter Configuration for ArcMDTI (code_new)
===========================================================
Centralizes all architectural, optimization, and angular-margin hyperparameters
to ensure full consistency across 5-fold CV, cold-start evaluation, ablation
studies, and hyperparameter sensitivity experiments.
"""


class hyperparameter:
    def __init__(self):
        # 1. Sequence & Vocabulary Configuration
        # Remapped CHARISOSMISET strictly covers 1..64 (vocab=65) and CHARPROTSET covers 1..25 (vocab=26),
        # ensuring 100% checkpoint compatibility and zero out-of-bounds index errors.
        self.char_dim = 128
        self.prot_max_len = 1000
        self.drug_max_len = 100
        self.drug_vocab_size = 65
        self.prot_vocab_size = 26
        self.ss_vocab_size = 5  # 0: pad, 1: Coil (C/T/S/I), 2: Helix (H/G), 3: Strand (E/B)

        # 2. Hybrid Encoder & Cross-Attention Architecture
        self.depth = 1          # Transformer encoder depth
        self.heads = 4          # Number of attention heads
        self.cnn_dropout = 0.2
        self.attn_dropout = 0.3
        self.mlp_dropout = 0.4

        # 3. Training & Optimization Schedule (can also be overridden via --epochs / --patience or env vars)
        import os
        self.Epoch = int(os.environ.get("ARCMDTI_EPOCHS", 100))
        self.Patience = int(os.environ.get("ARCMDTI_PATIENCE", 25))
        self.Batch_size = int(os.environ.get("ARCMDTI_BATCH_SIZE", 128))
        self.Learning_rate = 3e-4
        self.weight_decay = 1e-2
        self.scheduler_T0 = 10
        self.scheduler_Tmult = 2
        self.grad_clip = 1.0

        # 4. ArcFace Hyperspherical Margin & Focal Loss Parameters
        self.arcface_s = 15.0   # Hypersphere radius / cosine scale factor s
        self.arcface_m = 0.15   # Additive angular margin penalty m (radians)
        self.focal_alpha = 0.5  # Focal loss balancing factor alpha
        self.focal_gamma = 1.0  # Focal loss hard-example focusing parameter gamma

        # 5. Runtime Metadata
        self.dataset_name = "Enzyme"

    def select_dataset_config(self, dataset_path_or_name):
        import os
        base = os.path.basename(str(dataset_path_or_name)).replace(".txt", "")
        self.dataset_name = base
        print(
            f"[ArcMDTI Config] Dataset: {self.dataset_name} | "
            f"dim={self.char_dim}, depth={self.depth}, heads={self.heads} | "
            f"lr={self.Learning_rate}, wd={self.weight_decay}, bs={self.Batch_size} | "
            f"ArcFace(s={self.arcface_s}, m={self.arcface_m}) + "
            f"Focal(alpha={self.focal_alpha}, gamma={self.focal_gamma})"
        )

