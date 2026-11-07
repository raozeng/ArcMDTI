# -*- coding:utf-8 -*-
import os
import numpy as np
import torch


class EarlyStopping:
    """
    Early stops the training if validation metric (1 - AUC) doesn't improve after a given patience.
    """

    def __init__(self, savepath="checkpoint.pt", patience=25, verbose=False, delta=0.0):
        self.savepath = savepath
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, epoch=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation metric improved ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")

        if os.path.isdir(self.savepath):
            save_file = os.path.join(self.savepath, "valid_best_checkpoint.pth")
        else:
            save_file = self.savepath

        torch.save(model.state_dict(), save_file)
        self.val_loss_min = val_loss

