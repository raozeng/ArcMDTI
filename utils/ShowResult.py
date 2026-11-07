# -*- coding:utf-8 -*-
import os
import numpy as np


def show_result(DATASET, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List, Ensemble=False):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.std(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.std(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.std(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.std(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.std(AUPR_List)

    os.makedirs(f"./{DATASET}", exist_ok=True)
    if not Ensemble:
        print("The model's results:")
        filepath = f"./{DATASET}/results.txt"
    else:
        print("The ensemble model's results:")
        filepath = f"./{DATASET}/ensemble_results.txt"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Accuracy(std):{Accuracy_mean:.4f}({Accuracy_var:.4f})\n")
        f.write(f"Precision(std):{Precision_mean:.4f}({Precision_var:.4f})\n")
        f.write(f"Recall(std):{Recall_mean:.4f}({Recall_var:.4f})\n")
        f.write(f"AUC(std):{AUC_mean:.4f}({AUC_var:.4f})\n")
        f.write(f"PRC(std):{PRC_mean:.4f}({PRC_var:.4f})\n")
    print(f"Accuracy(std):{Accuracy_mean:.4f}({Accuracy_var:.4f})")
    print(f"Precision(std):{Precision_mean:.4f}({Precision_var:.4f})")
    print(f"Recall(std):{Recall_mean:.4f}({Recall_var:.4f})")
    print(f"AUC(std):{AUC_mean:.4f}({AUC_var:.4f})")
    print(f"PRC(std):{PRC_mean:.4f}({PRC_var:.4f})")

