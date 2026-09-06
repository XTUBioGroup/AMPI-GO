import warnings
import click
import numpy as np
import pandas as pd
import scipy.sparse as ssp
import torch
import dgl
from pathlib import Path
import os
from tqdm.auto import tqdm, trange
import networkx as nx
import torch.nn as nn
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn.functional as F
import pickle as pkl
import time

from objective import AverageMeter
from Evaluation import new_compute_performance_deepgoplus

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, reduction='mean'):
        """
        Args:
            gamma: Focusing parameter, usually set to 2.0.
            alpha: Positive-sample weight (0–1).
                   Because positive samples are extremely rare (<1%), use a larger alpha (such as 0.75 or 0.9)
                   to tell the model that misclassifying a positive sample is very costly.
            reduction: 'mean' or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        # 1. Compute standard BCE loss without reducing, preserving the original shape
        # pred: logits, target: 0/1 labels
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 2. Compute pt (the probability assigned to the correct class)
        pt = torch.exp(-ce_loss)
        
        # 3. Compute the focal term
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 4. Apply alpha weighting (critical step)
        # The original logic here was incorrect. The correct approach is:
        # positive samples * alpha, negative samples * (1 - alpha)
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_loss

        # 5. Apply reduction at the end
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def merge_result(cob_df_list):
    save_dict = {}
    save_dict['protein_id'] = []
    save_dict['gos'] = []
    save_dict['predictions'] = []
    
    for idx, row in cob_df_list[0].iterrows():
        save_dict['protein_id'].append(row['protein_id'])
        save_dict['gos'].append(row['gos'])
        pred_gos = {}
        # merge
        for go, score in row['predictions'].items():
            pred_gos[go] = score
        for single_df in cob_df_list[1:]:
            pred_scores = single_df[single_df['protein_id']==row['protein_id']].reset_index().loc[0, 'predictions']
            for go, score in pred_scores.items():
                pred_gos[go] += score
        # average
        avg_pred_gos = {}
        for go, score in pred_gos.items():
            avg_pred_gos[go] = score/len(cob_df_list)
        
        save_dict['predictions'].append(avg_pred_gos)
        
    df = pd.DataFrame(save_dict)
    
    return df



def test_performance_gnn_inter(
    model,
    dataloader,
    test_pid_list,
    test_interpro,
    test_y_matrix,
    idx_goid,
    goid_idx,
    ont,
    device,
    loss_fn=None,
    valid_ppi_graph=None,
    save=False,
    save_file=None,
    evaluate=True,
    with_relations=True,
    go_file='go.obo'  # ✅ New parameter
):
    model.eval()
    
    pred_labels = []
    save_dict = {'protein_id': [], 'gos': [], 'predictions': []}

    # 1. Loss function
    if loss_fn is None: 
        loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
    
    test_loss_vals = AverageMeter()
    
    # 2. Validation loop
    with torch.no_grad():
        for batch_idx, (x_test, sample_idx, y_test) in enumerate(dataloader):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            feats = x_test.ndata['x']

            # InterPro features
            idx_np = sample_idx.cpu().numpy().astype(int)
            row = test_interpro[idx_np]
            inter_features = (
                torch. from_numpy(row.indices).long().to(device),
                torch.from_numpy(row. indptr).long().to(device),
                torch.from_numpy(row.data).float().to(device)
            )

            # PPI graph
            if valid_ppi_graph is not None:
                ppi_graph_list = [valid_ppi_graph[i] for i in idx_np]
                batched_ppi_graph = dgl.batch(ppi_graph_list).to(device)
                logits = model(inter_features, x_test, feats, batched_ppi_graph)
            else:
                logits = model(inter_features, x_test, feats)

            # Loss
            loss = loss_fn(logits, y_test)
            test_loss_vals.update(loss.item(), len(y_test))

            # Probabilities
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            pred_labels.append(probs)
            
            # Debug (print only the first batch)
            if batch_idx == 0:
                print(f"\n🔍 [Valid Debug] Predicted probability distribution: Max={probs.max():.5f}, Mean={probs.mean():.5f}")
                if probs.max() < 0.01:
                    print("⚠️ Warning: The maximum predicted probability is still below 0.01; Fmax may be 0!")

    pred_labels = np. vstack(pred_labels)
    
    # 3. Build the DataFrame
    print("Building the evaluation DataFrame...")
    
    # Debug: Check the mapping
    print("\n🔍 [ID Mapping Debug] ----------------")
    if len(idx_goid) > 0:
        first_idx = list(idx_goid.keys())[0]
        first_go = idx_goid[first_idx]
        print(f"Index type:  {type(first_idx)} (Expect: int)")
        print(f"GO ID type: {type(first_go)} (Expect: str)")
        print(f"Example mapping: {first_idx} -> {first_go}")
        if not str(first_go).startswith("GO:"):
            print("⚠️ Warning: The GO ID is missing the 'GO:' prefix!")
    else:
        print("⚠️ Warning: The idx_goid dictionary is empty!")
    print("---------------------------------------\n")

    num_labels = pred_labels.shape[1]
    
    for rowid in tqdm(range(pred_labels. shape[0]), desc="Building DF", leave=False):
        save_dict['protein_id']. append(test_pid_list[rowid])
        
        # Ground-truth label set
        true_indices = np.where(test_y_matrix[rowid] == 1)[0]
        true_gos = set()
        for idx in true_indices:
            if idx in idx_goid:  # ✅ Add a guard
                true_gos. add(idx_goid[idx])
        save_dict['gos'].append(true_gos)
        
        # Prediction dictionary: {GO_ID: score}
        current_pred = pred_labels[rowid]
        pred_gos = {}
        for i in range(num_labels):
            if i in idx_goid:  # ✅ Add a guard
                pred_gos[idx_goid[i]] = float(current_pred[i])
        save_dict['predictions'].append(pred_gos)

    df = pd.DataFrame(save_dict)
    
    # ✅ Save results
    if save and save_file:
        with open(save_file, 'wb') as fw:
            pkl.dump(df, fw)
        print(f"✅ Results saved to {save_file}")
    
    # 4. Evaluation
    if evaluate:
        if os.path.exists(go_file):
            new_fmax, new_aupr, new_t = new_compute_performance_deepgoplus(
                df, go_file, ont, idx_goid, goid_idx, with_relations
            )
            return new_fmax, new_aupr, new_t, df, test_loss_vals. avg
        else:
            print(f"❌ Error: {go_file} was not found; skipping metric computation")
            return 0.0, 0.0, 0.0, df, test_loss_vals.avg
    else:
        return df
