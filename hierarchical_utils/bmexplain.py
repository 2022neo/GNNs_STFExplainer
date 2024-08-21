"""
    Implementation of the bmexplainer, which transfert input layer explanation to high level
"""

import math
import time
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_auc_score, precision_recall_curve
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F

EPS = 1e-6



class BMExplainer(nn.Module):
    def __init__(
        self,
        data,
        raw_tree,
        model,
        device,
        masked_adj,
    ):
        super(BMExplainer, self).__init__()
        self.device = device
        self.raw_tree = raw_tree
        self.tree = {k:torch.nn.functional.one_hot(torch.LongTensor(v)).float().to(self.device) for k,v in raw_tree.items()}
        self.model = model.to(self.device)
        self.model.eval()

        self.layer_edge_labels = {}
        if hasattr(data, 'edge_label'):
            for (r,c),(l_sub,l_sup) in zip(data.edge_index.T.tolist(),data.edge_label):
                if r<=c:
                    continue
                for layer,assign in raw_tree.items():
                    self.layer_edge_labels.setdefault(layer,[])
                    if l_sup==1 or l_sub==1:
                        self.layer_edge_labels[layer].append((assign[r],assign[c]))


        with torch.no_grad():
            data = data.to(self.device)
            logits = self.model(data=data)
            ori_pred = logits.argmax(-1).data.cpu().item() if logits.shape[-1]>1 else logits.data.cpu()

        #init edge_weight
        edge_index = data.edge_index
        nodesize = data.x.shape[0]
        # if isinstance(data.edge_weight,torch.Tensor):
        #     edge_weight = data.edge_weight
        #     adj = torch.sparse_coo_tensor(
        #     edge_index, edge_weight, (nodesize, nodesize)
        #     ).to_dense().to(self.device)
        # else:
        edge_weight = torch.ones(edge_index.shape[-1]).to(self.device)
        adj = torch.sparse_coo_tensor(
        edge_index, edge_weight, (nodesize, nodesize)
        ).to_dense().to(self.device)

        self.adj = adj
        self.data = data
        self.ori_pred = ori_pred
        
        self.layer_adjs = {}
        self.layer_masked_adjs = {}
        with torch.no_grad():
            for layer, assign_matrix in self.tree.items():
                num_nodes = assign_matrix.size()[1]
                self.layer_adjs[layer] = ((assign_matrix.T @ adj @ assign_matrix)>0).float().to(self.device)
                self.layer_edge_labels.setdefault(layer,[])
                self.layer_edge_labels[layer] = list(set(self.layer_edge_labels[layer]))
                assign_matrix_T = assign_matrix.T
                aggregated_matrix = assign_matrix_T.unsqueeze(0).unsqueeze(-1) * masked_adj.to(self.device) * assign_matrix_T.unsqueeze(1).unsqueeze(-2)
                aggregated_matrix, _ = torch.max(aggregated_matrix, dim=-1)
                aggregated_matrix, _ = torch.max(aggregated_matrix, dim=-1)
                self.layer_masked_adjs[layer] = aggregated_matrix.float().to(self.device)

    
    def evaluation(self,eval_func,recall_N,ce_auc,acc_auc,spars,spars_auc):
        with torch.no_grad():
            data = self.data
            adj = self.adj
            ori_pred = self.ori_pred
            for layer in self.raw_tree:
                assign_matrix = self.tree[layer]
                layer_adj = self.layer_adjs[layer]
                layer_edge_label = self.layer_edge_labels[layer]

                layer_mask_sigmoid =  self.layer_masked_adjs[layer]

                layer_sym_mask = (layer_mask_sigmoid + layer_mask_sigmoid.t()) / 2
                layer_masked_adj = layer_adj * layer_sym_mask


                eval_func(
                    data=data,adj=adj,model=self.model,ori_pred=ori_pred,
                    layer_adj=layer_adj,layer_masked_adj=layer_masked_adj,layer_edge_label=layer_edge_label,
                    assign_matrix=assign_matrix,
                    recall_N=recall_N,ce_auc=ce_auc,acc_auc=acc_auc,spars=spars,spars_auc=spars_auc)

        return len(layer_edge_label)

    def evaluation_reg(self,eval_func_reg,reals,preds,spars_auc):
        with torch.no_grad():
            data = self.data
            adj = self.adj
            ori_pred = self.ori_pred
            for layer in self.raw_tree:
                assign_matrix = self.tree[layer]
                layer_adj = self.layer_adjs[layer]
                layer_edge_label = self.layer_edge_labels[layer]

                layer_mask_sigmoid =  self.layer_masked_adjs[layer]

                layer_sym_mask = (layer_mask_sigmoid + layer_mask_sigmoid.t()) / 2
                layer_masked_adj = layer_adj * layer_sym_mask


                eval_func_reg(
                    data=data,adj=adj,model=self.model,ori_pred=ori_pred,
                    layer_adj=layer_adj,layer_masked_adj=layer_masked_adj,layer_edge_label=layer_edge_label,
                    assign_matrix=assign_matrix,spars_auc=spars_auc,
                    reals=reals,preds=preds)

        return len(layer_edge_label)


