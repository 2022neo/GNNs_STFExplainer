import torch.nn as nn
from torch import Tensor
from typing import Tuple, List, Dict, Optional
import torch
from torch_geometric.nn.conv import MessagePassing
from torch.optim import Adam
import tqdm
import time
import numpy as np
from math import sqrt
from torch_geometric.data import Data
import torch.nn.functional as F

EPS = 1e-6

class PGExplainer(nn.Module):
    def __init__(self, model, in_channels: int, device, epochs: int = 20,
                 lr: float = 0.005, coff_size: float = 0.01, coff_ent: float = 5e-4,
                 t0: float = 5.0, t1: float = 1.0, sample_bias: float = 0.0, batch_size = 100):
        super(PGExplainer, self).__init__()
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.in_channels = in_channels
        self.explain_graph = True

        # training parameters for PGExplainer
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.coff_size = coff_size
        self.coff_ent = coff_ent
        self.t0 = t0
        self.t1 = t1
        self.init_bias = 0.0
        self.sample_bias = sample_bias
        
        # Explanation model in PGExplainer
        self.elayers = nn.ModuleList()
        self.elayers.append(nn.Sequential(nn.Linear(in_channels, 64), nn.ReLU()))
        self.elayers.append(nn.Linear(64, 1))
        self.elayers.to(self.device)
        
    def __loss__(self, prob: Tensor, ori_pred: int):
        if prob.shape[-1]>1:
            logit = F.softmax(prob.squeeze(), dim=-1)
            logit = logit[ori_pred]
            logit = logit + EPS
            pred_loss = - torch.log(logit)
        else:
            pred_loss = F.l1_loss(prob, ori_pred.to(self.device))

        # size
        edge_mask = self.sparse_mask_values
        size_loss = self.coff_size * torch.sum(edge_mask)

        # entropy
        edge_mask = edge_mask * 0.99 + 0.005
        mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
        mask_ent_loss = self.coff_ent * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss
        return loss
        
    def concrete_sample(self, log_alpha: Tensor, beta: float = 1.0, training: bool = True):
        r""" Sample from the instantiation of concrete distribution when training """
        if training:
            bias = self.sample_bias
            random_noise = torch.rand(log_alpha.shape) * (1 - 2 * bias) + bias
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs
    
    def explain(self,
                data,
                embed: Tensor,
                tmp: float = 1.0,
                training: bool = False,
                **kwargs)\
            -> Tuple[float, Tensor]:
        
        edge_index = data.edge_index
        nodesize = embed.shape[0]
        col, row = edge_index
        f1 = embed[col]
        f2 = embed[row]
        f12self = torch.cat([f1, f2], dim=-1)
        
        #init edge_weight
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
            
        # using the node embedding to calculate the edge weight
        h = f12self.to(self.device)
        for elayer in self.elayers:
            h = elayer(h)
        values = h.reshape(-1)
        values = self.concrete_sample(values, beta=tmp, training=training)
        self.sparse_mask_values = values
        mask_sparse = torch.sparse_coo_tensor(
            edge_index, values, (nodesize, nodesize)
        )
        mask_sigmoid = mask_sparse.to_dense()
        # set the symmetric edge weights
        sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
        masked_adj = adj * sym_mask
        # inverse the weights before sigmoid in MessagePassing Module
        
        # the model prediction with edge mask
        self.model.eval()
        probs = self.model(data=data,adj=masked_adj)
        return probs, masked_adj
    
    def train_explanation_network(self, dataset):
        r""" training the explanation network by gradient descent(GD) using Adam optimizer """
        optimizer = Adam(self.elayers.parameters(), lr=self.lr)
        if self.explain_graph:
            with torch.no_grad():
                dataset_indices = list(range(len(dataset)))
                self.model.eval()
                emb_dict = {}
                ori_pred_dict = {}
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid].to(self.device)
                    logits = self.model(data=data)
                    emb = self.model.get_emb(data=data)
                    if not isinstance(emb,Tensor):
                        emb = emb[0]
                    emb_dict[gid] = emb.data.cpu()
                    ori_pred_dict[gid] = logits.argmax(-1).data.cpu() if logits.shape[-1]>1 else logits.data.cpu()
            # train the mask generator
            duration = 0.0
            # with torch.autograd.detect_anomaly():
            with tqdm.tqdm(total=self.epochs, desc="training epoch") as pbar:
                for epoch in range(self.epochs):
                    loss = 0.0
                    tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
                    self.elayers.train()
                    optimizer.zero_grad()
                    tic = time.perf_counter()
                    for gid in dataset_indices:
                        data = dataset[gid]
                        data.to(self.device)
                        prob, edge_mask = self.explain(data=data, embed=emb_dict[gid].to(self.device), tmp=tmp, training=True)
                        loss_tmp = self.__loss__(prob, ori_pred_dict[gid])
                        loss_tmp.backward()
                        loss += loss_tmp.item()
                    pbar.set_postfix({'loss' : f'{loss:.5f}'})
                    pbar.update(1)
                    optimizer.step()
                    duration += time.perf_counter() - tic
                    # print(f'Epoch: {epoch} | Loss: {loss}')
                
    def forward(self,data):
        self.model.eval()
        self.elayers.eval()
        data.to(self.device)
        logits = self.model(data=data)
        pred_labels = logits.argmax(dim=-1) if logits.shape[-1]>1 else logits.data.cpu()
        embed = self.model.get_emb(data=data)
        if not isinstance(embed,Tensor):
            embed = embed[0]
        if self.explain_graph:
            # original value
            # probs = probs
            # label = pred_labels
            # masked value
            _, edge_mask = self.explain(data=data, embed=embed.to(self.device), tmp=1.0, training=False)
            # selected_nodes = calculate_selected_nodes(data, edge_mask[data.edge_index[0], data.edge_index[1]], top_k)
            # masked_node_list = [node for node in range(data.x.shape[0]) if node in selected_nodes]
            # maskout_nodes_list = [node for node in range(data.x.shape[0]) if node not in selected_nodes]
            # value_func = GnnNetsGC2valueFunc(self.model, target_class=label)

            # masked_pred = gnn_score(masked_node_list, data,
            #                         value_func=value_func,
            #                         subgraph_building_method='zero_filling')

            # maskout_pred = gnn_score(maskout_nodes_list, data, value_func,
            #                          subgraph_building_method='zero_filling')

            # sparsity_score = 1 - len(selected_nodes) / data.x.shape[0]
        else:
            raise
        return edge_mask
