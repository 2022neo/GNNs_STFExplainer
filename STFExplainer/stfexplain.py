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
import math
from ase.data import  chemical_symbols
from collections import OrderedDict

from torch.nn import ModuleList, Linear as Lin
from torch_geometric.nn import BatchNorm, ARMAConv,GCNConv
from torch_geometric.nn import global_add_pool,global_max_pool,global_mean_pool,global_sort_pool



EPS = 1e-6

class STFExplainer(nn.Module):
    def __init__(self, model, device, emb_channels, dataset, epochs: int = 20,
                 lr: float = 0.005, coff_size: float = 1e-4, coff_ent: float = 1e-2,
                 t0: float = 5.0, t1: float = 1.0,batch_size=0,hidden_dim=50,use_inputopt=False,n_layers=2, pool_net = True, pool_red='cat'):
        super(STFExplainer, self).__init__()
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.batch_size = batch_size
        self.use_inputopt=use_inputopt

        # training parameters for STFExplainer
        self.epochs = epochs
        self.lr = lr
        self.coff_size = coff_size
        self.coff_ent = coff_ent
        self.t0 = t0
        self.t1 = t1
        self.sparse_mask_values = {}
        self.ratio=0.5

        in_channels = dataset.x.shape[-1]
        # if hasattr(dataset, 'edge_attr') and dataset.edge_attr is not None:
        #     e_in_channels = dataset.edge_attr.shape[-1]
        # else:
        #     e_in_channels = 0
        self.edge_net = EdgeMaskNet(
                n_in_channels=in_channels,
                n_emb_channels = emb_channels,
                # e_in_channels=e_in_channels,
                hid=hidden_dim,
                n_layers=n_layers,
                pool_net = pool_net,
                pool_red = pool_red,
                ).to(self.device)

        
    def __loss__(self, prob: Tensor, ori_pred: int, layer):
        if prob.shape[-1]>1:
            logit = F.softmax(prob.squeeze(), dim=-1)
            logit = logit[ori_pred]
            logit = logit + EPS
            pred_loss = - torch.log(logit)
        else:
            pred_loss = F.l1_loss(prob, ori_pred.to(self.device))

        # size
        edge_mask = self.sparse_mask_values[layer]
        size_loss = self.coff_size * edge_mask.mean()

        # entropy
        # edge_mask = edge_mask * 0.99 + 0.005
        # mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)

        mask_ent = -edge_mask * torch.log(edge_mask + EPS) - (1 - edge_mask) * torch.log(1 - edge_mask + EPS)
        mask_ent_loss = self.coff_ent * mask_ent.mean()

        loss = pred_loss + size_loss + mask_ent_loss
        return loss
        
    def concrete_sample(self, log_alpha: Tensor, beta: float = 1.0, training: bool = True):
        r""" Sample from the instantiation of concrete distribution when training """
        if training:
            random_noise = torch.rand(log_alpha.size()).to(self.device)
            gate_inputs = torch.log2(random_noise) - torch.log2(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta + EPS
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs
    
    def explain(self,
                data,
                embed: Tensor,
                assign= None,
                layer=0,
                tmp: float = 1.0,
                training: bool = False,
                **kwargs)\
            -> Tuple[float, Tensor]:

        edge_index = data.edge_index
        nodesize = data.x.shape[0]
        edge_weight = torch.ones(edge_index.shape[-1]).to(self.device)
        adj = torch.sparse_coo_tensor(
        edge_index, edge_weight, (nodesize, nodesize)
        ).to_dense().to(self.device)
        embed = embed.to(self.device)



        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr

        else:
            edge_attr = None
            
        if assign == None:
            padj = adj
            pedge_index = edge_index
            pnodesize = nodesize
        else:
            assign = assign.to(self.device)
            with torch.no_grad():
                padj = ((assign.T @ adj @ assign)>0).float().to(self.device)
                pedge_index = edge_index
                pedge_index = torch.nonzero(padj).T
                pnodesize = assign.shape[-1]
                assert pedge_index.shape[0]==2

        values = self.edge_net(
            x = data.x,
            emb = embed,
            edge_index = data.edge_index,
            pedge_index = pedge_index,
            edge_attr = edge_attr,
            assign=assign,
        ).view(-1)
        values = self.concrete_sample(values, beta=tmp, training=training)
        

        self.sparse_mask_values[layer] = values

        if training:
            with torch.no_grad():
                top_k = math.ceil((torch.rand(1)*self.ratio+(1-self.ratio))*len(values))
                pos_idx = values.reshape(-1).sort(descending=True).indices[:top_k]
            mask_sparse = torch.sparse_coo_tensor(
                pedge_index[:,pos_idx], values[pos_idx], (pnodesize, pnodesize)
                )
        else:
            mask_sparse = torch.sparse_coo_tensor(
                pedge_index, values, (pnodesize, pnodesize)
            )

        mask_sigmoid = mask_sparse.to_dense()
        # set the symmetric edge weights
        sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
        layer_masked_adj = padj * sym_mask

        if layer == 0 or assign == None:
            masked_adj = layer_masked_adj
        else:
            masked_adj = adj * (assign @ layer_masked_adj @ assign.T)
        # inverse the weights before sigmoid in MessagePassing Module
        
        # the model prediction with edge mask
        self.model.eval()
        probs = self.model(data=data,adj=masked_adj)
        return (probs, layer_masked_adj, padj, adj)
    
    def train_explanation_network(self, dataset,dataset_gid,raw_trees,trees_gid,tmp = None):
        r""" training the explanation network by gradient descent(GD) using Adam optimizer """
        with torch.no_grad():
            dataset_indices = list(range(len(dataset)))
            self.model.eval()
            emb_dict = {}
            tree_dict = {}
            ori_pred_dict = {}
            for gid in tqdm.tqdm(dataset_indices):
                data = dataset[gid].to(self.device)
                raw_tree = raw_trees[trees_gid.index(dataset_gid[gid])]
                tree = {k:torch.nn.functional.one_hot(torch.LongTensor(v)).float() for k,v in raw_tree.items()}
                logits = self.model(data=data)
                emb = self.model.get_emb(data=data)
                if not isinstance(emb,Tensor):
                    emb = emb[0]
                emb_dict[gid] = emb.data.cpu()
                if self.use_inputopt:
                    tree[0]=None
                tree_dict[gid] = tree
                ori_pred_dict[gid] = logits.argmax(-1).data.cpu() if logits.shape[-1]>1 else logits.data.cpu()

        # train the mask generator
        duration = 0.0
        optimizer = Adam(self.edge_net.parameters(), lr=self.lr)
        # Print the parameters being optimized
        # for i, param_group in enumerate(optimizer.param_groups):
        #     print(f"Parameters in Group {i + 1}:")
        #     for param in param_group['params']:
        #         print(param.shape)

        # torch.autograd.detect_anomaly(True)
        with tqdm.tqdm(total=self.epochs, desc="training epoch") as pbar:
            for epoch in range(self.epochs):
                loss = 0.0
                temp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs)) if tmp is None else tmp
                self.edge_net.train()
                optimizer.zero_grad()
                tic = time.perf_counter()
                for gid in dataset_indices:
                    data = dataset[gid]
                    tree = tree_dict[gid]
                    data.to(self.device)
                    for layer,assign_matrix in tree.items():
                        prob, layer_masked_adj,layer_adj, adj = self.explain(data=data, embed=emb_dict[gid].to(self.device),assign = assign_matrix,layer=layer, tmp=temp, training=True)
                        # loss_tmp = self.__loss__(prob.squeeze(), ori_pred_dict[gid],layer)
                        loss_tmp = self.__loss__(prob, ori_pred_dict[gid],layer)
                        loss_tmp.backward()
                        loss += loss_tmp.item()
                    if self.batch_size >0 and gid+1 % self.batch_size ==0:
                        optimizer.step()
                optimizer.step()
                duration += time.perf_counter() - tic
                pbar.set_postfix({'loss' : f'{loss:.5f}'})
                pbar.update(1)
                print('epoch loss', f'{loss:.5f}')
                

    def evaluation(self,eval_func, dataset, dataset_gid,raw_trees,trees_gid,dump_result=False):
        explain_result = []
        recall_N,ce_auc,acc_auc,spars = [],[],[],[]
        spars_auc = {
            0.5:[],
            0.6:[],
            0.7:[],
            0.8:[],
            0.9:[],
            0.99:[],
            0.999:[],
        }
        self.model.eval()
        self.edge_net.eval()
        with torch.no_grad():
            dataset_indices = list(range(len(dataset)))
            ori_pred_dict = {}
            for gid in tqdm.tqdm(dataset_indices):
                data = dataset[gid].to(self.device)
                raw_tree = raw_trees[trees_gid.index(dataset_gid[gid])]
                tree = {k:torch.nn.functional.one_hot(torch.LongTensor(v)).float() for k,v in raw_tree.items()}

                logits = self.model(data=data)
                emb = self.model.get_emb(data=data)
                if not isinstance(emb,Tensor):
                    emb = emb[0]
                ori_pred = logits.argmax(-1).data.cpu() if logits.shape[-1]>1 else logits.data.cpu()
                edge_label_tree = {}
                if hasattr(data, 'edge_label'):
                    for (r,c),(l_sub,l_sup) in zip(data.edge_index.T.tolist(),data.edge_label):
                        if r<=c:
                            continue
                        for layer,assign in raw_tree.items():
                            edge_label_tree.setdefault(layer,[])
                            if l_sup==1 or l_sub==1:
                                edge_label_tree[layer].append((assign[r],assign[c]))
                explain_tree = {}
                for layer in raw_tree:
                    assign_matrix = tree[layer].to(self.device)
                    prob, layer_masked_adj, layer_adj, adj = self.explain(data=data, embed=emb.to(self.device),assign = assign_matrix,layer=layer, training=False)
                    edge_label_tree.setdefault(layer,[])
                    edge_label_tree[layer] = list(set(edge_label_tree[layer]))
                    layer_edge_label = edge_label_tree[layer]
                    eval_func(
                        data=data,adj=adj,model=self.model,ori_pred=ori_pred,
                        layer_adj=layer_adj,layer_masked_adj=layer_masked_adj,layer_edge_label=layer_edge_label,
                        assign_matrix=assign_matrix,
                        recall_N=recall_N,acc_auc=acc_auc,ce_auc=ce_auc,spars=spars,spars_auc=spars_auc)
                    if not dump_result:
                        continue
                    explain_tree[layer] = {
                        'gid':dataset_gid[gid],
                        'full_adj': adj.detach().cpu().numpy(),
                        'assign_mat':assign_matrix.detach().cpu().numpy(),
                        'layer_saliency_adj' : (layer_masked_adj * layer_adj).detach().cpu().numpy(),
                        'layer_edge_label':layer_edge_label,
                    }

                explain_result.append(explain_tree)

        return recall_N,ce_auc,acc_auc,spars,spars_auc, explain_result


    def evaluation_reg(self,eval_func_reg, dataset,dataset_gid,raw_trees,trees_gid,dump_result=False):
        print(f'dump_result={dump_result}')
        explain_result = []
        reals,preds = [[] for _ in range(11)],[[] for _ in range(11)]
        spars_auc = {
            0.1:[],
            0.05:[],
            0.01:[],
            0.005:[],
            0.001:[],
        }
        self.model.eval()
        self.edge_net.eval()
        with torch.no_grad():
            dataset_indices = list(range(len(dataset)))
            ori_pred_dict = {}
            for gid in tqdm.tqdm(dataset_indices):
                data = dataset[gid].to(self.device)
                raw_tree = raw_trees[trees_gid.index(dataset_gid[gid])]
                tree = {k:torch.nn.functional.one_hot(torch.LongTensor(v)).float() for k,v in raw_tree.items()}

                logits = self.model(data=data)
                emb = self.model.get_emb(data=data)
                if not isinstance(emb,Tensor):
                    emb = emb[0]
                ori_pred = logits.argmax(-1).data.cpu() if logits.shape[-1]>1 else logits.data.cpu()

                edge_label_tree = {}
                if hasattr(data, 'edge_label'):
                    for (r,c),(l_sub,l_sup) in zip(data.edge_index.T.tolist(),data.edge_label):
                        if r<=c:
                            continue
                        for layer,assign in raw_tree.items():
                            edge_label_tree.setdefault(layer,[])
                            if l_sup==1 or l_sub==1:
                                edge_label_tree[layer].append((assign[r],assign[c]))
                explain_tree = {}
                for layer in raw_tree:
                    assign_matrix = tree[layer].to(self.device)
                    prob, layer_masked_adj, layer_adj, adj = self.explain(data=data, embed=emb.to(self.device),assign = assign_matrix,layer=layer, training=False)
                    edge_label_tree.setdefault(layer,[])
                    edge_label_tree[layer] = list(set(edge_label_tree[layer]))
                    layer_edge_label = edge_label_tree[layer]
                    eval_func_reg(
                        data=data,adj=adj,model=self.model,ori_pred=ori_pred,
                        layer_adj=layer_adj,layer_masked_adj=layer_masked_adj,layer_edge_label=layer_edge_label,
                        assign_matrix=assign_matrix,
                        reals=reals,preds=preds,spars_auc=spars_auc)
                    if not dump_result:
                        continue
                    explain_tree[layer] = {
                        'structure_id':data.structure_id,
                        'node_symbol' : [chemical_symbols[n] for n in data.x.argmax(-1)],
                        'node_position':data.pos.detach().cpu().numpy(),
                        'full_adj': adj.detach().cpu().numpy(),
                        'assign_mat':assign_matrix.detach().cpu().numpy(),
                        'layer_saliency_adj' : (layer_masked_adj * layer_adj).detach().cpu().numpy(),
                        'layer_edge_label':layer_edge_label,
                    }


                # layer = 0
                # prob, layer_masked_adj, layer_adj, adj = self.explain(data=data, embed=emb.to(self.device),assign = None,layer=layer, training=False)
                # explain_tree[layer] = (layer_masked_adj.detach().cpu(),layer_adj.detach().cpu(),None)
                explain_result.append(explain_tree)
        return reals,preds,spars_auc,explain_result


class MLP(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, act=nn.Tanh()):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(OrderedDict([
                ('lin1', Lin(in_channels, hidden_channels)),
                ('act', act),
                ('lin2', Lin(hidden_channels, out_channels))
                ]))
     
    def forward(self, x):
        return self.mlp(x)


class EdgeMaskNet(torch.nn.Module):

    def __init__(self,
                 n_in_channels,
                 n_emb_channels,
                 pool_net = True,
                 pool_red = 'cat',
                 hid=72, n_layers=3):
        super(EdgeMaskNet, self).__init__()
        self.pool_net=pool_net
        self.pool_red=pool_red

        if self.pool_net:
            self.node_lin = Lin(n_in_channels, hid)
            self.embe_lin = Lin(n_emb_channels, hid)
            self.convs = ModuleList()
            self.batch_norms = ModuleList()
            for _ in range(n_layers):
                conv = ARMAConv(in_channels=hid, out_channels=hid)
                self.convs.append(conv)
                self.batch_norms.append(BatchNorm(hid))
            self.mlp = MLP(3 * 2 * 2 * hid, hid, 1)
        else:
            self.mlp = MLP(2 * (n_emb_channels+n_in_channels), hid, 1)
        self._initialize_weights()
        

    def forward(self, x, emb, edge_index, pedge_index, edge_attr=None,assign=None):
        if self.pool_net:
            x = torch.flatten(x, 1, -1)
            x = F.relu(self.node_lin(x))
            for conv, batch_norm in zip(self.convs, self.batch_norms):
                x = F.relu(conv(x, edge_index))
                x = batch_norm(x)
                
            emb = torch.flatten(emb, 1, -1)
            emb = F.relu(self.embe_lin(emb))

        x = torch.cat([x,emb],dim =-1)
        batch = assign.argmax(-1) if assign is not None else None
        if self.pool_red=='cat':
            if assign == None:
                x = torch.cat([x,x,x],dim =-1)
            else:
                p1 = global_add_pool(x, batch)
                p2 = global_max_pool(x, batch)
                p3 = global_mean_pool(x, batch)
                x = torch.cat([p1,p2,p3],dim =-1)
        elif self.pool_red=='add':
            x = global_add_pool(x, batch)
        elif self.pool_red=='max':
            x = global_max_pool(x, batch)
        elif self.pool_red=='avg':
            x = global_mean_pool(x, batch)
        else:
            raise

        edge_index[0, :]
        e = torch.cat([x[pedge_index[0, :]], x[pedge_index[1, :]]], dim=1)
        return self.mlp(e)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight) 
