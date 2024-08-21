import torch
from sklearn.metrics import roc_auc_score,accuracy_score
import torch.nn.functional as F

def eval_func_reg(data,adj,model,ori_pred,layer_adj,layer_masked_adj,layer_edge_label,assign_matrix,reals,preds,spars_auc):
    layer_masked_adj = layer_masked_adj * layer_adj

    #acc_auc
    # std_tensor = torch.ones_like(layer_masked_adj, dtype=torch.float) /1000000
    # mean_tensor = layer_masked_adj
    # layer_masked_adj = torch.normal(mean=mean_tensor, std=std_tensor).sigmoid()

    spars_instance = {k:0 for k in spars_auc}
    for thred in (layer_masked_adj.unique()).sort(descending=True).values:
        masked_adj = adj * (assign_matrix @ (layer_masked_adj>=thred).float() @ assign_matrix.T)
        logits = model(data=data,adj=masked_adj).view(-1)
        marg = abs(logits.cpu()-ori_pred)/ori_pred
        spars_tmp = max(1-((layer_masked_adj >=thred).float().sum().item()/layer_adj.sum().item()),0)
        assert spars_tmp>=0
        for k in spars_instance:
            if marg<=k and spars_tmp>spars_instance[k]:
                spars_instance[k] = spars_tmp
    for k in spars_auc:
        spars_auc[k].append(spars_instance[k])


    for ind, ratio in enumerate([r* 0.1 for r in range(0,11)]):
        if ratio <= 0.05:
            masked_adj = adj * 0
        elif ratio <= 0.95:
            top_k = round(layer_adj.sum().item()* ratio)
            edge_index = torch.nonzero(layer_masked_adj, as_tuple=False).t()
            edge_weight = layer_masked_adj[edge_index[0], edge_index[1]]

            # threshold = 1.0 if len(edge_weight)==0 else float(edge_weight.reshape(-1).sort(descending=True).values[max(0,min(top_k, edge_weight.shape[0]-1))])
            # masked_adj = adj * (assign_matrix @ (layer_masked_adj>=threshold).float() @ assign_matrix.T)

            if len(edge_weight)==0:
                masked_adj = adj * 0
            else:
                pos_idx = edge_weight.reshape(-1).sort(descending=True).indices[:top_k]
                edge_index = edge_index[:,pos_idx]
                masked_adj_tmp = torch.sparse_coo_tensor(
                edge_index, torch.ones(edge_index.shape[-1]).to(edge_index.device), layer_masked_adj.shape
                ).to_dense()
                masked_adj = adj * (assign_matrix @ (masked_adj_tmp).float() @ assign_matrix.T)


        else:
            masked_adj = adj


        logits = model(data=data,adj=masked_adj)
        # if ratio > 0.95 and logits.data.cpu().item() != ori_pred.data.cpu().item():
        #     print(logits,ori_pred)
        #     pribnt(masked_adj)
        #     raise

        preds[ind].append(logits.item())

        reals[ind].append(ori_pred.item())



def eval_func(data,adj,model,ori_pred,layer_adj,layer_masked_adj,layer_edge_label,assign_matrix,recall_N,spars,acc_auc,spars_auc,ce_auc):
    layer_masked_adj = layer_masked_adj * layer_adj

    if layer_edge_label is not None and len(layer_edge_label)>0:
        #recall@N
        top_k = len(layer_edge_label)
        edge_index = torch.nonzero(layer_masked_adj, as_tuple=False).t()
        edge_weight = layer_masked_adj[edge_index[0], edge_index[1]]
        threshold = float(edge_weight.reshape(-1).sort(descending=True).values[min(top_k, edge_weight.shape[0]-1)])

        hard_mask = (edge_weight >=threshold)
        pred_edge = edge_index[:,hard_mask].T.tolist()


        for e in layer_edge_label:
            if [e[0],e[1]] in pred_edge:
                recall_N.append(1)
            elif [e[1],e[0]] in pred_edge:
                recall_N.append(1)
            else:
                recall_N.append(0)

        #spars
        spars.append(1-((layer_masked_adj >=threshold).float().sum().item()/layer_adj.sum().item()))

        #acc_rho
        # if len(edge_weight)==0:
        #     masked_adj = adj * 0
        # else:
        #     pos_idx = edge_weight.reshape(-1).sort(descending=True).indices[:top_k]
        #     edge_index = edge_index[:,pos_idx]
        #     masked_adj_tmp = torch.sparse_coo_tensor(
        #     edge_index, torch.ones(edge_index.shape[-1]).to(edge_index.device), layer_masked_adj.shape
        #     ).to_dense()
        #     masked_adj = adj * (assign_matrix @ (masked_adj_tmp).float() @ assign_matrix.T)
        # masked_adj = adj * (assign_matrix @ (layer_masked_adj>=threshold).float() @ assign_matrix.T)
        # logits = model(data=data,adj=masked_adj).view(-1)
        # assert logits.shape[-1]>1
        # logit = F.softmax(logits.squeeze(), dim=-1)
        # logit = - torch.log(logits[ori_pred] + EPS)
        # ce_auc.append(logit.item())

        # logit = F.softmax(prob.squeeze(), dim=-1)
        # logit = logit[ori_pred]
        # logit = logit + EPS
        # pred_loss = - torch.log(logit)

        # mi = logits[ori_pred] + 1e-6
        # mi = - torch.log(mi)
        # acc_auc.append(mi.item()*1e7)
        # hard_pred = logits.argmax(-1).data.cpu().item()
        # if hard_pred == ori_pred:
        #     acc_rho.append(1)
        # else:
        #     acc_rho.append(0)
    else:
        spars_instance = {k:0 for k in spars_auc}
        for thred in (layer_masked_adj.unique()).sort(descending=True).values:
            masked_adj = adj * (assign_matrix @ (layer_masked_adj>=thred).float() @ assign_matrix.T)
            logits = model(data=data,adj=masked_adj).view(-1)
            assert logits.shape[-1]>1
            logit = F.softmax(logits.squeeze(), dim=-1)
            logit = logit[ori_pred]
            spars_tmp = max(1-((layer_masked_adj >=thred).float().sum().item()/layer_adj.sum().item()),0)
            assert spars_tmp>=0
            for k in spars_instance:
                if logit>=k and spars_tmp>spars_instance[k]:
                    spars_instance[k] = spars_tmp
        for k in spars_auc:
            spars_auc[k].append(spars_instance[k])

    #acc_auc
    for ratio in [r* 0.1 for r in range(0,11)]:
        if ratio <= 0.05:
            masked_adj = adj * 0
        elif ratio <= 0.95:
            top_k = round(layer_adj.sum().item()* ratio)
            edge_index = torch.nonzero(layer_masked_adj, as_tuple=False).t()
            edge_weight = layer_masked_adj[edge_index[0], edge_index[1]]
            # threshold = 1.0 if len(edge_weight)==0 else float(edge_weight.reshape(-1).sort(descending=True).values[max(0,min(top_k, edge_weight.shape[0]-1))])
            # masked_adj = adj * (assign_matrix @ (layer_masked_adj>=threshold).float() @ assign_matrix.T)
            if len(edge_weight)==0:
                masked_adj = adj * 0
            else:
                pos_idx = edge_weight.reshape(-1).sort(descending=True).indices[:top_k]
                edge_index = edge_index[:,pos_idx]
                masked_adj_tmp = torch.sparse_coo_tensor(
                edge_index, torch.ones(edge_index.shape[-1]).to(edge_index.device), layer_masked_adj.shape
                ).to_dense()
                masked_adj = adj * (assign_matrix @ (masked_adj_tmp).float() @ assign_matrix.T)
        else:
            masked_adj = adj

        logits = model(data=data,adj=masked_adj).view(-1)
        assert logits.shape[-1]>1
        if not (layer_edge_label is not None and len(layer_edge_label)>0):
            logits = F.softmax(logits.squeeze(), dim=-1)
            logit = - torch.log(logits[ori_pred] + EPS)
            ce_auc.append(logit.item())
            

        hard_pred = logits.argmax(-1).data.cpu().item()
        if hard_pred == ori_pred:
            acc_auc.append(1)
        else:
            acc_auc.append(0)

EPS = 1e-6