from pgexplainer import PGExplainer
import pickle
import sys
sys.path.append('../experiments/')
sys.path.append('../hierarchical_utils/')
import numpy as np
import torch
from tqdm import tqdm

from BAHierarchicalShapes2p.utils.models import GCN_BAHShapes,BAHierarchicalShapesDataset
from torch_geometric.data import  Data
from BAHierarchicalShapes2p.utils.edge_parser import get_explain_labels
import pickle


data_path = '../experiments/BAHierarchicalShapes2p/dataset/BAHierarchicalShapes_2proto_10000_data.pkl'
ckpt_path = '../experiments/BAHierarchicalShapes2p/ckpt/bm_BApred2p_result_0.93_ckpt.pt'
assigns_path = '../experiments/BAHierarchicalShapes2p/assign_mats/knowledge_tree.pkl'

with open(assigns_path,'rb') as f:
    trees,trees_gid = pickle.load(f)


def get_dataset(data_path):
    used_gid = []
    with open(data_path,'rb') as f:
        edge_index_list,feats,labels,node_ids = pickle.load(f)
    selected_gid,selected_edge_labels = get_explain_labels(data_path)
    data_list=[]
    for gid,edge_label in zip(selected_gid,selected_edge_labels):
        edge_index,feat,label,node_id = edge_index_list[gid],feats[gid],labels[gid],node_ids[gid]
        data = Data(x=torch.Tensor(feat), 
                    edge_index=torch.tensor(edge_index, dtype=torch.long).T, 
                    edge_label = edge_label,
                    num_nodes=len(feat),
                    y=torch.tensor([int(label)], dtype=torch.long),
                    node_id=node_id,
                    )
        data_list.append(data)
        used_gid.append(gid)
    dataset = BAHierarchicalShapesDataset(data_list)
    print(len(used_gid),len(dataset))
    return dataset,used_gid
dataset,dataset_gid = get_dataset(data_path)

def get_model(ckpt_path):
    with open(ckpt_path,'rb') as f:
        ckpt,gnn_args = torch.load(f,map_location='cuda:0')
    print(gnn_args)
    model = GCN_BAHShapes(
        input_dim=10,
        hidden_dim=gnn_args['hidden_dim'],
        label_dim=2,
        num_layers=gnn_args['num_layers'],
        pre_fc_count=gnn_args['pre_fc_count'],
        post_fc_count=gnn_args['post_fc_count'],
        no_act=gnn_args['no_act'],
        add_self_loops=gnn_args['add_self_loops'],
        mix_pool=gnn_args['mix_pool'],
        )
    print(model.load_state_dict(ckpt,strict=True))
    return model
model = get_model(ckpt_path)

import pgexplainer
from importlib import reload
reload(pgexplainer)
import bmexplain
reload(bmexplain)
import metric
reload(metric)



recall_N_10,ce_auc_10,acc_auc_10,spars_10 = [],[],[],[]
sparse_auc_10 = {}
for _ in range(10):
    explainer = pgexplainer.PGExplainer(model,in_channels=2*32,device='cuda',epochs=10,lr=0.01)
    explainer.train_explanation_network(dataset)

    mask_gid = dataset_gid
    masked_adjs = []
    for data in tqdm(dataset):
        mask = explainer(data=data)
        masked_adjs.append(mask.detach().cpu())

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

    dataset_inds = list(range(len(dataset)))
    for ind in tqdm(dataset_inds):
        data = dataset[ind]
        raw_tree = trees[trees_gid.index(dataset_gid[ind])]
        masked_adj = masked_adjs[mask_gid.index(dataset_gid[ind])]
        bmexplainer = bmexplain.BMExplainer(data=data,raw_tree=raw_tree,model=model,device='cuda',masked_adj=masked_adj)
        bmexplainer.evaluation(eval_func=metric.eval_func,recall_N=recall_N,ce_auc=ce_auc,acc_auc=acc_auc,spars=spars,spars_auc=spars_auc)
    recall_N_10.append(np.mean(recall_N))
    ce_auc_10.append(np.mean(ce_auc))
    acc_auc_10.append(np.mean(acc_auc))
    spars_10.append(np.mean(spars))
    for k,v in spars_auc.items():
        sparse_auc_10.setdefault(k,[])
        sparse_auc_10[k].append(np.mean(v))

print(f'acc_auc_10 ${np.mean(acc_auc_10):.10f}\pm{np.std(acc_auc_10):.10f}$ ')
print(f'recall_N_10 ${np.mean(recall_N_10):.10f}\pm{np.std(recall_N_10):.10f}$ ')
print(f'spars_10 ${np.mean(spars_10):.10f}\pm{np.std(spars_10):.10f}$ ')
with open('../baseline_results.txt','a') as f:
    f.write('pgexplainer bah 10cv ')
    f.write(f'acc_auc_10 ${np.mean(acc_auc_10):.10f}\pm{np.std(acc_auc_10):.10f}$ ')
    f.write(f'recall_N_10 ${np.mean(recall_N_10):.10f}\pm{np.std(recall_N_10):.10f}$ ')
    f.write(f'spars_10 ${np.mean(spars_10):.10f}\pm{np.std(spars_10):.10f}$ ')
    f.write('\n')
