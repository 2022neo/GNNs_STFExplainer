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
model.eval()
ori_preds = []
with torch.no_grad():
    for data in dataset:
        ori_pred = model(data).view(-1)
        ori_preds.append(ori_pred.cpu())


from importlib import reload
from sklearn.metrics import r2_score,mean_absolute_error
import bmexplain
reload(bmexplain)
import metric
reload(metric)
import explain
reload(explain)


recall_N_10,ce_auc_10,acc_auc_10,spars_10 = [],[],[],[]
sparse_auc_10 = {}
for _ in range(10):
    prog_args=explain.arg_parse()
    explainer = explain.Explainer(
        model=model,
        dataset=dataset,
        preds=ori_preds,
        args=prog_args,
        writer=None,
        print_training=False,
        graph_mode=True,
        graph_idx=prog_args.graph_idx,
    )
    masked_adjs = []
    mask_gid = []
    for ind in tqdm(range(len(dataset))):
        data = dataset[ind]
        edge_list = data.edge_index.T.numpy()
        masked_adj_batch = explainer.explain_graphs(graph_indices=[ind])
        masked_adjs.extend(masked_adj_batch)
        mask_gid.append(dataset_gid[ind])

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
    f.write('gnnexplainer bah 10cv ')
    f.write(f'acc_auc_10 ${np.mean(acc_auc_10):.10f}\pm{np.std(acc_auc_10):.10f}$ ')
    f.write(f'recall_N_10 ${np.mean(recall_N_10):.10f}\pm{np.std(recall_N_10):.10f}$ ')
    f.write(f'spars_10 ${np.mean(spars_10):.10f}\pm{np.std(spars_10):.10f}$ ')
    f.write('\n')

