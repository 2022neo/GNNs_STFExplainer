import pickle
import sys
import torch
import numpy as np
from tqdm import tqdm
sys.path.append('../experiments/')
sys.path.append('../hierarchical_utils/')

from BAHierarchicalShapes2p.utils.models import GCN_BAHShapes,BAHierarchicalShapesDataset
from BAHierarchicalShapes2p.utils.edge_parser import get_explain_labels,get_knowledge_tree
from torch_geometric.data import  Data


data_path = '../experiments/BAHierarchicalShapes2p/dataset/BAHierarchicalShapes_2proto_10000_data.pkl'
ckpt_path = '../experiments/BAHierarchicalShapes2p/ckpt/bm_BApred2p_result_0.93_ckpt.pt'
assigns_path = '../experiments/BAHierarchicalShapes2p/assign_mats/knowledge_tree.pkl'




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



with open(assigns_path,'rb') as f:
    trees,trees_gid = pickle.load(f)
dataset,dataset_gid = get_dataset(data_path)
model = get_model(ckpt_path)



from importlib import reload
import stfexplain
reload(stfexplain)
import bmexplain
reload(bmexplain)
import metric
reload(metric)




def evaluate_params(hidden_dim,lr,tmp,n_layers,batch_size,coff_size):
    recall_N_10,ce_auc_10,acc_auc_10,spars_10 = [],[],[],[]
    sparse_auc_10 = {}
    for _ in range(10):
        explainer = stfexplain.STFExplainer(model,emb_channels=32,dataset=dataset,device='cuda',coff_size=coff_size,\
            epochs=40,hidden_dim=hidden_dim,lr=lr,use_inputopt=False,n_layers=n_layers,batch_size=batch_size)
        explainer.train_explanation_network(dataset=dataset,dataset_gid=dataset_gid,raw_trees=trees,trees_gid=trees_gid,tmp=tmp)
        recall_N,ce_auc,acc_auc,spars,spars_auc, explain_result = explainer.evaluation(eval_func=metric.eval_func,dataset=dataset,dataset_gid=dataset_gid,raw_trees=trees,trees_gid=trees_gid,dump_result=False)
        recall_N_10.append(np.mean(recall_N))
        ce_auc_10.append(np.mean(ce_auc))
        acc_auc_10.append(np.mean(acc_auc))
        spars_10.append(np.mean(spars))
        for k,v in spars_auc.items():
            sparse_auc_10.setdefault(k,[])
            sparse_auc_10[k].append(np.mean(v))
        # with open('./recall_N.txt','a') as f:
        #     f.write(f'{np.mean(recall_N)}')
        #     f.write(' ')
        # with open('./bah_acc_auc.txt','a') as f:
        #     f.write(f'{np.mean(acc_auc)}')
        #     f.write(' ')

    print(f'acc_auc_10 ${np.mean(acc_auc_10):.10f}\pm{np.std(acc_auc_10):.10f}$ ')
    print(f'recall_N_10 ${np.mean(recall_N_10):.10f}\pm{np.std(recall_N_10):.10f}$ ')
    print(f'spars_10 ${np.mean(spars_10):.10f}\pm{np.std(spars_10):.10f}$ ')



    # with open('../experiments/BAHierarchicalShapes2p/explains/STFExplains_bah.pkl','wb') as f:
    #     pickle.dump(explain_result, f)
    # print(len(explain_result))

    with open(histo_path,'a') as f:
        f.write(f'acc_auc_10 ${np.mean(acc_auc_10):.10f}\pm{np.std(acc_auc_10):.10f}$ ')
        f.write(f'recall_N_10 ${np.mean(recall_N_10):.10f}\pm{np.std(recall_N_10):.10f}$ ')
        f.write(f'spars_10 ${np.mean(spars_10):.10f}\pm{np.std(spars_10):.10f}$')
        f.write(f'params:{hidden_dim,lr,tmp,n_layers,batch_size,coff_size} ')
        f.write('\n')

    
histo_path='../STFExplainer/crossv/params_bah_cv.txt'
# with open(histo_path,'r') as f:
#     histo = [eval(line.split('params:')[-1].strip()) for line in f.readlines()]

def main_param():
    for _ in range(3):
        hidden_dim,lr,tmp,n_layers,batch_size,coff_size=(32, 0.001, None, 0, 0, 0.5)
        evaluate_params(hidden_dim,lr,tmp,n_layers,batch_size,coff_size)


if __name__=='__main__':
    main_param()


