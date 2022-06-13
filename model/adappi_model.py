import os
import sys
import time
import numpy as np
import scipy.sparse as sp
import warnings
import json

DIR_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(DIR_BASE) 
from util.data_processing import get_network_data, load_complex, \
    load_cliques_set, load_json, filter_protein
from adappi.trainer import run


def normalize_adj(adj, type='sym'):
    """Totally same as AGC paper
    Symmetrically normalize adjacency matrix. Derived from github"""
    if type == 'sym':
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    elif type == 'rw':
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj_normalized = d_mat_inv.dot(adj)
        return adj_normalized


def preprocess_adj(adj, type='sym', loop=True):
    """Totally same as AGC paper
    Preprocessing of adjacency matrix for simple
    GCN model and conversion to tuple representation."""
    if loop:
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj, type=type)
    return adj_normalized


def AdaPPI(reference_complex, reference_pathway):
    warnings.filterwarnings("ignore")
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = ' '    # set cpu

    path = f'{data_path}{data_flag[0]}/'
    dataset = data_flag[1]
    print(f'====Get {dataset}{data_flag[2]} Graph Data====')
    if data_flag[0] in ('SGD',):
        data_path_sgd = f'{data_path}SGD/graph_data/'
        adj = sp.load_npz(f'{data_path_sgd}adj.npz')
        feature = sp.load_npz(f'{data_path_sgd}feature.npz')
        protein_map_id_dict = load_json(data_path_sgd, 'node_map.json')
        # split difference nodes
        protein_node_list = list(protein_map_id_dict.keys())
        node_complex = filter_protein(protein_node_list, reference_complex) 
        node_pathway = filter_protein(protein_node_list, reference_pathway)
        print(f'node_complex={len(node_complex)}, node_pathway={len(node_pathway)}')
    elif data_flag[0] in ('PC12',):
        data_path_ = f'{data_path}{data_flag[0]}/{pc_graph_data}/'
        adj = sp.load_npz(f'{data_path_}adj.npz')
        feature = sp.load_npz(f'{data_path_}feature.npz')
        protein_map_id_dict = load_json(data_path_, 'node_map.json')
        node_complex, node_pathway = None, None  
    else:
        adj, feature = get_network_data(path, dataset)
        protein_map_id_dict = None
        node_complex, node_pathway = None, None 

    if sp.issparse(feature):
        feature = feature.todense()
    if sp.issparse(adj):
        adj = np.array(adj.todense())
    
    print(f'adj shape={adj.shape}, feature shape ={feature.shape}')
    print(f'edge = {np.sum(adj)}, node average desity={np.mean(adj.sum(1))}')

    # D^-1/2 A D^-1/2 or D^-1 A
    adj_normalized = preprocess_adj(adj)
    # G = 1/2（I + D^-1/2 A D^-1/2）
    g = (sp.eye(adj_normalized.shape[0]) + adj_normalized) / 2

    if data_flag[0] == 'Krogan-core':
        param = {
            'k': 30, 
            'lr': 0.0001, 
            'epoches': 500, 
            'max_conv_time': 10,  
            'threshold': 0.99,  
            'batch_size': feature.shape[0],  
            'hidden_size': 200,  
            'inter_dist_coefficient': 1,  
            'intra_dist_coefficient': 1, 
            'lr_decay': 1,  
            'max_epoch': 0,  
            'max_grad_norm': 0,  
            'data_path': data_path,
            'data_flag': data_flag,
            'tmp_path': tmp_path,
            'filter_clique_flag': True,
        }
    elif data_flag[0] == 'SGD':
        param = {
            'k': 100,                       
            'lr': 0.0001,                  
            'epoches': 500,                
            'max_conv_time': 10,           
            'threshold': 0.99,              
            'batch_size': feature.shape[0], 
            'hidden_size': 200,             
            'inter_dist_coefficient': 1,  
            'intra_dist_coefficient': 1,    
            'lr_decay': 1,                  
            'max_epoch': 0,                 
            'max_grad_norm': 0,             
            'dataset_name': dataset,
            'data_path': data_path,
            'data_flag': data_flag,
            'tmp_path': tmp_path,
            'filter_clique_flag': True,
        }
    elif data_flag[2] == 'humancyc':
        param = {
            'k': 50,                        
            'lr': 0.0001,                   
            'epoches': 500,                
            'max_conv_time': 10,            
            'threshold': 0.99,              
            'batch_size': feature.shape[0],
            'hidden_size': 200,             
            'inter_dist_coefficient': 1/7, 
            'intra_dist_coefficient': 1,    
            'lr_decay': 1,                  
            'max_epoch': 0,                 
            'max_grad_norm': 0,  
            'dataset_name': dataset,
            'data_path': data_path,
            'data_flag': data_flag,
            'tmp_path': tmp_path,
            'filter_clique_flag': True,
        }
    else:
        param = {}

    save_info_flag = f"{param['data_flag'][1]}_inter_{param['inter_dist_coefficient']}_max_{param['max_conv_time']}_k_{param['k']}"
    param['save_info_flag'] = save_info_flag

    print(json.dumps(param, indent=4))

    param_protein = {
        'cliques_set': cliques_set,
        'protein_map_id_dict': protein_map_id_dict,
        'reference_golden_standard': reference_golden_standard,
        'reference_complex': reference_complex,
        'reference_pathway': reference_pathway,
        'node_complex': node_complex,
        'node_pathway': node_pathway
    }

    run(feature, g.toarray(), adj, param, param_protein, cluster_k=param['k'])


if __name__ == '__main__':
    proj_path = f'{DIR_BASE}/'

    data_path = f'{proj_path}dataset/'
    tmp_path = f'{proj_path}tmp_adappi/'

    pc_data = 'humancyc'    # pid, humancyc, panther

    all_data = [
        ('Krogan-core', 'krogan2006core', ''), 
        ('COLLINS', 'collins', ''),
        ('Krogan14k', 'krogan14k', ''),
        ('DIP', 'dip', ''),
        ('BIOGRID', 'biogrid', ''),
        ('SGD', 'sgd', ''),  # SGD includes complex and pathway
        ('PC12', 'pc12', pc_data)  # only pathway
    ]
    data_flag = all_data[0]

    # ==== load clique set ====
    cliques_set = load_cliques_set(data_flag, tmp_path)

    pc_graph_data = 'graph_data'

    print('==== Load reference protein modules ====')
    if data_flag[0] == 'SGD':
        data_path_sgd_ = f'{data_path}{data_flag[0]}/{pc_graph_data}/'
        reference_complex = load_complex(f'{data_path_sgd_}', 'golden_standard_complex.txt')
        reference_pathway = load_complex(f'{data_path_sgd_}', 'golden_standard_pathway.txt')
        reference_golden_standard = reference_complex + reference_pathway
        print(f'reference_complex = {len(reference_complex)}')
        print(f'reference_pathway = {len(reference_pathway)}')
        print(f'reference_golden_standard = {len(reference_golden_standard)}')
    elif data_flag[0] == 'PC12':
        pc_graph_data = f'graph_data_{pc_data}'
        data_path_ = f'{data_path}{data_flag[0]}/{pc_graph_data}/'
        reference_pathway = load_complex(f'{data_path_}', 'golden_standard_pathway.txt')
        reference_golden_standard = reference_pathway
        reference_complex, reference_pathway = None, None
        print('reference_golden_standard len =', len(reference_golden_standard))
    else:
        reference_golden_standard = load_complex(f'{data_path}', 'golden_standard.txt')
        reference_complex, reference_pathway = None, None
        print('reference_golden_standard len =', len(reference_golden_standard))

    AdaPPI(reference_complex, reference_pathway)