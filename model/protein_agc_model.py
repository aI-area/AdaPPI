import os
import sys
import time
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

DIR_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(DIR_BASE) 

from util.data_processing import get_network_data, load_complex, load_json, \
    load_cliques_set, filter_protein
from model.core_attachment import predict_protein_functional_modules


def normalize_adj(adj, type='sym'):
    """Symmetrically normalize adjacency matrix."""
    if type == 'sym':
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        # d_inv_sqrt = np.power(rowsum, -0.5)
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # return adj*d_inv_sqrt*d_inv_sqrt.flatten()
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
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if loop:
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj, type=type)
    return adj_normalized


def to_onehot(prelabel):
    k = len(np.unique(prelabel))
    label = np.zeros([prelabel.shape[0], k])
    label[range(prelabel.shape[0]), prelabel] = 1
    label = label.T
    return label


def to_onehot_fuzzy(prob_cluster, fuzzy_num, c):

    def select_fuzzy_num():
        row = []
        col = []
        for i, pc in enumerate(prob_cluster.T):
            # 找出最大几个数的下标
            top = np.argpartition(pc, -fuzzy_num)[-fuzzy_num:].tolist()
            col += top
            row += [i] * len(top)
        row = np.array(row)
        col = np.array(col)
        return (row, col)

    # ind = select_fuzzy_num()

    ind = np.where(prob_cluster.T >= 1/50)
    
    label = np.zeros([prob_cluster.shape[1], c])
    label[ind] = 1

    return label.T


def square_dist(onehot, feature):
    if sp.issparse(feature):
        feature = feature.todense()
    feature = np.array(feature)

    m, n = onehot.shape
    count = onehot.sum(1).reshape(m, 1)
    count[count==0] = 1

    mean = onehot.dot(feature)/count
    a2 = (onehot.dot(feature*feature)/count).sum(1)
    pdist2 = np.array(a2 + a2.T - 2*mean.dot(mean.T))

    intra_dist = pdist2.trace()
    inter_dist = pdist2.sum() - intra_dist
    intra_dist /= m
    inter_dist /= m * (m - 1)
    return intra_dist


def dist(prelabel, feature):
    k = len(np.unique(prelabel))
    intra_dist = 0

    for i in range(k):
        Data_i = feature[np.where(prelabel == i)]

        Dis = euclidean_distances(Data_i, Data_i)
        n_i = Data_i.shape[0]
        if n_i == 0 or n_i == 1:
            intra_dist = intra_dist
        else:
            intra_dist = intra_dist + 1 / k * 1 / (n_i * (n_i - 1)) * sum(sum(Dis))

    return intra_dist


def AGC(reference_complex, reference_pathway):

    path = f'{data_path}{data_flag[0]}/'
    dataset = data_flag[1]

    print(f'====Get {dataset}{data_flag[2]} Graph Data====')
    if data_flag[0] in ('SGD',):
        data_path_ = f'{data_path}{data_flag[0]}/{pc_graph_data}/'
        adj = sp.load_npz(f'{data_path_}adj.npz')
        feature = sp.load_npz(f'{data_path_}feature.npz')
        protein_map_id_dict = load_json(data_path_, 'node_map.json')
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
    
    ks = {
        'krogan2006core': 30,
        'collins': 30,
        'krogan14k': 30,
        'dip': 30,
        'biogrid': 30,
        'sgd': 100,
        'pc12': {
            'humancyc': 50,
            'pid': 50,
            'panther': 50,
        }
    }
    if data_flag[1]!='pc12':
        k = ks[data_flag[1]]  
    else:
        k = ks[data_flag[1]][data_flag[2]]
    print(f'set k = {k} of k-means')
    
    print('====Start Protein-AGC model====')
    intra_list = []
    intra_list.append(10000)

    max_iter = 60
    rep = 2
    t = time.time()
    adj_normalized = preprocess_adj(adj)
    adj_normalized = (sp.eye(adj_normalized.shape[0]) + adj_normalized) / 2

    print('====Using cluster by k-means====')

    predict_labels = np.zeros(feature.shape[0])
    tt = 0
    while 1:
        tt = tt + 1
        power = tt
        intraD = np.zeros(rep)
        
        feature = adj_normalized.dot(feature)

        u, _, _ = sp.linalg.svds(feature, k=k, which='LM')

        for i in range(rep):
            kmeans = KMeans(n_clusters=k).fit(u)
            predict_labels = kmeans.predict(u)
            onehot = to_onehot(predict_labels)  # [k, n]
            
            intraD[i] = square_dist(onehot, feature)

        if data_flag[0] in ('SGD',) or data_flag[2] in ('humancyc', ):
            commit_msg = calculate_protein(u, adj, protein_map_id_dict, 
                node_complex, node_pathway, reference_complex, 
                reference_pathway, msg_flag=str(power))
        else:
            commit_msg = calculate_protein(feature, adj, protein_map_id_dict, 
                node_complex, node_pathway, reference_complex, 
                reference_pathway, msg_flag=str(power))

        intramean = np.mean(intraD)
        intra_list.append(intramean)

        if data_flag[0] in ('SGD',):
            print('both   :', commit_msg)
            print('power: {}'.format(power), 'intra_dist: {:.4f}'.format(intramean), 
                '--------------------------------------')
        else:
            print('power: {}'.format(power), 'intra_dist: {:.4f}'.format(intramean), commit_msg)

        if intra_list[tt] > intra_list[tt - 1] or tt > max_iter:
            print('best power: {}'.format(tt - 1))
            t = time.time() - t
            print('using time =', t)
            break


def calculate_protein(feature, adj, protein_map_id_dict, 
    node_complex, node_pathway, reference_complex, reference_pathway, msg_flag=''
):

    '''cosine_similarity'''
    feature_sim = cosine_similarity(feature)
    feature_sim -= np.eye(feature_sim.shape[0])     # del I matrix
    weight_adj = np.multiply(adj, feature_sim)      # weight adj based on original adj

    if data_flag[0] == 'SGD':
        commit_msg = predict_protein_functional_modules(
            data_path, tmp_path, data_flag,
            reference_golden_standard,
            reference_complex=reference_complex,
            reference_pathway=reference_pathway,
            node_complex=node_complex,
            node_pathway=node_pathway,
            cliques_set=cliques_set,
            weight_adj=weight_adj,
            save_flag=False,
            save_info_flag=f"output_{data_flag[1]}",
            sgd_protein_map_id_dict=protein_map_id_dict,
            filter_clique_flag=filter_clique_flag,
            msg_flag=msg_flag,
        )
    else:
        commit_msg = predict_protein_functional_modules(
            data_path, tmp_path, data_flag,
            reference_golden_standard,
            cliques_set=cliques_set,
            weight_adj=weight_adj,
            save_flag=False,
            save_info_flag=f"output_{data_flag[1]}{data_flag[2]}",
            sgd_protein_map_id_dict=protein_map_id_dict,
            filter_clique_flag=filter_clique_flag,
            msg_flag=msg_flag,
        )
    return commit_msg


if __name__ == '__main__':
    proj_path = f'{DIR_BASE}/'

    data_path = f'{proj_path}dataset/'
    tmp_path = f'{proj_path}tmp_protein_agc/'

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
    data_flag = all_data[1]

    '''
    Almost all nodes in ppi network participate in the pathway structure:
        pid {filter_clique_flag=False}  
        panther {filter_clique_flag=False}
    '''
    filter_clique_flag = True

    pc_graph_data = 'graph_data'

    # ==== load clique set ====
    cliques_set = load_cliques_set(data_flag, tmp_path)

    print('==== Load reference protein modules ====')
    if data_flag[0] == 'SGD':
        data_path_sgd_ = f'{data_path}{data_flag[0]}/{pc_graph_data}/'
        reference_complex = load_complex(f'{data_path_sgd_}', 'golden_standard_complex.txt')
        reference_pathway = load_complex(f'{data_path_sgd_}', 'golden_standard_pathway.txt')
        reference_golden_standard = reference_complex + reference_pathway
        print(f'reference_complex = {len(reference_complex)}')
        print(f'reference_pathway = {len(reference_pathway)}')
        print(f'reference_golden_standard = {len(reference_golden_standard)}')
    elif data_flag[0] =='PC12':
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

    AGC(reference_complex, reference_pathway)
   