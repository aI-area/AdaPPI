import os
import pandas as pd
import numpy as np
import pickle
import json

def filter_protein(protein_node_list, reference):
    ref = []
    for tmp in reference:
        ref += tmp
    
    return list_intersection(protein_node_list, ref)


def load_cliques_set(data_flag, tmp_path):
    """
    for larger sgd_cliques_set.pickle file with 160M
    """

    cliques_set_name = f'{data_flag[1]}{data_flag[2]}_cliques_set'

    if not os.path.exists(f'{tmp_path}/{cliques_set_name}.pickle'):
        cliques_set = None
        print('====No saved clique====')
    else:
        cliques_set = load_pickle(f'{tmp_path}/', f'{cliques_set_name}.pickle')
        print(f'Loading saved {cliques_set_name}.pickle')
    return cliques_set


def list_intersection(l1, l2):
    return list(set(l1).intersection(set(l2)))


def list_difference(l1, l2):
    return list(set(l1).difference(set(l2)))


def list_union(l1, l2):
    return list(set(l1).union(set(l2)))


def load_pickle(path, file_name):
    # print(f'loading:{path}{file_name}')
    with open(f'{path}{file_name}', 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(path, file_name, data_pd):
    if not os.path.exists(path):
        os.mkdir(path)
    with open(f'{path}{file_name}', 'wb') as f:
        pickle.dump(data_pd, f)


def load_json(path, file_name):

    with open(f'{path}{file_name}', 'r') as f:
        json_ = json.loads(f.read())
    return json_


def save_json(path, file_name, data_json):
    if not os.path.exists(path):
        os.mkdir(path)
    with open(f'{path}{file_name}', 'w') as f:
        json.dump(data_json, f, indent=4)


def load_complex(path, file_name, display_flag=True):
    if display_flag:
        print(f'Loading {path}{file_name}')

    complex = []
    with open(f'{path}{file_name}', 'r') as f:

        lines = f.readlines()

        for line in lines:
            node_list = line.strip('\n').strip(' ').split(' ')
            complex.append(node_list)

    return complex


def filter_modules(
    protein_set_list, protein_name_list, min_protein_num=3,
    only_intersect_flag=False
):  
    """
        filter protein set 
    """
    new_protein_set_list = []

    for ref in protein_set_list:
        one = list_intersection(ref, protein_name_list)
        if len(one) >= min_protein_num:     # 至少3个元素在里面才要
            if only_intersect_flag:
                # 在pathway中过滤掉不属于其基础蛋白的节点，
                #   包括complex的节点和（不用于组合complex和pathway）的中间蛋白
                # 若过滤complex也同理
                new_protein_set_list.append(one)
            else:
                # 过滤金标准，吧不属于该网络的节点的ref，就去除
                new_protein_set_list.append(ref)
    return new_protein_set_list


def save_complex(path, file_name, complex_set, Map_dic):
    if not os.path.exists(path):
        os.mkdir(path)

    with open(f"{path}{file_name}", "w") as f:
        for i in range(len(complex_set)):

            line = ""
            for m in complex_set[i]:
                line += Map_dic[m] + " "
            line += "\n"

            f.write(line)


def load_edge_weight(path, data_name):

    print(f'Loading {path}{data_name}_attr_sim.txt')

    edge_weight = []

    with open(f'{path}{data_name}_attr_sim.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            one = line.strip('\n').strip(' ').split(' ')
            edge_weight.append(one)

    return edge_weight


def load_attr_vector(path, data_name):
    # print(f'Loading {path}{data_name}_attr_vector.txt')

    node_feature_dict = dict()

    with open(f'{path}{data_name}_attr_vector.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line == '\n':
                continue
            one = line.strip('\n').strip(' ').split(' ')
            # one = line.strip('\n').strip(' ').split('\t')
            node = one[0]
            vector = [float(v) for v in one[1:]]
            node_feature_dict[node] = vector

    return node_feature_dict  # by order


def load_go_info(path, data_name):
    """
    node information from 'go_slim_mapping.tab.txt'
    
    P for biological process(Bp) 
    F for molecular function(Mf)
    C for cellular component(Cc)

    Since GO slims of Cc include some protein complexes information, 
        only select Bp and Mf
    """
    
    print(f'Loading {path}{data_name}_go_information.txt')

    node_go_info = []

    with open(f'{path}{data_name}_go_information.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            one = line.strip('\n').strip(' ').split(' ')
            node = one[0]
            go_tag = one[1:]
            node_go_info.append((node, go_tag))

    return node_go_info


def load_node_edge(path, data_name):
    print(f'Loading {path}{data_name}.txt')

    nodes = []
    edges = []
    
    with open(f'{path}{data_name}.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            edges.append(line.strip('\n').split('\t'))
            nodes += line.strip('\n').split('\t')
    return list(set(nodes)), edges


def load_node_order_and_edge_dict(path, data_name, display_flag=True):
    if display_flag:
        print(f'Loading {path}{data_name}.txt')

    nodes = []
    edges = []
    
    with open(f'{path}{data_name}.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            node_1, node_2 = line.strip('\n').split('\t')

            if node_1 not in nodes:
                nodes.append(node_1)
            if node_2 not in nodes:
                nodes.append(node_2)

            edge_dict = dict()
            edge_dict['node_1'] = node_1
            edge_dict['node_2'] = node_2
            edges.append(edge_dict)

    return nodes, edges


def save_node_emb(path, file_name, node_order, node_emb):

    assert len(node_order) == len(node_emb)

    if not os.path.exists(path):
        os.mkdir(path)
    
    with open(f'{path}{file_name}', 'w') as f:
        for node, emb in zip(node_order, node_emb):
            str_emb = ' '.join([str(e) for e in emb])
            f.write(f'{node} {str_emb}\n')


def load_node_emb(path, file_name):

    emb = []
    with open(f'{path}{file_name}', 'r') as f:
        lines = f.readlines()
        for line in lines:
            one = line.strip('\n').split(' ')
            emb.append([float(o) for o in one[1:]])
    return np.array(emb)


def cos_sim(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    result=dot_product / ((normA * normB) ** 0.5)
    return result


def save_node_sim(path, file_name, node_order, edge_list, node_emb):

    assert len(node_order) == len(node_emb)

    if not os.path.exists(path):
        os.mkdir(path)

    # get node embedding
    node_emb_list = []
    for node, emb in zip(node_order, node_emb):
        node_emb_dict = {'node_name': node, 'node_vector': emb}
        node_emb_list.append(node_emb_dict)

    with open(f'{path}{file_name}', 'w') as f:

        for edge_dict in edge_list:
            temp1 = 0
            temp2 = 0
            for node_emb_dict in node_emb_list:
                if edge_dict['node_1'] == node_emb_dict['node_name']:
                    v1 = np.array(node_emb_dict['node_vector'])
                    temp1 = 1

            for node_emb_dict in node_emb_list:
                if edge_dict['node_2'] == node_emb_dict['node_name']:
                    v2 = np.array(node_emb_dict['node_vector'])
                    temp2 = 1

            if temp1 == 1 and temp2 == 1:
                result = cos_sim(v1, v2)
                node_1 = edge_dict['node_1']
                node_2 = edge_dict['node_2']
                f.write(f'{node_1} {node_2} {result}\n')


def save_cluster(path, file_name, pre_label_onehot, node_list, k):

    assert pre_label_onehot.shape[0] == len(node_list)

    node_arr = np.array(node_list)

    if not os.path.exists(path):
        os.mkdir(path)

    # save all protein name in same cluster
    with open(f'{path}{file_name}', 'w') as f:
        for i in range(k):
            ind = np.where(pre_label_onehot==i)[0]
            f.write(' '.join(node_arr[ind])+'\n')
            f.write(' '.join([str(ind_) for ind_ in ind.tolist()])+'\n')


def load_node_sim(path, file_name):

    Dic_map = {}
    index = 0
    Node1 = []
    Node2 = []
    Weight = []
    All_node = set([])
    All_node_index = set([])

    with open(f"{path}{file_name}", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) == 3:
                Node1.append(line[0])
                All_node.add(line[0])
                Node2.append(line[1])
                All_node.add(line[1])
                Weight.append(float(line[2]))

                if line[0] not in Dic_map:
                    Dic_map[line[0]] = index
                    All_node_index.add(index)
                    index += 1

                if line[1] not in Dic_map:
                    Dic_map[line[1]] = index
                    All_node_index.add(index)
                    index += 1

    Map_dic = {}
    for key in Dic_map.keys():
        Map_dic[Dic_map[key]] = key

    return Map_dic, All_node_index


def load_network_adj(path, data_name):
    print(f'Loading {path}Network_{data_name}.txt')

    adj = []
    
    with open(f'{path}Network_{data_name}.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip('\n').strip(' ').split(' ')
            adj.append([int(r) for r in row])

    return np.array(adj)


def load_node_attr(path, data_name):
    print(f'Loading {path}Attribute_{data_name}.txt')

    feature_onehot = []
    
    with open(f'{path}Attribute_{data_name}.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip('\n').strip(' ').split(' ')
            feature_onehot.append([int(r) for r in row])

    return np.array(feature_onehot)


def count_different(data_pd, key):

    element_list = []
    for element in data_pd[key].to_list():
        if len(element)>0:
            element_list += element
    
    return list(set(element_list))


def joint_data_info(path, data_name, save_flag=False):

    print(f'Joint {data_name} info of nodes & edges.')

    print('\n====get node go infomation====')
    node_go_info = load_go_info(path, data_flag[1])
    print('There are', len(node_go_info), 'nodes info in', data_flag[1])

    print('\n====get node feature====')
    node_feature_dict = load_attr_vector(path, data_flag[1])
    print('There are', len(node_feature_dict), 'nodes with feature in', data_flag[1])
    print('Node feature dimension is', len(node_feature_dict['YAL001C']))

    print('\n====get egde weight (similarity of nodes)====')
    edge_weight = load_edge_weight(path, data_flag[1])

    node_pd = pd.DataFrame(node_go_info, columns=['node_protein', 'go_tags'])
    node_pd['feature'] = node_pd['node_protein'].apply(lambda np: node_feature_dict[np])
    tag_diff_list = count_different(node_pd, 'go_tags')

    edge_pd = pd.DataFrame(edge_weight, columns=['node_protein_1', 'node_protein_2', 'similarity'])

    print(node_pd)
    print('There are', len(tag_diff_list), 'different go tags.')
    print(tag_diff_list[:5])
    print(edge_pd)

    if save_flag:
        save_pickle(f'{path}pd_data_files/', f'{data_name}_node_pd.pickle', node_pd)
        save_pickle(f'{path}pd_data_files/', f'{data_name}_edge_pd.pickle', edge_pd)
    

def get_network_data(path, data_name):

    adj = load_network_adj(path, data_name)
    feature_onehot = load_node_attr(path, data_name)

    return adj, feature_onehot

