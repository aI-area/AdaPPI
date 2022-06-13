import os
import sys
import numpy as np
import networkx as nx

DIR_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(DIR_BASE)

from util.data_processing import load_node_sim, save_complex, save_pickle, filter_modules
from evaluator.compare_performance import get_score


def load_clique(cliques_file_path):
    cliques_set = []

    with open(cliques_file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            temp_set = []
            line = line.strip().split()
            for i in range(1, len(line)):
                temp_set.append(int(line[i]))
            cliques_set.append(temp_set)

    return cliques_set


def get_clique(adj):
    edge = [(n1, n2) for n1, n2 in zip(adj.nonzero()[0], adj.nonzero()[1])]

    G = nx.Graph()
    G.add_edges_from(edge)

    cliques = nx.find_cliques(G)

    return [c for c in cliques if len(c)>=3]


def f_key(a):
    return a[-1]


def density_score(temp_set, matrix):
    temp_density_score = 0.
    # O(n^2)
    for m in temp_set:
        for n in temp_set:
            if n != m and matrix[m, n] != 0:
                temp_density_score += matrix[m, n]

    # clique_density = sum(wij)/n(n-1)
    temp_density_score = temp_density_score / (len(temp_set) * (len(temp_set) - 1))
    return temp_density_score


def find_core_cliques(new_cliques_set):
    
    seed_clique = []

    while True:
        temp_cliques_set = []
        if len(new_cliques_set) >= 2:

            seed_clique.append(new_cliques_set[0])

            for i in range(1, len(new_cliques_set)):
                if len(new_cliques_set[i].intersection(new_cliques_set[0])) == 0:
                    temp_cliques_set.append(new_cliques_set[i])

                elif len(new_cliques_set[i].intersection(new_cliques_set[0])) >= 2 and \
                    len(new_cliques_set[i].difference(new_cliques_set[0])) >= 3:
                    temp_cliques_set.append(new_cliques_set[i].difference(new_cliques_set[0]))

            new_cliques_set = temp_cliques_set

        elif len(new_cliques_set) == 1:
            seed_clique.append(new_cliques_set[0])
            break
        else:
            break

    return seed_clique


def expand_core_cliques(seed_clique, all_protein_set, matrix, expand_thres):
    expand_set = []
    complex_set = []

    for instance in seed_clique:
        # avg_node_score = density_score(instance, matrix)

        # 计算单个蛋白在clique中的相关性
        temp_set = set([])
        for j in all_protein_set.difference(instance):
            temp_score = 0.
            for n in instance:
                temp_score += matrix[n, j]
            temp_score /= len(instance)

            # 假如一个新蛋白的加入之后，相似度超越阈值，
            if temp_score >= expand_thres:
                temp_set.add(j)
        expand_set.append(temp_set)

    for i in range(len(seed_clique)):
        complex_set.append(seed_clique[i].union(expand_set[i]))

    return complex_set


def predict_protein_functional_modules(data_path, tmp_path, data_flag, reference_golden_standard,
    reference_complex=None, reference_pathway=None, node_complex=None, node_pathway=None,
    cliques_set=None, weight_adj=None, save_flag=False, save_info_flag=None,
    sgd_protein_map_id_dict=None, filter_clique_flag=False, msg_flag='',
    min_protein_num=3,
):

    id_map_protein_dict, All_node_index = load_node_sim(
        f'{data_path}{data_flag[0]}/{data_flag[2]}/',
        f'{data_flag[1]}_attr_sim.txt',
    )

    if data_flag[0] in ('SGD', 'PC12') and sgd_protein_map_id_dict is not None:
        # id_map_protein_dict of SGD and PC12 is difference with others
        id_map_protein_dict = {v:k for k, v in sgd_protein_map_id_dict.items()}
        
    # find reference sets belong to the current ppi
    reference_golden_standard = filter_modules(
        reference_golden_standard, 
        list(id_map_protein_dict.values()), 
        min_protein_num=min_protein_num,
        only_intersect_flag=False
    )

    if cliques_set is None:     
        cliques_set = get_clique(weight_adj)
        save_pickle(f'{tmp_path}', f'{data_flag[1]}{data_flag[2]}_cliques_set.pickle', cliques_set)

        if data_flag[0] in ('SGD', ):
            print('Avoid saving the file repeatedly, you can re-execute the code after saving the file once')
            assert False

    # calculate density
    avg_clique_score = 0.
    tmp_cliques_set = []
    for instance in cliques_set:
        clique_score = density_score(instance, weight_adj)
        avg_clique_score += clique_score
        tmp_cliques_set.append(instance+[clique_score])
    avg_clique_score /= len(tmp_cliques_set)

    # delete cliques that their density are less than average density
    if filter_clique_flag:
        tmp_cliques_set = [tmp for tmp in tmp_cliques_set if tmp[-1]>=avg_clique_score]
    
    # sort cliques by density
    tmp_cliques_set.sort(key=f_key, reverse=True)

    # delete information of density
    new_cliques_set = []
    for i in range(len(tmp_cliques_set)):
        temp_set = set([])
        for j in range(len(tmp_cliques_set[i]) - 1):
            temp_set.add(tmp_cliques_set[i][j])
        new_cliques_set.append(temp_set)
 
    # find core cliques
    seed_clique = find_core_cliques(new_cliques_set)
    
    # expand core cliques
    expand_thres = 0.3  # setting by GANE
    complex_set = expand_core_cliques(seed_clique, All_node_index, weight_adj, expand_thres)

    if save_flag:
        if save_info_flag is not None and len(msg_flag)>0:
            step = msg_flag.split(' ')[-1]
            save_complex(
                f'{tmp_path}/{save_info_flag}/', 
                f'final_{data_flag[1]}{data_flag[2]}_attr_output_{step}.txt', 
                complex_set, id_map_protein_dict
            )
        else:
            save_complex(
                tmp_path, 
                f'final_{data_flag[1]}_attr_output_complex.txt', 
                complex_set, id_map_protein_dict
            )

    # id to protein name
    predicted_modules = []
    for complex_ in complex_set:
        predicted_modules.append([id_map_protein_dict[c] for c in complex_])

    if reference_complex is not None:
        assert data_flag[0] in ('SGD',)
        # find reference complexes belong to the current ppi
        reference_complex = filter_modules(
            reference_complex, list(id_map_protein_dict.values()), 
            min_protein_num=min_protein_num,
            only_intersect_flag=False
        )
        # find predict complexes belong to the current ppi
        predicted_complex_ = filter_modules(
            predicted_modules, node_complex, 
            min_protein_num=min_protein_num,
            only_intersect_flag=False
        )
        score_msg = get_score(reference_complex, predicted_complex_)
        print(f'complex: {msg_flag}{score_msg}')

    if reference_pathway is not None:
        assert data_flag[0] in ('SGD',)
        # find reference pathways belong to the current ppi
        reference_pathway = filter_modules(
            reference_pathway, list(id_map_protein_dict.values()), 
            min_protein_num=min_protein_num,
            only_intersect_flag=False
        )
        # find predict pathways belong to the current ppi
        predicted_pathway_ = filter_modules(
            predicted_modules, node_pathway, min_protein_num=min_protein_num,
            only_intersect_flag=False
        )
        score_msg = get_score(reference_pathway, predicted_pathway_)
        print(f'pathway: {msg_flag}{score_msg}')

    return get_score(reference_golden_standard, predicted_modules)

