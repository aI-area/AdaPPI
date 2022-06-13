import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from model.adappi.loss_estimator import get_intra_loss
from model.adappi.repres_learner import conv_act
from model.core_attachment import predict_protein_functional_modules


def run(x, graph_filter, adj, params, param_protein, cluster_k=7):
    """
    initializing param
    feature and adj matrix run in graph convolution with ACT
    calculate loss after k-means in graph embedding
    loss function = λ_tig * L_tig + λ_sep * 1/L_sep

    :param x: X = {x_1, ..., x_M}, where M is the number of nodes, shape: M * D
    :param graph_filter: Normalized adjacency matrix used for graph convolution
    :param adj_sp: sparse adjacency of graph
    :param cluster_k: nb. of classes
    :param params: all parameters used for model training
    :return: None
    """
    tf.set_random_seed(110)

    feature = tf.compat.v1.placeholder(tf.float32, shape=(None, x.shape[1]))
    plc_graph_filter = tf.compat.v1.placeholder(tf.float32, shape=graph_filter.shape)

    # for act
    # Accumulated halt signal that is below saturation value (variable threshold)
    p_t = tf.zeros(params["batch_size"], dtype=tf.float32, name="halting_probability")

    # Accumulated halt signal that is above saturation value (variable threshold)
    exceeded_p_t = tf.zeros_like(p_t, dtype=tf.float32, name="p_t_compare")
    # Index of halting convolution step
    n_t = tf.zeros(params["batch_size"], dtype=tf.float32, name="n_updates")

    # RNN model
    rnn_cell = GRUCell(params["hidden_size"])

    # Initialized state for rnn model
    state = rnn_cell.zero_state(params["batch_size"], tf.float32)

    # acculated output, i.e. y_t for {y_t^1, ..., y_t^N(t)}
    outputs_acc = tf.zeros_like(feature, dtype=tf.float32, name="output_accumulator")

    # If a node have been already halted, the mask variable is assigned as '0', otherwise is '1'
    batch_mask = tf.fill([params["batch_size"]], True, name="batch_mask")

    # ===== learn representation =====
    embedding, _, _, pt, final_n_t, _ \
        = conv_act(batch_mask, exceeded_p_t, p_t, n_t, plc_graph_filter,
                   state, feature, outputs_acc, params["max_conv_time"],
                   params["threshold"], rnn_cell
          )

    # get and record R_t
    final_r_t = 1 - pt

    # get intra_loss and inter_loss, drop other loss for save memory
    intra_loss, inter_loss, k_means_init_op, k_means_train_op = get_intra_loss(embedding, cluster_k)
    inter_loss_1 = params["inter_dist_coefficient"] * 1 / inter_loss
    loss = intra_loss + inter_loss_1

    # ===== backpropagation =====
    var_lr = tf.Variable(0.0, trainable=False)

    if params["max_grad_norm"] <= 0:
        optimizer = tf.compat.v1.train.AdamOptimizer(var_lr)
        train_step = optimizer.minimize(loss)
    else:
        tvars = tf.compat.v1.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), params["max_grad_norm"])
        optimizer = tf.compat.v1.train.AdamOptimizer(var_lr)
        train_step = optimizer.apply_gradients(zip(grads, tvars))

    # ===== Run already built computational graph =====
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

    with tf.compat.v1.Session(config=config) as sess:
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)

        # init k-means for get intra_loss & inter_loss
        sess.run(k_means_init_op, feed_dict={feature: x, plc_graph_filter: graph_filter})

        print('====train start====')

        for step in range(params["epoches"]):
            lr_decay = params["lr_decay"] ** max(step - params["max_epoch"], 0.0)
            sess.run(tf.compat.v1.assign(var_lr, params["lr"] * lr_decay))

            # start train k-means for get intra_loss & inter_loss
            _, total_loss, _, i_loss, in_loss, nt, embedding_y, rt \
                = sess.run([train_step, loss, k_means_train_op, intra_loss,
                            inter_loss_1, final_n_t, embedding, final_r_t],
                           feed_dict={feature: x, plc_graph_filter: graph_filter}
                        )

            train_msg = '[%d step(s), loss: %g, lr: %g] ' % (step + 1, total_loss, sess.run(var_lr))
            pt_msg = 'intra_loss=' + str(i_loss) + '|inter_loss=' + str(in_loss) + ' -> '

            # show number message of different convolution layers
            nt_msg = 'Nt:' + str(show_nt_msg(nt))
              
            start_time = time.time()

            if params['data_flag'][0] in ('SGD') or params['data_flag'][2] in ('humancyc',):
                u, _, _ = sp.linalg.svds(
                    np.array(embedding_y, dtype=np.float32), 
                    k=cluster_k, which='LM'
                )
            else:
                u = embedding_y

            commit_msg = calculate_protein(
                u, adj, params, param_protein, flag=f'power: {step+1}'
            )

            msg = train_msg + pt_msg + nt_msg
            if params['data_flag'][0] == 'SGD':
                print(f'         power: {step+1}{commit_msg}' , 'using time = {:.2f}'.format(time.time()-start_time))
            else:
                print(f'power: {step+1}{commit_msg}' , 'using time = {:.2f}'.format(time.time()-start_time))
            
            print(msg)
            print('----------------------------------')


def calculate_protein(feature, adj, params, param_protein, flag=''):

    '''cosine_similarity'''
    feature_sim = cosine_similarity(feature)
    feature_sim -= np.eye(feature_sim.shape[0])     # del I matrix
    weight_adj = np.multiply(adj, feature_sim)      # weight adj based on original adj

    if params['data_flag'][0] == 'SGD':
        commit_msg = predict_protein_functional_modules(
            params['data_path'], params['tmp_path'], params['data_flag'],
            param_protein['reference_golden_standard'],
            reference_complex=param_protein['reference_complex'],
            reference_pathway=param_protein['reference_pathway'],
            node_complex=param_protein['node_complex'],
            node_pathway=param_protein['node_pathway'],
            cliques_set=param_protein['cliques_set'],
            weight_adj=weight_adj,
            save_flag=False,
            save_info_flag=params['save_info_flag'],
            sgd_protein_map_id_dict=param_protein['protein_map_id_dict'],
            filter_clique_flag=params['filter_clique_flag'],
            msg_flag=flag,
        )
    else:
        commit_msg = predict_protein_functional_modules(
            params['data_path'], params['tmp_path'], params['data_flag'],
            param_protein['reference_golden_standard'],
            cliques_set=param_protein['cliques_set'],
            weight_adj=weight_adj,
            save_flag=True,
            save_info_flag=params['save_info_flag'],
            sgd_protein_map_id_dict=param_protein['protein_map_id_dict'],
            filter_clique_flag=params['filter_clique_flag'],
            msg_flag=flag,
        )
    return commit_msg


def show_nt_msg(nt):
    """
    show number message of different halting convolution layers in nt

    :param nt: Index of halting convolution step
    :return nt_dict: number message of different halting convolution layers
    """

    nt_set = set(nt)
    nt_dict = {}

    for nt_node in nt_set:
        nt_dict[nt_node] = len(np.where(nt == nt_node)[0])

    return nt_dict


