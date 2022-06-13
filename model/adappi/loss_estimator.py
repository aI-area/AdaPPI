import tensorflow as tf


def get_intra_loss(embedding, cluster_k):
    """

    get k-means result
    intra_dist and inter_dist of embedding function from AGC

    intra_dist = L_tig = 1/|C| Σ 1/|c|(|c|-1) Σ||xj-xj|| (xi, xj belong to same cluster)
    inter_dist = L_sep = 1/|C| Σ 1/|c|(|c|-1) Σ||xj-xj|| (xi, xj belong to different cluster respectively)

    :param embedding: embedding in graph-act
    :param cluster_k: number of clusters
    :return intra_dist_loss, inter_dist_loss, k-means_init_op and k-means_train_op

    """
    # K-Means Parameters
    kmeans = tf.contrib.factorization.KMeans(inputs=embedding,
                                             num_clusters=cluster_k,
                                             initial_clusters='kmeans_plus_plus',
                                             distance_metric='cosine',
                                             use_mini_batch=True)

    # Build KMeans graph
    training_graph = kmeans.training_graph()

    _, cluster_idx, _, _, init_op, train_op = training_graph

    # cluster_idx is tuple
    pred_label = cluster_idx[0]

    # using index to get node onehot (k,n),
    index = pred_label

    # get nodes that belongs to first cluster become onehot's first col (n,1)
    zero_one_hot = tf.expand_dims(tf.cast(tf.equal(pred_label, 0), tf.int64), -1)

    for _ in range(cluster_k - 1):
        index = index - 1
        index_one_hot = tf.expand_dims(tf.cast(tf.equal(index, 0), tf.int64), -1)
        # concat other cluster
        zero_one_hot = tf.concat([zero_one_hot, index_one_hot], 1)

    # zero_one_hot has concated all onehot, (n, k), int64
    # onehot is (k, n) and float32
    onehot = tf.cast(tf.transpose(zero_one_hot), tf.float32)

    # number of every cluster
    count = tf.expand_dims(tf.reduce_sum(onehot, 1), -1)
    # count = print_debug(count, 'number of each cluster', True)

    '''calculate euclidean_distances with count for mean'''
    # [(k, n) * (n, d)] / number of cluster => (k, d)
    mean = tf.matmul(onehot, embedding) / count

    # feature * feature => xi^2, (k, d).sum(1) => (k, 1)
    square_f = tf.reduce_sum(tf.matmul(onehot, embedding * embedding) / count, 1)

    # col & row vector of square_f
    c_f = tf.reshape(square_f, [-1, 1])
    r_f = tf.reshape(square_f, [1, -1])

    # c_f + r_f => (k, k)     mean.dot(mean.T) => (k, k)
    eu_dist = c_f + r_f - 2 * tf.matmul(mean, tf.transpose(mean))

    # euclidean_distances make (intra_dist + inter_dist) lower
    intra_dist = tf.linalg.trace(eu_dist)

    # all eu_dist = intra_dist + inter_dist
    inter_dist = tf.reduce_sum(eu_dist) - intra_dist

    intra_dist /= cluster_k
    inter_dist /= cluster_k * (cluster_k - 1)

    return intra_dist, inter_dist, init_op, train_op
