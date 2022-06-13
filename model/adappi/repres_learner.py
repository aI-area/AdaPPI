import tensorflow as tf
from tensorflow.contrib.rnn import static_rnn


def conv_act(batch_mask, exceeded_p_t, p_t, n_t, graph_filter, state, feature,
             output_acc, max_conv_time, threshold, rnn_cell):
    """
    Code is derived from https://github.com/DeNeutoy/act-tensorflow/blob/
    master/src/act_cell.py

    Assume the nb. of nodes is M, dimension of node representation is D

    :param batch_mask: If a node have been already halted, the mask variable
    is assigned as '0', otherwise '1'
    :param exceeded_p_t: Accumulated halt signal that is exactly above
    saturation value
    :param p_t: Accumulated halt signal that is exactly below saturation value
    :param n_t: Index of halting convolution step
    :param graph_filter: graph convolution function that G in AGC paper
    :param state: internal initial state of a sequence-to-sequence model
    :param feature: X: X = {x_1, ..., x_M}, where M is the number of nodes,
    shape: M * D
    :param output_acc: acculated output, i.e. y_t for {y_t^1, ..., y_t^N(t)}
    :param max_conv_time: maximum convolution times
    :param threshold: saturation value set for marking exit signal
    :param rnn_cell: A specified sequence-to-sequence model

    :return embedding: final feature after iteration
    :return mask: final batch_mask after iteration
    :return fin_p_t_compare: final exceeded_p_t after iteration
    :return fin_p_t: final p_t after iteration
    :return fin_n_t: final n_t after iteration
    :return last_conv_embed: node representation from last convolution,
    not linearized combination of all convolution layers like embedding

    """

    def act_step(lc_batch_mask, p_t_compare, lc_p_t, lc_n_t, lc_state,
                 input_x, lc_output_acc):
        """
        Executing update of node representation and convolution along
        iterations.

        This function is regarded as main body of a loop.

        :param lc_batch_mask: Same as batch_mask, but inside iteration
        :param p_t_compare: Same as exceeded_p_t, but inside iteration
        :param lc_p_t: Same as p_t, but inside iteration
        :param lc_n_t: Same as n_t, but inside iteration
        :param lc_state: Same as state, but inside iteration
        :param input_x: Same as feature, but inside iteration
        :param lc_output_acc: Same as output_acc, but inside iteration

        :return new_lc_batch_mask: updated batch_mask along with iteration
        :return p_t_compare: updated exceeded_p_t along with iteration
        :return lc_p_t: updated p_t along with iteration
        :return lc_n_t: updated n_t along with iteration
        :return new_lc_state: updated state along with iteration
        :return y: updated feature (convoluted representation) along with
        iteration
        :return lc_output_acc: updated ouput_acc along with iteration
        """

        # Execute current convolution on last layer
        y = tf.matmul(graph_filter, input_x)

        # Difference between current and last layer is considered as final input
        sub_conv = y

        _, new_lc_state = static_rnn(cell=rnn_cell,
                                     inputs=[sub_conv],
                                     initial_state=lc_state,
                                     scope=type(rnn_cell).__name__)

        # Equation: p_t^n = sigmoid(W*s+b)
        halt = tf.squeeze(tf.layers.dense(new_lc_state, 1,
                                          activation=tf.nn.sigmoid,
                                          use_bias=True,
                                          kernel_initializer=None),
                          squeeze_dims=1)

        # Multiply by the previous mask as if we stopped before,
        # we don't want to start again
        # if we generate a p less than p_t-1 for a given example.
        # True: need to keep iteration; False: need to stop
        new_lc_batch_mask = tf.logical_and(tf.less(lc_p_t + halt, threshold),
                                           lc_batch_mask)
        new_float_mask = tf.cast(new_lc_batch_mask, tf.float32)

        # Only increase the prob accumulator for the examples
        # which haven't already passed the threshold. This
        # means that we can just use the final prob value per
        # example to determine the remainder.
        lc_p_t += halt * new_float_mask

        # This accumulator is used solely in the While loop condition.
        # we multiply by the PREVIOUS batch mask, to capture probabilities
        # that have gone OVER 1-eps THIS iteration.
        p_t_compare += halt * tf.cast(lc_batch_mask, tf.float32)

        # Only increase the counter for those probabilities that
        # did not go over 1-eps in this iteration.
        # As iteration need to be continued, new_float_mask=1.0;
        # otherwise, new_float_mask=0.0
        lc_n_t += new_float_mask

        # Halting condition (halts, and uses the remainder when this is FALSE):
        # If any batch element still has both a prob < 1 - epsilon AND counter
        # < N we continue, using the outputed probability p.
        n_t_condition = tf.less(lc_n_t, max_conv_time)
        final_iteration_condition = tf.logical_and(new_lc_batch_mask,
                                                   n_t_condition)

        # Variable R(t) in paper
        remainder = tf.expand_dims(1.0 - lc_p_t, -1)
        probability = tf.expand_dims(halt, -1)

        # Choosing remainder of probability
        update_weight = tf.where(final_iteration_condition,
                                 probability, remainder)

        float_mask = tf.expand_dims(tf.cast(lc_batch_mask, tf.float32), -1)
        # states_acc += (new_state * update_weight * float_mask)
        # updated convolution result
        lc_output_acc += (y * update_weight * float_mask)

        return [new_lc_batch_mask, p_t_compare, lc_p_t, lc_n_t,
                new_lc_state, y, lc_output_acc]

    def should_continue(lc_batch_mask, p_t_compare, lc_p_t, lc_n_t,
                        lc_state, input_x, lc_output_acc):
        """
        While loop stops when this predicate is FALSE.
        Ie all (probability < 1-eps AND counter < N) are false.

        This function is regarded as condition of a loop.

        :param lc_batch_mask: Same as batch_mask, but inside iteration
        :param p_t_compare: Same as exceeded_p_t, but inside iteration
        :param lc_p_t: Same as p_t, but inside iteration
        :param lc_n_t: Same as n_t, but inside iteration
        :param lc_state: Same as state, but inside iteration
        :param input_x: Same as feature, but inside iteration
        :param lc_output_acc: Same as output_acc, but inside iteration

        :return continue_flag: whether iteration should be continued - for
        every node
        """

        continue_flag = tf.reduce_any(
            tf.logical_and(
                tf.less(p_t_compare, threshold),
                tf.less(lc_n_t, max_conv_time)))

        return continue_flag

    mask, fin_p_t_compare, fin_p_t, fin_n_t, _, last_conv_embed, embedding = \
        tf.while_loop(should_continue, act_step,
                      loop_vars=[batch_mask, exceeded_p_t, p_t, n_t,
                                 state, feature, output_acc])

    return embedding, mask, fin_p_t_compare, fin_p_t, fin_n_t, last_conv_embed
