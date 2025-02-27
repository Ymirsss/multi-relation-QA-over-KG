import tensorflow as tf
from tensorflow.keras import layers
from Environment import d

initializer = tf.keras.initializers.GlorotNormal()

#就是实现公式（5），用一个perceptron计算action得分
class Perceptron(layers.Layer):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.W_L1 = tf.Variable(initializer([d, 2*d]), trainable=True)
        self.W_L2 = tf.Variable(initializer([d]), trainable=True)

    def call(self, r_star, H_t, q_t_star):
        if r_star.shape == (d, 1):
            r_star = tf.transpose(r_star)

        r_star = tf.reshape(r_star, [d])

        concat_input = tf.reshape(tf.concat([H_t, q_t_star], 0), [2*d, 1])
        
        logits_1 = tf.reshape(tf.matmul(self.W_L1, concat_input), [d])
        out_1 = tf.nn.relu(logits_1)
        out_2 = tf.math.multiply(self.W_L2, out_1)
        semantic_score = tf.math.multiply(r_star, out_2)
        semantic_score = tf.reduce_sum(semantic_score, 0)

        return semantic_score

