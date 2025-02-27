import tensorflow as tf
from tensorflow.keras import layers
from Environment import d

initializer = tf.keras.initializers.GlorotNormal()

#实现公式（2）（3）（4），即用注意力机制实现relation-aware q representation
class Attention(layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()
        self.W = tf.Variable(initializer([d]), trainable=True)
        self.b = tf.Variable(0, dtype=tf.float32, trainable=True)

    def call(self, r_star, q_t):
        if r_star.shape == (d, 1):
            r_star = tf.transpose(r_star)

        r_star = tf.reshape(r_star, [d])

        beta_stars = tf.map_fn(lambda w_t_m: self.beta_stars(r_star, w_t_m), q_t)
        alpha_stars = tf.nn.softmax(beta_stars)
        
        q_t_stars = tf.math.multiply(alpha_stars, q_t)
        q_t_star = tf.reduce_sum(q_t_stars, 0)

        return q_t_star

    def beta_stars(self, r_star, w_t_m):
        output = r_star * w_t_m
        b_star = self.W * output + self.b
        return b_star
