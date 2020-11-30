from itertools import combinations
import tensorflow as tf


class DEIMOS_Model(tf.keras.Model):
    
    def __init__(self):
        super.__init__(self)
        
        self.is_learning = True

        # static parameters
        self.anneal_factor = 0 # need to figure out how to update
        self.lr = 0.01
        self.epsilon = 1e-10

        # parameters to optimize
        self.lamb = tf.Variable(0.)
        self.

    def call(self, inputs):
        pass

    def upper_bound(self):
        if self.is_learning:
            return 0.95 - self.lamb
        return 0.95 - self.anneal_factor

    def lower_bound(self):
        if self.is_learning:
            return 0.455 + 0.1 * self.lamb
        return 0.95 - self.anneal_factor

    def loss_w(self, feats):
        feats += tf.math.reduce_min(feats)
        feats = tf.linalg.normalize(feats, axis=1)
        loss = 0
        for tens_1, tens_2 in combinations(feats, 2):
            dot_prod = tf.reduce_sum(tens_1 * tens_2) + self.epsilon
            if dot_prod < self.lower_bound():
                loss -= tf.math.log(1 - dot_prod)
            elif dot_prod > self.upper_bound():
                loss -= tf.math.log(dot_prod)
        return loss

    def loss_l(self):
        return self.upper_bound() - self.lower_bound()
