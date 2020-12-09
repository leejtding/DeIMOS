from itertools import combinations
import tensorflow as tf


class DEIMOS_Model(tf.keras.Model):
    
    def __init__(self, n_clusters):
        super(DEIMOS_Model, self).__init__()

        # static parameters
        self.lr = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        #self.epsilon = 1e-10
        self.u_coeff = 1
        self.l_coeff = 0.1
        self.n_clusters = n_clusters

        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

        # parameters to optimize
        self.lamb = 0

        self.conv1 = tf.keras.layers.Conv2D(32, 3, strides=2, activation='relu', padding='valid')
        self.batch_norm1 = tf.keras.layers.BatchNormalization(axis=1)
        self.conv2 = tf.keras.layers.Conv2D(32, 3, strides=1, activation='relu', padding='same')
        self.batch_norm2 = tf.keras.layers.BatchNormalization(axis=1)
        self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu', padding='same')
        self.batch_norm3 = tf.keras.layers.BatchNormalization(axis=1)
        self.conv4 = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu', padding='same')
        self.batch_norm4 = tf.keras.layers.BatchNormalization(axis=1)
        self.fc1 = tf.keras.layers.Dense(self.n_clusters, activation='relu')
        self.batch_norm5 = tf.keras.layers.BatchNormalization(axis=1)
        self.fc2 = tf.keras.layers.Dense(self.n_clusters, activation='softmax')

        '''
        Dimensionality Check
        Initial: (b, 227, 227, 1)
        After conv1: (b, 112, 112, 32)
        After pool1: (b, 56, 56, 32)
        After conv2: (b, 56, 56, 32)
        After pool2: (b, 28, 28, 32)
        After conv3: (b, 28, 28, 64)
        After pool3: (b, 14, 14, 64)
        After conv4: (b, 14, 14, 64)
        After pool4: (b, 7, 7, 64)
        After flatten: (b, 12544)
        After fc1: (b, self.n_clusters)
        After fc2: (b, self.n_clusters)
        '''
    
    def pretrain_setup(self, n_classes):
        self.n_pretrain_classes = n_classes
        self.pretrain_fc_out = tf.keras.layers.Dense(n_classes, name='pretrain_output')
        self.pretrain_fc_out.trainable = True
        self.pretrain_lr = 0.01

    def call(self, inputs, training=False, mask=None):
        outs = self.conv1(inputs)
        outs = self.batch_norm1(outs, training=training)
        outs = self.max_pool(outs)
        outs = self.conv2(outs)
        outs = self.max_pool(outs)
        outs = self.batch_norm2(outs, training=training)
        outs = self.conv3(outs)
        outs = self.max_pool(outs)
        outs = self.batch_norm3(outs, training=training)
        outs = self.conv4(outs)
        outs = self.max_pool(outs)
        outs = self.batch_norm4(outs, training=training)
        outs = tf.keras.layers.Flatten()(outs)
        outs = self.fc1(outs)
        outs = self.batch_norm5(outs, training=training)
        outs = self.fc2(outs)
        return outs

    def get_clusters(self, inputs):
        '''
        inputs: Tensor with dimension (_, self.n_clusters)
        '''
        #inputs += tf.math.reduce_min(inputs)
        #inputs, _ = tf.linalg.normalize(inputs, axis=1)
        return tf.argmax(inputs, axis=1)

    def upper_bound(self):
        return 0.95 - self.u_coeff * self.lamb

    def lower_bound(self):
        return 0.455 + self.l_coeff * self.lamb

    def loss_w(self, feats):
        #feats += tf.math.reduce_min(feats)
        feats, _ = tf.linalg.normalize(feats, axis=1)
        loss = 0
        for tens_1, tens_2 in combinations(feats, 2):
            #dot_prod = tf.reduce_sum(tens_1 * tens_2) + self.epsilon
            dot_prod = tf.reduce_sum(tens_1 * tens_2)
            if dot_prod < self.lower_bound():
                loss -= tf.math.log(1 - dot_prod)
            elif dot_prod > self.upper_bound():
                loss -= tf.math.log(dot_prod)
        if loss == 0:
            return None
        return loss

    def loss_l_update(self):
        self.lamb -= self.lr * (self.u_coeff - self.l_coeff)
    
    # Pretrain call and loss
    def call_pretrain(self, inputs):
        outs = self.conv1(inputs)
        outs = self.batch_norm1(outs)
        outs = self.max_pool(outs)
        outs = self.conv2(outs)
        outs = self.max_pool(outs)
        outs = self.batch_norm2(outs)
        outs = self.conv3(outs)
        outs = self.max_pool(outs)
        outs = self.batch_norm3(outs)
        outs = self.conv4(outs)
        outs = self.max_pool(outs)
        outs = self.batch_norm4(outs)
        outs = tf.keras.layers.Flatten()(outs)
        outs = self.fc1(outs)
        outs = self.batch_norm5(outs)
        outs = self.pretrain_fc_out(outs)
        return outs

    def loss_pretrain(self, logits, labels):
        labels = tf.one_hot(labels, self.n_pretrain_classes)
        loss = tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True)
        return tf.reduce_mean(loss)
