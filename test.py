import tensorflow as tf
import numpy as np
from model import DEIMOS_Model
from preprocess import get_data
# pretend we have this function ^^
import random

tf.random.set_seed(123)
np.random.seed(123)
random.seed(123)

_, unlabeled_data = get_data('data/hirise-map-proj-v3_2')
batched_ds = unlabeled_data.batch(64)

model = DEIMOS_Model()
n_epochs = 2

for _ in range(n_epochs):
    for batch in batched_ds:
        with tf.GradientTape() as tape:
            out_feats = model(batch)
            loss = model.loss_w(out_feats)
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    model.loss_l_update()


