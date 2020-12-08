import tensorflow as tf
import numpy as np
from sklearn.metrics import silhouette_score
from model import DEIMOS_Model
from preprocess import get_data
from utils import tsne_visualization
import random

tf.random.set_seed(123)
np.random.seed(123)
random.seed(123)

tsne_params = {
    'learning_rate': 200
}

labeled_data, unlabeled_data = get_data('data/hirise-map-proj-v3_2')
batched_ds = unlabeled_data.batch(90)
#batched_ds = labeled_data.batch(90)

model = DEIMOS_Model(n_clusters=10)
n_epochs = 1

# Train loop
for i in range(n_epochs):
    print(i)
    for j, batch in enumerate(batched_ds):
        with tf.GradientTape() as tape:
            out_feats = model(batch)
            loss = model.loss_w(out_feats)
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    model.loss_l_update()

# Cluster prediction
predictions = []
for i, batch in enumerate(batched_ds):
    out_feats = model(batch, training=False)
    if not i:
        full_output = out_feats
    else:
        full_output = np.vstack((full_output, out_feats))
    batch_preds = model.get_clusters(out_feats).numpy()
    predictions = np.concatenate((predictions, batch_preds))
print(full_output[:10, :])
print(np.histogram(predictions))
# Silhouette score currently doesn't work since all one cluster
#score = silhouette_score(full_output, predictions) 
#print(score)
tsne_visualization(full_output, **tsne_params)

