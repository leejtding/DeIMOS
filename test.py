import tensorflow as tf
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from model import DEIMOS_Model
from preprocess import get_data
from utils import tsne_visualization
import random, sys

tf.random.set_seed(123)
np.random.seed(123)
random.seed(123)

tsne_params = {
    'learning_rate': 200
}

labeled_data, unlabeled_data = get_data('data/hirise-map-proj-v3_2')

model = DEIMOS_Model(n_clusters=4)

# Pretrain the model using labeled data
if len(sys.argv) >= 2 and sys.argv[1] == 'pretrain':
    model.pretrain_setup(7)
    n_pretrain_epochs = 20
    for i in range(n_pretrain_epochs):
        print(i)
        labeled_data = labeled_data.shuffle(200)
        batched_label_ds = labeled_data.batch(50)
        curr_losses = []
        for j, (batch_img, batch_label) in enumerate(batched_label_ds):
            with tf.GradientTape() as tape:
                logits = model.call_pretrain(batch_img)
                loss = model.loss_pretrain(logits, batch_label)
                grads = tape.gradient(loss, model.trainable_variables)
                model.pretrain_optimizer.apply_gradients(zip(grads, model.trainable_variables))
            curr_losses.append(loss)
        print(f'Epoch loss: {np.mean(curr_losses)}')
    print(tf.reduce_mean(tf.cast(tf.argmax(logits, axis=1) == tf.cast(batch_label, tf.int64), float)))

    if len(sys.argv) == 3 and sys.argv[2] == 'kmeans':
        kmeans_model = KMeans(2)
        for i, batch in enumerate(unlabeled_data.batch(50)):
            out_feats = model.call_feat_output(batch)
            if not i:
                full_output = out_feats
            else:
                full_output = np.vstack((full_output, out_feats))
        kmeans_model.fit(full_output)
        predictions = kmeans_model.predict(full_output)
        score = silhouette_score(full_output, predictions)
        print(f'Silhouette Score: {score}')
        tsne_visualization(full_output, predictions, **tsne_params)
        exit()


cluster_vars = [tens for tens in model.trainable_variables if \
    tens.name != 'pretrain_output/kernel:0' and tens.name != 'pretrain_output/bias:0']

# Train loop
n_epochs = 10
for i in range(n_epochs):
    print(i)
    unlabeled_data = unlabeled_data.shuffle(200)
    batched_ds = unlabeled_data.batch(50)
    for j, batch in enumerate(batched_ds):
        with tf.GradientTape() as tape:
            out_feats = model(batch)
            #print(tf.argmax(out_feats, axis=1))
            #exit()
            loss = model.loss_w(out_feats)
            if loss is None:
                continue
            grads = tape.gradient(loss, cluster_vars)
            model.optimizer.apply_gradients(zip(grads, cluster_vars))
    print(tf.argmax(out_feats, axis=1))
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

print(np.histogram(predictions))
score = silhouette_score(full_output, predictions)
print(f'Silhouette Score: {score}')
tsne_visualization(full_output, predictions, **tsne_params)

