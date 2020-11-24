import tensorflow as tf
from preprocessing import get_unlabeled_dataset, get_labeled_dataset
# pretend we have this function ^^

tf.random.set_seed(123)

unlabeled_dataset = get_unlabeled_dataset()
labeled_dataset = get_labeled_dataset()


