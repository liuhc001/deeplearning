from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import pickle
import os

maxlen = 512
max_features = 100000
embedding_dims = 10
pos = __file__.rfind('/')
if pos == -1:
    pos = __file__.rfind('\\')
    if pos == -1:
        cur = os.path.join('data', 'reuters21578')
    else:
        cur = os.path.join(__file__[:__file__.rfind('\\')], 'data', 'reuters21578')
else:
    cur = os.path.join(__file__[:pos], 'data', 'reuters21578')


def load_data():
    data_path = os.path.join(cur, 'train_data')
    test_path = os.path.join(cur, 'test_data')
    tk_path = os.path.join(cur, 'Token')
    topic_iw = os.path.join(cur, 'topic_iw')
    topic_wi = os.path.join(cur, 'topic_wi')
    with open(tk_path, 'rb') as f:
        tk = pickle.load(f)
    with open(topic_iw, 'rb') as f:
        topic_index_word = pickle.load(f)
    with open(topic_wi, 'rb') as f:
        topic_word_index = pickle.load(f)

    def change_data(x1):
        features = {"label": tf.io.FixedLenFeature((), tf.int64, default_value=0),
                    "data": tf.io.VarLenFeature(tf.int64)}
        parsed_features = tf.io.parse_single_example(x1, features)
        return parsed_features["data"], parsed_features["label"]

    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(change_data)
    dataset_test = tf.data.TFRecordDataset(test_path)
    dataset_test = dataset_test.map(change_data)
    return tk, topic_index_word, topic_word_index, dataset, dataset_test


(tk, topic_iw, topic_wi, train_data, test_data) = load_data()


# https://keras.io/layers/recurrent/
class MyModel(keras.Model):
    def __init__(self, flag=1):
        super(MyModel, self).__init__()
        self.embedding = keras.layers.Embedding(max_features, embedding_dims, input_length=maxlen)
        if flag == 1:
            self.birnn = keras.layers.Bidirectional(keras.layers.LSTM(128))
        else:
            self.birnn = keras.layers.Bidirectional(keras.layers.GRU(128))
        self.d1 = keras.layers.Dense(len(topic_iw.keys()), activation='softmax')

    def call(self, x):
        embedding = self.embedding(x)
        x = self.birnn(embedding)
        output = self.d1(x)

        return output


model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


# @tf.function
def train_step(texts, labels):
    texts = tf.reshape(texts, [-1, maxlen])
    with tf.GradientTape() as tape:
        predictions = model(texts)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


# @tf.function
def test_step(texts, labels):
    texts = tf.reshape(texts, [-1, maxlen])
    predictions = model(texts)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


EPOCHS = 100
train_data = train_data.batch(32)
test_data = test_data.batch(32)
for epoch in range(EPOCHS):
    for texts, labels in train_data:
        train_step(texts.values, labels)
        print('.', end='')
    print('')

    for test_texts, test_labels in test_data:
        test_step(test_texts.values, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))

tf.saved_model.save(model, 'model_textbirnn')

# https://www.tinymind.cn/articles/4230
