from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
import os

maxlen = 512
max_features = 100000
embedding_dims = 10
pos = __file__.rfind('/')
if pos == -1:
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
    dataset_test = tf.data.TFRecordDataset(data_path)
    dataset_test = dataset_test.map(change_data)
    return tk, topic_index_word, topic_word_index, dataset, dataset_test


(tk, topic_iw, topic_wi, train_data, test_data) = load_data()


# https://github.com/ShawnyXiao/TextClassification-Keras/blob/master/model/TextCNN/text_cnn.py
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = keras.layers.Embedding(max_features, embedding_dims, input_length=maxlen)
        self.conv1_3 = keras.layers.Conv1D(128, 3, activation='relu')
        self.pool1_3 = keras.layers.GlobalMaxPooling1D()
        self.conv1_4 = keras.layers.Conv1D(128, 4, activation='relu')
        self.pool1_4 = keras.layers.GlobalMaxPooling1D()
        self.conv1_5 = keras.layers.Conv1D(128, 5, activation='relu')
        self.pool1_5 = keras.layers.GlobalMaxPooling1D()
        self.concatenate = keras.layers.Concatenate()
        self.d1 = keras.layers.Dense(len(topic_iw.keys()), activation='softmax')
        self.d2 = keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        embedding = self.embedding(x)
        convs = []
        x3 = self.conv1_3(embedding)
        x3 = self.pool1_3(x3)
        convs.append(x3)
        x4 = self.conv1_4(embedding)
        x4 = self.pool1_4(x4)
        convs.append(x4)
        x5 = self.conv1_5(embedding)
        x5 = self.pool1_5(x5)
        convs.append(x5)
        x = self.concatenate(convs)
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


EPOCHS = 20
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

# https://www.tinymind.cn/articles/4230
