import tensorflow as tf
import numpy as np
# import keras
from tensorflow import keras

x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
              [6.0, 7.0, 8.0, 9.0, 10.0]])
y = [1, 2]

path_tmp = r'tf_record'


def write_record():
    writer = tf.python_io.TFRecordWriter(path_tmp)
    for i in range(x.shape[0]):
        feature = {'data': tf.train.Feature(float_list=tf.train.FloatList(value=x[i])),
                   'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[y[i]]))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()


def parser(x):
    features = tf.parse_single_example(x,
                                       features={'data': tf.VarLenFeature(tf.float32),
                                                 'label': tf.FixedLenFeature([], tf.int64)})
    return features


dataset = tf.data.TFRecordDataset(path_tmp)
# dataset.
dataset = dataset.map(parser)
dataset = dataset.repeat(10000)
# dataset = dataset.shuffle(100)
dataset = dataset.batch(1)
iter = dataset.make_initializable_iterator()
m = iter.get_next()
x = tf.reshape(m['data'].values, (-1, 5))
y = m['label']

with tf.Session() as sess:
    #    sess.run(tf.global_variables_initializer())
    sess.run(iter.initializer)
    # a1, a2 = sess.run([x, y])
    # print(a1)
    # print('----')
    sgd = keras.optimizers.SGD(lr=0.001)
    model = keras.Sequential()
    model.add(keras.layers.Dense(2, activation='softmax', input_dim=5))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd)
    h = model.fit(x, y, epochs=50, steps_per_epoch=2)
#print(h)
