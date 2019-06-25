import tensorflow as tf
import os
from nltk import clean_html
from bs4 import BeautifulSoup
import pickle
import numpy as np
import random

pos = __file__.rfind('/')
if pos == -1:
    cur = os.path.join(__file__[:__file__.rfind('\\')], 'reuters21578')
else:
    cur = os.path.join(__file__[:pos], 'reuters21578')
file_list = os.listdir(cur)

content_list = []
topic_list = []
str_all = ''
max_length = 0
for i in file_list:
    if i.endswith('.sgm'):
        with open(os.path.join(cur, i), encoding='utf-8', errors='ignore') as f:
            str_value = f.read()
            start_topic = last_pos = str_value.find('<TOPICS>')
            end_topic = last_pos = str_value.find('</TOPICS>', last_pos)
            while last_pos != -1:
                topic = str_value[start_topic + len('<TOPICS>'):end_topic].replace('<D>', '').replace('</D>', ';')
                topic = topic.split(';')[0]
                if len(topic) == 0:
                    start_topic = last_pos = str_value.find('<TOPICS>', last_pos)
                    end_topic = last_pos = str_value.find('</TOPICS>', last_pos)
                    continue
                start_body = last_pos = str_value.find('<BODY>', last_pos)
                end_body = last_pos = str_value.find('</BODY>', last_pos)
                body = str_value[start_body + len('<BODY>'):end_body].replace('<D>', '').replace('</D>', '')
                soup = BeautifulSoup(body, 'html.parser')
                body = soup.get_text().strip()[:-1]
                start_topic = last_pos = str_value.find('<TOPICS>', last_pos)
                end_topic = last_pos = str_value.find('</TOPICS>', last_pos)
                if len(body) == 0:
                    continue
                content_list.append(body)
                topic_list.append(topic)

tk = tf.keras.preprocessing.text.Tokenizer(num_words=100000)
tk.fit_on_texts(content_list)
with open(os.path.join(cur, 'Token'), 'wb') as f:
    pickle.dump(tk, f)
topic_index_word = {}
topic_word_index = {}
for i, topic in enumerate(set(topic_list)):
    topic_index_word[i] = topic
    topic_word_index[topic] = i
with open(os.path.join(cur, 'topic_iw'), 'wb') as f:
    pickle.dump(topic_index_word, f)
with open(os.path.join(cur, 'topic_wi'), 'wb') as f:
    pickle.dump(topic_word_index, f)

index = [i for i in range(len(topic_list))]
random.shuffle(index)

writer = tf.io.TFRecordWriter(os.path.join(cur, 'train_data'))
for i in index[:int(len(topic_list) * 2 / 3)]:
    l = [topic_word_index[topic_list[i]]]
    tmp = tk.texts_to_sequences([content_list[i]])
    tmp1 = tf.keras.preprocessing.sequence.pad_sequences(tmp, maxlen=512, padding='post', truncating='post')
    c = np.array(tmp1).reshape(-1).tolist()
    feature = {'label': tf.train.Feature(int64_list=tf.train.Int64List(value=l)),
               'data': tf.train.Feature(int64_list=tf.train.Int64List(value=c))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
writer.close()

writer = tf.io.TFRecordWriter(os.path.join(cur, 'test_data'))
for i in index[int(len(topic_list) * 2 / 3):]:
    l = [topic_word_index[topic_list[i]]]
    tmp = tk.texts_to_sequences([content_list[i]])
    tmp1 = tf.keras.preprocessing.sequence.pad_sequences(tmp, maxlen=512, padding='post', truncating='post')
    c = np.array(tmp1).reshape(-1).tolist()
    feature = {'label': tf.train.Feature(int64_list=tf.train.Int64List(value=l)),
               'data': tf.train.Feature(int64_list=tf.train.Int64List(value=c))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
writer.close()

print('finish')
