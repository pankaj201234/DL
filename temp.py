import os
import gc
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# from collections import Counter


data_path = './random_split/'
#print('Available data', os.listdir(data_path))

def read_data(partition):
    data = []
    for file in os.listdir(os.path.join(data_path, partition)):
        with open(os.path.join(data_path, partition, file)) as f:
            data.append(pd.read_csv(f, index_col=None))
    return pd.concat(data)

df_train = read_data('train')
df_test = read_data('test')
df_val = read_data('val')

# df_train.info()
# print(df_train.head())
#print(df_train)
#print(df_train.head(1)['sequence'].values[0])

# print('Train size: ', len(df_train))
# print('Val size: ', len(df_val))
# print('Test size: ', len(df_test))

def calc_unique_class(train, test, val):
    train_unq = np.unique(train['family_accession'].values)
    val_unq = np.unique(val['family_accession'].values)
    test_unq = np.unique(test['family_accession'].values)
    
    print('Number of unique classes in Train: ', len(train_unq))
    print('Number of unique classes in Val: ', len(val_unq))
    print('Number of unique classes in Test: ', len(test_unq))
    
# calc_unique_class(df_train, df_val, df_test)

df_train['seq_char_cnt'] = df_train['sequence'].apply(lambda x: len(x))
df_val['seq_char_cnt'] = df_val['sequence'].apply(lambda x: len(x))
df_test['seq_char_cnt'] = df_test['sequence'].apply(lambda x: len(x))

def plot_seq_count(df, data_name):
    sns.distplot(df['seq_char_cnt'].values)
    plt.title(f'Sequence char count: {data_name}')
    plt.grid(True)
    
# plt.subplot(1,3,1)
# plot_seq_count(df_train, 'Train')

# plt.subplot(1,3,2)
# plot_seq_count(df_val, 'Val')

# plt.subplot(1,3,3)
# plot_seq_count(df_test, 'Test')

# =============================================================================
# def get_code_freq(df, data_name):
#     df = df.apply(lambda x: " ".join(x))
#     codes = []
#     for i in df:
#         codes.extend(i)
#     codes_dict = Counter(codes)
#     codes_dict.pop(' ')
#     
#     print(f'Codes: {data_name}')
#     print(f'Total unique codes: {len(codes_dict.keys())}')
#     
#     df = pd.DataFrame({'Code': list(codes_dict.keys()), 'Freq': list(codes_dict.values())})
#     return df.sort_values('Freq', ascending=False).reset_index()[['Code','Freq']]
# 
# test_code_freq = get_code_freq(df_test['sequence'], 'Test')
# print(test_code_freq())
# =============================================================================

# print(df_train.groupby('family_id').size().sort_values(ascending=False).head(20))
# print(df_val.groupby('family_id').size().sort_values(ascending=False).head(20))
# print(df_test.groupby('family_id').size().sort_values(ascending=False).head(20))
# print(df_test.groupby('family_accession').size().sort_values(ascending=False).head(20))


classes = df_train['family_accession'].value_counts()[:250].index.tolist()  #taking only 250 top classes

train_sm = df_train.loc[df_train['family_accession'].isin(classes)].reset_index();
val_sm = df_val.loc[df_val['family_accession'].isin(classes)].reset_index();
test_sm = df_test.loc[df_test['family_accession'].isin(classes)].reset_index();

# calc_unique_class(train_sm, val_sm, test_sm)

codes = ['A','C','D','E','F','G','H','I','K','L',
         'M','N','P','Q','R','S','T','V','W','Y']

def create_dict(codes):
    char_dict = {}
    for index, val in enumerate(codes):
        char_dict[val] = index+1
        
    return char_dict

char_dict = create_dict(codes)
# print(char_dict)

def integer_encoding(data):
    encode_list = []
    for row in data['sequence'].values:
        row_encode = []
        for code in row:
            row_encode.append(char_dict.get(code, 0))
        encode_list.append(np.array(row_encode))
        
    return encode_list

train_encode = integer_encoding(train_sm)
val_encode = integer_encoding(val_sm)
test_encode = integer_encoding(test_sm)

# print(val_encode)

max_length = 50 # max length of sequences

train_pad = pad_sequences(train_encode, maxlen = max_length, padding='post', truncating='post')
val_pad = pad_sequences(val_encode, maxlen = max_length, padding='post', truncating='post')
test_pad = pad_sequences(test_encode, maxlen = max_length, padding='post', truncating='post')

# print(val_pad)
train_ohe = train_pad
val_ohe = val_pad
test_ohe = val_pad

# train_ohe = to_categorical(train_pad) #(184460, 50, 21) (22917, 50, 21) (22917, 50, 21)
# val_ohe = to_categorical(val_pad)
# test_ohe = to_categorical(test_pad)
# print(train_ohe.shape, val_ohe.shape, test_ohe.shape)
# print(val_ohe)

#del train_pad, val_pad, test_pad
#del train_encode, val_encode, test_encode
gc.collect()

le = LabelEncoder()

y_train_le = le.fit_transform(train_sm['family_accession']) #(184460,) (22917,) (22917,)
y_val_le = le.fit_transform(val_sm['family_accession'])
y_test_le = le.fit_transform(test_sm['family_accession'])
# print(y_train_le.shape, y_val_le.shape, y_test_le.shape)

y_train = to_categorical(y_train_le) #(184460, 250) (22917, 250) (22917, 250)
y_val = to_categorical(y_val_le)
y_test = to_categorical(y_test_le)
# print(y_train.shape, y_val.shape, y_test.shape)

del y_train_le, y_val_le, y_test_le
gc.collect()

n_nodes_hl1 = 100
n_nodes_hl2 = 100
n_nodes_hl3 = 100
n_classes = 250
loss1 = []
accuracy1 = []

x = tf.placeholder(tf.float32,[None,50])
y = tf.placeholder(tf.float32)

def neural_network_model(data):
    hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([50, n_nodes_hl1])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases' : tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.tanh(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.tanh(l2)
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.tanh(l3)
    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    output = tf.nn.tanh(output)
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.015).minimize(cost)
    hm_epochs = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for j in range(train_ohe.shape[0]):
                _,c = sess.run([optimizer,cost],feed_dict={x:train_ohe[j,:], y:y_train[j,:]})
                epoch_loss += c
            print('Epoch', epoch,'completed out of ',hm_epochs, 'loss ', epoch_loss)
            loss1.append(epoch_loss)
    pred = sess.run(prediction, feed_dict={x:test_ohe})
    correct = tf.equal(tf.argmax(pred,1),tf.argmax(y_test,1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    accu = float(accuracy)
    print('Accuracy: ', accu)
    accuracy1.append(accu)
    #plt.plot(loss1)
    
if __name__ == '__main__':
    for q in range(10):
        print('Run :', q+1)
        loss1 = []
        train_neural_network(x)
    plt.xlabel('Runs', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.plot(accuracy1)
    Sum = sum(accuracy1)
    print('Average accuarcy :', Sum/10)
    

















