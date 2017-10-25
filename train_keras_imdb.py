import numpy as np
import time
import preprocessing
import argparse
import gensim
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

MAX_NB_WORDS = 100000  # only consider 10000 most commonly occuring words

WORD2VEC_FILE = '/hdd/data/GoogleNews-vectors-negative300.bin'

BATCH_SIZE = 64

NUM_EPOCHS = 10

EMBEDDING_DIM = 50

DROPOUT_RATE = 0.5

NUM_FILTERS = 128

FILTER_SIZES = "3, 4, 5"

PRE_TRAINED = False


def load_data(positive_data_file, negative_data_file):
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [preprocessing.clean_str(sent) for sent in x_text]
    # x_text = [sent.split(' ') for sent in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    return [x_text, y]


def create_word2vec_embedding(word_index, embedding_dim):
    miss_count = 0
    embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))

    print("loading wor2vec vectors...")
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_FILE, binary=True)
    for word, i in word_index.items():
        try:
            embedding_vector = word2vec_model[word]
        except KeyError:
            embedding_vector = None
            miss_count += 1

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            # randomly initialized unmatch words
            embedding_matrix[i] = np.random.rand(embedding_dim)
            pass

    print('(miss count / words) %d/%d=%f' %
          (miss_count, len(word_index), miss_count * 1.0 / len(word_index)))

    return embedding_matrix

def create_random_embedding(word_index, embedding_dim):
    embedding_matrix = np.random.rand(len(word_index)+1, embedding_dim)
    return embedding_matrix

"""
  pre_trained: True/False
  word_index: Vocabulary, a python dictionary object (e.g. key: word, value: word_id)
  embedding_dim: Embedding dimension
"""
def create_embedding_layer(pre_trained, word_index, embedding_dim, max_sequence_length):
    if pre_trained:
        embedding_matrix = create_word2vec_embedding(word_index, embedding_dim)
        embedding_layer = Embedding(input_dim=len(word_index)+1,
                                    output_dim=embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=max_sequence_length,
                                    trainable=True)     # non-static setting

    else:
        # embedding_matrix = create_random_embedding(word_index, embedding_dim)
        embedding_layer = Embedding(input_dim=len(word_index)+1,
                                    output_dim=embedding_dim,
                                    input_length=max_sequence_length)

    return embedding_layer


def create_model_topology(embedding_layer, embedding_dim, filter_sizes, num_filters, max_sequence_length, num_labels=2):
    sequence_input = Input(shape=(max_sequence_length, ), dtype='int32')
    embedding_sequences = embedding_layer(sequence_input)

    conv_blocks = []
    for filter_size in filter_sizes:
        conv = Conv1D(filters=num_filters,
                      kernel_size=filter_size,
                      padding='valid',
                      activation='relu')(embedding_sequences)

        conv = MaxPool1D(pool_size=max_sequence_length - filter_size + 1)(conv)
        conv_blocks.append(conv)
    # end of convolution layers

    conv_merge = concatenate(conv_blocks, axis=1) if len(conv_blocks) > 1 else conv_blocks[0]
    flatten = Flatten()(conv_merge)
    dropout = Dropout(args.dropout)(flatten)
    preds = Dense(num_labels, activation='softmax')(dropout)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    print(model.summary())

    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CNN model for text classification')
    parser.add_argument('--mode', dest='mode', action='store', required=False, default='train')
    parser.add_argument('--num_filters', dest='num_filters', action='store', required=False, default=NUM_FILTERS)
    parser.add_argument('--filter_sizes', dest='filter_sizes', action='store', required=False, default=FILTER_SIZES)
    parser.add_argument('--batch_size', dest='batch_size', action='store', required=False, default=BATCH_SIZE)
    parser.add_argument('--num_epochs', dest='num_epochs', action='store', required=False, default=NUM_EPOCHS)
    parser.add_argument('--embedding_dim', dest='embedding_dim', action='store', required=False, default=EMBEDDING_DIM)
    parser.add_argument('--dropout', dest='dropout', action='store', required=False, default=DROPOUT_RATE)
    parser.add_argument('--pre_trained', dest='pre_trained', action='store', required=False, default=PRE_TRAINED)
    parser.add_argument('--save_model', dest='save_model', action='store', required=False, default=True)
    parser.add_argument('--model_file', dest='model_file', action='store', required=False, default=None)

    args = parser.parse_args()

    assert args.mode == 'train' or args.mode == 'test'
    if args.mode == 'test':
        assert args.model_file is not None
    filter_sizes = [int(size) for size in args.filter_sizes.split(",")]
    assert len(filter_sizes) > 0

    # Load data
    sentences, labels = load_data("data/imdb/train/imdb_pos.txt",
                                  "data/imdb/train/imdb_neg.txt")
    # sentences, labels = load_data("data/rt-polaritydata/rt-polarity.pos",
    #                               "data/rt-polaritydata/rt-polarity.neg")

    # tokenize
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters="")
    tokenizer.fit_on_texts(sentences)
    print("Vocabulary size: %d" % len(tokenizer.word_index))

    # max_sequence_length = max([len(sent.split(" ")) for sent in sentences])
    # print("max sequence length: %d" % max_sequence_length)

    sequences = tokenizer.texts_to_sequences(sentences)
    data = pad_sequences(sequences, maxlen=None, padding="post", truncating='post')  # Unlimited
    print("data shape: %s" % str(data.shape))

    max_sequence_length = data.shape[1]
    num_labels = labels.shape[1]

    x = data
    y = labels

    # define embedding layer
    embedding_layer = create_embedding_layer(pre_trained=args.pre_trained,
                                             word_index=tokenizer.word_index,
                                             embedding_dim=args.embedding_dim,
                                             max_sequence_length=max_sequence_length)

    model = create_model_topology(embedding_layer=embedding_layer,
                                  embedding_dim=args.embedding_dim,
                                  filter_sizes=filter_sizes,
                                  num_filters=args.num_filters,
                                  max_sequence_length=max_sequence_length,
                                  num_labels=num_labels)

    # train the model or test
    if args.mode == 'train':
        # train
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=10)

        early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')
        model.fit(x_train, y_train,
                  batch_size=args.batch_size,
                  epochs=args.num_epochs,
                  verbose=1,
                  validation_data=(x_val, y_val),
                  callbacks=[early_stopping])
        loss, acc = model.evaluate(x_val, y_val)

        print('Validation set loss:', loss)
        print('Validation set accuracy:', acc)

        if args.save_model:
            timestamp = str(int(time.time()))
            model.save_weights('models/keras_%s.h5' % timestamp)

    elif args.mode == 'test':
        # evaluation
        # Load data
        sentences, labels = load_data("data/imdb/test/imdb_pos.txt",
                                      "data/imdb/test/imdb_neg.txt")

        sequences = tokenizer.texts_to_sequences(sentences)
        x_test = pad_sequences(sequences, maxlen=None, padding="post", truncating="post")  # Unlimited

        model.load_weights(args.model_file)
        # model.predict()

        pass
