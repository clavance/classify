import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed
from keras import backend as K
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# this implementation is very largely based on:
# https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py
# https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2

# for loading data
def load_data(file_location):
    return pd.read_csv(file_location, header='infer', encoding='latin1')


# for plotting training loss, val loss
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    # plotting function
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Number of epochs')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Number of epochs')
    plt.legend()

    plt.show()


# define function to plot confusion matrix
def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion Matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Reds')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                if (cm[i, j] == 0):
                    formatting = 0
                else:
                    formatting = "{:0.3f}".format(cm[i, j])

                plt.text(j, i, formatting,
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.05)
    plt.gcf().subplots_adjust(bottom=0.05)
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# dot product operation
def dot_product(x, kernel):
    return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)

# define a HAN layer
class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def run_model(df, MAX_WORDS, MAX_SENTS, MAX_SENT_LENGTH, VALIDATION_SPLIT, EMBEDDING_DIM):
    x = df['Text']
    y = df['Class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1000, stratify=y)

    # split data into sentences, using spacy sentencizer
    # since we are only using spacy for sentencizing, we only invoke the sentencizer
    nlp = spacy.blank("en")
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    x_train_sentences = pd.Series([doc for doc in nlp.pipe(x_train)]).apply(lambda x: [sent for sent in x.sents])

    # oov_token=True reserves a token for unknown words (rather than ignoring the word)
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token=True)
    tokenizer.fit_on_texts(x_train.values)

    data = np.zeros((len(x_train_sentences), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

    # i is each sample index number
    # samples is each sample
    for i, samples in enumerate(x_train_sentences):
        # j is the sentence index number
        # sentences is each sentence
        for j, sentences in enumerate(samples):
            if j < MAX_SENTS:
                # wordTokens is list of tokens
                wordTokens = text_to_word_sequence(str(sentences))
                k = 0
                # word is each individual token
                for _, word in enumerate(wordTokens):
                    if k < MAX_SENT_LENGTH:
                        if word not in tokenizer.word_index:  # remove special characters
                            continue
                        if tokenizer.word_index[word] < MAX_WORDS:
                            data[i, j, k] = tokenizer.word_index[word]
                            k += 1

    print('Total %s unique tokens.' % len(tokenizer.word_index))

    labels = pd.get_dummies(y_train.values)

    print('Shape of samples tensor:', data.shape)
    print('Shape of labels tensor:', labels.shape)

    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    print('Number of samples of each class in training and validation set')
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))

    embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SENT_LENGTH,
                                trainable=True)

    word_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    word_sequences = embedding_layer(word_input)
    word_lstm = Bidirectional(LSTM(100, return_sequences=True))(word_sequences)
    word_dense = TimeDistributed(Dense(200))(word_lstm)
    word_att = AttentionWithContext()(word_dense)
    wordEncoder = Model(word_input, word_att)

    sent_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    sent_encoder = TimeDistributed(wordEncoder)(sent_input)
    sent_lstm = Bidirectional(LSTM(100, return_sequences=True))(sent_encoder)
    sent_dense = TimeDistributed(Dense(200))(sent_lstm)
    sent_att = AttentionWithContext()(sent_dense)
    predictions = Dense(20, activation='softmax')(sent_att)
    model = Model(sent_input, predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    print(model.summary())

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=50, batch_size=50, callbacks=[es])

    # vectorize test set the same way
    x_test_sentences = pd.Series([doc for doc in nlp.pipe(x_test)]).apply(lambda x: [sent for sent in x.sents])

    x_test_sentences = []
    for doc in nlp.pipe(x_test):
        x_test_sentences.extend(sent.text for sent in doc.sents)

    test_data = np.zeros((len(x_test_sentences), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    test_labels = pd.get_dummies(y_test.values)

    print('Shape of test samples tensor:', test_data.shape)
    print('Shape of test labels tensor:', test_labels.shape)
    print('Number of samples of each class in test set: ', test_labels.sum(axis=0))

    for i, samples in enumerate(x_test_sentences):
        # j is the sentence number
        # sentences is each sentence
        for j, sentences in enumerate(samples):
            if j < MAX_SENTS:
                # wordTokens is list of tokens
                wordTokens = text_to_word_sequence(str(sentences))
                k = 0
                # word is each individual token
                for _, word in enumerate(wordTokens):
                    if k < MAX_SENT_LENGTH:
                        if word not in tokenizer.word_index:  # remove special characters
                            continue
                        if tokenizer.word_index[word] < MAX_WORDS:
                            test_data[i, j, k] = tokenizer.word_index[word]
                            k = k + 1

    x_test = test_data
    y_test_single = y_test # need this for classification report f1
    y_test = test_labels

    loss, accuracy = model.evaluate(x_train, y_train, verbose=2)
    print("Training Accuracy: {:.4f}".format(accuracy))

    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plot_history(history)

    y_pred = np.argmax(model.predict(x_test), axis=1) + 1
    cm = confusion_matrix(y_test_single, y_pred, labels=None, sample_weight=None)
    print("\nClassification report summary:")
    print(classification_report(y_test_single, y_pred, labels=[i + 1 for i in range(20)], digits=3))

    return model, cm

# ------------------------------------------ SAMPLE CODE ------------------------------------------ #
df = load_data('/.../data.csv')

MAX_WORDS = 261737  # number of unique tokens in our training set
MAX_SENTS = 300 # we stick to max. 300 sentences per sample
MAX_SENT_LENGTH = 300 # we stick to max. 300 tokens per sentence
VALIDATION_SPLIT = 0.2 # same training:validation split, 80:20
EMBEDDING_DIM = 300  # we stick to embedding dimensions of 300

model, cm = run_model(df, MAX_WORDS, MAX_SENTS, MAX_SENT_LENGTH, VALIDATION_SPLIT, EMBEDDING_DIM)

plot_confusion_matrix(cm, normalize=False, target_names=[i for i in range(1,21)], title='Confusion Matrix')
