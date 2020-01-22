import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Input, Model
from keras.layers import Bidirectional, concatenate, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dense, Dropout, Flatten, LSTM, Masking
from sklearn.metrics import classification_report, confusion_matrix

# cleanup, word vectorizer and plot functions are based on https://www.kaggle.com/enerrio/scary-nlp-with-spacy-and-keras

# function for preprocessing
def cleanup_text(docs, embedding, logging=False):
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    stop_words.add('//')
    stop_words.add('°')
    stop_words.add('â\x80\x94')

    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Cleaned %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        # lowercase if not pronoun
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']

        if embedding == "word_embeddings":
            # remove stopwords, punctuation
            tokens = [tok for tok in tokens if tok not in stop_words and tok not in string.punctuation]

        elif embedding == "sentence_embeddings":
            # remove stopwords, but don't remove punctuation (need it to split sentences)
            tokens = [tok for tok in tokens if tok not in stop_words]

        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


# define function to shorten text to maximum of MAX_LENGTH words (to limit size of word embeddings)
def shorten_text_word(docs, MAX_LENGTH, logging=False):
    texts = []
    counter = 1

    for doc in docs:

        if counter % 1000 == 0 and logging:
            print("Shortened %d out of %d documents." % (counter, len(docs)))

        counter += 1

        words = []
        wordcount = 0

        for word in doc.split():
            wordcount += 1

            if wordcount <= MAX_LENGTH:
                words.append(word)

        words = ' '.join(words)
        texts.append(words)

    return pd.Series(texts)


def shorten_text_sent(docs, MAX_LENGTH, logging=False):
    texts = []
    counter = 1

    for sample in docs:

        if counter % 1000 == 0 and logging:
            print("Shortened %d out of %d documents." % (counter, len(docs)))

        counter += 1

        doc = nlp(sample)

        sentences = []
        sentcount = 0

        for sent in doc.sents:
            sentcount += 1

            if sentcount <= MAX_LENGTH:
                sentences.append(str(sent))

        sentences = ' '.join(sentences)
        texts.append(sentences)

    return pd.Series(texts)


# define function to obtain word embeddings
def vectorize_text_word(docs, logging=False):
    counter = 1
    all_vectors = []

    for doc in docs:
        if counter % 100 == 0 and logging:
            print("Vectorized %d out of %d documents." % (counter, len(docs)))

        counter += 1
        vectors = []

        for word in doc.split():
            if nlp(word).has_vector:
                vectors.append(nlp(word).vector)

            else:
                vectors.append(np.zeros((300,), dtype="float32"))

        all_vectors.append(vectors)

    return pd.Series(all_vectors)


# define function to obtain sentence embeddings
def vectorize_text_sent(docs, MAX_LENGTH, logging=False):
    counter = 1
    all_vectors = []

    for sample in docs:
        if counter % 100 == 0 and logging:
            print("Vectorized %d out of %d documents." % (counter, len(docs)))

        doc = nlp(sample)
        counter += 1
        vectors = []

        for sent in doc.sents:
            # remove punctuation from each sentence before vectorizing, because this step was skipped in cleanup
            tokens = [str(tok) for tok in sent if str(tok) not in string.punctuation]
            tokens = " ".join(tokens)

            if len(vectors) < MAX_LENGTH:
                vectors.append(nlp(tokens).vector)

        all_vectors.append(vectors)

    return pd.Series(all_vectors)


# define function to load data
def load_data(file_location):
    return pd.read_csv(file_location, header='infer', encoding='latin1')


# define function to plot training and validation loss
def plot_history(history, architecture, NB_EPOCHS, early_stopping):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    title = ""
    if architecture == "mlp":
        title += "Multilayer perceptron, "
    elif architecture == "cnn":
        title += "CNN, "
    elif architecture == "ngram_cnn":
        title += "CNN (Kim, 2014), "
    elif architecture == "lstm":
        title += "LSTM, "
    elif architecture == "lstm":
        title += "Bi-LSTM, "
    title += str(NB_EPOCHS) + " epochs"

    if early_stopping == True:
        title += " with early stopping"
    title += "\n"
    plt.rcParams["axes.titlesize"] = 10
    plt.suptitle(title)
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


# define general function for embeddings
def get_embeddings(df, embedding, MAX_LENGTH):
    # add a new column to the dataframe, with the cleaned up texts

    if embedding == "word_embeddings":
        # add a new column to the dataframe, with the shortened and cleaned up texts
        # texts are shortened to a maximum of 300 words
        df['Shortened'] = shorten_text_word(df['Cleaned'], MAX_LENGTH, logging=True)
        # add a new column to the dataframe, with the word embedding vectors
        # note: this takes several hours to run and uses a lot of memory (especially depending on max number of words chosen)
        # once run, df['Vectors'] has the vectors of each text
        # spacy returns each embedding as a vector of size 300
        # eg. if word embeddings are selected and length=300:
        # this results in dimension (300, 300) per sample; 300 numbers per word, and 300 words max, but possibly fewer words
        df['Vectors'] = vectorize_text_word(df["Shortened"], logging=True)

    elif embedding == "sentence_embeddings":
        df['Cleaned_sent'] = cleanup_text(df['Text'], embedding, logging=True)
        df['Shortened'] = shorten_text_sent(df['Cleaned_sent'], MAX_LENGTH, logging=True)
        df['Vectors'] = vectorize_text_sent(df['Shortened_sent'], MAX_LENGTH, logging=True)

    else:
        print("Select \"word_embeddings\" or \"sentence_embeddings\"!")

    return df


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


# define general function for models
def run_model(df, architecture, MAX_LENGTH, NB_EPOCHS, EARLY_STOPPING=False, PATIENCE=0):

    y = df['Class'].values

    x = []
    for i in range(df.shape[0]):
        x.append(df['Vectors'].values[i][:MAX_LENGTH])

    # post-pad all texts shorter than 300 words to make all samples of dimensions (300, 300)
    x = pad_sequences(x, padding='post')

    # split into training and test sets, 80:20
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1000, stratify=y)

    # we need the non-one hot classes for our classification report
    y_test_single = y_test

    # we need to one hot encoded classes to pass into our network
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    VAL_SET = math.floor((x_train.shape[0])*0.2) # Val Set = Training Set * 0.2 = 2587

    if architecture == "mlp":
        model = Sequential()
        model.add(Flatten(input_shape=(MAX_LENGTH, 300)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(20, activation='softmax'))

    elif architecture == "cnn":
        inputs = Input(shape=(MAX_LENGTH, 300))
        # masking = Masking(mask_value=0)(inputs) # masking in Keras not supported with Conv1D
        m = Conv1D(256, 3, strides=1, padding='same', activation='relu')(inputs)
        m = MaxPooling1D(pool_size=2)(m)
        m = Conv1D(512, 3, strides=1, padding='same', activation='relu')(m)
        m = GlobalMaxPooling1D()(m)
        outputs = Dense(20, activation='softmax')(m)
        model = Model(inputs=inputs, outputs=outputs, name='CNN')

    elif architecture == "ngram_cnn":
        inputs = Input(shape=(MAX_LENGTH, 300))

        convolutions = []
        filters = [3, 4, 5]

        for i in filters:
            m = Conv1D(filters=100, kernel_size=i, strides=1, activation='relu')(inputs)
            m = MaxPooling1D(pool_size=3)(m)
            convolutions.append(m)

        conv = concatenate(convolutions, axis=1)
        conv = Dropout(0.5)(conv)
        conv = Flatten()(conv)
        conv = Dense(256, activation='relu')(conv)
        conv = Dropout(0.5)(conv)
        conv = Dense(20, activation='softmax')(conv)
        model = Model(inputs, conv)

    elif architecture == "lstm":
        inputs = Input(shape=(MAX_LENGTH, 300,), dtype='float32')
        masking = Masking(mask_value=0)(inputs)
        lstm = LSTM(256)(masking)
        output = Dense(20, activation='softmax')(lstm)
        model = Model(inputs, output)

    elif architecture == "bi_lstm":
        inputs = Input(shape=(MAX_LENGTH, 300,), dtype='float32')
        masking = Masking(mask_value=0)(inputs)
        lstm = Bidirectional(LSTM(256))(masking)
        output = Dense(20, activation='softmax')(lstm)
        model = Model(inputs, output)


    else:
        print("Choose from one of the architectures: \"mlp\", \"cnn\", \"ngram_cnn\", \"lstm\", \"bi_lstm\"")

    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if not EARLY_STOPPING:
        history = model.fit(x_train[:-VAL_SET], y_train[:-VAL_SET],
                            epochs=NB_EPOCHS,
                            verbose=1,
                            validation_data=(x_train[-VAL_SET:], y_train[-VAL_SET:]),
                            batch_size=50)

    else: # using early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)

        history = model.fit(x_train[:-VAL_SET], y_train[:-VAL_SET],
                            epochs=NB_EPOCHS,
                            verbose=1,
                            validation_data=(x_train[-VAL_SET:], y_train[-VAL_SET:]),
                            batch_size=50, callbacks=[es])


    loss, accuracy = model.evaluate(x_train, y_train, verbose=2)
    print("Training Accuracy: {:.4f}".format(accuracy))

    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    # argmax of softmax output returns 0-based indices, so +1 to get the actual class prediction
    # we need to compare this with the non-one hot encoded version of true class labels, hence y_test_single
    y_pred = np.argmax(model.predict(x_test), axis=1) + 1
    cm = confusion_matrix(y_test_single, y_pred, labels=None, sample_weight=None)

    print("\nClassification report summary:")
    print(classification_report(y_test_single, y_pred, labels=[i + 1 for i in range(20)], digits=3))

    # set matplotlib plot style for plotting chart of training and validation accuracy/loss
    plt.style.use('ggplot')
    plot_history(history, architecture, NB_EPOCHS, EARLY_STOPPING)

    return model, cm

# ------------------------------------------ SAMPLE CODE ------------------------------------------ #

# function load_data takes in string of file location, loads data into dataframe
df = load_data('/.../data.csv')

# load the largest spacy english model (must be downloaded in advance)
# other options: 'en_core_web_sm', 'en_core_web_md'
nlp = spacy.load('en_core_web_lg')

# by default, the spacy model takes in inputs of max length 1,000,000. our dataset has max text length 2,871,868
nlp.max_length = 2871868

# choose settings
MAX_LENGTH = 300
NB_EPOCHS = 50
PATIENCE = 5
EARLY_STOPPING = True

# we run the function to add embeddings from spacy to our dataframe, function takes in 3 parameters, returns our df
# 1 - df - the pandas dataframe containing our data
# 2 - embedding - the type of embedding (options: "word_embeddings", "sentence_embeddings")
# 3 - MAX_LENGTH - the max number of words per sample to be vectorized (if "word_embeddings" selected)
#     or max number of sentences per sample to be vectorized (if "sentence_embeddings" selected)
#     (options: an integer between 1 to 300,000, but due to memory requirements, ideally <=1,000)
#     note: setting length=1000 already loads ~75GB of embeddings in memory
df = get_embeddings(df, "word_embeddings", 300)


# we run the neural network model, function takes in 5 parameters
# 1 - df - the pandas dataframe containing our data
# 2 - architecture - the type of architecture (options: "mlp", "cnn", "ngram_cnn", "lstm", "bi_lstm")
#     "ngram_cnn" is based on the CNN implemented in (Kim, 2014)
# 3 - MAX_LENGTH - the max number of words per sample, or max number of sentences per sample (must be same chosen above)
# 4 - NB_EPOCHS - the number of epochs over which to train
# 5 - EARLY_STOPPING - whether or not model will use early stopping (options: True or False)
# 6 - PATIENCE - if using early stopping, set number of epochs for which validation loss does not decrease before
#     early stopping is triggered (i.e. patience) (options: any integer < NB_EPOCHS)
model, cm = run_model(df, "mlp", MAX_LENGTH, NB_EPOCHS, EARLY_STOPPING, PATIENCE)

# plot confusion matrix and normalised confusion matrix
plot_confusion_matrix(cm, normalize=False, target_names=[i for i in range(1,21)])
plot_confusion_matrix(cm, normalize=True, target_names=[i for i in range(1,21)])
