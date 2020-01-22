import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE


# cleanup functions and plot_common_tokens are largely based on:
# https://www.kaggle.com/enerrio/scary-nlp-with-spacy-and-keras

def cleanup_text_token(docs):
    texts = []
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    stop_words.add('//')
    stop_words.add('°')
    stop_words.add('â\x80\x94')

    for doc in docs:
        nlp.max_length = len(doc)
        doc = nlp(doc, disable=['parser', 'ner'])
        # lowercase if not pronoun
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stop_words and tok not in string.punctuation]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


# when we clean up text to pass to the sentencizer, we do not remove punctuation
def cleanup_text_sentence(docs):
    texts = []
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    stop_words.add('//')
    stop_words.add('°')
    stop_words.add('â\x80\x94')

    for doc in docs:
        nlp.max_length = len(doc)
        doc = nlp(doc, disable=['parser', 'ner'])
        # lowercase if not pronoun
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stop_words]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


# we can generate plots of the 25 most commonly occurring tokens in each class, with their counts
def plot_common_tokens(tok_clean, i):

    counter = Counter(tok_clean)
    tokens = [j[0] for j in counter.most_common(25)]
    counts = [j[1] for j in counter.most_common(25)]

    plt.figure(figsize=(25, 12))
    fig = sns.barplot(x=tokens, y=counts)
    title = '25 most commonly occurring tokens in Class ' + str(i)

    if i == 1:
        title += ' (General, financial and institutional matters)'

    elif i == 2:
        title += ' (Customs Union and free movement of goods)'

    elif i == 3:
        title += ' (Agriculture)'

    elif i == 4:
        title += ' (Fisheries)'

    elif i == 5:
        title += ' (Freedom of movement for workers and social policy)'

    elif i == 6:
        title += ' (Right of establishment and freedom to provide services)'

    elif i == 7:
        title += ' (Transport policy)'

    elif i == 8:
        title += ' (Competition policy)'

    elif i == 9:
        title += ' (Taxation)'

    elif i == 10:
        title += ' (Economic and monetary policy and free movement of capital)'

    elif i == 11:
        title += ' (External relations)'

    elif i == 12:
        title += ' (Energy)'

    elif i == 13:
        title += ' (Industrial policy and internal market)'

    elif i == 14:
        title += ' (Regional policy and coordination of structural instruments)'

    elif i == 15:
        title += ' (Environment, consumers and health protection)'

    elif i == 16:
        title += ' (Science, information, education and culture)'

    elif i == 17:
        title += ' (Law relating to undertakings)'

    elif i == 18:
        title += ' (Common Foreign and Security Policy)'

    elif i == 19:
        title += ' (Area of freedom, security and justice)'

    else:
        title += ' (People\'s Europe)'

    fig.set(xlabel='Tokens', ylabel='Number of occurrences', title=title)
    plt.show()


def load_data(file_location):
    return pd.read_csv(file_location, header='infer', encoding='latin1')


def visualise_data(df, v, t):
    x = df['Cleaned'].values
    y = df['Class'].values

    if t == "full":
        if v == "count":
            vectorizer = CountVectorizer()

        if v == "tfidf":
            vectorizer = TfidfVectorizer()

        X = vectorizer.fit_transform(x)

        # first reduce dimensionality to 50-D with LSA
        svd = TruncatedSVD(n_components=50, n_iter=5, random_state=1000)
        X = svd.fit_transform(X)
        print(np.shape(X)) # shape should be (16169, 50), i.e. 16169 samples in 50D

        # further reduce to 2-D with TSNE
        # this 2-step reduction is recommended in Scikit-learn documentation
        tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
        X_tsne = tsne.fit_transform(X)
        print(np.shape(X_tsne)) # shape should be (16169, 2), i.e. 16169 samples in 2D

        df_tsne = pd.DataFrame()
        df_tsne["x-dimension"] = X_tsne[:,0]
        df_tsne["y-dimension"] = X_tsne[:,1]

        plt.figure(figsize=(16,10))

        sns.scatterplot(
            x="x-dimension", y="y-dimension",
            hue=y,
            palette=sns.color_palette("bright", 20),
            data=df_tsne,
            legend="full",
            alpha=0.3
        )

        plt.title("Visualisation of "+ t +" dataset set with " + v + " vectorizer")
        plt.savefig("dataset_visualisation_" + t + "_" + v + ".pdf")
        plt.show()

    if t == "test":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1000, stratify=y)

        if v == "count":
            vectorizer = CountVectorizer()

        if v == "tfidf":
            vectorizer = TfidfVectorizer()

        vectorizer.fit(x_train)
        X_test = vectorizer.transform(x_test)

        # first reduce dimensionality to 50-D with LSA
        svd = TruncatedSVD(n_components=50, n_iter=5, random_state=1000)
        X = svd.fit_transform(X_test)
        print("data shape after LSA:", np.shape(X))  # shape should be (3234, 50), i.e. 3234 samples in 50D

        # further reduce to 2-D with TSNE
        # this 2-step reduction is recommended in Scikit-learn documentation
        tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
        X_tsne = tsne.fit_transform(X)
        print("data shape after tSNE:", np.shape(X_tsne))  # shape should be (3234, 2), i.e. 3234 samples in 2D

        df_tsne = pd.DataFrame()
        df_tsne["x-dimension"] = X_tsne[:, 0]
        df_tsne["y-dimension"] = X_tsne[:, 1]

        plt.figure(figsize=(32, 20))

        sns.scatterplot(
            x="x-dimension", y="y-dimension",
            hue=y_test,
            palette=sns.color_palette("bright", 20),
            data=df_tsne,
            legend="full",
            alpha=0.3
        )

        plt.title("Visualisation of " + t + " dataset set with " + v + " vectorizer")
        plt.savefig("dataset_visualisation_" + t + "_" + v + ".pdf")
        plt.show()


def visualise_embeddings():

    # plot visualisation of spaCy word embeddings
    tokens = ['fishing', 'fishery', 'fisheries', 'aquaculture', 'sea',
              'nuclear', 'energy', 'environmental', 'pollution', 'residue',
              'financial', 'economic', 'monetary', 'deficit', 'investment',
              'spain', 'poland', 'czech', 'estonia', 'europe',
              'shipping', 'transport', 'drive', 'carriage', 'road']

    vectors = [nlp(tok).vector for tok in tokens]

    tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=3)

    vectors = tsne.fit_transform(vectors)

    plt.figure(figsize=(12, 8))

    for v in vectors[0:5]:
        plt.scatter(v[0], v[1], c='r')
    for v in vectors[5:10]:
        plt.scatter(v[0], v[1], c='b')
    for v in vectors[10:15]:
        plt.scatter(v[0], v[1], c='g')
    for v in vectors[15:20]:
        plt.scatter(v[0], v[1], c='k')
    for v in vectors[20:25]:
        plt.scatter(v[0], v[1], c='m')

    texts = []

    for i, j in zip(tokens, vectors):
        x, y = j
        # plt.text(x, y, i, size=10)
        texts.append(plt.text(x, y, i, size=10))

    plt.title('Visualisation of spaCy word embeddings with TSNE dimensionality reduction of 25 common words in dataset')
    adjust_text(texts)
    plt.savefig("spacyplot.pdf")


def class_distributions(df):
    # convert classes from string to int (to facilitate counting)
    df['Class'] = df['Class'].astype(int)

    # prints counts of samples in each class
    print("Class Count")
    print(df['Class'].value_counts().sort_index())

    # plots bar chart of the distribution of classes
    x_list = [str(i) for i in range(1,21)]
    fig = sns.barplot(x=x_list, y=df['Class'].value_counts().sort_index(), order=x_list)
    fig.set(xlabel='Classes', ylabel='Number of samples', title='Distribution of samples across classes')
    plt.show()

    text = ' '.join([text for text in df['Text']])

    # Average tokens per sample
    # Average sentences per sample
    # Average tokens per sentence
    # Plot 25 most common tokens in each class

    for i in range(1,21):
        class_text = [text for text in df[df['Class'] == i]['Text']]
        tokens_clean = cleanup_text_token(class_text)
        tokens_clean = ' '.join(tokens_clean).split()
        tokencount = len(tokens_clean)
        print("Average tokens per sample in Class " + str(i) + ": ", tokencount/len(class_text))

        # first get lowercased and cleaned tokens
        sent_clean = cleanup_text_sentence(class_text)
        # join tokens back into sentences
        sent_clean = ' '.join(sent_clean)

        sentcount = 0

        # loading the entire string takes too much memory, so it's divided into 10 batches to be processed
        ind_array = [0]
        mult = len(sent_clean)/10
        for j in range(1,11):
            ind_array.append(int(mult*j))

        for k in range(10):
            sent_part = sent_clean[ind_array[k]:ind_array[k+1]]
            nlp.max_length = len(sent_part)
            doc = nlp(sent_part, disable=['ner'])
            # split and count sentences with sentencizer
            for sent in doc.sents:
                sentcount += 1

        print("Average sentences per sample in Class " + str(i) + ": ", sentcount / len(class_text))
        print("Average tokens per sentence in Class " + str(i) + ": ", tokencount / sentcount)

        plot_common_tokens(tokens_clean, i)


# ------------------------------------------ SAMPLE CODE ------------------------------------------ #
df = load_data('/.../data.csv')
nlp = spacy.load('en_core_web_lg')
visualise_data(df, "count", "full")
