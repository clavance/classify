import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import spacy
import string
from mlxtend.evaluate import mcnemar_table, mcnemar
from scipy.stats import randint
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from spacy.lang.en.stop_words import STOP_WORDS


# --------------------------------- FUNCTION TO LOAD DATA --------------------------------- #
def load_data(file_location):
    return pd.read_csv(file_location, header='infer', encoding='latin1')


# --------------------------------- FUNCTION FOR PREPROCESSING --------------------------------- #
def cleanup_text_token(docs):
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    stop_words.add('//')
    stop_words.add('°')
    stop_words.add('â\x80\x94')
    nlp = spacy.load('en_core_web_lg')
    texts = []
    for doc in docs:
        nlp.max_length = len(doc)
        doc = nlp(doc, disable=['parser', 'ner'])
        # lowercase if not pronoun
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stop_words and tok not in string.punctuation]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


# --------------------------------- FUNCTION TO RUN MODEL --------------------------------- #

def run_model(df, vectorizer, classifier):
    # load data
    x = df['Cleaned'].values
    y = df['Class'].values

    # split dataset into training and test sets, with 80:20 split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1000, stratify=y)

    if vectorizer == "count":
        vectorizer = CountVectorizer()

    if vectorizer == "tfidf":
        vectorizer = TfidfVectorizer()

    vectorizer.fit(x_train)

    X_train = vectorizer.transform(x_train)
    X_test = vectorizer.transform(x_test)

    if classifier == "naive_bayes":
        classifier = MultinomialNB()

    if classifier == "decision_tree":
        classifier = DecisionTreeClassifier() # manual search tried, but default hyperparameters were best

    if classifier == "random_forest":
        clf = RandomForestClassifier() # default n_estimators=100

        # define random search space based on decision tree depth
        hyp = {"n_estimators": [50, 100, 150, 200], # number of trees in the forest
               "max_depth": [40, 50, None], # max depth of tree
               "max_features": [10, 20, 'sqrt', None],
               "min_samples_split": randint(1, 11),
               "bootstrap": [True, False], # to use bagging or not
               "criterion": ["gini", "entropy"]} # gini impurity or information gain

        # random search over 5-fold cross validation (stratified k-fold by default)
        random_search = RandomizedSearchCV(clf, hyp, random_state=1, n_iter=100, cv=5, verbose=1, n_jobs=-1)
        search_result = random_search.fit(X_train, y_train)

        n_estimators = search_result.best_estimator_.get_params()['n_estimators']
        max_depth = search_result.best_estimator_.get_params()['max_depth']
        max_features = search_result.best_estimator_.get_params()['max_features']
        min_samples_split = search_result.best_estimator_.get_params()['min_samples_split']
        bootstrap = search_result.best_estimator_.get_params()['bootstrap']
        criterion = search_result.best_estimator_.get_params()['criterion']

        print("Random search results: ")
        print("Best n_estimators: ", n_estimators)
        print("Best max_depth: ", max_depth)
        print("Best max_features:", max_features)
        print("Best max_features:", min_samples_split)
        print("Best bootstrap:", bootstrap)
        print("Best criterion:", criterion)

        # set the classifier to the one with best hyperparameters from random search
        classifier = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            max_features=max_features,
                                            min_samples_split=min_samples_split,
                                            bootstrap=bootstrap,
                                            criterion=criterion)

    if classifier == "logistic_regression":
        # by a manual search the lbfgs solver showed best results
        # number of max iterations is increased to allow lbfgs solver to converge
        # compare loss functions over 5-fold cross validation
        ovr_clf = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000)
        ovr_score = cross_val_score(ovr_clf, X_train, y_train, cv=5).mean()

        mce_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        mce_score = cross_val_score(mce_clf, X_train, y_train, cv=5).mean()

        # choose the better performing hyperparameters
        if (ovr_score > mce_score):
            classifier = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000)
        else:
            classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)


    if classifier == "linear_svm":
        clf = svm.LinearSVC(max_iter=1000)

        hyp = {"loss": ['hinge', 'squared_hinge'],
               "multi_class": ['ovr', 'crammer_singer']}

        random_search = RandomizedSearchCV(clf, hyp, random_state=1, n_iter=20, cv=5, verbose=1, n_jobs=-1)
        search_result = random_search.fit(X_train, y_train)

        loss = search_result.best_estimator_.get_params()['loss']
        multi_class = search_result.best_estimator_.get_params()['multi_class']

        print("Best loss: ", loss)
        print("Best multi_class:", multi_class)

        classifier = svm.LinearSVC(loss=loss, multi_class=multi_class, max_iter=1000)


    if classifier == "nonlinear_svm":
        clf = svm.SVC()
        hyp = {"gamma": ['auto', 'scale'],
               "kernel": ['poly', 'rbf', 'sigmoid']}

        random_search = RandomizedSearchCV(clf, hyp, random_state=1, n_iter=20, cv=5, verbose=1, n_jobs=-1)
        search_result = random_search.fit(X_train, y_train)

        gamma = search_result.best_estimator_.get_params()['gamma']
        kernel = search_result.best_estimator_.get_params()['kernel']

        print("Best gamma: ", gamma)
        print("Best kernel:", kernel)

        classifier = svm.SVC(gamma=gamma, kernel=kernel)

    if classifier == "knn":
        classifier = KNeighborsClassifier(n_neighbors=5)  # change k-value as needed

    if classifier == "mlp":
        clf = MLPClassifier()
        hyp = {"hidden_layer_sizes": [(64,), (64,64), (64,64,64), (128,), (128,128),
                                      (128.128,128), (256,256,256), (512,512,512)]}

        grid_search = GridSearchCV(clf, hyp, cv=5)
        search_result = grid_search.fit(X_train, y_train)

        hidden_layer_sizes = search_result.best_estimator_.get_params()['hidden_layer_sizes']

        print("Best hidden layer size:", hidden_layer_sizes)

        classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, verbose=True) # uses reLU, adam by default

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # print metrics
    print("\nClassification report summary:")
    print(classification_report(y_test, y_pred, labels=[i + 1 for i in range(20)], digits=3))

    print("Accuracy:", classifier.score(X_test, y_test))
    print("Macro-F1:", f1_score(y_test, y_pred, average='macro'))

    # if decision tree or random forest, generates plot of tree
    if classifier == "decision_tree" or classifier == "random_forest":

        # print 5 most important tokens:
        swapped_vocab = dict([(value, key) for key, value in vectorizer.vocabulary_.items()])
        print("5 most important tokens: ")
        for i in np.argsort(classifier.feature_importances_)[-5:][::-1]:
            print(swapped_vocab[i])

        from sklearn.externals.six import StringIO
        from sklearn.tree import export_graphviz
        import pydotplus

        dot_data = StringIO()

        if classifier == "decision_tree":
            export_graphviz(classifier, out_file=dot_data,
                            filled=True, rounded=True,
                            special_characters=True)
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

            graph.write_pdf("decision_tree.pdf")

        else:
            # get a random one of the 100 trees in the forest
            export_graphviz(classifier.estimators_[random.randint(1,101)], out_file=dot_data,
                            filled=True, rounded=True,
                            special_characters=True)
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

            graph.write_pdf("random_forest.pdf")

    # if logistic regression, plot most important terms
    if classifier == "logistic_regression":
        plot_lr_coef(classifier, vectorizer)

    # get confusion matrix for plot
    cm = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)

    return vectorizer, classifier, cm


# --------------------------------- PLOT CONFUSION MATRIX --------------------------------- #
# function to plot confusion matrix or normalised confusion matrix
# heavily based on: https://medium.com/@deepanshujindal.99/how-to-plot-wholesome-confusion-matrix-40134fd402a8

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
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclassified={:0.4f}'.format(accuracy, misclass))
    plt.show()



# --------------------------------- ANALYSIS OF LOGISTIC REGRESSION COEFFICIENTS --------------------------------- #
# find the most important words for each class using the logistic regression model coefficients
# classifier.coef_ returns the coefficients of the logistic regression model. shape: (20, 232483)
# this means that for each class, there are 232483 coefficients, one each for each word in the vocabulary, indicating
# how important each word in the vocabulary is for that class

def plot_lr_coef(classifier, vectorizer):
    # swap the key:value pairs in the vocabulary dictionary, to make accessing them easier
    swapped_vocab = dict([(value, key) for key, value in vectorizer.vocabulary_.items()])

    # print greatest weighted single word for each class
    # for i in range(np.shape(classifier.coef_)[0]):
    #     index = np.argmax(classifier.coef_[i])
    #     print(swapped_vocab[index])

    # to get the 5 most and least weighted words for each class
    for i in range(np.shape(classifier.coef_)[0]):
        index_list = classifier.coef_[i].argsort()[-5:][::-1]
        print("5 heaviest weighted words in Class ", i + 1, ":")
        for j in index_list:
            print("weight: ", classifier.coef_[i][j], ", word: ", swapped_vocab[j])
        print("\n")

        index_list_neg = classifier.coef_[i].argsort()[:5][::1]
        print("5 least weighted words in Class ", i + 1, ":")
        for j in index_list_neg:
            print("weight: ", classifier.coef_[i][j], ", word: ", swapped_vocab[j])
        print("\n")

    # to plot the distribution of coefficients in a bubble plot
    # first get the coefficients of the classifier and save them
    box_plot_data = classifier.coef_

    # at this point, the data shape is (20, 232483), we need to swap the axes
    # then shape becomes (232483, 20)
    box_plot_data = box_plot_data.swapaxes(0, 1)

    # labels for x-axis
    labels = []
    for i in range(20):
        j = i + 1
        text = "Class " + str(j)
        labels.append(text)

    plt.boxplot(box_plot_data, notch='True', patch_artist=True, labels=labels)
    fig = plt.figure()
    fig.set_size_inches(20,10)
    ax = fig.add_subplot(111)
    ax.boxplot(box_plot_data, widths=0.6, patch_artist=True)
    plt.title("Distribution of Logistic Regression Coefficient Weights")
    plt.xlabel("Classes")
    plt.ylabel("Value of Coefficient Weights")
    plt.show()


# --------------------------------- STATISTICAL SIGNIFICANCE TEST  --------------------------------- #
# compare two classifiers to see if their difference in performance is statistically significant
# function takes in both classifiers and vectorizers as parameters, returns chi2 and p-value
# uses modified McNemar's test
# See documentation at http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/

def stat_test(df, classifier1, classifier2):
    x = df['Cleaned'].values
    y = df['Class'].values

    # split dataset into training and test sets, with 80:20 split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1000, stratify=y)

    # vectorizer for first classifier
    # vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer()

    vectorizer.fit(x_train)
    X_test = vectorizer.transform(x_test)

    y_pred_1 = classifier1.predict(X_test)

    # vectorizer for second classifier
    # vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer()

    vectorizer.fit(x_train)
    X_test = vectorizer.transform(x_test)

    y_pred_2 = classifier2.predict(X_test)

    contingency_table = mcnemar_table(y_target=y_test,
                       y_model1=y_pred_1,
                       y_model2=y_pred_2)

    print(contingency_table)

    chi2, p_val = mcnemar(ary=contingency_table, corrected=True)
    print('chi-squared:', chi2)
    print('p-value:', p_val)


# --------------------------------- PLOT KNN GRAPH  --------------------------------- #
# hard coded function
def knn_plot():
    x1 = np.linspace(1, 15, num=15)
    x2 = np.linspace(1, 15, num=15)
    y1 = [0.888, 0.856, 0.864, 0.862, 0.856, 0.850, 0.847, 0.842, 0.836, 0.835, 0.832, 0.829, 0.824, 0.820, 0.820]
    y2 = [0.829, 0.768, 0.777, 0.786, 0.772, 0.769, 0.757, 0.697, 0.687, 0.684, 0.686, 0.683, 0.673, 0.666, 0.663]
    ya = [0.920, 0.900, 0.915, 0.908, 0.908, 0.904, 0.904, 0.902, 0.902, 0.900, 0.898, 0.893, 0.893, 0.891, 0.892]
    yb = [0.878, 0.814, 0.885, 0.826, 0.822, 0.810, 0.815, 0.809, 0.808, 0.804, 0.802, 0.796, 0.793, 0.789, 0.790]

    y3 = [0.888, 0.888, 0.884, 0.879, 0.869, 0.867, 0.859, 0.858, 0.848, 0.849, 0.846, 0.841, 0.837, 0.835, 0.833]
    y4 = [0.829, 0.829, 0.815, 0.808, 0.798, 0.794, 0.786, 0.772, 0.759, 0.762, 0.755, 0.751, 0.744, 0.742, 0.690]
    yc = [0.920, 0.920, 0.925, 0.922, 0.914, 0.912, 0.912, 0.910, 0.909, 0.908, 0.904, 0.903, 0.901, 0.901, 0.897]
    yd = [0.877, 0.877, 0.885, 0.893, 0.832, 0.826, 0.828, 0.825, 0.820, 0.818, 0.809, 0.810, 0.809, 0.807, 0.796]

    plt.figure(figsize=(24, 10))
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-', label='Count (uniform)')
    plt.plot(x1, ya, '.-', label='TF-IDF (uniform)')
    plt.plot(x1, y3, 'o-', label='Count (inverse)')
    plt.plot(x1, yc, '.-', label='TF-IDF (inverse)')
    plt.title('Summary of k-NN performance')
    plt.ylabel('Accuracy/Micro-F1')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, 'o-', label='Count (uniform)')
    plt.plot(x2, yb, '.-', label='TF-IDF (uniform)')
    plt.plot(x2, y4, 'o-', label='Count (inverse)')
    plt.plot(x1, yd, '.-', label='TF-IDF (inverse)')
    plt.xlabel('k-value (number of nearest neighbours)')
    plt.ylabel('Macro-F1')
    plt.legend()
    plt.show()

# ------------------------------------------ SAMPLE CODE ------------------------------------------ #

# load dataset as pandas frame
df = load_data('/.../data.csv')

# # preprocess data (output already included in our data file)
# df['Cleaned'] = cleanup_text_token(df['Text'])

# the run_model function takes three parameters, returns a trained classifier, vectorizer, and confusion matrix
# 1 - df - the dataframe we loaded above
# 2 - vectorizer - the name of the vectorizer (options: "count" or "tfidf")
# 3 - classifier - the name of the classifier
#     (options: "naive_bayes", "decision_tree", "random_forest", "logistic_regression", "linear_svm", "nonlinear_svm", "knn", "mlp")
vectorizer, classifier, cm = run_model(df, "count", "logistic_regression")

# run statistical test by passing in 2 classifiers to compare
# stat_test(classifier1, classifier2)

# NOTE: function requires matplotlib v3.1.0 or earlier to run (NOT v3.1.1, which is the current build, which has a bug)
# prints confusion matrix
plot_confusion_matrix(cm, normalize=False, target_names=[i for i in range(1,21)], title='Confusion Matrix')

# prints normalised confusion matrix
plot_confusion_matrix(cm, target_names=[i for i in range(1,21)], title='Normalised Confusion Matrix')
