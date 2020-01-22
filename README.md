# Project files

This is a summary of how to run the models and visualisations in this project.

We provide the files for running all the models discussed in the project.

+ Download and save the [__data.csv__](https://www.dropbox.com/s/v8qhwjwobp8ti00/data.csv?dl=0) file. 
+ Due to the file size limitations in CATe, we have hosted the file [here](https://www.dropbox.com/s/v8qhwjwobp8ti00/data.csv?dl=0).
+ It contains 3 columns per line:
  1. "Text": consisting of the raw text samples, extracted from HTML source
  2. "Class": the class label belonging to the text sample
  3. "Cleaned": the preprocessed text samples, for easy access. We discuss the preprocessing steps thoroughly in the project, and also provide the functions for preprocessing in our files.
+ Files provided:
  1. _dataset.py_: get word counts, visualise dataset
  2. _machine_learning_models.py_: run machine learning models
  3. _deep_learning_models.py_: run deep learning models (excl. HAN)
  4. _han.py_: run hierarchical attention network (HAN) model
  5. _decision_boundary_visualisation.py_: visualisation for k-NN classifier
  6. _extract.py_: extract text samples from raw html files
  7. _decision_tree.pdf_: image of decision tree visualisation
  8. _random_forest.pdf_: image of random forest visualisation

## Libraries
+ [tensorflow](tensorflow.org) (machine learning library)
+ [keras](keras.io) (ML library, used with Tensorflow backend)
+ [pandas](pandas.pydata.org) (for data handling)
+ [spaCy](spacy.io) (for various NLP tasks)
+ [mlxtend](http://rasbt.github.io/mlxtend/) (for some plotting and additional functions, statistical testing)
+ [adjustText](https://adjusttext.readthedocs.io/en/latest/) (to adjust matplotlib plot overlaps)
+ [seaborn](seaborn.pydata.org) (for visualisation)
+ [matplotlib](matplotlib.org) (for visualisation)
+ [beautifulsoup4](pypi.org/project/beautifulsoup4) (for scraping raw html text files)


## Installation

Note that we do not use the latest version of matplotlib (v3.1.1), because of a bug in plotting confusion matrices.

With Python 3.x, using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install scikit-learn tensorflow keras pandas spacy mlxtend adjustText seaborn matplotlib==3.1.0 beautifulsoup4
python -m spacy download en_core_web_lg
```

## Usage of _dataset.py_

Import all dependencies and function definitions in the file.

We acknowledge that the cleanup_text functions and plot_tokens_clean function are based on [this](https://www.kaggle.com/enerrio/scary-nlp-with-spacy-and-keras) kernel.

+ Store data in a pandas dataframe, 'df'
+ Pass in as a string the file location where data.csv has been saved.

```python
df = load_data('/.../data.csv') 
```

+ load the largest nlp model from spaCy

```python
nlp = spacy.load('en_core_web_lg') 
```

+ this function takes loaded pandas dataframe as argument
+ prints the counts of each sample in each class
+ plots bar chart of distributions of samples in each class
+ prints counts of tokens/sentences per sample
+ prints bar charts of occurrences of most common words in each class
```python
class_distributions(df)
```

+ plot a visualisation of spaCy's embeddings of some common words 
```python
visualise_embeddings()
```

+ visualise our dataset with count or TF-IDF vectorizer, with dimensionality reduction to 2D
+ takes in the dataframe and two strings as parameters
+ the first string is the vectorizer ("count" or "tfidf")
+ the second string chooses whether to visualise the full dataset or only the test set ("full" or "test")

```python
visualise_data(df, "count", "full")
```

## Usage of _machine_learning_models.py_

Import all dependencies and function definitions in the file.

We acknowledge that the plot_confusion_matrix function is based on [this](https://medium.com/@deepanshujindal.99/how-to-plot-wholesome-confusion-matrix-40134fd402a8) article.


+ Store data in a pandas dataframe, 'df'
+ Pass in as a string the file location where data.csv has been saved.
```python
df = load_data('/.../data.csv') 
```

+ the run_model function takes three parameters, returns a trained classifier, vectorizer, and confusion matrix; prints classification report from sklearn, accuracy and macro-F1
+ parameters:
  1. df - the dataframe we loaded above
  2. vectorizer - the name of the vectorizer (options: "count" or "tfidf")
  3. classifier - the name of the classifier
   + classifier options:
      + "naive_bayes"
      + "decision_tree"
      + "random_forest", 
      + "logistic_regression"
      + "linear_svm",
      + "nonlinear_svm"
      + "knn"
      + "mlp"
```python
vectorizer, classifier, cm = run_model(df, "count", "logistic_regression")
```

+ next, run another classifier if desired

```python
vectorizer2, classifier2, cm2 = run_model(df, "tfidf", "linear_svm")
```

+ print confusion matrix
+ takes confusion matrix object returned by run_model as argument
+ if normalize=True, prints a normalised confusion matrix
+ change title text string as desired
```python
plot_confusion_matrix(cm, normalize=False, target_names=[i for i in range(1,21)],
 title='Confusion Matrix')
```

+ run a mcnemar's test if desired (pass in two classifiers returned from run_model)
```python
stat_test(df, classifier1, classifier2)
```

+ generate plot of logistic regression coefficients
+ pass in classifier and vectorizer objects as arguments
+ only works if passing in a logistic regression classifier
```python
plot_lr_coef(classifier, vectorizer)
```

+ generate k-NN performance graph used in our report
+ values in this plot are hard-coded from our results
```python
knn_plot()
```

## Usage of _deep_learning_models.py_

Import all dependencies and function definitions in the file.

+ Store data in a pandas dataframe, 'df'
+ Pass in as a string the file location where data.csv has been saved.
```python
df = load_data('/.../data.csv') 
```

+ load the largest nlp model from spaCy
+ by default, the spacy model takes in inputs of max length 1,000,000 chars
+ our dataset has max char length 2,871,868, so we need to configure this

```python
nlp = spacy.load('en_core_web_lg')
nlp.max_length = 2871868
```

+ choose some options
+ if using word embeddings, MAX_LENGTH controls max number of words per sample
+ if using sentence embeddings, MAX_LENGTH controls max number of sentences per sample

```python
MAX_LENGTH = 300 # the max length per sample (choose wisely)
NB_EPOCHS = 50 # number of epochs over which to run. choose some integer, ideally <100
EARLY_STOPPING = True
PATIENCE = 5 # patience for early stopping, if set to True. choose some integer < NB_EPOCHS
```

+ we run the function to add embeddings from spacy to our dataframe
+ function takes in 3 parameters, returns our df
  1. df - the pandas dataframe containing our data
  2. embedding - the type of embedding (options: "word_embeddings", "sentence_embeddings")
  3. MAX_LENGTH - options: an integer between 1 to 300,000 but due to memory requirements, ideally <=1,000. Note: setting length=1000 already loads ~75GB of embeddings in memory
```python
df = get_embeddings(df, "word_embeddings", MAX_LENGTH)
```

+ we run the neural network model, function takes in 6 parameters, including those defined above:
  1. df - the pandas dataframe containing our data
  2. architecture - the type of architecture
   (options: "mlp", "cnn", "ngram_cnn", "lstm", "bi_lstm")
  + note: "ngram_cnn" is based on the CNN implemented in (Kim, 2014) as discussed in our report
  3. MAX_LENGTH
  4. NB_EPOCHS
  5. EARLY_STOPPING 
  6. PATIENCE
```python
model, cm = run_model(df, "mlp", MAX_LENGTH, NB_EPOCHS, EARLY_STOPPING, PATIENCE)
```

+ plot confusion matrix (normalised or not), if desired
```python
plot_confusion_matrix(cm, normalize=False, target_names=[i for i in range(1,21)])
```

## Usage of _han.py_

We acknowledge that the implementation of the dot_product function and AttentionWithContext layer, and general design of the network, is based on [this](https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py) and [that](https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2) repository.

However, based on those existing implementations, we configured the network to work in our context and for our purposes.

Import all dependencies and function definitions in the file.

+ Store data in a pandas dataframe, 'df'
+ Pass in as a string the file location where data.csv has been saved.
```python
df = load_data('/.../data.csv') 
```
+ set some parameters which can be chosen
```python
MAX_WORDS = 261737  # number of unique tokens in our training set
MAX_SENTS = 300 # we stick to max. 300 sentences per sample
MAX_SENT_LENGTH = 300 # we stick to max. 300 tokens per sentence
VALIDATION_SPLIT = 0.2 # same training:validation split, 80:20
EMBEDDING_DIM = 300  # we stick to embedding dimensions of 300
```

+ run the model
```python
model, cm = run_model(df, MAX_WORDS, MAX_SENTS, MAX_SENT_LENGTH, VALIDATION_SPLIT, EMBEDDING_DIM)
```

+ plot the confusion matrix
```python
plot_confusion_matrix(cm, normalize=False, target_names=[i for i in range(1,21)], title='Confusion Matrix')
```

## Usage of _decision_boundary_visualisation.py_

Run the entire python script.

We used a list of 20 distinct colours from [here](https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/).

+ It generates a plot of our k-NN classifier's decision boundaries.
+ The dimensions of our features have been reduced to 2D, so this is not representative of the actual classifiers we trained.
+ In the file there is a variable, _vectorizer_, which is set to the _TfidfVectorizer()_.
+ Change this to _CountVectorizer()_ if desired.
```python
# vectorizer = CountVectorizer()  
vectorizer = TfidfVectorizer() # choose vectorizer
```
+ There are also variables, _k=1_ and _weights='distance'_.
+ _k_ can be reset to any desired value of _k_, and the corresponding plot will be generated.
+ _weights_ can be set to _'distance'_ or _'uniform'_.
```python
k = 1 # choose value for k-NN  
# weights = 'uniform'  
weights = 'distance' # choose distance weighting  
classifier = KNeighborsClassifier(n_neighbors=k, weights='distance')
```
+ run the entire python script once decided.



## Usage of _extract.py_

This file extracts the text samples from the raw HTML files from the original EURLEX dataset. This can be accessed [here](http://www.ke.tu-darmstadt.de/resources/eurlex).

To run this file, first download and unzip all raw HTML files from [here](http://www.ke.tu-darmstadt.de/files/resources/eurlex/eurlex_html_EN_NOT.zip). Note the directory in which these files have been saved.

Create a new empty directory where the extracted text files will be stored.

+ Import dependencies

```python
import os
from bs4 import beautifulsoup
```

+ Initialise variables containing the filepath of the directory where all the raw HTML files are, and the directory where the extracted text files will be stored.
```python
base_dir = "/..." # the location of raw HTML files
second_dir = "/..." # the location of extracted files
```

Run the rest of the script.

The output of this process is the body text of each file, with a CELEX ID. We then matched the CELEX IDs with the document IDs in [this file](http://www.ke.tu-darmstadt.de/files/resources/eurlex/eurlex_ID_mappings.csv.gz).

At this stage, we have the body texts, with document IDs.  We then matched the document IDs with the top level directory codes, which are the class labels in the file _id2class_eurlex_DC_l1.qrels,_ from [this zip file](http://www.ke.tu-darmstadt.de/files/resources/eurlex/eurlex_id2class.zip).

Note that the output of this process has been saved in a format which is easy to work with, especially with a pandas dataframe, in the file __data.csv__.


## Acknowledgements
EURLEX dataset modified with permission.

Access the original dataset [here](http://www.ke.tu-darmstadt.de/resources/eurlex).

See the original paper [here](http://www.ke.tu-darmstadt.de/publications/papers/loza10eurlex.pdf).

Access the full EUR-Lex repository [here](https://eur-lex.europa.eu/homepage.html).

EUR-Lex data used and modified with permissions and remains the property of:

'© European Union, https://eur-lex.europa.eu, 1998-2019'
