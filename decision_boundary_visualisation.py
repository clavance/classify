import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('/.../data.csv', header='infer', encoding='latin1')
x = df['Cleaned'].values
y = df['Class'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1000, stratify=y)

# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer() # choose vectorizer

vectorizer.fit(x_train)
X_train = vectorizer.transform(x_train)
X_test = vectorizer.transform(x_test)

svd = TruncatedSVD(n_components=50, n_iter=5, random_state=1000)
X_train = svd.fit_transform(X_train)
X_test = svd.fit_transform(X_test)

tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
X_train = tsne.fit_transform(X_train)
X_test = tsne.fit_transform(X_test)

k = 1 # choose value for k-NN
# weights = 'uniform'
weights = 'distance' # choose distance weighting
classifier = KNeighborsClassifier(n_neighbors=k, weights='distance')

classifier.fit(X_train, y_train)

# list of 20 simple, distinct colours from:
# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
colors = '#e6194b,#3cb44b,#ffe119,#4363d8,#f58231,#911eb4,#46f0f0,#f032e6,#bcf60c,#fabebe,#008080,#e6beff,#9a6324,#fffac8,#800000,#aaffc3,#808000,#ffd8b1,#000075,#808080,#ffffff,#000000'

fig = plt.figure(figsize=(40,25))
fig = plot_decision_regions(X_test, y_test, clf=classifier, colors=colors)
plt.legend(loc=2, prop={'size':25})
plt.xlabel('x-dimension')
plt.ylabel('y-dimension')
plt.title('k-NN decision boundary visualisation, k='+str(k))
plt.show()
