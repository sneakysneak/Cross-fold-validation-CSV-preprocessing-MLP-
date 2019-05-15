import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import numpy
# random number of random state
seed = 9
numpy.random.seed(seed)
# Read in data
theData = pd.read_csv('dataset.csv', delimiter = ',')
# Supervised learning - last column is the output data
# Set x to the input data, 0-56
X = theData.values[:,0:56]
# Set y to the output data 57
Y = theData.values[:,57]
# Pre-process the input data
# robust scale is slightly more efficient
X = preprocessing.robust_scale(X)
# X = preprocessing.minmax_scale(X)
# X = preprocessing.scale(X)
# PCA ~38 choses the "best" num of columns to compare
pca = PCA(n_components = 38)
pca.fit(X)
X = pca.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
# Classifier - classification
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(15, 5), random_state=1)
# clf.fit(X, Y)
clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(15, 5),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=200, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)

# First fit
clf = clf.fit(X_train, Y_train)
# After  predict
Y_prediction = clf.predict(X_test)
print("Predicted result: ", accuracy_score(Y_test, Y_prediction))
# Shuffle data
# Split the data in 10 parts, use 20% of it for test, random state 1 to randomize
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
# Trying with StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# Print result
results = cross_val_score(clf, X, Y, cv=cv)
# Implementing with StratifiedKFold  for cross validation
resultsWithKFold = cross_val_score(clf, X, Y, cv=kfold)
print("Main results: ")
print("Cross fold validation results: ", results)
print("Cross fold validation results mean: ", results.mean())
print("\nResults with StratKFold - for curiosity \n")
print("Cross fold validation results with StratKFold: ",resultsWithKFold)
print("Cross fold validation results mean with StratKFold: ",resultsWithKFold.mean())
