import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

from data import load_dataset, one_hot

if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid = load_dataset('threshold')
    #y_train = one_hot(y_train)
    #y_valid = one_hot(y_valid)

    y_train = np.reshape(y_train, y_train.shape[0])
    y_valid = np.reshape(y_valid, y_valid.shape[0])

    clf = LinearSVC(
        C=0.9,
        verbose=1
    )

    #clf = LogisticRegression(
    #    C=0.9, 
    #    solver='lbfgs', 
    #    multi_class='multinomial', 
    #    n_jobs=-1,
    #    verbose=1)
    


    clf.fit(x_train, y_train)
    print(clf.score(x_valid, y_valid))
