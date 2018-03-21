import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

from data import load_dataset, save_array

if __name__ == '__main__':
    
    results = np.zeros((30, 3))
    i = 0

    for dataset in ['big', 'og', 'threshold']:
        print('Loading...')
        x_train, y_train, x_valid, y_valid = load_dataset(dataset)
        print('Done')

        y_train = np.reshape(y_train, y_train.shape[0])
        y_valid = np.reshape(y_valid, y_valid.shape[0])

        for C in np.logspace(-4,0,10):
            print('Dataset: {}, C: {}'.format(dataset, C))

            clf = LinearSVC(
                C=C, 
                verbose=0)

            clf.fit(x_train, y_train)
        
            train_acc = clf.score(x_train, y_train)
            valid_acc = clf.score(x_valid, y_valid)
            
            print('C: {}'.format(C))
            print('training accuracy: {}'.format(train_acc))
            print('validation accuracy: {}\n'.format(valid_acc))

            results[i,0] = C
            results[i,1] = train_acc
            results[i,2] = valid_acc
            i += 1
        print('\n') 
    save_array(results, 'Q1_res.csv')


