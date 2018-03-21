import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from data import load_dataset, one_hot

FIG_PATH = '../report/figures/'
npoints = 10

def showCM(cm,methodname):
    # Show confusion matrix in a separate window
    fig1 = plt.figure(1)
    plt.matshow(cm)
    plt.title('Confusion matrix for '+methodname)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    fig1.savefig(FIG_PATH+methodname+'CM.pdf')

if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid = load_dataset('threshold')
    #y_train = one_hot(y_train)
    #y_valid = one_hot(y_valid)

    y_train = np.reshape(y_train, y_train.shape[0])
    y_valid = np.reshape(y_valid, y_valid.shape[0])
    Cspan = np.logspace(-3 ,2,num =npoints)
    cm = np.zeros(npoints,10,10)
    score = np.zeros(npoints)
    for i in range(npoints):
        
    
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
        y_pred = clf.fit(x_train, y_train).predict(x_valid)

        # Compute confusion matrix
        cm[i,:,:] = confusion_matrix(y_valid, y_pred)
        score[i] = clf.score(x_valid, y_valid)
        print('Score for i = '+i+': '+score[i])
        imax = np.argmax(score)
        Cmax = Cspan[imax]
    showCM(cm[imax,:,:].reshape(10,10),'LinearSVM')
    np.savetxt(FIG_PATH+'LinearSVMCM.csv', np.array(score)[None,:], delimiter=' & ' ,newline=' \\\\\n',fmt = "%s")
    np.savetxt(FIG_PATH+'LinearSVMCMbest.csv', np.array(cm[imax,:,:]).reshape(10,10), delimiter=' & ' ,newline=' \\\\\n',fmt = "%s")
    np.savetxt(FIG_PATH+'LinearSVMCspan.csv', np.array(Cspan)[None,:], delimiter=' & ' ,newline=' \\\\\n',fmt = "%s")

        


