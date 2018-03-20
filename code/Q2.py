import numpy as np
import matplotlib.pyplot as plt

from neural_net import NeuralNetwork
from data import load_dataset, one_hot

# for testing purposes
def xor_data():
    X = np.array([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
    Y = np.array([[1.,0.], [0.,1.], [0.,1.], [1.,0.]])
    return X, Y, X[:], Y[:]

if __name__ == '__main__':

    #x_train, y_train, x_valid, y_valid = xor_data() 
    
    print('Loading dataset')
    x_train, y_train, x_valid, y_valid = load_dataset('threshold')
    print('Done loading')

    print('One hotting data')
    y_train = one_hot(y_train)
    y_valid = one_hot(y_valid)

    nn = NeuralNetwork(
        #layers=[x_train.shape[1], 5, y_train.shape[1]], 
        layers=[x_train.shape[1], 500, 200, y_train.shape[1]], 
        activations=['tanh', 'tanh'], 
        learning_rate=0.1,
        epochs=5,
        name='threshold_500x200')

    training_costs, validation_costs = nn.fit(
        x_train, 
        y_train, 
        x_valid, 
        y_valid, 
        minibatch_size=50, 
        verbose=True,
        save_step=1)
    
    plt.plot(training_costs, color='red')
    plt.plot(validation_costs, color='blue')
    plt.show()
