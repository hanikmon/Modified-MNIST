import numpy as np
import matplotlib.pyplot as plt

from neural_net import NeuralNetwork
from data import load_dataset, one_hot, load_array, save_array

# for testing purposes
def xor_data():
    X = np.array([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
    Y = np.array([[1.,0.], [0.,1.], [0.,1.], [1.,0.]])
    return X, Y, X[:], Y[:]

if __name__ == '__main__':

    #x_train, y_train, x_valid, y_valid = xor_data() 
    
    print('Loading dataset')
    x_train, y_train, x_valid, y_valid = load_dataset('big')
    print('Done loading')
    
    print('One hotting data')
    y_train = one_hot(y_train)
    y_valid = one_hot(y_valid)


    nn = NeuralNetwork.load('big_350')
    preds = nn.predict(x_valid)
    save_array(preds, 'big_350_preds')
    print(nn.accuracy(x_valid, y_valid))




    
    #nn = NeuralNetwork(
    #    #layers=[x_train.shape[1], 5, y_train.shape[1]], 
    #    layers=[x_train.shape[1], 350, y_train.shape[1]], 
    #    activations=['tanh'], 
    #    learning_rate=0.03,
    #    epochs=15,
    #    name='big_350')

    #training_costs, validation_costs = nn.fit(
    #    x_train, 
    #    y_train, 
    #    x_valid, 
    #    y_valid, 
    #    minibatch_size=100, 
    #    verbose=True,
    #    save_step=1)
    
    #plt.plot(training_costs, color='red', label='training error')
    #plt.plot(validation_costs, color='blue', label='validation error')
    #plt.title('Learning curve for one hidden layer with 350 neurons')
    #plt.legend()
    #plt.show()
    
    #print(training_costs)
    #print(validation_costs)

    
    #nn = NeuralNetwork(
    #    #layers=[x_train.shape[1], 5, y_train.shape[1]], 
    #    layers=[x_train.shape[1], 200, 100, y_train.shape[1]], 
    #    activations=['tanh', 'tanh'], 
    #    learning_rate=0.03,
    #    epochs=15,
    #    name='big_200_100')

    #training_costs, validation_costs = nn.fit(
    #    x_train, 
    #    y_train, 
    #    x_valid, 
    #    y_valid, 
    #    minibatch_size=100, 
    #    verbose=True,
    #    save_step=1)
    
    #plt.plot(training_costs, color='red', label='training error')
    #plt.plot(validation_costs, color='blue', label='validation error')
    #plt.title('Learning curve for two hidden layers, 200 and 100 neurons')
    #plt.legend()
    #plt.show()
    #print(training_costs)
    #print(validation_costs)
