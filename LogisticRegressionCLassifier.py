import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.special import logsumexp
from dimension_reduction import *
from helper_functions import *
import argparse

class LogisticRegression:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        onehot_encoder = OneHotEncoder(sparse=False)
        self.Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1,1))

    def fit(self, iter=1000, step=0.01, mu=0.01):
        self.loss_steps = self.gradient_descent(max_iter=iter, step=step, mu=mu)

    def plot_loss(self, step):
        title = 'Training Data Size: ' + str(self.X.shape[0]) + ' Step size = '+str(step)
        return self.loss_steps.plot(
            title=title,
            x='step', 
            y='loss',
            xlabel='iterations',
            ylabel='loss'
        )

    def predict(self, H):
        Z = - H @ self.W
        P = self.softmax(Z, axis=1)
        return np.argmax(P, axis=1)

    def softmax(self, Z, axis):
        sum =  logsumexp(Z, axis=axis).reshape(Z.shape[0],1)
        P = np.exp( Z - sum)
        return P

    def loss(self):

        Z = - self.X @ self.W
        N = self.X.shape[0]
        loss = 1/N * (np.trace(self.X @ self.W @ self.Y_onehot.T) + np.sum(logsumexp(Z, axis=1)))
        return loss

    def gradient(self, mu):

        Z = - self.X @ self.W
        P = self.softmax(Z, axis=1)
        N = self.X.shape[0]
        gradient = 1/N * (self.X.T @ (self.Y_onehot - P)) + 2 * mu * self.W
        return gradient

    def gradient_descent(self, max_iter=1000, step=0.1, mu=0.01):

        self.W = np.zeros((self.X.shape[1], self.Y_onehot.shape[1]))
        iter = 0
        step_lst = [] 
        loss_lst = []
        W_lst = []
    
        while iter < max_iter:
            iter += 1
            self.W -= step * self.gradient(mu)
            step_lst.append(iter)
            W_lst.append(self.W)
            loss_lst.append(self.loss())

        dataframe = pd.DataFrame({
            'step': step_lst, 
            'loss': loss_lst
        })
        return dataframe

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-training', '--training',required=False, default=10000, type=int)
    parser.add_argument('-testing', '--testing',required=False, default=1000, type=int)
    parser.add_argument('-step', '--step',required=False, default=0.1, type=float)
    parser.add_argument('-pca', '--pca',required=False, default=False, type=bool)
    parser.add_argument('-mda', '--mda',required=False, default=False, type=bool)
    parser.add_argument('-iter', '--iter',required=False, default=100, type=int)
    parser.add_argument('-reg', '--reg',required=False, default='0.01', type=float)
    parser.add_argument('-random', '--random',required=False, default=50000, type=int)
    args = vars(parser.parse_args())


    n_train = args['training']
    n_test = args['testing']
    step_size = args['step']
    usePca = args['pca']
    useMDA = args['mda']
    iterations = args['iter']
    regularization = args['reg']
    random = args['random']

    # get training and testing data
    training_data, y_train, testing_data, y_test = get_training_testing_data()
    flattened = np.array([training_data[i].flatten() for i in range(training_data.shape[0])])
    
    # Perform PCA/MDA whichever true
    print('Reducing dimensions......')
    if usePca:
        projected = doPCA(flattened)
    else:
        projected = doMDA(flattened, y_train, dim=9)
    
    # train model

    print('Fitting the Logistic Regression model')
    Y = y_train
    model = LogisticRegression(projected[:n_train], Y[:n_train])
    model.fit(iter=iterations, step=step_size, mu=regularization)
    model.plot_loss(step_size)
    plt.show()

    # predict
    pred = model.predict(projected[random : random + n_test]) == y_train[random : random + n_test]
    accuracy = np.where(pred == True)[0].shape[0] * 100 / n_test
    print('Accuracy of model = ', str(accuracy))
