import numpy as np
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser

from EM_algorithm import EM

def display_mean(mean, label, w, h):
    mean = np.where(mean>0.5, 1, 0)
    mean = mean.reshape(w, h)
    print('class %d'%label)
    for i in range(mean.shape[1]):
        for j in range(mean.shape[0]):
            print(mean[i, j], end=' ')
        print()
    print('\n\n')

def confusion_matrix(y, y_pred, label):
    sub_y = np.where(y==label, 1, 0)
    sub_y_pred = np.where(y_pred==label, 1, 0)
    
    tp = np.sum((sub_y_pred == 1) & (sub_y == 1))
    fp = np.sum((sub_y_pred == 1) & (sub_y == 0))
    tn = np.sum((sub_y_pred == 0) & (sub_y == 0))
    fn = np.sum((sub_y_pred == 0) & (sub_y == 1))
    return tp, fp, tn, fn

def confusion_matrix_report(y, y_pred, label):
    tp, fp, tn, fn = confusion_matrix(y, y_pred, label)
    print('\n\n'+'-'*50)
    print('\nConfusion Matrix %d:'%label)
    print('\t\t%23s%23s'%('Predict number %d'%label, 'Predict not number %d'%label))
    print('Is number %d  %23s %23s'%(label, tp, fp))
    print('Isn\'t number %d %21s %23s\n'%(label, fn, tn))
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print('Sensitivity (Successfully predict number %d)    : %.5f'%(label, sensitivity))
    print('Specificity (Successfully predict not number %d): %.5f'%(label, specificity))

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--k", help="k-component (the number of categories).", default=10, type=int)
    parser.add_argument("--path", help="the dataset path.", default='./mnist')
    parser.add_argument("--wh", help="the length and width of input.", default=(28, 28), type=int)
    args = parser.parse_args()

    k = args.k
    path = args.path
    w, h = args.wh

    # load data
    X = np.load(os.path.join(path, 'X_train.npy'))
    y = np.load(os.path.join(path, 'y_train.npy')).reshape(-1)
    X = np.reshape(X, (X.shape[0], w*h))
    X = np.where(X > 100, 1, 0)
    labels = np.unique(y)
    models = dict()
    k = int(k / len(labels))

    for label in labels:
        X_subset = X[y==label]
        models[label] = EM(k)
        models[label].fit(X_subset)

    likelihoods = np.ndarray(shape=(len(labels), X.shape[0]))

    for label in labels:
        likelihoods[label] = models[label].predict(X)

    predicted_labels = np.argmax(likelihoods, axis=0)

    y_pred = predicted_labels
    accuracy = np.mean(y_pred == y)
    print('accuracy: %.5f\n'%accuracy)

    # Display Results
    for label in labels:
        mean = models[label].mu
        display_mean(mean, label, w, h)

    accuracy = np.mean(y_pred == y)

    for label in range(len(labels)):
        confusion_matrix_report(y, y_pred, label)

    print()
    print('Total iteration to converge: %d'%10)
    print('Total error rate\n: %.5f'%(1-accuracy))