"""
MNIST Numpy Assignments
Advanced Topics in Machine Learning, UCL (COMPGI13)
Model Performance Visualisation Functions
Author: Adam Hornsby
"""

import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./output/' + title + '.png', bbox_inches='tight');
    plt.clf();
    plt.cla();

def generate_confusion_plot(y, y_hat, model_title):

    cnf_matrix = confusion_matrix(y, y_hat)
    np.set_printoptions(precision=2)
    # plot unnormalised
    plt.figure()
    plot_confusion_matrix(cnf_matrix, range(0,10),
                          normalize=False,
                          title=model_title + 'confusion_matrix',
                          cmap=plt.cm.Blues)

    # plot normalised
    plt.figure()
    plot_confusion_matrix(cnf_matrix, range(0,10),
                          normalize=True,
                          title=model_title + 'confusion_matrix_norm',
                          cmap=plt.cm.Blues)

def plot_train_test_loss(epoch_id, train_loss, test_loss, title):
    """Plot the training and test set loss on a line chart"""

    plt.plot(epoch_id, train_loss, label='Training Set', color='gray')
    plt.plot(epoch_id, test_loss, label='Test Set', color='blue')

    plt.legend(loc='lower left')
    plt.xlabel('Training Iteration')
    plt.ylabel('Error')

    plt.tight_layout()
    plt.savefig('./output/' + title + '.png', bbox_inches='tight');
    plt.clf();
    plt.cla();