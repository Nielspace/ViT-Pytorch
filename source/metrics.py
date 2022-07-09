
import torch 
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns


def Confusion_matrix(y, pred, vis=True):
    """
    Calculates the confusion matrix for the model.
    """
    confusion_matrix = torch.zeros(400, 400)
    for t, p in zip(y.view(-1), pred.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    if vis:
        plt.figure(figsize=(10, 10))
        sns.heatmap(confusion_matrix.numpy())
        plt.savefig('metadata/confusion_matrix.png', dpi=300)
        plt.show()


def metrics(y, pred): 
    classification_report__ = classification_report(y, pred)
    accuracy_score__ = accuracy_score(y, pred)
    precision_score__ = precision_score(y, pred)
    recall_score__ = recall_score(y, pred)
    f1_score__ = f1_score(y, pred)

    return classification_report__, accuracy_score__, precision_score__, recall_score__, f1_score__ 


