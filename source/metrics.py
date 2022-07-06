


def confusion_matrix(y, pred):
    """
    Calculates the confusion matrix for the model.
    """
    confusion_matrix = torch.zeros(400, 400)
    for t, p in zip(y.view(-1), pred.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    return cm

def AUC_ROC(self):
    """
    Calculates the AUC ROC for the model.
    """
    y_pred = self.predict(self.X_test)
    return roc_auc_score(self.y_test, y_pred)

def f1scores(self): 
    """
    Calculates the f1 scores for the model.
    """
    y_pred = self.predict(self.X_test)
    return f1_score(self.y_test, y_pred, average='micro')

