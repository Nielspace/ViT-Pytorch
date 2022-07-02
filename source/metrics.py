


def confusion_matrix(self):
    """
    Calculates the confusion matrix for the model.
    """
    
    y_pred = self.predict(self.X_test)
    cm = confusion_matrix(self.y_test, y_pred)
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

