import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
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
        plt.savefig("metadata/confusion_matrix.png", dpi=300)
        plt.show()


def metrics(y, pred):
    classification_report__ = classification_report(y, pred)
    accuracy_score__ = accuracy_score(y, pred)
    precision_score__ = precision_score(y, pred)
    recall_score__ = recall_score(y, pred)
    f1_score__ = f1_score(y, pred)

    return (
        classification_report__,
        accuracy_score__,
        precision_score__,
        recall_score__,
        f1_score__,
    )


def auroc(model, data, N_classes=400, device="cpu"):
    model.eval()
    y_test = []
    y_score = []
    with torch.no_grad():
        with tqdm(data, unit="iteration") as test_epoch:
            test_epoch.set_description(f"Epoch {epoch}")
            for i, (inputs, classes) in enumerate(test_epoch):
                inputs = inputs.to(device)
                y_test.append(F.one_hot(classes, N_classes).numpy())

                try:
                    bs, ncrops, c, h, w = inputs.size()
                except:
                    bs, c, h, w = inputs.size()
                    ncrops = 1
                if ncrops > 1:
                    outputs = model(inputs.view(-1, c, h, w))
                    outputs = outputs.view(bs, ncrops, -1).mean(1)
                else:
                    outputs, _ = model(inputs)
                y_score.append(outputs.cpu().numpy())
    y_test = np.array([t.ravel() for t in y_test])
    y_score = np.array([t.ravel() for t in y_score])
    # print(y_true)
    # print(y_pred)

    """
    compute ROC curve and ROC area for each class in each fold
    """

    fpr = dict()
    tpr = dict()
    local_roc_auc = dict()
    for i in range(N_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(y_test[:, i]), np.array(y_score[:, i]))
        local_roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    local_roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(N_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(N_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= N_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    local_roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})"
        "".format(local_roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})"
        "".format(local_roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(N_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})"
            "".format(i, local_roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([-1e-2, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristics")
    plt.legend(loc="lower right")
    plt.show()



