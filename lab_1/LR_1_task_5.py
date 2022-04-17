import numpy as np
import pandas as pd
df = pd.read_csv("data_metrics.csv")
df.read()

threash = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype("int")
df['predicted_LR'] = (df.model_LR >= 0.5).astype("int")
df.head()


from sklearn.metrics import confusion_matrix
res_base = confusion_matrix(df.actual_label.values, df.predicted_RF.values)



def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))


def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))


def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

def miedviediev_confusion_matrix(y_true, y_pred):
    res = [[0, 0], [0, 0]]

    if len(y_true) != len(y_pred):
        res = [[-1, -1], [-1, -1 ]]
    else:
        res[0][0] = find_TN(y_true, y_pred)
        res[0][1] = find_FP(y_true, y_pred)
        res[1][0] = find_FN(y_true, y_pred)
        res[1][1] = find_TP(y_true, y_pred)

    return res


res = miedviediev_confusion_matrix(df.actual_label.values, df.predicted_RF.values)


assert np.array_equal( res, res_base)

from sklearn.metrics import accuracy_score
accuracy_score(df.actual_label.values, df.predicted_RF.values)


def miedvidiev_accuracy_score(y_true, y_pred):
    metric = miedviediev_confusion_matrix(y_true, y_pred)

    return (metric[1][1] + metric[0][0]) / (metric[1][1] + metric[0][0] + metric[0][1] + metric[1][0])



from sklearn.metrics import recall_score
recall_score(df.actual_label.values, df.predicted_RF.values)


def miedviediev_recall(y_true, y_pred):
    metric = miedviediev_confusion_matrix(y_true, y_pred)

    return (metric[1][1]) / (metric[1][1] + metric[1][0])



from sklearn.metrics import precision_score
precision_score(df.actual_label.values, df.predicted_RF.values)


def miedviediev_precision(y_true, y_pred):
    metric = miedviediev_confusion_matrix(y_true, y_pred)
    return (metric[1][1]) / (metric[1][1] + metric[0][1])



from sklearn.metrics import f1_score
f1_score(df.actual_label.values, df.predicted_RF.values)


def miedviediev_f1(y_true, y_pred):
    precision = miedviediev_precision(y_true, y_pred)
    recall = miedviediev_recall(y_true, y_pred)

    return (2 * (precision * recall)) / precision + recall



from sklearn.metrics import roc_curve
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve( df.actual_label.values, df.model_LR.values)


import matplotlib.pyplot as plt

plt.plot(fpr_RF, tpr_RF, 'r-', label="RF")
plt.plot(fpr_LR, tpr_LR, "b-", label="LR")
plt.plot([0, 1], [0, 1], "k-", label="random")
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], "g-", label="perfect")
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


from sklearn.metrics import roc_auc_score
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print("AUC RF :%.3f", auc_RF)
print("AUC RF :%.3f", auc_LR)


plt.plot(fpr_RF, tpr_RF, 'r-', label="RF AUC :%.3f")
plt.plot(fpr_LR, tpr_LR, "b-", label="LR AUC :%.3f")
plt.plot([0, 1], [0, 1], "k-", label="random")
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], "g-", label="perfect")
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


#TASK 2.6
