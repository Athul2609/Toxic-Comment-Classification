import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report,confusion_matrix

a=np.load("./all_labels.npy")
s=a.shape[0]
for i in range(a.shape[1]):
    print(f"the propotions of 1s is {((np.sum(a[:,i]))/s)*100} \n")