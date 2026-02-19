import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import metrics as metrics

def fprr(inConf, oodConf, truePositiveRate):
    inTotal = len(inConf)

    if inTotal == 0 and len(oodConf) == 0:
        falsePositives = 0
    elif inTotal == 0:
        falsePositives = 0
    else:
        recalls = int(np.floor(truePositiveRate * inTotal))
        thresh = np.sort(inConf)[-recalls]
        falsePositives = np.sum(oodConf >= thresh)
        
    oodTotal = len(oodConf)
    falsePositiveRate = falsePositives / max(1, oodTotal)

    return falsePositiveRate

def auc(inConf, oodConf):
    conf = np.concatenate((inConf, oodConf))
    inIndicator = np.concatenate((np.ones_like(inConf), np.zeros_like(oodConf)))

    falsePositiveRate, truePositiveRate, _ = metrics.roc_curve(inIndicator, conf)
    inPrecision, inRecall, _ = metrics.precision_recall_curve(inIndicator, conf)
    oodPrecision, oodRecall, _ = metrics.precision_recall_curve(1 - inIndicator, 1 - conf)

    auroc = metrics.auc(falsePositiveRate, truePositiveRate)
    inAupr = metrics.auc(inRecall, inPrecision)
    outAupr = metrics.auc(oodRecall, oodPrecision)

    return auroc, inAupr, outAupr

def neco(feature_id_train, feature_id_val, feature_ood, neco_dim):
    ss = StandardScaler()
    complete_vectors_train = ss.fit_transform(feature_id_train)
    complete_vectors_test = ss.transform(feature_id_val)
    complete_vectors_ood = ss.transform(feature_ood)

    pca_estimator = PCA(feature_id_train.shape[1])
    _ = pca_estimator.fit_transform(complete_vectors_train)
    cls_test_reduced_all = pca_estimator.transform(complete_vectors_test)
    cls_ood_reduced_all = pca_estimator.transform(complete_vectors_ood)

    cls_test_reduced = cls_test_reduced_all[:, :neco_dim]
    cls_ood_reduced = cls_ood_reduced_all[:, :neco_dim]
    l_ID = []
    l_OOD = []

    for i in range(cls_test_reduced.shape[0]):
        sc_complet = LA.norm((complete_vectors_test[i, :]))
        sc = LA.norm(cls_test_reduced[i, :])
        sc_finale = sc/sc_complet
        l_ID.append(sc_finale)
    for i in range(cls_ood_reduced.shape[0]):
        sc_complet = LA.norm((complete_vectors_ood[i, :]))
        sc = LA.norm(cls_ood_reduced[i, :])
        sc_finale = sc/sc_complet
        l_OOD.append(sc_finale)
    l_OOD = np.array(l_OOD)
    l_ID = np.array(l_ID)

    score_id = l_ID
    score_ood = l_OOD
    auc_ood = auc(score_id, score_ood)[0]
    recall = 0.95
    fpr_ood, _ = fprr(score_id, score_ood, recall)
    print(f'NECO: Area Under Curve (ROC)={auc_ood}, False Positive Rate={fpr_ood}')
    