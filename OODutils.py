import numpy as np
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
