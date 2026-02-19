import numpy as np
import torch
from numpy.linalg import norm
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm
import OODutils as utils

recall = 0.95

def maxSoftmaxProb(inSoftmaxVal, oodSoftmax):
    inScore = inSoftmaxVal.max(axis=-1)
    oodScore = oodSoftmax.max(axis=-1)
    oodAreaUnderCurve = utils.auc(inScore, oodScore)[0]
    oodFalsePositiveRate = utils.fprr(inScore, oodScore, recall)
    print(f'Max Softmax Probability: Area Under Curve (ROC)={oodAreaUnderCurve}, False Positive Rate={oodFalsePositiveRate}')

def maxLogitScore(inLogitVal, oodLogit):
    inScore = inLogitVal.max(axis=-1)
    oodScore = oodLogit.max(axis=-1)
    oodAreaUnderCurve = utils.auc(inScore, oodScore)[0]
    oodFalsePositiveRate = utils.fprr(inScore, oodScore, recall)
    print(f'Max Logit Score: Area Under Curve (ROC)={oodAreaUnderCurve}, False Positive Rate={oodFalsePositiveRate}')

def mahalanobis(inFeatureTrain, trainLabels, inFeatureVal, oodFeature, numClasses):
    trainMeans = []
    trainFeatCentered = []
    for i in tqdm(range(numClasses)):
        fs = inFeatureTrain[trainLabels == i]
        m = fs.mean(axis=0)
        trainMeans.append(m)
        trainFeatCentered.extend(fs - m)

    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.array(trainFeatCentered).astype(np.float64))
    mean = torch.from_numpy(np.array(trainMeans)).cuda().float()
    prec = torch.from_numpy(ec.precision_).cuda().float()
    inScore = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in tqdm(torch.from_numpy(inFeatureVal).cuda().float())])
    oodScore = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in tqdm(torch.from_numpy(oodFeature).cuda().float())])
    oodAreaUnderCurve = utils.auc(inScore, oodScore)[0]
    oodFalsePositiveRate = utils.fprr(inScore, oodScore, recall)
    print(f'Mahalanobis: Area Under Curve (ROC)={oodAreaUnderCurve}, False Positive Rate={oodFalsePositiveRate}')

def energyScore(inLogitVal, oodLogit):
    inScore = logsumexp(inLogitVal, axis=-1)
    oodScore = logsumexp(oodLogit, axis=-1)
    oodAreaUnderCurve = utils.auc(inScore, oodScore)[0]
    oodFalsePositiveRate = utils.fprr(inScore, oodScore, recall)
    print(f'Energy Score: Area Under Curve (ROC)={oodAreaUnderCurve}, False Positive Rate={oodFalsePositiveRate}')

def vim(inFeatureTrain, inFeatureVal, oodFeature, inLogitTrain, inLogitVal, oodLogit, u):
    DIM = 300
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(inFeatureTrain - u)
    eigVals, eigenVectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigenVectors.T[np.argsort(eigVals * -1)[DIM:]]).T)
    inVlogitTrain = norm(np.matmul(inFeatureTrain - u, NS), axis=-1)
    alpha = inLogitTrain.max(axis=-1).mean() / inVlogitTrain.mean()
    inVlogitVal = norm(np.matmul(inFeatureVal - u, NS), axis=-1) * alpha
    inEnergyVal = logsumexp(inLogitVal, axis=-1)
    inScore = -inVlogitVal + inEnergyVal
    oodEnergy = logsumexp(oodLogit, axis=-1)
    oodVlogit = norm(np.matmul(oodFeature - u, NS), axis=-1) * alpha
    oodScore = -oodVlogit + oodEnergy
    oodAreaUnderCurve = utils.auc(inScore, oodScore)[0]
    oodFalsePositiveRate = utils.fprr(inScore, oodScore, recall)
    print(f'ViM: Area Under Curve (ROC)={oodAreaUnderCurve}, False Positive Rate={oodFalsePositiveRate}')
