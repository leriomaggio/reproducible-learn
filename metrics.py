## This code is written by Davide Albanese <albanese@fbk.eu>

## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np


def error(ya, yp):

    ya_arr, yp_arr = np.asarray(ya), np.asarray(yp)
    if ya_arr.shape[0] != yp_arr.shape[0]:
        raise ValueError("ya, yp: shape mismatch")
    return np.sum(ya_arr != yp_arr) / ya_arr.shape[0]


def accuracy(ya, yp):

    ya_arr, yp_arr = np.asarray(ya), np.asarray(yp)
    if ya_arr.shape[0] != yp_arr.shape[0]:
        raise ValueError("ya, yp: shape mismatch")
    return np.sum(ya_arr == yp_arr) / ya_arr.shape[0]


def confusion_matrix(ya, yp, classes=None):
    """
    actual (rows) x predicted (cols)
    """

    if classes is None:
        classes = np.unique(np.concatenate((ya, yp)))      
    else:
        classes = np.asarray(classes, dtype=np.int)
    
    k = classes.shape[0]

    cm = np.zeros((k, k), dtype=np.int)
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            cm[i, j] += np.sum(np.logical_and(ya == ci, yp == cj))

    return cm, classes


def confusion_matrix_binary(ya, yp):
    """
    Returns TN, FP, FN, TP 
    (correct rejection, false alarm or Type I error, miss or Type II error, hit)
    """

    classes = np.unique(np.concatenate((ya, yp)))
    if classes.shape[0] != 2:
        raise ValueError("Binary confusion matrix is defined for binary classification only")
    
    cm, _ = confusion_matrix(ya, yp, classes=classes)
    
    return cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]


def sensitivity(ya, yp):
    """ or true positive rate, hit rate, recall
    TP / P = TP / (TP + FN)
    """
    
    TN, FP, FN, TP = confusion_matrix_binary(ya, yp)
    if TP == 0.0:
        return 0.0
    else:
        return TP / (TP + FN)


def specificity(ya, yp):
    """or true negative rate
    TN / N = TN / (FP + TN) = 1 - FPR
    """

    TN, FP, FN, TP = confusion_matrix_binary(ya, yp)
    if TN == 0.0:
        return 0.0
    else:
        return TN / (FP + TN)


def fpr(ya, yp):
    """false positive rate or fall-out
    FP / N = FP / (FP + TN)
    """
    
    TN, FP, FN, TP = confusion_matrix_binary(ya, yp)
    if FP == 0.0:
        return 0.0
    else:
        return FP / (FP + TN)


def ppv(ya, yp):
    """positive predictive value or precision
    TP / (TP + FP)
    """

    TN, FP, FN, TP = confusion_matrix_binary(ya, yp)
    if TP == 0.0:
        return 0.0
    else:
        return TP / (TP + FP)


def npv(ya, yp):
    """negative predictive value
    TN / (TN + FN)
    """

    TN, FP, FN, TP = confusion_matrix_binary(ya, yp)
    if TN == 0.0:
        return 0.0
    else:
        return TN / (TN + FN)


def fdr(ya, yp):
    """false discovery rate
    FP / (FP+TP)
    """
    
    TN, FP, FN, TP = confusion_matrix_binary(ya, yp)
    if FP == 0.0:
        return 0.0
    else:
        return FP / (FP + TP)


def F1_score(ya, yp):
    precision = ppv(ya, yp)
    recall = sensitivity(ya, yp)
    return (2 * precision * recall) / (precision + recall)


def auc_wmw(ya, yp):
    """Compute the AUC by using the Wilcoxon-Mann-Whitney
    statistic.
    """

    ya_arr, yp_arr = np.asarray(ya), np.asarray(yp)
    classes = np.unique(ya_arr)
    if classes.shape[0] != 2:
        raise ValueError("AUC is defined for binary classification only")
    bn = (ya_arr == classes[0])
    bp = (ya_arr == classes[1])
    auc = 0.0
    for i in yp[bp]:
        for j in yp[bn]:
            if i > j:
                auc += 1.0
    return auc / (np.sum(bn) * np.sum(bp))


def dor(ya, yp):
    """Diagnostic Odds Ratio
    """

    TN, FP, FN, TP = confusion_matrix_binary(ya, yp)
    return (TP / FN) / (FP / TN)


def _expand(x, y):
    K = np.unique(np.concatenate((x, y)))
    X = np.zeros((x.shape[0], K.shape[0]), dtype=np.int)
    Y = np.zeros((y.shape[0], K.shape[0]), dtype=np.int)
    for i, k in enumerate(K):
        X[x==k, i] = 1
        Y[y==k, i] = 1
    return X, Y
        

def KCCC(x, y):
    """ K-category correlation coefficient.
    """

    EPS = np.finfo(np.float).eps
    k = x.shape[1]
    
    xn = x - np.mean(x, axis=0)
    yn = y - np.mean(y, axis=0)
    cov_xy = np.sum(xn * yn) / k
    cov_xx = np.sum(xn * xn) / k
    cov_yy = np.sum(yn * yn) / k

    cov_xxyy = cov_xx * cov_yy
    if cov_xxyy > EPS:
        rk = cov_xy / np.sqrt(cov_xx * cov_yy)
    else:
        rk = 0.0
    
    return rk


def KCCC_discrete(x, y, one_hot_enc=True):
    if one_hot_enc:
        X, Y = _expand(x, y)
    else:
        X, Y = x, y
    return KCCC(X, Y)
