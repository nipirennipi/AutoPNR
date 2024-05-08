from sklearn.metrics import roc_auc_score
import numpy as np


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ctr_score(y_true, y_score, k=1):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.mean(y_true)


def evaluate(impid_list, label_list, score_list):
    impres = {}
    for impid, label, score in zip(impid_list, label_list, score_list):
        if impid not in impres:
            impres[impid] = {}
            impres[impid]['label'] = []
            impres[impid]['score'] = []
        impres[impid]['label'].append(label)
        impres[impid]['score'].append(score)

    auc_list, mrr_list, ndcg5_list, ndcg10_list = [], [], [], []
    for impid in impres.keys():
        label = impres[impid]['label']
        score = impres[impid]['score']

        imp_auc = roc_auc_score(label, score)
        imp_mrr = mrr_score(label, score)
        imp_ndcg5 = ndcg_score(label, score, k=5)
        imp_ndcg10 = ndcg_score(label, score, k=10)

        auc_list.append(imp_auc)
        mrr_list.append(imp_mrr)
        ndcg5_list.append(imp_ndcg5)
        ndcg10_list.append(imp_ndcg10)

    auc = np.mean(auc_list)
    mrr = np.mean(mrr_list)
    ndcg5 = np.mean(ndcg5_list)
    ndcg10 = np.mean(ndcg10_list)
    return auc, mrr, ndcg5, ndcg10
