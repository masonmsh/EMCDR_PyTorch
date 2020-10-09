import os
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn

from model import MF, MLP


def parse_args():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument('--dataset', type=str, default='', help="d")
    parser.add_argument('--dim', type=int, default=16, help="embedding")
    parser.add_argument('--layers', type=int, default=1, help="l")
    parser.add_argument('--batchsize', type=int, default=2, help="b")
    parser.add_argument('--topk', type=int, default=10, help="tk")
    return parser.parse_args()


def default_args():
    return 'Musical_Patio'


def test_dict(dataset):
    tp = 'data/%s/test.txt' % dataset
    testDict = {}
    with open(tp, 'r') as f:
        for line in f.readlines():
            test = list(map(int, line.strip().split()))
            idx = test.pop(0)
            testDict[idx] = test
    return testDict


def train_mat(dataset):
    tp = 'data/%s/t.csv' % dataset
    data = pd.read_csv(tp, ',', names=['u', 'i', 'r', 't'], engine='python')
    return data[['u', 'i']]


def load_data(dataset, mode):
    dp = 'data/%s/%s.csv' % (dataset, mode)
    data = pd.read_csv(dp, ',', names=['u', 'i', 'r', 't'], engine='python')
    data.sort_values(by=['u', 'i'], inplace=True)
    n_user = data['u'].max() + 1
    n_item = data['i'].max() + 1
    return n_user, n_item


def load_model(dataset, dim, layers):
    sp = 'pretrain/%s/MF_s.pth.tar' % dataset
    tp = 'pretrain/%s/MF_t.pth.tar' % dataset
    mp = 'pretrain/%s/Mapping.pth.tar' % dataset
    s_user, s_item = load_data(dataset, 's')
    t_user, t_item = load_data(dataset, 't')
    mf_s = MF(s_user, s_item, dim)
    mf_t = MF(t_user, t_item, dim)
    mapping = MLP(dim, layers)
    mf_s.load_state_dict(torch.load(sp))
    mf_t.load_state_dict(torch.load(tp))
    mapping.load_state_dict(torch.load(mp))
    return mf_s, mf_t, mapping


def batch_user(n_user, batch_size):
    for i in range(0, n_user, batch_size):
        yield list(range(i, min(i+batch_size, n_user)))


def pos_item(trainMat, users):
    items = []
    for u in users:
        item = trainMat.loc[trainMat['u'] == u]['i'].values
        items.append(item)
    return(items)


def get_label(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def recall_precision(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def ndcg_k(test_data, r, k):
    pred_data = r[:, :k]
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = get_label(groundTrue, sorted_items)
    ret = recall_precision(groundTrue, r, k)
    ndcg = ndcg_k(groundTrue, r, k)
    return {'precision': ret['precision'], 'recall': ret['recall'], 'ndcg': ndcg}


def main(dataset, dim, layers, batchsize, topk):
    testDict = test_dict(dataset)
    trainMat = train_mat(dataset)
    mf_s, mf_t, mapping = load_model(dataset, dim, layers)
    results = {'precision': 0, 'recall': 0, 'ndcg': 0}

    with torch.no_grad():
        test_users = list(testDict.keys())
        ratings = []
        truths = []
        for users in batch_user(len(test_users), batchsize):
            groundTrue = [testDict[u] for u in users]
            user_embed = mapping(mf_s.get_embed(users))
            rating = mf_t.get_rating(user_embed)
            positem = pos_item(trainMat, users)
            train_idx = []
            train_item = []
            for idx, item in enumerate(positem):
                train_idx.extend([idx]*len(item))
                train_item.extend(item)
            rating[train_idx, train_item] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=topk)
            ratings.append(rating_K.cpu())
            truths.append(groundTrue)
        X = zip(ratings, truths)
        test_results = []
        for x in X:
            test_results.append(test_one_batch(x, topk))
        for result in test_results:
            results['precision'] += result['precision']
            results['recall'] += result['recall']
            results['ndcg'] += result['ndcg']
        results['precision'] /= float(len(test_users))
        results['recall'] /= float(len(test_users))
        results['ndcg'] /= float(len(test_users))

        print('[%d] precision = %.4f, recall = %.4f, NDCG = %.4f' %
              (topk, results['precision'], results['recall'], results['ndcg']))


if __name__ == '__main__':
    args = parse_args()
    args.dataset = default_args()
    main(args.dataset, args.dim, args.layers, args.batchsize, args.topk)
