import os
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
from time import time
from time import strftime
from time import localtime
import logging

from model import MF, MLP


def parse_args():
    parser = argparse.ArgumentParser(description="Latent Space Mapping")
    parser.add_argument('--dataset', type=str, default='', help="d")
    parser.add_argument('--dim', type=int, default=16, help="embedding")
    parser.add_argument('--layers', type=int, default=1, help="l")
    parser.add_argument('--lr', type=float, default=0.001, help="lr")
    parser.add_argument('--reg', type=float, default=0, help="r")
    parser.add_argument('--epochs', type=int, default=100, help='e')
    parser.add_argument('--batchsize', type=int, default=2, help="b")
    return parser.parse_args()


def default_args():
    return 'Musical_Patio'


def overlap_user(dataset):
    dp = 'data/%s/test.txt' % dataset
    with open(dp, 'r') as f:
        l = [int(line.strip().split()[0]) for line in f.readlines()]
    return max(l)+1


def load_data(dataset, mode):
    dp = 'data/%s/%s.csv' % (dataset, mode)
    data = pd.read_csv(dp, ',', names=['u', 'i', 'r', 't'], engine='python')
    data.sort_values(by=['u', 'i'], inplace=True)
    n_user = data['u'].max() + 1
    n_item = data['i'].max() + 1
    return n_user, n_item


def load_model(dataset, dim):
    sp = 'pretrain/%s/MF_s.pth.tar' % dataset
    tp = 'pretrain/%s/MF_t.pth.tar' % dataset
    s_user, s_item = load_data(dataset, 's')
    t_user, t_item = load_data(dataset, 't')
    mf_s = MF(s_user, s_item, dim)
    mf_t = MF(t_user, t_item, dim)
    mf_s.load_state_dict(torch.load(sp))
    mf_t.load_state_dict(torch.load(tp))
    return mf_s, mf_t


def batch_user(n_user, batch_size):
    for i in range(0, n_user, batch_size):
        yield list(range(i, min(i+batch_size, n_user)))


def train(mapping, opt, mse_loss, u, y):
    out = mapping(u)
    loss = mse_loss(out, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.cpu().item()


def main(dataset, dim, layers, lr, reg, epochs, batchsize):
    n_user = overlap_user(dataset)
    print(n_user)
    logging.info(str(n_user))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mf_s, mf_t = load_model(dataset, dim)
    mapping = MLP(dim, layers)
    mf_s = mf_s.to(device)
    mf_t = mf_t.to(device)
    mapping = mapping.to(device)
    opt = torch.optim.Adam(mapping.parameters(), lr=lr, weight_decay=reg)
    mse_loss = nn.MSELoss()

    start = time()
    for epoch in range(epochs):
        loss_sum = 0
        for users in batch_user(n_user, batchsize):
            us = torch.tensor(users).long()
            us = us.to(device)
            u = mf_s.get_embed(us)
            y = mf_t.get_embed(us)
            loss = train(mapping, opt, mse_loss, u, y)
            loss_sum += loss
        print('Epoch %d [%.1f] loss = %f' % (epoch, time()-start, loss_sum))
        logging.info('Epoch %d [%.1f] loss = %f' %
                     (epoch, time()-start, loss_sum))
        start = time()

    mfile = 'pretrain/%s/Mapping.pth.tar' % dataset
    torch.save(mapping.state_dict(), mfile)
    print('save [%.1f]' % (time()-start))
    logging.info('save [%.1f]' % (time()-start))


if __name__ == '__main__':
    args = parse_args()
    # args.dataset = default_args()
    log_dir = "log/%s/" % args.dataset
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, "%s_%s_%s_%s" % (
        args.dataset, args.layers, args.dim, strftime('%Y-%m-%d--%H-%M-%S', localtime()))), level=logging.INFO)
    main(args.dataset, args.dim, args.layers, args.lr,
         args.reg, args.epochs, args.batchsize)
