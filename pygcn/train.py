from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from pygcn.utils import load_data, accuracy, ThreeDMatchGraphData, ThreeDMatchGraphTestData, load_test_data
from pygcn.models import GCN


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def train(epoch):
    t = time.time()
    model.train()
    epoch_loss_train = []
    epoch_acc_train = []
    predict_num = []
    count = 0
    for adj, features, labels, _ in dataloader:
        optimizer.zero_grad()
        features = features[0]
        labels = labels[0]
        adj = adj[0]
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        output = model(features, adj)
        prediction = output.data.max(1)[1]
        predict = prediction.data.cpu().numpy()
        idx = np.where(predict == 1)[0]
        loss_train = F.nll_loss(output, labels)
        # loss_train = FocalLoss(gamma=0)(output, labels)
        acc_train = accuracy(output, labels)
        loss_train.backward()
        optimizer.step()
        epoch_loss_train.append(loss_train.item())
        epoch_acc_train.append(acc_train.item())
        predict_num.append(len(idx))
        count+=1
        # if count % 100 == 0:
        #     print("loss_train: ", np.mean(epoch_loss_train))
        #     print("acc_train: ", np.mean(epoch_acc_train))
        #     print("mean_predict_num: ", np.mean(predict_num))

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
    print("mean_predict_num: ", np.mean(predict_num))
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(np.mean(epoch_loss_train)),
          'acc_train: {:.4f}'.format(np.mean(epoch_acc_train)),
          'time: {:.4f}s'.format(time.time() - t))

def set_model():
    model = GCN(nfeat=4,
                nhid=32,
                nclass=2,
                dropout=0.3)
    checkpoint = torch.load('./pygcn/models/direction_distance_checkpoint_20.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    return model

def get_predict_points(content_path, cite_path, model):
    adj, features, labels, inv_map = load_test_data(content_path, cite_path)
    features = features.cuda()
    adj = adj.cuda()
    output = model(features, adj)
    acc_train = accuracy(output, labels)
    prediction = output.data.max(1)[1]
    predict = prediction.data.cpu().numpy()
    idx = np.where(predict == 1)[0]
    points = []
    for i in idx:
        points.append(inv_map[i])
    return points, acc_train





if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    adj, features, labels = load_data()

    # Model and optimizer
    model = GCN(nfeat=4,
                # nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=2,
                # nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    # if args.cuda:
    #     model.cuda()
    #     features = features.cuda()
    #     adj = adj.cuda()
    #     labels = labels.cuda()

    # Train model
    t_total = time.time()
    # dataloader = DataLoader(ThreeDMatchGraphData("./dataset/threedmatch_graph/train/",50 , 'train'), shuffle=True, pin_memory= False)
    # load_epoch = 20
    # checkpoint = torch.load('./pygcn/models/checkpoint_{}.pth'.format(load_epoch))
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    files = []
    gt_idx = 0
    with open('./features_tmp/list.txt') as f:
        sets = f.readlines()
        sets = [x.strip().split() for x in sets]
    for s in sets:
        set_name = s[0]
        with open("./dataset/threedmatch_test/" + set_name + "-evaluation/LoMatch_gt_overlap.log") as f:
            lines = f.read().splitlines()
        pts_num = int(s[1])
        matching_pairs = []
        for i in range(pts_num):
            for j in range(i + 1, pts_num):
                matching_pairs.append([i, j, pts_num])
        for m in matching_pairs:
            ratio_success = 0
            while ratio_success != 1:
                overlap_gt = lines[gt_idx].split(',')
                if m[0] == int(overlap_gt[0]) and m[1] == int(overlap_gt[1]):
                    ratio = float(overlap_gt[2])
                    ratio_success = 1
                gt_idx += 1
            if ratio > 0.1:
                files.append(set_name + '_%03d_%03d' % (m[0], m[1]))
    dataloader = DataLoader(ThreeDMatchGraphTestData("./dataset/threedmatch_graph/test_v_dd/", files), shuffle=True)
    if args.cuda:
        model.cuda()
    #
    for epoch in range(args.epochs):
        train(epoch)
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, './pygcn/models/direction_distance_checkpoint_{}.pth'.format(epoch))
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    files = []
    with open('./features_tmp/list.txt') as f:
        sets = f.readlines()
        sets = [x.strip().split() for x in sets]
    for s in sets:
        set_name = s[0]
        pts_num = int(s[1])
        matching_pairs = []
        for i in range(pts_num):
            for j in range(i + 1, pts_num):
                matching_pairs.append([i, j, pts_num])
        for m in matching_pairs:
            files.append(set_name + '_%03d_%03d' %(m[0], m[1]))
    # dataloader = DataLoader(ThreeDMatchGraphTestData("./dataset/threedmatch_graph/test/", files))
    # dataloader = DataLoader(ThreeDMatchGraphData("./dataset/threedmatch_graph/valid/",50, 'valid'))
    # evaluation
    before_ratios = []
    after_ratios = []
    mean_true_num = []
    with open("./dataset/threedmatch_test/" + sets[0][0] + "-evaluation/LoMatch_gt_overlap.log") as f:
        lines = f.read().splitlines()
    gt_idx = 0
    acc = []
    for adj, features, labels in dataloader:
        overlap_gt = float(lines[gt_idx].split(',')[2])
        gt_idx+=1
        features = features[0]
        labels = labels[0]
        adj = adj[0]
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        # output = model(features, adj)
        # prediction = output.data.max(1)[1]
        np_labels = labels.cpu().numpy()
        # np_predict = prediction.data.cpu().numpy()
        pos_num = np.count_nonzero(np_labels)
        neg_num = len(np_labels) - pos_num
        before_ratios.append(pos_num)
        after_ratios.append(neg_num)
        # acc_test = accuracy(output, labels)
        # acc.append(acc_test.item())
        # # print(overlap_gt, pos_num / ( pos_num + neg_num ))
        # true_positive = 0
        # false_positive = 0
        #
        # for i in range(len(np_labels)):
        #     if np_labels[i] == 1 and np_predict[i] == 1:
        #         true_positive += 1
        #     if np_labels[i] == 0 and np_predict[i] == 1:
        #         false_positive += 1
        # if overlap_gt < 0.1:
        #     continue
        # if true_positive + false_positive == 0:
        #     continue
        # # print(pos_num, neg_num, true_positive, false_positive)
        # before_ratio = pos_num / (pos_num + neg_num)
        # after_ratio = true_positive / (true_positive + false_positive)
        # before_ratios.append(before_ratio)
        # after_ratios.append(after_ratio)
        # mean_true_num.append(true_positive)
        # if len(mean_true_num) % 100 == 0:
        #     print(len(mean_true_num))
    # print(np.mean(acc))
    print(np.mean(before_ratios))
    print(np.mean(after_ratios))
    # print(np.mean(mean_true_num))


