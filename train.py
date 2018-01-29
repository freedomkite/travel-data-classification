# encoding:utf-8
import data_pro as pro
import numpy as np
import torch
import lstm
import torch.utils.data as D
from torch.autograd import Variable
import torch.nn.functional as F
import random
from sklearn import cross_validation

'''training data'''
train_data = pro.load_data('train_pad.txt')
word_dict = {'unk': 0}
word_dict = pro.build_dict(train_data, word_dict)
train_tag = pro.load_data('tag.txt')
tag_dict = {}
tag_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
# tag_dict=pro.build_dict(train_tag,tag_dict)

import argparse

parser = argparse.ArgumentParser(description='question classification')
parser.add_argument('-embed_dim', type=int, default=50)
parser.add_argument('-embed_num', type=int, default=len(word_dict))
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-hidden_size', type=int, default=100)
parser.add_argument('-batch_size', type=int, default=20)
parser.add_argument('-epochs', type=int, default=300)
parser.add_argument('-t_size', type=int, default=100)
parser.add_argument('-class_num', type=int, default=len(tag_dict))
parser.add_argument('-train', type=str, default='true')
parser.add_argument('-f', type=str)

args = parser.parse_args()
model = lstm.BiLSTM(args)
if torch.cuda.is_available():
    model = model.cuda()

print model
optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()


def prediction(out, y):
    predict = torch.max(out, 1)[1].long()
    correct = torch.eq(predict, y)
    acc = correct.sum().float() / float(correct.data.size()[0])
    return (acc * 100).cpu().data.numpy()[0], predict


###############################################################


# vectorize
def vector(sents, tags, word_dict, tag_dict):
    max_len = len(sents[0])
    sent_vec = np.zeros((len(sents), max_len), np.float32)
    # tag_vec = np.zeros((len(sents), len(tag_dict)), dtype=int)
    for i, (sent, tag) in enumerate(zip(sents, tags)):
        vec = [word_dict[w] if w in word_dict else 0 for w in sent]
        sent_vec[i, :] = vec
    tag_vec = [tag_dict[t[0]] for t in tags]

    return sent_vec, tag_vec


tmp_acc = 0

dev_data = train_data[-5000:]
dev_tag = train_tag[-5000:]
train_data = train_data[:-5000]
train_tag = train_tag[:-5000]

for epoch in range(args.epochs):
    acc = 0
    k = 0
    train_loss = 0
    for i in range(0, len(train_data), 64):
        if i + 64 < len(train_data):
            sent_vec, tag_vec = vector(train_data[i:i + 63], train_tag[i:i + 63], word_dict, tag_dict)
        else:
            sent_vec, tag_vec = vector(train_data[i:len(train_data)], train_tag[i:len(train_data)], word_dict, tag_dict)
        train = torch.from_numpy(sent_vec.astype(np.int64))
        y_tensor = torch.LongTensor(tag_vec)
        train_datasets = D.TensorDataset(data_tensor=train, target_tensor=y_tensor)
        train_dataloader = D.DataLoader(train_datasets, args.batch_size, shuffle=True, num_workers=2)
        for (x, y) in train_dataloader:
            x = Variable(x)
            y = Variable(y)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            out = model(x)
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pre, predic = prediction(out, y)
            acc += pre
            k += 1
            train_loss += loss
        # print train_loss/k
    print 'epoch:', epoch, 'train_acc:', acc / k, '%   loss:', train_loss.cpu().data.numpy()[0] / k
    dev_acc = 0
    dev_k = 0
    dev_loss = 0
    for i in range(0, len(dev_data), 64):
        if i + 64 < len(dev_data):
            sent_vec, tag_vec = vector(dev_data[i:i + 63], dev_tag[i:i + 63], word_dict, tag_dict)
        else:
            sent_vec, tag_vec = vector(dev_data[i:len(dev_data)], dev_tag[i:len(dev_data)], word_dict, tag_dict)
        train = torch.from_numpy(sent_vec.astype(np.int64))
        y_tensor = torch.LongTensor(tag_vec)
        train_datasets = D.TensorDataset(data_tensor=train, target_tensor=y_tensor)
        train_dataloader = D.DataLoader(train_datasets, args.batch_size, shuffle=True, num_workers=2)
        for (x, y) in train_dataloader:
            x = Variable(x)
            y = Variable(y)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            out = model(x)
            loss = loss_func(out, y)
            pre, predic = prediction(out, y)
            dev_acc += pre
            dev_k += 1
            dev_loss += loss
        # print dev_loss/k
    print 'epoch:', epoch, 'dev_acc:', dev_acc / dev_k, '%   loss:', dev_loss.cpu().data.numpy()[0] / dev_k
    if tmp_acc < dev_acc:
        tmp_acc = dev_acc
        test_data = pro.load_data('test_seg.txt')
        result = []

        print len(test_data)
        for line in test_data:
            sent_vec = np.array([[word_dict[w] if w in word_dict else 0 for w in line]])
            test = torch.from_numpy(sent_vec.astype(np.int64))
            x = Variable(test)
            out = model(x)
            # print out.size()
            predict = torch.max(out, 1)[1].long()
            # print predict
            # print out
            # print list(predict.cpu().data.numpy())
            result += list(predict.cpu().data.numpy())
        with open('test_result.txt', 'w') as f_w:
            new_label_dict = {}
            for key, value in tag_dict.items():
                new_label_dict[value] = key
            print len(result)
            for ind in result:
                # print ind,new_label_dict[ind],y_test[i],t_data[2][i]
                # f_test.write((label_dict.keys()[ind]+'\n').encode('utf-8'))
                f_w.write((new_label_dict[ind] + '\n').encode('utf-8'))

