#encoding:utf-8
import data_pro as pro
import numpy as np
import torch
import lstm
import torch.utils.data as D
from torch.autograd import Variable
import torch.nn.functional as F
import random

'''training data'''
train_data=pro.load_data('train_pad.txt')
word_dict={'unk':0}
word_dict=pro.build_dict(train_data,word_dict)
train_tag=pro.load_data('tag.txt')
tag_dict={}
tag_dict=pro.build_dict(train_tag,tag_dict)

import argparse
parser=argparse.ArgumentParser(description='question classification')
parser.add_argument('-embed_dim',type=int,default=50)
parser.add_argument('-embed_num',type=int,default=len(word_dict))
parser.add_argument('-dropout',type=float,default=0.4)
parser.add_argument('-hidden_size',type=int,default=100)
parser.add_argument('-batch_size',type=int,default=20)
parser.add_argument('-epochs',type=int,default=300)
parser.add_argument('-t_size',type=int,default=100)
parser.add_argument('-class_num',type=int,default=len(tag_dict))
parser.add_argument('-train',type=str,default='true')


args=parser.parse_args()
model=lstm.BiLSTM(args)
if torch.cuda.is_available():
    model=model.cuda()

print model
optimizer = torch.optim.Adam(model.parameters())
loss_func=torch.nn.CrossEntropyLoss()


def prediction(out, y):
    predict = torch.max(out, 1)[1].long()
    correct = torch.eq(predict, y)
    acc = correct.sum().float() / float(correct.data.size()[0])
    return (acc * 100).cpu().data.numpy()[0],predict
###############################################################


#vectorize
def vector(sents,tags,word_dict,tag_dict):
    max_len=len(sents[0])
    sent_vec=np.zeros((len(sents),max_len),np.float32)
    #tag_vec = np.zeros((len(sents), len(tag_dict)), dtype=int)
    for i,(sent,tag) in enumerate(zip(sents,tags)):
        vec=[word_dict[w] if w in word_dict else 0 for w in sent]
        sent_vec[i,:]=vec
    tag_vec = [tag_dict[t[0]] for t in tags]


    return sent_vec,tag_vec

for epoch in range(args.epochs):
    acc=0
    k=0
    train_loss=0
    for i  in range(0,len(train_data),64):
        if i+64<len(train_data):
            sent_vec,tag_vec=vector(train_data[i:i+63],train_tag[i:i+63],word_dict,tag_dict)
        else:
            sent_vec, tag_vec = vector(train_data[i:len(train_data)], train_tag[i:len(train_data)], word_dict, tag_dict)
        train = torch.from_numpy(sent_vec.astype(np.int64))
        y_tensor = torch.LongTensor(tag_vec)
        train_datasets = D.TensorDataset(data_tensor=train, target_tensor=y_tensor)
        train_dataloader = D.DataLoader(train_datasets, args.batch_size, shuffle=True, num_workers=2)
        for (x, y) in train_dataloader:
            x=Variable(x)
            y=Variable(y)
            if torch.cuda.is_available():
                x=x.cuda()
                y=y.cuda()
            out = model(x)
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pre, predic = prediction(out, y)
            acc += pre
            k+=1
            train_loss+=loss
        print train_loss/k
    print 'epoch:', epoch, 'test_acc:', acc /k, '%   loss:', train_loss.cpu().data.numpy()[0] / k











'''


if args.train=='true':
    output = open('test.log', 'w+')
    output.write('-' * 50 + '\n')
    output.flush()
    max_acc = 0.0
    step = 0
    for i in range(args.epochs):
        print "epochs:",i
        acc=0.0
        l=0.0
        k=0
        for (x_cat, y) in train_dataloader:
            x,pos,y = data_unpack(x_cat, y)
            # print 'x:', x, 'pos:', pos
            #print y
            out = model1(x, pos)
            #print out.data,y
            loss = loss_func(out, y)
            l += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print loss,prediction(out,y)
            pre,_=prediction(out,y)
            acc+=pre
            k+=1
        print 'epoch:', i, 'acc:', acc / k, '%   loss:', l.cpu().data.numpy()[0] / k
        test_acc=0.0
        j=0
        test_l=0.0
        result=[]
        y_test=[]
        for (t_x_cat,t_y) in test_dataloader:
            t_x,t_pos,t_y=data_unpack(t_x_cat,t_y)
            t_out=model1(t_x,t_pos)
            loss = loss_func(t_out, t_y)
            test_l += loss
            t_pre,predic=prediction(t_out,t_y)
            #print 'predic:',predic.data,'t_y:',t_y.data
            test_acc+=t_pre
            j+=1
            result+=list(predic.cpu().data.numpy())
            y_test+=list(t_y.cpu().data.numpy())
        print 'epoch:', i, 'test_acc:', test_acc / j, '%   loss:', test_l.cpu().data.numpy()[0] / j
        #print label_dict.items()
        new_label_dict={}
        for key,value in label_dict.items():
            new_label_dict[value]=key
        output.write('epoch:'+str(i)+'test_acc:'+str(test_acc / j)+ '%   loss:'+str(test_l.cpu().data.numpy()[0] / j)+'\n')
        output.flush()

        if test_acc/j>max_acc:
            step=i
            max_acc=test_acc/j
            torch.save(model1.state_dict(),'model.pt')
            with open('result.txt','w') as f_test:
                print 'max_acc:---------------------------------------------------------------',max_acc


                for ind in result:
                    #print ind,new_label_dict[ind],y_test[i],t_data[2][i]
                    #f_test.write((label_dict.keys()[ind]+'\n').encode('utf-8'))
                    f_test.write((new_label_dict[ind]+'\n').encode('utf-8'))
        if i>=args.epochs-1:
            output.write('max_acc:'+'-'*40+str(max_acc)+'\n')
            output.flush()
            output.close()
if args.train=='false':
    model1.load_state_dict(torch.load('model.pt'))
    test_acc = 0.0
    j = 0
    test_l = 0.0
    result = []
    y_test = []
    for (t_x_cat, t_y) in test_dataloader:
        t_x, t_pos, t_y = data_unpack(t_x_cat, t_y)
        t_out = model1(t_x, t_pos)
        loss = loss_func(t_out, t_y)
        test_l += loss
        t_pre, predic = prediction(t_out, t_y)
        # print 'predic:',predic.data,'t_y:',t_y.data
        test_acc += t_pre
        j += 1
        result += list(predic.cpu().data.numpy())
        y_test += list(t_y.cpu().data.numpy())
    print 'test_acc:', test_acc / j, '%   loss:', test_l.cpu().data.numpy()[0] / j
    with open('test_result.txt','w') as f_w:
        new_label_dict = {}
        for key, value in label_dict.items():
            new_label_dict[value] = key
        for ind in result:
            # print ind,new_label_dict[ind],y_test[i],t_data[2][i]
            # f_test.write((label_dict.keys()[ind]+'\n').encode('utf-8'))
            f_w.write((new_label_dict[ind] + '\n').encode('utf-8'))


'''




















    #
    #
    # for x_i,p_i,y_i in zip(list(x),list(pos),y):
    #     #print 'x:', x_i, 'pos:', p_i
    #     x_i = Variable(torch.from_numpy(x_i))
    #     p_i = Variable(torch.from_numpy(p_i))
    #     #print y_i
    #     y_i = Variable(torch.LongTensor(torch.from_numpy(y_i)))
    #     #print y_i
    #     out=model1(x_i,p_i)
    #     #print out,y_i
    #     #print out,y_i
    #     loss = loss_func(out, y_i)
    #     l+=loss
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     k+=1
    #     #print k
    #     #print loss.data
    #     #print out
    #     print torch.max(out,1)
    #     predict = torch.max(out, 1)[1].long()
    #
    #     print predict
    #     correct = torch.eq(predict, y_i)
    #     #acc = correct.sum().float() / float(correct.data.size()[0])
    #
    # print l/float(k)






