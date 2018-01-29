#encoding:utf-8
import torch
import  torch.nn as nn
import torch.nn.functional as F
from torch.autograd import  Variable
torch.manual_seed(1)

class BiLSTM(nn.Module):
    def __init__(self,args):
        super(BiLSTM,self).__init__()
        # self.args=args
        # self.embed=nn.Embedding(args.embed_num, args.embed_dim)
        # self.pos_embedding = nn.Embedding(args.pos_size, args.pos_dim)
        #self.dropout = nn.Dropout(args.dropout)
        #
        # self.lstm = nn.LSTM(args.input_size, args.hidden_size,
        #                     bidirectional=False,
        #                     batch_first=False)
        #                     #dropout=args.dropout_rnn)
        #
        # self.myw = Variable(torch.randn(args.max_len,args.hidden_size), requires_grad=True)
        # self.linearOut = nn.Linear(args.max_len, args.class_num)
        # self.softmax = nn.Softmax()
        self.args = args
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        #self.dropout = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(args.embed_dim, args.hidden_size,
                            bidirectional=True,
                            batch_first=True,
                            dropout=args.dropout)
        # dropout=args.dropout_rnn)
        # self.myw = Variable(torch.randn(args.hidden_size * 2,args.hidden_size*2), requires_grad=True).cuda()
        self.linearOut = nn.Linear(args.hidden_size*2, args.class_num)
        self.softmax = nn.Softmax()
    def forward(self,x,):
        if torch.cuda.is_available():
            hidden = (Variable(torch.zeros(2, x.size(0), self.args.hidden_size)).cuda(),
                      Variable(torch.zeros(2, x.size(0), self.args.hidden_size)).cuda())
        else:
            hidden = (Variable(torch.zeros(2, x.size(0), self.args.hidden_size)),
                      Variable(torch.zeros(2, x.size(0), self.args.hidden_size)))
        x = self.embed(x)
        #x=self.dropout(x)
        x, lstm_h = self.lstm(x, hidden)
        x = F.tanh(x)
        # for idx in range(x.size(0)):
        #     h= torch.mm(x[idx], self.myw)
        #     if idx == 0:
        #         output = torch.unsqueeze(h, 0)
        #     else:
        #         output = torch.cat([output, torch.unsqueeze(h, 0)], 0)
        x=torch.transpose(x,1,2)
        x = F.max_pool1d(x, x.size(2))
        #print x.data.size()
        x = self.linearOut(x.view(x.size(0),1,-1))
        x = self.softmax(x.view(x.size(0),-1))

        return x



