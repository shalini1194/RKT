# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch import optim
import json

os.environ["CUDA_LAUNCH_BLOCKING"]=str(1)
def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()[0],seq.size()[1]
    subsequent_mask = torch.tril(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8))
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    #print(subsequent_mask)
    return subsequent_mask.to(torch.float)


class LSTMM(nn.Module):



    def __init__(self, args):
        super(LSTMM, self).__init__()
        self.ques_emb = nn.Parameter(F.normalize(torch.rand(args.num_questions, args.embedding_dim),dim=1))
        self.embedding_dim=args.embedding_dim
        self.time_emb = nn.Embedding(2*
                                     args.time_span+1, args.embedding_dim)
        # self.time_emb2 = nn.Embedding(args.time_span2 + 1, args.embedding_dim)
        self.emb_dropout = nn.Dropout(args.dropout)
        self.RNN = nn.RNN(args.embedding_dim,args.embedding_dim,dropout=args.dropout, batch_first=False)
        self.lin_layer = nn.Linear(2* args.embedding_dim, args.embedding_dim)
        self.lin_layer3 = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.lin_layer2 = nn.Linear(args.embedding_dim, 1)
        self.input_layer =nn.Linear(2*args.embedding_dim,args.embedding_dim)
        self.len = args.max_len
        self.dropout3 = nn.Dropout(args.dropout)
        self.dropout4 = nn.Dropout(args.dropout)
        self.dropout2 = nn.Dropout(args.dropout)
        self.dropout = nn.Dropout(args.dropout)


    def forward(self,hist_ques, corr_seq, time,time2,next,corr):
        # seq_len=torch.Tensor(seq_len).cuda()
        # seq_len, perm_idx = seq_len.sort(0, descending=True)
        mask_seq = (hist_ques != 0).to(torch.float).unsqueeze(-1)

        ques_corr= torch.cat([self.ques_emb[hist_ques], corr_seq.to(torch.float32).unsqueeze(-1).repeat(1, 1, self.embedding_dim)], dim=-1)

        ques_corr = self.dropout(ques_corr)
        ques_corr =self.input_layer(ques_corr)

        time = self.time_emb(time)
        # time2=self.time_emb2(time2)
        #input = torch.cat([hist_ques,hist_ques*time],dim=-1)
        input=ques_corr
        input =self.lin_layer3(input)


        #print (mask_seq.shape)
        #input = torch.cat([input,time], dim=-1)*mask_seq

        y,h = self.RNN(input)
        #y = y*mask_seq

        mask=get_subsequent_mask(next)


        #y=torch.sum(corr.unsqueeze(-1)*y.unsqueeze(2).repeat(1,1,corr.shape[1],1),2)
        y=y*mask_seq

        # print(seq_len)
        # print(target_ex.shape)
        # print(target_ex[:,seq_len-1])
        # print(z.shape)

        # print(y.shape)
        #print(time.shape)

        #y=torch.cat([y,time],dim=-1)

        target_ex = self.ques_emb[next]
        v = torch.cat([y, target_ex], dim=-1)*mask_seq

        #w=torch.cat([y,self.ques_emb[hist_ques]],dim=-1)*mask_seq


        y=self.lin_layer(v)
        y= self.dropout2(y)

        #y=nn.ReLU()(y)

        y=self.lin_layer2(y)
        y=self.dropout3(y)

        return nn.Sigmoid()(y),nn.Sigmoid()(v)




class TopicRNNModel(nn.Module):
    """
    双向RNN（GRU）建模题面
    """

    def __init__(self,args, dropout = 0.2):
        super(TopicRNNModel, self).__init__()

        #self.embedding = nn.Embedding(wcnt, emb_size, padding_idx=0)
        #if num_layers > 1:

        self.rnn = nn.GRU(args.embedding_dim, args.embedding_dim,
                              bidirectional=True,
                              dropout=dropout)
        self.output = nn.Linear(args.embedding_dim * 2,
                            args.embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(args.num_words, args.embedding_dim)
        # else:
        #     self.emb_size = topic_size // 2
        #     self.rnn = nn.GRU(emb_size, topic_size // 2, 1,
         #                     bidirectional=True)

    def forward(self, input, mask):
        #x = self.embedding(input)
        # print(x.size())
        # exit(0)
        # seq_len, perm_idx = seq_len.sort(0, descending=True)
        # input = input[perm_idx]
        input = self.embedding(input)
        #y=nn.utils.rnn.pack_padded_sequence(input,seq_len, batch_first=True)
        y,_ = self.rnn(input)
        y= y*mask
        #y, _ = nn.utils.rnn.pad_packed_sequence(y,batch_first=True)
        #y = y[perm_idx]

        y = self.output(y)
        y = self.dropout(y)
        z,_=torch.max(y,dim=1)

        return z
class GraphAttentionLayer(nn.Module):
    def __init__(self,args):
        super(GraphAttentionLayer, self).__init__()
        self.projection = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.lin_layer = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.target_layer = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.rel_emb = nn.Embedding(4,args.embedding_dim)
        self.dropout = args.dropout
        self.in_features = args.embedding_dim
        self.out_features = args.embedding_dim
        self.alpha = 1e-3

        self.W = nn.Parameter(torch.zeros(size=(self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.corr_emb = nn.Embedding(2,args.embedding_dim)
        self.final_layer = nn.Linear(2*args.embedding_dim,1)
        self.layer = nn.Linear(self.out_features,1)
        self.max_len = args.max_len-1


    def loss(self, seq,input, neigh, corrs, next):
        input1 = self.lin_layer(input)
        neigh1 = self.lin_layer(neigh)
        M =self.max_len
        subseq_mask = get_subsequent_mask(neigh)
        input2= input1.repeat(1,self.max_len,1)
        a_input = torch.cat([input2, (neigh1+self.corr_emb(corrs))], dim=-1)

        e = self.leakyrelu(torch.matmul(a_input, self.a))

        e = e.repeat(1, 1, self.max_len) * subseq_mask

        zero_vec = -9e15*torch.ones_like(e)

        attention = torch.where(seq.unsqueeze(-1) > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = attention.unsqueeze(-1)* input.unsqueeze(2).repeat(1,M,M,1)
        att= torch.sum(h_prime,dim=2)
        att = F.elu(att)
        result =torch.cat([att, self.lin_layer(next)],dim=-1)
        result= self.final_layer(result)
        result = nn.Sigmoid()(result)

        return result




# class QDP(nn.Module):
#     def __init__(self, args):
#         super(QDP, self).__init__()
#         print("*** Initializing the QDP model ***")
#
#         self.q_linear = nn.Linear(3*args.embedding_dim, args.embedding_dim)
#         self.k_linear = nn.Linear(args.embedding_dim , 1)
#         self.v_linear = nn.Linear(3*args.embedding_dim, args.embedding_dim)
#         self.relation_emb =nn.Embedding(num_relations, args.embedding_dim)
#         self.lin_layer = nn.Linear(args.embedding_dim, args.embedding_dim)
#         self.lin_layer2 = nn.Linear(args.embedding_dim, args.embedding_dim)
#         self.max_len = max_neigh+max_concept
#
#
#         print("*** JODIE initialization complete ***\n\n")
#
#     def forward(self,target_ques, neighbors, relations,mask_neigh, concept, mask_concept):
#
#         relations = torch.cat([torch.LongTensor(relations).cuda(), torch.ones([concept.shape[0],concept.shape[1]], dtype= torch.long).cuda()*3],dim=1)
#         mask = torch.Tensor(np.concatenate((mask_neigh,mask_concept),axis=-1)).cuda().unsqueeze(-1)
#         q = torch.cat([neighbors,concept],dim=1)
#         r = self.relation_emb(relations)
#         neighbor_emb = torch.cat([q, r],dim=-1)*mask
#         target_ques_emb = target_ques
#         target_ques_emb2 = target_ques_emb.unsqueeze(1).repeat(1,self.max_len, 1)
#         new_ques_emb = torch.cat(([target_ques_emb2, neighbor_emb]),dim=-1)*mask
#         new_ques_emb = self.q_linear(new_ques_emb)
#         new_ques_emb = self.k_linear(new_ques_emb).squeeze(-1)
#         new_ques_emb = nn.LeakyReLU()(new_ques_emb)
#         new_ques_emb[new_ques_emb==0] = 1e-9
#         new_ques_emb = nn.Softmax()(new_ques_emb)
#         new_ques_emb = new_ques_emb.unsqueeze(-1)*neighbor_emb
#         new_ques_emb = torch.sum(new_ques_emb, dim=1)
#         new_ques_emb = torch.cat([new_ques_emb, target_ques_emb],dim=-1)
#         new_ques_emb = self.v_linear(new_ques_emb)
#
#         return new_ques_emb

# class DGAT(nn.Module):
#     def __init__(self, args):
#         super(DGAT, self).__init__()
#
#         print("*** Initializing the JODIE model ***")
#         self.q_linear = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
#         self.k_linear = nn.Linear(args.embedding_dim, 1)
#         self.max_len = args.max_len
#         self.v_linear = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
#         self.Pos = nn.Parameter(torch.Tensor(args.max_len-1,args.embedding_dim))
#         self.lin_layer = nn.Linear(2*args.embedding_dim, args.embedding_dim)
#         self.lin_layer2 = nn.Linear(args.embedding_dim, args.embedding_dim)
#         self.lin_layer3 = nn.Linear(args.embedding_dim, 1)
#
#         print("*** JODIE initialization complete ***\n\n")
#
#
#     def forward(self, neighbors,next_ques,mask):
#         '''
#
#         :param q: [batch, 1, emb_size]
#         :param k: [batch, 50,emb_size]
#         :param v:[batch,50,emb_size]
#         :param d_k:
#         :param time:
#         :param type:
#         :return:
#         '''
#         mask = torch.Tensor(mask).cuda().unsqueeze(-1)
#         neighbors =self.lin_layer(neighbors)
#         neighbor_emb =(neighbors+ self.Pos.unsqueeze(0))*mask #(B,n,emb)
#
#         # new_ques_emb = torch.cat(([next_ques.unsqueeze(1).repeat(1,self.max_len-1,1,1),
#         #                            neighbor_emb.unsqueeze(2).repeat(1,1,self.max_len-1,1)]), dim=-1)
#         new_ques_emb = torch.matmul(next_ques,torch.transpose(neighbor_emb,1,2))
#         new_ques_emb = new_ques_emb / (neighbor_emb.shape[-1] ** 0.5)
#         #new_ques_emb = self.q_linear(new_ques_emb)
#         #new_ques_emb = self.k_linear(new_ques_emb).squeeze(-1)#32*49*49
#         diag_vals = np.ones((new_ques_emb.shape[1],new_ques_emb.shape[2]))  # (T_q, T_k)
#         tril = np.tril(diag_vals) # (T_q, T_k)
#         mask_tril = torch.Tensor(tril).cuda().unsqueeze(0).repeat(new_ques_emb.shape[0],1,1)
#         # (h*N, T_q, T_k)
#
#
#         #new_ques_emb = nn.LeakyReLU()(new_ques_emb)*mask_tril
#         new_ques_emb = new_ques_emb*mask_tril
#         new_ques_emb[new_ques_emb == 0] = 1e-9
#         new_ques_emb = nn.Softmax()(new_ques_emb)
#         # neighbor_emb1 = neighbor_emb.transpose(1,2).contiguous().view(neighbor_emb.shape[0]*neighbor_emb.shape[2],-1).transpose(0,1)#B*64
#         # new_ques_emb1 = new_ques_emb.contiguous().view(-1,new_ques_emb.shape[-1]) #B*49
#         new_ques_emb2 = torch.bmm(new_ques_emb ,neighbor_emb)
#         #new_ques_emb = new_ques_emb2.view(neighbor_emb.shape)
#         #new_ques_emb = torch.sum(new_ques_emb, dim=1).squeeze(1)
#         new_ques_emb2 = self.lin_layer3(new_ques_emb2)
#         new_ques_emb2 = nn.Sigmoid()(new_ques_emb2)
#         return new_ques_emb2
#
#
