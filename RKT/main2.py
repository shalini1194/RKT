# -*- coding: utf-8 -*-

"""
执行模型训练、测试等
"""
# from comet_ml import Experiment
# experiment = Experiment(api_key="MCyyqYKWkKKMm0O15B1BadVOg",
  #                      project_name="general", workspace="shalini1194")
import argparse
import numpy as np
import json
import torch.optim as optim
np.random.seed(0)
import os
from SAKT_model2 import  *


from torch import autograd
from numpy import savetxt
import torch
from torch.nn import functional as F
import torch.nn as nn
import random
import csv

import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from collections import  defaultdict
#crossEntropyLoss = nn.CrossEntropyLoss()
seed=1194
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
#PYTHONHASHSEED=seed
pre_req = defaultdict(list)
os.environ['CUDA_LAUNCH_BLOCKING']=str(1)


def computeRePos(time_seq, time_span):
    size = time_seq.shape[0]
    #print(time_seq.shape)
    #ßtime_diff = abs(time_seq[1:]-time_seq[:-1])
    # print(np.log2(np.max(time_diff)))
    # time_diff2=[]
    # ques_pre ={}
    # for j in range(1,len(time_seq)):
    #     if time_seq[j] in ques_pre.keys():
    #         time_diff2.append(time_seq[j]-time_seq[ques_pre[time_seq[j]]])
    #     else:
    #         time_diff2.append(0)
    #     ques_pre[time_seq[j]]=j

    # if len(np.nonzero(time_diff))==0:
    #     time_scale=1
    # else:
    #     time_scale = np.min(time_diff[np.nonzero(time_diff)])
    #     # print(time_scale)
    #     if time_scale == 0:
    #         time_scale = 1
    #
    time_matrix= (np.abs(np.repeat(np.expand_dims(time_seq, axis=-1),size,-1).reshape((size*size,1)) - \
                 np.repeat(np.expand_dims(time_seq,axis=0),size,axis=0).reshape((size*size,1))))
    #print(np.max(time_matrix))
    # #print(time_matrix[time_matrix<0])
    # #print(time_matrix)
    time_matrix[time_matrix>time_span] = time_span
    time_matrix = time_matrix.reshape((size,size))
    #print(time_matrix.nonzero())
    return (time_matrix)

def compute_corr(prob_seq,corr_dic):
    corr= np.zeros((prob_seq.shape[0]-1,prob_seq.shape[0]-1))
    for i in range(1,prob_seq.shape[0]):
        for  j in range(i):

            corr[i-1][j]=corr_dic[prob_seq[i]][prob_seq[j]]
    return corr


def one_hot(indices, depth):
    encoding = np.concatenate((np.eye(depth), [np.zeros(depth)]))
    return encoding[indices]
def process_problems_and_corrects( problem_seqs, correct_seqs, time_seq,max_num_problems, timespan):
    """
    This function aims to process the problem sequence and the correct sequence into a DKT feedable X and y.
    :param problem_seqs: it is in shape [batch_size, None]
    :param correct_seqs: it is the same shape as problem_seqs
    :return:
    """
    x_problem_seqs = problem_seqs[:, :-1]
    x_correct_seqs = correct_seqs[:, :-1]


    y_problem_seqs = problem_seqs[:, 1:]
    y_correct_seqs = correct_seqs[:, 1:]


    result = (x_problem_seqs, x_correct_seqs,y_problem_seqs, y_correct_seqs)
    return result


def read_data_from_csv_file(fileName, file2,max_num_problems,num_questions, timespan):

    f4 = open(file2, 'r')
    corr_dic=defaultdict(lambda:defaultdict(lambda :0))

    for i in f4:
        i_=i.strip().split(',')
        corr_dic[int(i_[0])][int(i_[1])]=float(1)

    #print('here')
    rows = []
    prob_seq ,corr_seq, concept_seq,time_seq, time_seq2,rel_seq= [],[],[],[],[],[]

    with open(fileName, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)

    index = 0
    max2,max3=0,0
    while (index < len(rows) - 1):
        #print(index)
        problems_num = int(len(rows[index + 1]))

        if (problems_num <1):
            index += 4
            continue
        if problems_num>100:
            index+=4
            continue
        if problems_num <= max_num_problems:
            problems = np.zeros(max_num_problems,int)
            correct = np.zeros(max_num_problems,int)
            time = np.zeros((max_num_problems,max_num_problems))

            time2=  np.zeros((max_num_problems, max_num_problems))

            problems[-len(rows[index+1]):] = [int(i) for i in rows[index + 1]]
            correct[-len(rows[index+2]):] =  [round(float(i)) for i in rows[index + 2]]

            time_raw = [float(i) for i in rows[index + 3]]

            time[-len(rows[index+3]):,-len(rows[index+3]):] = computeRePos(np.array(time_raw), timespan)
            # if len(rows[index+2])>1:
            #     time[-(len(rows[index+3])-1):] = computeRePos(np.array(time_raw), timespan)
            corr=compute_corr(problems,corr_dic)

            prob_seq.append(problems)
            corr_seq.append(correct)
            time_seq.append(time)
            rel_seq.append(corr)
            #time_seq2.append(np.round(time2))

        else:
            start_idx = 0
            while max_num_problems + start_idx <= problems_num:

                problems = [int(i) for i in rows[index + 1][start_idx:max_num_problems + start_idx]]
                correct =  [round(float(i)) for i in rows[index + 2][start_idx:max_num_problems + start_idx]]

                time_raw = [float(i) for i in rows[index + 3][start_idx:max_num_problems + start_idx]]

                time = computeRePos(np.array(time_raw), timespan)
                #print("Time Re pos")
                corr = compute_corr(np.array(problems), corr_dic)
                #print(corr.shape)

                # print("corr")
                start_idx += max_num_problems

                prob_seq.append(problems)
                corr_seq.append(correct)
                time_seq.append(time)
                rel_seq.append(corr)
                #time_seq2.append(np.round(time2))

        max2=max(max2,np.max(time))
        # print(time)
        # print(np.max(time))
       # # max3 = max(max3, np.max(time2))

        index += 4
    #f2.close()
    print("Processing Data")
    result= process_problems_and_corrects( np.array(prob_seq), np.array(corr_seq), np.array(time_seq), num_questions, timespan)



    return result,rel_seq,np.array(time_seq),max2
    #return result,np.array(time_seq), np.array(time_seq2), concept_seq, np.array(seq_len)

def read_relations_file( file2,neg_occ,num_questions,cnt):

    neighbor=defaultdict(list)
    neg_neighbor = defaultdict(list)
    relations=defaultdict(list)
    concept = defaultdict(list)
    max_neigh,max_concept,min_neigh=1200,0,100000

    # for i in f:
    #     #print(i)
    #     i_=i.strip().split(',')
    #
    #     neighbor[i_[0]].append(i_[1])
    #     relations[i_[0]].append(1)
    #     if len(neighbor[i_[0]])>max_neigh:
    #         max_neigh= len(neighbor[i_[0]])
    f=open(file2,'r')

    for i in f:
        i_ = i.strip().split(',')
        if len(neighbor[i_[0]])<max_neigh:
            neighbor[i_[0]].append(i_[1])
            relations[i_[0]].append(2)

    f.close()
    f=open(neg_occ,'r')
    for i in f:
        i_ = i.strip().split(',')
        if len(neg_neighbor[i_[0]])<max_neigh:
            neg_neighbor[i_[0]].append(i_[1])


    f.close()

    concept2que={}

    neighbor_ret = np.zeros((num_questions,max_neigh))
    negative_neig_ret = np.zeros((num_questions, max_neigh))
    relations_ret = np.zeros((num_questions,max_neigh))
    concept_ret = np.zeros((num_questions))
    mask_neigh =np.zeros((num_questions, max_neigh))
    mask_concept = np.zeros((num_questions, max_concept))
    score_ret= np.zeros((num_questions,1))

    conceptid = 1
    for i, v in concept2que.items():
        for j in v:
            concept_ret[j] = conceptid
        conceptid+=1

    score = defaultdict(lambda :0)
    # f2 = open(score_file, 'r')
    # for i in f2:
    #     i_ = i.strip().split(',')
    #     score[int(i_[0])] = i_[1]

    for i in range(num_questions):
        #score_ret[i]=float(score[i])
        if len(neighbor[str(i)])==0:
            continue
        neighbor_ret[i][:len(neighbor[str(i)])]=neighbor[str(i)]
        negative_neig_ret[i][:len(neg_neighbor[str(i)])]=neg_neighbor[str(i)]
        cnt_neg=len(neg_neighbor[i])

        while cnt_neg<max_neigh:
            j=random.randint(1,num_questions-1)
            if j in neighbor_ret[i]:
                continue
            if j in negative_neig_ret[i]:
                continue
            negative_neig_ret[i][cnt_neg]=j
            cnt_neg+=1

        relations_ret[i][:len(relations[str(i)])]=relations[str(i)]

        mask_neigh[i][:len(neighbor[str(i)])]=1
    np.save('neighbor'+str(cnt),neighbor_ret)
    np.save('neg_neighbor1'+str(cnt), negative_neig_ret)
    np.save('relation1'+str(cnt), relations_ret)
    np.save('mask_neigh1'+str(cnt), mask_neigh)

    neighbor_ret=np.load('neighbor'+str(cnt)+'.npy')
    negative_neig_ret =np.load('neg_neighbor1'+str(cnt)+'.npy')
    relations_ret = np.load('relation1'+str(cnt)+'.npy')
    mask_neigh = np.load('mask_neigh1'+str(cnt)+'.npy')
    max_neigh=1200
    return neighbor_ret, negative_neig_ret,relations_ret, mask_neigh,max_neigh
    #return neighbor_ret[3:835], relations_ret[3:835], concept_ret[3:835], mask_neigh[3:835], mask_concept[3:835], max_neigh,max_concept
    # item2con = {}
    # item2con[0]=0
    # for i in f:
    #     i_=i.strip().split(' ')
    #
    #     item2con[int(i_[0])]= int(i_[1])-num_questions+1
    # item2con[1]=1
    # return item2con

def build_relation_graph(file, num_questions):
    f = open(file,'r')
    ques_pre = []
    neg_ques_pre = []
    ques_dict=defaultdict(list)
    relation_dict=defaultdict(list)
    maxlen=0
    for i in f:
        i_=i.strip().split(',')
        ques_pre.append([int(i_[0]),int(i_[1]),int(i_[2])])
        ques_dict[int(i_[0])].append(int(i_[1]))
        relation_dict[int(i_[0])].append(int(i_[2]))
        maxlen = max(maxlen, len(ques_dict[int(i_[0])]))
    exercises=[]
    for i in ques_pre:
        exercise = random.randint(1,num_questions-1)
        while 1:
            if exercise not in ques_dict[i[0]] or exercise != i[0]:
                break
            exercise =random.randint(1,num_questions-1)
        exercises.append(exercise)

    ques_pre =np.array(ques_pre)
    #print(pre_req)
    ques_list=np.zeros((num_questions,maxlen ))
    relation_list = np.zeros((num_questions, maxlen))
    for i in range(num_questions):
        ques_list[i,:len(ques_dict[i])]=(ques_dict[i])
        relation_list[i,:len(ques_dict[i])] = relation_dict[i]
    return ques_pre[:,0], ques_pre[:,1], ques_pre[:,2],  np.array(exercises), ques_list, relation_list

def process_word_file(file, num_questions=836, max_words_problem=200, emb_size=50):

    f= open(file,'r')
    f.readline()
    question_emb = torch.zeros(num_questions+1,max_words_problem, dtype = torch.int64)
    # for i in f:
    #     i_ = i.strip().split(',')
    #     question_emb[int(i_[0])][:len(i_) - 1] = torch.LongTensor([int(i_[j]) for j in range(1, len(i_))])
    # return question_emb
    # cnt=0
    for i in f:
        cnt+=1
        print(cnt)


        i_=i.strip().split(',')

        #print(i_[0])
        j=0
        while j*emb_size+1<len(i_):

            k=0

            while(k<emb_size and j*emb_size+k+1<len(i_)):
                try:
                    fl = float(i_[j*emb_size+k+1])
                except:
                    k+=1
                    continue

                question_emb[int(i_[0])][j][k]=fl
                k+=1
            j+=1

    # question_emb[(ques_len < 1).nonzero()][0]=  torch.zeros(300)
    # ques_len[(ques_len < 1).nonzero()] = 1
    torch.save(question_emb, 'question_tensor.pt')
    question_emb=torch.load( 'question_tensor.pt')
    return question_emb
# def split(students):
#     train_index , test_index=[],[]
#     for cnt,i in enumerate(students[0]):
#         print(len(i))
#
#     return train_index, test_index


def compute_auc(preds, labels):

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    auc = metrics.roc_auc_score(labels, preds)
    return auc
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='junyi', type=str)
    parser.add_argument('--time_span', default=100000, type=int)
    parser.add_argument('--time_span2', default=256, type=int)
    parser.add_argument('--exercise_emb', default='assist_question_emb.csv', type=str)
    parser.add_argument('--exercise_sim', default='junyi_corr', type=str)
    #parser.add_argument('--exercise_sim', default='exercise_diff.csv', type=str)
    parser.add_argument('--exercise_sim3', default='occ4', type=str)
    parser.add_argument('--exercise_sim1', default='occ2', type=str)
    parser.add_argument('--num_concepts', default=41, type=int) #876
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--concept2que', default='concept2que.json', type=str)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--max_len', default=50 , type=int)
    parser.add_argument('--pos', default=False, type=bool)
    parser.add_argument('--num_words', default=200, type=int)
    parser.add_argument('--negative_sample',default='data/synthetic/synthetic_neg.csv', type=str)
    parser.add_argument('--num_text_len', default=50, type=int)
    parser.add_argument('--score_file', default='assist_score.csv', type=str)
    #parser.add_argument('--num_questions',default=2781, type=int) #684
    #parser.add_argument('--num_questions', default=51, type=int)
    #parser.add_argument('--num_questions', default=65789, type=int)  # 684
    parser.add_argument('--num_questions', default=768, type=int)


    args = parser.parse_args()
    data_path = 'data/'+args.dataset+'/'+args.dataset + "2.csv"

    max_len = args.max_len
    print('Started reading log data.')

    neighbor_ret, neg_neigh,relations_ret,  mask_neigh,  max_neigh = \
        read_relations_file( args.exercise_sim1,args.negative_sample, args.num_questions,1)
    # neighbor_ret2, neg_neigh2, relations_ret2, mask_neigh2, max_neigh = \
    #     read_relations_file(args.exercise_sim2, args.negative_sample,  args.num_questions,2)
    # neighbor_ret3, neg_neigh3, relations_ret3, mask_neigh3, max_neigh = \
    #     read_relations_file(args.exercise_sim3, args.negative_sample,  args.num_questions,3)
    #print(max_neigh)
    #adj_pos_target, adj_pos_neigh, adj_pos_rel,  adj_neg_neigh, neigh_list, reln_list = build_relation_graph(args.exercise_rel, args.num_questions)
    # print(adj_pos.nonzero()[0].shape)
    # adj_pos_target, adj_pos_neigh, adj_pos_rel, adj_neg_neigh, neigh_list,reln_list = torch.LongTensor(
    #     adj_pos_target).cuda(), torch.LongTensor(adj_pos_neigh).cuda(), torch.LongTensor(
    #     adj_pos_rel).cuda(),   torch.LongTensor(adj_neg_neigh).cuda(), torch.LongTensor(neigh_list).cuda(), torch.LongTensor(reln_list).cuda()

    #print(pre_req)

    students,corr_rel,time_seq,args.time_span= read_data_from_csv_file(data_path,args.exercise_sim,args.max_len,args.num_questions, args.time_span)

    #print(np.nonzero(corr_rel))
    args.time_span = int(args.time_span)
    num_students = len(students[0])
    print("Number of students: {}".format(num_students))
    print(args.time_span)
    print(args.time_span2)
    #conept_emb = nn.Parameter(torch.Tensor(args.num_concepts, args.embedding_dim).cuda())
    # neighbors, relations, concepts, mask_neigh, mask_concept,max_neigh,max_concept =\
    #     read_relations_file(args.exercise_rel,args.num_questions)

    print('Started reading word file.')

    print('Finished reading word file')
    #m_topic = TopicRNNModel(args.num_text_len, args.num_words,output_size = args.embedding_dim).cuda()

    print("Started reading relation file:")

    print("Ended read relation file.")


    model =SAKT(args).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # for name, param in model.named_parameters():
    #
    #     if param.requires_grad:
    #         param.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    #m_GAT = GraphAttentionLayer(args).cuda()
    bs = args.batch_size

    cnt=0
    f1 = open('assist_corr', 'r')
    sim=defaultdict(list)
    pre=-1
    for i in f1:
        #
        i_ = i.strip().split(',')

        sim[int(i_[0])].append(int(i_[1]))

    #index = np.random(1,len(stud))
    with autograd.detect_anomaly():
        print("Starting to train")
        f_result = open('result_'+args.dataset+str(cnt)+'_'+str(args.embedding_dim)+'.csv', 'w')
        cnt+=1
        best_auc=0

        for ep in range(args.num_epochs):
            print(ep)

            loss, total_loss, rmse, mae,auc = 0,0,0,0,0

            i= 0
            total_loss1,total_loss2=0,0

            while (i  <0.8*num_students):
                #print("print relations")
                if i%5000==0:
                    print(i)
                loss=0
                # if i in range(25344, 76003):
                #     i+=1
                #     continue

                #print("Modeling  test logs")
                hist_seq = torch.LongTensor(students[0][i: i+bs]).cuda()

                corr_seq1 = torch.LongTensor(students[1][i:i + bs]).cuda()
                #print(time_seq[i:i+bs].shape)
                time_seq1 = torch.Tensor(time_seq[i:i+bs,1:,1:]).cuda()
                time_seq2 = torch.LongTensor(students[1][i:i+bs]).cuda()

                rel = torch.Tensor(corr_rel[i:i+bs]).cuda()
                #hist_seq =
                next = torch.LongTensor(students[2][i: i + bs]).cuda()
                corr_seq = torch.Tensor(students[3][i:i+bs]).cuda()
                target_pos = next.contiguous().view(-1)
                target_pos2=hist_seq.contiguous().view(-1)
                hist_seq1 = (hist_seq+args.num_questions*corr_seq1)
                target_id = target_pos.nonzero() * args.num_questions + next[next > 0].view(-1, 1)
                corr_seq2 = corr_seq1.unsqueeze(-1).repeat(1,1,args.embedding_dim).to(torch.float32)


                pred,pred2 = model.forward(hist_seq,next,corr_seq1,time_seq1,time_seq2,next,rel)

                corr_seq = corr_seq.view(-1)
                corr_seq = corr_seq[target_pos.nonzero()].view(-1)


                pred =pred.view(-1)[target_pos.nonzero()].view(-1)

                #print(torch.isnan(pred).any())
                loss2=nn.BCELoss(reduction = 'sum')(pred, corr_seq)
                #print(loss2)
                target_que=target_pos[target_pos>0]
                #target_que=np.unique(target_que.cpu().numpy())

                #print(loss)

                total_loss1+=loss2


                optimizer.zero_grad()

                auc+=compute_auc(pred, corr_seq)
                loss2.backward()
                optimizer.step()
                i += bs
            #model.update(ques_emb_new)

            print(total_loss1/(i))
            print(auc/(i/bs))
            print("starting to relations ")
            j = 0
            bs2=256

            total_loss = 0
            cnt=0
            #
            # while j < args.num_questions:
            #     loss1 = model.loss(torch.arange(j, min(j + bs2,args.num_questions)).cuda(),
            #                     torch.LongTensor(neighbor_ret[j:j + bs2]).cuda(),
            #                        torch.LongTensor(neg_neigh[j:j+bs2]).cuda(),
            #                        torch.LongTensor(relations_ret[j:j+bs2]).cuda(),
            #                        torch.Tensor(mask_neigh[j:j+bs2]).cuda())
            #
            # #     # loss1 += model.loss(torch.arange(j, min(j + bs2, args.num_questions)).cuda(),
            # #     #                    torch.LongTensor(neighbor_ret2[j:j + bs2]).cuda(),
            # #     #                    torch.LongTensor(neg_neigh2[j:j + bs2]).cuda(),
            # #     #                    torch.LongTensor(relations_ret2[j:j + bs2]).cuda(),
            # #     #                    torch.Tensor(mask_neigh2[j:j + bs2]).cuda())
            # #     # loss1 += model.loss(torch.arange(j, min(j + bs2, args.num_questions)).cuda(),
            # #     #                     torch.LongTensor(neighbor_ret3[j:j + bs2]).cuda(),
            # #     #                     torch.LongTensor(neg_neigh3[j:j + bs2]).cuda(),
            # #     #                     torch.LongTensor(relations_ret3[j:j + bs2]).cuda(),
            # #     #                     torch.Tensor(mask_neigh3[j:j + bs2]).cuda())
            # #
            #     total_loss += loss1
            #     optimizer.zero_grad()
            #     loss1.backward()
            #     optimizer.step()
            #     j += bs2
            # #
            # print(total_loss / args.num_questions)

            cnt+=1
            print("Starting to test")

            p ,actual_label= [], []

            #f=open('attention','w')


            total_cnt,total_cnt2=0,0
            while i<num_students:
            #for i in range(25344, 33779,bs):

               #  loss=0
                #loss = model.loss( ques_emb,adj_pos_target,adj_pos_neigh, adj_pos_rel, adj_neg_target, adj_neg_neigh, adj_neg_rel)
                hist_seq = torch.LongTensor(students[0][i: i + bs]).cuda()
                corr_seq1 = torch.LongTensor(students[1][i:i + bs]).cuda()
                corr_seq = torch.Tensor(students[3][i:i + bs]).cuda()
                hist_seq1 = (hist_seq+ args.num_questions * corr_seq1)
                time_seq1 = torch.Tensor(time_seq[i:i+bs,1:,1:]).cuda()
                time_seq2 = torch.LongTensor(students[1][i:i+bs]).cuda()


                next = torch.LongTensor(students[2][i: i + bs]).cuda()
                next_seq = students[2][i:i+bs]
                hist_seq2=students[0][i:i+bs]
                mask_que=np.zeros(next_seq.shape)
                rel = torch.Tensor(corr_rel[i:i + bs]).cuda()

                # if i+bs>=num_students:
                #     to_check,next_que,pre_que=[],[],[]
                #     it = 0
                #     while it < next_seq.shape[0]:
                #
                #         f = 0
                #         for j in range(next_seq.shape[1]):
                #             if next_seq[it][j]!=0:
                #                 total_cnt2+=1
                #             if int(next[it][j]) in sim.keys():
                #
                #                 for k in range(j,0,-1):
                #                     if int(hist_seq2[it][k]) in sim[int(next[it][j])]:
                #                         mask_que[it][j] = 1
                #                         if i + bs  >=num_students:
                #                             next_que.append(next[it][j])
                #                             pre_que.append(hist_seq2[it][j])
                #                             to_check.append(k)
                #                             # print(next[it][j])
                #                             # print(hist_seq2[it][j])
                #                             # print(corr_rel[it][j])
                #                         total_cnt += 1
                #                         f = 1
                #                         break
                #             if f == 0:
                #                 mask_que[it][j] = 0
                #
                #         it += 1
                #print("Testind")
                target_pos = next.contiguous().view(-1)
                #print(mask_que[i-n:i-n+bs].nonzero())
                #target_pos = torch.Tensor(mask_que).cuda().view(-1)


                #target_id = target_pos.nonzero() * args.num_questions + next[next > 0].view(-1, 1)
                corr_seq2 = corr_seq1.to(torch.float32).unsqueeze(-1).repeat(1, 1, args.embedding_dim)



                pred, att = model.forward( hist_seq,next,corr_seq1,time_seq1,time_seq2,next.cuda(),rel)


                #pred, att = model.test(sim,next, hist_seq, corr_seq1,mask_que)

                corr_seq = corr_seq.view(-1)

                corr_seq = corr_seq[target_pos.nonzero()].view(-1)
                pred = pred.view(-1)[target_pos.nonzero()].view(-1)

                i += bs

                p.extend(pred.data.cpu().view(-1).numpy())
                actual_label.extend(corr_seq.data.cpu().numpy())


            print(total_cnt)
            print(total_cnt2)
            p, actual_label = np.array(p), np.array(actual_label)
            target_names = ['class 0', 'class 1' ]
            print(metrics.classification_report(actual_label, np.array([round(i) for i in p]), target_names=target_names))
            rmse = mean_squared_error(p, actual_label)
            mae = mean_absolute_error(p, actual_label)
            fpr, tpr, _ = metrics.roc_curve(actual_label, p,  pos_label=1)
            roc_auc = metrics.auc(fpr, tpr)
            acc = metrics.accuracy_score(actual_label, np.array([round(i) for i in p]))
            if roc_auc> best_auc:
                best_auc = roc_auc
                print(next_seq[-1])
                print("Test MAE"+str(mae))
                print("Test RMSE"+str(rmse))
                print("Test AUC"+str(roc_auc))
                print("Test ACC"+ str(acc))
                # f_result.write("Test MAE:" + str(mae)+'\n')
                # f_result.write("Test RMSE:" + str(rmse)+'\n')
                # f_result.write("Test AUC:" + str(roc_auc)+'\n')
                # f_result.write("Test ACC:"+ str(acc)+'\n')

                idx = np.nonzero(mask_que)
                attn = att.data.cpu().numpy()

                # print(to_check)
                # print(next_que[idx])
                # # # model.print(torch.LongTensor(next_que).cuda(), torch.LongTensor(pre_que).cuda(),torch.LongTensor([2,2,2,2]).cuda())
                # print(pre_que)
                # print(next.shape)
                # print(next.data.cpu().numpy()[idx])
                # print(hist_seq2[idx])
                f=open('result' , 'w')
                # # np.savetxt(f, np.array(to_check),delimiter=',')
                # # #np.savetxt(f,'\n')
                # print(att.shape)
                # print(att)
                #print(att)
                attn = attn[-1,:,:]
                print(attn.shape)
                print(attn)
                np.savetxt(f,attn)
                #np.savetxt(f, '\n')
                #f_result.write(str(ma))












