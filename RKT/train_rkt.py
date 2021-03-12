import argparse
import pandas as pd
from random import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import sparse
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from collections import  defaultdict

from model_rkt import RKT
from utils import *

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def compute_corr(prob_seq, next_seq, corr_dic):
    corr= np.zeros((prob_seq.shape[0],prob_seq.shape[1], prob_seq.shape[1]))
    for i in range(0,prob_seq.shape[0]):
        for  j in range(0,next_seq.shape[1] ):
            for k in range(j+1):
                corr[i][j][k]=corr_dic[next_seq[i][j]][prob_seq[i][k]]
    return corr
def get_data(df, file2, max_length, train_split=0.8, randomize=True):
    """Extract sequences from dataframe.
    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        max_length (int): maximum length of a sequence chunk
        train_split (float): proportion of data to use for training
    """

    pro_pro_sparse = sparse.load_npz('../data/pro_pro_sparse.npz')

    pro_pro_coo = pro_pro_sparse.tocoo()
    # print(pro_skill_csr)
    pro_pro_dense = pro_pro_coo.toarray()


    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    # skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
    #              for _, u_df in df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]



    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i + 1))[:-1] for i in item_ids]
    # skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s + 1))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]

    def chunk(list):
        if list[0] is None:
            return list
        list = [torch.split(elem, max_length) for elem in list]
        return [elem for sublist in list for elem in sublist]

    # Chunk sequences
    lists = (item_inputs, label_inputs, item_ids, labels)
    chunked_lists = [chunk(l) for l in lists]

    data = list(zip(*chunked_lists))
    if randomize:
        shuffle(data)

    # Train-test split across users
    train_size = int(train_split * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    return pro_pro_dense, train_data, val_data


def prepare_batches(train_data, batch_size, randomize=True):
    """Prepare batches grouping padded sequences.

    Arguments:
        data (list of lists of torch Tensor): output by get_data
        batch_size (int): number of sequences per batch

    Output:
        batches (list of lists of torch Tensor)
    """
    # if randomize:
    #     shuffle(train_data)
    batches = []
    train_y, train_skill, train_problem, timestamps, train_real_len = train_data[0], train_data[1], train_data[2], train_data[3], train_data[4]

    item_ids = [torch.LongTensor(i) for i in train_problem]
    timestamp = [torch.LongTensor(timestamp) for timestamp in timestamps]
    labels = [torch.LongTensor(i) for i in train_y]
    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i))[:-1] for i in item_ids]
    # skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]
    data = list(zip(item_inputs, label_inputs, item_ids, timestamp, labels))

    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        seq_lists = list(zip(*batch))

        inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
                          for seqs in seq_lists[:-1]]
        labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
        batches.append([*inputs_and_ids, labels])

    return batches


def train_test_split(data, split=0.8):
    n_samples = data[0].shape[0]
    split_point = int(n_samples*split)
    train_data, test_data = [], []
    for d in data:
        train_data.append(d[:split_point])
        test_data.append(d[split_point:])
    return train_data, test_data


def compute_auc(preds, labels):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    if len(torch.unique(labels)) == 1:  # Only one class
        auc = accuracy_score(labels, preds.round())
        acc = auc
    else:
        auc = roc_auc_score(labels, preds)
        acc = accuracy_score(labels, preds.round())
    return auc, acc


def compute_loss(preds, labels, criterion):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    return criterion(preds, labels)
def computeRePos(time_seq, time_span):
    batch_size = time_seq.shape[0]
    size = time_seq.shape[1]

    time_matrix= (torch.abs(torch.unsqueeze(time_seq, axis=1).repeat(1,size,1).reshape((batch_size, size*size,1)) - \
                 torch.unsqueeze(time_seq,axis=-1).repeat(1, 1, size,).reshape((batch_size, size*size,1))))

    # time_matrix[time_matrix>time_span] = time_span
    time_matrix = time_matrix.reshape((batch_size,size,size))


    return (time_matrix)
def get_corr_data(pro_num):
    pro_pro_dense = np.zeros((pro_num, pro_num))
    pro_pro_ = open('../../KT-GAT/ednet_corr')
    for i in pro_pro_:
        j = i.strip().split(',')
        pro_pro_dense[int(j[0])][int(j[1])] += int(float(j[2]))
    return pro_pro_dense


def train(train_data, val_data, pro_num, timestamp, timespan,  model, optimizer, logger, saver, num_epochs, batch_size, grad_clip):
    """Train RKT model.
    Arguments:
        train_data (list of tuples of torch Tensor)
        val_data (list of tuples of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        saver: wrapper for torch saving
        num_epochs (int): number of epochs to train for
        batch_size (int)
        grad_clip (float): max norm of the gradients
    """
    criterion = nn.BCEWithLogitsLoss()

    step = 0
    metrics = Metrics()
    corr_data = get_corr_data(pro_num)
    for epoch in range(num_epochs):

        train_batches = prepare_batches(train_data, batch_size)
        val_batches = prepare_batches(val_data, batch_size)

        # Training

        for item_inputs, label_inputs, item_ids, timestamp, labels in train_batches:
            # rel = compute_corr(item_inputs, item_ids, corr_data)

            rel = corr_data[(item_ids-1).unsqueeze(1).repeat(1,item_ids.shape[-1],1),(item_inputs-1).unsqueeze(-1).repeat(1,1,item_inputs.shape[-1])]
            item_inputs = item_inputs.cuda()
            time = computeRePos(timestamp, timespan)
            # skill_inputs = skill_inputs.cuda()
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            # skill_ids = skill_ids.cuda()

            preds, weights = model(item_inputs, label_inputs, item_ids, torch.Tensor(rel).cuda(), time.cuda())


            loss = compute_loss(preds, labels.cuda(), criterion)
            preds = torch.sigmoid(preds).detach().cpu()
            train_auc, train_acc = compute_auc(preds, labels)

            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1
            metrics.store({'loss/train': loss.item()})
            metrics.store({'auc/train': train_auc})

            # Logging
            if step == len(train_batches)-1:
                torch.save(weights, 'weight_tensor_rel')
            # print(step)
            if step % 1000 == 0:
                logger.log_scalars(metrics.average(), step)

                # weights = {"weight/" + name: param for name, param in model.named_parameters()}
                # grads = {"grad/" + name: param.grad
                #         for name, param in model.named_parameters() if param.grad is not None}
                # logger.log_histograms(weights, step)
                # logger.log_histograms(grads, step)

        # Validation

        model.eval()
        for item_inputs, label_inputs, item_ids, timestamp, labels in val_batches:
            rel = corr_data[
                (item_ids - 1).unsqueeze(1).repeat(1, item_ids.shape[-1], 1), (item_inputs - 1).unsqueeze(-1).repeat(1,
                                                                                                                     1,
                                                                                                            item_inputs.shape[
                                                                                                                         -1])]
            item_inputs = item_inputs.cuda()
            # skill_inputs = skill_inputs.cuda()
            time = computeRePos(timestamp, timespan)
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            # skill_ids = skill_ids.cuda()
            with torch.no_grad():
                preds,weights = model(item_inputs, label_inputs, item_ids, torch.Tensor(rel).cuda(), time.cuda())

                preds = torch.sigmoid(preds).cpu()


            val_auc, val_acc = compute_auc(preds, labels)
            metrics.store({'auc/val': val_auc, 'acc/val': val_acc})
        model.train()

        # Save model

        average_metrics = metrics.average()
        logger.log_scalars(average_metrics, step)
        print(average_metrics)
        stop = saver.save(average_metrics['auc/val'], model)
        if stop:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RKT.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/rkt')
    parser.add_argument('--savedir', type=str, default='save/rkt')
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--embed_size', type=int, default=200)
    parser.add_argument('--num_attn_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=5)
    parser.add_argument('--encode_pos', action='store_true')
    parser.add_argument('--max_pos', type=int, default=10)
    parser.add_argument('--drop_prob', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=10)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--timespan', default=100000, type=int)

    args = parser.parse_args()


    # print(len(train_data))
    data = np.load('data/ednet.npz')

    y, skill, problem, timestamp, real_len = data['y'], data['skill'], data['problem'], data['time'] , data['real_len']
    skill_num, pro_num = data['skill_num'], data['problem_num']
    print('problem number %d, skill number %d' % (pro_num, skill_num))

    # divide train test set
    train_data, test_data = train_test_split([y, skill, problem, timestamp, real_len])
    num_items = pro_num
# num_items = int(full_df["item_id"].max() + 1)
    # num_skills = int(full_df["skill_id"].max() + 1)

    model = RKT(num_items, args.embed_size, args.num_attn_layers, args.num_heads,
                  args.encode_pos, args.max_pos, args.drop_prob).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Reduce batch size until it fits on GPU
    while True:
        # try:
            # Train
        param_str = (f'{args.dataset},'
                     f'batch_size={args.batch_size},'
                     f'max_length={args.max_length},'
                     f'encode_pos={args.encode_pos},'
                     f'max_pos={args.max_pos}')
        logger = Logger(os.path.join(args.logdir, param_str))
        saver = Saver(args.savedir, param_str)

        train(train_data, test_data, pro_num, timestamp, args.timespan, model, optimizer, logger, saver, args.num_epochs,
              args.batch_size, args.grad_clip)
        break
        # except RuntimeError:
        #     args.batch_size = args.batch_size // 2
        #     print(f'Batch does not fit on gpu, reducing size to {args.batch_size}')

    logger.close()

    param_str = (f'{args.dataset},'
                  f'batch_size={args.batch_size},'
                  f'max_length={args.max_length},'
                  f'encode_pos={args.encode_pos},'
                  f'max_pos={args.max_pos}')
    saver = Saver(args.savedir, param_str)
    model = saver.load()
    # test_data, _ = get_data(test_df, args.max_length, train_split=1.0, randomize=False)
    test_batches = prepare_batches(test_data, args.batch_size, randomize=False)
    corr_data = get_corr_data()
    test_preds = np.empty(0)

    # Predict on test set
    model.eval()
    correct = np.empty(0)
    for item_inputs, label_inputs, item_ids, labels in test_batches:
        rel = corr_data[
            (item_ids - 1).unsqueeze(1).repeat(1, item_ids.shape[-1], 1), (item_inputs - 1).unsqueeze(-1).repeat(1,
                                                                                                                 1,
                                                                                                                 item_inputs.shape[
                                                                                                                     -1])]
        item_inputs = item_inputs.cuda()
        # skill_inputs = skill_inputs.cuda()
        label_inputs = label_inputs.cuda()
        item_ids = item_ids.cuda()
        # skill_ids = skill_ids.cuda()
        with torch.no_grad():
            preds = model(item_inputs, label_inputs, item_ids)
            preds = torch.sigmoid(preds[labels >= 0]).flatten().cpu().numpy()
            test_preds = np.concatenate([test_preds, preds])
        labels = labels[labels>=0].float()
        correct = np.concatenate([correct, labels])


    print("auc_test = ", roc_auc_score(correct, test_preds))
    print("acc_test = ", accuracy_score(correct, test_preds))
