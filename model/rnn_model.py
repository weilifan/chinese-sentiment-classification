# -*-coding:utf-8-*-
import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

import dataset_train
from vocab import Vocab

import pickle

batch_size = 512
voc_model = pickle.load(open("../models/vocab.pkl", "rb"))
sequence_max_len = 100
Vocab()

SAVE_PATH = 'weights/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    """
    reviews, labels = zip(*batch)
    # reviews = torch.LongTensor([voc_model.transform(i, max_len=sequence_max_len) for i in reviews])
    reviews = torch.LongTensor(reviews)
    labels = torch.LongTensor(labels)
    return reviews, labels


def get_dataset():
    return dataset_train.DMSCDataset(train)


def get_dataloader(imdb_dataset, train):
    validation_split = .2
    dataset_size = len(imdb_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    _train_loader = DataLoader(imdb_dataset, shuffle=False, batch_size=batch_size,
                               sampler=train_sampler)
    _val_loader = DataLoader(imdb_dataset, shuffle=False, batch_size=batch_size,
                             sampler=valid_sampler)
    loader = _train_loader if train else _val_loader
    return loader


class DMSCDModel(nn.Module):
    def __init__(self, num_embeddings, padding_idx):
        super(DMSCDModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=200, padding_idx=padding_idx).to()
        self.rnn = nn.RNN(input_size=200, hidden_size=64, num_layers=2, nonlinearity='tanh', bias=True,
                          batch_first=True, bidirectional=True, dropout=0.5)
        self.fc1 = nn.Linear(64 * 2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """
        input_embeded = self.embedding(input)  # torch.Size([512, 100, 200])

        output, h_n = self.rnn(input_embeded)  # output torch.Size([512, 100, 128]), h_n torch.Size([4, 512, 64])
        # out :[batch_size,hidden_size*2]
        out = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)  # 拼接正向最后一个输出和反向最后一个输出
                                                                # h_n[-1, :, :] torch.Size([512, 64])
                                                                # h_n[-2, :, :] torch.Size([512, 64])
                                                                # out torch.Size([512, 128])
        # 进行全连接
        out_fc1 = self.fc1(out)  # out_fc1 torch.Size([512, 64])
        # 进行relu
        out_fc1_relu = F.relu(out_fc1)

        # 全连接
        out_fc2 = self.fc2(out_fc1_relu)  # out_fc2 torch.Size([512, 2])
        return out_fc2


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def validate(model, data_loader, criteon):
    model.eval()
    dev_loss, label_list, pred_list = [], [], []

    for tokens, labels in data_loader:

        tokens = torch.stack(tokens, dim=1).to(device())
        labels = labels.to(device())
        with torch.no_grad():
            preds = model(tokens)
        loss = criteon(preds, labels)
        dev_loss.append(loss.item())
        pred_list.append(preds.detach().cpu().numpy())
        label_list.append(labels.detach().cpu().numpy())

    pred_list = np.argmax(np.concatenate(pred_list, axis=0), axis=1)
    label_list = np.concatenate(label_list, axis=0)

    correct = (pred_list == label_list).sum()
    return np.array(dev_loss).mean(), float(correct) / len(label_list)



def train(optimizer, lr, weight_decay, epoch, clip):
    dataset = get_dataset()
    train_loader = get_dataloader(dataset, train=True)
    val_loader = get_dataloader(dataset, train=False)

    model = DMSCDModel(len(voc_model), voc_model.PAD).to(device())
    # model.load_state_dict(torch.load("weights/rnn_model_epoch41_0.7151.pt", map_location=DEVICE))
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    criteon = nn.CrossEntropyLoss()

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for i in range(epoch):
        bar = tqdm(train_loader, total=len(train_loader))
        model.train()
        train_loss, label_list, pred_list = [], [], []
        for idx, (tokens, labels) in enumerate(bar):
            tokens = torch.stack(tokens, dim=1).to(device())
            target = labels.to(device())

            optimizer.zero_grad()

            preds = model(tokens)
            loss = criteon(preds, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            train_loss.append(loss.item())
            label_list.append(labels.detach().cpu().numpy())
            pred_list.append(preds.detach().cpu().numpy())

            bar.set_description("epcoh:{}  idx:{}   loss:{:.6f}".format(i, idx, loss.item()))

        train_loss = np.array(train_loss).mean()
        val_loss, val_acc = validate(model, val_loader, criteon)

        pred_list = np.argmax(np.concatenate(pred_list, axis=0), axis=1)
        label_list = np.concatenate(label_list, axis=0)

        correct = (pred_list == label_list).sum()
        train_acc = float(correct) / len(label_list)

        torch.save(model.state_dict(), SAVE_PATH + "rnn_model_epoch{}_{:.4f}.pt".format(i+42, val_acc))
        print('Training loss:{}, Val loss:{}'.format(train_loss, val_loss))
        print("train acc:{:.4f}, val acc:{:4f}".format(train_acc, val_acc))

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

    plt.figure()
    plt.plot(train_loss_list)
    plt.plot(val_loss_list)
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.legend(['train', 'val'])
    plt.show()

    plt.figure()
    plt.plot(train_acc_list)
    plt.plot(val_acc_list)
    plt.xlabel("epoch")
    plt.ylabel('accuracy')
    plt.legend(['train', 'val'])
    plt.show()


def predict(sentence,max_len, weights_path):

    # weibo_loader = BertLoader(batch_size, ROOT_PATH, max_len)
    # test_loader = weibo_loader.get_test_loader()

    # construct data loader
    model = DMSCDModel(len(voc_model), voc_model.PAD).to(device())
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model = model.to(DEVICE)
    # criterion = nn.CrossEntropyLoss()

    text_tokens = [word for word in jieba.cut(sentence)]  # 直接使用jieba分词
    tokens = voc_model.transform(text_tokens, max_len=max_len)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

    preds = F.softmax(model(tokens), dim=-1)

    return preds


if __name__ == '__main__':
    # SAVE_PATH = 'weights/'
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(optimizer="sgd", lr=5e-4, weight_decay=1e-3, epoch=8, clip=0.8)
    # predict('好看的，赞，推荐给大家', max_len=100, weights_path="weights/rnn_model_epoch3_0.6007.pt")
    # predict('什么破烂反派，毫无戏剧冲突能消耗两个多小时生命，还强加爱情戏。脑残片好圈钱倒是真的', max_len=100, weights_path="weights/rnn_model_epoch3_0.6007.pt")

    # imdb_dataset = get_dataset()
    # imdb_model = ImdbModel(len(voc_model), voc_model.PAD).to(device())
    # train(imdb_model, imdb_dataset, 4)
    # test(imdb_model, imdb_dataset)
    #
    #
    # data = '挺好看的'
    # voc_model = pickle.load(open("./models/vocab.pkl", "rb"))
    # review = [word for word in jieba.cut(data)]  # 直接使用jieba分词
    #
    # voc_result = voc_model.transform(review, max_len=100)
    # print("哈哈",voc_result)
    # print("呵呵", len(voc_result))
    #
    #
    #
    # print("呵呵", torch.tensor(voc_result, dtype=torch.long).unsqueeze(0).shape)
    # print(torch.tensor(voc_result, dtype=torch.long).unsqueeze(0).shape)
    # print(imdb_model(torch.tensor(voc_result, dtype=torch.long).unsqueeze(0)))
