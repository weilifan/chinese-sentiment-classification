# import pandas as pd
# import numpy as np
#
# data = pd.read_csv("../data/DMSC.csv")
# print(data.size)
# star_array = np.array(data["Star"])
# star_list = star_array.tolist()
# print("差评个数", len([i for i in star_list if i <= 3]))
# print("好评个数", len([i for i in star_list if i > 3]))

# -*-coding:utf-8-*-
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from model import dataset_train
from vocab import Vocab

import jieba

batch_size = 512  # batch size
voc_model = pickle.load(open("../models/vocab.pkl", "rb"))
sequence_max_len = 100  # 一个句子的最大长度
Vocab()


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
    return dataset_train.ImdbDataset(train)


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

class ImdbModel(nn.Module):
    def __init__(self, num_embeddings, padding_idx):
        super(ImdbModel, self).__init__()
        embedding_dim = 200

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=200, padding_idx=padding_idx)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(3, embedding_dim)),
            nn.ReLU(inplace=True)

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(4, embedding_dim)),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(5, embedding_dim)),
            nn.ReLU(inplace=True)
        )
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(300, 2)
        )

    def forward(self, input):
        input_embeded = self.embedding(input)  # torch.Size([512, 100, 200])
        # input_embeded_viewed = input_embeded.view(input_embeded.size(0), -1)
        input_embeded_viewed = input_embeded.unsqueeze(dim=1)  # torch.Size([512, 1, 100, 200])
        feature1 = self.conv1(input_embeded_viewed).squeeze(
            -1)  # squeeze前torch.Size([512, 100, 98, 1])，后torch.Size([512, 100, 98])
        feature2 = self.conv2(input_embeded_viewed).squeeze(-1)  # torch.Size([512, 100, 97])
        feature3 = self.conv3(input_embeded_viewed).squeeze(-1)  # torch.Size([512, 100, 96])

        feature1 = self.max_pool(feature1).squeeze(-1)  # torch.Size([512, 100])
        feature2 = self.max_pool(feature2).squeeze(-1)  # torch.Size([512, 100])
        feature3 = self.max_pool(feature3).squeeze(-1)  # torch.Size([512, 100])

        features = torch.cat([feature1, feature2, feature3], dim=-1)  # torch.Size([512, 300])
        out = self.classifier(features)

        return out


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

    model = ImdbModel(len(voc_model), voc_model.PAD).to(device())
    model.load_state_dict(torch.load("weights/cnn_model_epoch29_0.7439.pt", map_location=DEVICE))
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

        torch.save(model.state_dict(), SAVE_PATH + "cnn_model_epoch{}_{:.4f}.pt".format(i+30, val_acc))
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
    model = ImdbModel(len(voc_model), voc_model.PAD).to(device())
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model = model.to(DEVICE)
    # criterion = nn.CrossEntropyLoss()

    text_tokens = [word for word in jieba.cut(sentence)]  # 直接使用jieba分词
    tokens = voc_model.transform(text_tokens, max_len=max_len)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

    preds = F.softmax(model(tokens), dim=-1)

    # entroy = nn.CrossEntropyLoss()
    # target1 = torch.tensor([0])
    # target2 = torch.tensor([1])
    #
    # print(entroy(preds, target1),entroy(preds, target2))
    print("output",preds)


if __name__ == "__main__":
    SAVE_PATH = 'weights/'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train(optimizer="sgd", lr=5e-4, weight_decay=1e-3, epoch=70, clip=0.8)
    predict('好看的，赞，推荐给大家', max_len=100, weights_path="weights/cnn_model_epoch29_0.7439.pt")
    predict('无理由特效,全程很尴尬，这一星是给幕后辛苦的特效人员的', max_len=100, weights_path="weights/cnn_model_epoch29_0.7439.pt")
