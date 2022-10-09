# -*-coding:utf-8-*-
import pickle
import re

from torch.utils.data import Dataset, DataLoader

import jieba
import pandas as pd
import numpy as np

from model.vocab import Vocab

Vocab()


class ImdbDataset(Dataset):
    def __init__(self, train=True, sequence_max_len=100):
        # super(ImdbDataset,self).__init__()

        self.sequence_max_len = sequence_max_len
        self.voc_model = pickle.load(open("../models/vocab.pkl", "rb"))

        df = pd.read_csv("../data/DMSC.csv")
        review_array = np.array(df["Comment"])  # 将dataframe 转换为矩阵
        self.total_review = review_array.tolist()  # 将一维矩阵转化为列表

        star_array = np.array(df["Star"])  # 将dataframe 转换为矩阵
        self.total_star = star_array.tolist()  # 将一维矩阵转化为列表

    def __getitem__(self, idx):
        # file = self.total_path[idx]
        # 从txt获取评论并分词
        # review = tokenlize(open(file, "r", encoding="utf-8").read())

        review = tokenlize(self.total_review[idx]) #直接使用jieba分词

        voc_result = self.voc_model.transform(review, max_len=self.sequence_max_len)
        # 获取评论对应的label
        # label = int(file.split("_")[-1].split(".")[0])
        # label = 0 if label < 5 else 1

        label = self.total_star[idx]
        label = 0 if label <= 3 else 1

        return voc_result, label

    def __len__(self):
        return len(self.total_review)

    def get_num_embeddings(self):
        return len(self.voc_model)

    def get_padding_idx(self):
        return self.voc_model.PAD


def tokenlize(sentence):
    """
    进行文本分词
    :param sentence: str
    :return: [str,str,str]
    """

    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    # sentence = sentence.lower()  # 把大写转化为小写
    sentence = re.sub("<br />", "", sentence)
    sentence = re.sub("|".join(fileters), " ", sentence)
    sentence = "".join(sentence.split())  # 去掉所有空格
    # sentence = re.sub("I'm","I am",sentence)
    # sentence = re.sub("isn't","is not",sentence)
    # result = [i for i in sentence.split(" ") if len(i) > 0]

    result = [word for word in jieba.cut(sentence) if len(word) > 0]  # 直接使用jieba分词


    return result


# def unzip_file(zip_src, dst_dir):
#     """
#     解压缩
#     :param zip_src:
#     :param dst_dir:
#     :return:
#     """
#     r = zipfile.is_zipfile(zip_src)
#     if r:
#         fz = zipfile.ZipFile(zip_src, 'r')
#         bar = tqdm(fz.namelist())
#         bar.set_description("unzip  " + zip_src)
#         for file in bar:
#             fz.extract(file, dst_dir)
#     else:
#         print('This is not zip')


# 以下为调试代码
def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    """
    reviews, labels = zip(*batch)

    return reviews, labels


# def test_file(train=True):
#     if not os.path.exists("./data/download"):
#         unzip_file("./data/data.zip", "./data/download")
#     data_path = r"./data/download"
#     data_path += r"/train" if train else r"/test"
#     total_path = []  # 保存所有的文件路径
#     for temp_path in [r"/pos", r"/neg"]:
#         cur_path = data_path + temp_path
#         total_path += [os.path.join(cur_path, i) for i in os.listdir(cur_path) if i.endswith(".txt")]
#     print(total_path)


if __name__ == "__main__":

    imdb_dataset = ImdbDataset(True)
    my_dataloader = DataLoader(imdb_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    for data in my_dataloader:
        print(data)
        # vocab_model = pickle.load(open("./models/vocab.pkl", "rb"))
        # print(data[0][0])
        # result = vocab_model.transform(data[0][0], 100)
        # print(result)
        break

    # unzip_file("./data/a.zip", "./data/download")
    # if os.path.exists("./data/download"):
    #     print("T")

    # data = open("./data/download/train/pos\\10032_10.txt", "r", encoding="utf-8").read()
    # result = tokenlize("--or something like that. Who the hell said that theatre stopped at the orchestra pit--or even at the theatre door?")
    # result = tokenlize(data)
    # print(result)

    # test_file()
