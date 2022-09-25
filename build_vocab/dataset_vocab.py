# -*-coding:utf-8-*-

from torch.utils.data import Dataset

import jieba
import pandas as pd
import numpy as np

from model.vocab import Vocab

Vocab()


class ImdbDataset(Dataset):
    def __init__(self, train=True, sequence_max_len=100):
        # if not os.path.exists("./data/download"):
        #     unzip_file("./data/test.zip", "./data/download")
        #     unzip_file("./data/train.zip", "./data/download")
        # self.sequence_max_len = sequence_max_len
        # data_path = r"./data/download"
        # data_path += r"/train" if train else r"/test"
        # self.total_path = []  # 保存所有的文件路径
        # # self.voc_model = pickle.load(open("./models/vocab.pkl", "rb"))
        # for temp_path in [r"/pos", r"/neg"]:
        #     cur_path = data_path + temp_path
        #     self.total_path += [os.path.join(cur_path, i) for i in os.listdir(cur_path) if i.endswith(".txt")]

        df = pd.read_csv("D:/assignment/22_09\语义计算\chinese-sentiment-classification\data\DMSC.csv")
        review_array = np.array(df["Comment"])  # 将dataframe 转换为矩阵
        self.total_review = review_array.tolist()  # 将一维矩阵转化为列表

        star_array = np.array(df["Star"])  # 将dataframe 转换为矩阵
        self.total_star = star_array.tolist()  # 将一维矩阵转化为列表

    def __getitem__(self, idx):
        # 从txt获取评论并分词
        # review = tokenlize(open(file, "r", encoding="utf-8").read())
        review = [word for word in jieba.cut(self.total_review[idx])]
        return review

    def __len__(self):
        return len(self.total_review)

    # def get_num_embeddings(self):
    #     return len(self.voc_model)
    #
    # def get_padding_idx(self):
    #     return self.voc_model.PAD


# def tokenlize(sentence):
#     """
#     进行文本分词
#     :param sentence: str
#     :return: [str,str,str]
#     """
#
#     fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
#                 '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
#                 '“', ]
#     sentence = sentence.lower()  # 把大写转化为小写
#     sentence = re.sub("<br />", " ", sentence)
#     # sentence = re.sub("I'm","I am",sentence)
#     # sentence = re.sub("isn't","is not",sentence)
#     sentence = re.sub("|".join(fileters), " ", sentence)
#     result = [i for i in sentence.split(" ") if len(i) > 0]
#
#     return result


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
    # reviews, labels = zip(*batch)

    return batch


