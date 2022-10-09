import os
import pickle

from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset_vocab
from model.vocab import Vocab


def get_dataloader(train=True):
    imdb_dataset = dataset_vocab.ImdbDataset(train, sequence_max_len=100)
    my_dataloader = DataLoader(imdb_dataset, batch_size=200, shuffle=True, collate_fn=dataset_vocab.collate_fn)
    return my_dataloader


if __name__ == '__main__':

    ws = Vocab()
    dl_train = get_dataloader(True)
    for reviews in tqdm(dl_train, total=len(dl_train)):
        # reviews是一个二维向量，是一整个batch里所有的review[[token1, token2,..., token n],[token1, token2,..., token n],[token1, token2,..., token n]]
        for sentence in reviews:
            # 一个reviews是一个batch，一个sentence是一条评论[token1, token2,..., token n]
            ws.fit(sentence)

    ws.build_vocab()
    # print(len(ws))
    # print(ws.dict)
    #
    # ret = ws.transform(["美队", "好帅", "自动", "捂", "眼睛"], max_len=13)
    # print(ret)
    #
    # ret = ws.inverse_transform(ret)
    # print(ret)


    if not os.path.exists("../models"):
        os.makedirs("../models")
    pickle.dump(ws, open("../models/vocab.pkl", "wb"))
