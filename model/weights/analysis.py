import os

import plotly.graph_objects as go
import plotly.offline as py

root_dir_fc = "fc_weight"  # 词向量文件根目录
root_dir_rnn = "rnn_weight"
root_dir_cnn = "cnn_weight"
root_dir_bert = "bert"
root_dir_bert_shuffle = "bert_after_shuffle"
fileList_fc = os.listdir(root_dir_fc)  # 列出文件夹下所有的目录与文件
fileList_rnn = os.listdir(root_dir_rnn)
fileList_cnn = os.listdir(root_dir_cnn)
fileList_bert = os.listdir(root_dir_bert)
fileList_bert_shuffle = os.listdir(root_dir_bert_shuffle)


# 遍历文件
def traverse_file(fileList):
    stored_list = []
    dic = {}
    x = []
    y = []
    for i in range(len(fileList)):
        filename = fileList[i]
        stored_list.append(filename.split("epoch")[1].split(".pt")[0])
    for i in stored_list:
        dic[int(i.split("_")[0])] = float(i.split("_")[1])
    for i in sorted(dic):
        x.append(i)
        y.append(dic[i])
    return x, y


def figure():
    trace0 = go.Scatter(
        x=traverse_file(fileList_fc)[0][:50],
        y=traverse_file(fileList_fc)[1][:50],
        mode='lines + markers',
        name='fnn'
    )
    trace1 = go.Scatter(
        x=traverse_file(fileList_cnn)[0],
        y=traverse_file(fileList_cnn)[1],
        mode='lines + markers',
        name='cnn'
    )
    trace2 = go.Scatter(
        x=traverse_file(fileList_rnn)[0],
        y=traverse_file(fileList_rnn)[1],
        mode='lines + markers',
        name='rnn'
    )
    trace3 = go.Scatter(
        x=traverse_file(fileList_bert)[0],
        y=traverse_file(fileList_bert)[1],
        mode='lines + markers',
        name='bert-before-shuffle'
    )
    trace4 = go.Scatter(
        x=traverse_file(fileList_bert_shuffle)[0],
        y=traverse_file(fileList_bert_shuffle)[1],
        mode='lines + markers',
        name='bert'
    )
    data = [trace0, trace1, trace2, trace4]

    py.iplot(data)


figure()
