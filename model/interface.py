from tkinter import *
import fc_model
import rnn_model
import cnn_model
import fine_tuning_bert_model


def start_analysis():
    fc_result.delete('1.0', 'end')
    rnn_result.delete('1.0', 'end')
    cnn_result.delete('1.0', 'end')
    fine_tuning_bert_result.delete('1.0', 'end')

    text = text_input.get("1.0", "end")
    fc_outcome = fc_model.predict(text, max_len=100, weights_path="weights/fc_model_epoch75_0.6821.pt")
    rnn_outcome = rnn_model.predict(text, max_len=100, weights_path="weights/rnn_model_epoch49_0.7221.pt")
    cnn_outcome = cnn_model.predict(text, max_len=100, weights_path="weights/cnn_model_epoch49_0.7545.pt")
    bert_outcome = fine_tuning_bert_model.predict(text, max_len=100, weights_path="weights/epoch0_0.8407.pt")

    fc_result.insert("insert", fc_outcome)
    rnn_result.insert("insert", rnn_outcome)
    cnn_result.insert("insert", cnn_outcome)
    fine_tuning_bert_result.insert("insert", bert_outcome)


root = Tk()
root.title("Chinese Sentiment Classification")
root.minsize(100, 60)
button1 = Button(root, text="开始判断", command=start_analysis)
button1.grid(column=0, row=10)  # grid dynamically divides the space in a grid
# button1.pack()

input_label = Label(root, text='文本框插入字符：')
text_input = Text(root, width=70, height=10)
input_label.grid(column=0, row=0)
text_input.grid(column=0, row=1)

fc_label = Label(root, text='fc：')
fc_result = Text(root, width=70, height=5)
fc_label.grid(column=0, row=2)
fc_result.grid(column=0, row=3)

rnn_label = Label(root, text='rnn：')
rnn_result = Text(root, width=70, height=5)
rnn_label.grid(column=0, row=4)
rnn_result.grid(column=0, row=5)

cnn_label = Label(root, text='cnn：')
cnn_result = Text(root, width=70, height=5)
cnn_label.grid(column=0, row=6)
cnn_result.grid(column=0, row=7)

fine_tuning_bert_label = Label(root, text='fine-tuned bert-base：')
fine_tuning_bert_result = Text(root, width=70, height=5)
fine_tuning_bert_label.grid(column=0, row=8)
fine_tuning_bert_result.grid(column=0, row=9)

root.mainloop()
