# chinese-sentiment-classification
1. 将Kaggle上的https://www.kaggle.com/datasets/utmhikari/doubanmovieshortcomments 下载并存放在data文件夹下。
![image](https://user-images.githubusercontent.com/57277850/205491322-0bb5c4ce-a80c-4f17-937f-d49db32754c8.png) 
2. 在运行model/中每个模型对应的脚本的train函数后，即可获得模型参数。
![image](https://user-images.githubusercontent.com/57277850/205491350-90c7baec-98c7-4c76-93f4-2718d9ee805a.png)
3. 将得到的模型参数添加到interface.py中start_analysis对应的不同显示文本区域后，通过运行interface.py可使用本项目的图形用户界面进行影评文本情感分析。
![image](https://user-images.githubusercontent.com/57277850/205491500-05b1c0af-a236-4f01-943e-45fee2800e21.png)
4. 在model/weights下的127.0.0.1.html可查看每种不同的模型在训练50个epoch内，在验证集上的预测准确度变化。
![image](https://user-images.githubusercontent.com/57277850/205491528-a9f44dae-ea8d-4f73-8432-f4c23d55118b.png)
