import pandas as pd
import numpy as np

data = pd.read_csv("../data/DMSC.csv")
print(data.size)
star_array = np.array(data["Star"])
star_list = star_array.tolist()
print("差评个数", len([i for i in star_list if i <= 3]))
print("好评个数", len([i for i in star_list if i > 3]))

'''
好看的，赞，推荐给大家
比我预想的还要无聊
无理由特效,全程很尴尬，这一星是给幕后辛苦的特效人员的
剧情有点离谱，后面还注水严重，男二女二到底带资多少进组啊，戏份比男女主还多，剪辑的故意穿插，霸总剧情也是大可不必。
这部剧给我的感觉 是真实 温暖 还有坚定 它的故事是缓缓叙述的但又十分扣人心弦，里面讲了好几个家庭，人物形象都很饱满。疫情下的生活，压抑和多变，但是只要坚持下去才会有希望。 人生永远都在大考，也许过程比结局更有意义，在过程里的成长和摸索都是意义非凡的，我想。


说得比较隐晦：
比我预想的还要无聊（fc不行）
点手撕鬼子那劲儿了（fc和cnn不行）

评价的是演员：
太胖了，真的……每一个浑圆的胳膊，结实的双下巴，两人油腻的演出都在敲打着我的心（fc、cnn、rnn都不行）
虞书欣的夹子音好难听啊（fc、cnn、rnn都不行）
刘亦菲也许不会想到，自己也有演技碾压对手的一天（fc、cnn、rnn都不行）

阴阳怪气：
你们不要黑郭导好么，人家这部电影完美地诠释了“场景的华丽堆砌”、“没主线即主线”、“如何塑造空洞的人物灵魂”、“论特写镜头如何滥用”等词句。郭导的良苦用心岂是你们这群低级影迷能理解的（fc太行）
'''