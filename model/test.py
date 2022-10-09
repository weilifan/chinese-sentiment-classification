import pandas as pd
import numpy as np

data = pd.read_csv("../data/DMSC.csv")
print(data.size)
star_array = np.array(data["Star"])
star_list = star_array.tolist()
print("差评个数", len([i for i in star_list if i <= 3]))
print("好评个数", len([i for i in star_list if i > 3]))