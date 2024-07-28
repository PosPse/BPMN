from tqdm import tqdm
import time
a = [ i for i in range(100)]
for i in tqdm(a):  # 使用trange代替range，并显示进度条
    # print(i)
    time.sleep(0.1)  # 模拟耗时操作gggggg