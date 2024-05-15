import os
import re

data_dir = 'data/李文鑫data/data1/有标签/'
data_files = os.listdir(data_dir)
target = 'reimbursement'
for data_file in data_files:
    with open(data_dir + data_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = re.sub('\n', '', line)
            temp = line.split(" ")
            if target in temp:
                print(data_file)
print('end')