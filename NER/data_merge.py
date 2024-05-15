import os



data_dir = 'data\\李文鑫data\\Texts\\'
data_files = os.listdir(data_dir)
total = ''
for data_file in data_files:
    with open(data_dir + data_file, 'r', encoding='utf-8') as reader:
        total += reader.read()
        total += '\n'
with open('total', 'w', encoding='utf-8') as writer:
    writer.write(total)

