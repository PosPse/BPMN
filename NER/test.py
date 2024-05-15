from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('/home/btr/bpmn/NER/model/stanford-corenlp-4.5.5',lang='en')
sentence = 'If the requested amount is greater than 1M$ an approval must be requested.'
sentence_parse = nlp.parse(sentence)
print(sentence_parse)
# import torch

# # 获取可用的 GPU 数量
# gpu_count = torch.cuda.device_count()
# print(f"Available GPUs: {gpu_count}")

# # 获取每个 GPU 的名称和设备编号
# for i in range(gpu_count):
#     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
# import re
# input = "I am submitting the job description for consideration and waiting for the approval."
# word_list = re.findall(r'\w+|[^\w\s]', input)
# print(word_list)