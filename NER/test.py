# from stanfordcorenlp import StanfordCoreNLP

# nlp = StanfordCoreNLP('/home/btr/bpmn/NER/model/stanford-corenlp-4.5.5',lang='en')
# sentence = 'If the requested amount is greater than 1M$ an approval must be requested.'
# sentence_parse = nlp.parse(sentence)
# print(sentence_parse)
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
import spacy
import numpy as np

# 加载SpaCy的英文模型
nlp = spacy.load("en_core_web_sm")

# 示例文本
text = "If else"

# 使用SpaCy进行依存句法分析
doc = nlp(text)
print(doc)
# 定义依存关系的编码方式，这里使用one-hot编码
# 首先，获取所有可能的依存关系类型
dep_types = set(token.dep_ for token in doc if not token.is_punct)
dep_types = list(dep_types)

# 初始化one-hot编码字典
one_hot_encoding = {dep: np.zeros(len(dep_types)) for dep in dep_types}
for dep in dep_types:
    one_hot_encoding[dep][dep_types.index(dep)] = 1
print(one_hot_encoding)
# 提取依存关系并编码
encoded_dependencies = []
for token in doc:
    if not token.is_punct:
        dep_type = token.dep_
        # 假设我们只考虑直接的子节点和其依存关系
        if token.head != token:
            head_text = token.head.text
            dep_vector = one_hot_encoding[dep_type]
            encoded_dependencies.append((token.text, head_text, dep_vector))

# 打印编码后的依存关系
for token_text, head_text, dep_vector in encoded_dependencies:
    print(f"Token: {token_text}, Head: {head_text}, Dependency Vector: {dep_vector}")