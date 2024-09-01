# %%
# 使用LLM生成过程图
import json
import requests
from tqdm import tqdm
def get_llm_response():
    dataset_list = None
    with open('/home/btr/bpmn/LLMEnG/my-data/datasets5.json', 'r') as f:
        dataset_list = json.load(f)
    for i in range(39, len(dataset_list)):
        dataset = dataset_list[i]
        filename = dataset['filename']
        # if filename == '42_data.txt':
        #     print(i)
        #     return
        # else:
        #     continue
        text = dataset['text']
        node_token_list = dataset['node_token_list']
        ner_results = []
        for i, node_token in enumerate(node_token_list):
            ner_results.append(f"{i}: '{node_token}'")
        ner_results = '\n'.join(ner_results)
        prompt = f'''
I want to generate a Procedural Graph based on Procedural Document. To accurately generate a Procedural Graph, I use a NER tool to identify activities, signal words indicating branches, and conditions in the Procedural Document. They are arranged in the order in the Procedural Document. Each item is represented as "{id}: {text}", where id represents the graph node identifier and text represents the text information corresponding to the graph node.
You should generate a Procedural Graph line by line in the form of "[src_id, tgt_id]" until the last item, and keep the text of the nodes, signal words, and conditions consistent with the original program document.

Here are some examples:

###
"Procedural Document":
The credit company receives the credit information from the customer , If the requested amount is greater than 1M $ an approval must be requested . If the requested amount is lower or equal to 1M $ the company assess the risk of the credit . After the assessment , if the risk is high , an approval must be requested ; but if the risk is low the credit is accepted . After the approval request , the credit could be accepted or rejected ; in both cases , an email is sent to the customer.

###
"NER results":
0: "The credit company receives the credit information from the customer",
1: "If",
2: "the requested amount is greater than 1M $",
3: "an approval must be requested",
4: "If",
5: "the requested amount is lower or equal to 1M $",
6: "the company assess the risk of the credit",
7: "After",
8: "the assessment",
9: "if",
10: "the risk is high",
11: "an approval must be requested",
12: "but if",
13: "the risk is low",
14: "the credit is accepted",
15: "After",
16: "the approval request",
17: "the credit could be accepted or rejected",
18: "in both cases",
19: "an email is sent to the customer"

###
"Procedural Graph":
[0, 1],
[1, 2],
[2, 3],
[0, 4],
[4, 5],
[5, 6],
[6, 7],
[7, 8],
[8, 9],
[9, 10],
[10, 11],
[9, 12],
[12, 13],
[13, 14],
[3, 15],
[11, 15],
[15, 16],
[16, 17],
[17, 18],
[18, 19]

###
"Procedural Document":
If a not collected bag is identified , CVS must check the number of resolution retries conducted . If the number of retries is greater than an specified number , the bag must be returned to the production area for disassemble ; in all other cases , CVS must try to contact the customer by phone and remind to pick up the bag . If the contact is successful and the customer withdrawn the order , the bag must be returned to the production area for disassemble , but if it is successful and the customer promises to pick up the bag , CVS must increase the retry count on the bag label . Also , this number must be increased if the contact is unsuccessful .

###
"NER results":
0: "If",
1: "a not collected bag is identified",
2: "CVS must check the number of resolution retries conducted",
3: "If",
4: "the number of retries is greater than an specified number",
5: "the bag must be returned to the production area for disassemble",
6: "in all other cases",
7: "CVS must try to contact the customer by phone and remind to pick up the bag",
8: "If",
9: "the contact is successful and the customer withdrawn the order",
10: "the bag must be returned to the production area for disassemble",
11: "but if",
12: "it is successful and the customer promises to pick up the bag",
13: "CVS must increase the retry count on the bag label",
14: "Also",
15: "this number must be increased",
16: "if",
17: "the contact is unsuccessful"

###
"Procedural Graph":
[0, 1],
[1, 2],
[2, 3],
[3, 4],
[4, 5],
[2, 6],
[6, 7],
[7, 8],
[8, 9],
[9, 10],
[7, 11],
[11, 12],
[12, 13],
[7, 14],
[14, 15],
[15, 16],
[16, 17]

Now you need to generate the corresponding Procedural Graph of the following Procedural Document and NER results (Note that the result output in JSON format only contains the "result" field. No other redundant content is allowed.):

###
"Procedural Document":
{text}

###
"NER results":
{ner_results}

###
"Procedural Graph":
'''
        edge_index = None
        if filename == '41_data.txt':
            break
        else:
            while True:
                url = 'http://localhost:11434/api/generate'
                data = {
                'model': 'llama3.1:70b',
                'prompt': prompt,
                'format': 'json',
                'stream': False,
                }
                try:
                    print(filename)
                    response = requests.post(url, json=data)
                    result = response.json()['response']
                    result = json.loads(result)['result']
                    edge_index = result
                    break
                except:
                    pass
            dataset['edge_index'] = edge_index
    dataset_list_json = json.dumps(dataset_list)
    with open('/home/btr/bpmn/LLMEnG/my-data/datasets8-42.json', 'w') as f:
        f.write(dataset_list_json)
get_llm_response()