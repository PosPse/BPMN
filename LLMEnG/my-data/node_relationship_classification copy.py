# 使用LLM判断节点之间是否存在链接
import json
import requests
from tqdm import tqdm

link_type = [
    '[activity]->[activity]',
    '[activity]->[condition]',
    '[activity]->[sign-successor]',
    '[activity]->[sign-selection]',
    '[activity]->[sign-parallel]',
    '[activity]->[sign-loop]',
    '[condition]->[sign-successor]',
    '[condition]->[sign-selection]',
    '[condition]->[sign-parallel]',
    '[condition]->[sign-loop]',
    '[condition]->[activity]',
    '[sign-successor]->[activity]',
    '[sign-selection]->[activity]',
    '[sign-parallel]->[activity]',
    '[sign-loop]->[activity]'
    '[sign-successor]->[condition]',
    '[sign-selection]->[condition]',
    '[sign-parallel]->[condition]',
    '[sign-loop]->[condition]'
]
# activity -> activity
# activity -> condition
# activity -> sign-successor
# activity -> sign-selection
# activity -> sign-parallel
# activity -> sign-loop
# condition -> sign-successor
# condition -> sign-selection
# condition -> sign-parallel
# condition -> sign-loop
def get_llm_response():
    dataset_list = None
    with open('/home/btr/bpmn/LLMEnG/my-data/datasets5.json', 'r') as f:
        dataset_list = json.load(f)

    for dataset in tqdm(dataset_list):
        text = dataset['text']
        filename = dataset['filename']
        if filename != '34_data.txt':
            continue
        node_token_list = dataset['node_token_list']
        data_2_mask_single_signal_llm = dataset['data_2_mask_single_signal_llm']
        node_to_text = ''
        for i, node_token in enumerate(node_token_list):
            node_to_text += f'{i}: {node_token}\n'
        
        node_pair_list = []
        relation_matrix = [[0 for _ in range(len(node_token_list))] for _ in range(len(node_token_list))]
        for i in range(len(node_token_list)):
            for j in range(i+1, len(node_token_list)):
                if f'{data_2_mask_single_signal_llm[i]}->{data_2_mask_single_signal_llm[j]}' in link_type:
                    node_pair_list.append([i, j])
        prompt = f'''
I want to generate a Procedural Graph based on Procedural Document. To accurately generate a Procedural Graph, I use a NER tool to identify activities, signal words indicating branches, and conditions in the Procedural Document. They are arranged in the order in the Procedural Document. Each item is represented as "{{id: text}}", where id represents the graph node identifier and text represents the text information corresponding to the graph node.

You need to determine whether each item in the given node pair list is a reasonable edge. That is, each item in the list represents the number of any two nodes. You need to determine whether there is a direct edge between them based on the node numbers and their associated business texts, and explain the reason. Note that the results are output in JSON format. Each item corresponds to a node pair that needs to be judged. Each item includes two items: "result" and "explain". The value of "result" is 0 or 1. If there is a directly connected edge, it is 1, otherwise it is 0. "explain" is the explanation of the result. You only need to output the results according to the format without outputting any redundant content.

Here is an example:

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
Node Pair List:
[[0, 1], [1, 2], [1, 3]]

### 
Result:
[
    {{
    "result": 1,
    "explain":"‘The credit company receives the credit information from the customer’ indicates the beginning of the business process. ‘If’ appears directly after it, indicating the start of a branch event, so there is a direct edge between nodes 0 and 1."
    }},
    {{
        "result": 1,
        "explain":"'if' represents a branch, and the following 'the requested amount is greater than 1M $' is the condition of the branch, so there is a direct edge connecting nodes 1 and 2."
    }},
    {{
        "result": 0,
        "explain":"According to the Procedural Document, 'if' represents a branch, which is directly followed by a condition, i.e. 'the requested amount is greater than 1M $', and 'an approval must be requested' is executed only when the condition is met, so there is no direct edge connecting nodes 1 to 3, and the correct path is 1->2->3."
    }}
]

Now you need to refer to the given example and judge each item in the node pair list in turn according to the Procedural Document and the NER results.

###
"Procedural Document":
{text}

###
"NER results":
{node_to_text}

###
Node Pair List:
{node_pair_list}

### 
Result:

'''
        # print(prompt)
        while True:
            url = 'http://localhost:11434/api/generate'
            data = {
            'model': 'llama3.1:latest',
            'prompt': prompt,
            'format': 'json',
            'stream': True,
            "options": {
                    "num_ctx": 20480
                }
            }
            try:
                print(filename)
                response = requests.post(url, json=data)
                result = response.json()['response']
                print(result)
                # result = json.loads(result)['result']
                # relation_matrix[i][j] = int(result)
                # relation_matrix[j][i] = int(result)
                break
            except:
                pass
    #     dataset['relation_matrix'] = relation_matrix
    # dataset_list_json = json.dumps(dataset_list)
    # with open('/home/btr/bpmn/LLMEnG/my-data/datasets9-llama3.1-8B-1.json', 'w') as f:
    #     f.write(dataset_list_json)
get_llm_response()