# 使用LLM判断节点之间是否存在链接
import json
import requests
from tqdm import tqdm
def get_llm_response():
    dataset_list = None
    with open('/home/btr/bpmn/LLMEnG/my-data/datasets5.json', 'r') as f:
        dataset_list = json.load(f)

    for i in tqdm(range(1, len(dataset_list))):
        dataset = dataset_list[i]
        node_token_list = dataset['node_token_list']
        node_to_text = ''
        for i, node_token in enumerate(node_token_list):
            node_to_text += f'{i}: {node_token}\n'
        
        relation_matrix = [[0 for _ in range(len(node_token_list))] for _ in range(len(node_token_list))]
        for i in range(len(node_token_list)):
            for j in range(i+1, len(node_token_list)):
                prompt = f'''
    Below is a text describing a business process. I will draw a flowchart based on this text.
    The credit company receives the credit information from the customer , If the requested amount is greater than 1M $ an approval must be requested. If the requested amount is lower or equal to 1M $ the company assess the risk of the credit . If the requested amount is lower or equal to 1M $ the company assess the risk of the credit . After the assessment , if the risk is high , an approval must be requested ; but if the risk is low the credit is accepted . After the approval request , the credit could be accepted or rejected ; in both cases , an email is sent to the customer .

    Below is the above business process text in the diagram for the node information:

    {node_to_text}
    ##Task
    Determine whether node {i} and node {j} have directly connected edges based on the node information in the graph for the business process text and the service process text, and give a reason.

    ##Response
    The result is output in JSON format, including two items: "result" and "explain". The value of "result" is 0 or 1. If there is a directly connected edge, it is 1, otherwise it is 0. "explain" is the explanation of the result. You only need to output the results according to the format, and there is no need to output any redundant content.
    '''
                url = 'http://localhost:11434/api/generate'
                data = {
                'model': 'llama3.1:70b',
                'prompt': prompt,
                'format': 'json',
                'stream': False,
                }
                try:
                    print(dataset['filename'], len(node_token_list), i, j)
                    response = requests.post(url, json=data)
                    result = response.json()['response']
                    result = json.loads(result)['result']
                    relation_matrix[i][j] = int(result)
                    relation_matrix[j][i] = int(result)
                except:
                    dataset_list_json = json.dumps(dataset_list)
                    with open('/home/btr/bpmn/LLMEnG/my-data/datasets7.json', 'w') as f:
                        f.write(dataset_list_json)
                    return
        dataset['relation_matrix'] = relation_matrix
    dataset_list_json = json.dumps(dataset_list)
    with open('/home/btr/bpmn/LLMEnG/my-data/datasets7.json', 'w') as f:
        f.write(dataset_list_json)
get_llm_response()