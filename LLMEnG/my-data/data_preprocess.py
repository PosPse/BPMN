import json
import requests
from tqdm import tqdm
dataset_list = None
with open ('/home/btr/bpmn/LLMEnG/my-data/datasets4.json', 'r') as f:
    dataset_list = json.load(f)
for dataset in tqdm(dataset_list):
    data_2_mask_single_signal_llm = dataset['data_2_mask_single_signal_llm']
    node_token_list = dataset['node_token_list']
    sign_index_list = [index for index, value in enumerate(data_2_mask_single_signal_llm) if value not in ['[activity]', '[condition]']]
    signal_token_llm_list = []
    prompt_1 = []
    prompt_2 = []
    for index in sign_index_list:
        start = ' '.join(node_token_list[:index])
        end = ' '.join(node_token_list[index+1:])
        medium = f'[{node_token_list[index]}]'
        p = ' '.join([start, medium, end])
        prompt_1.append(p)
        prompt_2.append(medium)
    for p1, p2 in zip(prompt_1, prompt_2):
        prompt = f'''
I want to extract activities from a business process text described in natural language and determine the relationship between two activities.

Below I give you a business process description text, in which the connecting words or phrases between activities are represented by [...] symbols.
"{p1}"

##Task
Please determine the type of relationship represented by the word or phrase {p2} and the context in which it is located, and give an explanation. There are 4 types of relationships, as follows:
[sign-selection]: Indicates a selection relationship between activities.
[sign-successor]: indicates a direct sequential relationship between activities
[sign-parallel]: indicates a concurrent relationship between activities.
[sign-loop]: indicates a loop relationship between activities.

##Response
The results are output in JSON format, including "result" and "explain". The value of "result" is the relationship type, such as "[sign-selection]". "explain" is the explanation of the result. You only need to output the results according to the format without outputting any redundant content.
'''
        url = 'http://localhost:11434/api/generate'
        data = {
            'model': 'llama3.1:latest',
            'prompt': prompt,
            'format': 'json',
            'stream': False,
        }
        try:
            response = requests.post(url, json=data)
            result = response.json()['response']
            result = json.loads(result)['result']
            signal_token_llm_list.append(result)
        except:
            dataset_list_json = json.dumps(dataset_list)
            with open('/home/btr/bpmn/LLMEnG/my-data/datasets6.json', 'w') as f:
                f.write(dataset_list_json)
    dataset['signal_token_llm_list'] = signal_token_llm_list
dataset_list_json = json.dumps(dataset_list)
with open('/home/btr/bpmn/LLMEnG/my-data/datasets6.json', 'w') as f:
    f.write(dataset_list_json)