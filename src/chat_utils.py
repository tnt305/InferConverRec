import random
import json
import os
import re
import subprocess
##### Try to extract from dbpedia raw instead of entity


def get_entities_and_ids():
  # Opening JSON file
    with open('/home/thiendc/InferConverRec/src/data/redial/entity2id.json') as json_file:
        data = json.load(json_file)

    entity_id = {}
    id_entity = {}
    for k, v in data.items():
        k = k.split("/")[-1]
        k = k.replace(">", "")
        if "(" and ")" in k:
            k=re.sub("[\(\[].*?[\)\]]", "", k)
        k = k.replace("_", " ")
        k = k.strip()
        k = k.lower()

        entity_id[k] = v
        id_entity[v] = k

    return entity_id, id_entity
entity_id, id_entity = get_entities_and_ids()
all_entities = list(entity_id.keys())
apologise_error_404 = [
    "Sorry, I don't seem to know the person or movie you are talking about!",
    "Ah, my circuits seem to have been fried. I can't find the movie / person you are referring to!",
    "Sorry, error 404, that movie / person has not been found.",
    "I apologise, but can you please check your spelling. I can't comprehend what you are saying!",
]

def input_to_jsonl(input_str):
    single_user_path = '/home/thiendc/InferConverRec/src/data/redial/sample_input_data_processed.jsonl'
    isExisting = os.path.exists(single_user_path)
    current_str = ''
    
    if isExisting:
        with open(single_user_path, 'r') as json_str:
            current_str = json.load(json_str)
            print(f"current_str: {current_str}")
            # Add new input string from user
            current_str['context'].append(input_str)
    else:
        current_str = {"context": [input_str], "resp": "", "rec": [], "entity": []}
    
    doesMovieExist = True
    # Check string for special regex: regex indicates movies $Angelina Jolie$ $Transformers$
    result = re.findall(r'\$.*?\$', input_str)
    
    if result:
        result = [i.replace('$', '').lower() for i in result]
        movie_ids = []

    print("************************************************************")
    print("MOVIE IS", result)
    print("************************************************************")
    
    for r in result:
        if r not in all_entities:
            doesMovieExist = False
        for k, v in entity_id.items():
            if k == r:
                print("************************************************************")
                print("MOVIE ID IS", v)
                print("************************************************************")
                current_str['rec'].append(v)
                #current_str['entity'].append(v)

    with open(single_user_path, 'w') as outfile:
        jout = json.dumps(current_str)
        outfile.write(jout)

    return doesMovieExist

def from_pred_output_resp_to_input():
    single_user_path = '/home/thiendc/InferConverRec/src/data/redial_gen/sample_input_data_processed.jsonl'
    pred_reply = "/home/thiendc/InferConverRec/src/save/redial/dialogpt5e4_sample_input.jsonl"
    
    # Read prediction data
    with open(pred_reply, 'r') as json_str:
        pred_str = json.load(json_str)
        print(f"pred_str: {pred_str}")

    curr_str = {}
    with open(single_user_path, 'r') as outfile:
        curr_str = json.load(outfile)
        print(f"curr_str: {curr_str}")
    outfile.close()

    curr_str['resp'] = pred_str['pred']

    with open(single_user_path, 'w') as outfile:
        jout = json.dumps(curr_str)
        outfile.write(jout)
    outfile.close()


def run_inference():
    subprocess.run([
        "python", "infer_conv.py",
        "--dataset", "redial",
        "--split", "sample_input",
        "--tokenizer", "utils/dialogpt",
        "--model", "utils/dialogpt_model",
        "--text_tokenizer", "utils/roberta",
        "--text_encoder", "utils/roberta_model",
        "--n_prefix_conv", "20",
        "--prompt_encoder", "/home/thiendc/InferConverRec/src/output_dir/conv/dialogpt5e4/best",
        "--per_device_eval_batch_size", "128",
        "--context_max_length", "200",
        "--resp_max_length", "183",
        "--prompt_max_length", "128",
        "--entity_max_length", "32"
    ], check=True)

    # Change directory
    # os.chdir("/home/thiendc/projects/InferConverRec/src")
    
    # Copy files
    subprocess.run(["cp", "-r", "data/redial/.", "data/redial_gen/"], check=True)
    
    # Run merge.py
    subprocess.run(["python", "data/redial_gen/merge.py", "--gen_file_prefix", "dialogpt5e4"], check=True)
    
    from_pred_output_resp_to_input()

    subprocess.run([
        "accelerate", "launch", "--num_processes", "4",
         "infer_rec.py",
        "--dataset", "redial_gen",
        "--tokenizer", "microsoft/DialoGPT-small",
        "--model", "microsoft/DialoGPT-small",
        "--text_tokenizer", "roberta-base",
        "--text_encoder", "roberta-base",
        "--n_prefix_rec", "20",
        "--prompt_encoder", "/home/thiendc/InferConverRec/src/output_dir/rec1/best",
        "--num_train_epochs", "5",
        "--per_device_train_batch_size", "32",
        "--per_device_eval_batch_size", "64",
        "--gradient_accumulation_steps", "1",
        "--num_warmup_steps", "530",
        "--context_max_length", "128",
        "--prompt_max_length", "200",
        "--entity_max_length", "32",
        "--learning_rate", "5e-5",
    ], check=True)

def read_from_jsonl_to_user():
    single_user_path = '/home/thiendc/InferConverRec/src/data/redial_gen/sample_input_data_processed.jsonl'
    isExisting = os.path.exists(single_user_path)
    if isExisting:
        with open(single_user_path, 'r') as json_str:
            current_str = json.load(json_str)
            print(f"current_str: {current_str}")
    
    movie_count = 0
    current_str['resp'] = current_str['resp'].replace('System: ', '')
    print("------------------------------------")
    print('ĐÂy là current string', current_str)
    print("------------------------------------")
    if "<movie>" in current_str['resp']:
        split_str = current_str['resp'].split("<movie>")
        final_str = []

        for idx, j in enumerate(split_str):
            if idx != len(split_str)-1:
                final_str.append(j)
                if current_str['entity'] == []:
                    final_str.append(random.choice(all_entities))
                else:
                    final_str.append(id_entity[current_str['entity'][-movie_count-1]])
                movie_count +=1
            else:
                final_str.append(j)

        # current_str['resp'] = final_str
        # last_str = ' '.join(final_str)
        # print("------------------------------------")
        # print('Đây là text cuối cùng', last_str)
        # print("------------------------------------")
        return ' '.join(final_str)
    else:
        return current_str['resp']

def chat(input_str):
    doesEntityExist = input_to_jsonl(input_str)
    print("***************")
    print("Does entity exist:", doesEntityExist)
    print("***************")
    if doesEntityExist:
        run_inference()
        return read_from_jsonl_to_user()
    else:
        return random.choice(apologise_error_404)