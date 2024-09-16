import random
import json
import os
import re
import subprocess
from test_inference import CRSInference
##### Try to extract from dbpedia raw instead of entity


def get_entities_and_ids():
  # Opening JSON file
    with open('/home/thiendc/projects/InferConverRec/src/data/redial/entity2id.json') as json_file:
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
    single_user_path = '/home/thiendc/projects/InferConverRec/src/data/redial/sample_input_data_processed.jsonl'
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

    with open(single_user_path, 'w') as outfile:
        jout = json.dumps(current_str)
        outfile.write(jout)

    return doesMovieExist

def from_pred_output_resp_to_input():
    single_user_path = '/home/thiendc/projects/InferConverRec/src/data/redial_gen/sample_input_data_processed.jsonl'
    pred_reply = "/home/thiendc/projects/InferConverRec/src/save/redial/conv_sample_input.jsonl"
    
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

# Global CRSInference object
crs_inference = None

def initialize_chatbot(device):
    global crs_inference
    if crs_inference is None:
        crs_inference = CRSInference()
    print("CRS Inference initialized.")

def run_inference():



