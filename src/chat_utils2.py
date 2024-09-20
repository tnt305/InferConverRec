import random
import json
import os
import re
import subprocess
from accelerate import Accelerator
import json
import random
import multiprocessing

from refinement import *
from generator.conv import GenerateConversation
from generator.rec import GenerateRecommendation
from generator.hooks import hook_sentences ,promotional_sentences
from fuzzywuzzy import fuzz
multiprocessing.set_start_method('spawn', force=True)

# def get_entities_and_ids():
#   # Opening JSON file
#     with open('/home/thiendc/InferConverRec/src/data/redial/entity2id.json') as json_file:
#         data = json.load(json_file)

#     entity_id = {}
#     id_entity = {}
#     for k, v in data.items():
#         k = k.split("/")[-1]
#         k = k.replace(">", "")
#         if "(" and ")" in k:
#             k=re.sub("[\(\[].*?[\)\]]", "", k)
#         k = k.replace("_", " ")
#         k = k.strip()
#         k = k.lower()

#         entity_id[k] = v
#         id_entity[v] = k

#     return entity_id, id_entity
# entity_id, id_entity = get_entities_and_ids()
# all_entities = list(entity_id.keys())
apologise_error_404 = [
    "Sorry, I don't seem to know the person or movie you are talking about!",
    "Ah, my circuits seem to have been fried. I can't find the movie / person you are referring to!",
    "Sorry, error 404, that movie / person has not been found.",
    "I apologise, but can you please check your spelling. I can't comprehend what you are saying!",
]

# def chat(input_str):
#     doesEntityExist = input_to_jsonl(input_str)
#     print("***************")
#     print("Does entity exist:", doesEntityExist)
#     print("***************")
#     if doesEntityExist:
#         run_inference()
#         return read_from_jsonl_to_user()
#     else:
#         return random.choice(apologise_error_404)

RECOMMEND_GPUS = [6, 5, 4, 3 , 2]  # GPUs dành cho task recommend

def run_infer_rec_on_specific_gpus():
    # Lưu giá trị CUDA_VISIBLE_DEVICES hiện tại
    original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    
    try:
        # Sử dụng chính xác 4 GPU đã chỉ định
        gpu_list = ','.join(map(str, RECOMMEND_GPUS))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        
        print(f"Running infer_rec.py on GPUs {gpu_list}")
        
        # Chạy infer_rec.py
        subprocess.run([
            "accelerate", "launch", 
            "--num_processes", "4",  # Chính xác 4 processes cho 4 GPU
            "--multi_gpu",  # Thêm flag này để sử dụng nhiều GPU
            "infer_rec.py",
            "--dataset", "redial_gen",
            "--tokenizer", "./utils/dialogpt",
            "--model", "./utils/dialogpt_model",
            "--text_tokenizer", "./utils/roberta",
            "--text_encoder", "./utils/roberta_model",
            "--n_prefix_rec", "20",
            "--prompt_encoder", "./output_dir/rec1/best",
            "--per_device_eval_batch_size", "16",
            "--gradient_accumulation_steps", "2",
            "--context_max_length", "128",
            "--prompt_max_length", "128",
            "--entity_max_length", "32",
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running infer_rec.py: {e}")
        # Xử lý lỗi ở đây nếu cần
    finally:
        # Khôi phục lại giá trị CUDA_VISIBLE_DEVICES ban đầu
        if original_cuda_devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)

# def clear_cached():
#     file_path = "./data/redial/sample_input_data_processed.jsonl"
#     try:
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             return "The garbage has been cleared!"
#         else:
#             return "Cache has already been cleared!"
#     except Exception as e:
#         return f"An error occurred while clearing cache: {str(e)}"
global_conv_module = None

def initialize_conv_module():
    global global_conv_module
    if global_conv_module is None:
        global_conv_module = GenerateConversation()

def chat(input_str):
    global global_conv_module
    doesEntityExist = input2jsonl(input_str, entity2id_json = './data/redial_gen/entity2id.json')    

    print("***************")
    print("Does entity exist:", doesEntityExist)
    print("***************")
    
    if doesEntityExist:
        # Initialize conv_module if it hasn't been initialized yet
        initialize_conv_module()
        # Use the global conv_module
        predict_conversation = global_conv_module.generate_conversations()
        print('Đây là conversation', predict_conversation) # list

        save_path = './data/redial/sample_input_data_processed.jsonl'
        try:
            with open(save_path, 'r') as json_file:
                data = json.load(json_file)
        except json.JSONDecodeError:
            # If the file is empty or invalid JSON, create a new dictionary
            data = {"context": [], "resp": "", "rec": [], "entity": []}
        if isinstance(predict_conversation, list):
            predict_conversation = ' '.join(predict_conversation)

        predict_conversation = rewrite2(predict_conversation)
        predict_conversation = predict_conversation.replace("<|endoftext|>", "")
        predict_conversation = predict_conversation.replace("System:", "")
        
        if predict_conversation.strip() == "":
            predict_conversation = random.choice(hook_sentences)
        
        if is_valid_sentence(predict_conversation):
            predict_conversation = rewrite(predict_conversation)

        data['resp'] = predict_conversation  

        with open(save_path, 'w') as json_file:
            json.dump(data, json_file)

        subprocess.run(["cp", "-r", "data/redial/.", "data/redial_gen/"], check=True)

        # rec_module.generate_recommendation()
        run_infer_rec_on_specific_gpus()
        # subprocess.run([
        #     "accelerate", "launch", 
        #     "--num_processes", "2",  # Chính xác 4 processes cho 4 GPU
        #     "--multi_gpu",  # Thêm flag này để sử dụng nhiều GPU
        #     "infer_rec.py",
        #     "--dataset", "redial_gen",
        #     "--tokenizer", "/home/thiendc/InferConverRec/src/utils/dialogpt",
        #     "--model", "/home/thiendc/InferConverRec/src/utils/dialogpt_model",
        #     "--text_tokenizer", "/home/thiendc/InferConverRec/src/utils/roberta",
        #     "--text_encoder", "/home/thiendc/InferConverRec/src/utils/roberta_model",
        #     "--n_prefix_rec", "20",
        #     "--prompt_encoder", "/home/thiendc/InferConverRec/src/output_dir/rec1/best",
        #     "--per_device_eval_batch_size", "16",
        #     "--gradient_accumulation_steps", "2",
        #     "--context_max_length", "128",
        #     "--prompt_max_length", "128",
        #     "--entity_max_length", "32",
        # ], check=True)

        # Rec list
        with open('/home/thiendc/InferConverRec/src/data/redial_gen/movie_ids.json', 'r') as moviesl:
            list_movies = json.load(moviesl)
        with open('/home/thiendc/InferConverRec/src/recommedations.json', 'r') as f:
            movie_rec = json.load(f)
        movie_rec = movie_rec[-1][0]
        data['rec'] = [i for i in movie_rec['recommendation'] if i in list_movies][:predict_conversation.count("<movie>")]
        response = data['resp']
        recommended_movies = data['rec']

        with open('/home/thiendc/InferConverRec/src/data/redial/entity2id.json', 'r') as f:
            movielink2id =  json.load(f)
        id2movielink = {v: k for k, v in movielink2id.items()}

        for movie_id in recommended_movies:
            if '<movie>' in response:
                if isinstance(movie_id, list):
                    movie_id = movie_id[0]  # Take the first item if it's a list
                movie_link = id2movielink.get(movie_id, str(movie_id))  # Use get() with a default value
                movie_name = movie_link.rstrip("/").split('/')[-1].strip('>').replace("_", " ")
                response = response.replace("<movie>", movie_name, 1)  # Replace only one occurrence
            else:
                response = response + " " + random.choice(promotional_sentences)
                movie_link = id2movielink.get(movie_id, str(movie_id))
                movie_name = movie_link.rstrip("/").split('/')[-1].strip('>').replace("_", " ")
                response = response + " " + movie_name + "."
        
        with open('/home/thiendc/InferConverRec/src/data/redial/sample_input_data_processed.jsonl', 'r') as file:
            line = file.readline().strip()
        data = json.loads(line)
        data['context'].append(response)
        with open('/home/thiendc/InferConverRec/src/data/redial/sample_input_data_processed.jsonl', 'w') as file:
            json.dump(data, file)
        
        # clear_cached()
        return response

    else:
        response = random.choice(apologise_error_404)
        with open('/home/thiendc/InferConverRec/src/data/redial/sample_input_data_processed.jsonl', 'r') as file:
            line = file.readline().strip()
        data = json.loads(line)
        data['context'].append(response)
        with open('/home/thiendc/InferConverRec/src/data/redial/sample_input_data_processed.jsonl', 'w') as file:
            json.dump(data, file)
        # clear_cached()
        return response

def get_entities_and_ids(json_file_path):
    with open(json_file_path, 'r') as json_file:
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

    return entity_id

def fuzzy_search_entities(entities):
    matched_entities = []
    for entity in entities:
        similarities = [(entity_name, fuzz.ratio(entity, entity_name)) for entity_name in get_entities_and_ids('/home/thiendc/InferConverRec/src/data/redial_gen/entity2id.json').keys()]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = [entity_name for entity_name, similarity in similarities[:5] if similarity > 60]
        matched_entities.extend(top_matches)
    return matched_entities

def input2jsonl(input_str, entity2id_json):
    save_path = '/home/thiendc/InferConverRec/src/data/redial/sample_input_data_processed.jsonl'
    is_exist = os.path.exists(save_path)

    current_str = ''

    if is_exist:
        with open(save_path, 'r') as json_file:
            current_str = json.load(json_file)
            current_str['context'].append(input_str)
    else:
        current_str = {"context": [input_str], "resp": "", "rec": [], "entity": []}
    
    entities = re.findall(r'\$.*?\$', input_str)
    if entities:
        entities = [entity.replace('$', '').lower() for entity in entities]
        # list_entities = set(get_entities_and_ids(entity2id_json))

        matched_entities = fuzzy_search_entities(entities)
        entity_id = get_entities_and_ids('/home/thiendc/InferConverRec/src/data/redial_gen/entity2id.json')
        if matched_entities:

            entity_ids = [entity_id[entity] for entity in matched_entities if entity in get_entities_and_ids(entity2id_json)]
            # random_text= random.choice(['chose', 'picked', 'consider', 'found'])
            print("-------------------------------------------------------------------")
            # print(f"Movies that you {random_choose} earlier: {', '.join(matched_entities)}")
            # print(f"Corresponding IDs : {', '.join(map(str, entity_ids))}")
            # print("-------------------------------------------------------------------")
            current_str['rec'].extend(entity_ids)

            does_movie_exist = True
        else:
            does_movie_exist = False
    else:
        does_movie_exist = False
    
    with open(save_path, 'w') as outfile:
        jout = json.dumps(current_str)
        outfile.write(jout)
    
    return does_movie_exist
