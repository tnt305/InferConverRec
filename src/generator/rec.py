import sys
import os

# Thêm đường dẫn của thư mục gốc vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import json
from transformers import AutoTokenizer, AutoModel
from src.model_gpt2 import PromptGPT2forCRS
from src.model_prompt import KGPrompt
from src.dataset_rec import CRSRecDataset, CRSRecDataCollator
from src.dataset_dbpedia import DBpedia
from src.evaluate_rec import RecEvaluator
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
from src.config import gpt2_special_tokens_dict, prompt_special_tokens_dict

def load_recommendation_model(model_path, tokenizer_path, text_encoder_path, prompt_encoder_path):
    accelerator = Accelerator(device_placement=False, mixed_precision = 'fp16')
    device = accelerator.device

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    # Load model
    model = PromptGPT2forCRS.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    # Load text encoder
    text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(text_encoder_path)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)

    # Load KG
    kg = DBpedia(dataset="redial_gen").get_entity_kg_info()

    # Load prompt encoder
    prompt_encoder = KGPrompt(
        model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=8,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
        n_prefix_rec= 20
    )
    prompt_encoder.load(prompt_encoder_path)
    prompt_encoder = prompt_encoder.to(device)

    return model, tokenizer, text_tokenizer, text_encoder, prompt_encoder

def generate_recommendation(model, tokenizer, text_tokenizer,  text_encoder, prompt_encoder, accelerator):
    # Thiết lập các tham số
    ## khởi tạo cùng chat_utils
    # accelerator = Accelerator(device_placement=False, mixed_precision = 'fp16')
    # device = accelerator.device

    args = {
        "dataset": "redial_gen",
        "split": "sample_input",
        "num_workers": 4,
        "context_max_length": 128,
        "entity_max_length": 32,
        "prompt_max_length": 128,
        "n_prefix_rec": 20,
        "num_bases": 8,
        "per_device_eval_batch_size": 64,
    }

    # Tải KG
    kg = DBpedia(dataset=args["dataset"]).get_entity_kg_info()

    # Chuẩn bị dataset và dataloader
    dataset = CRSRecDataset(
        args["dataset"], 
        args["split"], 
        tokenizer = tokenizer,
        context_max_length=args["context_max_length"],
        prompt_tokenizer = text_tokenizer, 
        prompt_max_length=args["prompt_max_length"],
        entity_max_length=args.entity_max_length,
    )
    data_collator = CRSRecDataCollator(
        tokenizer=tokenizer, 
        device=accelerator.device,
        context_max_length=args["context_max_length"],
        entity_max_length=args["entity_max_length"],
        pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer= text_tokenizer, 
        prompt_max_length=args["prompt_max_length"]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args["per_device_eval_batch_size"],
        num_workers=args["num_workers"],
        collate_fn=data_collator,
    )

    # Chuẩn bị model và dataloader với Accelerator
    model, text_encoder, prompt_encoder, dataloader = accelerator.prepare(
        model, text_encoder, prompt_encoder, dataloader
    )

    # Sinh recommend
    model.eval()
    prompt_encoder.eval()
    text_encoder.eval()

    recommendations = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            token_embeds = text_encoder(**batch['prompt']).last_hidden_state
            prompt_embeds = prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                output_entity=True,
                use_rec_prefix=True
            )
            batch['context']['prompt_embeds'] = prompt_embeds
            batch['context']['entity_embeds'] = accelerator.unwrap_model(prompt_encoder).get_entity_embeds()

            outputs = model(**batch['context'], rec=True)
            logits = outputs.rec_logits[:, kg['item_ids']]
            ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
            ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
            labels = batch['context']['rec_labels']

            recommendations.extend(
                [
                    {
                        'gt': labels[i].tolist() if isinstance(labels[i], torch.Tensor) else labels[i],
                        'recommendation': ranks[i]
                    }
                    for i in range(len(labels))
                ]
            )

            # evaluator.evaluate(ranks, labels)
            unique_set = set(json.dumps(item, sort_keys=True) for item in total_list)

            # Convert JSON strings back to dictionaries
            unique_list = [json.loads(item) for item in unique_set]

    # Lấy kết quả đánh giá
    # metrics = evaluator.report()
    
    return recommendations[-1]  # Trả về recommend đầu tiên và metrics
