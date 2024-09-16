import os
import sys
import time
import json
import subprocess
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from time import time

from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_dbpedia import DBpedia
from dataset_conv import CRSConvDataCollator, CRSConvDataset
from dataset_rec import CRSRecDataset, CRSRecDataCollator
from evaluate_conv import ConvEvaluator
from evaluate_rec import RecEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt
import numpy
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'



class CRSInference():
    def __init__(self):
        self.accelerator = Accelerator(device_placement=False, mixed_precision = 'fp16')
        self.device = self.accelerator.device

        self.tokenizer = AutoTokenizer.from_pretrained('utils/dialogpt')
        self.tokenizer.add_special_tokens(gpt2_special_tokens_dict)
        
        # Initialize model
        self.model = PromptGPT2forCRS.from_pretrained('utils/dialogpt_model')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = self.model.to(self.device)
        
        # Initialize text encoder
        self.text_tokenizer = AutoTokenizer.from_pretrained('utils/roberta')
        self.text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
        self.text_encoder = AutoModel.from_pretrained('utils/roberta_model')
        self.text_encoder.resize_token_embeddings(len(self.text_tokenizer))
        self.text_encoder = self.text_encoder.to(self.device)

    
    def initialize_dynamic_components(self, types = 'conv'):
        assert types in ['conv', 'rec']
        if types == 'conv':
            kg = DBpedia(dataset='redial', debug=False).get_entity_kg_info()
            self.kg = kg
            self.prompt_encoder = KGPrompt(
            self.model.config.n_embd, self.text_encoder.config.hidden_size, 
            self.model.config.n_head, self.model.config.n_layer, 2,
            n_entity= kg['num_entities'], num_relations=kg['num_relations'], num_bases=8,
            edge_index=kg['edge_index'], edge_type=kg['edge_type'],
            n_prefix_conv=20
            )
            self.prompt_encoder.load('output_dir/prompt/dialogpt1e3/best')
            self.prompt_encoder = self.prompt_encoder.to(self.device)
        else:
            kg = DBpedia(dataset= 'redial_gen', debug=False).get_entity_kg_info()
            self.kg = kg
            self.prompt_encoder = KGPrompt(
            self.model.config.n_embd, self.text_encoder.config.hidden_size, 
            self.model.config.n_head, self.model.config.n_layer, 2,
            n_entity= kg['num_entities'], num_relations=kg['num_relations'], num_bases=8,
            edge_index=kg['edge_index'], edge_type=kg['edge_type'],
            n_prefix_conv=20
            )
            self.prompt_encoder.load('output_dir/prompt/dialogpt1e3/best')
            self.prompt_encoder = self.prompt_encoder.to(self.device)
        return self.prompt_encoder, self.kg


    def generate_conversation(self):
        prompt_encoder, kg = self.initialize_dynamic_components(types = 'conv')
        dataset = CRSConvDataset(
            dataset = 'redial', split = 'sample_input',  tokenizer = self.tokenizer, debug= False,
            context_max_length = 128, resp_max_length=81, entity_max_length=32,
            prompt_tokenizer = self.text_tokenizer, prompt_max_length= 128)

        data_collator_generator = CRSConvDataCollator(
            tokenizer = self.tokenizer, device= self.device, gen=True, use_amp = self.accelerator.use_fp16, debug= False,
            ignore_pad_token_for_loss= True,
            context_max_length= 128, resp_max_length=81,
            entity_max_length= 32, pad_entity_id= kg['pad_entity_id'],
            prompt_tokenizer= self.text_tokenizer
        )
        dataloader = DataLoader(
            dataset,
            batch_size= 32,
            num_workers = 4,
            collate_fn = data_collator_generator,
        )
        # gen_dir = os.path.join('save', 'redial')
        # os.makedirs(gen_dir, exist_ok=True)
        # model_name = 'output_dir/conv/dialogpt5e4/final'.split('/')[-2]
        # gen_file_path = os.path.join(gen_dir, f'{model_name}_sample_input.jsonl')
        # evaluator = ConvEvaluator(tokenizer= self.tokenizer, log_file_path=gen_file_path)
        for batch in tqdm(dataloader, disable= not self.accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds = self.text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=False,
                    use_conv_prefix=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds

                gen_seqs = self.accelerator.unwrap_model(self.model).generate(
                    **batch['context'],
                    max_new_tokens= 50,
                    no_repeat_ngram_size=3,
                    pad_token_id= self.tokenizer.pad_token_id,  # Add this line
                    eos_token_id=self.tokenizer.eos_token_id # add this also
                )
                gen_resp_ids = []
                for gen_seq, length in zip(gen_seqs, batch['context_len']):
                    gen_seq = [token_id for token_id in gen_seq if token_id != self.tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq[length:])
                
                for resp_ids in gen_resp_ids:
                    decoded_resp = self.tokenizer.decode(resp_ids, skip_special_tokens= False)
                    print("-----------------------------------")
                    print("Generated response:", decoded_resp)
                    print("-----------------------------------")

                # evaluator.evaluate(gen_resp_ids, batch['resp'], log = self.accelerator.is_local_main_process)

    def copy_and_merge(self):
        start = time()
        subprocess.run(["cp", "-r", "data/redial/.", "data/redial_gen/"], check=True)
        subprocess.run(["python", "data/redial_gen/merge.py", "--gen_file_prefix", "dialogpt5e4"], check=True)
        print(f"Elapsed time: {(time() - start):.5f} seconds")

    def generate_recommend(self):
        prompt_encoder, kg = self.initialize_dynamic_components(types = 'rec')
        test_dataset = CRSRecDataset(
            dataset= 'redial_gen', split='sample_input', debug= False,
            tokenizer= self.tokenizer, context_max_length= 128, use_resp= False,
            prompt_tokenizer= self.text_tokenizer, prompt_max_length= 128,
            entity_max_length= 32,
        )
        data_collator = CRSRecDataCollator(
            tokenizer= self.tokenizer, device= self.device, debug= False,
            context_max_length= 128, entity_max_length= 32,
            pad_entity_id=kg['pad_entity_id'],
            prompt_tokenizer= self.text_tokenizer, prompt_max_length= 128,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=64,
            collate_fn=data_collator,
        )
        evaluator = RecEvaluator()
        prompt_encoder, test_dataloader = self.accelerator.prepare(
            prompt_encoder, test_dataloader
        )

        # Only show the progress bar once on each machine.
        # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="Training..")

        # save model with best metric
        # metric, mode = 'loss', -1
        # assert mode in (-1, 1)
        # if mode == 1:
        #     best_metric = 0
        # else:
        #     best_metric = float('inf')
        # best_metric_dir = os.path.join('outputdir/rec', 'best')
        # os.makedirs(best_metric_dir, exist_ok=True)


        # test
        test_loss = []
        total_list = []
        prompt_encoder.eval()
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                token_embeds = self.text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=True,
                    use_rec_prefix=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = self.accelerator.unwrap_model(prompt_encoder).get_entity_embeds()

                outputs = self.model(**batch['context'], rec=True)
                test_loss.append(float(outputs.rec_loss))
                logits = outputs.rec_logits[:, kg['item_ids']]
                ranks = torch.topk(logits, k= 10, dim=-1).indices.tolist()
                ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                labels = batch['context']['rec_labels']
                total_list.extend(
                    [
                        {
                            'gt': labels[i].tolist() if isinstance(labels[i], torch.Tensor) else labels[i],
                            'recommendation': ranks[i].tolist() if isinstance(ranks[i], torch.Tensor) else ranks[i]
                        }
                    ]
                    for i in range(len(labels))
                )
                # evaluator.evaluate(ranks, labels)
