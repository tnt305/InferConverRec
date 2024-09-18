import torch
from transformers import AutoTokenizer, AutoModel
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt
from dataset_conv import CRSConvDataset, CRSConvDataCollator
from dataset_dbpedia import DBpedia
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
from config import gpt2_special_tokens_dict, prompt_special_tokens_dict

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class GenerateConversation:
    def __init__(self):
        # self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator(device_placement=False, mixed_precision='fp16')

        # Load KG
        self.kg = DBpedia(dataset= 'redial', debug= False).get_entity_kg_info()

        # Load tokenizers and models
        self.tokenizer = AutoTokenizer.from_pretrained('utils/dialogpt')
        self.tokenizer.add_special_tokens(gpt2_special_tokens_dict)
        self.model = PromptGPT2forCRS.from_pretrained('utils/dialogpt_model')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = self.model.to(self.device)

        self.text_tokenizer = AutoTokenizer.from_pretrained('utils/roberta')
        self.text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
        self.text_encoder = AutoModel.from_pretrained('utils/roberta_model')
        self.text_encoder.resize_token_embeddings(len(self.text_tokenizer))
        self.text_encoder = self.text_encoder.to(self.device)

        self.prompt_encoder = KGPrompt(
            self.model.config.n_embd, self.text_encoder.config.hidden_size, 
            self.model.config.n_head, self.model.config.n_layer, 2,
            n_entity=self.kg['num_entities'], num_relations=self.kg['num_relations'], 
            num_bases= 8, edge_index=self.kg['edge_index'], 
            edge_type=self.kg['edge_type'], n_prefix_conv= 20
        )
        
        self.prompt_encoder.load('/home/thiendc/InferConverRec/src/output_dir/conv/dialogpt1e3_2nd/best')
        self.prompt_encoder = self.prompt_encoder.to(self.device)
        self.prompt_encoder = self.accelerator.prepare(self.prompt_encoder)

        # Prepare dataset and dataloader
        self.dataset = CRSConvDataset(
            'redial', 'sample_input', self.tokenizer, debug= False,
            context_max_length=200, resp_max_length= 183,
            entity_max_length=32,
            prompt_tokenizer=self.text_tokenizer, prompt_max_length=128
        )
        self.data_collator = CRSConvDataCollator(
            tokenizer= self.tokenizer, device=self.device, gen=True, 
            use_amp= self.accelerator.use_fp16, debug= False,
            ignore_pad_token_for_loss= False,
            context_max_length=200, resp_max_length=183,
            entity_max_length=32, pad_entity_id=self.kg['pad_entity_id'],
            prompt_tokenizer=self.text_tokenizer
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size= 64,
            num_workers= 2,
            collate_fn=self.data_collator,
        )

        # Set up generation directory
        gen_dir = os.path.join('save', 'redial')
        os.makedirs(gen_dir, exist_ok=True)
        model_name = '/home/thiendc/InferConverRec/src/output_dir/conv/dialogpt1e3_2nd/best'.split('/')[-2]
        self.gen_file_path = os.path.join(gen_dir, f'{model_name}_sample_input.jsonl')

    def generate_conversations(self):
        self.prompt_encoder.eval()

        with torch.no_grad():
            for batch in tqdm(self.dataloader, disable=not self.accelerator.is_local_main_process):
                token_embeds = self.text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds = self.prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=False,
                    use_conv_prefix=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds

                gen_seqs = self.accelerator.unwrap_model(self.model).generate(
                    **batch['context'],
                    max_new_tokens= 64,
                    no_repeat_ngram_size=3,
                    temperature=0.9,
                    length_penalty=1.5,
                )
                gen_resp_ids = []
                for gen_seq, length in zip(gen_seqs, batch['context_len']):
                    gen_seq = [token_id for token_id in gen_seq if token_id != self.tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq[length:])
                
                decoded_resps = self.tokenizer.batch_decode(gen_resp_ids, skip_special_tokens=False)
                print("-----------------------------------")
                print("-----------------------------------")
                print("Generated response:", decoded_resps)

                # with open(self.gen_file_path, 'a') as f:
                #     for context, resp in zip(batch['raw_context'], decoded_resps):
                #         json.dump({'context': context, 'response': resp}, f)
                #         f.write('\n')
        return decoded_resps