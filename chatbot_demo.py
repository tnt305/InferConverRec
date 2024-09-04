import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, RobertaModel, AutoModel
from accelerate import Accelerator

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained models and tokenizers
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small").to(device)
tokenizer.add_special_tokens({
    'pad_token': '<pad>',
    'additional_special_tokens': ['<movie>'],
})
model.config.pad_token_id = tokenizer.pad_token_id

text_tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
text_tokenizer.add_special_tokens({
    'additional_special_tokens': ['<movie>'],
})
text_encoder = RobertaModel.from_pretrained("FacebookAI/roberta-base")
text_encoder = text_encoder.to(device)

# Load and inspect your pre-trained prompt encoder
pre_trained_prompt_state = torch.load("/kaggle/working/InferConverRec/src/output_dir/dialogpt_prompt-pre_prefix-20_redial/best/model.pt", map_location=device)
conv_prompt_encoder_state = torch.load("/kaggle/working/InferConverRec/src/output_dir/dialogpt_redial-resp/best/model.pt", map_location=device)
rec_prompt_encoder_state = torch.load("/kaggle/working/InferConverRec/src/output_dir/dialogpt_rec_redial/best/model.pt", map_location=device)

# Function to create a linear layer from state dict
def create_linear_from_state(state_dict):
    if isinstance(state_dict, dict):
        if 'weight' in state_dict:
            weight = state_dict['weight']
            bias = state_dict.get('bias')
        else:
            weight = next(iter(state_dict.values()))
            bias = None
    else:
        weight = state_dict
        bias = None

    if isinstance(weight, dict):
        weight = next(iter(weight.values()))

    if not isinstance(weight, torch.Tensor):
        raise ValueError(f"Weight must be a tensor, got {type(weight)}")

    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=(bias is not None)).to(device)
    with torch.no_grad():
        linear.weight.copy_(weight)
        if bias is not None:
            linear.bias.copy_(bias)
    return linear

# Create linear layers
pre_trained_prompt = create_linear_from_state(pre_trained_prompt_state)
conv_prompt_encoder = create_linear_from_state(conv_prompt_encoder_state)
rec_prompt_encoder = create_linear_from_state(rec_prompt_encoder_state)

# Print shapes for debugging
print("Pre-trained prompt weight shape:", pre_trained_prompt.weight.shape)
print("Conv prompt encoder weight shape:", conv_prompt_encoder.weight.shape)
print("Rec prompt encoder weight shape:", rec_prompt_encoder.weight.shape)

# Set up Accelerator
accelerator = Accelerator()
model, text_encoder, pre_trained_prompt, conv_prompt_encoder, rec_prompt_encoder = accelerator.prepare(
    model, text_encoder, pre_trained_prompt, conv_prompt_encoder, rec_prompt_encoder
)

# Function to get recommendations
def get_recommendations(context):
    try:
        context_ids = text_tokenizer.encode(context, return_tensors="pt", max_length=200, truncation=True).to(device)
        context_embeds = text_encoder(context_ids).last_hidden_state
        
        print("Context embeds shape:", context_embeds.shape)
        
        # Adjust dimensions
        pre_trained_prompt_embeds = pre_trained_prompt(context_embeds.mean(dim=1).unsqueeze(1))
        print("Pre-trained prompt embeds shape:", pre_trained_prompt_embeds.shape)
        
        rec_prompt = rec_prompt_encoder(pre_trained_prompt_embeds)
        print("Rec prompt shape:", rec_prompt.shape)
        
        with torch.no_grad():
            rec_output = model.generate(
                inputs_embeds=rec_prompt,
                max_length=32,
                num_return_sequences=3,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        recommendations = [tokenizer.decode(rec, skip_special_tokens=True) for rec in rec_output]
        return recommendations
    except Exception as e:
        print(f"Error in get_recommendations: {e}")
        return ["Error occurred while generating recommendations."]

# Chatbot function
def chatbot(message, history):
    try:
        context = " ".join([f"{turn[0]} {turn[1]}" for turn in history]) + " " + message
        
        context_ids = text_tokenizer.encode(context, return_tensors="pt", max_length=200, truncation=True).to(device)
        context_embeds = text_encoder(context_ids).last_hidden_state
        print('Context embed shape:', context_embeds.shape)
        
        # Adjust dimensions
        pre_trained_prompt_embeds = pre_trained_prompt(context_embeds.mean(dim=1).unsqueeze(1))
        print('Pre-trained prompt embeds shape:', pre_trained_prompt_embeds.shape)
        
        conv_prompt = conv_prompt_encoder(pre_trained_prompt_embeds)
        print('Conv prompt shape:', conv_prompt.shape)
        
        with torch.no_grad():
            output = model.generate(
                inputs_embeds=conv_prompt,
                max_length=183,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        recommendations = get_recommendations(context + " " + response)
        
        response += "\n\nBased on our conversation, you might like these recommendations:\n"
        response += "\n".join([f"- {rec}" for rec in recommendations])
        
        return response
    except Exception as e:
        print(f"Error in chatbot: {e}")
        return "An error occurred while processing your message. Please try again."

# Create the Gradio interface
iface = gr.ChatInterface(
    chatbot,
    title="Conversational Recommendation Chatbot",
    description="This is a demo of a conversational recommendation system with pre-trained prompts. Ask me about movies, books, or activities!",
    examples=[
        "Can you recommend a good movie?",
        "What's a fun activity for the weekend?",
        "I'm looking for a new book to read. I enjoy science fiction."
    ],
    cache_examples=True
)

# Launch the interface
iface.launch(share = True)
