import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, RobertaModel
from accelerate import Accelerator

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained models and tokenizers
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small").to(device)
text_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
text_encoder = RobertaModel.from_pretrained("roberta-base").to(device)

# Load and inspect your pre-trained prompt encoder
pre_trained_prompt_state = torch.load("/kaggle/working/InferConverRec/src/output_dir/dialogpt_prompt-pre_prefix-20_redial/best/model.pt", map_location=device)
print("Pre-trained prompt state keys:", pre_trained_prompt_state.keys())

# Load and inspect your trained prompts for conversation and recommendation
conv_prompt_encoder_state = torch.load("/kaggle/working/InferConverRec/src/output_dir/dialogpt_redial-resp/best/model.pt", map_location=device)
print("Conversation prompt encoder state keys:", conv_prompt_encoder_state.keys())

rec_prompt_encoder_state = torch.load("/kaggle/working/InferConverRec/src/output_dir/dialogpt_rec_redial/best/model.pt", map_location=device)
print("Recommendation prompt encoder state keys:", rec_prompt_encoder_state.keys())

# Function to create a linear layer from state dict
def create_linear_from_state(state_dict):
    if 'weight' in state_dict and 'bias' in state_dict:
        linear = torch.nn.Linear(state_dict['weight'].size(1), state_dict['weight'].size(0)).to(device)
        linear.load_state_dict(state_dict)
    else:
        # Assuming the state dict is the weight matrix itself
        linear = torch.nn.Linear(state_dict.size(1), state_dict.size(0), bias=False).to(device)
        linear.weight.data = state_dict
    return linear

# Create linear layers
pre_trained_prompt = create_linear_from_state(pre_trained_prompt_state)
conv_prompt_encoder = create_linear_from_state(conv_prompt_encoder_state)
rec_prompt_encoder = create_linear_from_state(rec_prompt_encoder_state)

# Set up Accelerator
accelerator = Accelerator()
model, text_encoder, pre_trained_prompt, conv_prompt_encoder, rec_prompt_encoder = accelerator.prepare(
    model, text_encoder, pre_trained_prompt, conv_prompt_encoder, rec_prompt_encoder
)

# Function to get recommendations
def get_recommendations(context):
    # Tokenize the context
    context_ids = text_tokenizer.encode(context, return_tensors="pt", max_length=200, truncation=True).to(device)
    context_embeds = text_encoder(context_ids).last_hidden_state
    
    # Generate pre-trained prompt
    pre_trained_prompt_embeds = pre_trained_prompt(context_embeds)
    
    # Prepare the recommendation prompt
    rec_prompt = rec_prompt_encoder(pre_trained_prompt_embeds)
    
    # Generate recommendation
    with torch.no_grad():
        rec_output = model.generate(
            inputs_embeds=rec_prompt,
            max_length=32,  # Adjust based on your entity_max_length
            num_return_sequences=3,  # Get top 3 recommendations
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode recommendations
    recommendations = [tokenizer.decode(rec, skip_special_tokens=True) for rec in rec_output]
    return recommendations

# Chatbot function
def chatbot(message, history):
    # Combine history and new message
    context = " ".join([f"{turn[0]} {turn[1]}" for turn in history]) + " " + message
    
    # Tokenize the input
    context_ids = text_tokenizer.encode(context, return_tensors="pt", max_length=200, truncation=True).to(device)
    context_embeds = text_encoder(context_ids).last_hidden_state
    
    # Generate pre-trained prompt
    pre_trained_prompt_embeds = pre_trained_prompt(context_embeds)
    
    # Prepare the conversation prompt
    conv_prompt = conv_prompt_encoder(pre_trained_prompt_embeds)
    
    # Generate a response
    with torch.no_grad():
        output = model.generate(
            inputs_embeds=conv_prompt,
            max_length=183,  # Based on your resp_max_length
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Get recommendations
    recommendations = get_recommendations(context + " " + response)
    
    # Append recommendations to the response
    response += "\n\nBased on our conversation, you might like these recommendations:\n"
    response += "\n".join([f"- {rec}" for rec in recommendations])
    
    return response

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
iface.launch()
