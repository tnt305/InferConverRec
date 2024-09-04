import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, RobertaModel
from accelerate import Accelerator

# Load your trained models and tokenizers
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
text_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
text_encoder = RobertaModel.from_pretrained("roberta-base")

# Load your trained prompts
conv_prompt_encoder = torch.load("/path/to/prompt for conversation")
rec_prompt_encoder = torch.load("/path/to/prompt for recommendation")

# Set up Accelerator
accelerator = Accelerator()
model, text_encoder, conv_prompt_encoder, rec_prompt_encoder = accelerator.prepare(
    model, text_encoder, conv_prompt_encoder, rec_prompt_encoder
)

# Function to get recommendations
def get_recommendations(context):
    # Tokenize the context
    context_ids = text_tokenizer.encode(context, return_tensors="pt", max_length=200, truncation=True)
    
    # Prepare the recommendation prompt
    rec_prompt = rec_prompt_encoder(context_ids)
    
    # Generate recommendation
    with torch.no_grad():
        rec_output = model.generate(
            input_ids=context_ids,
            max_length=32,  # Adjust based on your entity_max_length
            num_return_sequences=3,  # Get top 3 recommendations
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones(context_ids.shape, dtype=torch.long),
            prefix=rec_prompt
        )
    
    # Decode recommendations
    recommendations = [text_tokenizer.decode(rec, skip_special_tokens=True) for rec in rec_output]
    return recommendations

# Chatbot function
def chatbot(message, history):
    # Combine history and new message
    context = " ".join([f"{turn[0]} {turn[1]}" for turn in history]) + " " + message
    
    # Tokenize the input
    input_ids = tokenizer.encode(context + tokenizer.eos_token, return_tensors="pt", max_length=200, truncation=True)
    
    # Prepare the conversation prompt
    conv_prompt = conv_prompt_encoder(input_ids)
    
    # Generate a response
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=183,  # Based on your resp_max_length
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones(input_ids.shape, dtype=torch.long),
            prefix=conv_prompt
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
    description="This is a demo of a conversational recommendation system. Ask me about movies, books, or activities!",
    examples=[
        "Can you recommend a good movie?",
        "What's a fun activity for the weekend?",
        "I'm looking for a new book to read. I enjoy science fiction."
    ],
    cache_examples=True
)

# Launch the interface
iface.launch()
