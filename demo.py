import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load your trained model and tokenizer
model_name = "microsoft/DialoGPT-small"  # Replace with your trained model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the function that generates responses
def generate_response(user_input, chat_history=[]):
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([torch.tensor(chat_history), new_user_input_ids], dim=-1) if chat_history else new_user_input_ids
    
    chat_history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return response, chat_history

# Set up the Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message here:")
    clear = gr.Button("Clear")

    def respond(message, chat_history):
        response, chat_history = generate_response(message, chat_history)
        chat_history = chat_history.tolist()
        chatbot.update(response)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the Gradio app
demo.launch()
