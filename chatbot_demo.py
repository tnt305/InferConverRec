import numpy as np
import json
import os
import re
import gradio 
import random
from src.chat_utils import chat


os.environ['MKL_THREADING_LAYER']= 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

apologise_error_404 = [
    "Sorry, I couldn't find any information on that.",
    "I apologize, but I don't have details on that movie.",
    "Unfortunately, I couldn't locate that entity."
]

def chatbot(user_input, chat_history):
    try:
        response = chat(user_input)
        chat_history.append((user_input, response))
        return response
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        chat_history.append((user_input, error_message))
        return error_message

demo = gradio.ChatInterface(
    fn= chatbot,
    title="Movie Recommendation Chatbot",
    description="Chat about movies and get recommendations. Mention movies by enclosing them in dollar signs, like $Movie Title$",
    examples=[  "Hi, can you recommend a $fantasy$ movie?", 
                "Do you have any movies similar to $The avengers$?", 
                "Can I check with you if you have any movies that has $Brad Pitt$ like $Fight Club$ ?",
                "Hello, i need help"],
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear Chat",
)

if __name__ == "__main__":
    demo.launch(share = True)
    
