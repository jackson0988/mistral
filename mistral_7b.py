import os
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="CohereLLM", page_icon=":rocket:", layout="wide")

# Initialize the chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.markdown("<h1 style='text-align:center'>MistralLLM</h1>", unsafe_allow_html=True)
st.write("<h3 style='text-align:center'>This app uses a pre-trained Mistral model to generate text.</h3>", unsafe_allow_html=True)

# Create a container for the chat history and input area
container = st.container()

# Divide the container into two columns
col1, col2 = container.columns([2, 1]) # 2/3 for chat history, 1/3 for input area

# Display chat history in the left column
with col1:
    for message in st.session_state["chat_history"]:
        if message["role"] == "user":
            st.write("You: ", message["content"])
        else:
            st.write("Assistant: ", message["content"])

# Input area in the right column
with col2:
    user_input = st.text_input("Enter your message:", key="user_input", placeholder="Type your message here...")

    if user_input:
        # Add user input to chat history
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        # Initialize the LLM
        model_path = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

        # Generate response based on the entire chat history
        chat_history_strings = [f"{message['role']}: {message['content']}" for message in st.session_state["chat_history"]]
        prompt = "\n".join(chat_history_strings)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        response = model.generate(input_ids, max_length=2500, temperature=0.3, num_return_sequences=1)
        response_text = tokenizer.decode(response[0], skip_special_tokens=True)

        # Add assistant response to chat history
        st.session_state["chat_history"].append({"role": "assistant", "content": response_text})

        # Clear the user input area
        st.session_state["user_input"] = ""
