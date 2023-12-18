import streamlit as st
from llama import BasicModelRunner
import lamini
import time

lamini.api_key = "e0e10d2e1a21a8b0b15078556006171de05f1782286815512cb0c36a6b9989da"

# List of LLAMA models
llama_models = [
    "hf-internal-testing/tiny-random-gpt2",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-70m-v0",
    "EleutherAI/pythia-70m-deduped-v0",
    "EleutherAI/neox-ckpt-pythia-70m-deduped-v0",
    "EleutherAI/neox-ckpt-pythia-70m-v1",
    "EleutherAI/neox-ckpt-pythia-70m-deduped-v1",
    "EleutherAI/gpt-neo-125m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-160m-deduped-v0",
    "EleutherAI/neox-ckpt-pythia-70m",
    "EleutherAI/neox-ckpt-pythia-160m",
    "EleutherAI/neox-ckpt-pythia-160m-deduped-v1",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-410m-v0",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-410m-deduped-v0",
    "EleutherAI/neox-ckpt-pythia-410m",
    "EleutherAI/neox-ckpt-pythia-410m-deduped-v1",
    "cerebras/Cerebras-GPT-111M",
    "cerebras/Cerebras-GPT-256M",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "Intel/neural-chat-7b-v3-1",
    "mistralai/Mistral-7B-Instruct-v0.1"
]


# Dropdown for selecting LLAMA model
selected_model = st.selectbox("Select LLAMA Model", llama_models)

# Initialize LLAMA model
llm = BasicModelRunner(selected_model)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get LLAMA response
    assistant_response = llm(prompt)

    # Display assistant response in chat message container with streaming effect
    with st.empty():
        full_response = ""
        for chunk in assistant_response.split():
            full_response += chunk + " "
            st.markdown(full_response + "▌")
            time.sleep(0.1)  # Adjust the sleep duration for the desired speed

        st.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
