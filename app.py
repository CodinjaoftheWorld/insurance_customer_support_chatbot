import pandas as pd
import openai
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pickle
import time
import streamlit as st
from streamlit import session_state
from streamlit import secrets
from PIL import Image
import requests
from io import BytesIO


_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')
# print(openai.api_key)
# openai.api_key = st.secrets["OPENAI_API_KEY"]

# print(openai_api_key)


# Masterdata path
masterdata_df_file_path = "/home/gaurav/Documents/LLMs/customer_support/final/masterdata/masterdata.pickle"
index_file_path = "/home/gaurav/Documents/LLMs/customer_support/final/masterdata/index.pickle"


def main():
    load_dotenv()
    st.title("ChatGPT-like clone")

    # Access the OpenAI API key
    # openai_api_key = st.secrets["openai_api_key"]

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

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
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})





if __name__=="__main__":
    main()