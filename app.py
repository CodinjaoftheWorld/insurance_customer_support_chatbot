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
masterdata_df_file_path = "/home/gaurav/Documents/LLMs/insurance_customer_support_chatbot/output/insurance_masterdata.pickle"
index_file_path = "/home/gaurav/Documents/LLMs/insurance_customer_support_chatbot/output/index.pickle"


def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_masterdata_pickle(path):
    df = pd.read_pickle(path)
    return df

def get_context(index_file_path, prompt, masterdata_df_file_path, k):
    index = load_pickle_file(index_file_path)
    query_embedding = openai.Embedding.create(input = [prompt], model="text-embedding-ada-002")['data'][0]['embedding']
    k = k
    distance, indices = index.search(np.array([query_embedding]), k)
    master_df = load_masterdata_pickle(masterdata_df_file_path)
    top_k_prompt = "".join(master_df.iloc[indices[0].flatten()]['text'].tolist())
    return top_k_prompt

def process_content(text):
    return text.split("}<question/>")[0].split("{")[-1]

def reset_conversation():
  st.session_state.messages = []
  


def main():
    load_dotenv()
    st.title("Your Insurance Assistant! 🛡️🚗🏥")

    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: DarkRed;
        }
    </style>
    """, unsafe_allow_html=True) 


    with st.sidebar:
        c1, c2, c3 = st.columns([0.1, 0.8, 0.1])
        with c2:
            st.markdown(
                """
            <style>
            .stButton > button {
            color: dimgray;
            background: lightblack;
            width: 220px;
            height: 50px;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )
            st.button('Reset Chat', on_click=reset_conversation)
    
    # Access the OpenAI API key
    # openai_api_key = st.secrets["openai_api_key"]

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    print('messages: ', type(st.session_state.messages))


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        # print("message: ", message)
        if message["role"] != "system":
            if message["role"] == "user":
                with st.chat_message(message["role"]):
                    st.markdown(process_content(message["content"]))
            else:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    system_message = f"""
     You will be provided with customer service queries for the insurance company. Input will be a string which has the customer query and context to answer the query. To answer the query you have to perfomr the following steps:
    
    Step-1 : Understand the input query and relevant context. Take the context from context xml tags in the user content and answer the question present in-between the question xml tags.
    Step-2 : Divide the overall query into smaller tasks and answer the each step to find the final answer.
    Step-3 : You must find the answer youself and then give the response. Do not respond without finding the answers by yourself.
    Step-4 : For answering the query keep the response conversational.
    Step-5 : Always append the text "Good bye!" in the response and restrict your response to 50 words only.
    """

    # Add system message to chat history
    st.session_state.messages.append({"role": "system", "content": system_message})
    
    # Accept user input
    if prompt := st.chat_input("What is up?"):

        relevant_context = get_context(index_file_path, prompt, masterdata_df_file_path, 2)  
        prompt_context = "<question>{"+prompt+"}<question/>, <context>{"+relevant_context+"}<context/>"        
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt_context})
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