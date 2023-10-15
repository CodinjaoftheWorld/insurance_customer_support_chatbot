import pandas as pd
import openai
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pickle
import streamlit as st
from streamlit import session_state
from streamlit import secrets


# Set OpenAI API Key as the environemnt variable
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')

# Config Paths
OUTPUT_FILE_DIR = "output"
OUTPUT_MASTERDATA_FILE_NAME = "insurance_masterdata.pickle"
OUTPUT_INDEX_FILE_NAME = "index.pickle"

# Getting the root directory path
ROOT_DIR = os.path.dirname(
os.path.abspath(__file__)
) 

# Masterdata file path
masterdata_df_file_path = os.path.join(ROOT_DIR, OUTPUT_FILE_DIR, OUTPUT_MASTERDATA_FILE_NAME)
index_file_path = os.path.join(ROOT_DIR, OUTPUT_FILE_DIR, OUTPUT_INDEX_FILE_NAME)


# Function to load the pickle file
def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Function to load the masterdata pickle file
def load_masterdata_pickle(path):
    df = pd.read_pickle(path)
    return df

# Function to extract the top K question and answers similar to the user prompt
def get_context(index_file_path, prompt, masterdata_df_file_path, k):
    index = load_pickle_file(index_file_path)
    query_embedding = openai.Embedding.create(input = [prompt], model="text-embedding-ada-002")['data'][0]['embedding']
    k = k
    distance, indices = index.search(np.array([query_embedding]), k)
    master_df = load_masterdata_pickle(masterdata_df_file_path)
    top_k_prompt = "".join(master_df.iloc[indices[0].flatten()]['text'].tolist())
    return top_k_prompt

# Function to process respone and extract the original user prompt
def process_content(text):
    return text.split("}<question/>")[0].split("{")[-1]

# This function in triggered when user click on "Start new chat" button on the left side of the screen
def reset_conversation():
    if len(st.session_state.messages) == 0:
        st.session_state.conversations = []    
        st.session_state.clicked = []
        st.session_state.expander_history = []
    else:
        st.session_state.conversations = [] 
        for msg in st.session_state.messages:
            if msg['role'] != 'system':
                st.session_state.conversations.append(msg)
        st.session_state.expander_history.append(st.session_state.conversations)
        st.session_state.messages = []
        st.session_state.clicked = []

# This function is used to show the old conversations on the main chat screen.
def show_conversation(conv, idx):
    # Display chat messages from history on app rerun
    for message in conv:
        if message["role"] == "user":
            with st.chat_message(message["role"]):
                st.markdown(process_content(message["content"]))
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# This function is used to load the conversation in the session state of clicked variable when "Start new chat" button is clicked 
def load_conversation(conv, idx):
    st.session_state.clicked = []
    if len(st.session_state.clicked) == 0:
        st.session_state.clicked.append({'conversation': conv, 'idx': idx })
    else:
        pass 


# Main streamlit function
def main():
    # Set the Openai key in the environment variable
    load_dotenv()

    # Set the head of streamlit UI
    st.title("Your Insurance Assistant! 🛡️🚑🏥")
    
    # set the cutom markdown for sidebar
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: DarkRed;
        }
    </style>
    """, unsafe_allow_html=True) 
    
    # Create the sidebar
    with st.sidebar:
        # Initialize the varibale to save all the conversation 
        if 'conversations' not in st.session_state:
            st.session_state.conversations = []
        # Initialize the varibale to store the conversation on the click of "Start new chat" button 
        if 'clicked' not in st.session_state:
            st.session_state.clicked = []
        # Initialize the flag to load the history of the conversation in the sidebar
        if 'load_history' not in st.session_state:
            st.session_state.load_history = False
        # Initialize the main chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        # Initialize the varibale to store the hostory of convesations for expandable buttons in the side bar.
        if "expander_history" not in st.session_state:
            st.session_state.expander_history = []    
        # Set a default model
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-3.5-turbo"

        # Defining columns with in the sidebar to allign "Start new chat" button in the center. 
        c1, c2, c3 = st.columns([0.1, 0.8, 0.1])
        with c2:
            #Custome css for "Start new chat" button
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
            st.button('Start new chat', on_click=reset_conversation)
            
        #Custome css for expandable buttons
        st.markdown(
            '''
            <style>
            .streamlit-expanderHeader {
                background-color: white;
                color: black; # Adjust this for expander header color
            }
            .streamlit-expanderContent {
                background-color: white;
                color: black; # Expander content color
            }
            </style>
            ''',
            unsafe_allow_html=True
        )
        
        # Loop to create expandable buttons for old chats same as chatGPT
        for index, item in enumerate(st.session_state.expander_history):
            if item != []:
                head = True
                for idx, conv in enumerate(item):
                    if head:
                        if conv['role'] == 'user': 
                            heading = conv['content'].split("}<question/>")[0].replace("<question>{", "")[:40]
                            head = False
                expander = st.expander(f"{heading}")
                clicked = expander.button("Show details", key = index, use_container_width=True)
                if clicked:
                    load_conversation(item, index)
                    st.session_state.load_history = True 

    

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

    # Custom System prompt. Telling OpenAI api to consider the customer query and relevant context for answering the query and some steps to process the information before proving the response  
    system_message = f"""
     You will be provided with customer service queries for the insurance company. Input will be a string which has the customer query and context to answer the query. To answer the query you have to perfomr the following steps:
    
    Step-1 : Understand the input query and relevant context. Take the context from context xml tags in the user content and answer the question present in-between the question xml tags.
    Step-2 : Divide the overall query into smaller tasks and answer the each step to find the final answer.
    Step-3 : You must find the answer youself and then give the response. Do not respond without finding the answers by yourself.
    Step-4 : For answering the query keep the response conversational.
    Step-5 : Always restrict your response to 50 words.
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

    # check if the "Show details" button is clicked with in the expandable button
    if st.session_state.load_history:
        conv = st.session_state.clicked[0]['conversation']
        idx = st.session_state.clicked[0]['idx']
        show_conversation(conv, idx)
        st.session_state.load_history = False


if __name__=="__main__":
    main()