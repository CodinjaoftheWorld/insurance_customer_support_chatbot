import pandas as pd
import openai
import os
from dotenv import load_dotenv, find_dotenv
# from openai.embeddings_utils import get_embedding
import numpy as np
import pickle
import faiss



_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')
# print(openai.api_key)

# PATHS
INPUT_FILE_DIR = "input"
INPUT_FILE_NAME = "customer_support.csv"
EMBEDDING_FILE_NAME = "embedding_array.pickle"
OUTPUT_FILE_DIR = "masterdata"
OUTPUT_MASTERDATA_FILE_NAME = "masterdata.pickle"
OUTPUT_INDEX_FILE_NAME = "index.pickle"


def create_text(row):
    return "question- " +str(row['question'])+ "\n" + "answer- "+str(row['answer'])



def main():
    





if __name__ == "__main__":
    try: 
        main()
    except Exception as e:
        logging.exception(e)
        raise e