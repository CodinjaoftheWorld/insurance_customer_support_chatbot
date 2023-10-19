import pandas as pd
import openai
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pickle
import faiss


# Set the OPENAI API Key as the environment variable
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')

# Config Paths
INPUT_FILE_NAME = "insurance.csv"
EMBEDDING_FILE_NAME = "embedding_array.pickle"
INPUT_FILE_DIR = "input"
OUTPUT_FILE_DIR = "output"
OUTPUT_MASTERDATA_FILE_NAME = "insurance_masterdata.pickle"
OUTPUT_INDEX_FILE_NAME = "index.pickle"


def create_text(row):
    # Process the text before sending it to LLM
    return "question- " +str(row['question'])+ "\n" + "answer- "+str(row['answer'])

def generate_embedding_array(embeddings, EMBEDDING_FILE_NAME, OUTPUT_FILE_DIR):
    # Generate embeddings in numpy array and saving it in a pickle format
    all_embeddings = []
    for i in embeddings:
        all_embeddings.append(i['embedding'])
    embedding_array = np.array(all_embeddings)

    with open(f"{OUTPUT_FILE_DIR}/{EMBEDDING_FILE_NAME}", 'wb') as pickle_file:
        pickle.dump(embedding_array, pickle_file)
    return


def create_faiss_index(embeddings_path, OUTPUT_INDEX_FILE_NAME, OUTPUT_FILE_DIR):
    # Getting the root directory path
    ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
    ) 

    # Defining the dimensions for openai embedding model which is 1536
    d = 1536
    # Index creation
    index = faiss.IndexFlatIP(d)

    # Opening the embedding saved in numpy array
    with open(embeddings_path, 'rb') as f:
        embeddings_content = f.read()
    embeddings = pickle.loads(embeddings_content)
    embeddings = embeddings.astype(np.float32)

    # Adding embedding to the index
    index.add(np.array(embeddings, dtype='float32'))

    # Saving the index pickle file to the output directory
    os.chdir(os.path.join(ROOT_DIR, OUTPUT_FILE_DIR))
    with open(OUTPUT_INDEX_FILE_NAME, 'wb') as file:
        pickle.dump(index,file)
    return

def convert_masterdata_to_pickle(df, OUTPUT_MASTERDATA_FILE_NAME, OUTPUT_FILE_DIR):
    # Create the root diretory path
    ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
    )
    # Reading the master data and converting it into pickle file
    df.to_pickle(os.path.join(ROOT_DIR, OUTPUT_FILE_DIR, OUTPUT_MASTERDATA_FILE_NAME))
    return

def main():
    # Create the root diretory path
    ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
    )
    # Create the input file path
    input_file_path = os.path.join(ROOT_DIR, "input", INPUT_FILE_NAME)

    # Read the masterdata csv file
    df = pd.read_csv(input_file_path)
    # Create the input for mebedding creation
    df['text'] = df.apply(lambda row: create_text(row), axis=1)

    # Genertae embeddings for all the rows
    embeddings = openai.Embedding.create(input = df['text'].tolist(), model="text-embedding-ada-002")['data'] 

    # Create numpy array for embeddings
    generate_embedding_array(embeddings, EMBEDDING_FILE_NAME, OUTPUT_FILE_DIR)

    # Create FAISS index
    create_faiss_index(os.path.join(ROOT_DIR, OUTPUT_FILE_DIR, EMBEDDING_FILE_NAME), OUTPUT_INDEX_FILE_NAME, OUTPUT_FILE_DIR)       

    # Create masterdata for input file
    convert_masterdata_to_pickle(df, OUTPUT_MASTERDATA_FILE_NAME, OUTPUT_FILE_DIR)




if __name__ == "__main__":
    main()