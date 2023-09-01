import pandas as pd
import torch
import re
from bs4 import BeautifulSoup
import spacy
from contractions import fix
import multiprocessing
from tqdm import tqdm
import os

def parse_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    paragraph = soup.find('p')
    return paragraph.text if paragraph else ""

def remove_links(text):
    # Regular expression to match URLs
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    # Remove URLs from the text
    cleaned_text = re.sub(url_pattern, '', text)
    return cleaned_text

def clean(text):
    # text=remove_links(text)
    # print(f"\noriginal text:\n{text}")
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # text =  parse_html(text)
    # print(f"html parse text:\n{text}")

    text = text.lower()

    # Process the text with spaCy
    doc = text.split()

    # Expand contractions using the contractions library
    text = " ".join([fix(token) for token in doc])

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # nlp = spacy.load("en_core_web_sm")

    # # Process the text with spaCy
    # doc = nlp(text)

    # # Remove stop words and lemmatize
    # cleaned_tokens = [token.lemma_ for token in doc if not token.is_stop]

    # # Join the cleaned tokens back into a sentence
    # cleaned_text = " ".join(cleaned_tokens)
    # cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # cleaned_text = cleaned_text.strip()
    # # Split the text into words
    # words = cleaned_text.split()

    # # Filter out words that are only one character long
    # filtered_words = [word for word in words if len(word) > 1]

    # # Join the filtered words back into a text
    # text = ' '.join(filtered_words)
    # # print(f"\nfinal text:\n{cleaned_text}")

    return text

def single_row(row):
    row["text"]=row["comment_text"].apply(clean)
    indexs=row.index
    label_dict={}
    for i in range(len(row)):
        label_row=((row.iloc[i,2:8]).values)
        label_row=[float(j) for j in label_row]
        # label_row=torch.tensor(label_row,dtype=torch.float32)
        label_dict[indexs[i]]=label_row
    row["label"]=pd.Series(label_dict,dtype='O')
    return row

if __name__ == "__main__":
    train_df = pd.read_csv(r"C:\Users\athul\myfiles\projects\toxic comment classifier\data\train.csv")
    # train_df = train_df.head(100) 

    output_path_folder="./processed_data"
    
    if not os.path.exists(output_path_folder):
        os.makedirs(output_path_folder)

    num_processes = multiprocessing.cpu_count()

    if(num_processes>8):
        num_processes=8

    # chunk_size = len(train_df)//num_processes 
    chunk_size = 100
    chunks = [train_df[i:i + chunk_size] for i in range(0, len(train_df), chunk_size)]

    # print(f"the length of chunk is {len(chunks)}")
    # print(f"the type of chunk is {type(chunks)}")
    # print(f"the type of chunk0 is {type(chunks[0])}")
    # print(f"the len of chunk0 is {len(chunks[0])}")
    # print(f"the len of chunk4 is {len(chunks[-1])}")
    # print(f"the head  of chunk0 is \n{chunks[0].head()}")
    # print(f"the head of chunk4 is \n{chunks[-1].head()}")
    # print(chunks)

    with multiprocessing.Pool(num_processes) as pool:
        processed_chunks = list(tqdm(pool.imap(single_row, chunks), total=len(chunks)))

    # print(f"the length of chunk is {len(processed_chunks)}")
    # print(f"the type of chunk is {type(processed_chunks)}")
    # print(f"the type of chunk0 is {type(processed_chunks[0])}")
    # print(f"the len of chunk0 is {len(processed_chunks[0])}")
    # # print(f"the len of chunk4 is {len(processed_chunks[4])}")
    # print(f"the head  of chunk0 is \n {processed_chunks[0].head()}")
    # print(f"the head  of chunk0 is \n {processed_chunks[456].head()}")
    # print(f"the head of chunk4 is \n {processed_chunks[-1].head()}")
    # # print(chunks)


    train_df = pd.concat(processed_chunks, ignore_index=True)

    train_df = train_df.drop(train_df.columns[1:8], axis=1)
    print(train_df.head())
    train_df.to_csv(os.path.join(output_path_folder, "p1_data.csv"), index=False)
