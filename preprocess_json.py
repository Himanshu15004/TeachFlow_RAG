import requests
import os
import json
import pandas as pd
import joblib
def create_embeddings(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    return r.json()["embeddings"]   # returns list of embedding vectors


json_folder = r"D:\RAG_Teaching_Assitant\Jsons"
json_files = os.listdir(json_folder)

my_dicts = []
chunk_id = 0

for json_file in json_files:

    with open(os.path.join(json_folder, json_file), "r", encoding="utf-8") as f:
        content = json.load(f)

    print(f"\n Creating Embeddings for: {json_file}")

    
    chunk_texts = [chunk["text"] for chunk in content["chunks"]]

    embeddings = create_embeddings(chunk_texts)

    for i, chunk in enumerate(content["chunks"]):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)

df = pd.DataFrame.from_records(my_dicts)
joblib.dump(df, 'embeddings.joblib')
