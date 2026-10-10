import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    return r.json()

df = joblib.load("embeddings.joblib")

incoming_query = input("Where is flowcharts taught in this?")
question_embedding = create_embedding([incoming_query])[0]

similarities = cosine_similarity(np.vstack(df["embedding"]), [question_embedding]).flatten()
top_results = 5
max_indx = similarities.argsort()[::-1][:top_results]
new_df = df.loc[max_indx]

chunks_json = new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")

prompt = f'''
You are an expert teaching assistant for Sharadha Kapra's Data Structures and Algorithms course.

Below is the list of video subtitle chunks. Each chunk contains:
- video title
- video number
- start time in seconds
- end time in seconds
- transcript text spoken in that time range

Here are the chunks:
{chunks_json}

----------------------------------------------------

The user asked the following question:
"{incoming_query}"

Your task:
1. Understand the user's question.
2. Search the video chunks to find the most relevant segments.
3. Answer naturally like a human tutor (do NOT mention JSON or chunk format).
4. Tell the user:
   - Which video contains the answer.
   - The exact timestamp range.
   - What content is taught in that period.
5. If the question is unrelated to the course, say:
   "I can only answer questions related to the course content."

Respond in this exact format:

Video No: <video number>
Time Range: <start time> : <end time>
Explanation: <your explanation here>
'''

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)
