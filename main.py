import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import faiss
import numpy as np
from db import SessionLocal, Conversation, init_db
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

app = FastAPI(title="MedChatBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # allow your front-end origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Llama model and tokenizer
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained("./llama_3.2_3B")
model = AutoModelForCausalLM.from_pretrained("./llama_3.2_3B", torch_dtype=torch.bfloat16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load FAISS index and question mapping
index = faiss.read_index("faiss_index.index")
qa_pairs = np.load("qa_pairs.npy", allow_pickle=True).tolist()
embedder = SentenceTransformer("all-mpnet-base-v2")

class Query(BaseModel):
    query: str

def retrieve_context(query_text, top_k=3):
    q_embedding = embedder.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(q_embedding)
    distances, indices = index.search(q_embedding, top_k)
    retrieved = [qa_pairs[idx] for idx in indices[0]]
    return retrieved

def construct_prompt(user_query, retrieved_context):
    context_text = "\n\n".join(retrieved_context)
    prompt = (
        "Below is some background information derived from similar medical Q&A pairs. "
        "Use this background solely to inform your answer, but do not include it verbatim in your final response.\n\n"
        f"Background:\n{context_text}\n\n"
        f"Patient's Question: {user_query}\n\n"
        "Answer :"
    )
    return prompt

init_db()

import re
def remove_space_before_punctuation(s: str) -> str:
    return re.sub(r'\s+([?!.:,;])', r'\1', s)

@app.post("/chat")
async def chat(query: Query):
    retrieved = retrieve_context(query.query)
    prompt = construct_prompt(query.query, retrieved)
    
    print("prompt is: ", prompt)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,           
        do_sample=True,               
        temperature=0.4,              
        top_k=50,                     
        top_p=0.9,                    
        repetition_penalty=1.2,       
        no_repeat_ngram_size=3,       
        pad_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    prompt_norm = remove_space_before_punctuation(prompt)
    answer_norm = remove_space_before_punctuation(answer)
    
    answer_norm = answer_norm.removeprefix(prompt_norm)
    print("answer_norm is: ", answer_norm)
    
    # Log conversation to the database
    db = SessionLocal()
    conv = Conversation(user_query=query.query, prompt=prompt, answer=answer)
    db.add(conv)
    db.commit()
    db.close()
    
    return {"answer": answer_norm}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
