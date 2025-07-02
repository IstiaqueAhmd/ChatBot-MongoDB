from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
from dotenv import load_dotenv
from uuid import uuid4
import google.generativeai as genai
import os
from datetime import datetime
from google.generativeai import embedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load .env
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")
embed_model = genai.EmbeddingModel(model_name="models/embedding-001")

def find_relevant_chats(user_query, session_id, top_k=5):
    query_vec = embedding.embed_content(user_query, model="models/embedding-001")["embedding"]
    query_vec = np.array(query_vec).reshape(1, -1)

    all_chats = list(db[session_id].find({"embedding": {"$exists": True}}))
    
    if not all_chats:
        return []

    chat_vectors = np.array([chat["embedding"] for chat in all_chats])
    similarities = cosine_similarity(query_vec, chat_vectors)[0]

    sorted_indices = similarities.argsort()[::-1][:top_k]
    return [all_chats[i] for i in sorted_indices]


# Mongo setup
client = MongoClient(os.getenv("MONGO_URI"))
db = client["chatbot"]

# FastAPI setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Show session list
@app.get("/", response_class=HTMLResponse)
async def list_sessions(request: Request):
    collections = db.list_collection_names()
    sessions = [c for c in collections if c.startswith("session_")]
    return templates.TemplateResponse("index.html", {"request": request, "sessions": sessions})


# Start a new session
@app.post("/start-session")
async def start_session(topic: str = Form(...)):
    session_id = f"session_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:6]}"
    db[session_id].insert_one({
        "sender": "system",
        "message": f"New session started on: {topic}",
        "timestamp": datetime.utcnow()
    })
    return RedirectResponse(url=f"/chat/{session_id}", status_code=303)


# Show chat in a session
@app.get("/chat/{session_id}", response_class=HTMLResponse)
async def chat_ui(request: Request, session_id: str):
    chats = list(db[session_id].find({}, {"_id": 0}).sort("timestamp", 1))
    return templates.TemplateResponse("chat.html", {"request": request, "chats": chats, "session_id": session_id})


# Send a message in a session
@app.post("/send/{session_id}")
async def send_message(session_id: str, message: str = Form(...)):
    vec = embedding.embed_content(message, model="models/embedding-001")["embedding"]
    db[session_id].insert_one({
        "sender": "user",
        "message": message,
        "embedding": vec,
        "timestamp": datetime.utcnow()
    })

    prompt = ""
    context = find_relevant_chats(message, session_id, top_k=5)
    print(context)

    response = model.generate_content(message)
    bot_reply = response.text.strip()
    
    vec = embedding.embed_content(bot_reply, model="models/embedding-001")["embedding"]
    db[session_id].insert_one({
        "sender": "bot",
        "message": bot_reply,
        "embedding": vec,
        "timestamp": datetime.utcnow()
    })

    return RedirectResponse(url=f"/chat/{session_id}", status_code=303)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.2", port=8000)