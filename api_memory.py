# api_server.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from rag_memory import qa_chain, store_chat_in_memory


app = FastAPI(title="RAG Chat API")


# Add this BEFORE defining endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost"],  # or ["https://your-joomla-domain.com"]
    allow_methods=["POST", "GET", "OPTIONS"],  # allow POST, OPTIONS, GET, etc.
    allow_headers=["*"],
)

# in-memory chat history (per session this can be improved with DB/Redis)
chat_history = []

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    global chat_history
    result = qa_chain({"question": query.question, "chat_history": list(chat_history)})
    answer = result["answer"]

    # save in memory store
    store_chat_in_memory(query.question, answer)

    # append to conversation
    chat_history.append((query.question, answer))

    return {
        "question": query.question,
        "answer": answer,
        }

@app.get("/")
def root():
    return {"message": "RAG Chatbot API is running!"}

if __name__ == "__main__":
    uvicorn.run("api_memory:app", host="0.0.0.0", port=8000)
