# api_server.py
from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from rag_memory import qa_chain, store_chat_in_memory, format_chunks


app = FastAPI(title="RAG Chat API")


# Add this BEFORE defining endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost","*"],  # or ["https://your-joomla-domain.com"]
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],  # allow POST, OPTIONS, GET, etc.
    allow_headers=["*"],
)


# in-memory chat history (per session this can be improved with DB/Redis)
chat_histories = {}

class Query(BaseModel):
    question: str



@app.post("/chat")
async def chat(query: Query, x_session_id: str = Header(...)):
    history = chat_histories.get(x_session_id, [])
    result = qa_chain({"question": query.question, "chat_history": list(history)})
    answer = result["answer"]    


    if not query.question:
        return {"answer":"Question cannot be empty."}

    # save in memory store
    store_chat_in_memory(query.question, answer)


    standalone_q = qa_chain.question_generator.run({
        "chat_history": list(history),
        "question": query.question,
    })

        # Append to session-specific history
    history.append((standalone_q, answer))
    chat_histories[x_session_id] = history

    condense_prompt_text = qa_chain.question_generator.prompt.format(
        chat_history=history[:-1],
        question=query.question
    )

    # 2️⃣ Prepare custom_prompt (the one used for answer)
    # You can optionally include retrieved context if available
    fake_context = "The context here are the retrieved chunks."  
    custom_prompt_text = qa_chain.combine_docs_chain.llm_chain.prompt.format(
        context=fake_context,
        question=standalone_q
    )

    src_docs = result.get("source_documents", []) or []
    retrieved_chunks = format_chunks(src_docs, max_items=10, max_chars=5000)

    return {
        "question": query.question,
        "standalone_question": standalone_q,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "chat_history": history[:-1],
        "condense_prompt_text": condense_prompt_text,
        "custom_prompt_text": custom_prompt_text
    }

@app.get("/")
def root():
    return {"message": "RAG Chatbot API is running!"}


if __name__ == "__main__":
    uvicorn.run("api_memory:app", host="0.0.0.0", port=8000)


