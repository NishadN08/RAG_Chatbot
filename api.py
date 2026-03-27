# -------------------------
# Imports
# -------------------------

import asyncio
import traceback
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import date
import json
import uvicorn
from rag import app, combine_chain, question_generator, fallback_chain, format_chunks

# Configure CORS (Cross-Origin Resource Sharing)

api = FastAPI(title="LangGraph RAG Chatbot Streaming API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://144.174.11.43",
        "http://144.174.11.43/joomla_hello",
        "http://localhost",
        "*"
        ],  # or ["https://your-joomla-domain.com"]
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS","*"],  # allow POST, OPTIONS, GET, etc.
    allow_headers=["*"],
)

# -------------------------------
# Request Schema
# -------------------------------

class Query(BaseModel):
    question: str

chat_histories: dict[str, List[dict]] = {}

# -------------------------------
# Streaming Endpoint
# -------------------------------

@api.post("/chat/stream")
async def chat_stream(query: Query, x_session_id: str = Header(...)):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    session_id = x_session_id
    if session_id not in chat_histories:
        chat_histories[session_id] = []

    history = chat_histories[session_id]
    last_three = history[-3:]  # gets last 3 questions


    async def generate_answer():        
        try:
            # Run the graph
            result = await app.ainvoke({
                "question": query.question,
                "chat_history": last_three,
                })  # or app.invoke(...) if sync

            # Send standalone question
            yield f"data: {json.dumps({'type': 'standalone_question', 'content': result['question']})}\n\n"

            # Stream LLM answer token by token (simulate here if your LLM doesn't support streaming)
            answer_text = result["answer"]
            for token in answer_text.split():
                yield f"data: {json.dumps({'type': 'token', 'content': token + ' '})}\n\n"

                await asyncio.sleep(0.02)  # 80ms per token (adjust as needed)

            # Send final answer
            yield f"data: {json.dumps({'type': 'answer_complete', 'content': answer_text})}\n\n"

            # Send question reformulation prompt
            condense_prompt_text = question_generator.prompt.format(
                chat_history=last_three,
                question=query.question,
                today=date.today().isoformat(),
                keywords = result.get("keywords", "")
                )
            yield f"data: {json.dumps({'type': 'memory', 'content': {'condense_prompt_text': condense_prompt_text, 'chat_history' : last_three, 'keywords': result.get('keywords', '')}})}\n\n" # //////////


            # Send rag prompt + docs
            fake_context = "The context here are the retrieved chunks."     # You can optionally include retrieved context if available
            custom_prompt_text = combine_chain.llm_chain.prompt.format(
            context=fake_context,
            question=result['question'],
            today=date.today().isoformat()
            )
            retrieved_chunks = format_chunks(result.get("docs", []))
            yield f"data: {json.dumps({'type': 'sources', 'content': {'custom_prompt_text' : custom_prompt_text, 'retrieved_chunks' : retrieved_chunks}})}\n\n"

            #Fallback prompt text
            fallback_prompt_text = fallback_chain.prompt.format(question=result['question'])
            yield f"data: {json.dumps({'type': 'fallback', 'content': fallback_prompt_text})}\n\n"


            # Send updated chat history (after answer)
            yield f"data: {json.dumps({'type': 'chat_history', 'content': history})}\n\n"

            # Done
            yield f"data: {json.dumps({'type': 'done', 'content': answer_text})}\n\n"

            # Update session chat history
            q_num = len(history) + 1
            history.append({f"Question {q_num}": result["question"], "Answer": answer_text})
            chat_histories[session_id] = history
            
        except Exception as e:
            # Send error as event
            error_msg = str(e)
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"

    return StreamingResponse(
        generate_answer(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # nginx-friendly
        }
    )

@api.get("/")
def root():
    return {"message": "FSU-SC RAG Chatbot API is running!"}


if __name__ == "__main__":
    uvicorn.run("api:api", host="0.0.0.0", port=8000)
