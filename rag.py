
# =========================================================
# Imports
# =========================================================
import os, json, re, shutil
from datetime import datetime, date
from typing import Dict, Any, List, TypedDict

# LangChain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Retrievers & Encoders(for Reranking)
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Chains & Prompts
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain, StuffDocumentsChain

# LangGraph
from langgraph.graph import StateGraph, END

from langchain_community.cache import SQLiteCache
from langchain_classic.globals import set_llm_cache

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)


# from dotenv import load_dotenv
# from langsmith import traceable

# load_dotenv()  # loads .env into environment variables
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# print("Tracing:", os.getenv("LANGCHAIN_TRACING_V2"))
# print("Project:", os.getenv("LANGCHAIN_PROJECT"))


# =========================================================
# Config
# =========================================================

DATA_DIR = "./"
NEW_FILE = "new.jsonl"
OLD_FILE = "old.jsonl"
CHROMA_NEW_DIR = "./chroma_new"
CHROMA_OLD_DIR = "./chroma_old"


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = "gpt-oss:20b"

CHUNK_SIZE = 1700
CHUNK_OVERLAP = 0

ALLOWED_META_TYPES = (str, int, float, bool, type(None))

# =========================================================
# Utilities (cleaning, dates, metadata)
# =========================================================

def clean_text_block(text: str) -> str:
    """
    Clean raw scraped text by removing noise and duplicates and normalizing spacing.
    """
    # Remove webmaster/edit info & placeholders
    text = re.sub(r"Edit Information.*?(removed\.)", "", text, flags=re.DOTALL)
    text = re.sub(r"Please Email.*?removed\.", "", text, flags=re.DOTALL)
    text = re.sub(r"Click here to email the webmaster.*?go away\.", "", text)

    # Fix inline duplicate labels
    text = re.sub(r"\b(Research Interests|Education|Publications|Address|Contact Info):\s*\1\b", r"\1:", text)

    # Collapse repeated words/emails/URLs inline
    text = re.sub(r"\b(\S+)( \1)+\b", r"\1", text)

    # Remove all newlines and collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()


    # De-duplicate lines while keeping order
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    return "\n".join(unique_lines)

def extract_date_from_url(url: str) -> str:
    """
    Extracting semester and date from the URLs.
    Converts date ranges to semester + year ("Semester YYYY").
    If no date is found, returns the current semester and year.
    """
    url_lower = url.lower()
    semester = None
    year = None

    # Full date pattern like 2025-09-17 YYYY-MM-DD
    match_full = re.search(r"(20\d{2})[-_/](0[1-9]|1[0-2])[-_/](0[1-9]|[12]\d|3[01])", url_lower)
    if match_full:
        year = int(match_full.group(1))
        month = int(match_full.group(2))
        semester = (
            "Spring" if 1 <= month <= 4 else
            "Summer" if 5 <= month <= 7 else
            "Fall"
        )
        return f"{semester} {year}"
    
    # Month-day-year pattern like M-D-YYYY or MM-DD-YYYY
    match_mdy = re.search(r"(0?[1-9]|1[0-2])[-_/](0?[1-9]|[12]\d|3[01])[-_/](20\d{2})", url_lower)
    if match_mdy:
        month = int(match_mdy.group(1))
        year = int(match_mdy.group(3))
        semester = (
            "Spring" if 1 <= month <= 4 else
            "Summer" if 5 <= month <= 7 else
            "Fall"
        )
        return f"{semester} {year}"

    # Year-month pattern like YYYY-MM
    match_year_month = re.search(r"(20\d{2})[-_/](0[1-9]|1[0-2])", url_lower)
    if match_year_month:
        year = int(match_year_month.group(1))
        month = int(match_year_month.group(2))
        semester = (
            "Spring" if 1 <= month <= 4 else
            "Summer" if 5 <= month <= 7 else
            "Fall"
        )
        return f"{semester} {year}"
    
    # Month-year patterns like MM-YY or MM-YYYY
    match_month_year = re.search(r"(0[1-9]|1[0-2])[-_/](\d{2}|\d{4})", url_lower)
    if match_month_year:
        month = int(match_month_year.group(1))
        year_raw = match_month_year.group(2)

        # Convert YY → YYYY
        if len(year_raw) == 2:
            yy = int(year_raw)
            year = 2000 + yy if yy <= 30 else 1900 + yy
        else:
            year = int(year_raw)

        semester = (
            "Spring" if 1 <= month <= 4 else
            "Summer" if 5 <= month <= 7 else
            "Fall"
        )
        return f"{semester} {year}"

    # Semester + year pattern like fall-2024 or spring_2023
    match_semester = re.search(r"(spring|summer|fall)[-_ ]?(20\d{2}|19\d{2})", url_lower)
    if match_semester:
        semester = match_semester.group(1).capitalize()
        year = int(match_semester.group(2))
        return f"{semester} {year}"

    # Year + semester pattern like 2023-spring or 2024-fall
    match_year_semester = re.search(r"(20\d{2}|19\d{2})[-_ ]?(spring|summer|fall)", url_lower)
    if match_year_semester:
        year = int(match_year_semester.group(1))
        semester = match_year_semester.group(2).capitalize()
        return f"{semester} {year}"

    # Short year codes with semester like fa19, sp16, su21, fall21, spring23, summer16
    match_short = re.search(r"(?<![a-z])(fa|fall|sp|spring|su|summer)(\d{2,4})(?!\d)", url_lower)
    if match_short:
        code_map = {
            "fa": "Fall", "fall": "Fall",
            "sp": "Spring", "spring": "Spring",
            "su": "Summer", "summer": "Summer"
        }
        semester = code_map.get(match_short.group(1))
        year_str = match_short.group(2)
        year = int(year_str) if len(year_str) == 4 else 2000 + int(year_str)   
        return f"{semester} {year}"
    

    # Only Year YYYY (fallback)
    match_year = re.search(r"(20\d{2}|19\d{2})(?!\d)", url_lower)
    if match_year:
        year = int(match_year.group(1))
        return str(year)

    # Default(if the URL does not contain any dates): current semester + current year
    now = datetime.now()
    month = now.month
    semester = (
        "Spring" if 1 <= month <= 4 else
        "Summer" if 5 <= month <= 7 else
        "Fall"
    )

    return f"{semester} {now.year}"

def force_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (md or {}).items():
        if isinstance(v, ALLOWED_META_TYPES):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = ", ".join(str(x) for x in v)
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out

def sanitize_metadata(d: Document) -> Document:
    md = dict(d.metadata or {})
    return Document(page_content=d.page_content, metadata=force_metadata(md))

# =========================================================
# Loader
# =========================================================

def load_jsonl(path: str) -> List[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            url = rec.get("url", "") or ""

            # Skip URLs with pagination or view parameters
            if (
                "?start=" in url
                or "?view=" in url
                or "?option=" in url
                or "tags" in url
            ):
                continue

            text = clean_text_block(rec.get("text", ""))
            if not text:
                continue

            md = {
                "source": rec.get("url", ""),
                "title": rec.get("title", ""),
                "date": extract_date_from_url(rec.get("url", "")),
            }

            docs.append(Document(page_content=text, metadata=md))

    return docs



def format_chunks(docs: list):
    """
    Convert retrieved chunks into JSON-friendly format and returns them for debugging.
    """
    retrieved_chunks = []
    if not docs:
        # When no documents are returned by the retriever
        return [{"title": "[none]", "source": "", "text": ""}] 

    # Iterate only through top N chunks
    for i, d in enumerate(docs, 1):
        
        meta = d.metadata or {} # Extract metadata safely
        title = meta.get("title", "") or "[No Title]"
        url = meta.get("source", "") or ""

        # Clean snippet: flatten newlines and limit character size
        text = (d.page_content or "").replace("\n", " ").strip()
        # if len(text) > max_chars:
        #     text = text[:max_chars] + " ..."

        retrieved_chunks.append({
            "id": i,
            "title": title,
            "url": url,
            "text": text
        })
    return retrieved_chunks

# =========================================================
# Build Vector Store Retrievers & Cross-Encoder Reranker
# =========================================================


def build_retriever(jsonl_file: str, persist_dir: str):
    docs = load_jsonl(os.path.join(DATA_DIR, jsonl_file))
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    for c in chunks:
        c.page_content = f"[Source: {c.metadata.get('source')}] [Title: {c.metadata.get('title')}] {c.page_content}"

    chunks = [sanitize_metadata(c) for c in chunks]
    chunks = filter_complex_metadata(chunks)
    
    print(f"[DEBUG] {jsonl_file}: {len(docs)} documents loaded, {len(chunks)} chunks created.")
    
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)

    vs = Chroma.from_documents(documents=chunks, embedding=embedding_model,
                               persist_directory=persist_dir, collection_name="corpus")
    
    base = vs.as_retriever(search_kwargs={"k": 700})
    
    cross_encoder = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=15)
    compressed = ContextualCompressionRetriever(base_retriever=base, base_compressor=reranker)
    
    return compressed, vs , reranker

new_retriever, new_vs, new_reranker = build_retriever(NEW_FILE, CHROMA_NEW_DIR)
old_retriever, old_vs, old_reranker = build_retriever(OLD_FILE, CHROMA_OLD_DIR)

# =========================================================
# Keyword Vector Store (for spelling corrections)
# =========================================================

KEYWORDS_FILE = "clean_keywords.txt"

def load_keyword_documents(path: str) -> List[Document]:
    """
    Load keywords from a file and convert each keyword into a Document.
    """
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            keyword = line.strip()
            if not keyword:
                continue
            docs.append(Document(page_content=keyword, metadata={"keyword": keyword}))
    return docs

keyword_docs = load_keyword_documents(KEYWORDS_FILE)
print(f"[DEBUG] Loaded {len(keyword_docs)} keyword documents.")


# Using the same embedding model
keyword_embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

KEYWORD_VS_DIR = "./chroma_keywords"

if os.path.isdir(KEYWORD_VS_DIR):
    shutil.rmtree(KEYWORD_VS_DIR)

keyword_vs = Chroma.from_documents(
    documents=keyword_docs,
    embedding=keyword_embedding_model,
    persist_directory=KEYWORD_VS_DIR,
    collection_name="keywords"
)

keyword_retriever = keyword_vs.as_retriever(search_kwargs={"k": 10})

def _keyword_matches(question: str) -> str:
    """
    Retrieve closest keywords from the keyword vector store.
    """
    results = keyword_retriever.invoke(question)
    keywords = []
    seen = set()
    for doc in results:
        kw = doc.metadata.get("keyword", "")
        if kw and kw not in seen:
            seen.add(kw)
            keywords.append(kw)
    if not keywords:
        return "None"
    return ", ".join(keywords)

# =========================================================
# LLM + Prompts
# =========================================================

set_llm_cache(SQLiteCache(database_path="./langchain_cache.db"))

llm = OllamaLLM(model=OLLAMA_MODEL)


# qa_prompt = PromptTemplate(
#     input_variables=["context", "question", "today"],
#     template="""
# You are a Helpful AI assistant. Answer the question **strictly using the given context**. 

# The context consists of multiple chunks of text, each with a date indicating when the information was relevant.
# Each chunk may mention multiple people or events — focus **only** on the person or topic directly asked about.

# Today's date is: {today}

# Your goal is to provide the most **recent and accurate** information available.

# Instructions:
# 1. Review all provided chunks carefully.
# 2. Prefer chunks with newer dates (more recent events or updates).
# 3. Prioritize information that directly matches the question subject.
# 4. If the information is outdated or uncertain, clearly mention that.
# 5. If asked about a person, include some additional relevant details.
# 6. Never include information about unrelated people or topics.
# 7. If no relevant information is found,  respond **Only** with: "I could not find any relevant information about your question."


# Question:
# {question}

# Context:
# {context}

# Guidelines:
# - Give the final answer based on the most recent chunk(s).
# - Do not merge unrelated or outdated information.
# - If unsure, state the uncertainty instead of guessing.
# - Do NOT infer, guess, or combine information across chunks.
# - Answer only using the context above.
# - Use **Markdown** for all answers, using tables, bullet points, and sections as appropriate.
# - Be concise and factual, and focused only on the query subject.
# """
# )

# If multiple chunks mention the same fact:
# - Use the information from the most recent date only.

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a factual question-answering assistant.

Answer the question using ONLY the provided context.
Prioritize information that directly matches the question subject.

Rules (MANDATORY):
- Do NOT restate the question.
- Do NOT explain your reasoning or selection of information.
- Do NOT include information not directly required to answer the question.
- Do NOT add interpretations, summaries, or conclusions beyond the context.

Opening Statement Rule:
- If the context explicitly contains a clear answer, begin with ONE short factual opening sentence that directly addresses the question.
- The opening sentence MUST be fully supported by the context.
- If no clear opening sentence can be formed from the context, omit the opening sentence entirely.

List Formatting Rule:
- Use bullet points ONLY if the question explicitly asks for a list.
- If using bullet points:
  - Add a short factual heading ONLY if the context explicitly supports a grouping or category.
  - Headings must be neutral, factual, and derived from the context.
  - Do NOT invent or infer categories.

If the answer is not explicitly present in the context:
- Respond EXACTLY with:
"I could not find any relevant information about your question."

Context:
{context}

Question:
{question}

Output format:
- A direct answer only
- Optional single-sentence opening statement (if applicable)
- Use bullet points ONLY if the question explicitly asks for a list
- Otherwise, use a short factual paragraph
"""
)
# - Do NOT mention sources, URLs, documents, or chunk metadata.

qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

combine_chain = StuffDocumentsChain(
    llm_chain=qa_chain,
    document_variable_name="context"
)

# -------------------------------------
# Question Reformulation Prompt
# -------------------------------------
condense_prompt = PromptTemplate(
    input_variables=["chat_history", "question", "today"],
    template="""
You are a precise query rewriting assistant.

Your task is to rewrite the user's latest question into a fully self-contained, standalone question
that preserves the original meaning and intent.

You MUST follow these rules:

1. Use the chat history ONLY if the latest question depends on it.
2. If the latest user question is not directly related to the latest or last question in the chat history, return it unchanged.
3. Do NOT answer the question.
4. Do NOT add new information.
5. Do NOT remove important constraints.
6. Preserve ALL temporal references (time-related words).
7. Do NOT change the date or time references if its correctly specified in the question.

IMPORTANT TIME HANDLING:

- Today's date is: {today}
- If the user uses vague time expressions such as:
"this semester", "this term", "currently", "now", "this year", "recently", "this month"

Only Then you MAY rewrite them into explicit, Semester using the year and month from today's date.
            
- A "semester" is defined as:
    Spring semester: January to April
    Summer semester: June to July
    Fall semester: August to December

- Do NOT guess semester names (Spring/Fall).
- Do NOT invent dates.
- Only make the time reference explicit.

The user question may contain spelling errors, grammatical mistakes, or typos.
You should try to correct them if you feel there is a spelling error, using the lexical matching from an internal directory.
The internal directory contains names of faculty, staff, courses, research areas, and other entities.
Closest matching terms from the internal keyword directory:
{keywords}


OUTPUT RULES:

- Output ONLY the rewritten question.
- Do NOT include quotes.
- Do NOT include explanations.

Chat History:
{chat_history}

Latest User Question:
{question}
Standalone question:"""
)

# LLM chain for reformulating the question
question_generator = LLMChain(llm=llm, prompt=condense_prompt)


fallback_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an assistant for the Florida State University Department of Scientific Computing.

Rules:
- Stay strictly within the domain of Florida State University and its Department of Scientific Computing.
- Do NOT explain your reasoning.
- Do NOT describe your thought process.
- Do NOT mention what you searched or considered.
- Do NOT speculate or guess specific facts.
- Produce ONLY the final answer.

If the answer is uncertain or cannot be verified:
- State this briefly and factually in one or two sentences.
- Also try to mention you can look up more information on www.sc.fsu.edu if needed in the answer.

Question:
{question}

Answer:
"""
)

fallback_chain = LLMChain(llm=llm, prompt=fallback_prompt)

# =========================================================
# LangGraph State
# =========================================================

class RAGState(TypedDict):
    question: str
    docs: List[Document]
    answer: str
    chat_history: List[dict]
    keywords: str

# =========================================================
# Nodes
# =========================================================

# @traceable(name="Reformulate Question")
def reformulate_node(state: RAGState):
    """
    Use the question_generator LLMChain to rewrite follow-up questions
    into a standalone question using chat history.
    """
    chat_history = state.get("chat_history", [])
    question = state["question"]
    today = date.today().isoformat()
    keyword_context = _keyword_matches(question)
    
    # Rewrite the question
    rewritten = question_generator.invoke({
        "chat_history": chat_history,
        "question": question,
        "today": today,
        "keywords": keyword_context
    })["text"].strip()
    
    return {"question": rewritten, "keywords": keyword_context}

# @traceable(name="Retrieve NEW Corpus")
def retrieve_new_node(state: RAGState):
    q = state["question"]
    final_docs = new_retriever.invoke(q)
    return {"docs": final_docs}

# @traceable(name="Answer from NEW Corpus")
def answer_new_node(state: RAGState):
    q = state["question"]
    docs = state["docs"]
    today = date.today().isoformat()
    result = combine_chain.invoke({"input_documents": docs, "question": q, "today": today}) 
    return {"answer": result["output_text"].strip()}


# @traceable(name="Retrieve OLD Corpus")
def retrieve_old_node(state: RAGState):
    q = state["question"]
    final_docs = old_retriever.invoke(q)
    return {"docs": final_docs}

# @traceable(name="Answer from OLD Corpus")
def answer_old_node(state: RAGState):
    q = state["question"]
    docs = state["docs"]
    today = date.today().isoformat()
    result = combine_chain.invoke({"input_documents": docs, "question": q, "today": today}) 
    return {"answer": result["output_text"].strip()}

# @traceable(name="Fallback LLM Answer")
def fallback_llm_node(state: RAGState):
    question = state["question"]
    result = fallback_chain.invoke({"question": question})
    return {"answer": result["text"].strip()}

# =========================================================
# Build Graph
# =========================================================

graph = StateGraph(RAGState)

graph.add_node("reformulate", reformulate_node) 
graph.add_node("retrieve_new", retrieve_new_node)
graph.add_node("answer_new", answer_new_node)
graph.add_node("retrieve_old", retrieve_old_node)
graph.add_node("answer_old", answer_old_node)
graph.add_node("fallback_llm", fallback_llm_node)



graph.set_entry_point("reformulate")
graph.add_edge("reformulate", "retrieve_new")
graph.add_edge("retrieve_new", "answer_new")
# Conditional edges after new answer
def next_after_answer_new(state):
    answer = state.get("answer", "").lower()
    if "could not find any relevant information" in answer or "could not find any information" in answer:
        return "retrieve_old"
    return END
graph.add_conditional_edges("answer_new", next_after_answer_new)
graph.add_edge("retrieve_old", "answer_old")
def next_after_answer_old(state):
    answer = state.get("answer", "").lower()
    if "could not find any relevant information" in answer or "could not find any information" in answer:
        return "fallback_llm"
    return END
graph.add_conditional_edges("answer_old", next_after_answer_old)
graph.add_edge("fallback_llm", END)
app = graph.compile()
