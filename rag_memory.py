# %%
import os, shutil, re
import json
from typing import Dict, Any


from langchain.schema import Document

# Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store / utils
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# Embeddings / LLM
from langchain.embeddings import HuggingFaceEmbeddings   
from langchain_community.llms import Ollama

# Retrievers / Encoders
from langchain.retrievers import EnsembleRetriever, BM25Retriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Chains / prompts
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain, StuffDocumentsChain

from langchain.retrievers import EnsembleRetriever

# =========================
# Config
# =========================
JSONL_PATH = "sc_test_3.jsonl"
PERSIST_DIR = "./chroma_pdf_db"
COLLECTION_NAME = "fsu_sc"

MEMORY_PERSIST_DIR = "./chroma_memory_db"
MEMORY_COLLECTION = "chat_memory"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:72b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

SHOW_TOPK_IN_CONTEXT = 5        # from the retriever (what the LLM actually saw)
PREVIEW_CHARS = 4000  

# =========================
# Helpers
# =========================
ALLOWED_META_TYPES = (str, int, float, bool, type(None))

def clean_text_block(text: str) -> str:
    """Clean raw scraped text by removing noise and duplicates."""
    # Remove webmaster/edit info
    text = re.sub(r"Edit Information.*?(removed\.)", "", text, flags=re.DOTALL)
    text = re.sub(r"Please Email.*?removed\.", "", text, flags=re.DOTALL)

    # Remove placeholders
    text = re.sub(r"Click here to email the webmaster.*?go away\.", "", text)

    # Fix inline duplicate labels
    text = re.sub(r"\b(Research Interests|Education|Publications|Address|Contact Info):\s*\1\b", r"\1:", text)

    # Collapse repeated words/emails/URLs inline
    text = re.sub(r"\b(\S+)( \1)+\b", r"\1", text)


    # Deduplicate lines while keeping order
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    return "\n".join(unique_lines)


def load_jsonl_as_docs(path: str) -> list[Document]:
    docs: list[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = rec.get("text", "") or ""
            if not text.strip():
                continue

             # ✅ Clean duplicates inside the text
            text = clean_text_block(text)

            md: Dict[str, Any] = {
                "source": rec.get("url", "") or "",
                "title": rec.get("title", "") or "",
                "emails": rec.get("emails", []),  # may be list → sanitize later
                "external_profile_links": rec.get("external_profile_links", [])
            }
            docs.append(Document(page_content=text, metadata=md))
    return docs


def coerce_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all metadata values are primitives (str/int/float/bool/None)."""
    out: Dict[str, Any] = {}
    for k, v in (md or {}).items():
        if isinstance(v, ALLOWED_META_TYPES):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = ", ".join(str(x) for x in v)  # e.g., emails list -> "a@x, b@y"
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out


def sanitize_metadata(d: Document) -> Document:
    """Flatten/clean metadata dict (no call to filter_complex_metadata here)."""
    md = dict(d.metadata or {})
    md.pop("anchor_texts", None)
    md.pop("out_links", None)
    md = coerce_metadata(md)
    return Document(page_content=d.page_content, metadata=md)


def assert_all_metadata_primitive(docs: list[Document]) -> None:
    for i, d in enumerate(docs):
        for k, v in (d.metadata or {}).items():
            if not isinstance(v, ALLOWED_META_TYPES):
                raise ValueError(f"Non-primitive metadata at doc #{i}, key '{k}': {type(v)} -> {v!r}")


def sources(source_docs: list[Document]) -> str:
    lines = []
    seen = set()
    for i, d in enumerate(source_docs, 1):
        src = d.metadata.get("source", "")
        title = d.metadata.get("title", "")
        key = (src, title)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{i}. {title or '[No Title]'}\n   {src}")
    return "\n".join(lines) if lines else "No sources returned."

def print_chunks(label: str, docs: list[Document], max_items: int = 5, max_chars: int = 500):
    print(f"\n----- {label} (showing up to {max_items}) -----")
    if not docs:
        print("[none]")
        return
    for i, d in enumerate(docs[:max_items], 1):
        meta = d.metadata or {}
        title = meta.get("title", "") or "[No Title]"
        url = meta.get("source", "") or ""
        snippet = (d.page_content or "").replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + " ..."
        print(f"\n[{i}] {title}\nURL: {url}\n---\n{snippet}")


def chunks_to_txt(chunks, file_path="retrieved_chunks.txt", max_chunks=None, max_chunk_chars=10000):
    with open(file_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            if max_chunks and i >= max_chunks:
                break
            title = chunk.metadata.get("title", "[No Title]")
            source = chunk.metadata.get("source", "[No Source]")
            text = chunk.page_content
            if len(text) > max_chunk_chars:
                text = text[:max_chunk_chars] + " ..."
            f.write(f"[{i+1}] Title: {title}\n")
            f.write(f"Source: {source}\n")
            f.write(f"Text:\n{text}\n")
            f.write("-" * 80 + "\n")
    print(f"Retrieved chunks saved to: {file_path}")

def save_chunks_to_txt(chunks: list[Document], file_path: str = "chunks.txt"):
    """
    Save each chunk's text (and optionally metadata) to a TXT file.
    
    Args:
        chunks: List of Document objects.
        file_path: Path to save the TXT file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"----- Chunk {i} -----\n")
            title = chunk.metadata.get("title", "[No Title]")
            source = chunk.metadata.get("source", "[No Source]")
            f.write(f"Title: {title}\n")
            f.write(f"Source: {source}\n")
            f.write(f"Text:\n{chunk.page_content}\n")
            f.write("\n\n")
    print(f"Saved {len(chunks)} chunks to {file_path}")

def dedupe_chunks(chunks):
    seen = set()
    unique_chunks = []
    for c in chunks:
        text = c.page_content.strip()
        if text not in seen:
            seen.add(text)
            unique_chunks.append(c)
    return unique_chunks    

# =========================
# 1) Load JSONL
# =========================
docs = load_jsonl_as_docs(JSONL_PATH)
print(f"Loaded {len(docs)} JSONL docs.")

# =========================
# 2) Split into chunks
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(docs)
print(f"Total number of chunks before sanitize: {len(chunks)}")

chunks = dedupe_chunks(chunks)
print(f"Total number of chunks after dedupe: {len(chunks)}")


# Sanitize metadata for Chroma (turn lists → strings, drop noisy fields)
chunks = [sanitize_metadata(c) for c in chunks]

# Optional extra guard (operates on list[Document], not dict)
chunks = filter_complex_metadata(chunks)

# Validate (will raise if anything non-primitive sneaks in)
assert_all_metadata_primitive(chunks)

print(f"Total number of chunks after sanitize:  {len(chunks)}")
if chunks:
    print("Sample chunk metadata:", chunks[0].metadata)
save_chunks_to_txt(chunks, "all_chunks.txt")


# =========================
# Memory Store
# =========================
def store_chat_in_memory(question: str, answer: str):
    doc_text = f"Human Message: {question}\nAI Message: {answer}"
    doc = Document(page_content=doc_text, metadata={"type": "chat_history"})
    memory_vectorstore.add_documents([doc])
    memory_vectorstore.persist()


# =========================
# 3) Embeddings + Chroma (persist)
# =========================
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

# If you want a clean rebuild each run, uncomment:
# if os.path.isdir(PERSIST_DIR):
#      import shutil; shutil.rmtree(PERSIST_DIR)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
)
vectorstore.persist()

# Delete previous memory directory if it exists
if os.path.exists(MEMORY_PERSIST_DIR):
    shutil.rmtree(MEMORY_PERSIST_DIR)

memory_vectorstore = Chroma(
    embedding_function=embedding_model,
    collection_name=MEMORY_COLLECTION,
    persist_directory=MEMORY_PERSIST_DIR,
)

# =========================
# Hybrid Retriever (Vector + BM25)
# =========================

# Vector retriever (semantic similarity only)
vector_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20},  # tune k (10 works well for names)
)

# BM25 retriever (keyword-based)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 20

# Combine them
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # 70% vector + 30% keyword
    
)

# Cross-encoder reranker
cross_encoder = HuggingFaceCrossEncoder(model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Wrap in LangChain's CrossEncoderReranker
compressor = CrossEncoderReranker(model=cross_encoder)


# Final retriever with reranking
retriever = ContextualCompressionRetriever(
    base_retriever=hybrid_retriever,
    base_compressor=compressor,
    k=10  # keep top-5 chunks after reranking
)
# =========================
# Memory Retriever (Hybrid + Memory)
# =========================

# Create memory retriever
memory_retriever = memory_vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}  # fewer docs since it's short Q&A
)

combined_retriever = EnsembleRetriever(
    retrievers=[retriever, memory_retriever],  # knowledge + memory
    weights=[0.8, 0.2]  # tune: more weight to knowledge base
)

# =========================
# 4) LLM (Ollama)
# =========================
llm = Ollama(model=OLLAMA_MODEL)

# =========================
# 5) Prompt + doc_chain
# =========================
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant.\n"
        # "Conversation so far:\n{chat_history}\n\n"
        # "Use ONLY the information provided below to answer the question.\n"
        "Use the following pieces of retrived context to answer the question\n"
        "If asked about a person give some more information on the person\n"
        # "If the new question depends on previous ones, use both history and info.\n"
        # "If unrelated, ignore history and answer only with the given info.\n\n"
        "Provide the answer DIRECTLY. If the answer is not present, reply ONLY with: Not available in the document\n"
        "Information:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ),
)

doc_chain = LLMChain(llm=llm, prompt=custom_prompt)
combine_docs_chain = StuffDocumentsChain(
    llm_chain=doc_chain,
    document_variable_name="context"  # must match {context} in prompt

)

# =========================
# 5) Memory Prompt + QA chain
# =========================

condense_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=(
        "Given the chat history and the latest user question and answer which might reference context in the chat history.\n"
        "Formulate a standalone question which can be understood without the chat history\n"
        "DO NOT answer the question, just reformulate it if needed and otherwise return it as it is.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Follow-up question: {question}\n\n"
        "Standalone question:"
    ),
)
question_generator = LLMChain(llm=llm, prompt=condense_prompt)



qa_chain = ConversationalRetrievalChain(
    retriever=combined_retriever,
    return_source_documents=True,
    question_generator=question_generator,
    combine_docs_chain=combine_docs_chain,
)

