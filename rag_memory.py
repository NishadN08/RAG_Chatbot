import os, shutil, re
import json
from typing import Dict, Any

# LangChain core imports and Text Splitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store / utils
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# Embeddings & LLM
from langchain_huggingface import HuggingFaceEmbeddings   
from langchain_ollama import OllamaLLM

# Retrievers & Encoders(for Reranking)
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Chains & Prompts
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain, ConversationalRetrievalChain, StuffDocumentsChain

# =========================
# Config
# =========================
JSONL_PATH = "sc_test_3.jsonl"
PERSIST_DIR = "./chroma_pdf_db"
COLLECTION_NAME = "fsu_sc"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = "gemma3:12b"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 50

ALLOWED_META_TYPES = (str, int, float, bool, type(None))

# =========================
# Helpers
# =========================

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

    # Remove all newlines and collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()


    # Deduplicate lines while keeping order
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    return "\n".join(unique_lines)

def format_chunks(docs: list, max_items, max_chars):
    """
    Convert retrieved chunks into JSON-friendly format.
    """
    retrieved_chunks = []
    if not docs:
        return [{"title": "[none]", "url": "", "snippet": ""}]

    for i, d in enumerate(docs[:max_items], 1):
        meta = d.metadata or {}
        title = meta.get("title", "") or "[No Title]"
        url = meta.get("source", "") or ""
        snippet = (d.page_content or "").replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + " ..."

        retrieved_chunks.append({
            "id": i,
            "title": title,
            "url": url,
            "snippet": snippet
        })
    return retrieved_chunks

# =========================
# Date Extractor
# =========================

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

# =========================================================
# Document Loading & Sanitization
# =========================================================
def load_jsonl_as_docs(path: str) -> list[Document]:
    """
    Load and clean JSONL documents into LangChain Document objects.
    """

    docs: list[Document] = []
    
    # Read JSONL file line-by-line
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not (line := line.strip()): # Skip empty / whitespace lines
                continue

            # Parse JSON safely
            try: 
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = rec.get("url", "") or ""

            # Skip URLs with pagination or view parameters
            if (
                "?start=" in url
                or "?view=" in url
                or "?option=" in url
            ):
                continue

            # Extract raw text
            text = rec.get("text", "") or ""

            #Clean duplicates inside the text
            text = clean_text_block(text)

             # If text becomes empty after cleaning → skip record
            if not text.strip():
                continue

            #Creating the Metadata
            md: Dict[str, Any] = {
                "source": rec.get("url", "") or "",
                "title": rec.get("title", "") or "",
                "emails": rec.get("emails", []),
                "external_profile_links": rec.get("external_profile_links", []),
                "date": extract_date_from_url(rec.get("url", "")),
            }

            # Convert into a LangChain Document
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

# Add date + source metadata into page_content
# This ensures the LLM sees date/source context directly in the text
for c in chunks:
    date = c.metadata.get("date", "[Date: Unknown]")
    source = c.metadata.get("source", "[Source: Unknown]")
    c.page_content = f"[Date: {date}] [Source: {source}] {c.page_content}"

print(f"Document split into {len(chunks)} chunks.")

# Sanitize metadata for Chroma (turn lists → strings, drop noisy fields)
chunks = [sanitize_metadata(c) for c in chunks]

# Optional extra guard (operates on list[Document], not dict)
chunks = filter_complex_metadata(chunks)

# Validate (will raise if anything non-primitive sneaks in)
assert_all_metadata_primitive(chunks)

# ==================================
# 3) Embeddings + Chroma (persist)
# ===================================
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

# If you want a clean rebuild each run, uncomment:
if os.path.isdir(PERSIST_DIR):
     shutil.rmtree(PERSIST_DIR)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
)
vectorstore.persist()


# =========================================================
# 4) Retriever Setup (Vector + Reranker)
# =========================================================

# 1) Standard vector-based retriever (semantic similarity search)
vector_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 700},  # tune k (10 works well for names)
)


# 2) Load Cross Encoder model for reranking
# This model scores text pairs (query, document) more accurately
cross_encoder = HuggingFaceCrossEncoder(model_name = CROSS_ENCODER_MODEL_NAME)

# 3) Wrap CrossEncoder in a LangChain-compatible reranker
# top_n=25 → After reranking, keep the best 25 chunks only
compressor = CrossEncoderReranker(
    model=cross_encoder,
    top_n=50
)

# 4) Build a hybrid retriever:
#    - First fetch top 500 via vector similarity
#    - Then rerank them using cross-encoder scoring
retriever = ContextualCompressionRetriever(
    base_retriever=vector_retriever,      # Vector search step
    base_compressor=compressor            # Cross-encoder reranker step
)

# =========================================================
# 5) Recency-Aware Retriever
# =========================================================

def semester_year_to_tuple(date_str: str) -> tuple[int, int]:
    """
    Assign a continuous offset relative to the current semester and year.
    Current semester/year = (current_year, 2)
    Previous semesters decrease by 1 each step
    Next semesters increase by 1 each step
    """
    # If date is empty or not a string → return invalid marker
    if not date_str or not isinstance(date_str, str):
        return (0, -999)

    # Normalize whitespace + capitalize first letter of each word
    date_str = date_str.strip().title()

    parts = date_str.split() # Split into ["Fall", "2023"]
    if len(parts) != 2:
        return (0, -999)

    semester, year_str = parts
    try:
        year = int(year_str) # Convert year string to integer
    except ValueError:
        return (0, -999)

    # Allowed semester names (fixed ordering)
    semesters = ["Spring", "Summer", "Fall"]
    if semester not in semesters:
        return (year, -999)
    
    # Get current year (e.g., 2025)
    current_year = int(datetime.now().year)

    # Convert semester to index 
    sem_index = semesters.index(semester)

    # Compute continuous offset so semesters can be compared numerically
    # Example:
    #   If current year = 2025:
    #       Fall 2025  → (2025 - 2025)*3 + 2 = 2
    #       Summer 2024 → (2024 - 2025)*3 + 1 = -2
    offset = (year - current_year) * 3 + sem_index 

     # Return tuple: (year, offset) for sorting
    return (year, offset)


class RecencyPriorityRetriever(BaseRetriever):
    """
    Sorts the retrieved chunks by the most recent chunks relative to the current semester/year.
    Returns the top k chunks after sorting
    """
    base_retriever: BaseRetriever
    top_k: int = 10

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        
        # Build: (doc, (year, offset))
        docs_with_offsets = [
            (d, semester_year_to_tuple(d.metadata.get("date", "")))
            for d in docs
        ]
        # Filter out invalid (year <= 0)
        docs_with_offsets = [dt for dt in docs_with_offsets if dt[1][0] > 0]

        if not docs_with_offsets:
            return docs  # fallback if no valid dates
        
        # Sort by year (desc) and offset (desc)
        sorted_docs = sorted(
            docs_with_offsets,
            key=lambda x: (x[1][0], x[1][1]),  # (year, offset)
            reverse=True
        )

        # Slice the top_k documents
        top_docs = sorted_docs[:self.top_k]

        # Return only the sorted Document objects
        return [d for d, _ in top_docs]
    
# # Attaching the cross-encoder retriever to the recency_priority_retriever
recency_priority_retriever = RecencyPriorityRetriever(base_retriever=retriever)

# =========================
# 6) LLM (Ollama)
# =========================
llm = Ollama(model=OLLAMA_MODEL)

# ==================================
# 7) Main Prompt + doc_chain
# ==================================
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=("""
        You are a Helpful AI assistant of The Department of Scientific Computing at Florida State University that answers questions **strictly** using the given context.
        The context consists of multiple chunks of text, each with a date indicating when the information was relevant.

        Each chunk may mention multiple people or events — focus **only** on the person or topic directly asked about.

        Your goal is to provide the most **recent and accurate** information available.

        Instructions:
        1. Review all provided chunks carefully.
        2. Prefer chunks with newer dates (more recent events or updates).
        3. Prioritize information that directly matches the question subject.
        4. Prefer newer dates over older ones.
        5. If the information is outdated or uncertain, clearly mention that.
        6. If asked about a person give some more information on the person.
        7. Never include information about unrelated people or topics.
        8. If no relevant information is found, make a response like I could not find any relevant information about your question.


        Question:
        {question}

        Context:
        {context}

        Guidelines:
        - Give the final answer based on the most recent chunk(s).
        - Do not merge unrelated or outdated information.
        - If two chunks conflict, choose the one with the latest date.
        - Answer only using the context above.
        - Be concise and factual, and focused only on the query subject.
        """
    ),
)

# Chain that combines the docs, fills the prompt and sends to LLM
doc_chain = LLMChain(llm=llm, prompt=custom_prompt)

# StuffDocumentsChain merges multiple documents into the "context" variable
combine_docs_chain = StuffDocumentsChain(
    llm_chain=doc_chain,
    document_variable_name="context"
)

# -------------------------------------
# 8) Question Reformulation Prompt
# -------------------------------------
condense_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=(
        "You are a Helpful AI assistant of The Department of Scientific Computing.\n"
        "Given the chat history and the latest user question and answer which might reference context in the chat history.\n"
        "If the latest user question is not related to the latest question in the chat history, return the question as it is\n"
        "Your task is to reformulate the latest user question into a standalone question which can be understood.\n"
        "DO NOT answer the question, just reformulate it if needed and otherwise return it as it is.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Follow-up question: {question}\n\n"
        "Standalone question:"
    ),
)

# LLM chain for reformulating the question
question_generator = LLMChain(llm=llm, prompt=condense_prompt)



# --------------------------------------
# 9) Final Conversational RAG Chain
# --------------------------------------
qa_chain = ConversationalRetrievalChain(
    retriever=recency_priority_retriever,       # recency-aware retriever + cross-encoder + vector retriever
    return_source_documents=True,               # Return chunks used for debugging
    question_generator=question_generator,      # Reformulate follow-up questions
    combine_docs_chain=combine_docs_chain,      # Final answering chain
)


