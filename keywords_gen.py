import json
import re
from typing import List
from tqdm import tqdm

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


# =========================================================
# CONFIG
# =========================================================

JSON_FILES = [
    "new1.jsonl",
    "old1.jsonl"
]

OUTPUT_FILE = "clean_keywords1.txt"

OLLAMA_MODEL = "llama3.1:8b"

# only control batching by number of docs
DOCS_PER_BATCH = 5


# =========================================================
# LLM
# =========================================================

llm = OllamaLLM(
    model=OLLAMA_MODEL,
    temperature=0
)


# =========================================================
# PROMPT
# =========================================================

prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are extracting searchable keywords from university website documents.

Extract all the important searchable keywords such as:
- faculty names
- research areas
- course codes
- software
- technical topics
- labs
- organizations

Note: These are just examples,you can extract any other unique and important keywords you find in the text.

Rules:
- Return ONLY a comma-separated list
- No explanations
- No numbering
- Normalize spacing
- Avoid duplicates

Retrieve all the words which are unique and important for search and discovery.
I am going to use these keywords to correct spelling errors and help users find relevant documents on a university website.
Documents:
{text}

Keywords:
"""
)


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# LOAD JSONL FILES
# =========================================================

def load_jsonl_files(files: List[str]) -> List[str]:

    docs = []

    for file in files:

        with open(file, "r", encoding="utf-8") as f:

            for line in f:

                if not line.strip():
                    continue

                obj = json.loads(line)

                title = obj.get("title", "")
                text = obj.get("text", "")

                combined = f"{title}. {text}"
                combined = clean_text(combined)

                docs.append(combined)

    return docs


# =========================================================
# BATCH DOCUMENTS (NO CHAR LIMIT)
# =========================================================

def batch_documents(docs: List[str]):

    batches = []

    for i in range(0, len(docs), DOCS_PER_BATCH):

        batch = docs[i:i + DOCS_PER_BATCH]

        batches.append("\n\n".join(batch))

    return batches


# =========================================================
# KEYWORD EXTRACTION
# =========================================================

def extract_keywords(batch_text):

    chain = prompt | llm

    result = chain.invoke({"text": batch_text})

    keywords = []

    for k in result.split(","):

        k = k.strip().lower()

        if len(k) > 2:
            keywords.append(k)

    return keywords

# =====================================================
# Cleaning Function
# =====================================================

def clean_keywords(line: str):
    """
    Clean and validate extracted keywords.
    Returns a cleaned keyword or None if it should be discarded.
    """

    line = line.strip().lower()

    # remove numbering (1. keyword)
    line = re.sub(r"^\d+\.?\s*", "", line)

    # remove markdown symbols
    line = re.sub(r"[*_#>`]", "", line)

    # remove sentences longer than 5 words
    if len(line.split()) > 5:
        return None

    # remove punctuation typical of sentences
    if re.search(r"[.:;!?]", line):
        return None

    # ignore very short strings
    if len(line) <= 2:
        return None

    return line


# =========================================================
# MAIN
# =========================================================

def main():

    docs = load_jsonl_files(JSON_FILES)
    print("Total documents loaded:", len(docs))

    batches = batch_documents(docs)
    print("Total batches:", len(batches))

    keyword_set = set()

    for batch in tqdm(batches):

        try:
            raw_keywords = extract_keywords(batch)

            for kw in raw_keywords:
                cleaned = clean_keywords(kw)

                if cleaned:
                    keyword_set.add(cleaned)

        except Exception as e:
            print("Error:", e)

    keywords = sorted(keyword_set)

    print("\nTotal unique keywords:", len(keywords))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for k in keywords:
            f.write(k + "\n")

    print("Keywords saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()