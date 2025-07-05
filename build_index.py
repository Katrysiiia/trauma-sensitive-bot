import os
import openai
from datasets import load_dataset
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SimpleNodeParser

# === enter your API key ===
openai.api_key = ""
# === 1. loading the dataset from HuggingFace ===
dataset = load_dataset("nbertagnolli/counsel-chat", split="train")

# === 2. documentation ===
documents = []
for item in dataset:
    question = (item.get("questionText") or "").strip()
    answer = (item.get("answerText") or "").strip()
    category = (item.get("category") or "").strip()

    if question and answer:
        text = f"[{category}]\nQuestion: {question}\nAnswer: {answer}"
        documents.append(Document(text=text))

print(f"✅ Documents loaded: {len(documents)}")

# === 3. Parsing without  SentenceSplitter because of the latest model===
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)

# === 4. Indexing
index = VectorStoreIndex(nodes)

# === 5. persisting the index to disk
PERSIST_DIR = "./storage/VectorStoreIndex"
index.storage_context.persist(persist_dir=PERSIST_DIR)

print("✅ Index built and saved successfully.")






