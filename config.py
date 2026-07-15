"""
Central configuration for the Nyaya Sahayak legal chatbot.
No external API keys needed — everything runs on local/open-source models.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
# Data & storage paths
# ---------------------------------------------------------------------
LEGAL_DOCS_DIR = BASE_DIR / "data" / "legal_docs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
GENERATED_COMPLAINTS_DIR = BASE_DIR / "generated_complaints"
FINE_TUNE_DATA_DIR = BASE_DIR / "data" / "fine_tune"
LORA_OUTPUT_DIR = BASE_DIR / "models" / "nyaya-aalap-lora"

for d in (VECTORSTORE_DIR, GENERATED_COMPLAINTS_DIR, FINE_TUNE_DATA_DIR, LORA_OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Models (all open-source, run locally — no API key required)
# ---------------------------------------------------------------------

EMBEDDING_MODEL_PRIMARY = "law-ai/InLegalBERT"
EMBEDDING_MODEL_FALLBACK = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

BASE_LLM_MODEL = "opennyaiorg/Aalap-Mistral-7B-v0.1-bf16"

USE_FINE_TUNED_ADAPTER = os.environ.get("USE_FINE_TUNED_ADAPTER", "false").lower() == "true"

LOAD_IN_4BIT = True

# ---------------------------------------------------------------------
# Inference backend — how answers actually get generated
# ---------------------------------------------------------------------
# "ollama"        -> calls a local Ollama server (CPU-friendly, no GPU needed).
# "transformers"  -> loads Aalap-Mistral-7B directly, needs a GPU.
INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "ollama")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

# ---------------------------------------------------------------------
# RAG settings
# ---------------------------------------------------------------------
CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 50
TOP_K_RETRIEVAL = 4
COLLECTION_NAME = "indian_law_corpus"

# ---------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.3

SYSTEM_INSTRUCTIONS = """You are Nyaya Sahayak, a legal-information assistant for Indian law.
Rules you must follow:
1. Answer ONLY using the legal context provided to you. If the context does not
   contain the answer, say so clearly instead of guessing.
2. Always mention the specific Act and Section you are relying on.
3. Always end with a reminder that this is general legal information, not a
   substitute for advice from a licensed advocate, and that the user should
   consult one before taking action or filing anything.
4. Keep the tone calm, clear, and supportive — many users asking this are
   under real stress (e.g. a case has been filed against them).
"""