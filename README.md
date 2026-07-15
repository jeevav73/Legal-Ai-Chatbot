# Nyaya Sahayak — Indian Law Legal-Aid Chatbot (Local ML, No API)

A RAG-based (Retrieval-Augmented Generation) chatbot that runs **entirely on open-source
models** — no Claude/OpenAI API key needed. It:

1. Answers questions about Indian law using retrieved legal context + a local LLM.
2. Suggests legal remedies for specific situations (e.g. "police filed a false case on
   me — what are my options?").
3. Auto-generates a draft complaint document (.docx) from a structured intake form, which
   the user should review with an advocate before filing.

> ⚠️ **Important**: This tool gives *general legal information and draft documents*, not
> legal advice from a licensed advocate. Every generated complaint/petition must be
> reviewed by a qualified lawyer before filing.

## 1. Models used (research summary)

| Purpose | Model | Why |
|---|---|---|
| Retrieval embeddings | `law-ai/InLegalBERT` | Domain-pretrained on Indian legal text (statutes, judgments) — over 1.8M downloads on HuggingFace, understands Indian legal phrasing much better than a generic embedding model. |
| Answer generation | `opennyaiorg/Aalap-Mistral-7B-v0.1-bf16` | Mistral-7B already instruction-fine-tuned by OpenNyAI for Indian legal tasks; runs comparably to GPT-3.5 on the legal tasks it was trained for. Fully open weights. |
| Further fine-tuning | QLoRA (via `peft` + `bitsandbytes`) on top of Aalap | Lets you teach it your own complaint-drafting style/format cheaply — needs only ~12-16GB VRAM, not a full GPU cluster. |

Both models are free, open-source, and downloaded automatically from HuggingFace on
first run (no login/token required for either).

## 2. Hardware requirements

- **Inference only (chatbot answering questions):** a GPU with ~6-8GB VRAM (4-bit
  quantized) is enough — e.g. RTX 3060/4060, or free Google Colab / Kaggle T4.
  CPU-only will work but will be slow (expect 30s-2min per answer).
- **Fine-tuning (train.py):** ~12-16GB VRAM recommended (free Colab/Kaggle T4 16GB
  works for QLoRA on a 7B model).

## 3. Project Structure

```
legal_chatbot_ai/
├── data/
│   ├── legal_docs/          # Legal reference text files (RAG source — add more here)
│   └── fine_tune/           # train.jsonl for QLoRA fine-tuning (auto-created with an example)
├── vectorstore/             # ChromaDB persisted embeddings (auto-created by ingest.py)
├── models/nyaya-aalap-lora/ # Your fine-tuned LoRA adapter lands here after train.py
├── generated_complaints/    # Output .docx complaints land here
├── config.py                # All settings — model names, paths, prompt instructions
├── ingest.py                # Builds the vector database from data/legal_docs
├── rag_engine.py            # Retrieval (InLegalBERT) + generation (Aalap) — the core pipeline
├── complaint_generator.py   # Structured intake -> LLM draft -> .docx complaint
├── train.py                 # QLoRA fine-tuning script (optional, for customizing Aalap further)
├── app.py                   # Streamlit UI (chat + complaint generator tabs)
├── cli.py                   # Terminal-only version (no Streamlit needed)
├── requirements.txt
├── .env.example
└── README.md
```

## 4. Setup (VS Code)

Open this folder in VS Code, then in the integrated terminal:

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies (this pulls in torch, transformers, peft, etc.)
pip install -r requirements.txt

# 3. (Optional) copy env file — only needed if you later use a gated HF model
cp .env.example .env

# 4. Build the vector database from the legal documents
python ingest.py

# 5a. Run the chat + complaint UI
streamlit run app.py

# 5b. OR run the terminal-only version
python cli.py
```

First run will download Aalap-Mistral-7B (~14GB) and InLegalBERT (~450MB) from
HuggingFace automatically — make sure you have disk space and a stable connection.

## 5. Fine-tuning the model on your own data

```bash
python train.py
```

The first run creates a starter example at `data/fine_tune/train.jsonl` showing the
expected format:
```json
{"instruction": "...", "input": "", "output": "..."}
```
Add more rows — the more good-quality, consistent examples you write, the better the
model gets at your exact complaint style. Good sources to expand this dataset:
- `opennyaiorg/aalap_instruction_dataset` (HuggingFace) — same format Aalap itself
  was trained on
- **IndicLegalQA** (HuggingFace, 2025) — Indian judicial Q&A dataset
- Your own curated examples of good complaint drafts / remedy explanations

Then re-run `python train.py`. Once done, set `USE_FINE_TUNED_ADAPTER=true` in `.env`
and restart `app.py`/`cli.py` — it will automatically load your LoRA adapter on top of
the base model.

## 6. Adding more legal reference data (for retrieval, not fine-tuning)

Drop plain `.txt` files into `data/legal_docs/`. Each file should cover one Act/topic
(e.g. `bns_wrongful_confinement.txt`, `it_act_cyber_crimes.txt`). Good public sources:
India Code (indiacode.nic.in), Indian Kanoon (indiankanoon.org), NCRB, and your State
Legal Services Authority's citizen guides.

After adding files, re-run:
```bash
python ingest.py
```

## 7. How the pipeline works, end to end

1. `ingest.py` reads every `.txt` in `data/legal_docs/`, splits into ~350-word
   overlapping chunks, embeds each chunk with InLegalBERT, and stores vectors + text
   in a local ChromaDB collection (`vectorstore/`).
2. `rag_engine.py`, on a user question:
   - Embeds the question with the same InLegalBERT model
   - Retrieves the top-k most similar chunks from ChromaDB
   - Builds a prompt instructing the model to answer *only* from that context and cite
     the Act/Section
   - Generates the answer locally with Aalap-Mistral-7B (4-bit quantized)
3. `complaint_generator.py` takes structured intake (who/what/when/where/evidence/relief
   requested), retrieves relevant legal context the same way, asks the local LLM to draft
   formal complaint prose, and renders it into a `.docx` via `python-docx` — with a
   built-in disclaimer footer reminding the user to get advocate review.

## 8. Known limitations to keep in mind

- Aalap is not a general-purpose Indian legal LLM — the paper notes it does not do any
  better than base Mistral-7B on the All India Bar Exam or LegalBench, only on the
  specific tasks it was trained on. Fine-tune it (Section 5) on the exact tasks you need
  (remedy explanation, complaint drafting) for best results.
- Always double-check Section numbers: India moved from IPC/CrPC/Evidence Act to
  BNS/BNSS/BSA in 2023-2024. Bare acts should be the final source of truth, not the
  model's memory.
- This is legal information + drafting assistance, not a lawyer. Keep the disclaimer
  visible in the UI (already included in `app.py`) and in every generated document
  (already included in `complaint_generator.py`).

## 9. Extending this later

- Add OCR/PDF ingestion (`pypdf`) so you can drop official PDF bare-acts straight in.
- Add a rules-based "case classifier" (cognizable/non-cognizable, bailable/non-bailable)
  before suggesting remedies, to reduce reliance on the LLM for structured facts.
- Add authentication + per-user history (SQLite is enough at small scale).
- Convert the base model to GGUF and run via `llama.cpp`/Ollama if you want CPU-only
  deployment without a GPU.
