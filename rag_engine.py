"""
rag_engine.py
Retrieval-Augmented Generation using:
  - InLegalBERT embeddings (retrieval half) via ChromaDB
  - Ollama (CPU-friendly) or Aalap-Mistral-7B via transformers (GPU) for generation
"""

import chromadb
import requests

import config
from ingest import load_embedding_model

_embed_model = None
_llm_model = None
_llm_tokenizer = None
_chroma_collection = None


def get_embedding_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = load_embedding_model()
    return _embed_model


def get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
        _chroma_collection = client.get_collection(config.COLLECTION_NAME)
    return _chroma_collection


def truncate_context(context_chunks, max_chars=None):
    max_chars = max_chars or config.MAX_CONTEXT_CHARS
    selected = []
    total = 0
    for chunk, source in context_chunks:
        entry = f"[Source: {source}]\n{chunk}\n\n"
        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                truncated_chunk = chunk[: max(0, remaining - len(source) - 20)].rstrip()
                selected.append((truncated_chunk + "...", source))
            break
        selected.append((chunk, source))
        total += len(entry)
    return selected


def get_llm():
    """Only used when config.INFERENCE_BACKEND == 'transformers' (GPU path)."""
    global _llm_model, _llm_tokenizer
    if _llm_model is not None:
        return _llm_model, _llm_tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading base model: {config.BASE_LLM_MODEL}")

    quant_config = None
    if config.LOAD_IN_4BIT and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    _llm_tokenizer = AutoTokenizer.from_pretrained(config.BASE_LLM_MODEL)
    _llm_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_LLM_MODEL,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    if config.USE_FINE_TUNED_ADAPTER:
        from peft import PeftModel
        _llm_model = PeftModel.from_pretrained(_llm_model, str(config.LORA_OUTPUT_DIR))

    return _llm_model, _llm_tokenizer


def _find_ollama_cli() -> str | None:
    import os
    import shutil

    ollama_cmd = shutil.which("ollama") or shutil.which("ollama.exe")
    if ollama_cmd:
        return ollama_cmd

    local_path = os.path.join(os.getenv("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
    if local_path and os.path.exists(local_path):
        return local_path

    return None


def generate_text(prompt: str, max_new_tokens: int = None) -> str:
    """Single entry point for text generation. Routes to Ollama (CPU) or
    transformers (GPU) depending on config.INFERENCE_BACKEND."""
    max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS

    if config.INFERENCE_BACKEND == "ollama":
        http_err = None
        try:
            response = requests.post(
                config.OLLAMA_URL,
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": config.OLLAMA_KEEP_ALIVE,
                    "options": {
                        "temperature": config.TEMPERATURE,
                        "num_predict": max_new_tokens,
                    },
                },
                timeout=config.OLLAMA_HTTP_TIMEOUT,
                stream=True,
            )
            response.raise_for_status()

            # Ollama returns NDJSON chunks for streamed responses.
            import json

            text_parts = []
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("done") is True and "response" in chunk:
                    text_parts.append(chunk.get("response", ""))
                    break
                if "response" in chunk:
                    text_parts.append(chunk["response"])

            joined = "".join(text_parts).strip()
            if joined:
                return joined

            raw_text = response.text
            try:
                data = json.loads(raw_text)
            except ValueError:
                data = raw_text
            if isinstance(data, dict):
                if "response" in data:
                    return data["response"].strip()
                if "text" in data:
                    return data["text"].strip()
                if "output" in data:
                    return data["output"].strip()
            elif isinstance(data, str):
                return data.strip()
            http_err = RuntimeError(f"Unexpected Ollama HTTP response: {data}")
        except Exception as e:
            http_err = e

        # CLI fallback: try running the local `ollama` CLI if the HTTP API fails.
        try:
            import subprocess
            import re

            ollama_cmd = _find_ollama_cli()
            if not ollama_cmd:
                raise RuntimeError("Ollama HTTP failed and no Ollama CLI found on PATH or default location.")

            proc = subprocess.run(
                [ollama_cmd, "run", config.OLLAMA_MODEL, *config.OLLAMA_CLI_FLAGS],
                input=prompt,
                capture_output=True,
                timeout=config.OLLAMA_CLI_TIMEOUT,
                encoding="utf-8",
                errors="replace",
            )
            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                stdout = (proc.stdout or "").strip()
                raise RuntimeError(
                    f"Ollama CLI failed (exit {proc.returncode}): {stderr or stdout}"
                )

            stdout = proc.stdout or ""
            ansi_re = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
            output = ansi_re.sub("", stdout).strip()
            if not output:
                raise RuntimeError("Ollama CLI returned empty output.")
            return output
        except Exception as e_cli:
            raise RuntimeError(
                f"Failed to generate text via Ollama HTTP ({http_err}) and CLI fallback ({e_cli})"
            ) from e_cli

    # --- transformers (GPU) backend ---
    import torch
    try:
        model, tokenizer = get_llm()
    except Exception as e:
        raise RuntimeError(f"Failed to load LLM model: {e}") from e

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    except Exception as e:
        raise RuntimeError(f"Tokenization failed: {e}") from e

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=config.TEMPERATURE,
                do_sample=config.TEMPERATURE > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}") from e

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return full_text[len(prompt_text):].strip()


def retrieve_context(question: str, top_k: int = None):
    top_k = top_k or config.TOP_K_RETRIEVAL
    embed_model = get_embedding_model()
    collection = get_chroma_collection()

    query_embedding = embed_model.encode([question], convert_to_numpy=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return list(zip(chunks, sources))


def build_prompt(question: str, context_chunks):
    truncated_chunks = truncate_context(context_chunks)
    if truncated_chunks:
        context_text = "\n\n".join(
            f"[Source: {source}]\n{chunk}" for chunk, source in truncated_chunks
        )
    else:
        context_text = "No relevant legal context was found. Answer based on the question only."

    prompt = (
        f"{config.SYSTEM_INSTRUCTIONS}\n\n"
        f"LEGAL CONTEXT:\n{context_text}\n\n"
        f"USER QUESTION: {question}\n\n"
        f"ANSWER:"
    )
    return prompt


def answer_question(question: str) -> dict:
    context_chunks = retrieve_context(question)
    prompt = build_prompt(question, context_chunks)
    answer = generate_text(prompt)

    return {
        "answer": answer,
        "sources": sorted(set(source for _, source in context_chunks)),
    }


if __name__ == "__main__":
    q = input("Ask a legal question: ")
    result = answer_question(q)
    print("\n--- ANSWER ---")
    print(result["answer"])
    print("\n--- SOURCES ---")
    print(", ".join(result["sources"]))