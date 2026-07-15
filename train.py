"""
train.py
Fine-tunes the base model (Aalap-Mistral-7B) further on YOUR OWN data using
QLoRA (4-bit quantization + LoRA adapters) — this needs far less GPU memory
than full fine-tuning (~12-16GB VRAM is enough; a free Colab/Kaggle T4 works).

STEP 1 — Prepare your training data
    Create a JSONL file at data/fine_tune/train.jsonl where each line is:
    {"instruction": "...", "input": "...", "output": "..."}

    Example line:
    {"instruction": "A user says the police filed a false case on them under
    a bailable offence. Explain their options and draft what they should ask
    for in a written complaint to the SP.",
     "input": "",
     "output": "1) Confirm the offence is bailable ... 2) You may apply for
     bail directly at the police station or before the Magistrate ... 3) Draft
     of complaint to SP: ..."}

    Good sources to build this dataset from:
      - opennyaiorg/aalap_instruction_dataset (HuggingFace) — same format Aalap
        itself was trained on, good for style reference
      - IndicLegalQA (HuggingFace, 2025 dataset) — Indian judicial QA pairs
      - Your own curated Q&A pairs about complaint drafting, written in the
        exact tone/structure you want the bot to answer in

STEP 2 — Run this script
    python train.py

STEP 3 — Use the fine-tuned model
    Set USE_FINE_TUNED_ADAPTER=true in your .env (or environment), then run
    app.py / cli.py as usual — rag_engine.py will automatically attach the
    LoRA adapter on top of the base model.
"""

import json

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

import config

TRAIN_FILE = config.FINE_TUNE_DATA_DIR / "train.jsonl"


def format_example(example):
    """Turns an {instruction, input, output} row into the prompt format
    the model will actually see during training and inference."""
    instruction = example["instruction"]
    extra_input = example.get("input", "")
    output = example["output"]

    if extra_input:
        prompt = f"{config.SYSTEM_INSTRUCTIONS}\n\nINSTRUCTION: {instruction}\n\nINPUT: {extra_input}\n\nANSWER:"
    else:
        prompt = f"{config.SYSTEM_INSTRUCTIONS}\n\nINSTRUCTION: {instruction}\n\nANSWER:"

    return {"text": f"{prompt} {output}"}


def main():
    if not TRAIN_FILE.exists():
        print(f"No training file found at {TRAIN_FILE}.")
        print("Create it first — see the docstring at the top of this file for the format.")
        # Write a tiny starter example so the path & format are obvious
        TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TRAIN_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "instruction": "A user says a false case was filed against them by police "
                                "under a bailable offence. Explain their options.",
                "input": "",
                "output": "First confirm whether the offence is bailable and cognizable. "
                          "If bailable, bail can be sought directly at the police station "
                          "or before the Magistrate as a matter of right. You may also file "
                          "a written complaint to the SP/DCP if you believe the case is "
                          "malicious, and consider a quashing petition before the High Court "
                          "under its inherent powers if the FIR discloses no offence at all. "
                          "This is general information only — consult a licensed advocate "
                          "before taking any action."
            }) + "\n")
        print(f"A starter example has been written to {TRAIN_FILE} so you can see the "
              f"expected format. Add more rows, then re-run this script.")
        return

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=str(TRAIN_FILE), split="train")
    dataset = dataset.map(format_example)
    print(f"Loaded {len(dataset)} training example(s).")

    print(f"Loading base model in 4-bit: {config.BASE_LLM_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_LLM_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_LLM_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Mistral attention modules
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(config.LORA_OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=1024,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving LoRA adapter to {config.LORA_OUTPUT_DIR}")
    model.save_pretrained(str(config.LORA_OUTPUT_DIR))
    tokenizer.save_pretrained(str(config.LORA_OUTPUT_DIR))

    print("\nDone. To use this fine-tuned model, set USE_FINE_TUNED_ADAPTER=true "
          "in your .env file and re-run app.py / cli.py.")


if __name__ == "__main__":
    main()
