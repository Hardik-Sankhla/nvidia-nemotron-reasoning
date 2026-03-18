"""LoRA training pipeline (skeleton) using Hugging Face + PEFT.

This script is a starting point. Customize `model_name` and dataset paths
before running on real infrastructure. This is intended for Phase 2 training
where you create a LoRA adapter compatible with vLLM.
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"

    # Expect a CSV with columns: prompt,answer
    dataset = load_dataset("csv", data_files={"train":"data/train.csv"})

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(example):
        text = example["prompt"] + "\nAnswer: " + example["answer"]
        return tokenizer(text, truncation=True, padding="max_length", max_length=1024)

    dataset = dataset["train"].map(preprocess, batched=False)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./lora-output",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

    trainer.train()

    model.save_pretrained("lora_adapter")


if __name__ == '__main__':
    main()
