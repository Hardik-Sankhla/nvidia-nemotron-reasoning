"""LoRA training pipeline driven by YAML config."""

import argparse

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

from src.utils.config import load_yaml


def run(config):
    model_name = config["model"]["name"]
    train_csv = config["data"]["train_csv"]

    dataset = load_dataset("csv", data_files={"train": train_csv})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = int(config["training"].get("max_length", 1024))

    def preprocess(example):
        text = example["prompt"] + "\nAnswer: " + example.get("answer", "")
        return tokenizer(text, truncation=True, padding="max_length", max_length=max_length)

    dataset = dataset["train"].map(preprocess, batched=False)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    lora = config["lora"]
    lora_config = LoraConfig(
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        target_modules=list(lora["target_modules"]),
        lora_dropout=float(lora["dropout"]),
    )

    model = get_peft_model(model, lora_config)

    training = config["training"]
    training_args = TrainingArguments(
        output_dir=training["output_dir"],
        per_device_train_batch_size=int(training["batch_size"]),
        num_train_epochs=float(training["num_train_epochs"]),
        logging_steps=int(training["logging_steps"]),
        save_steps=int(training["save_steps"]),
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

    adapter_dir = config["output"].get("adapter_dir", "lora_adapter")
    model.save_pretrained(adapter_dir)
    print(f"Saved adapter to {adapter_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter with YAML config")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_yaml(args.config)
    run(config)


if __name__ == '__main__':
    main()
