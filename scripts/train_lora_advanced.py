"""Advanced LoRA training pipeline with cleaning + synthetic augmentation."""

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

from src.data.synthetic import generate_dataset
from src.data.cleaning import clean_dataset


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"

    # Load real dataset
    dataset = load_dataset("csv", data_files="data/train.csv")["train"]

    # Clean and convert to list
    cleaned = clean_dataset(dataset)
    prompts = [d["prompt"] for d in cleaned]

    # Generate synthetic data (dry-run friendly when model is None)
    synthetic_data = generate_dataset(None, prompts[:100], variations=1)

    # Merge real + synthetic
    merged = cleaned + synthetic_data
    dataset = Dataset.from_list(merged)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(example):
        answer = example.get("answer", "")
        reasoning = example.get("synthetic_reasoning", "")
        text = example["prompt"] + "\nReasoning: " + reasoning + "\nAnswer: " + answer
        return tokenizer(text, truncation=True, padding="max_length", max_length=1024)

    dataset = dataset.map(preprocess)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./lora-output-advanced",
        per_device_train_batch_size=2,
        num_train_epochs=2,
        logging_steps=10,
        save_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained("lora_adapter")


if __name__ == '__main__':
    main()
