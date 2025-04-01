import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from data_loader import clean_text


df = pd.read_csv("/data/volunteer-descriptions.csv")
df = df.dropna(subset=["Description"]).drop_duplicates()
df["cleaned"] = df["Description"].apply(clean_text)

# Create sentence pairs (adjacent similar pairs)
sentence1 = df["cleaned"][:-1].tolist()
sentence2 = df["cleaned"][1:].tolist()
labels = [1.0] * len(sentence1)

dataset = Dataset.from_dict({
    "sentence1": sentence1,
    "sentence2": sentence2,
    "label": labels
})

# Load model & tokenizer
model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Apply PEFT using LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

model = get_peft_model(base_model, peft_config)

# Tokenize dataset
def tokenize(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./peft_volunteer_model",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_dir="./logs",
    save_strategy="epoch",
    fp16=True  # Only use if you have a GPU with float16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

# Save model
trainer.save_model("./peft_volunteer_model")
print("Model fine-tuned and saved to ./peft_volunteer_model")