import os
import json
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, GPT2TokenizerFast
from datasets import Dataset

script_directory = Path(__file__).parent
data_directory = script_directory / "data"
web_qsp__directory = data_directory / f"benchmarks/WebQSP"
artifacts_finetune_directory = data_directory / f"artifacts/WebQSPFineTuneGPT2"

training_strings = []
with open(web_qsp__directory / "webqsp.examples.train.json", "r") as f:
    data = json.load(f)
    for datapoint in data:
        for answer_str in datapoint['answers_str']:
            if answer_str:
                utterance = datapoint['utterance']
                sentence = f"Question: {utterance}\nAnswer: {answer_str}"
                training_strings.append(sentence)
                break

# Load the GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

dataset = Dataset.from_dict({'text': training_strings})
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=str(artifacts_finetune_directory),
    evaluation_strategy="no",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=4,
    per_device_train_batch_size=4,
    save_strategy="epoch",  # Save a checkpoint at the end of each epoch
    save_total_limit=1,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=None,
)

# Start the training
trainer.train()

model.save_pretrained(str(artifacts_finetune_directory))
tokenizer.save_pretrained(str(artifacts_finetune_directory))

