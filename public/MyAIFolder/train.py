import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)

print("Initializing 7M parameter model configuration...")

# 7.3M Parameter Configuration for character-level vocabulary
config = GPT2Config(
    vocab_size=256,  # 0-255 bytes
    n_positions=512,
    n_embd=272,
    n_layer=8,
    n_head=8,
    bos_token_id=None,
    eos_token_id=None,
)

model = GPT2LMHeadModel(config)
n_params = sum(p.numel() for p in model.parameters())
print(f"Total Model Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")


class TextDataset(Dataset):

  def __init__(self, file_path, block_size=128):
    with open(file_path, "r", encoding="utf-8") as f:
      self.text = f.read()

    byte_array = np.frombuffer(self.text.encode("utf-8"), dtype=np.uint8).copy()
    self.data = torch.from_numpy(byte_array).long()
    self.block_size = block_size

  def __len__(self):
    return max(0, len(self.data) - self.block_size)

  def __getitem__(self, idx):
    chunk = self.data[idx : idx + self.block_size + 1]
    return {"input_ids": chunk}


print("Loading training dataset...")
dataset = TextDataset("training_data.txt", block_size=128)


def custom_collate_fn(batch):
  input_ids = torch.stack([item["input_ids"] for item in batch])
  return {"input_ids": input_ids, "labels": input_ids.clone()}


# Optimized TrainingArguments with a fast step limit for local laptop training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    max_steps=7000,  # Forces training to stop after exactly 1,000 steps (~10 minutes)
    per_device_train_batch_size=4,  # Larger batch size speeds up CPU processing safely
    save_steps=200,
    save_total_limit=2,
    logging_steps=10,
    use_cpu=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=custom_collate_fn,
    train_dataset=dataset,
)

print("Starting fast training on your laptop...")
trainer.train()

model.save_pretrained("./my_7m_model")
print("Training complete! Model saved to './my_7m_model'.")