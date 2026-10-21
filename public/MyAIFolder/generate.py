import torch
from transformers import GPT2LMHeadModel

print("Loading your trained 7M model from './my_7m_model'...")
model = GPT2LMHeadModel.from_pretrained("./my_7m_model")
model.eval()  # Set model to evaluation mode


def generate_text(prompt_text, max_length=150):
  # Convert prompt string directly to raw byte/integer tokens (0-255)
  input_ids = torch.tensor(
      [list(prompt_text.encode("utf-8"))], dtype=torch.long
  )

  print(f"\nPrompt: '{prompt_text}'")
  print("Generating response...")

  with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        pad_token_id=0,  # Fallback padding token
    )

  # Convert the output token IDs back into bytes, then decode to string
  generated_bytes = bytes(output_ids[0].tolist())
  decoded_text = generated_bytes.decode("utf-8", errors="ignore")
  return decoded_text


# Test generation with a sample keyword/prompt from your database
prompt = "The"
result = generate_text(prompt, max_length=200)

print("\n--- Generated Output ---")
print(result)
print("------------------------")