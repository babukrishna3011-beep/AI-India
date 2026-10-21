# 7M Local AI Studio & Ephemeral Micro-Model Hub

This is the complete system bundle containing the backend PyTorch server, local database bridge, and web-based 100-parameter ephemeral model visualizer.

---

## 📁 Directory Structure & Functions of Each File

```text
MyAIFolder/
├── public/
│   ├── index.html           <-- Interactive Web Dashboard & 100-parameter JS Neural Net
│   └── model.safetensors    <-- Web downloadable replacement copy of 7M weights
├── app.py                   <-- FastAPI Python server (Inference & Weight Injection Engine)
├── requirements.txt         <-- Python package dependencies
├── wrangler.toml            <-- Cloudflare Pages deployment configuration
├── model.safetensors        <-- 7M model weights file (PyTorch / safetensors format)
├── data.db                  <-- 4MB SQLite context database
└── README.md                <-- Documentation and system guide
```

### Functions of Each Component:

1. **`app.py` (Backend API Engine)**
   - Connects your local 7M `model.safetensors` model weights to the web dashboard.
   - Handles SQLite database queries (`data.db`) for context retrieval.
   - Provides `/api/inject_weights` endpoint to modify 7M model weight matrices using converted numbers from the ephemeral model.

2. **`public/index.html` (Frontend & Visualizer)**
   - **100-Parameter Ephemeral Model**: Pure JavaScript neural network running real backpropagation on single words.
   - **Converted Numbers Exporter**: Generates ready-to-use PyTorch tensor code.
   - **Catastrophic Reset Button**: Intentional weight wiper for clean-state micro-training.
   - **Download Center**: Allows downloading full bundle zip or standalone 7M model weight replacements.

3. **`model.safetensors` (7M Model Weights)**
   - Stores the 7,000,000 parameters of your trained PyTorch language model.
   - If corrupted during weight manipulation experiments, use the **Model Only Download** option in the web UI to replace it without downloading the full setup again.

4. **`data.db` (SQLite Context Database)**
   - Holds structured notes, physics data, and text entries accessed by `app.py` during retrieval-augmented generation.

---

## 🐍 How to Add Converted Numbers Directly into the 7M Model in Python

When you train the 100-parameter toy model on a target word in `index.html`, it converts that word into a sequence of floating-point numbers. Here is how to inject those numbers into your 7M model's Python code:

### Method 1: Injecting into Token Embedding Matrix (`app.py`)

Open `app.py` or your model script and use the following function:

```python
import torch
from safetensors.torch import load_file, save_file

# 1. Paste converted numbers extracted from the 100-parameter model
converted_numbers = torch.tensor([
    -0.123456, 0.456789, 0.890123, -0.345678, 0.112233, ... # 100 values
], dtype=torch.float32)

# 2. Inject into 7M Model Weights
def modify_7m_model_weights(model_weights_path, token_id, converted_tensor):
    # Load 7M model safetensors dict
    weights = load_file(model_weights_path)
    
    # Locate embedding matrix tensor (e.g., 'embed_tokens.weight')
    embedding_matrix = weights["embed_tokens.weight"]
    
    # Modify the specific token row with converted numbers
    embedding_matrix[token_id, :len(converted_tensor)] += converted_tensor * 0.01
    
    # Save modified weights back to file
    save_file(weights, model_weights_path)
    print(f"Successfully modified 7M model embedding for Token #{token_id}!")

# Execute modification
# modify_7m_model_weights("model.safetensors", token_id=101, converted_tensor=converted_numbers)
```

### Method 2: Live API Weight Injection
With `python app.py` running, click the **Export Converted Numbers** button in `index.html`. The app automatically calls `/api/inject_weights` to modify the 7M model's active memory in real-time.

---

## 🔄 Replacing a Corrupted 7M Model File

If experimental weight modifications corrupt your `.safetensors` file:
1. Do **not** re-download the whole bundle.
2. Click **"Download 7M Model Weights Replacement"** in the Web UI dashboard (or download `model.safetensors` individually).
3. Overwrite the corrupted `model.safetensors` file inside your `MyAIFolder`.
4. Restart `app.py`.
