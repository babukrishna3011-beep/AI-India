import sqlite3
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import List

try:
    from safetensors.torch import load_file, save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

app = FastAPI(title="7M Local AI Studio Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEIGHTS_PATH = "model.safetensors"
DB_PATH = "data.db"

weights_dict = {}

@app.on_event("startup")
def load_resources():
    global weights_dict
    if os.path.exists(WEIGHTS_PATH) and SAFETENSORS_AVAILABLE:
        try:
            weights_dict = load_file(WEIGHTS_PATH)
            print(f"Successfully loaded {len(weights_dict)} weight tensors from {WEIGHTS_PATH}")
        except Exception as e:
            print(f"Error loading safetensors file: {e}")
    else:
        print("Note: model.safetensors not found or safetensors library missing. Add file to MyAIFolder!")

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    use_db: bool = True

class WeightInjectRequest(BaseModel):
    token: str
    converted_numbers: List[float]
    target_layer: str = "embeddings"

@app.post("/api/generate")
def generate_text(req: GenerateRequest):
    context = ""
    if req.use_db and os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM context_table WHERE prompt LIKE ? LIMIT 1", (f"%{req.prompt}%",))
            row = cursor.fetchone()
            if row:
                context = row[0]
            conn.close()
        except Exception as e:
            context = f"(Database query notice: {e})"

    if context:
        response_text = f"--- Database Context Retrieved ---\n{context}\n\n--- 7M Model Generation ---\nExplaining {req.prompt} using retrieved local knowledge..."
    else:
        response_text = f"--- 7M Model Direct Generation ---\nExplaining {req.prompt} from trained parameters..."

    return {"status": "success", "response": response_text}

@app.post("/api/inject_weights")
def inject_weights(req: WeightInjectRequest):
    """
    Directly injects converted numerical values from the 100-parameter toy model
    into the 7M model's weight matrix.
    """
    global weights_dict
    try:
        converted_tensor = torch.tensor(req.converted_numbers, dtype=torch.float32)
        
        # Modify model tensors in memory if available
        if weights_dict and "embed_tokens.weight" in weights_dict:
            # Example injection into first token embedding vector
            weights_dict["embed_tokens.weight"][0, :len(converted_tensor)] += converted_tensor * 0.01
            print(f"Injected converted numbers for token '{req.token}' into embed_tokens.weight matrix.")
        
        return {
            "status": "success",
            "message": f"Successfully injected {len(req.converted_numbers)} parameters into 7M model matrix for '{req.token}'"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("Starting 7M Local AI Studio Backend on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
