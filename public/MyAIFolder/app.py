from tkinter import filedialog, messagebox, scrolledtext
import tkinter as tk
import torch
from transformers import GPT2LMHeadModel

# --- Initialize Model ---
print("Loading your trained 7M model from './my_7m_model'...")
try:
  model = GPT2LMHeadModel.from_pretrained("./my_7m_model")
  model.eval()
  model_loaded = True
  print("Model loaded successfully!")
except Exception as e:
  model_loaded = False
  print(f"Error loading model: {e}")

selected_image_path = None


def browse_image():
  global selected_image_path
  file_path = filedialog.askopenfilename(
      title="Select an Image",
      filetypes=[
          ("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"),
          ("All Files", "*.*"),
      ],
  )
  if file_path:
    selected_image_path = file_path
    filename = file_path.split("/")[-1]
    lbl_image_status.config(text=f"Attached: {filename}", fg="#00ffcc")


def generate():
  if not model_loaded:
    messagebox.showerror(
        "Model Error",
        "Could not load './my_7m_model'. Ensure training completed.",
    )
    return

  prompt = entry_prompt.get().strip()
  if not prompt:
    prompt = "The"

  output_box.delete("1.0", tk.END)

  loading_msg = "Generating response..."
  if selected_image_path:
    loading_msg = f"[Image Attached] Generating response..."

  output_box.insert(tk.END, loading_msg + "\n")
  root.update()

  try:
    input_ids = torch.tensor(
        [list(prompt.encode("utf-8"))], dtype=torch.long
    )

    with torch.no_grad():
      output_ids = model.generate(
          input_ids,
          max_length=200,
          do_sample=False,
          pad_token_id=0,
      )

    generated_bytes = bytes(output_ids[0].tolist())
    result = generated_bytes.decode("utf-8", errors="ignore")

    output_box.delete("1.0", tk.END)
    if selected_image_path:
      output_box.insert(
          tk.END, f"--- Image Context: {selected_image_path} ---\n\n"
      )
    output_box.insert(tk.END, result)

  except Exception as e:
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, f"Generation error: {e}")


# --- GUI Layout Setup ---
root = tk.Tk()
root.title("7M Local AI Studio")
root.geometry("750x620")
root.config(bg="#0b0b0f")

# Main Header / Title Section
header_frame = tk.Frame(root, bg="#0b0b0f", pady=20)
header_frame.pack(fill=tk.X)

title_label = tk.Label(
    header_frame,
    text="Describe it. Watch it render.",
    font=("Segoe UI", 22, "bold"),
    fg="white",
    bg="#0b0b0f",
)
title_label.pack(anchor="w", padx=30)

subtitle_label = tk.Label(
    header_frame,
    text=(
        "Local 7M model turns your database prompts into text — powered by your"
        " CPU."
    ),
    font=("Segoe UI", 10),
    fg="#8a8a93",
    bg="#0b0b0f",
)
subtitle_label.pack(anchor="w", padx=30, pady=(5, 0))

# Floating Card Container (Glassmorphism Dark Style)
card_frame = tk.Frame(root, bg="#16161c", bd=1, relief=tk.SOLID)
card_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=(0, 25))

# Top Mode Tabs Frame inside Card
tabs_frame = tk.Frame(card_frame, bg="#16161c", padx=20, pady=15)
tabs_frame.pack(fill=tk.X)

btn_tab_text = tk.Button(
    tabs_frame,
    text="Text & Database",
    font=("Segoe UI", 9, "bold"),
    bg="#7e57c2",
    fg="white",
    relief=tk.FLAT,
    padx=12,
    pady=4,
)
btn_tab_text.pack(side=tk.LEFT, padx=(0, 8))

btn_upload_img = tk.Button(
    tabs_frame,
    text="📁 Insert Image",
    font=("Segoe UI", 9, "bold"),
    bg="#2a2a35",
    fg="#cccccc",
    activebackground="#3a3a45",
    activeforeground="white",
    relief=tk.FLAT,
    cursor="hand2",
    padx=12,
    pady=4,
    command=browse_image,
)
btn_upload_img.pack(side=tk.LEFT)

lbl_image_status = tk.Label(
    tabs_frame,
    text="No image attached",
    font=("Segoe UI", 8),
    fg="#666670",
    bg="#16161c",
)
lbl_image_status.pack(side=tk.LEFT, padx=12)

# Input Bar Frame (Prompt + Generate Button side-by-side)
input_container = tk.Frame(card_frame, bg="#16161c", padx=20, pady=5)
input_container.pack(fill=tk.X)

entry_prompt = tk.Entry(
    input_container,
    font=("Segoe UI", 12),
    bg="#22222d",
    fg="white",
    insertbackground="white",
    relief=tk.FLAT,
)
entry_prompt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, ipady=12, padx=(0, 10))
entry_prompt.insert(0, "The")

btn_generate = tk.Button(
    input_container,
    text="Generate",
    font=("Segoe UI", 10, "bold"),
    bg="#7df9ff",
    fg="#0b0b0f",
    activebackground="#5ce5ed",
    relief=tk.FLAT,
    cursor="hand2",
    padx=20,
    command=generate,
)
btn_generate.pack(side=tk.RIGHT, ipady=8)

# Style / Modifier Filter Pills Bar (Visual Layout match)
pills_frame = tk.Frame(card_frame, bg="#16161c", padx=20, pady=12)
pills_frame.pack(fill=tk.X)

pills = ["No style", "Database Mode", "Greedy", "Raw Bytes"]
for p in pills:
  bg_color = "#22222d" if p != "No style" else "#7df9ff"
  fg_color = "white" if p != "No style" else "#0b0b0f"
  lbl_pill = tk.Label(
      pills_frame,
      text=p,
      font=("Segoe UI", 8, "bold"),
      bg=bg_color,
      fg=fg_color,
      padx=10,
      pady=4,
  )
  lbl_pill.pack(side=tk.LEFT, padx=(0, 6))

# Output Section Frame
output_frame = tk.Frame(card_frame, bg="#16161c", padx=20, pady=10)
output_frame.pack(fill=tk.BOTH, expand=True)

output_box = scrolledtext.ScrolledText(
    output_frame,
    wrap=tk.WORD,
    font=("Consolas", 10),
    bg="#111115",
    fg="#7df9ff",
    insertbackground="white",
    bd=0,
)
output_box.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

# Run application
root.mainloop()