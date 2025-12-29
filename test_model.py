import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import get_json_schema

# --- Tool Definitions (Same as training for consistency) ---
def search_knowledge_base(query: str) -> str:
    """
    Search internal company documents, policies and project data.

    Args:
        query: query string
    """
    return "Internal Result"

def search_google(query: str) -> str:
    """
    Search public information.

    Args:
        query: query string
    """
    return "Public Result"

TOOLS = [get_json_schema(search_knowledge_base), get_json_schema(search_google)]

# Load the fine-tuned model
# Note: Ensure this matches the output_dir in train.py
checkpoint_dir = "functiongemma-270m-it-simple-tool-calling" 

print(f"Loading model from {checkpoint_dir}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        device_map="auto",
        torch_dtype="auto"
    )
except OSError:
    print(f"Error: Could not find model in {checkpoint_dir}. Make sure you have run train.py first.")
    exit(1)

# Test input
user_query = "What is the reimbursement limit for travel meals?"
print(f"User Query: {user_query}")

# Prepare the prompt
messages = [
    {"role": "user", "content": user_query}
]

# Apply template
inputs = tokenizer.apply_chat_template(
    messages,
    tools=TOOLS,
    return_tensors="pt",
    add_generation_prompt=True
).to(model.device)

# Generate response
print("Generating response...")
outputs = model.generate(inputs, max_new_tokens=128)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- Model Output ---")
print(result)
print("--------------------")
