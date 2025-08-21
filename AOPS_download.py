from datasets import load_dataset
from pathlib import Path
import logging
import os 
# load env 
from dotenv import load_dotenv

load_dotenv()

storage_path = Path(os.getenv("STORAGE_PATH"))
if not storage_path:
        raise ValueError("STORAGE_PATH environment variable is not set.")
log_path = storage_path / "filtered_out_indices.log"

# Set up logging
logging.basicConfig(filename=log_path, level=logging.INFO)

# Load dataset
ds = load_dataset(
    "DeepStudentLlama/AoPS-Instruct",
    name="2024_not_decontaminated",
    split="train"
)

# Add index to each row
ds = ds.map(lambda x, idx: {"orig_index": idx}, with_indices=True)

def is_valid(x):
    ans = x.get("rewritten_answers", [])
    q = x.get("rewritten_question", "")
    if ans is None or not any(isinstance(a, str) and a.strip() for a in ans):
        return False
    if not isinstance(q, str) or not q.strip():
        return False
    return True

# Save indices of removed entries
invalid_indices = [x["orig_index"] for x in ds if not is_valid(x)]
for idx in invalid_indices:
    logging.info(f"Removed index: {idx}")

# Apply filter
ds = ds.filter(is_valid)

# Merge context
ds = ds.map(lambda x: {
    "rewritten_context": "Question: " + x["rewritten_question"] +
                         "\nAnswers: " + "\n".join(x["rewritten_answers"])
})

# Save cleaned dataset
ds.save_to_disk(storage_path / "AoPS-Instruct-merged_context")
