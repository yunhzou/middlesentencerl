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
    "nvidia/OpenCodeReasoning",
    name="split_0",
    split="split_0"
)

# Save cleaned dataset
ds.save_to_disk(storage_path / "Nvidia_OpenCodeReasoning")
