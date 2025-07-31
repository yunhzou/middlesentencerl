from datasets import load_dataset

ds = load_dataset(
        "DeepStudentLlama/AoPS-Instruct",
        name="2024_not_decontaminated",      # default subset
        split="train")

print(ds[0])

ds = ds.map(lambda x: {"rewritten_context": "Question: "+ x["rewritten_question"] + "\n" + "Answers: " + "\n".join(x["rewritten_answers"])})

# save to disk as AoPS-Instruct-merged_context 
ds.save_to_disk("AoPS-Instruct-merged_context")