from datasets import load_dataset
import json
from agent import set_models, fix_code, run_in_sandbox
from tqdm import tqdm 


ds = load_dataset("bigcode/humanevalpack", split="test")

MODEL = "QWEN"

set_models(MODEL)

results = []
for example in tqdm(ds):

    buggy_code = example['declaration'] + example["buggy_solution"]
    test_code = example["test"]
    task_id = example["task_id"]
    prompt = example["prompt"]

    fixed_code = fix_code(buggy_code, test_code, prompt) 

    stdout, stderr = run_in_sandbox(fixed_code + "\n\n" + test_code)

    passed = not stderr

    results.append({"task_id": task_id, "passed": passed})


num_pass = sum(1 for r in results if r["passed"])
total = len(results)
pass1 = num_pass / total
print(f"pass@1 = {pass1:.3f} ({num_pass}/{total})")


with open(f"{MODEL}_evaluation_results.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
