import argparse
import json
from datasets import load_dataset
from tqdm import tqdm
from agent import set_models, fix_code, run_in_sandbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="QWEN", help="Model name: QWEN or CLAUDE")
    args = parser.parse_args()

    MODEL = args.model.upper()

    ds = load_dataset("bigcode/humanevalpack", split="test")

    set_models(MODEL)

    results = []
    num_pass = 0

    for example in tqdm(ds):
        buggy_code = example['declaration'] + example["buggy_solution"]
        test_code = example["test"]
        task_id = example["task_id"]
        prompt = example["prompt"]

        fixed_code = fix_code(buggy_code, test_code, prompt)
        stdout, stderr = run_in_sandbox(fixed_code + "\n\n" + test_code)

        passed = not stderr
        num_pass += passed
        results.append({"task_id": task_id, "passed": passed})

    total = len(results)
    pass1 = num_pass / total if total > 0 else 0
    print(f"pass@1 = {pass1:.3f} ({num_pass}/{total})")

    with open(f"{MODEL}_evaluation_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Results saved to {MODEL}_evaluation_results.jsonl")


if __name__ == "__main__":
    main()
