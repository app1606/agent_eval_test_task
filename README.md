# "Evaluation of LLM-Based Agentic Systems for Software Development Tasks" Test Task solution

The solution consists of two main scripts located in the `scripts` folder. `agent.py` contains my implementation of the AI-agent that fixes buggy code. For this task I went with two different models: 'Qwen3-0.6B' from Qwen and 'claude-sonnet-4-20250514' from Anthropic. First one runs locally, the second one sends requests to the server. 

Our agents are made in ReAct style, meaning they have 'reason -> act -> observe' scheme, at each observation step model decides whether it should continue or stop. To stop model from infinite loops we set `MAX_ATTEMPTS` argument, it limits the amount of requests sent to the model (each reason step makes one). Act step runs the code in a sandbox.

It's **important** to mention that this code was run on M3 Pro processor. I've added the support of cuda, but did not test it on CUDA-device. 

## Environment

- **OS:** macOS 14.6 (Apple M3 Pro)  
- **Python:** 3.13  
- **Libraries:** `torch`, `mlx_lm`, `transformers`, `langgraph`, `anthropic`, `datasets`, `tqdm`  
- **Execution:**  
  - Local inference on **Apple Silicon GPU via MPS** (Metal Performance Shaders)  
  - Remote inference via **Anthropic API** for Claude models  

## Settings
To set up Anthropic Model you should run the following command in bash:

```
export ANTHROPIC_API_KEY='YOUR_KEY_HERE'
```

## Usage

Example usage:

```
python3 ./scripts/evaluation.py --model=CLAUDE --max_attempts=2
```

Here `--model` argument may take two values: "CLAUDE" or "QWEN", it allows user to select respective model. `--max_attempts` argument is responsible for the number of requests to the model per try. 

## Dataset

The Python subset of [HumanEvalFix](https://huggingface.co/datasets/bigcode/humanevalpack) dataset was used to evaluate the models. 

## Metric and evaluation

To evaluate model's performance pass@1 metric was introduced. It measures the fraction of problems correctly solved by the agent’s first (and possibly iterative) attempt per task, following the original HumanEval definition. I've ran the whole evaluation pipeline three times: for Qwen model with max_attempts=5 and for Anthropic model with max_attempts=3 and 5. JSON files in `results` folder contain mapping 'test_id — result(Boolean)'. 

The maximal number of tokens for models differ, for Qwen it is set to 200, for Claude to 500. I did it because Qwen model tends to repeat the same sentence on the reasoning stage. Example output:

```
[Reason]  - The code is not correct because it doesn't handle the case where the list is empty.
         - The code is not correct because it doesn't handle the case where the list has only one element.
         - The code is not correct because it doesn't handle the case where the list has multiple elements but they are not close to each other.
         - The code is not correct because it doesn't handle the case where the list has elements that are not in order.
         - The code is not correct because it doesn't handle the case where the list has elements that are not in order.
         - The code is not correct because it doesn't handle the case where the list has elements that are not in order.
         - The code is not correct because it doesn't handle the case where the list has elements that are not in order.
         - The code is not correct because it doesn't handle the case where the list has elements that are not in order.
```

## Results 

| Model                  | MAX_ATTEMPTS | pass@1 |
|------------------------|---------------|--------|
| Qwen3-0.6B             | 5             | 0.030  |
| claude-sonnet-4-20250514 | 3           | 0.969  |
| claude-sonnet-4-20250514 | 5           | 1.000  |

## Conclusions

It is obvious that Qwen model is too small to solve this task, solving only 5 out of 164 examples. Claude model is, on the other hand, great for this task.  Example reasoning from Claude model: 

```
[Reason]  The bug is in the distance calculation. The code uses 'elem - elem2' which can be negative, but we need the absolute distance between two numbers. When comparing with threshold, we should use abs(elem - elem2) to get the actual distance regardless of which number is larger.
```

It gives precise reasoning and manages to solve most of the tasks in just 3 requests to the model, in most of the cases it's just one. 5 attempts limit is enough for it to solve all the tasks.

I saw poor performance from Qwen and decided to change my prompt, I've added 'prompt' field values to the prompt along with the buggy solution and tests, but it did not improve the quality of the solution, leaving it at around 3% success rate.  

