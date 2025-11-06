import os
import re
import json
import tempfile
import textwrap
import subprocess
from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from anthropic import Anthropic
from mlx_lm import load, generate

MODEL_NAME = "Qwen/Qwen3-0.6B"

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def load_model_tokenizer():
    print(f"Loading model: {MODEL_NAME} ...")
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if torch.backends.mps.is_available(): #MacOS
        model, tokenizer = load(MODEL_NAME)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map=None
        )
        model.to(device)

    return model, tokenizer

def generate_with_qwen(prompt: str, max_tokens: int = 200) -> str:
    if torch.backends.mps.is_available(): #MacOS
        out = generate(QWEN_MODEL, QWEN_TOKENIZER, prompt, max_tokens=max_tokens)
        return out.strip()
    else:
        messages = [{"role": "user", "content": prompt}]
        text = QWEN_TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = QWEN_TOKENIZER([text], return_tensors="pt").to(QWEN_MODEL.device)
        outputs = QWEN_MODEL.generate(**inputs, max_new_tokens=max_tokens)
        generated = QWEN_TOKENIZER.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return generated.strip()


def run_in_sandbox(code: str, timeout: int = 3):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(textwrap.dedent(code))
        f.flush()
        filename = f.name
    try:
        result = subprocess.run(
            ["python3", filename],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "Timeout"
    finally:
        try:
            os.remove(filename)
        except OSError:
            pass

class AgentState(TypedDict, total=False):
    buggy_code: str
    test_code: str
    prompt: str
    fixed_code: Optional[str]
    thought: Optional[str]
    stdout: Optional[str]
    stderr: Optional[str]
    history: List[Dict[str, Any]]
    stop: bool
    attempts: int

def initial_state(buggy_code: str, test_code: str, prompt: str) -> AgentState:
    return AgentState(
        buggy_code=buggy_code,
        test_code=test_code,
        prompt=prompt, 
        fixed_code=None,
        thought="",
        stdout="",
        stderr="",
        history=[],
        stop=False,
        attempts=0
    )

def reason(state: AgentState) -> AgentState:
    history_text = ""
    if state["history"]:
        history_text = "\n".join(
            [f"Thought: {h.get('thought','')}\nError: {h.get('stderr','')}" for h in state["history"]]
        )

    prompt = (
        f'''
        You are a Python code-fixing assistant. 
        Given a task prompt, a buggy function and its test code, propose a corrected version that passes all tests. 
        Explain your reasoning briefly.


        Return your answer strictly as JSON with fields:
        {{
          "thought": "...",
          "fixed_code": "..." 
        }}

        Task prompt:
        ---BEGIN TASK PROMPT---
        {state['prompt']}
        ---END TASK PROMPT---

        Buggy code:
        ---BEGIN BUGGY CODE---
        {state['buggy_code']}
        ---END BUGGY CODE---

        Tests:
        ---BEGIN TESTS---
        {state['test_code']}
        ---END TESTS---

        Previous attempts and errors:
        {history_text}
        '''
    )

    if MODEL_TYPE == "QWEN":
        text = generate_with_qwen(prompt)
    elif MODEL_TYPE == "CLAUDE":
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        text = ""
        try:
            text = response.content[0].text.strip()
        except Exception:
            text = ""

    
    json_str_match = re.search(r"\{.*\}", text, re.S)
    data = None
    if json_str_match:
        try:
            data = json.loads(json_str_match.group(0))
        except Exception:
            data = None

    if not data: # no fixes 
        data = {"thought": text or "No JSON response from model.", "fixed_code": state.get("buggy_code", "")}
        
    
    state["thought"] = data.get("thought", "")
    state["fixed_code"] = data.get("fixed_code", state.get("buggy_code", ""))
    print(state["fixed_code"])

    print(f"\n[Reason]  {state["thought"]}")
    
    return state

def act(state: AgentState) -> AgentState:
    fixed_code = state["fixed_code"]
    combined_code = fixed_code + "\n\n" + state["test_code"]
    stdout, stderr = run_in_sandbox(combined_code)
    state["stdout"], state["stderr"] = stdout, stderr

    return state

def observe(state: AgentState) -> AgentState:

    stdout = state["stdout"]
    stderr = state["stderr"]
    thought = state["thought"]
    attempts = state["attempts"] + 1
    state["attempts"] = attempts

    fixed_code = state["fixed_code"]
    history = state["history"]

    print(f"\n[Observe] Attempt {attempts}")

    history.append({
        "thought": thought,
        "fixed_code": fixed_code,
        "stdout": stdout,
        "stderr": stderr,
    })
    state["history"] = history

    if not stderr:
        print("Tests passed!")
        state["stop"] = True
        state["observation"] = "All tests passed successfully."
    else:
        if attempts >= MAX_ATTEMPTS:
            print(f"[Observe] Reached max attempts ({MAX_ATTEMPTS}). Stopping.")
            state["stop"] = True
            return state
        
        print("Tests failed, will retry...")
        print(stderr)
        state["stop"] = False
        state["observation"] = "Tests failed, need another fix."

    return state

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("reason", reason)
    graph.add_node("act", act)
    graph.add_node("observe", observe)

    graph.set_entry_point("reason")
    def continue_or_stop(state: AgentState):
        return END if state.get("stop") else "reason"

    graph.add_edge("reason", "act")
    graph.add_edge("act", "observe")
    graph.add_conditional_edges("observe", continue_or_stop)

    return graph.compile()

def fix_code(buggy_code, test_code, prompt):
    agent = build_graph()
    state = initial_state(buggy_code, test_code, prompt)
    result = agent.invoke(state)

    return result.get("fixed_code")

def set_models(model_type, max_attempts):
    global MODEL_TYPE, QWEN_MODEL, QWEN_TOKENIZER, MAX_ATTEMPTS
    MODEL_TYPE = model_type
    MAX_ATTEMPTS = max_attempts

    if MODEL_TYPE == "QWEN":
        QWEN_MODEL, QWEN_TOKENIZER = load_model_tokenizer()
