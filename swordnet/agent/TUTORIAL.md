# Agents: Qwen3 plans and deploys, gpt-oss writes the analysis

Two LLMs, both running on Trainium, both OpenAI-compatible:

| Agent | Where | Jobs |
|---|---|---|
| **Qwen3-8B** | our seat-145 chip, via the starter kit's `serve.sh` (vLLM) | plan each training round ([`plan.py`](plan.py)); choose the run to deploy ([`promote.py`](promote.py)) |
| **gpt-oss-20b** | the organisers' shared endpoint (`GPTOSS_BASE_URL` in `.env`) | write the analysis section of every run report ([`report.py`](report.py)) |

## The loop (straight from the starter kit's lesson)

The model proposes, a **checker** grades it, and the *reason* for a rejection goes back into the next
prompt. The checkers are plain Python (`plan.validate`, `promote.validate`) and their messages name
the fix: *"seat-147 runs yolo11m-obb, not 'yolo11x-obb'"*, *"batch=64 is too large for 40 training
images; use at most 20"*. After 3 rejected tries, or if the model server is down, a deterministic
fallback takes over, so training never stalls waiting on an agent.

## Reports

`data/runs/run-N/report.md`, also shown in the web UI's Reports tab:
- **Config**, **Results**, **Findings**, **Dataset**, **Compared with earlier runs**: generated directly
  from the metrics, so they are always correct;
- **Agent analysis**: gpt-oss-20b is given the findings and earlier runs and asked what to change
  next, what the humans may have missed (label quality, leakage, class balance, image size vs thin
  blades, speed vs accuracy), and whether more data or more epochs is the better investment.

Without `GPTOSS_BASE_URL`, Qwen writes the analysis instead (the section heading says which model).
Regenerate a report with `bridge report N`.

## Starting / checking Qwen

```bash
.venv/bin/python -m swordnet.bridge.cli status              # "Qwen3 ready"?
kubectl exec -it seat-145 -- bash -c 'cd /workspace && ./serve.sh --logs'
```
