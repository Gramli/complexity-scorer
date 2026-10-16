# Complexity Scorer

A Python training project that uses **Tinker** to fine-tune **Qwen3-8B** to estimate how difficult a user request is for a language model to answer correctly. The experiment compares the original model with a fine-tuned version on the same evaluation dataset.

The model reads a prompt and returns only JSON, for example `{"complexity": 0.50}`. It scores the reasoning and understanding needed, rather than just the prompt's length:

| Score | Meaning |
| --- | --- |
| 0.00 | Trivial: extracting a value or reformatting text |
| 0.25 | Easy: a few straightforward steps |
| 0.50 | Moderate: combining information or following several constraints |
| 0.75 | Hard: substantial reasoning, expertise, or tradeoffs |
| 1.00 | Very hard: deep reasoning across complex systems or uncertainty |

## How the experiment works

1. **Measure the baseline:** ask the original model to score 50 labeled prompts.
2. **Prepare training data:** validate 2,500 examples (500 per score) and convert them into chat conversations. Duplicate prompts and overlap with evaluation prompts are rejected.
3. **Fine-tune:** use Tinker Cookbook to train a LoRA adapter, a small set of model adjustments, for one pass through the training data.
4. **Compare:** evaluate the saved model checkpoint on the same 50 prompts, keeping the scoring instructions and generation settings unchanged.

The saved results show accuracy improving from **50% to 76%**, with mean absolute error falling from **0.125 to 0.06**. Accuracy means matching the expected score exactly; mean absolute error measures how far predictions are from the expected scores (lower is better). These results describe this small benchmark.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/baseline.py` | Evaluate either the original model or a trained checkpoint |
| `src/train.py` | Validate data, generate conversations, and start training |
| `src/scorer_config.py` | Model name, scoring instructions, and allowed scores |
| `src/scorer_schema.py`, `src/json_utils.py` | Shared message building, score validation, and JSONL reading |
| `src/verify_tinker.py` | Make a small inference request to check the Tinker connection |
| `data/` | Labeled prompts and generated training conversations; JSONL stores one JSON record per line |
| `results/` | Individual predictions and summary metrics from saved evaluations |
| `runs/` | Training configurations, logs, metrics, and remote checkpoint references |
| `tests/` | Offline checks for parsing, data validation, and shared helpers |

Training currently reads `data/train_2500.jsonl` and writes `data/train_2500_tinker.jsonl`. Evaluation uses `data/eval.jsonl`; keep those examples out of training.

## Run locally

Run these commands from the repository root in PowerShell. The project's documented training environment uses Python 3.13, Tinker 0.33.1, and Tinker Cookbook 0.5.7. You need a Tinker account with API access for evaluation and training.

```powershell
py -3.13 -m venv .venv-training
.\.venv-training\Scripts\Activate.ps1
python -m pip install "tinker-cookbook==0.5.7" "tinker==0.33.1"
tinker auth login

# Validate and convert the dataset locally, without calling Tinker.
python src/train.py --prepare-only

# Try five evaluation examples, then run the full baseline.
python src/baseline.py --limit 5
python src/baseline.py

# Train and copy the printed final sampler checkpoint path.
python src/train.py
python src/baseline.py --checkpoint "tinker://YOUR-RUN-ID/sampler_weights/final"
```

Evaluation and training use paid Tinker API calls. Evaluation replaces its corresponding result files. Training refuses to reuse a nonempty `runs/qwen3_8b_train_2500_epoch1/` directory; move the existing run to another directory under `runs/` before starting a fresh experiment. The trained weights stay in Tinker; the printed sampler checkpoint path lets the evaluator load them.

See [the detailed setup and training guide](docs/train_smoke.md) for environment troubleshooting and training settings. That guide still refers to `src/train_smoke.py`; the current script is `src/train.py`.
