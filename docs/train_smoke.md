# Complexity scorer 2,500-example fine-tune

Run commands from the repository root in PowerShell. No training has been started
during implementation.

## Files

- `src/scorer_config.py` contains the shared model name, unchanged baseline
  system prompt, and five allowed scores.
- `src/json_utils.py` reads JSONL files and rejects duplicate JSON properties.
  Each script still applies its own rules to the parsed records.
- `src/scorer_schema.py` validates scores and builds the shared system/user
  messages, so evaluation and training use the same input format.
- `src/train_smoke.py` validates the source dataset, generates chat conversations,
  and calls the standard Tinker Cookbook supervised training pipeline.
- `data/train_tinker.jsonl` is generated from `data/train.jsonl` after validation.
  Every conversation has system, user, and assistant messages. The assistant
  target is only JSON, such as `{"complexity":0.00}`. The `reasoning` field is ignored.
- `tests/test_train_smoke.py` checks validation and conversion offline, using
  temporary synthetic fixtures rather than either project dataset.
- `tests/test_data_contracts.py` checks prediction parsing and the differences
  between training and evaluation dataset rules.
- `tests/test_shared_helpers.py` checks file cleanup and independent message lists.

### Reading the Python code as a .NET developer

The shared files are Python modules: they group related functions, much like
static helper classes in C#. No instances or inheritance are needed here.

- `with` closes resources when a block exits, like C# `using`.
- `yield` returns one item at a time, like C# `yield return`. The JSONL reader
  only parses the next line when requested. `closing()` ensures an early stop
  also closes its file, including evaluation with `--limit`.
- `dict` is a dictionary. Chat messages use string keys such as `role` and `content`.
- `None` represents no value. A return type of `float | None` means a score or
  no valid prediction; the type annotation itself does not validate the input.
- `is_valid_complexity()` performs the actual score validation. Python treats
  booleans as integers, so the function explicitly rejects them.

Data flows from the shared JSONL reader into each script's validation, then into
evaluation or training. File-format parsing is shared; training-only checks such
as class balance and held-out separation remain in `train_smoke.py`.

The source training dataset must contain exactly 2,500 nonempty prompts, 500 for each
allowed score. Labels must be numbers; optional `level` values must match them.
Duplicate prompts and prompts shared with `data/eval.jsonl` cause an abort before
conversion or training. The comparison ignores leading/trailing whitespace;
the original prompt text is preserved in the generated conversations.

## Environment prerequisites

`data/train.jsonl` contains the 2,500 training examples used by this script: the
existing 500 examples plus 2,000 additions. The 500-example files remain available
as `data/train_500.jsonl` and `data/train_500_tinker.jsonl`. The original 50-example
files are preserved as `data/train_50_original.jsonl` and
`data/train_50_original_tinker.jsonl`. Keep `data/eval.jsonl` held out.

Install the stable Cookbook package recommended by the
[official installation guide](https://tinker-docs.thinkingmachines.ai/cookbook/quickstart/#installation),
using the project's Python and keeping Tinker at 0.33.1:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade "tinker-cookbook>=0.1" "tinker==0.33.1"
```

`--upgrade` asks pip to update an existing installation. The minimum version
excludes the empty `tinker-cookbook` 0.0.0 package, so incompatible dependencies
produce an installation error rather than permitting that package.

Before starting training, verify that the supervised training module imports:

```powershell
.\.venv\Scripts\python.exe -c "from tinker_cookbook.supervised import train; print('Cookbook ready')"
```

Proceed to training when this prints `Cookbook ready`. If pip still installs
0.0.0 or the import fails, the training environment is not ready. If installation
reports that no compatible PyTorch distribution is available for Python 3.15,
use a Python environment supported by the package dependencies and repeat the
installation and import check.

### Python 3.15 compatibility

The environment was checked after the import failure: it still contained
Cookbook 0.0.0, with no `supervised` module or declared dependencies. A dry run
with stable Cookbook 0.5.7 failed because no compatible `torch>=2.10` package
was available for Python 3.15. The package index offered compatible PyTorch
releases for Python 3.13.

Use a separate Python 3.13 environment for training:

```powershell
py install 3.13
py -3.13 -m venv .venv-training
.\.venv-training\Scripts\python.exe -m pip install --upgrade pip
.\.venv-training\Scripts\python.exe -m pip install "tinker-cookbook==0.5.7" "tinker==0.33.1"
.\.venv-training\Scripts\python.exe -c "from tinker_cookbook.supervised import train; print('Cookbook ready')"
```

These commands use the Windows Python installation manager and keep the existing
`.venv` available. Once the import check succeeds, use
`.\.venv-training\Scripts\python.exe` for the preparation, training, evaluation,
and test commands below in place of `.\.venv\Scripts\python.exe`.

No WandB extras are required. Use the existing Tinker authentication. The
`--prepare-only` command below uses only the standard library and can run before
the Cookbook installation is resolved.

## Prepare without training

```powershell
.\.venv\Scripts\python.exe src\train_smoke.py --prepare-only
```

This checks both input files, writes `data/train_tinker.jsonl`, and prints the
training configuration. It makes no Tinker service calls and does not load a
tokenizer.

## Start training

```powershell
.\.venv\Scripts\python.exe src\train_smoke.py
```

This always repeats validation and conversation generation before training:

| Setting | Value |
| --- | --- |
| Base model / tokenizer | `Qwen/Qwen3-8B` |
| Renderer | `qwen3` |
| Batch size | 16 |
| Maximum sequence length | 4096 |
| Learning rate | `1e-4` (Cookbook's default linear schedule) |
| LoRA rank | 32 |
| Epochs | 1 |
| Internal test split | None (`test_size=0`) |
| Periodic evaluation / checkpoints | Disabled |
| Loss targets | Last assistant message only |

The standard Cookbook dataset uses complete batches: with 2,500 records and batch
size 16, there are 156 batches per epoch, for 156 optimizer steps. It reshuffles each
epoch and omits the last four records in that epoch.

All training logs go to `runs/qwen3_8b_train_2500_epoch1/`. Cookbook saves its final training
state and sampler checkpoint automatically, even with periodic saves disabled.
After completion the script prints the log directory and reads the final sampler
path using the public `checkpoint_utils.get_last_checkpoint` helper. Checkpoints
are stored remotely in Tinker; their paths are recorded in the local run logs.

A nonempty run directory causes an abort to avoid silently resuming a previous
run. To repeat this experiment from the base model, first move the old run into
another directory under `runs/`.

On Windows, Cookbook receives project-relative dataset and log paths because
its `blobfile` reader interprets a drive prefix such as `D:` as a URL scheme.
The script temporarily runs Cookbook from the project root, so these paths also
work when the script is launched from another directory.

## Evaluate the trained checkpoint

The evaluator selects the trained weights only when you pass `--checkpoint`.
After training, copy the path printed beside `Final sampler checkpoint`:

```powershell
.\.venv\Scripts\python.exe src\baseline.py --checkpoint "tinker://YOUR-RUN-ID/sampler_weights/final"
```

Replace the example path with the exact path printed by your training run. Use
the sampler checkpoint, rather than the checkpoint saved for resuming training.
This evaluates the same `data/eval.jsonl` benchmark, system prompt, and sampling
settings, and saves results to `results/finetuned_qwen3_8b.jsonl` and
`results/finetuned_qwen3_8b_summary.json`. The summary records the checkpoint used.

Without `--checkpoint`, the command still evaluates the untouched base model and
writes the original `results/baseline_qwen3_8b*` files:

```powershell
.\.venv\Scripts\python.exe src\baseline.py
```

## Offline verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

API references: [conversation file builder](https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/supervised/fromconversationfilebuilder/),
[supervised Config](https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/supervised/config/),
[checkpoint lookup](https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/checkpoint_utils/get_last_checkpoint/),
and [Cookbook installation](https://tinker-docs.thinkingmachines.ai/cookbook/quickstart/).
