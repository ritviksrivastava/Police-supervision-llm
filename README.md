# police-supervision-llm

Fine-tuning a small language model to draft the analytical part of police **supervision notes**
(पर्यवेक्षण टिप्पणी) in Hindi, and — as a second project — to draft the **Report-2** (प्रतिवेदन-02) that a
senior office issues on the basis of a supervision note.

This repository contains code and notes only. **No case documents, datasets or model weights are included**, and
they should not be added: the source documents are real case records that name complainants, witnesses and accused.

| Step | Script | What it does |
|---|---|---|
| 1 | `01_convert_doc.ps1` | Legacy `.doc`/`.rtf` → `.docx` using Microsoft Word |
| 2 | `02_extract.py` (+ `krutidev.py`) | `.docx` → Unicode Hindi → paragraphs labelled by section → `records.jsonl` |
| 3 | `03_build_dataset.py` | Supervision-note training set (case facts → conclusion, lapses, directions) |
| 4 | `04_build_report2_pairs.py` | Supervision note → Report-2 training set (matched case by case) |
| 5 | `05_train.py` | LoRA fine-tuning |
| 6 | `06_evaluate.py` | Generation on the test set, metrics, side-by-side HTML report |
| 7 | `07_webui.py` | Drafting console: operators type the case facts, the model drafts the conclusion, lapses and directions |
| 8 | `08_merge_adapter.py` | Fold the LoRA adapter into a full checkpoint (for vLLM serving) |

`schema.json` describes the structured record the notes are parsed into.

---

## 1. The source documents

Two collections of Word files, expected in this layout next to the repository folder (see *Folder layout* below):

- **Supervision notes** – 7,062 Word files (5,222 `.docx`/`.docm`, 1,840 legacy `.doc`). About 5,000 are supervision
  notes; the rest are progress / final reports that follow a looser template.
- **Report-2 files** – 5,171 Word files; each says that the supervising officer's note "is attached and is to be
  treated as Report-2", then gives a summary, findings and directions.
- 1,312 scanned PDFs exist as well. They have no text layer and are **not used** (they would need Hindi OCR).

A supervision note follows a fairly fixed template:

| Section | Present in | Role in training |
|---|---|---|
| Header: office, case number/date, sections of law, place, dates, investigating officer | ~100% | input |
| प्राथमिकी (FIR) | 92% | input |
| घटनास्थल (scene) | 92% | input |
| अभियोजन साक्ष्य (complainant's statement) | 89% | input |
| साक्षी (witness statements) | 63% | input |
| अनुसंधान में प्रगति (progress of investigation) | 52% | input |
| गिरफ्तारी, समर्पण, जप्ती, प्रतिरक्षा साक्ष्य | 37–86% (usually "शून्य") | input |
| **निष्कर्ष** (conclusion) | 91% | **output** |
| **त्रुटि** (lapses) | 87% | **output** |
| **निर्देश** (directions to the investigating officer) | 86% | **output** |
| अभियुक्त विवरणी and signature block | 78% | dropped |

## 2. Data preparation

### 2.1 Legacy `.doc` → `.docx`

`01_convert_doc.ps1` drives Word through COM. It is resumable, copies files that are already `.docx` under a `.doc`
name, and can be run as several parallel workers (`-Shard i -Shards N`); six workers converted 1,840 files in about
40 minutes with no failures.

### 2.2 Legacy font → Unicode

About 89% of the files are typed in **Kruti Dev 010**, a pre-Unicode font: the stored characters are Latin
(`izkFkfedh`) and only *look* like Hindi (`प्राथमिकी`) in that font. Training on the raw text would teach the model
gibberish, so `krutidev.py` converts it:

- a glyph-to-Unicode table, extended with variants found in this corpus (curly quotes typed for श्/ष्, `Ÿ`, `Ê`, `´`);
- re-ordering of the short-i matra (typed *before* its consonant cluster) and of reph (typed *after* its syllable);
- conversion is done **per run of text according to its font** (resolved through run → character style → paragraph
  style → document default). English text, vehicle numbers, coordinates and so on, typed in other fonts inside the
  same paragraph, are left untouched. Consecutive runs in the legacy font are joined before conversion because one
  glyph sequence can span runs. Text whose font cannot be resolved is converted only if it looks like Kruti Dev.

Typists' own spelling habits are preserved (some offices consistently type ष for श).

### 2.3 Section splitting

`02_extract.py` reads paragraphs and table rows in document order and labels each with its section, using heading
patterns that tolerate numbering styles (`(1)`, `01.`, `(01.)`, `(क)`), spelling variants (`गिरफ्तारी`/`गिरफतारी`,
`निर्देश`/`निर्देष`) and offices that omit a heading altogether (an un-headed conclusion is recognised by its standard
opening "अबतक के अनुसंधान …", an un-headed direction by "अतः अनुसंधानकर्ता को … निर्देश/आदेश दिया जाता है").
Short header fields that reuse section names ("घटनास्थल :- <village>") stay in the header.

It also parses police station, case number, year, FIR date and sections of law from the case line, and classifies the
document (supervision note, supervision-cum-final report, final/progress report, Report-2).

Output: `dataset/<name>/records.jsonl` – one record per document with every paragraph labelled, so new tasks can be
built without re-reading the Word files.

### 2.4 Supervision-note dataset (`03_build_dataset.py`)

- **Input (user turn):** every paragraph before the conclusion, in original order.
- **Output (assistant turn):** `निष्कर्ष :- … / त्रुटि :- … / निर्देश :- 1. … 2. …` in a fixed layout
  (directions are re-numbered because Word's automatic list numbers are not part of the text).
- Kept: notes with both a conclusion (≥ 200 characters) and at least one direction.
- De-duplicated on the opening of the output text ("Copy of …" files are common).
- Split 90/5/5 **by case** (hash of police station + case number + year), so a case never appears in two splits.

Result: **4,708 examples** from 4,677 cases (train 4,250 / val 221 / test 237); 4,204 plain supervision notes and 504
combined supervision-cum-final reports (tagged `note_kind`). Left out: 1,154 progress/final reports, 641 duplicates,
300 notes without a usable conclusion or directions, 193 Report-2 files, 2 unreadable files.

Lengths (Qwen 3.5 tokenizer): median 6,264 tokens per example, 90% under 11,600, 99% under 20,900;
30.6M tokens in the training split. Hindi in this corpus costs roughly one token per 2.1 characters.

One office contributes almost half of the examples, so the model will lean towards that office's style.

### 2.5 Supervision note → Report-2 dataset (`04_build_report2_pairs.py`)

A report and a note are candidates when they share **case number and year**. Among the candidates the note that
contains the largest share of the report's distinct words is chosen, and the pair is kept only if that overlap is
high enough and the police station or FIR date also agrees. The scores separate cleanly: true pairs score ≥ 0.6,
false candidates (same number at another police station) 0.3–0.5.

Result: **1,576 pairs** (train 1,417 / val 76 / test 83), 1,369 of them with overlap ≥ 0.7. Each example carries
`match_overlap`; `match_audit.csv` lists every candidate with its scores. Only reports numbered 02 are used, and the
memo-number header lines of the report are dropped because they cannot be derived from the note.

Most of the unmatched reports belong to offices whose notes are not in the Word collection (some are probably among
the scanned PDFs).

### 2.6 Running the preparation

```
powershell -ExecutionPolicy Bypass -File 01_convert_doc.ps1
python 02_extract.py
python 03_build_dataset.py

powershell -ExecutionPolicy Bypass -File 01_convert_doc.ps1 -Src ..\report-2\word -Dst ..\report-2\word_converted
python 02_extract.py report-2 report2
python 04_build_report2_pairs.py
```

Needs Python 3.10+, `lxml`, and Microsoft Word for step 1.

## 3. Fine-tuning

### 3.1 Method (`05_train.py`)

- LoRA (rank 16, alpha 32, dropout 0.05) on the attention and MLP projections; base weights in bf16
  (`--load-4bit` switches to QLoRA for smaller GPUs). About 0.5% of the parameters are trained.
- **Loss on the assistant turn only**; the case facts are masked out. `--dry-run` prints where the mask starts and ends.
- Qwen's thinking mode is disabled in the chat template so the model learns to answer directly.
- Examples longer than `--max-len` tokens (default 16,384) are skipped rather than truncated, so no answer is cut.
- Batch size 1 with gradient accumulation 8, learning rate 1e-4, cosine schedule, 3% warm-up, 3 epochs,
  gradient checkpointing.
- Evaluation on the validation split each epoch; a resumable checkpoint every 100 steps.
- `--max-mem-gb` makes the process fail with an out-of-memory error instead of starving the machine.

```
python 05_train.py --model Qwen/Qwen3.5-4B --dry-run                                   # tokenisation + mask check
python 05_train.py --model Qwen/Qwen3.5-4B --limit 64 --epochs 1 --out ../runs/smoke   # end-to-end smoke test
python 05_train.py --model Qwen/Qwen3.5-4B --out ../runs/qwen35-4b-sn                  # full run
python 05_train.py --model Qwen/Qwen3.5-4B --data ../dataset/sn_to_report2 --out ../runs/qwen35-4b-r2
```

### 3.2 Notes from the first run (Qwen 3.5 4B on a single-GPU workstation)

Environment: Python 3.12, `torch 2.14.1+cu130`, `transformers 5.18.0`, `peft 0.21.2`, `accelerate 1.15.0`. The machine has one GPU and about 120 GB of memory shared between CPU and GPU.

- **Install the optimised kernels.** Qwen 3.5 uses linear-attention layers; without `flash-linear-attention` and
  `causal-conv1d`, `transformers` silently falls back to a slow reference implementation.
  `causal-conv1d` has to be built against the installed PyTorch:
  `pip install flash-linear-attention` and `pip install --no-build-isolation causal-conv1d` (with `nvcc` on `PATH`).

  | Setup | Training speed |
  |---|---|
  | Neither package | ~224 tokens/s |
  | `flash-linear-attention` only | ~530 tokens/s (short sample) |
  | Both | ~800–900 tokens/s |

- **Time.** 4,127 training examples fit in 16,384 tokens (28.0M tokens per epoch). At ~860 tokens/s one optimiser
  step (8 examples) takes about 66 s, an epoch about 9.5 hours and three epochs about 27 hours, plus roughly
  8 minutes per evaluation pass.
- **Memory.** Peak GPU allocation 56 GB with examples up to 16k tokens; PyTorch reserves up to ~75 GB because
  sequence lengths vary. Because CPU and GPU share one memory pool on this machine, the run was started from a
  wrapper script (not in this repository) that stops training if available system memory falls below 15 GB.
- **Unattended running.** The same wrapper waits until the GPU has been idle for five minutes before training, and
  an `@reboot` cron entry re-runs it after an unexpected reset. Because `05_train.py` resumes from the newest
  checkpoint in `--out`, this was enough to continue from step 600 after one reset, with about 15 minutes lost.
- **`transformers` 5 changes** met along the way: `warmup_ratio` is no longer accepted (the script passes a step
  count), and models are loaded with `dtype=` rather than `torch_dtype=`.
- **Result.** Validation loss 0.225 → 0.204 → 0.201 over the three epochs (most of the gain in epoch 2, no sign of
  over-fitting); training loss ended near 0.18. About 28 hours of training in total.

### 3.3 Evaluation (`06_evaluate.py`)

```
python 06_evaluate.py --model Qwen/Qwen3.5-4B --adapter ../runs/qwen35-4b-sn/final --out ../runs/qwen35-4b-sn/eval
python 06_evaluate.py --model Qwen/Qwen3.5-4B --out ../runs/qwen35-4b-base-eval      # untuned baseline
```

Writes `predictions.jsonl`, `scored.jsonl`, `metrics.json` and `report.html` (officer's text and model's text side
by side, worst cases first).

| Metric | Meaning | Officer's own text | Text from a different case |
|---|---|---|---|
| chrF (overall) | character n-gram similarity; tolerant of Hindi spelling variants | 100 | 39 |
| Directions covered | share of the officer's directions matched by a generated one | 100% | 24% |
| Law sections F1 | same sections of law cited in the finding | 1.00 | 0.03 |
| Verdict agreement | same finding: true / false / true but unproved (keyword rules) | 100% | 55% |
| Ungrounded fact share | vehicle numbers, dates, long numbers that appear nowhere in the note | 1% | 98% |

The last column is the floor produced by boilerplate and chance; a useful model has to be well above it. The scores
measure closeness to one officer's wording, so the HTML report is for reading and the numbers are for comparing runs.

### 3.4 Results (Qwen 3.5 4B, 233 test cases)

| Metric | Untuned base model | Fine-tuned | Fine-tuned + loop guard |
|---|---|---|---|
| Well-formed (all three parts) | 8% | 97% | 99% |
| chrF, overall | 43 | 78 | 80 |
| chrF, conclusion | 15 | 86 | 86 |
| Directions covered | 0% | 58% | 58% |
| Law sections F1 | 0.16 | 0.88 | 0.88 |
| Verdict agreement | 28% | 91% | 92% |
| Ungrounded fact share | 19% | 0.8% | 0.8% |
| Ran to the token limit | 29% | 24% | 2% |

- **Conclusions are reliable**: the right sections of law, the same finding as the officer in nine cases out of ten,
  and almost no invented numbers, dates or vehicle numbers.
- **Directions are the weak part.** In about a quarter of the cases greedy decoding fell into a loop, writing the same
  direction again and again. `--stop-on-repeat` ends generation once three earlier lines have been repeated and drops
  the repeats; that removes the junk (output length back to 1.0× the officer's) but does not add the missing
  directions – on those cases the model writes about 5 distinct directions where the officer wrote about 9.

## 4. Drafting console (`07_webui.py`)

A Gradio web page in Hindi. The operator fills in the case details and the sections that were the model's input
during training (प्राथमिकी, घटनास्थल, अभियोजन साक्ष्य, साक्षी, अनुसंधान में प्रगति, गिरफ्तारी, समर्पण,
तलाशी सह जप्ती, प्रतिरक्षा साक्ष्य); the model writes **निष्कर्ष**, **त्रुटि** and **निर्देश** into editable boxes,
streaming as it goes.

- The fields are laid out the way supervision notes are written, so the prompt matches the training format; empty
  arrest / seizure / defence fields become "शून्य।" as in real notes.
- Text typed or pasted in **Kruti Dev** is converted to Unicode automatically (a checkbox forces it for very short
  fields), and the converted text is written back into the form for checking.
- The loop guard runs during generation; the operator is told when it fired, because the directions may then be
  incomplete.
- Logins are required (salted scrypt hashes in `webui_users.json`, which is git-ignored). Nothing typed into the form is
  written to disk or logged – only timings.

```
python 07_webui.py add-user <name>                                         # asks for the password
python 07_webui.py serve --adapter ../runs/qwen35-4b-sn/final              # one draft at a time (transformers)
python 08_merge_adapter.py --adapter ../runs/qwen35-4b-sn/final --out ../runs/qwen35-4b-sn/merged
python 07_webui.py serve --adapter ../runs/qwen35-4b-sn/merged --vllm-url http://127.0.0.1:8000   # several users
```

**Serving several users.** With plain `transformers` each draft takes the GPU to itself (roughly one to several
minutes per draft), so simultaneous users queue. For 5–10 concurrent users the console can send generation to a
**vLLM** server, which batches requests on the GPU; the loop guard then works by closing the stream, which makes
vLLM drop the request. Serve the merged checkpoint locally, e.g.

```
VLLM_NO_USAGE_STATS=1 vllm serve ../runs/qwen35-4b-sn/merged --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.25 --max-model-len 20480 --max-num-seqs 10 --limit-mm-per-prompt '{"image": 0, "video": 0}'
```

`--gpu-memory-utilization` is a share of the whole GPU memory; keep it low on a machine whose memory is shared with
other work.

**Access.** The console listens on `127.0.0.1` only. It is reached through a private overlay network (Tailscale
`serve`, tailnet only – not `funnel`), with an access rule that lets operators reach the console's HTTPS port and
nothing else on the machine.

## 5. Folder layout

```
<project>/
  data/word/            supervision-note Word files        (not in the repository)
  data/word_converted/  .docx converted from .doc          (created by step 1)
  report-2/word/        Report-2 Word files                (not in the repository)
  dataset/supervision/  records.jsonl, train/val/test.jsonl, stats.json
  dataset/report2/      records.jsonl
  dataset/sn_to_report2/ train/val/test.jsonl, match_audit.csv, stats.json
  runs/                 checkpoints, adapters, merged model, evaluation output
  sn_pipeline/          this repository
```

## 6. Known limits

- Scanned PDFs are not used.
- Section splitting is rule-based; about 300 notes with unusual layouts are dropped rather than guessed at.
- Police station / case number parsing is best-effort (about 96% of examples); the rest fall back to a per-document key.
- Typing errors in the source documents are kept as they are.
- Model output is a draft for an officer to review, not a finished supervision note.
