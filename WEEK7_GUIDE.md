# 📘 Week 7 — Complete Beginner's Guide (QLoRA: Fine-Tuning an Open-Source Model)

This guide explains **every file** in `week7/` of the *LLM Engineering* repo — what each piece does, why it exists, and how it all fits together. (Companions: Weeks 1–2, 3–4, 5, and 6 guides.)

> **The one-sentence summary:** Week 6 fine-tuned GPT-4.1-nano through OpenAI's API and it *disappointed*. This week we do fine-tuning **the real way**: take Meta's open-source **Llama 3.2 3B**, freeze it, attach tiny trainable **LoRA adapters**, quantize it to 4 bits (**QLoRA**), and train it **ourselves on a GPU** on 800k product descriptions — producing **the best price-predictor of the entire course ($39.85 error), beating GPT-5.1 ($44.74)** with a model that fits in your hand.

> **Where the code lives:** Like Week 3, the heavy lifting runs on **Google Colab** (you need a GPU). Days 1, 3–4 and 5 are Colab links; **Day 2 and the shared utilities run locally** and are fully explained here. All three Colabs were fetched and are explained below too.

---

# 🧠 PART 0 — The Big Picture

## Why this week exists

Week 6's ending was a deliberate cliffhanger: fine-tuning *via API* didn't beat zero-shot frontier models. The course's answer: that was fine-tuning with **one hand tied behind your back**. This week you control the whole training stack — and with the right technique (QLoRA) and enough data (800k examples), a 3-billion-parameter model **out-learns** models 100× its size on this task.

## The two big ideas (in plain English)

### Idea 1 — LoRA: don't retrain the brain, add a tiny "attachment"

Fine-tuning *all* of Llama 3.2 3B means updating ~3 billion weights — and storing a full copy of the model per run. **LoRA** (Low-Rank Adaptation) is the elegant dodge:

- **Freeze** the entire base model. It never changes.
- For chosen layers, add a **side path**: two small matrices, `A` (down-project) and `B` (up-project), meeting in the middle at a tiny **rank `r`**.
- The layer's output becomes `original_weights + alpha × (B × A)`. Only `A` and `B` are trained.

A layer with 3,072×3,072 weights (9.4M) gets adapters of just `3072×r + r×3072` weights — with `r=32`, about 197k, **2% of the original**. Multiply across layers and your whole "fine-tune" is a few dozen MB instead of gigabytes. Intuition from the notebook: the adapters act like a compact "skill patch" pinned onto the frozen brain.

### Idea 2 — QLoRA: shrink the brain so it fits on a free GPU

**QLoRA = Quantization + LoRA.** Load the frozen base model in **4-bit** precision (NF4 — the exact `BitsAndBytesConfig` from Week 3!) so Llama 3.2 3B drops to ~1.7 GB, while the LoRA adapters train in normal precision on top. Result: serious fine-tuning on a **free Colab T4**.

## The week's "Order of play"

```
DAY 1  QLoRA                   Colab: feel quantization (16-bit vs 8-bit vs 4-bit),
                               load Ed's finished adapter, do the LoRA math by hand
DAY 2  Prompt Data + Base Model  (LOCAL) build prompt→completion training pairs
DAY 3  Train Part 1            Colab: all hyper-parameters explained
DAY 4  Train Part 2            Colab: run SFTTrainer, watch loss fall, push to the Hub
DAY 5  Eval                    Colab: score it against the 10k test set
```

## The final leaderboard (`results.ipynb`) — the payoff

Average absolute error on the test set (lower = better):

| Model | Error | Note |
|---|---|---|
| Constant baseline | $106.18 | the floor |
| Base Llama 3.2 3B (4-bit, **not** fine-tuned) | **$110.72** | worse than doing nothing! 🔥 |
| Human (Ed) | $87.62 | week 6 |
| NLP + Linear Regression | $76.81 | week 6 |
| **GPT-4.1-nano fine-tuned via API** | $75.91 | week 6's disappointment |
| Random Forest / XGBoost | $72.28 / $68.23 | week 6 |
| **Llama 3.2 fine-tuned on LITE data (20k)** | $65.40 | already good |
| Deep Neural Network | $63.97 / $46.49 | week 6 |
| GPT-4.1-nano zero-shot | $62.51 | week 6 |
| Grok 4.1 Fast / Gemini 3 Pro | $57.62 / $50.54 | week 6 |
| Claude 4.5 Sonnet | $47.10 | week 6 |
| GPT-5.1 | $44.74 | week 6's champion |
| 🥇 **Llama 3.2 3B + QLoRA on FULL data (800k)** | **$39.85** | **the course champion** |

Three lessons baked into those bars:
1. **The base model was terrible ($110.72)** — worse than always guessing the average. All that skill came from *your* training run.
2. **Data quantity mattered**: 20k examples → $65.40; 800k examples → $39.85. The single biggest lever was more data, not a cleverer technique.
3. **A 3B open model beat every frontier model** — because the task is narrow and the training was focused. (It also runs free, offline, and private forever.)

## What's in the folder

```
day1.ipynb            Launchpad → Colab "qlora intro"
day2.ipynb            FULL CODE (local): build the prompt/completion dataset
day3 and 4.ipynb      Launchpad → Colab "TRAINING"
day5.ipynb            Launchpad → Colab "Eval"
results.ipynb         The final leaderboard chart (above)
pricer/items.py       Week 6's Item + NEW prompt/completion methods
pricer/evaluator.py   Week 6's Tester (for Item objects) — reused on Colab
util.py               The Tester adapted to prompt/completion dicts (downloaded into Colabs)
```

---

# 🟦 PART 1 — The local files

## 1.1 `day2.ipynb` — Building the training data (prompt/completion pairs)

Training needs examples in the shape *"given this text, produce that text"* — not the chat-messages format of Week 6's API fine-tuning. Day 2 manufactures them:

### (a) Load data & tokenizer

```python
BASE_MODEL = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

train, val, test = Item.from_hub("ed-donner/items_full")   # 800k / 10k / 10k from Week 6
```

Note `Llama-3.2-3B` — the **base** model, not the `-Instruct` chat version. We're teaching it one focused job from scratch, and SFT just continues text; no chat template needed.

### (b) Measure, then choose a truncation cutoff

```python
token_counts = [item.count_tokens(tokenizer) for item in tqdm(items)]   # tokens per summary
# ...plot histogram...

CUTOFF = 110
cut = len([count for count in token_counts if count > CUTOFF])
print(f"With this CUTOFF, we will truncate {cut:,} items which is {cut/len(items):.1%}")
```

Same "plot before you decide" discipline as Week 6: the histogram of summary lengths shows most summaries fit comfortably under ~110 tokens, so a `CUTOFF = 110` keeps sequences short (→ fast, memory-cheap training) while losing only a sliver of products.

### (c) `Item.make_prompts` — the format magic (`pricer/items.py`)

```python
def make_prompts(self, tokenizer, max_tokens, do_round):
    """Make prompts and completions"""
    tokens = tokenizer.encode(self.summary, add_special_tokens=False)
    if len(tokens) > max_tokens:
        summary = tokenizer.decode(tokens[:max_tokens]).rstrip()   # truncate to the token budget
    else:
        summary = self.summary
    self.prompt = f"{QUESTION}\n\n{summary}\n\n{PREFIX}"           # QUESTION = "What does this cost to the nearest dollar?"
    self.completion = f"{round(self.price)}.00" if do_round else str(self.price)
```

One training example comes out as:

```
PROMPT:      What does this cost to the nearest dollar?

             Title: Excess V2 Distortion Pedal
             Category: Music Pedals
             ...

             Price is $
COMPLETION:  149.00
```

The model's whole job becomes: after seeing "Price is $", emit the right digits. Notice the truncation is done **on tokens, not characters** (`decode(tokens[:max_tokens])`) — character truncation could cut a word in half and leave broken text; token-level truncation is the correct way to hit a token budget.

### (d) Verify lengths & ship to the Hub

```python
prompt_token_counts = [item.count_prompt_tokens(tokenizer) for item in tqdm(items)]  # prompt+completion
# ...histogram: everything fits in ~128 tokens → this becomes MAX_SEQUENCE_LENGTH in training...

Item.push_prompts_to_hub("ed-donner/items_prompts_full", train, val, test)
```

`push_prompts_to_hub` stores the minimal training record per item — `{"prompt": ..., "completion": ...}` — as `items_prompts_lite` / `items_prompts_full`. That's the exact schema the training Colab consumes, and the same schema `util.py`'s Tester reads at eval time. (Day 2 ends with a link to a second Colab that walks the same steps live.)

## 1.2 `util.py` — the referee, retooled for the GPU

This is Week 6's `Tester` with three deliberate changes — a nice "spot the difference" exercise in adapting code to context:

1. **Datapoints are dicts now**: `truth = float(datapoint["completion"])` and the display title is parsed out of the prompt text (`datapoint["prompt"].split("Title: ")`). Week 7 evaluates directly against the prompt/completion dataset.
2. **No threads**: Week 6 ran API calls concurrently (`ThreadPoolExecutor`) because it was *I/O-bound* (waiting on the network). Here the predictor is a *local GPU generating one answer at a time* — sequential `for i in tqdm(range(self.size))`, since the GPU is already saturated.
3. **`clear_output(wait=True)`** before the final charts — hundreds of colored `$23 $41 $12...` prints get wiped so the two Plotly charts (guess-vs-truth scatter with the `y = x` dashed line, and the running-error chart with the shaded 95% CI) are the first thing you see.

Everything else — regex `post_process` to extract the number from generated text, traffic-light coloring, MRR-style honest reporting — is identical, so **results stay comparable with every Week 6 model**. The training Colab even downloads this file straight from GitHub:

```python
!wget -q https://raw.githubusercontent.com/ed-donner/llm_engineering/main/week7/util.py -O util.py
```

`pricer/evaluator.py` (local) is the Week 6 version kept for evaluating against `Item` objects if you prefer.

---

# 🟩 PART 2 — The three Colabs

## 2.1 Day 1 — "QLoRA intro": *feel* the memory savings

This Colab is deliberately experiential — same model, three precisions, restart in between (Runtime → Restart session, to clear the GPU):

```python
# 1) Full precision (16-bit)
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", device_map="auto")
print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:,.1f} GB")   # ≈ 6.4 GB

# 2) 8-bit
quant_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_config, device_map="auto")
# ≈ 3.4 GB

# 3) 4-bit NF4 (the Week 3 config, back for its greatest hit)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4")
# ≈ 1.7 GB — leaves room on a free T4 for training!
```

Then the payoff preview — **loading Ed's finished product**: an adapter is not a whole model, it's a small add-on you mount on the frozen base:

```python
from peft import PeftModel
fine_tuned_model = PeftModel.from_pretrained(base_model, "ed-donner/price-2025-11-30_15.10.55-lite")
```

`peft` = *Parameter-Efficient Fine-Tuning*, the HuggingFace library that implements LoRA. Printing `fine_tuned_model` shows the base model plus tiny `lora_A`/`lora_B` layers injected into the target modules.

### The LoRA math cells (do it by hand!)

```python
r = 32
lora_q_proj = 3072 * r + 3072 * r     # A: 3072->r, B: r->3072
lora_k_proj = 3072 * r + 1024 * r     # K and V project to a smaller dim (1024)
lora_v_proj = 3072 * r + 1024 * r
lora_o_proj = 3072 * r + 3072 * r
lora_layer  = lora_q_proj + lora_k_proj + lora_v_proj + lora_o_proj
params      = lora_layer * 28          # Llama 3.2 3B has 28 decoder layers
size        = (params * 4) / 1_000_000
# → ~21M params, ~85MB — that's the ENTIRE fine-tune for the lite run!
```

The full-data run then goes "pretty extreme": `r = 256` **and** LoRA on the MLP layers too (`gate_proj`, `up_proj`, `down_proj` in addition to `q/k/v/o_proj`) — with 800k training examples, there's enough signal to train a much bigger adapter. Same cell recomputes: a few hundred MB — still tiny next to the 6.4GB model.

## 2.2 Days 3–4 — "TRAINING": the main event

> **Free T4** for `LITE_MODE=True` (20k examples); a paid **A100** for the full 800k run.

### The hyper-parameter wall (every line explained)

```python
# overall
EPOCHS = 1 if LITE_MODE else 3              # passes over the data (week 6's training loop!)
BATCH_SIZE = 32 if LITE_MODE else 256       # examples per optimization step
MAX_SEQUENCE_LENGTH = 128                   # matches day 2's histogram — nothing gets cut
GRADIENT_ACCUMULATION_STEPS = 1             # simulate bigger batches by accumulating (1 = off)

# QLoRA
QUANT_4_BIT = True
LORA_R = 32 if LITE_MODE else 256           # adapter rank — capacity of the "skill patch"
LORA_ALPHA = LORA_R * 2                     # output scale; convention: 2×r
TARGET_MODULES = ["q_proj","v_proj","k_proj","o_proj"]            # attention (lite)
              # + ["gate_proj","up_proj","down_proj"]              # MLP too (full)
LORA_DROPOUT = 0.1                          # regularize the adapters

# training
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.01                         # start slow: LR ramps up over the first 1% of steps
LR_SCHEDULER_TYPE = 'cosine'                # then decay along a cosine curve (week 6's DNN again)
WEIGHT_DECAY = 0.001
OPTIMIZER = "paged_adamw_32bit"             # Adam that pages optimizer state to CPU when tight
capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8               # newer GPUs support bfloat16; T4 (7.x) uses fp16
```

The Colab pauses on **optimizers** with the key insight: plain **Adam/AdamW** stores rolling gradient statistics *per parameter* — roughly 2 extra numbers per weight — so on a 3B model that's gigabytes of extra GPU memory. `paged_adamw_32bit` (from the bitsandbytes library) pages that state to CPU RAM when the GPU gets tight. A free tier survival trick worth knowing.

Also here: logins for **HuggingFace** (to push the model) and **Weights & Biases** (wandb.ai — the industry-standard training dashboard; a free account + `WANDB_API_KEY` secret gets you live loss curves, learning-rate plots, and gradient histograms).

### Load model, data, and the two config objects

```python
dataset = load_dataset(DATASET_NAME)          # items_prompts_lite/full from day 2
train, val, test = dataset['train'], dataset['val'].select(range(VAL_SIZE)), dataset['test']

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token    # pad short sequences with the end-of-text token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_config, device_map="auto")

lora_parameters = LoraConfig(
    lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, r=LORA_R,
    bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES)

train_parameters = SFTConfig(
    output_dir=PROJECT_RUN_NAME,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIMIZER,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.001,
    max_grad_norm=0.3,                  # clip gradients (week 6's DNN trick, smaller cap)
    fp16=not use_bf16, bf16=use_bf16,
    max_length=MAX_SEQUENCE_LENGTH,
    group_by_length=True,               # batch similar-length examples together → less padding → faster
    logging_steps=LOG_STEPS, eval_strategy="steps", eval_steps=SAVE_STEPS, save_steps=SAVE_STEPS,
    report_to="wandb", run_name=RUN_NAME,
    push_to_hub=True, hub_model_id=HUB_MODEL_NAME, hub_strategy="every_save",  # checkpoint to HF as it trains
)
```

**SFT** = *Supervised Fine-Tuning*: training on `prompt → completion` pairs, exactly the day-2 format. `SFTTrainer`/`SFTConfig` come from **TRL** (Transformer Reinforcement Learning — HF's training library; here only the "S" part is used).

### Lift-off

```python
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=train,
    eval_dataset=val,
    peft_config=lora_parameters,     # ← PEFT: only the adapters will train
    args=train_parameters)

fine_tuning.train()                                          # the 4-stage loop from week 6, industrialized
fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True) # push the adapter to your HF account
wandb.finish()
```

`fine_tuning.train()` is Week 6's hand-rolled `forward → loss → backward → optimize` loop, productionized: automatic batching, mixed precision, cosine LR, checkpoint saves, eval passes, W&B logging. Watch the training loss *and* the validation loss in W&B — val loss is the honesty check for overfitting.

**Practical notes from the notebook:** training uploads checkpoints to the Hub every `SAVE_STEPS`; free Colabs can be reclaimed mid-run — a companion Colab shows how to **resume** from your last checkpoint (the trick: load the adapter with `is_trainable=True`).

## 2.3 Day 5 — "Eval": the moment of truth

*(This Colab requires a Google sign-in to open, but its shape is fully determined by the local files.)* The evaluation pattern:

1. Load the 4-bit base model (as Day 1).
2. Mount the adapter: `PeftModel.from_pretrained(base_model, "your-username/price-...")`.
3. For each of the first 200 test prompts: tokenize `prompt`, `model.generate(...)`, decode — the model emits something like `149.00`.
4. Feed every prediction through **`util.py`'s `evaluate(predictor, test)`** — same regex post-processing, same scatter + running-error charts, same $ error metric as Week 6.

You run it three times to fill in the leaderboard: **base model, no adapter** ($110.72 — worse than guessing the average!), **lite-trained adapter** ($65.40), and **full-trained adapter** ($39.85 — champion of the course). Then `results.ipynb` renders the final 17-bar chart reproduced in Part 0.

---

# 🎓 PART 3 — Consolidation

## Concepts introduced, week by day

| Concept | Where | One-liner |
|---|---|---|
| **LoRA** (low-rank adaptation) | D1, D3 | Freeze the model; train tiny A×B adapters on attention/MLP layers |
| Rank `r`, alpha, target modules | D1, D3 | Adapter capacity; output scale (2r); which layers get adapters |
| **QLoRA** | D1 | 4-bit quantized base + full-precision adapters = training on a free GPU |
| 16-bit vs 8-bit vs 4-bit footprints | D1 | 6.4 GB → 3.4 GB → 1.7 GB for the same 3B model |
| `PeftModel.from_pretrained` | D1, D5 | Adapters are small add-ons you mount on a shared base |
| Prompt/completion SFT format | D2 | Text-in → text-out pairs (vs Week 6's chat-messages API format) |
| Token-level truncation | D2 | Cut on tokens, not characters, to respect a token budget |
| **SFTTrainer / SFTConfig (TRL)** | D3–4 | Industrial version of week 6's 4-line training loop |
| LoraConfig (PEFT) | D3 | The adapter blueprint |
| Optimizer memory & `paged_adamw_32bit` | D3 | Adam keeps 2 extra numbers per weight; page them to CPU when tight |
| Warmup + cosine LR + max_grad_norm | D3 | The professional LR recipe (week 6's DNN, revisited) |
| bf16 vs fp16 by GPU capability | D3 | Newer GPUs use bfloat16; T4 falls back to fp16 |
| **Weights & Biases** | D3–4 | Live training telemetry: loss curves, LR, gradients |
| Checkpointing & resuming | D3–4 | Push checkpoints to the Hub; resume after Colab reclaims the runtime |
| Evaluating the base vs the adapter | D5 | Base $110.72 → lite $65.40 → full $39.85 |

## The week as a diagram

```
WEEK 6 DATA (items_full: 820k cleaned Amazon products)
        │  day2: tokenize → truncate to 110 tokens → wrap
        ▼
"What does this cost...?  <summary>  Price is $"  →  "149.00"
        │  push to hub: items_prompts_full
        ▼
COLAB (A100):  Llama-3.2-3B ──4-bit quantize (QLoRA)──► frozen base ~1.7GB
               + LoRA adapters (r=256, attention+MLP)  ← the ONLY thing trained
               SFTTrainer: 3 epochs, batch 256, lr 1e-4, cosine, wandb
        │  push adapter to HF Hub
        ▼
COLAB:  base + adapter → generate on test prompts → util.py Tester
        ▼
$39.85  🏆  beats GPT-5.1 ($44.74) with a 3B model you own
```

## Glossary additions

- **LoRA / QLoRA** — training small low-rank adapters on a frozen model / doing it on a 4-bit-quantized base.
- **PEFT** — HuggingFace's parameter-efficient fine-tuning library (LoRA's home).
- **Adapter** — the trained LoRA matrices, shipped as a small add-on to a base model.
- **Rank (r) / alpha** — adapter width (capacity) / output scaling factor (conventionally 2r).
- **Target modules** — the layer names (q/k/v/o_proj, gate/up/down_proj) that receive adapters.
- **SFT** — supervised fine-tuning: training on prompt→completion pairs.
- **TRL** — HuggingFace's training library (`SFTTrainer`, `SFTConfig`).
- **NF4 / double quant** — the 4-bit format and its compression-of-compression trick (Week 3 returns).
- **paged optimizer** — an optimizer whose state pages to CPU RAM when GPU memory runs out.
- **Warmup / cosine schedule** — learning rate ramps up over the first ~1% of steps, then decays on a cosine curve.
- **bf16 / fp16** — 16-bit float formats; bf16 is more robust but needs newer GPUs.
- **Weights & Biases (wandb)** — the standard dashboard for training runs.
- **Group by length** — batching similar-length examples to minimize padding waste.

## Running it yourself

1. **Prerequisites:** HuggingFace account + token (approve Llama 3.2 on its model page), free **Weights & Biases** account, and the Week 6 environment locally (`uv sync`).
2. **Day 2 (local):** run `week7/day2.ipynb` with `LITE_MODE = True` — it rebuilds the prompt dataset (or just use `ed-donner/items_prompts_lite` from the Hub and skip ahead).
3. **Day 1 Colab:** free T4, add `HF_TOKEN` secret; restart the session between the three quantization demos (the notebook tells you when).
4. **Days 3–4 Colab:** `LITE_MODE=True` trains ~20k examples × 1 epoch on a **free T4** in a few hours. Full mode (800k × 3 epochs) needs a paid A100 and roughly a day. Watch loss curves in W&B; checkpoints push to your HF account as it goes.
5. **Day 5 Colab:** load your adapter over the 4-bit base and run `evaluate(predictor, test)` — compare against base/no-adapter to see your training pay off.
6. Budget: the lite run is **free**; the full run is the one place the course suggests meaningful spend (~$10–15 of A100 time) — and it buys you the course's best model.

## Where this leaves the course

You've now trained a language model end-to-end with the same machinery the pros use — quantization, PEFT, SFT, telemetry, checkpointing — and you have empirical answers to the field's most practical question: *when does fine-tuning beat prompting?* (When you control the stack, the data is plentiful, and the task is narrow.) **Week 8** is the grand finale: an autonomous multi-agent system that recruits everything you've built — tools (W2), local models (W3), code generation & evaluation (W4), RAG (W5), and now your own fine-tuned model (W7) — to collaborate on a business problem.
