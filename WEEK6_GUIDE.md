# 📘 Week 6 — Complete Beginner's Guide (Fine-Tuning: "The Price Is Right" Capstone)

This guide explains **every file** in `week6/` of the *LLM Engineering* repo — what each piece does, why it exists, and how it all fits together. (Companions: Weeks 1–2, 3–4, and 5 guides.)

> **The one-sentence summary:** Instead of *prompting* models (Weeks 1–2), *running* them (Week 3), or *feeding* them documents (Week 5), this week we **change what a model has learned** — fine-tuning — inside the course's big capstone: **predict a product's price from its text description**, using ~820,000 real Amazon products. You'll build baselines, traditional ML, a PyTorch neural network, zero-shot frontier LLMs, and a fine-tuned GPT-4.1-nano — and race them all on one leaderboard.

> **Why this dataset/task?** Price prediction is perfect for teaching *training*: the answer is one number (easy to score automatically), and text→price requires genuine "knowledge" — no keyword trick will tell you a guitar pedal costs $149.

---

# 🧠 PART 0 — The Big Picture

## RAG vs fine-tuning (the distinction this week cements)

| | RAG (Week 5) | Fine-tuning (Week 6) |
|---|---|---|
| Changes | What's **in the prompt** | The model's **weights** |
| Best for | **Facts/knowledge** ("our refund policy") | **Style/skill/format** ("always answer like X") |
| Analogy | Open-book exam | Tutoring / practice |
| Cost | Cheap, instant updates | Training run + hosted private model |

Recall Week 5's opener: *"In week 6 we will try fine-tuning GPT ourselves."* Here it is.

## The week's "Order of play"

```
DAY 1  Data Curation      scrape → clean → filter → sample 820k Amazon products
DAY 2  Data Pre-processing  have an LLM rewrite every messy product into a clean summary
DAY 3  Evaluation + Traditional ML   define the error metric; random → constant →
                            linear regression → random forest → XGBoost
DAY 4  Neural Networks & LLMs  a HUMAN baseline(!) → vanilla PyTorch net → zero-shot
                            frontier models (GPT-5.1, Claude, Gemini, Grok…)
DAY 5  Fine-tuning          create your own private variant of GPT-4.1-nano via OpenAI's API
BONUS  "Redemption"         a deep residual neural network that nearly beats the frontier
```

## The final leaderboard (`results.ipynb`)

Average absolute error on the test set — **lower is better**:

| Model | Error | Color code |
|---|---|---|
| Constant (always the average price) | $106.18 | traditional ML |
| Linear Regression (3 features) | $101.56 | traditional ML |
| **Human (Ed himself!)** | **$87.62** | 👀 |
| NLP + Linear Regression (bag of words) | $76.81 | traditional ML |
| Random Forest | $72.28 | traditional ML |
| XGBoost | $68.23 | traditional ML |
| GPT-4.1 Nano **fine-tuned** (100 examples) | $75.91 | fine-tuned |
| Vanilla Neural Network (PyTorch) | $63.97 | deep learning |
| GPT-4.1 Nano zero-shot | $62.51 | frontier |
| Grok 4.1 Fast | $57.62 | frontier |
| Gemini 3 Pro | $50.54 | frontier |
| **Deep Neural Network (the "redemption")** | **$46.49** | deep learning |
| Claude 4.5 Sonnet | $47.10 | frontier |
| 🥇 **GPT-5.1** | **$44.74** | frontier |

Three plot twists worth savoring:
1. **Ed the human ($87.62) lost to several algorithms** — including the trivial Random Forest.
2. **Fine-tuning was the disappointment**: a fine-tuned nano didn't beat the *same* model zero-shot. Fine-tuning small data on a knowledge-heavy task doesn't work magic — a genuinely valuable lesson (this is exactly why the bonus notebooks are called *"redemption"*).
3. **A custom deep neural network ($46.49) nearly beat GPT-5.1** — classical ML craft is not dead!

## What's in the folder

```
day1.ipynb … day5.ipynb     The five labs, in order
results.ipynb               The final leaderboard bar chart (above)
redemption_train.ipynb      OPTIONAL: train the deep neural network yourself (~4h on a GPU Mac)
redemption_run.ipynb        …or download Ed's trained weights and just run it
human_in.csv / human_out.csv  The human experiment: 100 test products + Ed's own price guesses
pricer/                     The reusable Python package every notebook imports
jsonl/                      Where batch files & fine-tune files get written (starts empty, .gitkeep)
```

---

# 🟩 PART 1 — `pricer/`: the reusable package

Weeks 5 and 6 both turn notebook experiments into importable modules — this is the week's version.

## `pricer/items.py` — the data-point class

```python
class Item(BaseModel):                 # Pydantic model = data with validation
    title: str
    category: str
    price: float
    full: Optional[str] = None         # the raw cleaned text (day 1)
    weight: Optional[float] = None     # normalized to pounds (day 1)
    summary: Optional[str] = None      # the LLM rewrite (day 2)
    prompt: Optional[str] = None
    id: Optional[int] = None

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"

def make_prompt(self, text):           # used by Week 7's OpenAI-era training
    self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"

def test_prompt(self) -> str:          # the same prompt with the answer REMOVED
    return self.prompt.split(PREFIX)[0] + PREFIX
```

Plus two hub helpers used by every notebook: `Item.push_to_hub(name, train, val, test)` (wraps lists in HF `DatasetDict` and uploads) and `Item.from_hub(name)` (downloads and re-validates into `Item` objects). The **same train/val/test objects travel with you all week** via the Hub — which is why every notebook starts identically:

```python
train, val, test = Item.from_hub("ed-donner/items_full")   # 800k / 10k / 10k
# or items_lite: 20k / 1k / 1k for the cheap fast version
```

## `pricer/parser.py` — turning raw Amazon JSON into clean Items (day 1)

Raw Amazon metadata is a swamp: missing prices, part numbers, inconsistent units. `parse(datapoint, category)` is the filter funnel:

```python
MIN_CHARS = 600; MIN_PRICE = 0.5; MAX_PRICE = 999.49
MAX_TEXT_EACH = 3000; MAX_TEXT_TOTAL = 4000

def parse(datapoint, category):
    try:
        price = float(datapoint["price"])
    except ValueError:
        return None                        # no valid price -> reject
    if MIN_PRICE <= price <= MAX_PRICE:    # keep the $0.50–$999.49 range
        title = datapoint["title"]; description = ...; features = ...
        details = json.loads(datapoint["details"])
        weight = get_weight(details)       # pounds/ounces/grams/kg/hundredths -> POUNDS
        full = scrub(title, description, features, details)
        if len(full) >= MIN_CHARS:         # enough text to learn from
            return Item(title=title, category=category, price=price, full=full, weight=weight)
    # implicit: anything that fails any check returns None and is dropped
```

- **`scrub(...)`** drops useless fields (`Part Number`, `Best Sellers Rank`, …), flattens each section to single-spaced text (truncated to 3,000 chars each, 4,000 total), and — nice touch — **regex-deletes part numbers**: `r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b"` = "7+ character blobs mixing letters and digits" (like `B01D05U9NO`). Part numbers are noise that would let a model *memorize* items instead of *learning* prices.
- **`get_weight(details)`** normalizes every unit to pounds (ounces÷16, grams÷453.592, …), returning 0 when unknown.
- **Why the $1–$1000 range?** It tames the distribution: no $0.01 paperclips, no $50,000 machines — models train far better on this band (and the range is why the price histograms look like a hill, not a cliff).

## `pricer/loaders.py` — parallel dataset building

`ItemLoader(category).load()` downloads one Amazon category and parses it **using all your CPU cores**:

```python
CHUNK_SIZE = 1000
WORKERS = max(os.cpu_count() - 1, 1)

def load_in_parallel(self, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), total=chunk_count):
            results.extend(batch)
```

It splits the dataset into 1,000-row chunks (`chunk_generator` — a generator, like Week 2's `yield`), farms chunks to worker *processes* (true parallelism, same as Week 5's pro ingestion), and shows a `tqdm` progress bar. 8 categories × ~hundreds of thousands of rows each would be painfully slow single-threaded.

## `pricer/preprocessor.py` + `pricer/batch.py` — the day 2 machinery

**`Preprocessor`** wraps the "rewrite this product" LLM call and tracks tokens/cost per call (using LiteLLM's `response._hidden_params["response_cost"]`, as in Week 2):

```python
SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format...
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""
```

**`Batch`** is the star: a class that rewrites **thousands of products at 50% cost** using the **Batch API** (submit jobs, collect results within 24h — OpenAI/Groq give ~50% off for asynchronous processing):

```python
class Batch:
    BATCH_SIZE = 1_000

    def make_jsonl(self, item):        # one JSONL line = one API request, tagged with custom_id
        body = {"model": "openai/gpt-oss-20b",
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": item.full}],
                "reasoning_effort": "low"}
        return json.dumps({"custom_id": str(item.id), "method": "POST",
                           "url": "/v1/chat/completions", "body": body})

    @classmethod
    def create(cls, items, lite):   ...   # split items into chunks of 1,000
    @classmethod
    def run(cls):                   ...   # for each: write file → files.create(purpose="batch")
                                          #             → batches.create(completion_window="24h")
    @classmethod
    def fetch(cls):                 ...   # poll batches.retrieve(); when status=="completed",
                                          # download the output file and map each result back to
                                          # items[int(custom_id)].summary = ...
    @classmethod
    def save(cls) / load(cls):      ...   # pickle the batch state so you can stop/resume
```

The JSONL format (one JSON object per line) is the universal "many requests" file format — you'll meet it again in day 5 for fine-tuning. `custom_id` is how results find their way back to the right product after asynchronous processing. **Cost:** roughly $1 for the 22k-item lite set, ~$30 for all 820k — or **$0** by loading Ed's already-processed dataset (the recommended path).

## `pricer/evaluator.py` — the referee EVERY model answers to

Every single model this week is scored by one function:

```python
def evaluate(function, data, size=DEFAULT_SIZE, workers=WORKERS):
    Tester(function, data, size=size, workers=workers).run()
```

`Tester` in detail:

```python
@staticmethod
def post_process(value):                      # models return MESSY text ("about $45 dollars")
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", value)   # first number in the string
        return float(match.group()) if match else 0      # nothing numeric -> 0
    return value

def color_for(self, error, truth):            # per-datapoint traffic lights
    if error < 40 or error / truth < 0.2:  return "green"    # close, or within 20%
    elif error < 80 or error / truth < 0.4: return "orange"
    else: return "red"

def run(self):                                # parallel inference!
    with ThreadPoolExecutor(max_workers=self.workers) as ex:
        for title, guess, truth, error, color in tqdm(ex.map(self.run_datapoint, range(self.size)), ...):
            ...append and print a colored "$12 " per item...
    self.report()
```

- Threads (not processes) because the work is **I/O-bound** (waiting on APIs) — contrast with day 1's `ProcessPoolExecutor` for CPU-bound parsing. Default: `size=200` test items, `workers=5`.
- **`report()` draws two charts**: (1) a **scatter of guess vs truth** with a dashed `y = x` line — points on the line are perfect, points below it were underpriced — color-coded by error; (2) an **error-trend chart**: the running average error with a shaded **95% confidence interval** (`1.96 × sd/√n`), so you can literally watch the estimate stabilize as more items are scored. The title becomes e.g. *"Gpt 4.1 Nano results — Error: $62.51 ± $4.02"*.
- Because every model faces the **same 200 test items, same regex post-processing, same metric**, comparisons are fair. One evaluation harness, fourteen contenders.

## `pricer/deep_neural_network.py` — the redemption machine (bonus)

A proper deep-learning architecture, previewed here and unpacked in the redemption section below: `ResidualBlock` (Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm, then **add the input back**: `out += residual`), stacked 8 times with hidden size 4,096 → **289,128,449 parameters**. Its runner handles device selection (cuda/mps/cpu), log-normalized targets, L1 loss, AdamW, cosine learning-rate scheduling, and gradient clipping. Full walkthrough in Part 3.

---

# 🟦 PART 2 — The five days

## 2.1 `day1.ipynb` — Data curation (where the science actually happens)

> Notebook's own claim, and it's true: *"R&D with your dataset can often have a greater impact on performance than the fashionable hyper-parameter optimization."*

1. **Load & explore.** `load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Appliances", split="full")` pulls one category of Amazon product metadata. Cells poke at single rows, and find the most expensive appliance (a $19k commercial oven — with a fun "bargain" link).

2. **Parse everything.** `items = [parse(dp, "Appliances") for dp in tqdm(dataset)]` — the parser silently rejects missing prices, out-of-range prices, and thin descriptions.

3. **Visualize distributions (the heart of curation).** Histograms of **text lengths** and **prices** — you *look* at your data before modeling it. Then print the longest and weirdest items. This habit — plot first, model second — is the real lesson of day 1.

4. **Scale up to 8 categories** via `ItemLoader` (parallel parsing): Appliances, Automotive, Electronics, Office Products, Tools & Home Improvement, Cell Phones, Toys & Games, Musical Instruments → *"a grand total of ~900k items"*.

5. **Deduplicate.** Shuffle with `random.seed(42)`, then a classic Python idiom twice:

```python
seen = set()
items = [x for x in tqdm(items) if not (x.title in seen or seen.add(x.title))]
```

`set.add()` returns `None` (falsy), so this keeps the first occurrence of each title — then again on `full` text. Duplicate listings would leak *identical items* into both train and test, inflating scores (data leakage!).

6. **Weighted sampling to 820,000.** A subtle, clever cell:

```python
p = (prices - min) / (max - min)        # normalize price to 0..1
w = p**2                                # weight = price squared!
w[categories == "Tools_and_Home_Improvement"] *= 0.5
w[categories == "Automotive"] *= 0.05   # Automotive is over-represented: downweight hard
idx = np.random.choice(len(items), size=SIZE, replace=False, p=w)   # w = probability per item
```

Natural price data is a mountain of cheap items and a needle of expensive ones (look at the first histogram). Squaring the price biases the sample toward *expensive* items so the model sees enough of them; downweighting Automotive fixes category imbalance. Then re-plot: the sample is far more balanced than the raw data. **This single cell probably moved every number on the leaderboard.**

7. **Final split & push to the Hub.** First 800,000 → train, next 10,000 → validation, last 10,000 → test; also a "lite" 20k/1k/1k copy; `Item.push_to_hub(...)` uploads both (`ed-donner/items_raw_full`, `items_raw_lite`). From now on, everyone starts from the same data.

Scatterplots of price-vs-length and price-vs-weight close the day: neither correlates strongly — the model will need to actually read the words.

## 2.2 `day2.ipynb` — LLM data pre-processing ("LLMs are great at this!")

Raw Amazon text is ugly ("INSigNIA A3200 BT TOWEL RAIL 600x400 CHROME incl. Fixings, ----, PART# 9981/A…"). **Garbage prompts produce garbage models**, so every product gets rewritten by an LLM into the clean 5-line `Title/Category/Brand/Description/Details` format (that same `SYSTEM_PROMPT` from `preprocessor.py`).

The day demos three ways to run the rewrite at scale, in order of increasing sophistication:

1. **One call at a time** — `completion(model="groq/openai/gpt-oss-20b", reasoning_effort="low")`, printing tokens and cost (fractions of a cent each). Also shown: the same call against **local Ollama** (`model="ollama/llama3.2"`) at $0 — pick your trade-off.
2. **The Batch API, by hand** — build `jsonl/0_1000.jsonl` with `make_jsonl`/`make_file`, upload with `groq.files.create(purpose="batch")`, launch with `groq.batches.create(completion_window="24h")`, poll `batches.retrieve`, download `files.content(...)`, and stitch results back via `custom_id`. Half price for waiting up to 24h.
3. **The packaged `Batch` class** — `Batch.create(items, LITE_MODE)` → `Batch.run()` → `Batch.fetch()` (repeat until all batches are done), then set `item.full = None` (the raw text is no longer needed) and push the **finished dataset** to the Hub: `ed-donner/items_lite` / `items_full`. Every later day simply does `Item.from_hub(...)` — clean summaries and all, **$0**.

After this, an item looks like:

```
Title: Old Blood Noise Excess V2 Distortion Pedal
Category: Musical Instruments
Brand: Old Blood Noise Endeavors
Description: A compact distortion/modulation pedal...
Details: True bypass, 9V power...
```

## 2.3 `day3.ipynb` — The evaluation harness, baselines, traditional ML

Science 101: before celebrating a clever model, you need a **score to beat**. The score (from `evaluator.py`) is the **average absolute error in dollars** on 200 test items.

The models, from dumbest up:

```python
# 1. The floor
def random_pricer(item): return random.randrange(1, 1000)      # pure noise

# 2. The first real baseline
training_average = sum(item.price for item in train) / len(train)
def constant_pricer(item): return training_average             # always ~$113 -> error ~$106

# 3. Linear regression on 3 hand-made features
def get_features(item):
    return {"weight": item.weight,
            "weight_unknown": 1 if item.weight == 0 else 0,
            "text_length": len(item.summary)}
model = LinearRegression().fit(X_train, y_train)               # learns 3 coefficients

# 4. Linear regression on WORDS (bag of words!)
vectorizer = CountVectorizer(max_features=2000, stop_words='english')  # word-count columns
X = vectorizer.fit_transform(documents)                        # rows=items, cols=words
regressor = LinearRegression().fit(X, prices)
def natural_language_linear_regression_pricer(item):
    return max(regressor.predict(vectorizer.transform([item.summary]))[0], 0)  # never negative
```

**Bag of words** deserves a slow read: each product becomes a vector where column *w* = count of word *w*; linear regression learns a price **weight per word** — "premium" pushes the price up, "cable" pushes it down, and the prediction is the weighted sum. No neural network in sight, and it jumps from ~$102 → ~$77 error: **words matter**.

```python
# 5. Random Forest — an ENSEMBLE of decision trees
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4)
rf_model.fit(X[:15_000], prices[:15_000])
# A decision tree is a flowchart of IFs:
#   IF "TV" appears >3 times AND "LED" >2 times AND "HD" ≥1 -> predict $500
# A forest grows 100 such trees on random subsets and AVERAGES them -> less overfitting

# 6. XGBoost — boosting: trees built SEQUENTIALLY, each correcting the previous trees' errors
xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.1, n_jobs=4)
xgb_model.fit(X, prices)     # fast enough for all 800k rows -> error ~$68
```

**Bagging (RF: parallel trees, average) vs boosting (XGBoost: sequential trees, correct)** is the key ensemble distinction — the notebook's business box notes Ed's 15-hour Random Forest run reached $56.40, so *traditional ML is still a contender*.

## 2.4 `day4.ipynb` — A human, a neural network, and the frontier

**(a) The human baseline** — the most charming cell of the week. The first 100 test items are written to **`human_in.csv`** (summary + an empty column); Ed personally filled in guesses → **`human_out.csv`**; then:

```python
def human_pricer(item):
    idx = test.index(item)
    return human_predictions[idx]

evaluate(human_pricer, test, size=100)     # -> $87.62 average error
```

He was beaten by Random Forest, XGBoost, and (later) several LLMs. Humbling and memorable — pricing from text is genuinely hard.

**(b) A neural network from scratch (PyTorch sneak preview)**

```python
vectorizer = HashingVectorizer(n_features=5000, stop_words='english', binary=True)
X = vectorizer.fit_transform(documents)       # one-hot-ish word indicators, 5000 dims

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        self.layer1 = nn.Linear(input_size, 128)   # 5000 -> 128
        self.layer2..7 = 64 -> 64 ...              # a funnel of layers
        self.layer8 = nn.Linear(64, 1)             # -> ONE number: the price
        self.relu = nn.ReLU()                      # the "if positive, keep; else 0" activation
    def forward(self, x):
        output1 = self.relu(self.layer1(x))        # data flows layer1 -> ... -> layer8
        ...
        return self.layer8(output7)
```

Then the **training loop — the four stages you'll see in every framework ever:**

```python
loss_function = nn.MSELoss()                # how wrong? (squared error)
optimizer = optim.Adam(model.parameters(), lr=0.001)   # the update rule

for epoch in range(EPOCHS):                 # EPOCHS=2: two full passes through the data
    for batch_X, batch_y in train_loader:   # batches of 64 items
        optimizer.zero_grad()               # clear previous gradients
        outputs = model(batch_X)            # 1. FORWARD pass: predict
        loss = loss_function(outputs, batch_y)  # 2. LOSS: measure wrongness
        loss.backward()                     # 3. BACKWARD pass: blame each weight (gradients)
        optimizer.step()                    # 4. OPTIMIZE: nudge every weight the right way
```

*"Forward, loss, backward, optimize"* — memorize those four lines and you understand deep learning training. (Epochs, batches, learning rate: an **epoch** is one full pass over the training data; a **batch** is the chunk processed per update; the **learning rate** is step size.) Result: **$63.97** — already competitive with zero-shot GPT-4.1-nano!

**(c) Frontier LLMs, zero-shot** — no training at all, just prompting:

```python
def messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [{"role": "user", "content": message}]

def gpt_5__1(item):
    response = completion(model="gpt-5.1", messages=messages_for(item),
                          reasoning_effort='high', seed=42)
    return response.choices[0].message.content

evaluate(gpt_5__1, test)     # $44.74 — the week's gold standard
```

Same one-liner pattern for Claude Opus 4.5, Gemini 3 Pro, Grok 4.1 Fast, Gemini flash-lite… all routed through LiteLLM, all scored by the same `evaluate`. (The regex in `post_process` earns its keep here — models love replying "This would cost approximately $45.99.")

## 2.5 `day5.ipynb` — Fine-tuning a frontier model (OpenAI API)

Now the headline act: **create your own private variant of GPT-4.1-nano.** Three steps:

**Step 1 — Prepare data (JSONL again).** Each training example pairs the question with the *ideal answer*:

```python
def messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [{"role": "user", "content": message},
            {"role": "assistant", "content": f"${item.price:.2f}"}]   # <- the teaching signal!

def make_jsonl(items):
    result = ""
    for item in items:
        result += '{"messages": ' + json.dumps(messages_for(item)) + '}\n'
    return result.strip()

write_jsonl(fine_tune_train, "jsonl/fine_tune_train.jsonl")        # 100 examples
write_jsonl(fine_tune_validation, "jsonl/fine_tune_validation.jsonl")  # 50 examples

train_file = openai.files.create(file=f, purpose="fine-tune")      # upload both
```

**Step 2 — Launch the job.**

```python
openai.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=validation_file.id,
    model="gpt-4.1-nano-2025-04-14",     # the base model you're customizing
    seed=42,
    hyperparameters={"n_epochs": 1, "batch_size": 1},   # 1 pass, update after every example
    suffix="pricer")                     # your private model gets a name like ft:gpt-4.1-nano:...:pricer
```

Then poll `openai.fine_tuning.jobs.retrieve(job_id)` / `list_events(...)` (or watch platform.openai.com/finetune) until status `succeeded`.

**Step 3 — Use it like any model.**

```python
fine_tuned_model_name = openai.fine_tuning.jobs.retrieve(job_id).fine_tuned_model

def gpt_4__1_nano_fine_tuned(item):
    response = openai.chat.completions.create(model=fine_tuned_model_name,
                                              messages=test_messages_for(item),
                                              max_tokens=7)     # prices are short!
    return response.choices[0].message.content

evaluate(gpt_4__1_nano_fine_tuned, test)
```

**The honest result:** with 100 examples the fine-tuned nano scored ~$75.91 — *worse* than the same model zero-shot ($62.51). Ed's bigger runs (the comments at the bottom of the notebook: `79.29 - mini 2000`, `67.75 - nano 20,000` at a cost of $3.42) still never beat zero-shot frontier. **Why?** This task is about *knowledge* (what things cost), and fine-tuning teaches *behavior/format*, not facts — the exact RAG-vs-fine-tuning line from Part 0, now proven empirically. Note OpenAI itself recommends 50–100 examples; fine-tuning shines when you need consistent style/format/skill, not world knowledge.

---

# 🟧 PART 3 — The "Redemption" bonus: a serious deep neural network

The fine-tuning disappointment needed answering, and `redemption_train.ipynb` / `redemption_run.ipynb` (powered by `pricer/deep_neural_network.py`) deliver it: **a from-scratch PyTorch network scoring $46.49 — within $2 of GPT-5.1.**

### The upgrades over day 4's vanilla network — each one is a standard deep-learning tool

| Upgrade | Code | What it buys |
|---|---|---|
| **Residual ("skip") connections** | `out += residual` in `ResidualBlock` | Each block learns only a *correction* to its input, so gradients flow through many layers instead of vanishing — the idea that made 100+ layer networks trainable (ResNet, and inside every Transformer) |
| **Much wider & deeper** | 8 blocks × hidden size 4,096 → **289M parameters** | Capacity to actually absorb 800k examples (vs the vanilla net's ~330k) |
| **LayerNorm + Dropout** | `nn.LayerNorm`, `nn.Dropout(0.2)` | Stabilizes activations between layers; randomly drops 20% of neurons each step so the network can't over-rely on any single path (regularization) |
| **Predict `log(price)`, not price** | `y_train_log = torch.log(y + 1)` then normalize to mean 0/std 1; invert at inference with `torch.exp(pred * std + mean) - 1` | A $5 error on a $10 item hurts as much as a $50 error on a $100 item — relative errors instead of absolute; plus normalized targets are far easier to optimize |
| **L1 loss** | `nn.L1Loss()` (= mean absolute error) | Optimizes *directly* for the metric we're judged on (average dollars off), and is less spooked by outliers than squared error |
| **AdamW + cosine LR schedule + gradient clipping** | `optim.AdamW(lr=0.001, weight_decay=0.01)`, `CosineAnnealingLR(T_max=10)`, `clip_grad_norm_(max_norm=1.0)` | Modern optimizer with decoupled weight decay; learning rate glides down from 0.001 toward 0 on a cosine curve (big careful steps early, fine steps late); clipping stops rare exploding gradients from derailing training |
| **Device + seeds** | cuda / mps / cpu auto-pick; seeds everywhere | Runs on any GPU; reproducibility |

`redemption_train.ipynb` runs `runner.setup()` → `runner.train(epochs=5)` (~4 hours on an M1 Mac GPU) and you can watch the validation MAE walk down: $58.26 → $56.25 → $56.55 → $55.70 → **$53.15**, then `runner.save('deep_neural_network.pth')` (PyTorch's binary weights format). `redemption_run.ipynb` skips the wait: download Ed's `deep_neural_network.pth`, `runner.load('deep_neural_network.pth', 'cpu')`, and `evaluate(deep_neural_network, test)` → **$46.49**.

The meta-lesson is the one the course keeps hammering: **the fancy API trick (fine-tuning) lost to disciplined classical ML + neural network craft.** Technique choice beats novelty.

---

# 🎓 PART 4 — Consolidation

## Concepts introduced, week by day

| Concept | Where | One-liner |
|---|---|---|
| Data curation & leakage prevention | D1 | Filter, scrub part numbers, dedupe — before any modeling |
| Plot-before-you-model | D1 | Histograms of price/length reveal everything |
| **Weighted sampling** | D1 | `np.random.choice(p=w)` with price² weights fixes class imbalance |
| train / val / test splits | D1, all | Train to learn, val to tune, test to judge — never mix |
| HF Hub as dataset storage | D1–5 | `Item.push_to_hub` / `from_hub` |
| **Batch API** | D2 | Half-price async bulk inference; JSONL in, JSONL out |
| LLM data pre-processing | D2 | LLMs cleaning data for other models |
| Evaluation harness | D3 | One `evaluate()` for every model; post-process outputs with regex |
| Baselines first | D3 | Random & constant pricers define "worth it" |
| **Bag of words** | D3 | Word counts as features; per-word price weights |
| Ensembles: bagging vs boosting | D3 | Random Forest averages trees; XGBoost stacks corrections |
| **The 4-stage training loop** | D4 | forward → loss → backward → optimize |
| Epochs, batches, learning rate | D4 | Full passes, chunk sizes, step size |
| MSELoss vs L1Loss | D4/bonus | Squared vs absolute error |
| Human baseline | D4 | Ed: $87.62 — machines won |
| Zero-shot frontier prompting | D4 | No training; GPT-5.1 = $44.74 |
| **Fine-tuning via API** | D5 | JSONL examples → `fine_tuning.jobs.create` → private model |
| Fine-tuning's real scope | D5 | Teaches format/skill, not facts (the disappointing result) |
| Residual networks, LayerNorm, Dropout | bonus | The modern deep-learning toolkit |
| log-targets, AdamW, cosine LR, grad clipping | bonus | How professionals actually train |

## The week's arc as a diagram

```
Amazon-Reviews-2023 (HF)
   │ parse + scrub + filter ($0.50-$999, ≥600 chars)      pricer/parser.py
   │ dedupe by title & text; weighted sample (price², downweight Automotive)
   ▼
820k Items → train 800k / val 10k / test 10k  (+ 20k lite)  → pushed to HF Hub
   │ LLM rewrite of every product (groq gpt-oss-20b, Batch API, ~50% cost)
   ▼
clean summaries ─────────────┬──────────────────────────────────────────┐
                             ▼                                          ▼
              pricer/evaluator.py: evaluate(model, test, size=200) ──► avg $ error
                             │
   random $889 │ constant $106 │ lin-reg $102 │ words+LR $77 │ RF $72 │ XGB $68
   HUMAN $88   │ vanilla NN $64 │ nano-ft $76 │ nano $63    │ grok $58
   gemini $51  │ claude $47     │ DNN $46.5   │ GPT-5.1 $44.7 ◄── winner
```

## Glossary additions

- **Fine-tuning** — continued training of an existing model on your examples, producing a private variant.
- **Epoch / batch / learning rate** — one pass over the data / chunk per update / step size for weight updates.
- **Loss function** — the number measuring "how wrong" that training minimizes (MSE = squared; L1 = absolute).
- **Backpropagation ("backward pass")** — computing each weight's share of the blame (gradient).
- **Adam / AdamW** — adaptive optimizers that tune per-weight step sizes (W adds proper weight decay).
- **Cosine annealing / gradient clipping** — LR that decays on a cosine curve / cap on gradient size.
- **Residual connection** — adding a layer's input to its output so it learns only a correction.
- **LayerNorm / Dropout** — normalize activations between layers / randomly silence neurons to prevent overfitting.
- **Bag of words / CountVectorizer / HashingVectorizer** — text-as-word-count features (hashing: no vocabulary needed).
- **Decision tree / Random Forest / XGBoost** — flowchart model / averaged committee of trees / sequential error-correcting trees.
- **Overfitting** — memorizing training data at the expense of generalizing (why we hold out test data).
- **Data leakage** — test information sneaking into training (why we dedupe!).
- **JSONL** — one JSON object per line; the format of bulk APIs and fine-tune files.
- **Batch API** — asynchronous half-price inference submitted as a file of requests.
- **`.pth`** — PyTorch saved-weights file.

## Running it yourself

1. `uv sync` from the repo root; put `OPENAI_API_KEY`, `HF_TOKEN` (with write access), and optionally `GROQ_API_KEY` in `.env`.
2. **Free path (recommended):** skip days 1–2's heavy lifting entirely — from day 3 onward everything loads `ed-donner/items_lite` (set `LITE_MODE = True`). Run day3 → day4 → day5 in order.
3. **Full path:** day 1 takes ~30+ min of parallel parsing per category set; day 2 costs ~$1 (lite) / ~$30 (full) via Groq's Batch API; day 5 fine-tuning on 100 examples costs cents (~$3.42 for the 20k-example run).
4. Every model is scored with `evaluate(pricer_function, test)` — make your own pricer function and add it to `results.ipynb`'s list. Beating $44.74 is the standing challenge.
5. `LITE_MODE` toggles at the top of each notebook; the redemption DNN needs either ~4 GPU-hours (train) or the downloaded `deep_neural_network.pth` (run).
6. Gotchas from the labs: if `load_dataset` complains about `trust_remote_code`, run `!uv add --upgrade datasets==3.6.0` and restart the kernel; XGBoost on Mac may need `brew install libomp`.

## Where this leaves the course

You've now done the full ML lifecycle on real, messy data — curate, preprocess, baseline, model, evaluate, and *interpret honest results* (including the ones that disappoint). Week 7 goes deeper into the "training" half of the course — building and training a *from-scratch* transformer/open-source model pipeline (and revisiting this same pricer with more powerful techniques), and Week 8 assembles the autonomous multi-agent capstone that ties everything — tools, RAG, local models, fine-tuning — together.
