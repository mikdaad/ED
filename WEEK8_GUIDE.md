# 📘 Week 8 — Complete Beginner's Guide (The Finale: Autonomous Agentic AI)

This guide explains **every file** in `week8/` — the final week of the *LLM Engineering* course — what each piece does, why it exists, and how the whole system fits together. (Companions: Weeks 1–2, 3–4, 5, 6, 7 guides.)

> **The one-sentence summary:** Everything you built converges into **"The Price Is Right" — a team of collaborating AI agents** that scans real deal websites, estimates what products are *truly* worth (using your fine-tuned model from Week 7 running on serverless GPUs, plus a RAG-powered frontier model, plus a deep neural network), picks the single best bargain, and **pushes a notification to your phone** — autonomously, on a schedule, through a live dashboard.

> **What "agentic" finally means here:** Weeks 1–5, the *you* wrote the workflow (scrape → prompt → call). Week 8 Day 4 hands the workflow itself to the LLM: the model decides which tools to call, in what order, looping until the job is done. That's an **autonomous agent** — and you'll build it in ~40 lines, plus a production framework around it.

---

# 🧠 PART 0 — The Big Picture

## The mission

> Find genuine bargains on the internet. A *bargain* = a product offered **far below its true value**. Humans do this with intuition built from experience; the course builds it from three machine "experts" + a planner that coordinates them.

## The architecture (the whole week on one diagram)

```
                        ┌──────────────────────────────────────────────────┐
                        │            AUTONOMOUS PLANNING AGENT (gpt-5.1)   │
                        │   decides which tool to call next, in a loop     │
                        └───────┬──────────────────┬──────────────────┬────┘
                    tool 1      │        tool 2    │         tool 3   │
                                ▼                  ▼                  ▼
                    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
   RSS feeds        │  SCANNER AGENT   │ │ ENSEMBLE AGENT   │ │ MESSAGING AGENT  │
   (dealnews.com)──►│ scrapes deals,   │ │ 3 experts vote:  │ │ Claude writes a  │──► 📱 Pushover
                    │ gpt-5-mini picks │ │ FrontierAgent    │ │ catchy alert,    │    push notification
                    │ the best 5       │ │ SpecialistAgent  │ │ Pushover sends   │
                    │ (structured      │ │ NeuralNetwork-   │ │ it to your phone │
                    │  outputs JSON)   │ │ Agent (weighted  │ └──────────────────┘
                    └──────────────────┘ │ avg of the 3)    │
                                         └───────┬──────────┘
                       ┌─────────────────────────┼─────────────────────────┐
                       ▼                         ▼                         ▼
              FrontierAgent (RAG)      SpecialistAgent            NeuralNetworkAgent
              Chroma over 400k         your Week-7 QLoRA          Week-6 residual DNN
              Amazon products →        model on Modal.com         (local .pth weights)
              5 similar products       (serverless GPU)           (local .pth weights)
              + gpt-5.1 estimates      (serverless GPU)
```

## The week's "Order of play"

```
DAY 1  Modal.com serverless + SpecialistAgent   (your fine-tuned model becomes a web service)
DAY 2  RAG + FrontierAgent + EnsembleAgent      (inference-time techniques: the new champions)
DAY 3  ScannerAgent + MessagingAgent            (RSS scraping + phone notifications)
DAY 4  PlanningAgent → AutonomousPlanningAgent  (deterministic workflow vs LLM-driven agent)
DAY 5  The Gradio product + memory              (live dashboard, persistent state, finale)
```

## The final leaderboard (`results.ipynb`) — two new champions

Week 8 adds two entries to the running 17-model contest:

| Model | Error | Technique family |
|---|---|---|
| (weeks 6–7 entrants: constant $106 → fine-tuned-full $39.85…) | | training-time |
| **GPT-5.1 + RAG** (5 similar products in the prompt) | **$30.19** | 🟢 inference-time |
| 🥇 **Ensemble** (0.8·RAG + 0.1·fine-tuned + 0.1·DNN) | **$29.90** | 🟣 all three families |

The course's closing technical lesson: **you don't have to choose between training and prompting — combine them.** RAG on 800k real products beat everything from Weeks 6–7, and blending three independent experts beat RAG. (Ensembles work because models make *different* mistakes; averaging cancels them.)

## What's in the folder

```
day1.ipynb … day5.ipynb        The five labs
results.ipynb                  The final 19-model leaderboard
agents/                        The reusable multi-agent package (13 files)
deal_agent_framework.py        Orchestration + persistent memory (memory.json)
price_is_right.py              THE PRODUCT: live Gradio dashboard
pricer_ephemeral.py            Modal test: call your fine-tuned model ad-hoc
pricer_service.py / 2.py       Modal DEPLOYED services (function / class versions)
hello.py / llama.py            Modal 101: your first two cloud functions
log_utils.py                   ANSI log colors → HTML for the dashboard
memory.json                    The framework's persistent memory (2 example deals)
```

---

# 🟦 PART 1 — Day 1: Modal.com + the SpecialistAgent

## 1.1 The problem & the solution

Your Week 7 model lives in a Colab/GPU you no longer have. To use it in an app you need **GPU inference as a service**. [Modal](https://modal.com) is **serverless cloud infrastructure defined entirely in Python**: you decorate functions, deploy, and Modal builds the container, boots a GPU on demand, runs your code, and bills by the second (~$0.000164/s for a T4). No Docker, no DevOps.

### `hello.py` — your first cloud function

```python
app = modal.App("hello")
image = Image.debian_slim().pip_install("requests")     # the container recipe

@app.function(image=image)
def hello() -> str:
    import requests
    response = requests.get("https://ipinfo.io/json")   # where am I running?
    return f"Hello from {city}, {region}, {country}!!"
```

The day-1 notebook then shows the two calling modes:

```python
with app.run():
    hello.local()      # runs on YOUR machine
    hello.remote()     # Modal spins up a container in the cloud and runs it there
                       # → "Hello from Ashburn, Virginia, US!!"
```

`hello_europe` adds `region="eu"` to the decorator — your code, pinned to a geography. (Setup: create a Modal account, run `uv run modal token set --token-id ak-… --token-secret as-…`; troubleshooting tips are in the notebook, including the Windows token-file quirk.)

### `llama.py` — a GPU in the cloud

```python
image = Image.debian_slim().pip_install("torch", "transformers", "accelerate")
secrets = [modal.Secret.from_name("huggingface-secret")]   # HF token, injected as env var
GPU = "T4"
MODEL_NAME = "meta-llama/Llama-3.2-3B"

@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def generate(prompt: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, max_new_tokens=5)
    return tokenizer.decode(outputs[0])
```

This is Week 3 running in the cloud: imports *inside* the function (the container doesn't have them at build time), the HF secret injected, `device_map="auto"` onto the T4. Setup recipe (from the notebook): create a Modal secret named **huggingface-secret** with key `HF_TOKEN` = your token.

### `pricer_ephemeral.py` — the Week 7 model, ad-hoc

Same pattern, but the function loads the **QLoRA base + adapter** (exactly Week 7's inference stack: 4-bit `BitsAndBytesConfig`, `PeftModel.from_pretrained(..., revision=...)`), builds the prompt `"...Price is $"`, generates **5 tokens**, and regex-extracts the number. "Ephemeral" = created on-the-fly when called, destroyed after (~a minute of cold-start each time).

### `pricer_service.py` → `pricer_service2.py` — deployment

```bash
uv run modal deploy -m pricer_service    # (or modal deploy -m pricer_service2)
```

Deployment keeps the app resident in Modal's cloud so calls are fast. Version 1 exposes a plain function: `modal.Function.from_name("pricer-service", "price").remote(text)`. **Version 2 is the production pattern** — a class whose expensive setup happens ONCE:

```python
hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)  # persistent disk!

@app.cls(image=image.env({"HF_HUB_CACHE": CACHE_DIR}), secrets=secrets, gpu=GPU,
         min_containers=MIN_CONTAINERS, volumes={CACHE_DIR: hf_cache_volume})
class Pricer:
    @modal.enter()
    def setup(self):
        ...load tokenizer + 4-bit base + adapter ONCE, keep on self...

    @modal.method()
    def price(self, description: str) -> float:
        ...generate 5 tokens, extract number...   # reuses the loaded model!
```

Three production lessons in one file: **`@modal.enter()`** = load the model once per container, not per call; **`Volume`** = a persistent disk so the 6GB model download survives restarts; **`MIN_CONTAINERS`** = keep a warm container (2s responses, but it burns credits — the notebook explains `update_autoscaler(scaledown_window=1200)` for a 20-minute warm window as a middle ground).

## 1.2 `agents/agent.py` + `specialist_agent.py` — the Agent pattern begins

```python
class Agent:
    RED/GREEN/YELLOW/BLUE/MAGENTA/CYAN/WHITE = '\033[3xm'   # ANSI color codes
    BG_BLACK = '\033[40m'; RESET = '\033[0m'
    name: str = ""; color: str = ...

    def log(self, message):
        message = f"[{self.name}] {message}"
        logging.info(self.BG_BLACK + self.color + message + self.RESET)
```

A tiny abstract superclass whose entire job is **color-coded logging** — when ten agents run concurrently you can *see* who's talking (the dashboard later converts these ANSI codes to HTML via `log_utils.reformat`, e.g. `BG_BLACK+RED → <span style="color:#dd0000">`).

```python
class SpecialistAgent(Agent):
    """An Agent that runs our fine-tuned LLM that's running remotely on Modal"""
    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")   # connect to the deployed class
        self.pricer = Pricer()

    def price(self, description: str) -> float:
        result = self.pricer.price.remote(description)             # serverless call
        return result
```

**Every agent this week shares the same interface: `price(description) -> float`.** That uniformity is what makes the Ensemble (and the planner) possible.

---

# 🟩 PART 2 — Day 2: RAG, the FrontierAgent, and the Ensemble

## 2.1 The vector store (`day2.ipynb`)

Week 5's RAG recipe, at 20× scale and repurposed: instead of *answering questions*, the retrieved context helps *estimate prices* — "here are 5 similar products and what they sold for."

```python
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')   # free, local, 384-dim
client = chromadb.PersistentClient(path=DB)                              # DB = "products_vectorstore"

# embed 400,000 training products in batches of 1,000 (≈30 min on a GPU):
documents = [item.summary for item in train[i:i+1000]]
vectors   = encoder.encode(documents).astype(float).tolist()
metadatas = [{"category": item.category, "price": item.price} for item in train[i:i+1000]]
collection.add(ids=..., documents=documents, embeddings=vectors, metadatas=metadatas)
```

**The key design choice:** the product **price is stored in the metadata**. Retrieval returns documents *and their known prices* — turning the vector store into a "what have comparable things sold for?" engine. The notebook re-runs the familiar t-SNE 2D/3D plots (8 categories, colored — Week 5 déjà vu) and then does the money cell:

```python
def find_similars(item):
    vec = encoder.encode(item.summary)
    results = collection.query(query_embeddings=vec.astype(float).tolist(), n_results=5)
    return results['documents'][0], [m['price'] for m in results['metadatas'][0]]

def make_context(similars, prices):
    message = "For context, here are some other items that might be similar...\n\n"
    for similar, price in zip(similars, prices):
        message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
    return message
```

…then calls **GPT-5.1 with that context** and evaluates: `evaluate(gpt_5__1_rag, test)` → **$30.19**, the best single model of the course. The fine-tuned model needed a GPU-week; the RAG agent needed only a smarter prompt.

## 2.2 `agents/frontier_agent.py` — the RAG expert, productized

Same logic wrapped as an Agent (its `price()` = encode → query 5 similars → build context → call `gpt-5.1` with `seed=42, reasoning_effort="none"` → regex-extract the price). Fully commented, one log line per step.

## 2.3 `agents/neural_network_agent.py` — the third expert

Week 6's residual DNN, refactored for pure inference: `agents/deep_neural_network.py` keeps the architecture but **hardcodes the normalization constants** (`Y_STD = 1.0328…, Y_MEAN = 4.4349…` — the values training computed) and drops all optimizer/scheduler/training machinery. Lesson: *deployment strips a model down to forward passes.* You must download `deep_neural_network.pth` (link in the notebook) next to the code.

## 2.4 `agents/ensemble_agent.py` — three experts, one verdict

```python
class EnsembleAgent(Agent):
    def __init__(self, collection):
        self.specialist = SpecialistAgent()          # fine-tuned QLoRA on Modal
        self.frontier = FrontierAgent(collection)    # GPT-5.1 + RAG
        self.neural_network = NeuralNetworkAgent()   # Week 6 DNN, local weights
        self.preprocessor = Preprocessor()           # rewrite messy input first!

    def price(self, description: str) -> float:
        rewrite = self.preprocessor.preprocess(description)   # normalize to the standard format
        specialist = self.specialist.price(rewrite)
        frontier = self.frontier.price(rewrite)
        neural_network = self.neural_network.price(rewrite)
        return frontier * 0.8 + specialist * 0.1 + neural_network * 0.1
```

Two ideas: **preprocessing** (real-world deal text is messy; the Week 6 `Preprocessor` cleans it so all three models see the format they were trained/prompted on — default model configurable via `PRICER_PREPROCESSOR_MODEL`, defaulting to local `ollama/llama3.2`) and **weighted averaging** (weights chosen from the eval numbers: RAG is far more accurate, so it gets 80%). This is machine-learning's oldest superpower — *combining weak signals into a strong one* — now applied across an LLM, a fine-tuned LLM, and a neural network. Result: **$29.90, champion.**

---

# 🟨 PART 3 — Day 3: ScannerAgent + MessagingAgent

## 3.1 `agents/deals.py` — scraping real deals from RSS

```python
feeds = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]

class ScrapedDeal:
    def __init__(self, entry):                    # an RSS <entry>
        self.title = entry["title"]
        self.summary = extract(entry["summary"])  # BeautifulSoup cleanup (Week 1 skills!)
        self.url = entry["links"][0]["href"]
        stuff = requests.get(self.url)            # follow the link, scrape the full page
        ...
        self.truncate()                           # title≤100, details≤500, features≤500 chars

    def describe(self):
        return f"Title: {self.title}\nDetails: ...\nFeatures: ...\nURL: {self.url}"

    @classmethod
    def fetch(cls, show_progress=False) -> List[Self]:
        for feed_url in feeds:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                deals.append(cls(entry)); time.sleep(0.05)   # be polite
```

**RSS** = the structured "new posts" feed every news site publishes; `feedparser` reads it. Note the truncation — a politeness budget for the model's context window.

## 3.2 Structured Outputs, properly this time

The scanner's job: from ~30 raw deals, pick the **5 best** (clear product, unambiguous price) and summarize each — *watching out for the "$100 off" trap* (discount vs actual price). This is Week 8's upgrade of Week 1 Day 5's JSON mode — **Structured Outputs with a Pydantic schema**:

```python
class Deal(BaseModel):
    product_description: str = Field(description="Your clearly expressed summary ... focus on the item itself")
    price: float = Field(description="The actual price ... if a deal is $100 off the usual $300, respond with $200")
    url: str

class DealSelection(BaseModel):
    deals: List[Deal] = Field(description="Your selection of the 5 deals ...")

response = openai.chat.completions.parse(          # note: .parse, not .create!
    model="gpt-5-mini", messages=messages,
    response_format=DealSelection,                 # the schema IS the contract
    reasoning_effort="minimal")
result = response.choices[0].message.parsed        # a real DealSelection object — no json.loads!
result.deals = [deal for deal in result.deals if deal.price > 0]
```

The `Field(description=...)` strings are prompt engineering living inside the schema — the API guarantees the output *conforms* (the model is trained to emit exactly this structure; the SDK validates it). Week 5 used `response_format=Chunks` via LiteLLM; here it's OpenAI's native `.parse()`.

## 3.3 `agents/scanner_agent.py`

Wraps the above: `fetch_deals(memory)` filters out URLs already surfaced (memory = list of seen URLs — no repeat alerts!), `scan(memory)` returns a `DealSelection` or `None`, and `test_scan()` returns a hardcoded 4-deal selection used for offline testing (no scraping while developing — a genuinely good engineering habit).

## 3.4 `agents/messaging_agent.py` — the phone call

[Pushover](https://pushover.net) sends push notifications to your phone via a simple HTTP POST (free account; `PUSHOVER_USER` + `PUSHOVER_TOKEN` into `.env`):

```python
def push(self, text):
    payload = {"user": self.pushover_user, "token": self.pushover_token,
               "message": text, "sound": "cashregister"}       # 💰 the sound of savings
    requests.post("https://api.pushover.net/1/messages.json", data=payload)

def notify(self, description, deal_price, estimated_true_value, url):
    text = self.craft_message(description, deal_price, estimated_true_value)  # Claude writes it!
    self.push(text[:200] + "... " + url)
```

`craft_message` asks **Claude Sonnet 4.5** to write a 2–3-sentence, excitement-generating alert — an LLM call whose entire job is marketing copy for your deal.

---

# 🟪 PART 4 — Day 4: from workflow → to autonomous agent (the heart of the week)

## 4.1 First, the workflow — `agents/planning_agent.py` (deterministic)

```python
class PlanningAgent(Agent):
    DEAL_THRESHOLD = 50                    # only alert if discount > $50

    def __init__(self, collection):
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()

    def run(self, deal: Deal) -> Opportunity:
        estimate = self.ensemble.price(deal.product_description)
        discount = estimate - deal.price
        return Opportunity(deal=deal, estimate=estimate, discount=discount)   # ← from deals.py

    def plan(self, memory=[]):
        selection = self.scanner.scan(memory=memory)                # 1. find deals
        opportunities = [self.run(d) for d in selection.deals[:5]]  # 2. price them
        opportunities.sort(key=lambda opp: opp.discount, reverse=True)
        best = opportunities[0]                                     # 3. pick the best
        if best.discount > self.DEAL_THRESHOLD:
            self.messenger.alert(best)                              # 4. notify!
        return best if best.discount > self.DEAL_THRESHOLD else None
```

Solid, but **the human wrote the workflow**. Day 4's question: what if the *model* wrote it?

## 4.2 The rehearsal — fake tools in `day4.ipynb`

Before unleashing real (slow, costly) machinery, the notebook builds the agent loop with three **fake functions** (hardcoded deals, "$300 always", print-notifications) described to the model with the Week 2 Day 4 tool-JSON pattern, plus a `handle_tool_call` with a slick trick:

```python
def handle_tool_call(message):
    results = []
    for tool_call in message.tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        tool = globals().get(tool_name)        # ← look up the PYTHON FUNCTION BY NAME
        result = tool(**arguments) if tool else {}
        results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
    return results
```

Then **the agent loop** — the course's most important 10 lines:

```python
done = False
while not done:
    response = openai.chat.completions.create(model="gpt-5.1", messages=messages, tools=tools)
    if response.choices[0].finish_reason == "tool_calls":   # the model wants to DO something
        message = response.choices[0].message
        results = handle_tool_call(message)                 # we do it
        messages.append(message)                            # remember the request
        messages.extend(results)                            # remember the result
    else:
        done = True                                         # the model wrote its final answer
```

The user prompt gives the *goal*, not the steps: *"First scan… then for each deal estimate… pick the single most compelling deal… notify… then reply OK."* GPT-5.1 chains 10+ tool calls on its own — scan → estimate ×4 → notify → "OK". **This `while` loop is the same one from Week 2 Day 4, now the spine of an autonomous system.**

## 4.3 The real thing — `agents/autonomous_planning_agent.py`

Same loop; the fake functions become real methods wired to the real agents:

```python
class AutonomousPlanningAgent(Agent):
    MODEL = "gpt-5.1"

    def scan_the_internet_for_bargains(self) -> str:
        results = self.scanner.scan(memory=self.memory)
        return results.model_dump_json() if results else "No deals found"

    def estimate_true_value(self, description: str) -> str:
        estimate = self.ensemble.price(description)      # ← the whole 3-expert ensemble!
        return f"The estimated true value of {description} is {estimate}"

    def notify_user_of_deal(self, description, deal_price, estimated_true_value, url) -> Dict:
        if self.opportunity:
            self.log("...trying to notify the user a 2nd time; ignoring")   # guard rail!
        else:
            self.messenger.notify(...)
            self.opportunity = Opportunity(deal=Deal(...), estimate=..., discount=...)
        return "Notification sent ok"
```

Production touches worth noticing: a **guard rail** so the LLM can't spam notifications; `self.opportunity` captures the result for the caller even though the LLM drives; `handle_tool_call` uses an explicit name→method `mapping` (safer than the notebook's `globals()` trick); `plan()` seeds `self.memory` (seen URLs) into the scanner and runs the `while` loop until the model replies "OK". `day4.ipynb` ends by simply calling `agent.plan()` — and a real push notification arrives on the phone.

---

# 🟧 PART 5 — Day 5: the product

## 5.1 `day5.ipynb` — UI progressive disclosure

Four Gradio builds: (1) static header → (2) `gr.State` holding `Opportunity` objects + `gr.Dataframe` of deals → (3) `opportunities_dataframe.select(do_select, ...)` — **click a row → get the push notification again** → (4) hand over to the real code. (`gr.State` = data that survives between UI callbacks; `gr.SelectData` = which row you clicked.)

## 5.2 `deal_agent_framework.py` — orchestration + memory

```python
class DealAgentFramework:
    DB = "products_vectorstore"
    MEMORY_FILENAME = "memory.json"

    def read_memory(self):    # load past opportunities from disk → List[Opportunity]
    def write_memory(self):   # save them back
    @classmethod
    def reset_memory(cls):    # keep only the first 2 (a demo reset)

    def run(self) -> List[Opportunity]:
        self.init_agents_as_needed()                  # lazy init (agents are expensive to construct)
        result = self.planner.plan(memory=self.memory)
        if result:
            self.memory.append(result)                # remember what we alerted
            self.write_memory()                       # persist!
        return self.memory
```

**Memory** is the agentic loop's missing piece: `memory.json` (ship with 2 example deals — a Galaxy Watch Ultra estimated at $773.81 vs $350 asking = **$423.81 discount**, and an iPhone 14 Pro at $225.88 off) means restarts don't re-alert old deals, and the UI can show history immediately.

## 5.3 `price_is_right.py` — the dashboard (concurrency done gently)

The final product, `python price_is_right.py` — a Gradio Blocks app with: the **deals table**, a **live log panel**, and a **3D t-SNE plot** of the vector store (with a rotation camera). The interesting engineering is how a *slow, chatty* agent run stays a *responsive* UI:

```python
class QueueHandler(logging.Handler):        # capture every agent's colored log line...
    def emit(self, record): self.log_queue.put(self.format(record))

def run_with_logging(initial_log_data):
    log_queue = queue.Queue(); result_queue = queue.Queue()
    worker = threading.Thread(target=lambda: result_queue.put(do_run()))
    worker.start()                                    # ← agents run in a BACKGROUND THREAD
    for log_data, output, final in update_output(initial_log_data, log_queue, result_queue):
        yield log_data, output, final                 # ← UI updates stream back (generators, W2D2!)

def update_output(...):   # a generator: drain log queue → update HTML; check result queue → update table
    while True:
        try: message = log_queue.get_nowait(); yield ...
        except queue.Empty:
            ... if final_result is not None: break
            time.sleep(0.1)

timer = gr.Timer(value=300, active=True)              # every 5 minutes...
timer.tick(run_with_logging, ...)                     # ...run the whole agent framework again
```

The pattern: **worker thread** (the slow part) + **queues** (thread-safe hand-off) + **generator** (streaming updates into Gradio — the Week 2 `yield` pattern matured). `log_utils.reformat` converts agent ANSI colors into `<span style="color:...">` HTML so the log panel is as colorful as the terminal. Click a row → `do_select` → `messenger.alert(opportunity)` → your phone buzzes.

---

# 🎓 PART 6 — Consolidation & course wrap-up

## Concepts introduced this week

| Concept | Where | One-liner |
|---|---|---|
| **Serverless GPU inference (Modal)** | D1 | Python decorators → deployed GPU web services, no DevOps |
| `@app.function` / `@app.cls` / `@modal.enter()` | D1 | Ephemeral calls / class services / load-once setup |
| Modal `Secret` / `Volume` / `min_containers` | D1 | Secrets injection / persistent disk / keep-warm trade-off |
| RAG for estimation (not Q&A) | D2 | Prices in metadata → "what did similar items sell for?" |
| **Ensemble / weighted blending** | D2 | `0.8·RAG + 0.1·specialist + 0.1·DNN` = course champion $29.90 |
| Inference-only model refactor | D2 | Hardcode y_mean/y_std; strip training machinery |
| RSS scraping + politeness budgets | D3 | `feedparser` + truncation before prompting |
| **Structured Outputs (`.parse` + Pydantic)** | D3 | The schema *is* the contract; `Field(description=...)` steers the model |
| Push notifications | D3 | A 3-line HTTP POST to Pushover; Claude writes the copy |
| Workflow agent vs **autonomous agent** | D4 | Hand-written pipeline vs the `while finish_reason=="tool_calls"` loop |
| Tool call guards & result capture | D4 | "Notify only once"; capture `self.opportunity` for callers |
| Lazy init & agent memory | D5 | Construct agents on first use; `memory.json` = no repeat alerts |
| Threads + queues + generators | D5 | Slow agent work off-thread; UI streams updates live |
| `gr.Timer` | D5 | The agent re-runs itself every 5 minutes — a true daemon |

## The 8-week arc, complete

| Week | Superpower | Artifact |
|---|---|---|
| 1 | Call an LLM; prompts; tokens; statelessness | Website summarizer, brochure generator |
| 2 | Many providers; Gradio UIs; chat history; **tools** | Airline assistant with image+voice |
| 3 | Pipelines, tokenizers, transformers, quantization | Meeting-minutes from audio |
| 4 | LLMs write & execute compiled code; benchmarks | Python→C++/Rust racer |
| 5 | **RAG**: chunks, vectors, retrieval, evaluation | Insurellm expert assistant + eval dashboard |
| 6 | Fine-tuning via API; classic ML; honest evaluation | The Price Is Right, round 1 |
| 7 | **QLoRA** on open-source models | Your own Llama, champion of round 2 |
| 8 | **Autonomous multi-agent systems** | The deal-hunting swarm |

The course's technical thesis, now proven three ways: **prompting (W8 RAG $30.19) ≈ beats training (W7 QLoRA $39.85) ≈ beats naive API use — and blending everything ($29.90) beats each alone.** Choice of *technique for the task* is the LLM engineer's real skill.

## Glossary additions

- **Agentic AI** — LLMs that decide and execute multi-step workflows via tools, looping until done.
- **Serverless** — cloud functions that run on demand, billed per second, scaled automatically.
- **Ensemble** — combining multiple models' outputs (here, a weighted average).
- **RSS feed** — a site's machine-readable list of new posts; the scanner's data source.
- **Structured Outputs** — API-guaranteed conformance to a schema (Pydantic `Field`s steer the model).
- **Cold start / warm container** — first-call latency while a service loads its model / a kept-alive instance.
- **`gr.State` / `gr.Timer` / `gr.SelectData`** — Gradio's session state / scheduled ticks / click metadata.
- **ANSI escape codes** — terminal color codes; `log_utils` maps them to HTML spans.

## Running it yourself

1. **Accounts/keys in `.env`:** `OPENAI_API_KEY`, `HF_TOKEN`, plus **Modal** (`uv run modal token set …`) and **Pushover** (`PUSHOVER_USER`, `PUSHOVER_TOKEN`). Create the Modal secret named **huggingface-secret** (key `HF_TOKEN`).
2. **Day 1:** run the notebook; deploy your pricer — `uv run modal deploy -m pricer_service2` (first build ~10 min; then ~2s warm). To use *your* Week 7 model, change `HF_USER`/`RUN_NAME`/`REVISION` in `pricer_service2.py`; to use Ed's, leave as-is.
3. **Day 2:** builds `products_vectorstore/` (30 min GPU for 400k items — or use `LITE_MODE`). Download `deep_neural_network.pth` (link in the notebook) into `week8/`.
4. **Days 3–5:** run the notebooks, then `uv run price_is_right.py` → the dashboard opens, agents chatter in color, and within minutes your phone gets a real deal alert (only if discount > $50).
5. **Costs:** Modal pennies per pricing call; GPT-5.1/GPT-5-mini calls a few cents per run; everything else local/free.
6. Troubleshooting gold in the notebooks: Windows token-file relocation, `hf-secret` vs `huggingface-secret` naming, `PYTHONIOENCODING=utf-8` for emoji logs, and `DealAgentFramework.reset_memory()` for a clean slate.

**🎉 That's the whole course.** You've gone from your first `messages=[...]` call to a distributed, multi-model, autonomous system — with evaluation discipline at every step. Ed's closing note points onward: his **tech2ai** repo rebuilds the Week 8 planner with the OpenAI Agents SDK + MCP servers, and the guides folder + community-contributions are full of next steps. *You could not have picked a better time to be in this field.*
