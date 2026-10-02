# 🤖 LLM ENGINEERING BRIEF — Context File for Coding AI Agents

> **For the human:** give this file to your coding agent as standing context — paste it into your rules file (`.cursorrules`, `CLAUDE.md`, copilot-instructions.md), the agent's system prompt, or the first message of a task. It compresses the working knowledge of an 8-week LLM engineering course into the patterns, API signatures, and rules the agent needs to write correct code.
>
> **For the agent:** follow the conventions, patterns, and RULES below when implementing LLM projects. Code templates are canonical — match their exact field names and call signatures. Detailed per-project walkthroughs live in `WEEK*_GUIDE.md` and `MASTER_STEP_BY_STEP_GUIDE.md` in this repo; consult them when you need the full execution trace of a pattern.

---

## 1. UNIVERSAL PATTERN — every project has this shape

```
1. CONFIGURE   load .env, verify keys, build clients
2. GATHER      scrape URL / RSS / audio / DB / dataset
3. PREPARE     build messages / chunks / prompts / JSONL
4. CALL        chat.completions / pipeline / generate / train
5. LOOP        while tool_calls: run tool, append result, call again
6. PRESENT     display / UI / notification  +  EVALUATE against a fixed test set
```
When reading or writing any LLM code, locate all six. Missing #6 (evaluation) is the most common incomplete implementation.

## 2. PROJECT CONVENTIONS

```python
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)                      # ALWAYS override=True (re-reads after .env edits)
api_key = os.getenv('OPENAI_API_KEY')           # validate + print friendly diagnostic if missing
openai = OpenAI()                               # client auto-reads OPENAI_API_KEY
```
- RULE: secrets live ONLY in `.env` (`NAME=value` lines). Never hardcode keys, never commit `.env`.
- RULE: re-run `load_dotenv(override=True)` after any `.env` edit.
- Response unwrapping is ALWAYS: `response.choices[0].message.content` (dot/index chain — nothing shorter works).
- Multiple providers: OpenAI-compatible for OpenAI/Grok/Groq/OpenRouter/Ollama (base_url varies); native SDKs for Anthropic/Gemini; or LiteLLM for uniform routing.

## 3. THE MESSAGES LIST — core data structure

```python
messages = [
    {"role": "system",    "content": "Persona, rules, output format. Always FIRST."},
    {"role": "user",      "content": "the human's request + any material to analyze"},
    {"role": "assistant", "content": "model's own previous reply"},        # history
    {"role": "tool",      "content": "result of a tool call", "tool_call_id": "call_abc123"},
]
```
- RULE: models are STATELESS. "Memory" = resend the entire list with new items appended, every call. Nothing persists server-side.
- RULE: `system` first, newest `user` message LAST.
- Roles are labels YOUR code assigns: to make two models talk, one model's replies become `user` content in the other's list (multi-agent debate pattern).

## 4. CANONICAL LLM CALLS (OpenAI-compatible)

```python
# basic
response = openai.chat.completions.create(model="gpt-4.1-mini", messages=messages)

# streaming (typewriter) — a generator function
def stream_reply(messages):
    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""    # NOTE: .delta in streaming, .message otherwise
        yield result

# JSON mode — model constrained to valid JSON
response = openai.chat.completions.create(model=MODEL, messages=messages,
                                          response_format={"type": "json_object"})
data = json.loads(response.choices[0].message.content)    # still needs parsing; prompt MUST mention "JSON"

# structured outputs — Pydantic schema enforced by API (preferred over raw JSON mode)
from pydantic import BaseModel, Field
class Deal(BaseModel):
    product_description: str = Field(description="The product description, NOT the discount phrasing")
    price: float = Field(description="The actual price in dollars")
result = completion.parse(response_format=Deal)           # → typed object, no json.loads needed

# reasoning models: control thinking budget
#   reasoning_effort="high" for hard code/analysis, "minimal"/"none" for cheap extraction tasks

# audio: transcription + text-to-speech
transcript = openai.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f, response_format="text").text
speech = openai.audio.speech.create(model="tts-1", voice="onyx", input=text)      # → audio bytes

# image generation
img = openai.images.generate(model="dall-e-3", prompt=prompt, size="1024x1024")
# img.data[0].b64_json → base64 → Image.open(BytesIO(base64.b64decode(b64)))
```
Models used in the course (check current lineup when implementing): `gpt-5.1` (flagship), `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1-mini`, `gpt-4.1-nano-2025-04-14` (fine-tune base), `claude-sonnet-4-5`, `gemini-3-pro`, `grok-4.1-fast`, local `meta-llama/Llama-3.2-3B-Instruct`. Embeddings: `all-MiniLM-L6-v2` (local, 384-dim) or OpenAI embeddings.

## 5. PROMPT ENGINEERING RULES

- Split system (persona + rules + output format, set once) vs user (the ask + the material appended after it).
- One-shot/few-shot: include a worked example in the system prompt to pin format and conventions (e.g. JSON shape, absolute URLs).
- Cost guard: hard-truncate scraped material — `user_prompt = user_prompt[:5_000]`. Blunt but caps tokens/cost.
- Dynamic system prompts: ordinary `if` code may extend the system string per-request (policy in code, delivery by model).
- Grounding (RAG): "Answer only from the provided context; if it's not there, say you don't know."
- Tell the model the eval criteria it will be judged on — measurably improves output.
- Tool descriptions are prompts: the model picks tools by matching user intent against `description` text. Be specific.

## 6. TOOL CALLING / AGENT LOOP — the central mechanism

The model CANNOT execute code. It only writes a work order; YOUR code runs it.

```python
tools = [{"type": "function", "function": {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {"type": "object",
        "properties": {"destination_city": {"type": "string", "description": "The city the customer wants"}},
        "required": ["destination_city"], "additionalProperties": False}}}]

messages.append({"role": "user", "content": "How much for Tokyo?"})
response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

while response.choices[0].finish_reason == "tool_calls":          # ← THE AGENT LOOP
    message = response.choices[0].message
    messages.append(message)                                      # append the REQUEST (required!)
    for tool_call in message.tool_calls:
        args = json.loads(tool_call.function.arguments)           # arguments is a JSON STRING, not dict
        result = get_ticket_price(args["destination_city"])       # your function, your machine
        messages.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
reply = response.choices[0].message.content                       # finish_reason is now "stop"
```
- RULE: `finish_reason == "tool_calls"` means "not an answer — a work order"; `"stop"` means done.
- RULE: append BOTH the assistant's tool_call message AND every tool result before re-calling. Include `tool_call_id` on each result. Omitting either = API error.
- Any model that chains multiple tools in a loop until `stop` is an autonomous agent. Guard side-effecting tools (notify/send/post) with a once-only flag.

## 7. WEB SCRAPING PATTERN

```python
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36..."}  # fake Chrome or sites block you
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")
title = soup.title.string if soup.title else "No title found"
for irrelevant in soup.body(["script", "style", "img", "input"]):
    irrelevant.decompose()                                   # strip noise before text extraction
text = soup.body.get_text(separator="\n", strip=True)
return (title + "\n\n" + text)[:2_000]                       # always truncate
```
- RULE: `requests` sees only initial HTML — JS-rendered sites (e.g. openai.com) return little text. Not a bug; pick a different source.
- Link discovery: `[link.get("href") for link in soup.find_all("a")]` → drop empties → LLM call #1 selects relevant links (one-shot JSON example teaches relative→absolute) → scrape each → concatenate under `### Link: {type}` headers → LLM call #2 writes the artifact.
- RSS (preferred for monitoring — machine-readable, no scraping): `feedparser.parse(rss_url)` → `feed.entries[:10]` → fetch each linked page. Filter already-seen URLs via a memory file.

## 8. GRADIO UI PATTERNS

```python
# 1) quick wrapper: function → web app. Generator fn = live streaming output
gr.Interface(fn=stream_reply, inputs=[gr.Textbox(label="Message")],
             outputs=[gr.Markdown(label="Response")], title="...").launch(inbrowser=True)

# 2) chat with managed history — Gradio appends the new message BEFORE calling you
def chat(message, history):                     # history = list of {role, content} dicts (type="messages")
    history = [{"role": h["role"], "content": h["content"]} for h in history]     # normalize
    msgs = [{"role": "system", "content": SYSTEM}] + history + [{"role": "user", "content": message}]
    yield from stream_reply(msgs)
gr.ChatInterface(fn=chat, type="messages").launch()

# 3) Blocks: manual layout + event chaining
with gr.Blocks(title="...") as ui:
    message = gr.Textbox(); chatbot = gr.Chatbot(type="messages", value=[], height=500)
    audio = gr.Audio(autoplay=True); image = gr.Image()
    message.submit(add_user_msg, inputs=[message, chatbot], outputs=[message, chatbot]).then(
        respond, inputs=chatbot, outputs=[chatbot, audio, image])
ui.launch()
# add_user_msg returns ("", history + [{"role":"user","content":msg}])  → clears box, shows bubble instantly
# .then(respond, inputs=chatbot, ...)  → step 2 runs automatically with step 1's output
```
- RULE: keep UI-feedback events instant; put slow LLM calls in a `.then()` event.
- Multi-modal outputs: return `(reply_text, audio_bytes_or_path, PIL_image)` in output order.
- Auto-refresh: `gr.Timer(300).tick(fn=job)` re-runs a job every 300s.
- Live log panel: python `logging` + `QueueHandler(queue)`; background worker thread; a generator drains the queue and yields HTML (convert ANSI colors to `<span style="color:...">`), rendered by `gr.HTML`.

## 9. HUGGINGFACE LOCAL MODELS (open-weight, free per token, needs GPU)

```python
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")

# high-level wrapper — 6 internal steps: resolve model→tokenize→to(GPU)→forward→softmax→label
pipe = pipeline("sentiment-analysis", device="cuda")          # or ner / summarization / qa / asr ...
pipe("text")
asr = pipeline("automatic-speech-recognition", model="openai/whisper-medium.en",
               dtype=torch.float16, device="cuda")            # Whisper: audio file path → {"text": ...}

# manual control
tokenizer = AutoTokenizer.from_pretrained(MODEL); tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")  # model-specific format!
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", quantization_config=quant_config)
outputs = model.generate(inputs, max_new_tokens=80)           # autoregressive: 1 token/pass, appended
print(tokenizer.decode(outputs[0]))
del model, inputs, tokenizer, outputs; gc.collect(); torch.cuda.empty_cache()   # MANDATORY before next model
```
- RULE: 4-bit quantization ≈ 4× memory cut (6.4GB→1.7GB for 3B) with negligible loss — default on <24GB GPUs.
- RULE: a model's tokenizer is not interchangeable; `apply_chat_template` is the only correct way to format prompts.
- RULE: always free GPU memory (`del` + `gc.collect()` + `empty_cache()`) or the next load OOMs.

## 10. RAG (Retrieval-Augmented Generation) — answering from YOUR documents

```python
# INGEST (once): load → chunk → embed → persist
documents = [Document(page_content=text, metadata={"doc_type": folder_name}) for ...]
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")        # 384-dim, local, free
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="vector_db")

# QUERY: same embedding model! → retrieve top-k → inject into prompt
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
docs = retriever.invoke(question)
context = "\n\n".join(doc.page_content for doc in docs)
system = SYSTEM_PROMPT.format(context=context)   # "Answer only from this context..."
```
- RULE: query-time embedding MUST use the same model as ingest (coordinates from different models don't align).
- chunk_overlap saves facts sliced at boundaries; metadata (doc_type) enables filtering/visualization (t-SNE → Plotly for cluster sanity-checks).
- Upgrades: (a) QUERY REWRITE — small LLM call turns conversational follow-ups into standalone search queries before embedding; (b) RERANK — LLM re-orders top-20 (`response_format` rank schema) before injecting top few.
- EVAL: fixed test set (`question/keywords/reference_answer/category`); retrieval scored with MRR (`1/rank` of first chunk containing a keyword, averaged) + coverage; answers scored by LLM-as-judge returning a Pydantic `{feedback, accuracy 1-5, ...}` against the reference answer.

## 11. CLASSICAL ML + NEURAL NETS (tabular/text → numbers)

```python
# the race lineup, cheap→fancy (each beat the last on avg |error|):
random_pricer  # baseline noise
constant_pricer = lambda item: train_mean              # the bar any real model must clear
LinearRegression()                                     # on few numeric features
Pipeline([("vectorizer", CountVectorizer()), ("linear", LinearRegression())])   # bag of words: learned weight per word
RandomForestRegressor(n_estimators=100)                # averaged trees
XGBRegressor()                                         # sequential error-correcting trees (boosting)
```
Neural network training heartbeat (memorize; every framework is this):
```python
for epoch in range(EPOCHS):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()                  # 1 erase
        outputs = model(X_batch)               # 2 forward
        loss = loss_fn(outputs, y_batch)       # 3 score (MSE / L1)
        loss.backward()                        # 4 backprop: blame per weight
        optimizer.step()                       # 5 nudge weights downhill
```
- Robust-features DNN recipe: `HashingVectorizer` (no learned vocab → unseen words still map) → residual MLP → predict log-price then invert with stored `Y_MEAN/Y_STD` → L1 loss (outlier-robust) → AdamW + cosine LR schedule.
- Data hygiene: dedupe by title AND full text (data leakage = inflated scores); stratify/weight samples so rare cases are represented; fixed untouched test split; plot histograms at every pipeline stage.
- Batch API for bulk LLM preprocessing (50% cost): JSONL lines `{"custom_id": "<id>", "url": "/v1/chat/completions", "body": {"model": ..., "messages": [...]}}` → `files.create(purpose="batch")` → `batches.create(completion_window="24h")` → poll → results stitched back via `custom_id` (they return unordered).
- `post_process`: regex the first number out of messy model text before comparing.

## 12. FINE-TUNING

```python
# A) OpenAI hosted (cheap, style/skill adaptation): JSONL {"messages":[{"role":"user","content":prompt},
#    {"role":"assistant","content":completion}]} → files.create(purpose="fine-tune") →
openai.fine_tuning.jobs.create(model="gpt-4.1-nano-2025-04-14", training_file=file_id,
                               hyperparameters={"n_epochs": 1})
# B) QLoRA (open model, best quality-per-dollar; free Colab T4 suffices for 3B):
lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.1, task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj"])          # + MLP modules if data allows
cfg = SFTConfig(epochs=3, per_device_train_batch_size=..., learning_rate=1e-4,
                warmup_ratio=0.01, lr_scheduler_type="cosine", max_grad_norm=0.3,
                max_sequence_length=128, logging_steps=..., eval_steps=..., save_steps=...,
                push_to_hub=True, bf16=(torch.cuda.get_device_capability()[0] >= 8))
trainer = SFTTrainer(model=quantized_base, train_dataset=train, eval_dataset=val,
                     peft_config=lora, args=cfg)
trainer.train(); trainer.model.push_to_hub(REPO, private=True)   # only the ~85-300MB adapter uploads
# inference: base (4-bit) + PeftModel.from_pretrained(base, "user/adapter-repo")
```
- Training data schema: identical-format `prompt → completion` pairs ("What does this cost...\n\n<summary>\n\nPrice is $" → "149.00"). Token-count your inputs first; truncate to a fixed token CUTOFF.
- RULE: base model stays FROZEN (and 4-bit); only LoRA adapters train.
- Watch wandb/tensorboard: train loss ↓ while val loss ↑ = overfitting → fewer epochs. Optimizer `paged_adamw_32bit` pages optimizer state to CPU when GPU is tight.
- Honest expectation: hosted fine-tuning often teaches style, not accuracy (course: nano-FT $75.91 vs zero-shot GPT-5.1 $44.74); QLoRA on narrow task beat everything ($39.85).

## 13. DEPLOYMENT (Modal — serverless GPUs/containers)

```python
app = modal.App("pricer-service")
image = modal.Image.debian_slim().pip_install("transformers", "torch", "bitsandbytes", "accelerate", ...)
secrets = [modal.Secret.from_name("huggingface-secret")]     # created via: modal secret create huggingface-secret HF_TOKEN=hf_...
volume = modal.Volume.from_name("hf-hub-cache")               # persist model downloads between runs

@app.cls(image=image, secrets=secrets, gpu="A10G", volumes={"/cache": volume},
         scaledown_window=1200, min_containers=1)             # warm vs cheap trade-off
class Pricer:
    @modal.enter()
    def load(self): self.model = ...                          # ONE-TIME load; runs once per container
    @modal.method()
    def price(self, description: str) -> float: ...

# deploy: uv run modal deploy -m pricer_service2      (module, not file!)
# client (any machine):
Pricer = modal.Cls.from_name("pricer-service", "Pricer")
result = Pricer().price.remote("...")                        # .remote() = network call
```

## 14. MULTI-AGENT FRAMEWORK (putting it all together)

- One interface to rule them all: every agent exposes `price(description) -> float`. This is what lets an ensemble mix heterogeneous experts.
- Specialist pattern: Frontier (LLM+RAG over similar items with known prices) / fine-tuned model on serverless GPU / plain DNN forward pass.
- Ensemble: `0.8*frontier + 0.1*specialist + 0.1*dnn` — weight by reliability; tune as a hypothesis.
- Preprocessing agent before experts: a small local LLM rewrites messy real-world text into the exact clean format training data used. Garbage-in protection.
- Planner: `messages = [system, user(to-do list in English)] + tools`; loop on `finish_reason=="tool_calls"`; each iteration = one tool round-trip, full history resent. Deterministic alternative: hardcode the plan in Python (`for deal: estimate`) — prefer determinism when you can.
- Memory: persist seen-URLs/results to JSON (`memory.json`); filter duplicates each run.
- Notification: Pushover `requests.post("https://api.pushover.net/1/messages.json", data={"user":..., "token":..., "message": text[:200] + "... " + url, "sound": "cashregister"})`; have a second LLM write the blurb; once-only guard flag.

## 15. PITFALL CHECKLIST (the recurring failure modes)

1. `.env` edited but `load_dotenv(override=True)` not re-run → stale keys.
2. `response.content` instead of `.choices[0].message.content`.
3. Streaming: `.message.content` instead of `.delta.content`; forgetting `or ""` on None chunks.
4. JSON mode without mentioning JSON in the prompt → API error; no `response_format` → `json.loads` crash on prose.
5. Tool loop: not appending the assistant tool_call message / missing `tool_call_id` → API 400.
6. `finish_reason` never checked → treating a work order as an answer.
7. No truncation of scraped text → token explosion, cost blow-up, confused model.
8. Different embedding model at query time than at ingest → silent garbage retrieval.
9. Duplicate items across train/test (leakage) → fake scores.
10. Skipping `del model; gc.collect(); torch.cuda.empty_cache()` → CUDA OOM on next model.
11. Notebook cells run out of order → `NameError`s; fix with Restart & Run All.
12. Side-effect tools (send/notify/post) without once-only guards → spam.
13. Shipping without an eval harness → no way to know if a change helped.
14. Forgetting generated folders (`vector_db/`, caches) to rebuild after data changes → stale index.

## 16. SCOREBOARD TO BEAT (course benchmark, avg $ error, price prediction)

Constant 106.18 → classic ML 68.23 → NN 63.97 → GPT-5.1 zero-shot 44.74 → DNN 46.49 → QLoRA full 39.85 → **RAG-augmented GPT-5.1 30.19 → ensemble 29.90 (champion)**. Lesson: frontier + retrieval + ensemble > any single model; always compare against constant-baseline and zero-shot frontier.

---
*Full execution traces: `MASTER_STEP_BY_STEP_GUIDE.md` · Per-week deep dives: `WEEK1_WEEK2_GUIDE.md`, `WEEK3_WEEK4_GUIDE.md`, `WEEK5_GUIDE.md`, `WEEK6_GUIDE.md`, `WEEK7_GUIDE.md`, `WEEK8_GUIDE.md`.*
