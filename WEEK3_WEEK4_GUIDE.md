# 📘 Week 3 & Week 4 — Complete Beginner's Guide

This guide explains **every file** in `week3/` and `week4/` of the *LLM Engineering* repo — what each piece of code does, why it exists, and how it fits together. (Companion to the Week 1–2 guide.)

> **The one-sentence summary:** Weeks 1–2 used models running on *someone else's servers* via APIs. **Week 3 flips it: you run open-source models YOURSELF, on a GPU, using HuggingFace. Week 4 then has the LLMs write high-performance compiled code (C++/Rust) for you — and races them against each other.**

> **Where the code lives:** From Week 3 on, the course runs on **Google Colab** (free/cheap rented GPUs). The local notebooks in `week3/` are short "launchpads" with Colab links — the real labs are in Colab. This guide covers both the local files **and** the full contents of those Colabs (fetched and explained below).

---

# 🧠 PART 0 — The Big Picture

## New vocabulary you need first

| Term | Beginner translation |
|---|---|
| **Training** | Feeding a model data so it adjusts its internal knobs (*parameters/weights*) and learns. Costs millions. |
| **Fine-tuning** | More training on an already-trained model (Weeks 6–7). |
| **Inference** | *Using* a trained model to get outputs. Everything you did in Weeks 1–2 was inference. Weeks 3–5 are inference too. |
| **GPU** | The chip that does model math in parallel. You'll rent one (a free **Tesla T4**) on Colab. |
| **HuggingFace (HF)** | "The GitHub of AI" — hosts models, datasets, and the `transformers` Python library. You'll make a free account + **API token**. |
| **Colab** | Google's hosted Jupyter notebook with optional GPU runtimes. |
| **Quantization** | Squeezing model weights into fewer bits (e.g. 4 instead of 16) so a big model fits in a small GPU's memory. |

## How the two weeks connect

```
WEEK 3 — The HuggingFace ladder (each day = one rung up)      WEEK 4 — LLMs as programmers
Day 1  Colab setup (GPUs 101)                                 Day 3  Python → C++ via 4 frontier models,
Day 2  PIPELINES: 2-line AI (sentiment, translation,                 compile & race them
       summarization, image gen, speech…)                   Day 4  + open-source coder models
Day 3  TOKENIZERS: how text becomes numbers                 Day 5  + Rust, a harder benchmark,
Day 4  MODELS: load Llama yourself, quantize,                      and a full Gradio "IDE"
       generate, and peek inside a Transformer              + system_info.py, styles.py helpers
Day 5  (a) SEE inference working (token probabilities!)     Outcome: a working code-porting product
       (b) BUILD: audio → transcription → meeting minutes   + your first real benchmark
```

---

# 🟦 PART 1 — WEEK 3: Running models yourself with HuggingFace

## 1.1 How the week is wired (the local files)

Each local notebook is a signpost. Example, `week3/day2.ipynb` in its entirety:

> *"For this session we head to Google Colab and use this Notebook to explore the HuggingFace High Level API, pipelines."* + a `colab.research.google.com/drive/...` link.

**Why Colab?** Open-source models need a GPU. A T4 GPU on Colab's free tier is enough for everything this week — no purchases required. **One-time setup:** create a free HuggingFace account → Avatar menu → *Tokens* → create a token **with WRITE permissions** → in each Colab, click the 🔑 icon and add it as a secret named `HF_TOKEN`.

## 1.2 Day 1 — Colab 101 (`week3/day1.ipynb`)

A link to an intro Colab. The essentials it establishes:

- Colab = Jupyter in Google's browser. Cells run on a *runtime* — a virtual machine you can upgrade to a GPU via **Runtime → Change runtime type → T4 GPU**.
- The rest of the course uses the **T4** (16 GB GPU memory).
- **Pro-tip baked into every lab:** if you ever see `RuntimeError: CUDA is required but not available for bitsandbytes...`, don't touch package versions — Google silently swapped your runtime. Fix: *Kernel → Disconnect and delete runtime*, reload, reconnect a T4, rerun cells from the top.

## 1.3 Day 2 — HuggingFace **Pipelines**: AI in two lines

*Source: "week 3 day 2 - pipelines" Colab.* The `transformers` library has two levels: the **high-level API (pipelines)** — today — and the **low-level API (tokenizers + models)** — Day 4.

### The pattern (the whole lesson in 6 lines)

```python
my_pipeline = pipeline(task, model=..., device="cuda")   # STEP 1: create (downloads model)
result = my_pipeline(input)                              # STEP 2: call it as often as you like
```

- Omit `model=` and HF uses a sensible default for the task.
- `device="cuda"` = run on the NVIDIA GPU (`"mps"` on a Mac with Apple silicon).

### Setup cells (top of every Colab this week)

```python
!pip install -q --upgrade datasets==3.6.0 transformers==4.57.6 huggingface_hub diffusers

gpu_info = !nvidia-smi                     # shell command: is a GPU attached?
if gpu_info.find('failed') >= 0: print('Not connected to a GPU')

hf_token = userdata.get('HF_TOKEN')        # read the Colab secret
login(hf_token, add_to_git_credential=True)
```

`!` runs shell commands from a notebook; `nvidia-smi` is the GPU's dashboard; `login()` authenticates you to download models.

### The task tour — every demo in the Colab

```python
# 1. Sentiment analysis (default model)
my_simple_sentiment_analyzer = pipeline("sentiment-analysis", device="cuda")
my_simple_sentiment_analyzer("I'm super excited to be on the way to LLM mastery!")
# -> [{'label': 'POSITIVE', 'score': 0.99...}]

# 2. A BETTER model for the tricky negative case — just name another model!
better_sentiment = pipeline("sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment", device="cuda")
better_sentiment("I should be more excited to be on the way to LLM mastery!!")
# -> 3-star rating. Swapping one string swaps the entire model.
```

That second demo is the day's best lesson: *"I should be more excited…"* is grammatically positive but emotionally negative. The default model gets it wrong; a *different* open-source model gets it right. **Model choice is an engineering decision.**

```python
# 3. Named Entity Recognition — find people/orgs/places in text
ner = pipeline("ner", device="cuda")
ner("AI Engineers are learning about the amazing pipelines from HuggingFace in Google Colab from Ed Donner")

# 4. Question Answering — extract an answer FROM a provided context
question_answerer = pipeline("question-answering", device="cuda")
question_answerer(question="What are Hugging Face pipelines?",
                  context="Pipelines are a high level API for inference of LLMs with common tasks")

# 5. Summarization
summarizer = pipeline("summarization", device="cuda")
summarizer(text, max_length=50, min_length=25, do_sample=False)

# 6. Translation (default EN->FR, then a specific EN->ES model)
translator = pipeline("translation_en_to_fr", device="cuda")
translator = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es", device="cuda")

# 7. Zero-shot classification — categories YOU invent at runtime!
classifier = pipeline("zero-shot-classification", device="cuda")
classifier("Hugging Face's Transformers library is amazing!",
           candidate_labels=["technology", "sports", "politics"])

# 8. Text generation — a small LLM completing text
generator = pipeline("text-generation", device="cuda")
generator("If there's one thing I want you to remember about using HuggingFace pipelines, it's")
```

**Why #7 is special:** nobody trained a "technology/sports/politics" classifier — a general model understands the labels you pass in *at run time*. That flexibility is what makes transformers transformative.

### Beyond text: images & audio (via `diffusers`)

```python
# IMAGE generation — diffusion models (like Week 2's gpt-image-1, but open-source & local)
from diffusers import AutoPipelineForText2Image
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo",
        torch_dtype=torch.float16, variant="fp16")   # fp16 = half-precision = fits GPU
pipe.to("cuda")
image = pipe(prompt="A class of students learning AI engineering in a vibrant pop-art style",
             num_inference_steps=4, guidance_scale=0.0).images[0]   # only 4 steps = 'turbo'!

# TEXT-TO-SPEECH
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device='cuda')
embeddings_dataset = load_dataset("matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)  # pick a voice
speech = synthesiser("Hi to an artificial intelligence engineer, on the way to mastery!",
                     forward_params={"speaker_embeddings": speaker_embedding})
Audio(speech["audio"], rate=speech["sampling_rate"])    # playable audio right in the notebook
```

Two new ideas: **diffusion models** create images by starting from noise and de-noising step-by-step (`num_inference_steps` = how many passes; turbo models need only 4), and this TTS needs a **speaker embedding** — a numeric "voice print" pulled from a dataset to choose whose voice it sounds like.

---

## 1.4 Day 3 — Tokenizers (`week3/day3.ipynb`)

> Note: this particular Colab requires a Google sign-in to open, so its exact cells aren't quoted here — but the lesson is well-defined by the course: **look at different tokenizers side by side.**

It extends Week 1 Day 4 (where `tiktoken` showed GPT's tokens). In Colab you use HuggingFace's universal tokenizer class:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

ids = tokenizer.encode("Hi my name is Ed and I like banoffee pie")   # text -> token ids
tokenizer.decode(ids)                                                # ids -> text
```

The key insight: **every model ships with its own tokenizer**, and they disagree — the same sentence tokenizes differently for GPT (byte-pair encoding via `tiktoken`) vs Llama (SentencePiece-style) vs others. Special tokens (`<|begin_of_text|>`, `<|eot_id|>`) mark boundaries. This matters because tokens are the currency of cost, context limits, and — after Day 4 — literally the only thing the model sees.

## 1.5 Day 4 — **Models**: load Llama 3.1 yourself (`week3/day4.ipynb` + Colab)

The heart of the week: the **low-level API**. You'll load a real LLM onto a GPU, inspect its neural network, and generate text with PyTorch. (Llama models are *gated* — you must click "agree to terms" on their HF pages once; approval is quick.)

### Step 1 — Pick models & prepare the prompt

```python
LLAMA    = "meta-llama/Llama-3.2-1B-Instruct"        # or Meta-Llama-3.1-8B-Instruct (bigger!)
PHI      = "microsoft/Phi-4-mini-instruct"           # Microsoft's small model
GEMMA    = "google/gemma-3-270m-it"                  # Google's tiny model (no quant needed)
QWEN     = "Qwen/Qwen3-4B-Instruct-2507"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # a 'reasoning' distilled model

messages = [{"role": "user", "content": "Tell a joke for a room of Data Scientists"}]
```

Same `messages` format as Weeks 1–2 — but this time *we* have to do everything the API did for us.

### Step 2 — Quantization config (make it fit)

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # store weights in 4 bits, not 16
    bnb_4bit_use_double_quant=True,       # compress the compression info too
    bnb_4bit_compute_dtype=torch.bfloat16,# do the math in "brain float" 16-bit
    bnb_4bit_quant_type="nf4"             # 'normal float 4' — best quality for 4-bit
)
```

An 8-billion-parameter model at full precision needs ~16 GB; the T4 has 16 GB total. In 4 bits it needs ~4–5 GB — **quantization is the trick that lets free GPUs run big models**, at a tiny cost in accuracy.

### Step 3 — Tokenize (the chat template!)

```python
tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token            # padding filler = end-of-text token

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
```

`apply_chat_template` is the unsung hero: it converts your friendly `messages` list into the model's exact expected text format — with its special tokens, roles, and headers — then tokenizes it into a PyTorch tensor (`"pt"`) and moves it to the GPU. **This is what the OpenAI API was silently doing for you in Weeks 1–2.**

### Step 4 — Load the model and *look inside*

```python
model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto",
                                             quantization_config=quant_config)
memory = model.get_memory_footprint() / 1e6
print(f"Memory footprint: {memory:,.1f} MB")

model        # printing it dumps the entire neural network architecture!
```

The printout is your first real look at a **Transformer** (the 2017 Google architecture behind every modern LLM). Reading it top to bottom:

1. **Embedding layer** — converts each token id into a huge vector (4,096 numbers for Llama 3.1 8B).
2. **Decoder layers** — repeated blocks (32 in Llama 3.1 8B, 16 in Llama 3.2 1B), each containing **self-attention** (tokens "look at" other tokens to build context), an **MLP** (a multi-layer perceptron — the classic neuron layers that do much of the "thinking"), and **normalization** layers that keep the math stable.
3. **LM Head** — the final layer mapping back to a probability for every token in the vocabulary.

That's all a "Causal LM" is: embeddings → N × (attention + MLP) → token probabilities.

### Step 5 — Generate!

```python
outputs = model.generate(inputs, max_new_tokens=80)   # returns MORE token ids
tokenizer.decode(outputs[0])                          # ids -> readable text
```

(The first run looks like gibberish without the chat template's generation prompt — see below.) Then the notebook wraps everything reusable into one function:

```python
def generate(model, messages, quant=True, max_new_tokens=80):
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt",
                                              add_generation_prompt=True).to("cuda")
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
    streamer = TextStreamer(tokenizer)                       # prints tokens as they're born
    if quant:
        model = AutoModelForCausalLM.from_pretrained(model, quantization_config=quant_config).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model).to("cuda")
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                             max_new_tokens=max_new_tokens, streamer=streamer)

generate(PHI, messages)
generate(GEMMA, messages, quant=False)     # Gemma 270M is tiny — no quantization needed
generate(QWEN, messages)
generate(DEEPSEEK, messages, quant=False, max_new_tokens=500)   # reasoning models think long
```

Three subtle lessons in that function:

- **`add_generation_prompt=True`** appends the marker that says "your turn, assistant" — without it the model just continues the user's sentence instead of answering (a classic bug, celebrated in the notebook thanks to a student).
- **`TextStreamer(tokenizer)`** = streaming like Week 2, but hand-rolled: each new token is decoded and printed live.
- **`attention_mask`** tells the model which positions are real tokens vs padding.

### Step 6 — Housekeeping (GPU hygiene)

```python
del model, inputs, tokenizer, outputs   # drop references
gc.collect()                            # ask Python to garbage-collect
torch.cuda.empty_cache()                # return memory to the GPU pool
```

A T4 has ~15 GB — barely enough for one model — so you *must* clean up between models. This cell is the reason the notebook can load five models in a row.

---

## 1.6 Day 5 (part A) — *Watch* inference happen: `week3/visualizer.py` + `day5.ipynb`

Before building the meeting-minutes product, the course shows you something mind-bending: **the literal probability of every token as the model writes.** The local notebook runs:

```python
from visualizer import TokenPredictor, create_token_graph, visualize_predictions

message = "In one sentence, describe the color orange to someone who has never been able to see"
model_name = "gpt-4.1-mini"

predictor = TokenPredictor(model_name)
predictions = predictor.predict_tokens(message)
G = create_token_graph(model_name, predictions)
plt = visualize_predictions(G)
plt.show()
```

### How `TokenPredictor` extracts probabilities

```python
response = self.client.chat.completions.create(
    model=self.model_name,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=max_tokens,
    temperature=0,          # no randomness -> deterministic, reproducible picture
    logprobs=True,          # ← THE KEY: "also tell me the probabilities"
    seed=42,                # pin the random seed for reproducibility
    top_logprobs=3,         # give me the top 3 candidate tokens at each step
    stream=True,
)
for chunk in response:
    token = chunk.choices[0].delta.content
    logprob_dict = {item.token: item.logprob for item in chunk.choices[0].logprobs.content[0].top_logprobs}
    # probability = math.exp(logprob); keep the winner + top-2 alternatives
```

- **logprobs** = log of each candidate token's probability (logs are used because multiplying tiny probabilities underflows; `math.exp()` converts back to 0–1).
- The API has *always* been choosing from a probability distribution over the vocabulary — this proves Week 1's "next-token predictor" description literally.

### How the graph is drawn (the rest of `visualizer.py`)

- `create_token_graph` builds a `networkx.DiGraph`: a green START node → one blue node per generated token (labelled with its % probability) → a red END node, with each token's 2 runner-up candidates as grey side-nodes.
- `visualize_predictions` lays main tokens down a vertical spine, alternatives to the left/right, and draws with matplotlib. Result: a tall chart where you can see exactly where the model was *confident* (100%) vs *torn* (60/30/10 between words).

---

## 1.7 Day 5 (part B) — The product: **Meeting Minutes from audio** (Colab)

The week's capstone combines a new *modality* (audio) with the Day 4 skills. Goal: upload a meeting recording → get professional minutes with action items.

### Setup extras

```python
from google.colab import drive
drive.mount("/content/drive")                                  # attach YOUR Google Drive
audio_filename = "/content/drive/MyDrive/llms/denver_extract.mp3"  # a Denver City Council clip
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
audio_file = open(audio_filename, "rb")                        # 'rb' = read as raw bytes
```

### STEP 1 — Transcribe the audio (two options, same interface)

```python
# OPTION 1: open-source Whisper (OpenAI's model, hosted on HF, running on YOUR GPU)
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition",
                model="openai/whisper-medium.en",
                dtype=torch.float16, device='cuda',
                return_timestamps=True)
result = pipe(audio_filename)
transcription = result["text"]

# OPTION 2: OpenAI's paid API
transcription = openai.audio.transcriptions.create(
    model="gpt-4o-mini-transcribe", file=audio_file, response_format="text")
```

Note the pattern you now take for granted: *both* routes are essentially one call — the open-source route is just `pipeline(...)` on your own GPU (free, private) vs the API (easier, paid). The notebook displays both transcripts side by side to compare quality.

### STEP 2 — Analyze & write minutes (the Day 4 machinery, at work)

```python
system_message = """
You produce minutes of meetings from transcripts, with summary, key discussion points,
takeaways and action items with owners, in markdown format without code blocks.
"""
user_prompt = f"""
Below is an extract transcript of a Denver council meeting.
Please write minutes in markdown without code blocks, including:
- a summary with attendees, location and date
- discussion points
- takeaways
- action items with owners
Transcription:
{transcription}
"""
messages = [{"role": "system", "content": system_message},
            {"role": "user",   "content": user_prompt}]

tokenizer = AutoTokenizer.from_pretrained(LLAMA)
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto",
                                             quantization_config=quant_config)
outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)
display(Markdown(tokenizer.decode(outputs[0])))
```

**The full pipeline:** `audio file → Whisper (GPU) → transcript text → chat template → Llama 3.2 3B (quantized, on GPU) → streamed markdown minutes`. Two models chained = the same multi-step pattern as Week 1 Day 5's brochure, but now *you own the compute*. A linked student contribution even wraps this in Gradio with `TextIteratorStreamer` + background threads.

---

# 🟩 PART 2 — WEEK 4: LLMs as programmers (C++/Rust code generation)

## The project

Write a Python program that's painfully slow → have frontier LLMs **port it to C++/Rust** → compile it → run it → measure the speedup → crown a champion. Along the way you learn: LLM code generation in practice, executing generated code safely from Python, and benchmarking honestly.

## 2.1 `week4/system_info.py` — telling the LLM about *your* machine

**Problem:** compile commands differ on every OS (Mac/Windows/Linux; clang vs gcc vs MSVC; ARM vs Intel). **Solution:** don't hard-code — *ask the LLM*, and give it the facts. This file gathers those facts defensively:

```python
def _run(cmd, timeout=3):
    """Run a command safely. Returns stdout text or ''."""
    try:
        if isinstance(cmd, str):
            return subprocess.check_output(cmd, shell=True, text=True,
                                           stderr=subprocess.DEVNULL, timeout=timeout).strip()
        ...
    except Exception:
        return ""          # NEVER crash — every probe is best-effort
```

- `_run` executes a shell command and returns its output — or empty string on any failure. **Every check is wrapped in try/except** because this code runs on unknown machines.
- `_which(name)` (`shutil.which`) checks if a program exists on your PATH.
- `_os_block()` — OS, CPU architecture (Intel/ARM), even WSL and Mac-Rosetta detection; `_cpu_block()` — CPU brand, physical/logical core counts, and SIMD capability flags (AVX2, NEON…) that determine which optimization flags are legal; `_toolchain_block()` — which compilers/build tools are installed and their versions.
- The public API:

```python
def retrieve_system_info():        # a dict: {"os": ..., "package_managers": ..., "cpu": ..., "toolchain": ...}
def rust_toolchain_info():         # same idea for Rust: rustc/cargo/rustup paths & versions
```

**Why it matters:** this dict gets pasted *into the prompt*, so GPT-5 can reply "install LLVM with brew, then compile with `clang++ -Ofast ...`" — instructions guaranteed to fit *your* machine. **Prompting with environment context instead of hard-coding** is a genuinely useful agentic pattern.

## 2.2 `week4/day3.ipynb` — Python → C++, four frontier models, one race

### Step 1 — Let the LLM configure your compiler

```python
message = f"""
Here is a report of the system information for my computer.
I want to run a C++ compiler to compile a single C++ file called main.cpp ...
Please reply with whether I need to install any C++ compiler ... and the exact
compile_command and run_command I should use.
System information:
{system_info}
"""
response = openai.chat.completions.create(model=OPENAI_MODEL, messages=[...])
```

You paste the suggested commands into the next cell:

```python
compile_command = ["clang++", "-std=c++17", "-Ofast", "-mcpu=native", "-flto=thin",
                   "-fvisibility=hidden", "-DNDEBUG", "main.cpp", "-o", "main"]
run_command = ["./main"]
```

Beginner decode of those flags: `clang++` = the compiler; `-std=c++17` = language version; `-Ofast` = maximum optimization; `-mcpu=native` = tune for *this* CPU; `-flto=thin` = link-time optimization; `-DNDEBUG` = strip debug checks; `main.cpp -o main` = input → output binary. Note it's a **list** — that's what `subprocess` wants (no shell-quoting bugs).

### Step 2 — The porting prompts (familiar by now!)

```python
system_prompt = """
Your task is to convert Python code into high performance C++ code.
Respond only with C++ code. Do not provide any explanation other than occasional comments.
The C++ response needs to produce an identical output in the fastest possible time.
"""

def user_prompt_for(python):
    return f"""
Port this Python code to C++ with the fastest possible implementation ...
The system information is: {system_info}
Your response will be written to a file called main.cpp and then compiled and executed;
the compilation command is: {compile_command}
Respond only with C++ code.
Python code to port:
```python
{python}
```
"""
```

Notice **what's in the prompt**: the target machine info *and the exact compile command* — the model writes code knowing precisely how it will be judged. That's prompt engineering for code-gen.

### Step 3 — The porting function

```python
def port(client, model, python):
    reasoning_effort = "high" if 'gpt' in model else None     # think harder on GPT-5
    response = client.chat.completions.create(model=model, messages=messages_for(python),
                                              reasoning_effort=reasoning_effort)
    reply = response.choices[0].message.content
    reply = reply.replace('```cpp','').replace('```','')      # strip markdown code fences
    write_output(reply)                                        # save as main.cpp
```

Models love wrapping code in ` ```cpp ... ``` ` fences; the `.replace()` calls strip them so the file contains pure C++.

### Step 4 — The benchmark program: a Pi estimator

```python
pi = """
import time
def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result
start_time = time.time()
result = calculate(200_000_000, 4, 1) * 4     # 200 MILLION loop iterations!
...
"""
```

A numerical loop — Python's worst case, C++'s best case. It runs in ~19 s in Python.

### Step 5 — Compile, run, and race

```python
def run_python(code):
    exec(code, globals)                     # run the Python original in this notebook

def compile_and_run():
    subprocess.run(compile_command, check=True, text=True, capture_output=True)  # compile
    print(subprocess.run(run_command, check=True, text=True, capture_output=True).stdout)  # run ×3

port(openai, OPENAI_MODEL, pi);    compile_and_run()
port(anthropic, CLAUDE_MODEL, pi); compile_and_run()
port(grok, GROK_MODEL, pi);        compile_and_run()
port(gemini, GEMINI_MODEL, pi);    compile_and_run()
```

`subprocess.run(...)` executes an external program and waits. The key arguments: `check=True` → raise an exception if the command fails; `text=True` → give me strings not bytes; `capture_output=True` → catch stdout/stderr. `19.178207/0.082168` in the final cell is the honest way to compute speedup: *Python time ÷ compiled time*.

**Ed's day-3 leaderboard (Python ≈ 19.18 s baseline):**

| Place | Model | C++ runtime | Speedup |
|---|---|---|---|
| 4th | Claude Sonnet 4.5 | 0.104 s | ~184× |
| 3rd | GPT-5 | 0.082 s | ~233× |
| 2nd | Grok 4 | 0.018 s | ~1060× |
| 🥇 | Gemini 2.5 Pro | 0.013 s | ~1440× |

## 2.3 `week4/day4.ipynb` — Add open-source coder models + a Gradio UI

### A model→client switchboard (Week 2's trick, matured)

```python
models = ["gpt-5", "claude-sonnet-4-5-20250929", "grok-4", "gemini-2.5-pro",
          "qwen2.5-coder", "deepseek-coder-v2", "gpt-oss:20b",
          "qwen/qwen3-coder-30b-a3b-instruct", "openai/gpt-oss-120b"]

clients = {"gpt-5": openai, "claude-sonnet-4-5-20250929": anthropic, "grok-4": grok,
           "gemini-2.5-pro": gemini, "openai/gpt-oss-120b": groq,
           "qwen2.5-coder": ollama, "deepseek-coder-v2": ollama, "gpt-oss:20b": ollama,
           "qwen/qwen3-coder-30b-a3b-instruct": openrouter}

def port(model, python):
    client = clients[model]          # look up WHO to call — the name hides everything
    ...
```

One dictionary routes each model to the right provider: frontier models to their APIs, `qwen2.5-coder`/`deepseek-coder-v2`/`gpt-oss:20b` to **local Ollama**, and others to Groq/OpenRouter. `port()` no longer needs the client passed in.

### A safer `run_python` (stdout capture)

```python
def run_python(code):
    globals_dict = {"__builtins__": __builtins__}
    buffer = io.StringIO()           # an in-memory "fake file"
    old_stdout = sys.stdout          # remember the real screen
    sys.stdout = buffer              # redirect print() into the buffer
    try:
        exec(code, globals_dict)
        output = buffer.getvalue()
    except Exception as e:
        output = f"Error: {e}"       # never crash the UI
    finally:
        sys.stdout = old_stdout      # ALWAYS restore the screen
    return output
```

Redirecting `sys.stdout` is a classic Python trick: `print()` inside `exec`'d code lands in `buffer` so it can be shown in a UI textbox. Note `finally` guarantees restoration even on errors. (Also worth knowing: `exec()` runs arbitrary code — fine for your own experiments, never for untrusted input.)

`compile_and_run` gains the same treatment: a `try/except subprocess.CalledProcessError` prints the compiler's *stderr* (the error log) instead of crashing — essential when generated code sometimes doesn't compile!

### The UI (Gradio Blocks, week 2 skills applied)

```python
with gr.Blocks() as ui:
    with gr.Row():
        python = gr.Textbox(label="Python code:", lines=28, value=pi)   # pre-filled!
        cpp = gr.Textbox(label="C++ code:", lines=28)
    with gr.Row():
        model = gr.Dropdown(models, label="Select model", value=models[0])
        convert = gr.Button("Convert code")

    convert.click(port, inputs=[model, python], outputs=[cpp])   # ← the entire app logic

ui.launch(inbrowser=True)
```

Paste Python → pick a model → click Convert → C++ appears. Then run `compile_and_run()` in the notebook.

**Day-4 leaderboard adds the open-source challengers** (same 19.18 s baseline): Qwen2.5-Coder **FAIL** (produced non-working code), gpt-oss-120B ~14×, DeepSeek-Coder-v2 & Qwen3-30B ~168×, Claude ~184×, GPT-5 ~233×, **gpt-oss-20B ~238× (running free, locally!)**, Grok ~1060×, Gemini 2.5 Pro ~1440×. Lesson: **small open-source models can rival frontier models on specialized tasks like code.**

## 2.4 `week4/day5.ipynb` — Rust, a harder test, and a beautiful IDE

### Rust support

Day 5 generalizes everything to a second language. `system_info.rust_toolchain_info()` reports whether Rust is installed; another LLM call asks for the right commands:

```python
compile_command = ["/Users/ed/.cargo/bin/rustc", "main.rs",
    "-C", "opt-level=3",        # maximum optimization
    "-C", "target-cpu=native",  # use every instruction this CPU knows
    "-C", "codegen-units=1",    # compile as one unit = better optimization
    "-C", "lto=fat",            # aggressive link-time optimization
    "-C", "panic=abort",        # drop unwinding machinery = smaller, faster
    "-C", "strip=symbols",      # remove debug symbols
    "-o", "main"]
```

…then a single variable flips the whole app between languages:

```python
language = "Rust"   # or "C++"
extension = "rs" if language == "Rust" else "cpp"
system_prompt = f"Your task is to convert Python code into high performance {language} code..."
```

The prompts, file-writing (`main.rs`), and fence-stripping (now also `.replace('```rust','')`) all parameterize on `language`.

### The harder benchmark (and a trap!)

```python
python_hard = """# Be careful to support large numbers
def lcg(seed, a=1664525, c=1013904223, m=2**32):   # Linear Congruential Generator
    value = seed
    while True:
        value = (a * value + c) % m
        yield value

def max_subarray_sum(n, seed, min_val, max_val):
    ...   # O(n²) double loop over 10,000 pseudo-random numbers
def total_max_subarray_sum(...):  # ... repeated for 20 different seeds
"""
```

This is *much* harder: pseudo-random generation + an O(n²) double loop × 20 runs (~34 s in Python). And the first-line comment is the trap: Python integers are **arbitrary precision** (they never overflow), while C++/Rust integers are fixed-width. `a * value` with `a=1664525` and values up to 2³² **overflows a 32-bit int** — models that casually wrote `int` got *wrong answers* and failed the "identical output" requirement. **Speed means nothing without correctness** — the benchmark checks the printed number, not just the time.

### The IDE-grade UI (`week4/styles.py`)

```python
from styles import CSS

with gr.Blocks(css=CSS, theme=gr.themes.Monochrome(), title=f"Port from Python to {language}") as ui:
    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            python = gr.Code(label="Python (original)", value=python_hard, language="python", lines=26)
        with gr.Column(scale=6):
            cpp = gr.Code(label=f"{language} (generated)", language="cpp", lines=26)

    with gr.Row(elem_classes=["controls"]):
        python_run = gr.Button("Run Python", elem_classes=["run-btn", "py"])
        model = gr.Dropdown(models, value=models[0], show_label=False)
        convert = gr.Button(f"Port to {language}", elem_classes=["convert-btn"])
        cpp_run = gr.Button(f"Run {language}", elem_classes=["run-btn", "cpp"])

    with gr.Row(equal_height=True):
        python_out = gr.TextArea(label="Python result", lines=8, elem_classes=["py-out"])
        cpp_out = gr.TextArea(label=f"{language} result", lines=8, elem_classes=["cpp-out"])

    convert.click(fn=port, inputs=[model, python], outputs=[cpp])
    python_run.click(fn=run_python, inputs=[python], outputs=[python_out])
    cpp_run.click(fn=compile_and_run, inputs=[cpp], outputs=[cpp_out])
```

New Gradio powers on show:

- **`gr.Code`** — a real code editor with syntax highlighting (`language="python"/"cpp"`), replacing plain textboxes.
- **`gr.themes.Monochrome()` + `css=CSS`** — `styles.py` is pure CSS (dark cards, blue-tinted Python output, gold-tinted C++/Rust output, hover glows) applied via `elem_classes` hooks. You can style Gradio apps like any website.
- **`gr.Column(scale=6)`** — proportional column widths; three wired buttons = a complete IDE: *Run Python | pick model | Port | Run compiled*.

### Day-5 results (the plot twist)

On the Rust/harder benchmark, most frontier models **FAILED** (wrong output — the integer trap): Qwen2.5-Coder, Gemini 2.5 Pro, DeepSeek-Coder-v2, Qwen3-30B, Claude Sonnet 4.5, GPT-5. The podium was all **open-source GPT-OSS models**: gpt-oss-20B (0.000341 s), Grok 4 (0.000317 s), and **🥇 gpt-oss-120B (0.000304 s)** — about **111,000× faster than the Python** (`33.755209/0.000304`). A humbling, memorable lesson: model size ≠ coding skill; and *always* validate generated code against expected output.

---

# 🎓 PART 3 — Consolidation

## Concepts introduced, week by week

| Concept | Where | One-liner |
|---|---|---|
| Colab & GPU runtimes | W3 D1 | Rent a T4 free; `nvidia-smi` to verify |
| HuggingFace token & login | W3 all | `hf_token` Colab secret → `login()` |
| **Pipelines** (high-level API) | W3 D2 | `pipeline(task, model, device)` → call it |
| Model swapping via one string | W3 D2 | Change `model=` → change everything |
| Zero-shot classification | W3 D2 | Labels you invent at run time |
| Diffusion image generation | W3 D2 | De-noise step-by-step; `sdxl-turbo` in 4 steps |
| TTS with speaker embeddings | W3 D2 | A numeric "voice print" from a dataset |
| Training vs inference | W3 D2 | Adjusting weights vs using them |
| Per-model tokenizers | W3 D3 | `AutoTokenizer` — every model tokenizes differently |
| Chat templates | W3 D4 | `apply_chat_template` = what the API did for you |
| **Quantization** (4-bit, nf4) | W3 D4 | Big model, small GPU, tiny accuracy cost |
| Transformer anatomy | W3 D4 | Embedding → N × (attention + MLP + norm) → LM head |
| `generate()` + TextStreamer | W3 D4 | Token-by-token generation & streaming |
| `add_generation_prompt=True` | W3 D4 | Without it the model continues instead of answering |
| GPU memory hygiene | W3 D4 | `del` + `gc.collect()` + `empty_cache()` |
| **logprobs** | W3 D5 | See the probability of every candidate token |
| ASR / Whisper | W3 D5 | `pipeline("automatic-speech-recognition")` |
| Two-model pipelines | W3 D5 | Whisper → Llama = audio → minutes |
| Environment → prompt pattern | W4 all | `system_info.py` lets the LLM configure your machine |
| Code-fence stripping | W4 D3-5 | Models wrap code in ` ```cpp `; strip before writing files |
| `subprocess` execution | W4 D3-5 | `run(cmd, check=True, text=True, capture_output=True)` |
| stdout redirection + `exec` | W4 D4 | `io.StringIO` + `sys.stdout` swap to trap prints |
| Model→client routing dict | W4 D4 | One name → the right provider (Ollama/Groq/OpenRouter) |
| Benchmarking & speedup math | W4 D3-5 | `python_time / compiled_time`; run repeatedly |
| Correctness > speed | W4 D5 | Python's bigint vs C++/Rust overflow → FAIL |
| `gr.Code`, CSS themes | W4 D5 | IDE-style Gradio apps |

## The recurring skeleton (now with GPUs)

```
1. SETUP     pip installs → GPU check → login(HF token)
2. LOAD      pipeline(task)  —or—  AutoTokenizer + AutoModelForCausalLM (+ quant_config)
3. PROMPT    system + user messages → apply_chat_template → tensors on cuda
4. GENERATE  .generate(..., streamer) —or— pipeline(input); decode the token ids
5. CHAIN     feed output of model A into model B (Whisper → Llama; GPT → clang++)
6. CLEAN UP  del + gc.collect() + torch.cuda.empty_cache()
```

## Glossary additions

- **Pipeline** — HF's one-line wrapper connecting a task to a default model + preprocessing.
- **Quantization** — storing weights in fewer bits (4-bit NF4 here) to fit small GPUs.
- **Diffusion model** — generates images by iteratively removing noise.
- **Speaker embedding** — a vector encoding a voice's characteristics for TTS.
- **logprobs** — the model's per-token probability output; the window into its "confidence".
- **Attention / MLP / LM head** — the working parts of a Transformer layer.
- **Gated model** — an HF model you must request access to (Llama, Gemma).
- **Code fence** — the ` ```language ` markdown wrapper around generated code.
- **`subprocess`** — Python module for running external programs (compilers!).
- **LCG** — linear congruential generator, a classic pseudo-random number formula.
- **rustc / cargo** — the Rust compiler and its package/build manager.

## Running weeks 3–4 yourself

**Week 3 (Colab route — recommended):**
1. Free accounts: Google + [huggingface.co](https://huggingface.co). Create an HF token **with WRITE access**.
2. Open each `week3/dayN.ipynb`, click its Colab link, add `HF_TOKEN` under the 🔑 secrets, connect the **T4 GPU** runtime, run cells top-to-bottom.
3. For Llama/Gemma: accept license terms on the model pages first (takes ~2 minutes).
4. Day 5: put `denver_extract.mp3` (link in the Colab) into a Drive folder `llms/`, or upload your own audio. The local `day5.ipynb` visualizer needs only your `OPENAI_API_KEY` in `.env`.

**Week 4 (locally, in Cursor):**
1. `uv sync` from the repo root (per the lab's reminder) to refresh the environment.
2. Run `system_info.retrieve_system_info()` + the Day 3 prompt cell to get your machine's compile commands; paste them into `compile_command`/`run_command`. (Mac: install Xcode CLT or LLVM; Windows: MSVC or via WSL; or skip local execution and use an online compiler like programiz.com.)
3. Optional Rust: install via [rustup](https://rustup.rs) if you want Day 5's Rust mode.
4. Keep costs low by swapping in `gpt-5-nano` / `claude-haiku-4-5` / `gemini-3.1-flash-lite` (the notebooks include the exact lines).
5. Model outputs vary run to run — rerun the race and make your own leaderboard; that's half the fun.

## Where this leaves the course

You can now (1) run *any* open-source model on your own GPU, (2) chain models across modalities (audio→text→text), (3) generate, compile, execute and benchmark LLM-written systems code. Next: **Week 5** gives models knowledge bases with embeddings & RAG, **Weeks 6–7** fine-tune and train, and **Week 8** builds the fully autonomous multi-agent system that Week 4's "generate → execute → verify" loop already foreshadows.
