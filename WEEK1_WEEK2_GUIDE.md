# 📘 Week 1 & Week 2 — Complete Beginner's Guide

This guide explains **every core file** in the `week1/` and `week2/` folders of the *LLM Engineering* course repo — what each piece of code does, **why** it exists, and **how it all fits together**. No prior AI experience needed.

> **How to read this guide:** Each section covers one notebook/file in the order you'd study them. Code snippets are quoted from the actual files, followed by a plain-English walkthrough. A glossary and a "mental model" diagram are at the end.

---

# 🧠 PART 0 — The Mental Model (read this first!)

## What is an LLM, really?

An LLM (Large Language Model) like GPT, Claude, Gemini or Llama is a **text-prediction engine** living on a company's servers (or your own computer, if it's an open-source model). You send it text, it predicts the most likely continuation, and sends it back.

You never "run" the model yourself (in weeks 1–2). Instead, you **send a message over the internet** to an API (Application Programming Interface) — a web address that accepts requests and returns answers.

## The one API call you'll see 100 times

Almost every call in Weeks 1–2 looks like this:

```python
from openai import OpenAI
openai = OpenAI()   # the "client" — a thin wrapper around an HTTP call

messages = [
    {"role": "system",  "content": "You are a helpful assistant"},  # sets personality/task
    {"role": "user",    "content": "What is 2 + 2?"}                # what the human says
]

response = openai.chat.completions.create(model="gpt-4.1-mini", messages=messages)
print(response.choices[0].message.content)   # -> "2 + 2 equals 4."
```

Three things to burn into memory:

1. **`messages` is a list of dictionaries.** Each dict has a `role` and `content`.
   - `"system"` → invisible stage directions: who the assistant is, tone, rules.
   - `"user"` → the human's message.
   - `"assistant"` → the AI's previous replies (used to fake memory — see Week 1 Day 4).
   - `"tool"` → results of functions the AI asked you to run (see Week 2 Day 4).
2. **The response object is nested**: `response.choices[0].message.content` is the actual text.
3. **Every call is stateless.** The model remembers *nothing* between calls. You must re-send the whole conversation every time.

## The flow of both weeks

```
WEEK 1: learn to call models              WEEEK 2: build real products
                                          
Day 1  Website Summarizer (OpenAI)        Day 1  Talk to EVERY provider; make 2 bots argue
Day 2  Under the hood: HTTP, Gemini,      Day 2  Gradio: turn any function into a web UI
       and local models via Ollama        Day 3  A real streaming chatbot (ChatInterface)
Day 4  Tokens + the "memory illusion"     Day 4  TOOLS: let the AI call your Python/DB code
Day 5  Brochure Generator (multi-step     Day 5  Multimodal: + image gen, + voice, custom UI
       LLM pipeline) + streaming          Extra  Models drawing SVG art; end-of-week exercise
```

---

# 🟦 PART 1 — WEEK 1: Foundations

## 1.1 `week1/scraper.py` — the helper that reads websites

Both projects need website text as input, so the course provides a tiny web-scraper utility. **LLMs can't browse the web** — you must fetch the page and paste its text into the prompt.

```python
from bs4 import BeautifulSoup
import requests

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ..."}  # pretend to be a real browser

def fetch_website_contents(url):
    response = requests.get(url, headers=headers)          # 1. download the raw HTML
    soup = BeautifulSoup(response.content, "html.parser")  # 2. parse HTML into a searchable tree
    title = soup.title.string if soup.title else "No title found"
    if soup.body:
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()                         # 3. delete junk we don't want
        text = soup.body.get_text(separator="\n", strip=True)  # 4. extract just the text
    else:
        text = ""
    return (title + "\n\n" + text)[:2_000]                 # 5. truncate to 2,000 chars

def fetch_website_links(url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    links = [link.get("href") for link in soup.find_all("a")]  # every <a href="...">
    return [link for link in links if link]                    # drop empty ones
```

**Step by step:**

| Step | Code | Beginner translation |
|---|---|---|
| Download | `requests.get(url, headers=headers)` | Like typing a URL into a browser; returns raw HTML. The `User-Agent` header makes us look like Chrome so sites don't block us. |
| Parse | `BeautifulSoup(...)` | Turns a wall of HTML into a tree you can query ("give me the `<title>`", "give me the `<body>`"). |
| Clean | `decompose()` | Deletes `<script>`, `<style>`, `<img>`, `<input>` tags — menus, ads, code — leaving readable content. |
| Extract | `get_text(...)` | Pulls out only the visible words, one per line. |
| Truncate | `[:2_000]` | Keeps the prompt small. Long inputs cost more money and can exceed model limits. |

**Why it matters:** this is the classic "RAG-lite" pattern — get real-world data into the prompt.

**Limitations** (explained in the notebook): JavaScript-rendered sites (React apps, e.g. openai.com) return empty content, because `requests` only downloads the initial HTML. Sites behind Cloudflare may return 403 errors. Fixing that needs browser automation (Selenium/Playwright).

---

## 1.2 `week1/day1.ipynb` — Your first LLM project: a Website Summarizer

**Goal:** give it a URL → it replies with a summary. "The Reader's Digest of the internet."

### Cell-by-cell walkthrough

**(a) Imports & API key check**

```python
import os
from dotenv import load_dotenv
from scraper import fetch_website_contents        # our helper above
from IPython.display import Markdown, display     # pretty rendering inside Jupyter
from openai import OpenAI

load_dotenv(override=True)                        # reads the .env file into environment variables
api_key = os.getenv('OPENAI_API_KEY')             # pull the key out

if not api_key:
    print("No API key was found ...")             # friendly diagnostics for 3 common mistakes
elif not api_key.startswith("sk-proj-"):
    print("... doesn't start sk-proj- ...")
elif api_key.strip() != api_key:
    print("... space or tab characters ...")
else:
    print("API key found and looks good so far!")
```

- **`.env` file:** a local file (never committed to Git) holding secrets like `OPENAI_API_KEY=sk-proj-...`. `load_dotenv()` loads it; `os.getenv()` reads one variable.
- The `if/elif` chain is *defensive programming*: catch the three most common setup mistakes (missing key, wrong key type, stray whitespace) and print a helpful message instead of a cryptic failure later.

**(b) The very first call**

```python
message = "Hello, GPT! This is my first ever message to you! Hi!"
messages = [{"role": "user", "content": message}]

openai = OpenAI()
response = openai.chat.completions.create(model="gpt-5-nano", messages=messages)
response.choices[0].message.content
```

That's a complete LLM call in 3 lines. Notice `messages` is a *list* even with one item — the API always expects a list.

**(c) System prompt vs user prompt** — the key prompting concept:

```python
system_prompt = """
You are a snarky assistant that analyzes the contents of a website,
and provides a short, snarky, humorous summary, ignoring text that might be navigation related.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""

user_prompt_prefix = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.

"""
```

- **System prompt** = the job description + personality ("snarky"). Change this one word and the *same model* produces a completely different tone. This is the cheapest, most powerful knob in LLM engineering.
- **User prompt** = the actual request + the data. It's called a *prefix* because the website text gets glued onto the end of it.

**(d) Assembling messages with a function**

```python
def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_prefix + website}   # prefix + scraped text
    ]
```

**(e) The whole product in one function**

```python
def summarize(url):
    website = fetch_website_contents(url)                # 1. scrape
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages_for(website)                   # 2. build prompt
    )
    return response.choices[0].message.content           # 3. unwrap the answer

def display_summary(url):
    summary = summarize(url)
    display(Markdown(summary))                           # render as rich text in Jupyter
```

**The full pipeline:** `URL → requests+BeautifulSoup → text → messages list → OpenAI API → summary text → Markdown`.

The notebook then calls `display_summary("https://edwarddonner.com")`, `cnn.com`, `anthropic.com` to show it working on real sites.

**(f) Your turn (bottom of the notebook):** a skeleton asking you to build an **email subject-line generator** — same 4 steps: prompts → messages → call → print. Every LLM app you'll ever build is a variation of these 4 steps.

---

## 1.3 `week1/day2.ipynb` — How the API *really* works + free local models

This day pulls back the curtain. Three big ideas:

### Idea 1 — The `openai` package is just an HTTP wrapper

Before using the library, the notebook makes the identical call with raw HTTP:

```python
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

payload = {
    "model": "gpt-5-nano",
    "messages": [{"role": "user", "content": "Tell me a fun fact"}]
}

response = requests.post("https://api.openai.com/v1/chat/completions",
                         headers=headers, json=payload)
response.json()["choices"][0]["message"]["content"]
```

This is literally the same request the library makes: an HTTPS POST with JSON, authenticated by your key in the `Authorization` header. **The `openai` package contains no AI** — it's a convenience wrapper so you write Python instead of hand-built JSON.

### Idea 2 — "OpenAI-compatible endpoints": one client, many providers

The Chat Completions format became so popular that other providers cloned it. OpenAI's client lets you swap the destination:

```python
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)

response = gemini.chat.completions.create(model="gemini-3.1-flash-lite",
                                          messages=[{"role": "user", "content": "Tell me a fun fact"}])
```

Even though the code says `OpenAI(...)`, **no OpenAI model is involved** — we're just reusing the client to talk to Google's endpoint. This one trick powers half of Week 2.

### Idea 3 — Ollama: free models on your own machine

[Ollama](https://ollama.com) runs open-source models locally and *also* exposes an OpenAI-compatible endpoint at `http://localhost:11434/v1`:

```python
!ollama pull llama3.2                        # shell command: download Meta's Llama 3.2 (~2GB)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')   # api_key is ignored, but required

response = ollama.chat.completions.create(model="llama3.2",
                                          messages=[{"role": "user", "content": "Tell me a fun fact"}])
```

The notebook also pulls `deepseek-r1:1.5b` (DeepSeek reasoning distilled into Alibaba's Qwen). Note `api_key='ollama'` is a dummy — local servers don't check keys, but the client demands *something*.

**Homework given at the end of day 2:** rewrite the Day 1 summarizer to use Ollama instead of OpenAI. The official answer is `week1/solution.py` (same code, but `OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')` and `model="llama3.2"`), plus `solutions/day1_with_ollama.ipynb`.

**Trade-offs** (from the notebook):
- ✅ No API cost; data never leaves your machine
- ❌ Much weaker than frontier models; needs a decent computer

---

## 1.4 `week1/day4.ipynb` — Tokens, and the "illusion of memory"

Two "aha" lessons packed into one short notebook.

### Lesson A — Tokens: how LLMs see text

Models don't read letters or words; they read **tokens** — chunks of text mapped to numbers.

```python
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4.1-mini")

tokens = encoding.encode("Hi my name is Ed and I like banoffee pie")
# -> [13321, 617, 649, 282, 356, 322, 302, 905, 1944, 30903, 8069]  (just numbers!)

for token_id in tokens:
    print(f"{token_id} = {encoding.decode([token_id])}")
```

Output shows words split into pieces like `Hi`, ` my`, ` name`… Rare words split further (`banoffee` → multiple pieces). Why care?

- **Cost** is priced per token (input + output).
- **Context limits** are token counts, not words.
- The model literally predicts the next *token id*.

### Lesson B — LLMs have NO memory

```python
# Call 1
messages = [{"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hi! I'm Ed!"}]
# -> model replies "Hi Ed! ..."

# Call 2 — a brand-new call that omits the introduction
messages = [{"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What's my name?"}]
# -> "I'm sorry, but you haven't told me your name."   😱
```

Then the fix:

```python
messages = [
    {"role": "system",    "content": "You are a helpful assistant"},
    {"role": "user",      "content": "Hi! I'm Ed!"},
    {"role": "assistant", "content": "Hi Ed! How can I assist you today?"},   # replayed history!
    {"role": "user",      "content": "What's my name?"}
]
# -> "Your name is Ed!"   ✅
```

**The punchline:** every API call is stateless. "Memory" is a trick — *you* keep the conversation history in Python and re-send all of it every time. ChatGPT does exactly this. Consequences:
- You pay for the full conversation length on every turn (tokens grow as the chat grows).
- Week 2 Day 3 will automate this with Gradio's `history` parameter.

---

## 1.5 `week1/day5.ipynb` — The Brochure Generator (a multi-step LLM pipeline)

**Business challenge:** given a company name + website, generate a brochure for customers, investors and recruits. One page isn't enough — we want the About, Careers and Company pages too. But which links matter? That judgment call is *perfect* for an LLM.

**This is the first "agentic" pattern of the course: multiple LLM calls chained together, where call #1's output feeds call #2's input.**

### Step 1 — Ask the LLM to pick relevant links (and answer as JSON)

```python
link_system_prompt = """
You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include in a brochure about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.
You should respond in JSON as in this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""
```

Two techniques here:
- **One-shot prompting:** we show the model a worked example of the exact answer format we want.
- **Structured output via JSON mode:** the API call adds `response_format={"type": "json_object"}`, forcing valid JSON we can parse with `json.loads()` — no fragile text parsing.

```python
def get_links_user_prompt(url):
    user_prompt = f"""
Here is the list of links on the website {url} -
Please decide which of these are relevant web links for a brochure about the company,
respond with the full https URL in JSON format.
Do not include Terms of Service, Privacy, email links.
...
"""
    links = fetch_website_links(url)
    user_prompt += "\n".join(links)          # dump all raw links into the prompt
    return user_prompt

def select_relevant_links(url):
    response = openai.chat.completions.create(
        model=MODEL,   # 'gpt-5-nano'
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user",   "content": get_links_user_prompt(url)}
        ],
        response_format={"type": "json_object"}
    )
    links = json.loads(response.choices[0].message.content)
    return links   # {"links": [{"type": "about page", "url": "..."}, ...]}
```

**Why this is a great LLM use case:** deciding that `/about` matters but `/privacy` doesn't requires *nuanced understanding* — nightmarish to hand-code with string rules, trivial for an LLM.

### Step 2 — Scrape the landing page + every relevant link

```python
def fetch_page_and_all_relevant_links(url):
    contents = fetch_website_contents(url)
    relevant_links = select_relevant_links(url)          # ← LLM call #1 inside!
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    for link in relevant_links['links']:
        result += f"\n\n### Link: {link['type']}\n"
        result += fetch_website_contents(link["url"])    # scrape each chosen page
    return result
```

### Step 3 — Generate the brochure

```python
brochure_system_prompt = """
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""

def get_brochure_user_prompt(company_name, url):
    user_prompt = f"""
You are looking at a company called: {company_name}
Here are the contents of its landing page and other relevant pages;
use this information to build a short brochure of the company in markdown without code blocks.\n\n
"""
    user_prompt += fetch_page_and_all_relevant_links(url)
    user_prompt = user_prompt[:5_000]        # truncate: control cost & context size
    return user_prompt

def create_brochure(company_name, url):
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": brochure_system_prompt},
                  {"role": "user",   "content": get_brochure_user_prompt(company_name, url)}])
    display(Markdown(response.choices[0].message.content))
```

### Step 4 — Streaming (the typewriter effect)

```python
def stream_brochure(company_name, url):
    stream = openai.chat.completions.create(model="gpt-4.1-mini", messages=[...], stream=True)
    response = ""
    display_handle = display(Markdown(""), display_id=True)   # create an empty output box
    for chunk in stream:                                      # text arrives in pieces
        response += chunk.choices[0].delta.content or ''      # append each piece
        update_display(Markdown(response), display_id=display_handle.display_id)  # redraw the box
```

- `stream=True` makes the API send the answer as many small **chunks** (`delta.content`) instead of one big block. Same total tokens, but the user sees progress instantly.
- `display(...)` + `update_display(...)` keep refreshing the *same* Jupyter output cell → the typewriter animation.

**Tone trick:** swap the system prompt to the commented-out "humorous, entertaining, witty" version and the *same pipeline* produces a jokey brochure.

---

## 1.6 `week1/week1 EXERCISE.ipynb` + `solutions/week1 SOLUTION.ipynb`

**Challenge:** build a **technical tutor** — paste a technical question, get an explanation, from *both* GPT and local Llama 3.2.

The solution notebook, fully commented:

```python
MODEL_GPT = 'gpt-4o-mini'
MODEL_LLAMA = 'llama3.2'

question = """
Please explain what this code does and why:
yield from {book.get("author") for book in books if book.get("author")}
"""

system_prompt = "You are a helpful technical tutor who answers questions about python code, software engineering, data science and LLMs"
user_prompt = "Please give a detailed explanation to the following question: " + question

messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]

# 1) GPT with streaming
stream = openai.chat.completions.create(model=MODEL_GPT, messages=messages, stream=True)
response = ""
display_handle = display(Markdown(""), display_id=True)
for chunk in stream:
    response += chunk.choices[0].delta.content or ''
    response = response.replace("```", "").replace("markdown", "")   # strip code fences so Jupyter renders it
    update_display(Markdown(response), display_id=display_handle.display_id)

# 2) Llama 3.2 locally via the dedicated ollama client
import ollama
response = ollama.chat(model=MODEL_LLAMA, messages=messages)
display(Markdown(response['message']['content']))
```

Note the two ways to reach Ollama: the **OpenAI-compatible endpoint** (Day 2 style) or the purpose-built **`ollama` python package** (`ollama.chat(...)` returning a dict) — both are shown in this solution.

**Week 1 takeaway:** *scrape → prompt (system+user) → messages list → API call → unwrap `.choices[0].message.content`* is the universal skeleton; multi-call pipelines + streaming make it a product.

---

# 🟩 PART 2 — WEEK 2: APIs, UIs, Chatbots & Tools

## 2.1 `week2/day1.ipynb` — The model marketplace & two bots arguing

The busiest notebook: a tour of **every major way to reach an LLM**.

### (a) One client, seven providers

All keys load from `.env`, then — thanks to OpenAI-compatible endpoints — every provider gets an `OpenAI` client with a different `base_url`:

```python
anthropic_url = "https://api.anthropic.com/v1/"          # Claude
gemini_url    = "https://generativelanguage.googleapis.com/v1beta/openai/"
deepseek_url = "https://api.deepseek.com"
groq_url      = "https://api.groq.com/openai/v1"         # ultra-fast inference host
grok_url      = "https://api.x.ai/v1"
openrouter_url= "https://openrouter.ai/api/v1"           # ONE api → hundreds of models
ollama_url    = "http://localhost:11434/v1"              # your own machine, free

anthropic = OpenAI(api_key=anthropic_api_key, base_url=anthropic_url)
gemini    = OpenAI(api_key=google_api_key,    base_url=gemini_url)
# ... etc
```

After that, swapping models is one line: `anthropic.chat...`, `gemini.chat...`, `ollama.chat...`.

### (b) Reasoning effort — "thinking" as a dial

```python
easy_puzzle = [{"role": "user", "content":
    "You toss 2 coins. One of them is heads. What's the probability the other is tails? ..."}]

openai.chat.completions.create(model="gpt-5-nano", messages=easy_puzzle, reasoning_effort="minimal")
openai.chat.completions.create(model="gpt-5-nano", messages=easy_puzzle, reasoning_effort="low")
```

**Training-time vs inference-time scaling:** older models got smarter only by being *trained bigger* (expensive, infrequent). New "reasoning" models get smarter at *question time* by generating internal thinking tokens first. `reasoning_effort` is the dial: more thinking = better answers, more cost & latency. The notebook shows the same puzzle answered with different effort levels.

It then tests a classic trick question (the bookshelf worm puzzle) across `gpt-5-nano`, `claude-sonnet-4-5`, `gpt-5` and `gemini-2.5-pro` — some fall for it, some don't. A game-show "Steal or Share?" dilemma is put to Claude, GPT-OSS (via Groq), DeepSeek-Reasoner and Grok — a fun way to compare model *personalities*.

### (c) Native SDKs (for completeness)

```python
from google import genai
client = genai.Client()
response = client.models.generate_content(model="gemini-3.1-flash-lite", contents="Describe the color Blue ...")

from anthropic import Anthropic
client = Anthropic()
response = client.messages.create(model="claude-sonnet-4-5-20250929", messages=[...], max_tokens=100)
```

Different providers have their own libraries with different method names — but note `max_tokens` is *required* by Anthropic's native API (the OpenAI-compatible endpoint handles it for you).

### (d) Abstraction layers: OpenRouter, LangChain, LiteLLM

```python
# OpenRouter: one key → hundreds of models (here GLM 4.5 from Chinese startup z.ai)
openrouter.chat.completions.create(model="z-ai/glm-4.5", messages=tell_a_joke)

# LangChain: the heavyweight framework
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5-mini"); llm.invoke(tell_a_joke)

# LiteLLM: the featherweight router + automatic cost tracking
from litellm import completion
response = completion(model="openai/gpt-4.1", messages=tell_a_joke)
print(response._hidden_params["response_cost"])   # it computes what the call cost!
```

### (e) Prompt caching — paying (much) less for repeated prompts

The demo loads the full text of *Hamlet* (`week2/hamlet.txt`), asks a question about it twice, and prints token/cost stats via LiteLLM:

```python
with open("hamlet.txt", "r", encoding="utf-8") as f:
    hamlet = f.read()

question = [{"role": "user", "content": "In Hamlet, when Laertes asks 'Where is my father?' what is the reply?"}]

# 1st try WITHOUT the play: the model can't know / guesses
completion(model="gemini/gemini-3.1-flash-lite", messages=question)

# 2nd try: glue the entire play into the prompt
question[0]["content"] += "\n\nFor context, here is the entire text of Hamlet:\n\n" + hamlet
completion(...)   # expensive: ~200k input tokens, first time

# 3rd try: SAME prompt again
completion(...)   # cached_tokens jumps up, cost collapses
```

**How caching works:** providers hash the *prefix* of your prompt; if a later request starts with exactly the same tokens, the pre-computed work is reused.
- OpenAI/Gemini: automatic ("implicit"), ~4× cheaper on cache hits → put static content **first**, variable content **last**.
- Anthropic: explicit — you mark what to cache, pay 25% *more* to write the cache, then 10× *less* to read it.

### (f) ⭐ The finale: an adversarial conversation between two chatbots

The most important 30 lines of Week 2 Day 1 — this is the pattern every chatbot builds on:

```python
gpt_model = "gpt-4.1-mini"
claude_model = "claude-haiku-4-5"

gpt_system    = "You are a chatbot who is very argumentative; you disagree with anything..."
claude_system = "You are a very polite, courteous chatbot. You try to agree with everything..."

gpt_messages = ["Hi there"]      # plain Python lists holding each side's lines
claude_messages = ["Hi"]

def call_gpt():
    messages = [{"role": "system", "content": gpt_system}]
    for gpt, claude in zip(gpt_messages, claude_messages):     # weave the two histories
        messages.append({"role": "assistant", "content": gpt}) # GPT's own lines...
        messages.append({"role": "user", "content": claude})   # ...and Claude's as "user"
    response = openai.chat.completions.create(model=gpt_model, messages=messages)
    return response.choices[0].message.content

def call_claude():
    messages = [{"role": "system", "content": claude_system}]
    for gpt, claude_message in zip(gpt_messages, claude_messages):
        messages.append({"role": "user",      "content": gpt})          # mirrored!
        messages.append({"role": "assistant", "content": claude_message})
    messages.append({"role": "user", "content": gpt_messages[-1]})      # the new line to reply to
    response = anthropic.chat.completions.create(model=claude_model, messages=messages)
    return response.choices[0].message.content

# The debate loop
for i in range(5):
    gpt_next = call_gpt();    gpt_messages.append(gpt_next)     # GPT speaks, remember it
    claude_next = call_claude(); claude_messages.append(claude_next)  # Claude responds, remember it
```

**How to read it:** each model must believe *it* is the assistant. So for `call_gpt`, GPT's past lines get `role="assistant"` and Claude's lines get `role="user"` — and `call_claude` mirrors exactly that. Two separate history lists + two "viewpoints" on the same conversation.

The exercise at the end: extend to a 3-way conversation (the suggested trick: keep ONE shared conversation string and give each bot the transcript as text in the user prompt — many examples live in `community-contributions/`).

---

## 2.2 `week2/day2.ipynb` — Gradio: instant web UIs

**Gradio** turns a Python function into a shareable web page with one line. This is how every project from now on gets a face.

### The minimal app

```python
import gradio as gr

def shout(text):
    return text.upper()

gr.Interface(fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never").launch()
```

`gr.Interface` = "one function → one screen". `launch()` starts a local web server and opens the UI. `flagging_mode="never"` just disables Gradio's flag/report buttons. Variants shown: `launch(share=True)` (public tunnel URL), `launch(inbrowser=True)` (auto-open tab), `auth=("ed", "bananas")` (simple login), and a JavaScript snippet to force dark mode.

### A prettier UI + a real model

```python
message_input = gr.Textbox(label="Your message:", info="Enter a message for GPT-4.1-mini", lines=7)
message_output = gr.Textbox(label="Response:", lines=8)

view = gr.Interface(fn=message_gpt, title="GPT",
                    inputs=[message_input], outputs=[message_output],
                    examples=["hello", "howdy"], flagging_mode="never")
view.launch()
```

Components (`gr.Textbox`) replace strings for labels/sizes; `examples` gives clickable sample inputs; using `gr.Markdown` as the output renders the LLM's markdown properly.

### ⭐ Streaming with Python generators (`yield`)

```python
def stream_gpt(prompt):
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    stream = openai.chat.completions.create(model='gpt-4.1-mini', messages=messages, stream=True)
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result          # ← hands the partial answer to Gradio, which re-renders live
```

- A function with `yield` is a **generator**: each loop iteration *pauses* and hands `result` (the partial answer so far) to the caller — here Gradio, which updates the output box → typewriter effect in the browser. (Course guide: "Intermediate Python" covers generators.)
- `stream_model(prompt, model)` then uses `yield from result` to dispatch between `stream_gpt` and `stream_claude` from a `gr.Dropdown` selector — the model-picker UI.

### Capstone: the Brochure Generator gets a UI

Week 1 Day 5's logic, wrapped as a 3-input streaming app:

```python
def stream_brochure(company_name, url, model):
    yield ""                                                  # clear the box immediately
    prompt = f"Please generate a company brochure for {company_name}. Here is their landing page:\n"
    prompt += fetch_website_contents(url)
    if model == "GPT":      result = stream_gpt(prompt)
    elif model == "Claude": result = stream_claude(prompt)
    yield from result
```

Inputs: `[name_input, url_input, model_selector]` → one Markdown output. Type a company + URL in a browser and watch the brochure type itself.

---

## 2.3 `week2/day3.ipynb` — A real chatbot (`gr.ChatInterface`)

`gr.Interface` is stateless — it forgot your last message. A chatbot needs history. Gradio's `gr.ChatInterface` **manages history for you** and passes it to your callback:

```python
def chat(message, history):
    return "bananas"                       # step 1: prove the wiring works

gr.ChatInterface(fn=chat, type="messages").launch()
```

`history` arrives as a list of `{"role": ..., "content": ...}` dicts (`type="messages"` format). Next iteration shows what's inside by printing it into the reply.

### The real chat callback — the most important function of Week 2

```python
def chat(message, history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]      # normalize
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response                   # generator → streaming chatbot
```

Exactly the Week 1 Day 4 "memory illusion" pattern, automated: **system + full history + new message**, streamed via `yield`. (The dict-comprehension line copies Gradio's history objects into plain dicts — defensive tidying.)

### Prompt engineering in action (clothes-store assistant)

```python
system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage
the customer to try items that are on sale. Hats are 60% off, ... Encourage the customer to buy hats..."

system_message += "\nIf the customer asks for shoes, you should respond that shoes are not on sale today,
but remind the customer to look at hats!"
```

Then — dynamic, per-message prompting:

```python
def chat(message, history):
    ...
    relevant_system_message = system_message
    if 'belt' in message.lower():        # Python inspects the user's text!
        relevant_system_message += " The store does not sell belts; if you are asked for belts,
                                    be sure to point out other items on sale."
    messages = [{"role": "system", "content": relevant_system_message}] + history + [{"role": "user", "content": message}]
    ...
```

**Key lesson:** *your code and the LLM share control.* You can inspect the user's message in plain Python and inject extra instructions into the system prompt before calling the model. This hybrid "code + model" design shows up again immediately in Day 4.

---

## 2.4 `week2/day4.ipynb` — ⭐ TOOLS: giving the AI hands

**Project:** an airline customer-support assistant ("FlightAI"). The killer feature: the LLM can answer *"How much is a flight to Tokyo?"* using **your data**, by asking *you* to run *your* function. This is called **tool calling / function calling** and it's the foundation of all agentic AI.

### The tool: a plain Python function

```python
ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

def get_ticket_price(destination_city):
    price = ticket_prices.get(destination_city.lower(), "Unknown ticket price")
    return f"The price of a ticket to {destination_city} is {price}"
```

Nothing magical — an ordinary function with a dictionary. The magic is in *telling the model it exists*.

### Describing the tool in JSON (the model reads this, not your code)

```python
price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {"type": "string",
                                 "description": "The city that the customer wants to travel to"},
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": price_function}]
```

The `description` fields are prompt engineering: the model decides *when* and *how* to call the tool based on these words alone.

### The handshake — who actually runs the function?

> **Important:** the LLM *never* runs code. It can only *request* a tool call; your Python executes it and returns the result. The conversation then continues with the result in context.

```python
def chat(message, history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)  # pass tools!

    if response.choices[0].finish_reason == "tool_calls":        # model wants a tool
        message = response.choices[0].message                    # contains the requested call
        response = handle_tool_call(message)                     # WE run it
        messages.append(message)                                 # remember the request
        messages.append(response)                                # remember our result
        response = openai.chat.completions.create(model=MODEL, messages=messages)  # ask again

    return response.choices[0].message.content


def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    if tool_call.function.name == "get_ticket_price":
        arguments = json.loads(tool_call.function.arguments)     # JSON string → dict
        city = arguments.get('destination_city')
        price_details = get_ticket_price(city)                   # the real work
        response = {"role": "tool", "content": price_details, "tool_call_id": tool_call.id}
    return response
```

**The full round trip, in order:**

```
1. User:  "How much for Tokyo?"
2. Python → API:   messages + tools list
3. Model responds with finish_reason="tool_calls"
   and message.tool_calls[0] = {name:"get_ticket_price",
                                arguments:'{"destination_city":"Tokyo"}'}   ← JSON string!
4. Python: json.loads the arguments, calls get_ticket_price("Tokyo") → "$1400"
5. Python appends TWO messages: the assistant's tool request, then
   {"role":"tool", "content":"The price ... $1400", "tool_call_id": ...}
6. Python calls the API AGAIN. Now the model sees the result
   and writes: "A return ticket to Tokyo is $1,400." ✨
```

The `tool_call_id` ties each result to its request (essential when several tools are called at once).

### Two robustness upgrades shown in the notebook

**Multiple tool calls in one response** — `handle_tool_calls` (plural) loops over `message.tool_calls` and returns a *list* of tool messages; `chat` uses `messages.extend(responses)`.

**Chained tool calls** — change `if` to `while` so the model can request tools repeatedly:

```python
while response.choices[0].finish_reason == "tool_calls":
    message = response.choices[0].message
    responses = handle_tool_calls(message)
    messages.append(message)
    messages.extend(responses)
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
```

This little `while` loop is a genuine **agent loop**: model → tool → model → tool → … until the model finally answers in words.

### Upgrading the tool to a real database

The dictionary becomes an SQLite database (a file-based DB included with Python):

```python
import sqlite3
DB = "prices.db"

with sqlite3.connect(DB) as conn:
    conn.cursor().execute('CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price REAL)')

def get_ticket_price(city):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT price FROM prices WHERE city = ?', (city.lower(),))  # ? = safe param
        result = cursor.fetchone()
        return f"Ticket price to {city} is ${result[0]}" if result else "No price data available for this city"

def set_ticket_price(city, price):   # seed data + the basis of the homework
    with sqlite3.connect(DB) as conn:
        conn.cursor().execute('INSERT INTO prices (city, price) VALUES (?, ?)
                               ON CONFLICT(city) DO UPDATE SET price = ?', (city.lower(), price, price))
```

Same tool interface — now the data is persistent and updatable. **Exercise:** register a second tool (`set_ticket_price`) so users can say *"set the price to Berlin to $525"* and the assistant updates the database. The `?` placeholders prevent SQL injection (never f-string user data into SQL).

---

## 2.5 `week2/day5.ipynb` — Multimodal: voice + images + custom layout

The finale assembles everything into a **multimodal assistant**: it chats, calls tools, *speaks* its answers aloud, and *paints* a picture of your destination.

### Image generation

```python
import base64
from io import BytesIO
from PIL import Image

def artist(city):
    image_response = openai.images.generate(
        model="gpt-image-1-mini",
        prompt=f"An image representing a vacation in {city}, showing tourist spots ... pop-art style",
        size="1024x1024", n=1)
    image_base64 = image_response.data[0].b64_json     # image arrives as base64 text
    image_data = base64.b64decode(image_base64)        # decode to raw bytes
    return Image.open(BytesIO(image_data))             # wrap as a PIL Image
```

A *different* endpoint (`images.generate` instead of `chat.completions`). Base64 = a way to carry binary data inside text.

### Text-to-speech

```python
def talker(message):
    response = openai.audio.speech.create(model="gpt-4o-mini-tts", voice="onyx", input=message)
    return response.content        # raw audio bytes, ready for an <audio> player
```

### The upgraded agent loop — now with side-effects

```python
def chat(history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    cities = []
    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        responses, cities = handle_tool_calls_and_return_cities(message)   # ALSO collects cities
        messages.append(message); messages.extend(responses)
        response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    reply = response.choices[0].message.content
    history += [{"role": "assistant", "content": reply}]   # append to OUR history (we manage it now)
    voice = talker(reply)                                  # 🔊 speak the answer
    if cities:
        image = artist(cities[0])                          # 🎨 draw the destination
    return history, voice, image
```

`handle_tool_calls_and_return_cities` is the Day 4 loop, plus it collects which cities were looked up so the app knows what to draw.

### The custom UI: `gr.Blocks`

Gradio's three UI tiers: `gr.Interface` (simple form) → `gr.ChatInterface` (chatbot) → **`gr.Blocks`** (full layout control):

```python
def put_message_in_chatbot(message, history):
    return "", history + [{"role": "user", "content": message}]   # clear box, append user msg

with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500, interactive=False)
    with gr.Row():
        audio_output = gr.Audio(autoplay=True)
    with gr.Row():
        message = gr.Textbox(label="Chat with our AI Assistant:")

    message.submit(put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]).then(
        chat, inputs=chatbot, outputs=[chatbot, audio_output, image_output])
```

- `with gr.Row():` = components side by side (chat left, image right).
- **Event chaining:** pressing Enter runs `put_message_in_chatbot` first (updates history), then `.then(chat, ...)` runs the agent with the new history and writes back chat text + audio + image.
- Here *we* own the history list (no `ChatInterface` managing it) — the same statelessness lesson, now fully manual.

---

## 2.6 `week2/extra.ipynb` + `week2/revealer.py` — Models drawing SVG art

A fun coda inspired by Simon Willison's "pelican riding a bike" test: ask top models to write an **SVG** (an image described as XML shapes — a *coding* task, very different from diffusion-based image generation):

```python
challenge = "a panda rollerblading to work"
prompt = f"Generate an SVG of {challenge}. Respond with the SVG only, no code blocks."

def artist(model, effort=None):
    response = openrouter.chat.completions.create(model=model, messages=messages, reasoning_effort=effort)
    return heading, response.choices[0].message.content

results = [artist("openai/gpt-oss-120b"), artist("openai/gpt-5-nano", effort="low"),
           artist("deepseek/deepseek-v3.2"), artist("moonshotai/kimi-k2-thinking"),
           artist("x-ai/grok-4.1-fast"), artist("anthropic/claude-opus-4.5"),
           artist("openai/gpt-5.2", effort="high"), artist("google/gemini-3-pro-preview")]
```

All through **OpenRouter** — one client, eight different labs' models, side by side.

`revealer.py` makes the results watchable: it parses each model's SVG with `xml.etree.ElementTree`, injects a `<style>` block with a CSS `@keyframes reveal` animation, and staggers `animation-delay: 0.15s` across every drawable shape (`path`, `circle`, `rect`, …) — so the picture *draws itself* shape-by-shape in the notebook.

---

## 2.7 `week2/week2 EXERCISE.ipynb` — The end-of-week challenge

Combine everything into a polished **Technical Tutor** app (upgrading Week 1's exercise):

- ✅ Gradio UI (ChatInterface)
- ✅ Streaming answers (`yield`)
- ✅ System prompt giving it expertise/persona
- ✅ Model switcher (GPT / Claude / Ollama dropdown like Day 2)
- 🌟 Bonus: a **tool** (e.g. look up documentation)
- 🌟 Bonus: **audio in/out** (Gradio `gr.Audio(sources=["microphone"])` + `talker()`)

Commercial shapes this could take: language tutor, company onboarding assistant, course companion.

---

# 🎓 PART 3 — Concepts, patterns & cheat sheets

## Concepts introduced, week by week

| Concept | Where | One-line summary |
|---|---|---|
| Chat Completions API | W1 D1 | Send `messages` list → get `choices[0].message.content` |
| System vs user prompt | W1 D1 | System = persona/rules; user = request + data |
| OpenAI-compatible endpoints | W1 D2 | Same client code, different `base_url` → Gemini, DeepSeek, Groq, Ollama… |
| Ollama / local models | W1 D2 | Free, private, weaker; endpoint on `localhost:11434` |
| Tokens | W1 D4 | Text → numbers; cost & context are measured in tokens |
| Statelessness & history | W1 D4 | No memory — you re-send everything, every call |
| One-shot prompting | W1 D5 | Put an example of the desired output in the prompt |
| JSON mode | W1 D5 | `response_format={"type":"json_object"}` → parseable output |
| Multi-step LLM pipelines | W1 D5 | Output of call 1 (link picker) feeds call 2 (brochure writer) |
| Streaming | W1 D5, W2 D2/3 | `stream=True` + loop over chunks + `yield`/`update_display` |
| Reasoning effort | W2 D1 | Thinking-token dial: minimal→low→medium→high |
| Prompt caching | W2 D1 | Identical prompt prefix ⇒ drastically cheaper repeats |
| Role-played conversations | W2 D1 | Two history lists; each model sees itself as "assistant" |
| Gradio Interface | W2 D2 | `fn + inputs + outputs` = instant web app |
| Generators (`yield`) | W2 D2 | Pause/resume functions = live streaming UIs |
| ChatInterface & history | W2 D3 | Gradio keeps history, passes it to `chat(message, history)` |
| Dynamic system prompts | W2 D3 | Python inspects the message and augments the prompt |
| Tool/function calling | W2 D4 | Model *requests*; you execute; append `role:"tool"` result; call again |
| Agent loop | W2 D4/5 | `while finish_reason == "tool_calls":` — the seed of agentic AI |
| SQLite as a tool backend | W2 D4 | Persistent real data behind a tool |
| Multimodal | W2 D5 | `images.generate` (art), `audio.speech` (voice), Blocks UI |
| gr.Blocks event chaining | W2 D5 | `.submit(...).then(...)` pipelines between callbacks |

## The universal app skeleton (everything above, compressed)

```python
1. GATHER     fetch/scrape/receive real-world data        (scraper.py, DB, user input)
2. PROMPT     system prompt (persona+rules) + user prompt (data + ask), examples if needed
3. CALL       client.chat.completions.create(model, messages[, tools][, stream=True])
4. LOOP       if the model asked for tools: run them, append role:"tool" results, call again
5. STREAM     yield partial results to Gradio for live UI
6. REMEMBER   append the exchange to history; send full history next turn
```

## Glossary

- **LLM / Frontier model** — a large text-prediction model / the best models available (GPT, Claude, Gemini…).
- **API endpoint** — a URL a program can call, e.g. `https://api.openai.com/v1/chat/completions`.
- **Client library** — a thin wrapper turning Python calls into HTTP requests (`openai`, `anthropic`).
- **Prompt** — text sent to a model. **System** (instructions), **user** (human), **assistant** (model's own past words), **tool** (function results).
- **Token** — chunk of text (~¾ of a word) that models actually read; you pay per token.
- **Inference** — running a trained model to get an answer (vs *training* it).
- **Streaming** — receiving the answer in small chunks as it's generated.
- **JSON mode** — forcing the model to answer in machine-parseable JSON.
- **Tool calling** — the model outputs a structured "please run this function" request instead of prose.
- **Agent / Agentic AI** — LLMs wired into loops of tools and decisions so they can *act*, not just talk (the `while` loop of W2 D4 is the embryo; Week 8 builds the full thing).
- **Gradio** — Python library that wraps a function in a browser UI.
- **Ollama** — local server that runs open-source models and mimics the OpenAI API.
- **`.env` / environment variable** — a local secret file loaded with `load_dotenv()`, never committed to Git.

## Running these notebooks yourself

1. Complete the repo's `setup/` instructions (creates a `.venv` with every dependency and a `.env` file).
2. Open each notebook in Cursor/Jupyter, select the `.venv (Python 3.12.x)` kernel, and run cells **top to bottom** (Shift+Enter).
3. API keys go in `.env` (`OPENAI_API_KEY=sk-proj-...`, plus optional Anthropic/Google/… keys). Save the file and re-run `load_dotenv(override=True)` after editing.
4. For local models: install [Ollama](https://ollama.com), then `ollama pull llama3.2` (or `llama3.2:1b` on small machines). Keep it to llama3.2 — llama3.3 is 70B, too big for most laptops.
5. Cost control: prefer `gpt-4.1-nano`/`gpt-4.1-mini`; week 1–2 exercises cost cents, not dollars. Ollama is 100% free.
6. Common hiccups: `NameError` → run cells in order; 401 errors → bad/absent key; empty summaries → the site needs JavaScript (try another site); Ollama calls fail → run `ollama serve`.

## Where the course goes next (for context)

- **Week 3–4:** running and *evaluating* open-source models with GPUs (Colab), building benchmarks.
- **Week 5:** RAG — giving models a knowledge base with embeddings.
- **Week 6–7:** fine-tuning your own model.
- **Week 8:** a fully autonomous multi-agent system — the grown-up version of Day 4's little `while` loop.

*Everything in `week1/community-contributions/` and `week2/community-contributions/` is student homework variations of these same projects — great for inspiration once the core notebooks make sense.*
