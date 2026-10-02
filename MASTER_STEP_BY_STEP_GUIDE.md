# 🚀 MASTER STEP-BY-STEP GUIDE — How Every Project Actually Works
### ✨ Beginner Edition — every step explained in plain English

This is the **execution companion** to the six per-week guides (`WEEK1_WEEK2_GUIDE.md` … `WEEK8_GUIDE.md`). The week guides explain *concepts and why*. **This guide traces exactly what happens, in order, when the code runs** — what goes in, what gets transformed, what comes out, and what you'll see on screen.

**What's new in this edition:** every step now has a 🧠 **plain-English** translation, real-world 🏨 **analogies**, ⚠️ **common mistakes**, and ✅ **checkpoint questions** so you can test yourself as you go. No prior ML experience assumed — only basic Python.

> **Legend used throughout:**
> **STEP n** = the true execution order. **DATA** = what a variable *actually contains* at that moment. 🖥️ = what appears on your screen. 🧠 = plain-English translation. 🏨 = analogy. ⚠️ = trap to avoid.

---

<a id="primer"></a>
# 0️⃣ START HERE — 10 ideas that unlock the entire course

Read these once. Every one of the 16 sections that follow is built from these same ten blocks.

### 1. An LLM is a "text in → text out" machine you reach over the internet
There is no magic import. "Calling GPT" just means: **send an HTTPS request containing text to OpenAI's servers; a few seconds later you get a response containing text.** That's it. Everything else in the course is deciding *what text to send* and *what to do with what comes back*.

### 2. The API key is your membership card
`OPENAI_API_KEY=sk-proj-...` in your `.env` file is a secret password that (a) proves you're allowed in and (b) bills your account for usage. 🏨 Like a hotel key card: no card, no room; lose it, someone else can charge your room. That's why it lives in `.env` (never in code, never on GitHub) and why every notebook starts with `load_dotenv()`.

### 3. The `messages` list is the single most important structure in the course
Every chat-model call sends a **list of little dictionaries**, each with a `role` and `content`:

| role | who it pretends to be | 🏨 analogy |
|---|---|---|
| `system` | backstage instructions: personality, rules, format | the job description pinned to the employee's wall |
| `user` | the human speaking | the customer's words |
| `assistant` | the model's own previous replies | the employee's sent-mail folder |
| `tool` | results of a function the model asked you to run | a memo from another department answering the employee's question |

🧠 **The memory trick:** models have *no* memory between calls. "Remembering" = resending the whole list every time, with new lines appended. Once you see this, weeks 1, 2 and 8 all become the same picture.

### 4. Tokens are the model's puzzle pieces
Models don't read letters or words; they read **tokens** — chunks of text (~4 characters, or ¾ of an English word on average) mapped to numbers. Prices, speed limits (`max_new_tokens`), context windows ("how much fits in the prompt") and Week 7's truncation are all measured in tokens.

### 5. Two ways to use a model
- **API models (rented):** GPT, Claude, Gemini, Grok. Someone else's GPU, pay per token, one `pip install` and a key.
- **Local models (owned):** Llama, Phi, Gemma, Qwen. You download the *weights* (the learned numbers), load them onto a GPU (yours, Colab's, or Modal's), run them yourself — free per token, but you need hardware and usually **quantization** (storing weights in fewer bits, like saving a photo as a smaller JPEG: slightly blurry, 4× smaller).

### 6. JSON is the paperwork of the LLM world
**JSON** = a standardized way to write structured data as text: `{"city": "Tokyo", "price": 1400}`. It shows up constantly: the request you send, the response you get, tool descriptions, tool results, batch files, training data. When you see `json.loads(...)`, it means "turn JSON text into a Python dict."

### 7. Notebooks run top-to-bottom; variables *stay alive* between cells
A `.ipynb` notebook is a sequence of cells. When you run cell 1, its variables exist for every later cell — until the kernel restarts. ⚠️ The #1 beginner bug: running cell 5 without running cells 1–4 → `NameError`. Fix: **Kernel → Restart & Run All**. The same rule applies to Python scripts: `uv run price_is_right.py` executes top-to-bottom once.

### 8. `.env` + `load_dotenv(override=True)` = how secrets get into the program
`.env` is a text file of `NAME=value` lines. `load_dotenv(override=True)` copies those into environment variables; `os.getenv("NAME")` reads one back. `override=True` means "re-read the file even if I loaded it before" — re-run it after you edit `.env`.

### 9. Evaluate before you celebrate
Weeks 4–8 all install the same discipline: build a thing → build a **test set with known answers** → score the thing → *then* compare to alternatives. The course's scoreboard (random guess $106 average error → ensemble $29.90) is only meaningful because everything was measured the same way.

### 10. The universal 6-step pattern (Appendix A expands this)
Every single project in the course fits: **CONFIGURE → GATHER → PREPARE → CALL → LOOP → PRESENT.** If you can point at each of the six in code you've never seen, you can read any LLM project.

---

## Table of contents

0. [Start here — 10 ideas that unlock the course](#primer)
1. [Week 1 Day 1 — Website Summarizer (10 steps)](#w1d1)
2. [Week 1 Day 5 — Brochure Generator (12 steps)](#w1d5)
3. [Week 2 Day 1 — Two chatbots arguing (message-list trace)](#w2d1)
4. [Week 2 Day 2/3 — Gradio streaming & chat history](#w2gradio)
5. [Week 2 Day 4 — Tool calling: the full round trip (JSON trace)](#w2tools)
6. [Week 2 Day 5 — Multimodal Blocks app (event chain)](#w2blocks)
7. [Week 3 Day 2 — HuggingFace pipeline internals](#w3pipes)
8. [Week 3 Day 4 — Loading & running Llama yourself (12 steps)](#w3models)
9. [Week 3 Day 5 — Meeting minutes (audio → minutes, 8 steps)](#w3minutes)
10. [Week 4 — The Python→C++ race (11 steps)](#w4race)
11. [Week 5 — RAG: ingestion (7 steps) → query (6 steps) → evaluation](#w5rag)
12. [Week 6 — Data curation → batch rewrite → model race → fine-tune](#w6)
13. [Week 7 — QLoRA training, end to end (14 steps)](#w7)
14. [Week 8 — One full autonomous agent run (message-by-message trace)](#w8)
15. [Appendix A — The universal 6-step pattern](#appa)
16. [Appendix B — Complete run order for the whole course](#appb)
17. [Appendix C — Plain-English glossary of every term used](#appc)

---

<a id="w1d1"></a>
# 1️⃣ Week 1 Day 1 — The Website Summarizer, step by step

**What you're running:** `week1/day1.ipynb` — you type a website address, and GPT reads the site and writes a snarky summary.
**What you'll learn:** the anatomy of an LLM call — the pattern used in literally every later week.
**Prerequisites:** an OpenAI API key in `/home/user/ED/.env`, Jupyter kernel running.

### 🗺️ The big picture before any code
```
URL  →  download HTML  →  strip out junk, keep visible text  →  stuff text into a message
     →  send to GPT    →  GPT's summary comes back          →  display it
```
🏨 **Analogy:** you (the code) visit a shop, rip out everything except the words on the walls (scraping), photocopy the words (truncate), hand the photocopy to a consultant (GPT) with a sticky note saying "summarize this, be snarky" (the prompts), and pin the consultant's reply to a board (display).

---

### STEP 1 — Imports
```python
import os
from dotenv import load_dotenv
from scraper import fetch_website_contents     # local file, sits next to the notebook
from IPython.display import Markdown, display
from openai import OpenAI
```
🧠 **Line by line:**
- `import os` — Python's built-in toolkit for reading environment variables (like `.env` values).
- `from dotenv import load_dotenv` — a helper whose one job is "read `.env` → put values into environment variables."
- `from scraper import fetch_website_contents` — imports a function from `scraper.py`, a plain Python file **in the same folder as the notebook**. This is course-provided plumbing; importing runs its top-level code (defining two functions), but nothing executes yet.
- `from IPython.display import Markdown, display` — Jupyter's way to render pretty formatted text inside the notebook.
- `from openai import OpenAI` — the official Python client library that wraps the HTTPS calls for you.

🖥️ Nothing visible happens. A cell can run silently and still be essential — this is normal.

---

### STEP 2 — Load the secret key
```python
load_dotenv(override=True)          # reads .env → os.environ (override: replace existing)
api_key = os.getenv('OPENAI_API_KEY')
```
🧠 `load_dotenv(override=True)` opens the `.env` file sitting next to the notebook and copies each line into the process's environment. `override=True` says "if a variable was already set, replace it with the file's value" — important when you edit `.env` and re-run. `os.getenv(...)` then plucks the key out.

**DATA:** `api_key = "sk-proj-AbC123..."` — a string starting with `sk-proj-` (project key).

The notebook then has an `if/elif` chain that prints one of:
- 🖥️ `API key found and looks good so far!` — you're fine, continue.
- `No API key was found` — the `.env` file is missing, misnamed, or in the wrong folder.
- `API key found but only starts with sk-prob-...` (typo) or `...the headless` (old format) — fix the key itself.

⚠️ **Common mistake:** editing `.env` but not re-running this cell → the notebook still uses the old key. Re-run `load_dotenv(override=True)` after every edit.

---

### STEP 3 — First contact with the model
```python
messages = [{"role": "user", "content": "Hello, GPT! This is my first ever message to you! Hi!"}]
openai = OpenAI()                                            # client object; reads OPENAI_API_KEY itself
response = openai.chat.completions.create(model="gpt-5-nano", messages=messages)
response.choices[0].message.content
```
🧠 **What happens inside step 3, one beat at a time:**
1. `messages` is the famous list-of-dicts (see idea #3): here, one single `user` line. This is the smallest possible conversation.
2. `OpenAI()` builds a client object. It *automatically* picks up `OPENAI_API_KEY` from the environment — that's why step 2 mattered.
3. `.create(...)` sends the HTTPS POST to `https://api.openai.com/v1/chat/completions` with header `Authorization: Bearer sk-proj-...`. Your text travels to OpenAI's servers; a GPU there runs the model; JSON comes back. (This takes 1–10 seconds — the model is literally being run for you.)
4. The answer is buried in nested JSON: `response → choices (a list) → [0] → message → content`. Python digs it out with dot/index notation.

**DATA:** `response.choices[0].message.content` = a plain string, e.g.
🖥️ *"Hello! Welcome — it's great to meet you. What would you like to explore first?"*

🏨 **Analogy:** `create()` is mailing a letter and synchronously waiting for the reply; `choices[0].message.content` is opening the envelope and reading the letter inside (the envelope also has a postmark — usage stats, finish reason — which you ignore for now).

⚠️ **Common mistake:** writing `response.content` — too high up the nesting. Always the full chain: `.choices[0].message.content`.

---

### STEP 4 — Scrape the website
```python
ed = fetch_website_contents("https://edwarddonner.com")
```
Inside `scraper.py`, in order (this is the whole function, translated):
1. `requests.get(url, headers={"User-Agent": "Mozilla/5.0 ..."})` — downloads the page's raw HTML. The fake Chrome header makes the site treat us as a normal browser; some sites block unknown clients.
2. `BeautifulSoup(response.content, "html.parser")` — parses the HTML string into a **tree of tags** you can search. 🏨 A sieve: pour the messy HTML in, query the structured result.
3. `title = soup.title.string` — grabs the page title.
4. For every `<script>`, `<style>`, `<img>`, `<input>` inside `<body>`: `.decompose()` — **deleted**, because code, CSS, image tags and form fields are noise for a language model.
5. `soup.body.get_text(separator="\n", strip=True)` — walks what's left and returns only the visible words, one line per element.
6. `(title + "\n\n" + text)[:2_000]` — glue title + text, then **hard-truncate to 2,000 characters**. ⚠️ Pages are huge; a full site could be 200k+ characters = big cost and a confused model. Truncation is the blunt but effective fix.

**DATA:** `ed = "Ed Donner | Language models are wonderful\n\nLearn about AI\n\nI'm Ed Donner... (2000 chars)"` — one string. This is what the LLM will "see" of the website.

🧠 Note what the model does **not** see: layout, colors, images, JavaScript behavior. `requests` downloads the initial HTML only. (That's why, later, summarizing `openai.com` disappoints — its content is drawn by JavaScript after the initial HTML arrives, so there's little text to scrape.)

---

### STEP 5 — Define the two prompts
```python
system_prompt = "You are a snarky assistant that analyzes the contents of a website..."   # personality + rules
user_prompt_prefix = "Here are the contents of a website.\nProvide a short summary..."    # the ask
```
🧠 Two ordinary Python strings, and the two halves of every good LLM prompt:
- **system prompt** = who the model should *be* (personality, rules, output format). Set once, like a job description.
- **user prompt** = what you want *this time*. The word **"prefix"** is the hint: the website text gets appended *after* it.

---

### STEP 6 — Assemble the message list
```python
def messages_for(website):
    return [{"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt_prefix + website}]
```
🧠 This function is pure assembly — no model call. It glues the two strings from step 5 into the list-of-dicts format from step 3.

**DATA (conceptually what the list contains):**
```
[ {system: "You are a snarky assistant..."},
  {user:   "Here are the contents of a website. Provide a short summary...\n\nEd Donner | Language models are wonderful\n\nI'm Ed Donner..."} ]
```
🏨 The user message is a sandwich: instructions (bread) around the website text (filling).

---

### STEP 7 — The full product function
```python
def summarize(url):
    website = fetch_website_contents(url)      # step 4 again
    response = openai.chat.completions.create(model="gpt-4.1-mini", messages=messages_for(website))
    return response.choices[0].message.content
```
🧠 Everything before this step was ingredients; this is the recipe in 3 lines: scrape → call → unwrap the answer. (A different, slightly stronger model `gpt-4.1-mini` is used for the real task; `gpt-5-nano` was just the hello-world.)

---

### STEP 8–10 — Display
```python
display_summary("https://edwarddonner.com")    # calls summarize() → display(Markdown(summary))
```
🧠 `summarize()` gets the text; `Markdown(...)` marks it up; `display(...)` paints a formatted card in the notebook below the cell.

🖥️ A formatted, snarky summary. Repeat for `cnn.com`, `anthropic.com` — same function, different URL.

⚠️ **Failure mode to expect:** `openai.com` returns little or nothing (JavaScript-rendered site — `requests` only sees the initial HTML, as explained in step 4). This is a *scraping* limitation, not a bug in your code.

---

### ✅ Checkpoint — can you answer these?
1. Why does the summary cost money each time you re-run it? → *Each `create()` is a fresh API call; the model has no memory, so nothing is reused.*
2. What would break first if `edwarddonner.com`'s page were 500k characters? → *Nothing breaks, but cost and confusion rise — hence the 2,000-char truncation.*
3. What do the model and `requests` have in common? → *Both only ever see text.*

**The whole program in one line of flow:** `URL → HTTP GET → HTML → BeautifulSoup clean → 2000 chars → system+user messages → POST /chat/completions → choices[0].message.content → Markdown`.

---
<a id="w1d5"></a>
# 2️⃣ Week 1 Day 5 — The Brochure Generator, step by step

**What you're running:** `week1/day5.ipynb` — give it a company URL, get back a multi-page sales brochure.
**What you'll learn:** multi-step LLM workflows — using one LLM call to *steer* code, plus JSON mode and streaming.

### 🗺️ The big picture
Day 1 scraped **one** page. But a company's story is spread across its About, Careers and Products pages. Someone must decide *which pages matter* — and that someone is the LLM itself:

```
get all links → LLM call #1: "which of these are company pages?" → scrape those pages too
             → one big document → LLM call #2: "write a brochure" → display (or stream)
```
🏨 **Analogy:** you're writing a company profile. First you ask a librarian for the *map of the building* (the link list). Then a researcher reads the map and circles the three rooms worth visiting (LLM #1). You photocopy those rooms. Finally a writer turns the pile of photocopies into a brochure (LLM #2).

Two LLM calls, two different jobs: a **chooser** and a **writer**. This pattern — *model decides, code does* — grows into Week 8's autonomous agent.

---

### STEP 1 — Get all the links
```python
links = fetch_website_links("https://edwarddonner.com")
```
🧠 Inside the scraper: `requests.get` → BeautifulSoup → list comprehension over every anchor tag:
```python
[link.get("href") for link in soup.find_all("a")]   # "href" = the link's destination
```
then drop empty ones.

**DATA:** `["https://edwarddonner.com/", "/about", "mailto:ed@...", "/resources", ...]` — note the **relative** links like `/about`: no domain, just a path. They work when clicked *on* the site, but we'll need full URLs to scrape them (the one-shot example in step 2 quietly teaches the model to fix this).

---

### STEP 2 — Show the model an example answer (one-shot prompting)
```python
link_system_prompt = """You are provided with a list of links found on a webpage. ...
You should respond in JSON as in this example:
{ "links": [ {"type": "about page", "url": "https://full.url/goes/here/about"}, ... ] }"""
```
🧠 **One-shot prompting** = putting one worked example in the prompt so the model imitates its *shape* (zero-shot = no example; few-shot = several). Here the example does double duty: it fixes the output format (JSON with a `links` key) **and** the relative→absolute URL habit.

---

### STEP 3 — Build the user prompt
```python
def get_links_user_prompt(url):
    user_prompt = f"Here is the list of links on the website {url} ... Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "\n".join(links)          # all raw links, one per line
    return user_prompt
```
🧠 The ask + the raw material, in one string: "here's the list … ignore legal/mail links … and here are all the links." `"\n".join(links)` glues the list into one multiline string, one link per line — easy for the model to read line-by-line.

---

### STEP 4 — LLM call #1 (with JSON mode)
```python
def select_relevant_links(url):
    response = openai.chat.completions.create(
        model=MODEL, messages=[system, user], response_format={"type": "json_object"})
    links = json.loads(response.choices[0].message.content)   # JSON text → Python dict
    return links
```
🧠 **Line by line:**
- `response_format={"type": "json_object"}` — **JSON mode**. The model is now *constrained* to emit valid JSON, not chatty prose ("Sure! Here's the JSON you asked for: …" would crash `json.loads`). 🏨 Instead of asking someone to "please fill the form neatly," the form physically can't be submitted half-filled.
- `json.loads(...)` — the response is still just *text that looks like* `{"links": [...]}`; `loads` ("load string") converts it into a real Python dict you can index.

**DATA:** `{"links": [{"type": "about page", "url": "https://edwarddonner.com/about"}, {"type": "careers page", "url": "..."}]}`

⚠️ **Common mistake:** forgetting that even in JSON mode you must *mention* JSON in the prompt. The API requires it — the system prompt's example satisfies this.

---

### STEP 5 — Scrape the landing page + every selected page
```python
def fetch_page_and_all_relevant_links(url):
    contents = fetch_website_contents(url)                 # reuse Day-1 scraping
    relevant_links = select_relevant_links(url)            # ← LLM call #1 happens INSIDE
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    for link in relevant_links['links']:
        result += f"\n\n### Link: {link['type']}\n"
        result += fetch_website_contents(link["url"])      # scrape each chosen page
    return result
```
🧠 The heart of the project. Read the middle line again: **the LLM call is nested inside the scraping function.** The "gather" step now consults the model mid-way. Then a plain `for` loop scrapes each URL the model chose and appends every page into one giant string with `###` headers as section dividers (headers matter: they let the next call tell sections apart).

**DATA:** one big string (~5–10k chars): landing page + about page + careers page, each under a `### Link:` header.

---

### STEP 6 — Build the brochure prompt (with a cost guard)
```python
def get_brochure_user_prompt(company_name, url):
    user_prompt = f"You are looking at a company called: {company_name} ... build a short brochure...\n\n"
    user_prompt += fetch_page_and_all_relevant_links(url)
    user_prompt = user_prompt[:5_000]          # ← hard truncation: caps tokens, cost, context
    return user_prompt
```
🧠 Same shape as Day 1: instructions + material. New move: the **`[:5_000]` cost guard**. Even after the model filtered links, five pages can still blow past a sane context size. Truncation = a thermostat: simple, blunt, prevents runaway cost. (It *can* cut a page mid-sentence — accepted, because the landing page comes first and matters most.)

---

### STEP 7 — LLM call #2 writes the brochure
```python
create_brochure("HuggingFace", "https://huggingface.co")
```
🧠 Inside: `system = brochure_system_prompt` ("include company culture, customers, careers… in markdown") + the big user prompt from step 6 → one API call → `display(Markdown(result))`.

🖥️ A formatted brochure appears. Same unwrapping as Day 1 — by now, `choices[0].message.content` should feel like breathing.

---

### STEP 8 — Streaming version (the typewriter)
```python
def stream_brochure(company_name, url):
    stream = openai.chat.completions.create(model=..., messages=..., stream=True)   # ← key change
    response = ""
    display_handle = display(Markdown(""), display_id=True)     # create an EMPTY output box, keep its ID
    for chunk in stream:                                        # text arrives in many small pieces
        response += chunk.choices[0].delta.content or ''        # glue the new piece on
        update_display(Markdown(response), display_id=display_handle.display_id)  # REDRAW same box
```
🧠 **Line by line:**
- `stream=True` — instead of waiting for the full letter, OpenAI sends it as it's written: many small **chunks**, each carrying a fragment (`delta`) of text. The call returns immediately.
- `display(Markdown(""), display_id=True)` — print an empty box *once* and keep its ID number, like reserving a whiteboard.
- The `for chunk` loop: each pass appends the new fragment (`or ''` guards chunks that carry no text, e.g. role metadata) and **redraws the same whiteboard** with the longer text.
- `chunk.choices[0].delta.content` — note `delta`, not `message`: in streaming, fragments arrive in `.delta`.

**Timing of one loop pass:** a chunk like `"**About"` arrives → `response` grows → the box re-renders. Repeated ~200 times in a few seconds → 🖥️ text typing itself. Total time is the same as non-streaming; it just *feels* instant because the first word arrives in ~half a second.

🏨 **Analogy:** non-streaming = waiting for the whole letter; streaming = reading over the author's shoulder as she writes.

---

### ✅ Checkpoint
1. Why two LLM calls instead of one? → *Different jobs: choosing links needs the link list; writing needs full page contents. Splitting keeps each prompt small and each job checkable.*
2. What breaks if you delete `response_format={"type":"json_object"}`? → *The model may wrap JSON in prose; `json.loads` raises `JSONDecodeError`.*
3. Where exactly does LLM call #1 happen? → *Inside `fetch_page_and_all_relevant_links`, via `select_relevant_links`, before the scrape loop.*

---
<a id="w2d1"></a>
# 3️⃣ Week 2 Day 1 — Two chatbots arguing: the message-list trace

**What you're running:** the finale of `week2/day1.ipynb`. Two models, two personas, two history lists — and no user anywhere after the start.
**What you'll learn:** that the `messages` list is *yours to construct however you like* — including casting one model as "the user" of another.

### 🗺️ The big picture
Two chatbots talk to each other. GPT (argumentative) vs Claude (excessively polite). Each model only ever talks to **its own API** — they never connect to each other. Python is the postman: it takes what one said and delivers it to the other as if a human had typed it.

🏨 **Analogy:** two actors in separate soundproof rooms, each with a headset to a director. The director (your Python) reads actor A's line into actor B's ear, and vice versa. Each believes he's talking to a person.

---

### STEP 1 — Setup
```python
gpt_messages    = ["Hi there"]     # GPT's own lines (plain strings, NOT dicts)
claude_messages = ["Hi"]           # Claude's own lines
gpt_system    = "You are a chatbot who is very argumentative..."
claude_system = "You are a very polite, courteous chatbot..."
```
🧠 Two **separate** memory lists — one per character — holding plain strings this time (not dicts; the dicts are rebuilt fresh inside each call). And two system prompts that *are* the entire personality of the debate. Change these two strings and the whole argument changes: the drama comes 100% from the prompts.

---

### STEP 2 — `call_gpt()`: build GPT's view of the conversation
```python
def call_gpt():
    messages = [{"role": "system", "content": gpt_system}]
    for gpt, claude in zip(gpt_messages, claude_messages):     # pair the histories
        messages.append({"role": "assistant", "content": gpt})     # GPT's line → "assistant"
        messages.append({"role": "user", "content": claude})       # Claude's line → "user"
    response = openai.chat.completions.create(model=gpt_model, messages=messages)
    return response.choices[0].message.content
```
🧠 **Line by line:**
- Start with the system prompt: GPT's persona goes on first, always.
- `zip(gpt_messages, claude_messages)` — pairs up the two lists element-by-element: 🏨 like drawing socks from two drawers two-at-a-time: `("Hi there","Hi")`, then the next pair, until the shorter drawer empties.
- The two `append`s are **the perspective trick**: when *GPT* is called, GPT's own past lines play the `assistant` role (its "sent-mail"), and **Claude's lines are relabeled `user`** — so GPT experiences Claude as a human interlocutor.

**DATA — what GPT actually receives on turn 1:**
```
[{system: "…argumentative…"},
 {assistant: "Hi there"},        ← GPT's opener
 {user:      "Hi"}]              ← what Claude said → presented as "the user" to reply to
```
`call_claude()` is the exact mirror: GPT's lines → `user`, Claude's → `assistant`, plus one extra `{"role":"user", "content": gpt_messages[-1]}` at the end — the *new* GPT line Claude must respond to. (GPT's list already ends with its own last line as the implicit thing-to-answer; Claude's mirror needs it spelled out.)

🧠 Zoom out and admire: **there is no user anywhere.** The "user" role is just a label your code assigns. Any speaker can play any role.

---

### STEP 3 — The debate loop
```python
for i in range(5):
    gpt_next = call_gpt();       gpt_messages.append(gpt_next)      # speak → REMEMBER
    claude_next = call_claude(); claude_messages.append(claude_next)
```
🧠 Each pass: ask GPT for a line → *append it to GPT's memory* (this is the crucial half-step beginners forget — if you don't append, the debate never progresses and GPT re-answers "Hi there" forever) → same for Claude. Because each `call_*` rebuilds the message list from the full histories, every round carries everything said so far — the memory illusion from Week 1 Day 4, now driving a conversation between two machines.

**DATA evolution (that's the whole mechanism):**
```
Round 1: gpt=["Hi there","I don't see why we're greeting..."]  claude=["Hi","I'm sorry you feel..."]
Round 2: both lists grow by one; each new call re-sends EVERYTHING
```
🖥️ The notebook prints alternating `### GPT:` / `### Claude:` — an argument that escalates round by round, powered entirely by two system prompts and two growing lists.

⚠️ **Common mistake:** calling the APIs but not appending the replies — an infinite loop of the same opening exchange. **Append after every speak.**

---

### ✅ Checkpoint
1. What role does Claude's line play in GPT's request? → *`user` — the label is assigned by your code, not by who actually spoke.*
2. Why keep two separate lists? → *Each model needs its own lines marked `assistant` and the other's marked `user`; one mixed list would be wrong for both.*
3. What single edit makes them argue about pineapple pizza? → *Change the two system prompts. Personalities live in prompts, nothing else.*

---

<a id="w2gradio"></a>
# 4️⃣ Week 2 — Gradio streaming & chat history, step by step

**What you're running:** `week2/day2.ipynb` (UI basics) and `day3.ipynb` (a store chatbot with history + a dynamic system prompt).
**What you'll learn:** how a 5-line Python function becomes a working web app, and where "memory" in a chat UI really comes from.

### 🧠 What is Gradio, in one paragraph?
Gradio turns a Python function into a **web app**: it generates the whole browser interface (text boxes, buttons, chat bubbles), runs a local server, and wires the UI so that clicking Submit *calls your function* and shows what it returns. You write zero HTML/JS. 🏨 It's a restaurant front-of-house: it takes the customer's order (UI event), shouts it to the kitchen (your function), and plates whatever comes back (the output component).

## 4a. What happens when you click Submit (Interface + generator)

### STEP 1 — You launch
```python
gr.Interface(fn=stream_gpt, inputs=[message_input], outputs=[message_output], ...).launch()
```
🧠 `gr.Interface` says: "build a UI where `inputs` feed into function `fn`, and `outputs` display its result." `.launch()` does the heavy lifting: builds a small web app, starts a server on a free port, opens your browser. From now on the browser and Python talk behind your back. ⚠️ The cell "hangs" while the server runs — that's correct behavior, not a bug. Interrupt the kernel to stop it.

### STEP 2 — You type "hello" and press Enter
```python
def stream_gpt(prompt):
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    stream = openai.chat.completions.create(model='gpt-4.1-mini', messages=messages, stream=True)
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result            # ← PAUSE here; hand the partial string to Gradio; RESUME on next chunk
```
🧠 The new Python concept is the **generator**. A function containing `yield` doesn't run to completion like normal code. Each `yield`:
1. hands the current value (`result` so far) to whoever's listening (Gradio),
2. **pauses** the function exactly there,
3. resumes it on the next request (the next network chunk).

🏨 A normal function is a vending machine: money in, complete snack out. A generator is a barista pouring a latte: slide the cup (call the generator), get some coffee (yield), cup slides back for more, pour continues — until the cup is full and the pour ends.

So: chunk arrives → `result` grows → `yield` shows it in the browser → pause → next chunk → resume → … → stream ends → function exits. 🖥️ Text appears word-by-word, exactly like ChatGPT.

## 4b. ChatInterface — where history comes from

### STEP 1 — Gradio owns the history
```python
gr.ChatInterface(fn=chat, type="messages").launch()
```
🧠 `ChatInterface` is Interface-with-a-memory: Gradio itself keeps the running conversation and shows bubbles. `type="messages"` means "give my function the history in the OpenAI format (role/content dicts)" — the other legacy format used tuples. The magic question: when your `chat(message, history)` is called, where did `history` come from? **Gradio built it.** You never append to it yourself in this notebook — Gradio appends each new message *before* calling you.

### STEP 2 — Your callback gets called with accumulated state
```python
def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]   # normalize → plain dicts
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)
    ...
    yield response
```
🧠 The list-comprehension is just hygiene: guarantee plain dicts regardless of Gradio's wrapper objects. Then the eternal recipe: **system first + history middle + new message last.** Note what's *absent*: no code stores anything between calls. The model "remembers" because the entire history is resent on every call — statelessness, again.

**DATA — turn 3 of a conversation, the exact list sent:**
```
[{system: "You are a helpful assistant in a clothes store..."},     ← system always first
 {user: "Hi"}, {assistant: "Welcome! Hats are 60% off!"},           ← history (turns 1–2)
 {user: "and belts?"}]                                              ← the new message, LAST
```

### STEP 3 — Dynamic system prompt (code + model sharing control)
```python
relevant_system_message = system_message
if 'belt' in message.lower():                    # plain Python inspects the text
    relevant_system_message += " The store does not sell belts; ..."
```
🧠 Before building the messages, ordinary `if` code reads the user's text and *extends the system prompt* — Python decides policy ("we don't sell belts"), the model delivers it politely. The system prompt is not a constant; it's a **variable your code programs at call time**. This tiny `if` is the seed of every guardrail and router you'll later see (Week 5's query rewriting, Week 8's tool policy).

⚠️ **Common mistake:** building `messages = history + [new]` without the system prompt — the bot suddenly loses its personality and rules on every turn after the first.

---

### ✅ Checkpoint
1. What makes text appear progressively? → *The function is a generator; each `yield` repaints the output box with the partial result.*
2. On turn 10, how many user+assistant pairs are in the list sent to OpenAI? → *Nine pairs (18 messages) plus the system prompt plus the new user message — everything, every time.*
3. Who appends your new message to the history? → *Gradio, before calling `chat`. You only prepend the system prompt.*

---
<a id="w2tools"></a>
# 5️⃣ Week 2 Day 4 — Tool calling: the full round trip, JSON-traced

**What you're running:** the airline assistant in `week2/day4.ipynb`. User asks a price; the model *requests* your function; you run it; the model answers.
**What you'll learn:** the single most important mechanism of 2025 AI — it becomes Week 8's autonomous agent, and it's the engine of every "AI agent" product you've heard of.

### 🧠 The misconception to kill first
The model **cannot run your code.** It cannot browse, calculate, or query a database. It can only do one thing: read text, write text. Tool calling is a *protocol built out of that one ability*:

1. You tell the model (in text) "here are some tools that exist, with names and descriptions."
2. The model, when a question matches, replies with special text: *"please run `get_ticket_price` with these arguments."* (It still just wrote text — but formatted, and flagged `finish_reason="tool_calls"`.)
3. **Your Python** notices the flag, actually executes the function, and appends the result to the messages.
4. The model reads the result and finally answers in prose.

🏨 **Analogy:** the model is a consultant on the phone who can't touch your computer. You give him a list of forms the office can process ("Form T: ticket prices"). A customer asks about Tokyo; the consultant says "please have the office fill Form T for Tokyo." The office (your code) faxes back "$1400." The consultant tells the customer. The consultant never left his chair — **you did all the office work.**

---

### STEP 1 — Your tool (plain Python) and its JSON description
```python
ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}
def get_ticket_price(destination_city): ...

price_function = {"name": "get_ticket_price",
  "description": "Get the price of a return ticket to the destination city.",
  "parameters": {"type": "object",
    "properties": {"destination_city": {"type": "string", "description": "The city that the customer wants to travel to"}},
    "required": ["destination_city"], "additionalProperties": False}}
tools = [{"type": "function", "function": price_function}]
```
🧠 Two separate things, both yours:
- The **function itself** — boring, ordinary Python (a dict lookup).
- The **description** — a JSON "help-wanted ad" telling the model what the tool is *for* and what arguments it takes. `required` and `additionalProperties: False` keep the model from inventing missing or extra arguments.

🧠 The `description` fields are prompt engineering in disguise: the model chooses tools by matching the user's question against these descriptions. Vague description → wrong or missed tool calls.

---

### STEP 2 — First API call (with `tools=` attached)
```python
response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
```
User asked *"How much for Tokyo?"*. The model reads the tool ad, recognizes the match, and responds **without writing prose**:
```python
response.choices[0].finish_reason   # → "tool_calls"          (not "stop"!)
message.tool_calls[0].function.name # → "get_ticket_price"
message.tool_calls[0].function.arguments  # → '{"destination_city": "Tokyo"}'   ← a JSON STRING
```
🧠 **Read those three lines slowly; they're the whole trick:**
- `finish_reason` is the model's status report. `"stop"` = "I wrote my answer, done." `"tool_calls"` = "I'm not answering — I'm handing you a work order." Your code branches on exactly this field.
- `.name` — which tool the model chose.
- `.arguments` — ⚠️ a **string containing JSON**, not a dict. `'{"destination_city": "Tokyo"}'` has quotes around the whole thing. The model *wrote this text* — it never touched your function.

---

### STEP 3 — YOU execute the tool
```python
def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)   # '{"destination_city":"Tokyo"}' → dict
    city = arguments.get('destination_city')
    price_details = get_ticket_price(city)                 # dict lookup → "The price of a ticket to Tokyo is $1400"
    return {"role": "tool", "content": price_details, "tool_call_id": tool_call.id}
```
🧠 Line by line:
- `json.loads` — parse the model's JSON text into a real dict. (This is why JSON mode mattered in Week 1: the model is *generally* reliable at emitting parseable arguments here, and the API validates them against your declared schema.)
- `arguments.get('destination_city')` — pull out the argument. `.get` returns `None` instead of crashing if it's missing.
- `get_ticket_price(city)` — **your** function runs, in **your** process. The model had zero involvement in this line.
- The return value is shaped as a message with the special `role: "tool"` — a memo from the office. `tool_call_id` staples the answer to the exact request it answers (vital when there are several tool calls in flight).

---

### STEP 4 — Append BOTH messages, call again
```python
messages.append(message)      # the model's tool REQUEST
messages.append(response)     # your tool RESULT
response = openai.chat.completions.create(model=MODEL, messages=messages)   # NO tools needed now
```
🧠 Two appends, then the second call. The model needs to *see both* its own work order and the office's answer — omit either and the API errors out or the model gets confused.

**DATA — the final list the model sees on call #2:**
```
[{system: "You are a helpful assistant for an Airline called FlightAI..."},
 {user:      "How much for Tokyo?"},
 {assistant: <tool_calls: get_ticket_price({"destination_city":"Tokyo"})>},
 {tool:      "The price of a ticket to Tokyo is $1400",  tool_call_id: "call_abc123"}]
```
`finish_reason` is now `"stop"` and the content is: 🖥️ *"A return ticket to Tokyo is $1,400."*

🧠 Notice the conversation now has **four** entries for what the user experiences as one question. Agents are just this, many times over.

⚠️ **Common mistakes:** forgetting the `assistant` append (API rejects the orphan tool message); calling again *with* `tools=` when you don't want more tool use; forgetting `tool_call_id`.

---

### STEP 5 — The multi-tool / chained version
Two upgrades turn this into a real agent:
- `handle_tool_calls` (plural) loops over `message.tool_calls` — the model may request *several* tools at once — and returns a **list** of tool messages, added with `messages.extend(responses)`.
- The `if` becomes `while`:
```python
while response.choices[0].finish_reason == "tool_calls":   # keep looping while the model wants tools
    ...
```
🧠 That `while` is the **agent loop** — the model can now chain tools: ask for one, get the result, *then decide to ask for another*, and keep going until it finally answers (`stop`). Everything in Week 8 — scan → price → price → price → notify — is this loop with a to-do list. One-line idea, enormous consequences.

*(Later in day 4, `get_ticket_price` is rewritten to query SQLite: same interface, `SELECT price FROM prices WHERE city = ?` — proving the model never cared *how* the tool worked, only what it returned.)*

---

### ✅ Checkpoint
1. Who executes `get_ticket_price`? → *Your Python, your machine. The model only writes the request.*
2. What does `finish_reason == "tool_calls"` mean? → *"Don't read my text as the answer — read it as a work order."*
3. Why the second API call? → *The model is stateless and stopped mid-thought; you must show it its own request + your result and let it finish.*

---

<a id="w2blocks"></a>
# 6️⃣ Week 2 Day 5 — Multimodal Blocks app: the event chain

**What you're running:** `week2/day5.ipynb` — a chat app that answers about ticket cities, *and* speaks the reply aloud *and* paints a pop-art image of the city. **Multimodal** = text + audio + image in one app.
**What you'll learn:** Gradio's `Blocks` — full manual control of layout and, crucially, **event chaining**.

### 🧠 Interface vs Blocks
`gr.Interface` (day 2) is a one-function-from-a-box: zero layout control. `gr.Blocks` is the full workshop: you place components yourself (`with gr.Blocks() as ui:` …), and — the point of today — **wire events to events**. An event = "when X happens to component Y, run function Z, putting values from these components in and results into those."

### The chain, in code
```python
message.submit(put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]).then(
    chat, inputs=chatbot, outputs=[chatbot, audio_output, image_output])
```
🧠 Read it as a sentence: "When the box is submitted, run `put_message_in_chatbot`, reading from the textbox and the chat window, writing back to both. **Then** — automatically, when that finishes — run `chat`, reading the chat window, writing text, audio and image." `.then()` is the "and afterwards" glue: a two-step assembly line where step 2's *input* is step 1's *output*.

### STEP-BY-STEP on one user message ("Tell me about Paris"):
1. **Enter pressed** → event 1 fires: `put_message_in_chatbot("Tell me about Paris", history)` returns `("", history + [{"role":"user","content":"Tell me about Paris"}])` → two effects: the textbox is cleared, and the chatbot component shows your bubble instantly (UI feels fast *before* any LLM runs).
2. **`.then(...)` fires automatically** → event 2: `chat(history)` — with the history that *now includes* your Paris message.
3. Inside `chat`: builds `system + history` → API call **with tools** → `while finish_reason=="tool_calls"` loop (yesterday's agent loop, embedded in a UI): the DB tool runs `get_ticket_price("Paris")` → `$899`, and the city name is collected into `cities`.
4. Final text reply → `history += [{"role":"assistant","content":reply}]` (YOU own the history now — no ChatInterface here, you maintain the list yourself and hand it to the `chatbot` component).
5. Side effects from the same reply: `talker(reply)` → `openai.audio.speech.create(...)` returns audio bytes; `artist("Paris")` → `openai.images.generate(...)` → base64 → `Image.open(BytesIO(...))` becomes a PIL image. Two *more* API products chained onto the text.
6. Three outputs update together: 🖥️ chatbot text + 🔊 audio autoplays + 🖼️ pop-art Paris appears.

🧠 The deep lesson: one user action triggered a **pipeline** of two chained events, and step 2 fanned out to three different AI services. This is exactly how production apps compose LLMs — not one big call, but small steps wired by code.

⚠️ **Common mistakes:** forgetting `.then()` → history updates but nobody ever replies; returning wrong number of outputs → cryptic Gradio error; blocking too long in event 1 (keep UI-feedback steps instant, model calls in later events).

---

### ✅ Checkpoint
1. Why split into two events instead of one function? → *Instant visual feedback (clear box, show bubble) happens before the slow LLM call; `.then()` sequences them.*
2. Where does the image come from? → *`openai.images.generate` — a separate API — decoded from base64 into a PIL image.*
3. Which week's mechanism is the `while finish_reason=="tool_calls"` loop? → *Day 4's agent loop — same code, new home.*

---
<a id="w3pipes"></a>
# 7️⃣ Week 3 Day 2 — What `pipeline(...)` actually does inside

**Colab: "week 3 day 2".** **Prerequisites:** a Google Colab account with a **T4 GPU runtime** (Runtime → Change runtime type → T4 GPU) — free.
**What you'll learn:** that "running an AI model" = loading a neural network's weights and pushing numbers through it; the `pipeline()` helper is a friendly wrapper around 6 mechanical steps.

### 🧠 First, what even is a "model"?
A **transformer** — the architecture behind everything in this course — is a giant mathematical function with millions (or billions) of adjustable knobs called **weights**. Training = adjusting the knobs on tons of text. Using ("inference") = freezing the knobs and pushing your numbers through. The trained knobs are stored in files called **weights** or a **checkpoint**, downloaded from the HuggingFace **Hub** (think: GitHub, but for models).

The key component is **attention**: for each token, the model looks at every other token and decides which ones matter for interpreting it. In "The animal didn't cross the street because **it** was tired," attention lets `it` look back and lock onto `animal` (not `street`). Stack this ~16–100 times with progressively richer patterns, add an **MLP** (a per-token mini brain that transforms each token using everything attention gathered), and you get text understanding and generation.

### The 2-line API
```python
pipe = pipeline("sentiment-analysis", device="cuda")
pipe("I'm super excited about this.")
# → [{'label': 'POSITIVE', 'score': 0.9998}]
```
🧠 `device="cuda"` = "run on the GPU" (NVIDIA's math coprocessor; `cuda` is its programming platform). On CPU this would take seconds instead of milliseconds.

### The 6-step machine the two lines hide

1. **Create:** resolve the default model for the task (`distilbert/distilbert-base-uncased-finetuned-sst-2-english`), download weights + its tokenizer from the Hub (cached in `~/.cache/huggingface` — second run is instant).
2. **Call — tokenize:** text → token ids, padded/truncated to the model's expected length. (The **tokenizer** is model-specific: it maps text pieces ↔ numbers, and must be the *same* one used in training.)
3. **Tensors to GPU:** ids move to `cuda` (because `device="cuda"`). A **tensor** = multi-dimensional array (here, a grid of integers, one per token).
4. **Forward pass:** the network runs — attention + MLP layers, in sequence; outputs raw **logits** per class (`POSITIVE`, `NEGATIVE`). Logits = unbounded scores, one per possible answer.
5. **Post-process:** **softmax** → squashes logits into probabilities that sum to 1; pick the winning label.
6. **Return:** 🖥️ `[{'label': 'POSITIVE', 'score': 0.9998}]`.

🏨 **Analogy:** `pipeline()` is a car wash: you drive in with dirty text; inside — pre-rinse (tokenize), the actual wash (forward pass), wax (softmax), dry (label). You never see the brushes, and every wash uses the same tunnel with different brushes swapped in.

Every pipeline in the day (NER = find names of things, QA, summarization, translation, zero-shot classification, text generation) = same 6 steps, different model + pre/post-processing. The diffusion (image) and TTS (text-to-speech) examples swap step 4's network (image de-noiser / vocoder) — same wrapper. **Learn the tunnel once, use it for everything.**

### ✅ Checkpoint
1. What's cached after the first call? → *The weights + tokenizer in `~/.cache/huggingface`.*
2. What does softmax do? → *Turns raw scores into probabilities that sum to 1.*
3. Why does NER return spans like `{"word": "Sarah", "entity": "PER"}`? → *Different post-processing (step 5) for a token-tagging model — the tunnel is identical.*

---

<a id="w3models"></a>
# 8️⃣ Week 3 Day 4 — Running Llama yourself: 12 steps

**Colab: "Week 3 Day 4".** **Prerequisites:** T4 runtime, `HF_TOKEN` secret set in Colab, and Llama approved on the Hub (a 2-click form — it's a **gated model**: Meta wants to know who downloads it).
**What you'll learn:** the difference between *renting* a model through an API and *owning* one — downloading weights, formatting the model's internal prompt, generating, and cleaning up.

### 🧠 Why this day matters
Until now the model was a black box behind HTTPS. Today you download the actual neural network — Llama 3.2 3B (3 billion weights) — onto your own GPU and run it cell by cell. Every step you've treated as magic (`messages=...`) becomes visible: you'll see the raw text format the model expects, and the raw number-tokens it outputs.

---

1. `!pip install bitsandbytes accelerate transformers==4.57.6` — `!` runs a shell command instead of Python. Three libraries: **transformers** (load/run models), **accelerate** (place model layers across available memory), **bitsandbytes** (quantization — below).
2. `login(hf_token)` — your HF membership card; unlocks gated models like Llama.
3. Define `messages = [{"role": "user", "content": "Tell a joke for a room of Data Scientists"}]` — the same familiar list-of-dicts.
4. Build the quant config:
```python
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
```
🧠 **Quantization in one breath:** weights are normally stored as 16-bit numbers; 4-bit mode stores each in 4 bits with negligible quality loss. 🏨 Like shrinking a 500-page photo book into a 125-page thumbnail edition — you can still navigate it, it's just coarser. `nf4` is a 4-bit format tuned for neural weight distributions; `double_quant` compresses even the compression tables; `bfloat16` = the math happens at high precision even though storage is tiny. Result: a 6.4 GB model fits in ~1.7 GB — the difference between "fits on a free T4" and "doesn't."
5. `tokenizer = AutoTokenizer.from_pretrained(LLAMA)` — download the model's personal translator (text ↔ token ids). ⚠️ Tokenizers are not interchangeable: Llama's tokenizer knows Llama's vocabulary; using another model's ids would be gibberish to it.
6. `tokenizer.pad_token = tokenizer.eos_token` — housekeeping: batches of different-length texts need padding filler; reuse the "end of text" token as filler (processing happens per message, so nothing is lost).
7. `inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")` — **the big one.** Your friendly dict becomes the model's exact internal format, which looks like this:
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nTell a joke...<|eot_id|>
```
Every model family has its own secret punctuation like this; `apply_chat_template` is the official translator. Then `return_tensors="pt"` converts it to a PyTorch tensor of token ids, and `.to("cuda")` ships it to the GPU. **DATA:** `tensor([[128000, 128006,   882, ...]])` — print it! Your sentence is literally just numbers to the machine.
8. `model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config)` — weights stream from the Hub (several GB, a minute or two), and land on the GPU *as 4-bit* (never materializing the full-precision copy). `model.get_memory_footprint()` → 🖥️ ~1.7 GB (vs 6.4 GB unquantized). **`device_map="auto"`** = accelerate's autopilot: put each layer wherever memory allows.
9. `print(model)` — the whole Transformer anatomy scrolls past: embedding → 16 decoder layers (each = attention + MLP + norms) → LM head. There is no additional mystery — what you see is what runs.
10. `outputs = model.generate(inputs, max_new_tokens=80)` — **the autoregressive loop**: predict one token → append it → predict the next from the grown sequence → repeat, 80×. 🏨 Like typing on a phone keyboard that re-reads its whole suggestion each time. One token per pass — this loop is why generation is slower than reading.
11. `tokenizer.decode(outputs[0])` — ids → readable text. (First attempt looks odd without `add_generation_prompt=True` — the fixed `generate()` helper adds it, plus a `TextStreamer` to print tokens as they're born — the same typewriter effect as Week 1's streaming, now from *your* GPU.)
12. Cleanup: `del model, inputs, tokenizer, outputs` → `gc.collect()` → `torch.cuda.empty_cache()` — free the T4's 16 GB for the next model (Phi, Gemma, Qwen, DeepSeek — each re-runs steps 5–12). ⚠️ Forgetting cleanup = CUDA out-of-memory on the very next model. This is muscle memory for local-LLM work.

### ✅ Checkpoint
1. Where do the messages-dicts go before the model sees them? → *Through `apply_chat_template` into Llama's `<|start_header_id|>...` format, then into a token-id tensor.*
2. What does 4-bit quantization trade? → *A little precision for 4× memory — 6.4 GB → 1.7 GB.*
3. Why does generation loop? → *The model predicts ONE token per pass; `generate` repeats it, appending each new token.*

---
<a id="w3minutes"></a>
# 9️⃣ Week 3 Day 5 — Meeting minutes: audio → minutes in 8 steps

**Colab: "Week 3 Day 5".** **Prerequisites:** T4 runtime, the `denver_extract.mp3` file copied into your Google Drive folder `MyDrive/llms/`.
**What you'll learn:** chaining two different models — a speech-to-text model and a language model — into one pipeline you assemble from parts you already understand individually.

### 🗺️ The big picture
```
mp3 file → [Whisper: audio→text]  →  transcript string  →  [Llama: text→minutes]  →  markdown minutes
```
🏨 An assembly line with two workers: a stenographer (Whisper) produces the raw transcript; an executive assistant (Llama) turns it into polished minutes. Neither knows the other exists — the *string* between them is the only handoff.

---

1. `drive.mount("/content/drive")` — your Google Drive appears at `/content/drive`. (Colab machines are rented by the session — fresh disk each time — so files live in your Drive and get mounted in.)
2. `audio_file = open("/content/drive/llms/denver_extract.mp3", "rb")` — a raw bytes handle (`rb` = read binary; audio is not text).
3. **Transcribe (open-source):**
```python
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-medium.en", dtype=torch.float16, device="cuda")
result = pipe(audio_filename)
```
🧠 Whisper (OpenAI's open-source speech model — yes, the API company also gives models away) listens in chunks, transcribes each, joins them. `float16` = another half-precision trick (like quantization-lite) to fit the T4. **DATA:** `result["text"]` = the transcript string.
4. **Transcribe (API, for comparison):** `openai.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=audio_file, response_format="text")`. Same job, rented instead of owned. 🖥️ Both transcripts displayed side-by-side — the course's constant theme: *compare, don't assume.*
5. Wrap the chosen `transcription` into a system prompt ("produce minutes… summary, discussion points, takeaways, action items with owners") + user prompt. By now this is familiar: personality/rules first, material second.
6. Tokenize with the chat template → `.to("cuda")`; load `Llama-3.2-3B-Instruct` with the same 4-bit config from Day 4; `TextStreamer` attached.
7. `model.generate(inputs, max_new_tokens=2000, streamer=streamer)` — 🖥️ minutes type themselves, token by token, onto the page.
8. `tokenizer.decode(outputs[0])` → `display(Markdown(...))` — formatted minutes with owners per action item.

**Flow:** `mp3 → Whisper (GPU) → transcript string → chat template → quantized Llama (GPU) → markdown minutes`. Two models chained — and the beautiful part is that you already understood both halves separately on Days 2 and 4. Pipelines are just strings passed between models you know.

### ✅ Checkpoint
1. What's the interface between the two models? → *A plain Python string — the transcript.*
2. Why run Whisper locally instead of the API? → *Control, cost at scale, and privacy — the same trade-offs as any local-vs-API decision.*
3. What would a third stage look like? → *E.g., an LLM call that emails the minutes — just another string handoff.*

---

<a id="w4race"></a>
# 🔟 Week 4 — The Python→C++ race, step by step

**What you're running:** `week4/day3.ipynb` (+ `system_info.py`, `styles.py` nearby). Ask frontier LLMs to rewrite slow Python in C++, then **time** the results against each other.
**What you'll learn:** LLMs writing real, compiled code — and the course's eval discipline: same task, same flags, correctness required, speed measured.

### 🧠 Why C++ is faster (30-second version)
Python checks types and allocates memory *while your program runs* — flexible but slow. C++ does all that at **compile time**: a compiler (`clang++`) translates your code to native CPU instructions once, producing a binary; the CPU then just runs the instructions. Costs the price of one translation, paid up front, for 100–1000× speed on numeric loops.

### 🗺️ The big picture
```
probe machine → ask LLM for compile commands → send Python + machine info to each model
→ strip code fences → write main.cpp → compile → run 3× → speedup = python_time / cpp_time
```
The same machine report goes into the prompt so the model can't blame unfamiliar hardware — a clever eval trick: **control everything you're not testing.**

---

1. **Probe the machine:** `system_info = retrieve_system_info()` — `system_info.py` runs ~15 safe subprocess probes (`uname`, `sysctl`, `gcc --version`, …), each wrapped in try/except, returning a dict like `{"os": ..., "cpu": {"brand", "cores", "simd": ["AVX2",...]}, "toolchain": {"compilers": {...}}}`. **Every probe fails soft** (returns `""`) — the code must work on any machine, so a missing tool degrades the report instead of crashing.
2. **Ask the LLM how to compile** (prompt contains the probe report) → 🖥️ instructions + exact commands. You paste them in:
   `compile_command = ["clang++", "-std=c++17", "-Ofast", "-mcpu=native", "-flto=thin", "-DNDEBUG", "main.cpp", "-o", "main"]`, `run_command = ["./main"]`.
   🧠 Flags translated: `-std=c++17` = modern dialect; `-Ofast` = aggressive optimization; `-mcpu=native` = use every instruction this exact CPU knows (the SIMD report matters here!); `-flto=thin` = optimize across files; `-o main` = output binary name. **Lists, not strings**, for the commands = no shell-quoting landmines (a filename with spaces can't break `subprocess`).
3. **Define the task prompts:** system ("convert Python to high performance C++… respond only with C++") + user (the Python source + `system_info` + the compile command — the model knows exactly how it will be judged; telling it the eval criteria measurably improves output).
4. **`port(openai, OPENAI_MODEL, pi)`:** API call (`reasoning_effort="high"` for GPT) → reply text → `reply.replace('```cpp','').replace('```','')` — strip the **markdown code fences** models love to wrap code in → `write_output(reply)` writes `main.cpp` to disk.
   🧠 The `port` function takes the *client and model as arguments* — one function races every provider by swapping two strings.
5. **Baseline:** `run_python(pi)` `exec`s the Python — 🖥️ `Result: 3.14159... Execution Time: 19.178207 seconds` (200M loop iterations; the classic estimate-π-by-throwing-darts workload).
6. **`compile_and_run()`:** `subprocess.run(compile_command, check=True, text=True, capture_output=True)` → clang++ produces the `main` binary (`check=True` = raise on compile error — a model *can* emit broken C++!); then `subprocess.run(run_command, ...)` executes it ×3. 🖥️ Same result to many decimals, `Execution Time: 0.082168 seconds`.
7. **Score:** `19.178207 / 0.082168` → 🖥️ ~233× speedup.
8. **Repeat steps 4–6 for Claude, Grok, Gemini** — overwrite `main.cpp` each time, recompile, re-time. Leaderboard: Gemini 2.5 Pro fastest (~1440×). Same source, same flags, same machine — only the model varies.
9. **Day 4 adds a UI:** Gradio `convert.click(port, inputs=[model, python], outputs=[cpp])` — a dropdown of models + a paste box + a convert button. The `clients` dict maps a choice like "Grok" to its client object and model name (OpenAI/Anthropic/Grok/Gemini/Ollama/Groq/OpenRouter — one interface, many providers).
10. **Day 5 adds Rust:** `language = "Rust"` flips prompts/extension/commands (`rustc -C opt-level=3 -C target-cpu=native -C lto=fat …`). The harder benchmark hides a trap: Python integers never overflow (they grow arbitrarily), but C++/Rust fixed-size integers wrap around — so a model that translates too literally **fails correctness while being fast**. The lesson: translation is semantics, not syntax.
11. **The eval harness rule that makes the race fair:** every run is (a) same source, (b) same compile flags, (c) output must match Python's result — speed without correctness scores nothing.

### ✅ Checkpoint
1. Why feed the machine report into the prompt? → *So compile advice fits your actual CPU/GPU — and so no model can blame unfamiliar hardware. Controlled experiment.*
2. What if a model returns non-C++ prose or broken code? → *Fence-stripping handles prose; `check=True` turns compile errors into a visible failure instead of a fake "result."*
3. Where's the actual speedup number computed? → *`python_time / cpp_time` — one division, using the same machine for both.*

---
<a id="w5rag"></a>
# 1️⃣1️⃣ Week 5 — RAG, step by step (ingest → query → evaluate)

**What you're running:** `week5/` — first the day notebooks (build a knowledge base, ask it questions, score it), then the polished `implementation/` version (`ingest.py` → `app.py` → `evaluator.py`).
**What you'll learn:** the architecture behind most serious business AI: **RAG (Retrieval-Augmented Generation)** = let the model answer from *your* documents by physically placing the right passages into its prompt.

### 🧠 The problem RAG solves
An LLM has never read your company wiki. It will either say "I don't know" or — worse — confidently invent. Three possible fixes:
1. **Retrain** the model on your docs — astronomically expensive (Week 7 fine-tunes for a *narrow skill*, not knowledge).
2. **Stuff everything in the prompt** — context windows are big but finite; 50k documents won't fit.
3. **RAG: fetch only the few relevant passages and paste them into the prompt.** 🏨 An open-book exam: the model still writes the answer, but you hand it the right page first.

RAG needs a way to find "the right page" — enter **embeddings** and the **vector store**.

### 🧠 Embeddings & vector stores — the two ideas that power retrieval
- An **embedding** is a list of ~384–1536 numbers (a **vector**) produced by a small transformer ("embedding model") such that **texts with similar meanings get nearby vectors**. 🏨 Imagine giving every text GPS coordinates on a meaning map: "How do I reset my password?" and "I can't log in" land on the same block; "Q3 revenue" lands across town. Fun fact: countries cluster together, and the vector from king−man+woman lands near queen.
- A **vector store** (today: **Chroma**) is a filing cabinet for those coordinates: insert (vector + text + metadata), then query "give me the 4 entries nearest to *this* vector." The distance math (cosine similarity — measuring the angle between arrows, ignoring length) is solved; you just call the API.

The embedding model used everywhere in this course: `all-MiniLM-L6-v2` — small, fast, free, local, 384 dimensions.

---

## A. Ingestion (day 2, run once) — 7 steps

1. **Load documents:** `DirectoryLoader` per folder → `Document(page_content=text, metadata={"doc_type": "employees"})` × ~50 files. **Metadata** = a label stapled to each document (its category) — free info you'll use for filtering and colored visualizations later.
2. **Chunk:** `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)` → ~1,700 chunks. 🧠 Why chunk? (a) A whole document often doesn't fit the "right page" budget; (b) retrieval is more precise on small passages. **Overlap 200** = consecutive chunks share 200 characters — so a fact unlucky enough to sit at a boundary survives intact in *both* neighbors. "Recursive" = split on paragraph breaks first, then sentences, then words — respect the text's natural seams.
3. **Embed:** `HuggingFaceEmbeddings("all-MiniLM-L6-v2")` — each chunk → 384 floats. ~1,700 chunks in seconds on CPU.
4. **Store:** `Chroma.from_documents(chunks, embeddings, persist_directory="vector_db")` — vectors + text + metadata written to disk. ⚠️ This creates the `vector_db/` folder; delete it to force a clean rebuild.
5. **Sanity check:** `vectorstore._collection.count()` → 🖥️ "There are 1,7xx vectors with 384 dimensions." *(Check your work before building on it — course discipline again.)*
6. **Visualize:** pull all vectors → `TSNE(n_components=2)` → Plotly scatter → 🖥️ clusters by category — products/employees/contracts/company separate on their own. **t-SNE** squashes 384 dimensions into 2 for human eyes, preserving neighborhoods (it's a viewing tool, not the retrieval math). 🏨 Like zooming a globe into a flat map: distances distort, *closeness* survives. Meaning made visible: the machine really does file "similar things" together.
7. *(Day 5 PRO variant replaces steps 1–2 with an LLM call per document: GPT-4.1-nano returns Pydantic `Chunks`, each with `headline + summary + original_text`; all three are embedded, so chunks are findable by questions the raw text never answers — "which document is about the CEO's background?" works even though no sentence says that.)*

## B. Query time (day 3) — 6 steps

1. Reopen the store: `Chroma(persist_directory="vector_db", embedding_function=embeddings)`; `retriever = vectorstore.as_retriever()`. (No re-embedding — vectors are on disk. ⚠️ The *same* embedding model must be used at query time: coordinates from different maps don't line up.)
2. `docs = retriever.invoke("Who is Averi Lancaster?")` — internally: embed the question (same model!) → cosine-distance search → **top-4 chunks** returned.
3. `context = "\n\n".join(doc.page_content for doc in docs)` — the four hits become one readable block.
4. `system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)` — the facts are now physically inside the prompt, under rules like "answer only from the context; say you don't know otherwise." That second rule is the hallucination brake.
5. `llm.invoke([SystemMessage(system_prompt), HumanMessage(question)])` — GPT-4.1-nano answers **from the injected facts**, not its training. (`invoke` with LangChain message objects = the same API call you've made since Week 1, wearing a LangChain costume.)
6. Return `response.content` → Gradio chat. 🖥️ *"Averi Lancaster is the CTO and co-founder…"* — spelled right even though you misspelled it ("Avery") in your question, because retrieval matched **semantics, not keywords**. A keyword search would have found nothing here.

## C. Evaluation (day 4) — how a question is scored

1. Load 150 tests from `tests.jsonl` — each a dict: `question / keywords / reference_answer / category`. **Human-written gold answers** make objective scoring possible.
2. **Retrieval eval:** run `fetch_context(question)` → for each keyword compute `1/rank` of the first chunk containing it (MRR), position-discounted nDCG, and % coverage. Example: keyword found in chunk 4 → MRR contribution 0.25 (1/4). 🧠 **MRR (Mean Reciprocal Rank)** = a scoring system that loves finding the right thing *early*: rank 1 = full credit 1.0, rank 4 = quarter credit, not found = 0; average over everything.
3. **Answer eval:** `answer_question(...)` → an LLM judge compares generated vs reference and returns Pydantic `AnswerEval{feedback, accuracy 1–5, completeness, relevance}` ("wrong ⇒ accuracy 1"). ⚠️ Yes, an LLM grading an LLM sounds circular — mitigations: it sees the reference answer, it's forced into a strict schema, and spot-checking a few by hand is expected.
4. `evaluator.py` aggregates all 150 → color-coded metric cards + per-category bar charts. 🖥️ The dashboard tells you *which question types* your RAG fails — the difference between "it feels dumb sometimes" and an engineering work queue.

## D. Pro upgrades (day 5) — where each runs in the flow

- **Rerank:** after step 2 above, ask the LLM to re-order the 20 retrieved chunks (`response_format=RankOrder` — JSON, again); the right chunk demonstrably moves from ~position 6 → 1, and only the top few feed the prompt. 🏨 A fast librarian fetches 20 books; a senior colleague quickly flips through and hands you the 3 that actually answer the question.
- **Rewrite:** before embedding, a tiny LLM call turns conversational questions into standalone search queries (using history) — follow-ups like *"and his salary?"* finally retrieve properly, because they become "What is Sarah Chen's salary?" instead of the context-free pronoun soup.

### ✅ Checkpoint
1. Why must query-time embedding use the same model as ingest? → *Coordinates only align within one meaning-map; mixing models = searching the wrong city.*
2. Where does the model's knowledge of Averi come from? → *The context pasted into the system prompt at call time — nothing was retrained.*
3. What does chunk_overlap=200 protect against? → *Facts sliced in half at chunk boundaries.*

---
<a id="w6"></a>
# 1️⃣2️⃣ Week 6 — "The Price Is Right", step by step

**What you're running:** `week6/` — five day notebooks plus the reusable `pricer/` package.
**The mission:** given a product description, predict its price. Final scoreboard (average $ error): random 106 → classic ML 68 → GPT-5.1 zero-shot 44.74 → a purpose-built neural network 46.49 → *(Week 7's fine-tune will take it to 39.85)*.
**What you'll learn:** real data engineering at 900k scale, classical ML vs deep learning, and how to train a neural network — the loop at the heart of everything AI.

### 🧠 Machine-learning in one paragraph (read this before anything)
Instead of writing rules ("kitchen appliances cost ~$80"), you show the model thousands of *examples with known answers* and let it find the patterns itself — **training**. If a prediction is off, you compute a **loss** (a single "how wrong were you?" score), then nudge every internal knob (**weight**) slightly in the direction that would have made the loss smaller (**gradient descent** — like a hiker descending a hill in fog by always stepping downhill). Repeat a few million times → competence. **Classical ML** (linear regression, random forests) uses simple, human-inspectable patterns; **deep learning** stacks millions of knobs in layers and finds richer patterns — if you have the data to feed it.

## A. Data curation (`day1.ipynb` + `pricer/parser.py`)

🧠 Garbage in, garbage out — the week *starts* with data, not models. Goal: turn a messy Amazon dump into a clean, balanced, non-leaky dataset.

1. `load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Appliances", split="full")` — hundreds of thousands of raw product rows download from the HuggingFace Hub (this takes a while — it's real data, not a toy).
2. `ItemLoader(category).load()`: the dataset is split into 1,000-row chunks → `ProcessPoolExecutor` parses chunks **on all CPU cores in parallel**. Per row, `parse()` runs the funnel:
   - price parses as a number?
   - in range **$0.50–$999.49**? (kill $0 scrapes and ultra-luxury outliers — the task should be learnable)
   - text ≥600 chars after `scrub()`?
   - `scrub()` detail: drop part numbers via regex `(?=[A-Z0-9]{7,})(?=.*[A-Z])(?=.*\d)` (7+ chars of digits+CAPS = model codes like "B07XJ8CYS5", pure noise for pricing), normalize weight→pounds.
   - pass → an `Item` object; fail → `None` (silently dropped).
3. Concatenate 8 categories → ~900k Items. **Dedupe twice** with the `seen.add()` idiom (titles, then full text). 🧠 Duplicates matter beyond tidiness: if the same item appears in both train and test, your test scores are contaminated — the model "remembers" the answer. Killing duplicates = killing **data leakage**.
4. **Weighted sample to 820k:** `w = price_normalized**2` (boost expensive items — cheap $10 gadgets are everywhere, pricey goods are rare, and without boosting the model would learn "always guess $20"), Automotive ×0.05 (it dominated the pile), then `np.random.choice(len(items), 820_000, replace=False, p=w)`. Re-plot → balanced histogram. 🖥️ Price/length histograms and category bars at every stage — the notebooks *show* you the data transforming.
5. Split **800k train / 10k val / 10k test** → `Item.push_to_hub(...)` — datasets live on the Hub for the rest of the course (Weeks 7 and 8 just download them; and the 10k test items stay **untouched** by all training, ever — the only fair final exam).

## B. LLM pre-processing (`day2.ipynb` + `pricer/batch.py`)

🧠 The idea: the raw scrape is cluttered ("CERTIFIED REFURBISHED!! SHIPS IN 24H…"). An LLM can distill each into a clean 5-line summary. But 800k summaries at normal API prices = fortune. Enter the **Batch API**: hand OpenAI/Groq a *file* of 100k jobs; results arrive within 24h at **50% price**. 🏨 Post your letters in one mail sack at half postage, and pick up the sack of replies next day.

1. `for index, item in enumerate(items): item.id = index` — every item gets a stable number (0…819999).
2. `Batch.create(items, LITE_MODE)` — slices items into groups of 1,000 → `Batch` objects. (`LITE_MODE` = use the 25k smallest/cheapest items for experiments; flip to full later.)
3. `Batch.run()` — per batch: write JSONL — one line per job, each `{"custom_id": "17", "url": "/v1/chat/completions", "body": {model, messages}}` — → `groq.files.create(purpose="batch")` → `groq.batches.create(completion_window="24h")`. 🧠 **`custom_id` is the thread that ties results back**: since batches return *unordered*, each result carries its `custom_id` so you can find item #17. JSONL = JSON per line — a format you can stream file-style instead of one giant array.
4. Later, `Batch.fetch()` — poll `batches.retrieve()`; when `status=="completed"`, download the output file and for each line: `items[int(custom_id)].summary = <LLM's clean 5-line rewrite>`. The stitch-back is a plain dict lookup.
5. Push the finished dataset (`items_lite`/`items_full`) to the Hub. Everything downstream just does `Item.from_hub(...)` — the cleaning cost is paid **once** and shared by every later week.

## C. The model race (`day3`–`day5` + `pricer/evaluator.py`)

🧠 Race design: every contender — from "guess the average" to a 289-million-parameter network — faces the **same 200 test items**, scored the **same way**: `error = |guess − true price|`, averaged. One number per contender. No vibes.

1. Every contender is a function `f(item) -> price`. `evaluate(f, test, size=200)` runs them all through the same `Tester`.
2. `Tester.run_datapoint`: call f → `post_process` (regex the first number out of messy text — models write "The price is $23.99, which is…") → `error = abs(guess - truth)` → traffic-light color → print a colored `$23` per item. Watching hundreds of green/red numbers scroll = instant intuition about a model.
3. **Contenders in order, each beating the last:**
   - `random_pricer` (pure noise) → ~$1,000,000 error. The "you must be better than nothing" bar.
   - `constant_pricer` (always guess the training mean ≈ $106 err) — shockingly strong; *any* real model must beat this. It's also the **scale of the task**: prices are all over $0–$999.
   - `LinearRegression` on 3 features (weight, brand rank, category) — beats constant: features carry signal. (~$95 err)
   - `CountVectorizer` + LinearRegression — **bag of words**: every word gets a learned price-weight; "stainless" pushes up, "cable" pushes down. (~$76 err) Words are features!
   - `RandomForestRegressor` — 100 decision trees, each seeing a random slice of data/features, averaged. 🏨 Ask 100 slightly-different experts and average their guesses — individual quirks cancel. (~$72 err)
   - `xgb.XGBRegressor` — gradient boosting: trees built **sequentially**, each new tree trained to correct the previous ensemble's mistakes. (~$68 err) The classical-ML champion.
4. **The training loop** (day 4 neural network) — memorize these five lines; they are the heartbeat of all deep learning:
```python
for epoch in range(2):                      # one epoch = full pass through all 800k examples
    for X_batch, y_batch in loader:         # batches of 64 examples
        optimizer.zero_grad()               # 1. erase last round's note-taking
        outputs = model(batch_X)            # 2. FORWARD: guess prices (batch of 64)
        loss = MSELoss(outputs, batch_y)    # 3. score the guesses (mean squared error)
        loss.backward()                     # 4. BACKWARD: assign blame to every weight (gradients)
        optimizer.step()                    # 5. nudge every weight slightly downhill
```
🧠 `loss.backward()` = **backpropagation** — the chain rule applied through the whole network at once, computing "for each knob, which way and how strongly would turning it reduce the error?" `optimizer.step()` applies all those nudges at once. Then erase (step 1) and go again. Two epochs over 800k items → the simple NN lands at $63.97.
5. **Frontier zero-shot:** one prompt per item ("Estimate the price… respond with the price") via LiteLLM to GPT-5.1/Claude/Gemini/Grok → GPT-5.1 wins at **$44.74** — with zero training, pure world knowledge. Fine-tuned nano (day 5: JSONL `user/assistant` pairs → `files.create` → `fine_tuning.jobs.create(model="gpt-4.1-nano-2025-04-14", hyperparameters={"n_epochs":1})` → a private custom model) lands at **$75.91** — the honest disappointment: fine-tuning teaches *style* cheaply, not necessarily *accuracy*.
6. **Redemption DNN:** `DeepNeuralNetworkRunner.setup()` — `HashingVectorizer` (converts words to numbers with **no learned vocabulary** — any new word still maps somewhere; crucial because real products at inference time contain unseen words) → a **289M-parameter residual network** (residual = layers get shortcut connections, letting very deep nets train without forgetting) → log-normalize targets (predict log-price: relative errors matter more than absolute) → L1 loss (less rattled by outliers than MSE) → AdamW optimizer + cosine LR schedule (big confident steps early, tiny ones late — 🏨 approaching a destination, you shorten your stride). `train(5)` → validation MAE $58.26 → $53.15 → `save('deep_neural_network.pth')` → evaluate → **$46.49** — beats GPT-5.1 at ~$0 per prediction.

### ✅ Checkpoint
1. Why start with `constant_pricer`? → *It sets the bar any real model must clear and reveals the task's scale (≈$106 spread).*
2. Why does the Batch API halve your bill? → *You trade immediacy: results within 24h instead of seconds = half price.*
3. Name the 5 heartbeat lines. → *zero_grad → forward → loss → backward → step. Every deep-learning system ever trained runs some version of them.*

---
<a id="w7"></a>
# 1️⃣3️⃣ Week 7 — QLoRA training, end to end in 14 steps

**What you're running:** `week7/day2.ipynb` (local: build the training dataset) then Google Colabs days 3–4 (the actual training on a borrowed GPU) and day 5 (evaluation).
**The mission:** teach Llama-3.2-3B one narrow skill — reading a product summary and answering with just a price, like `"149.00"` — well enough to reach **$39.85** average error: the best single model of the course.
**What you'll learn:** fine-tuning — the difference between *telling* a model (prompting), *showing* it documents (RAG), and *changing the model itself* (training).

### 🧠 Fine-tuning in plain English
**Fine-tuning** = continue training a pre-trained model on a few thousand examples of exactly the input→output behavior you want. The model already knows language; you're teaching a *habit*.

**Full fine-tuning** would update all 3 billion weights — needs multiples of the model's size in GPU memory just for the training bookkeeping. Out of reach on free hardware.

**LoRA (Low-Rank Adaptation)** — the trick that makes today possible: **freeze the whole base model** and inject tiny new matrices (the **adapters**) next to the attention layers; train *only those*. They're like a thin cheat-sheet clipped onto a frozen textbook: <1–5% extra parameters, but enough capacity to learn a narrow behavior. 🏨 Don't retrain the surgeon; hand the surgeon a new clipboard of this hospital's procedures.

**QLoRA** = do LoRA **on top of a 4-bit quantized base model** (the Week 3 trick again). The frozen base sits compressed at ~1.7 GB; only the small adapters train at full precision. Result: serious fine-tuning on a **free Colab T4**.

### 🗺️ The pipeline
```
local:  800k items → build prompt/completion pairs → push to Hub
Colab:  download pairs → quantize base → attach LoRA adapters → SFTTrainer.train()
        → push adapter to Hub → day 5: base+adapter → evaluate on the untouched 10k
```

---

1. **Build training pairs** (`day2.ipynb`, local): `Item.from_hub("items_full")` → tokenize all summaries with `AutoTokenizer("meta-llama/Llama-3.2-3B")` → plot token-count histogram → most summaries fit under **CUTOFF = 110 tokens**. 🧠 Measure, then cut — not the other way round.
2. `item.make_prompts(tokenizer, 110, True)` — token-level truncate (safer than char-count: tokenizers are the model's reality), then each item becomes:
   `prompt = "What does this cost to the nearest dollar?\n\n<summary>\n\nPrice is $"`, `completion = "149.00"`.
   🧠 **That's the whole curriculum**: thousands of (question, answer) pairs in identical format. The model will learn "after 'Price is $', emit digits and a period." Narrow = learnable.
3. `Item.push_prompts_to_hub("items_prompts_full", ...)` — the training schema `{"prompt", "completion"}` goes to the Hub, so the Colab can download it without your laptop being awake.
4. **Colab, days 3–4:** `pip install bitsandbytes==0.48.2 trl==0.25.1 transformers==4.57.6` + `wget util.py` (the tester). **trl** = Transformer Reinforcement Learning library — despite the name, it hosts `SFTTrainer`, the industrial-grade trainer wrapper. Version pins matter: these libraries move fast and break pairwise.
5. Log in to HF **and** Weights & Biases — `wandb.login()`. **wandb** = live telemetry for training: loss curves in your browser while the GPU works. 🏨 A heart-rate monitor for the patient on the operating table — if loss stops falling, you want to know *now*, not in an hour.
6. Set the hyper-parameter wall (in caps — these are the knobs of the whole run):
   - `EPOCHS` (1 lite / 3 full) — complete passes over the data. Too many = **overfitting** (memorizing answers instead of learning the skill).
   - `BATCH_SIZE` (32/256) — examples per update; bigger = steadier gradients, more memory.
   - `MAX_SEQUENCE_LENGTH=128` — prompt+completion fits in 128 tokens (that's why CUTOFF=110 existed).
   - `LORA_R` (32/256) — adapter "rank" = the cheat-sheet's width; `LORA_ALPHA=2r` scales its influence; dropout 0.1 fights overfitting.
   - `TARGET_MODULES` — *which* layers get adapters: attention only (lite) or attention+MLP (full).
   - `LEARNING_RATE=1e-4` — step size of each knob-nudge. Too big: the model thrashes; too small: it never arrives.
   - `OPTIMIZER="paged_adamw_32bit"` — Adam's state pages to CPU RAM when the GPU is full (the Q in QLoRA's bag of tricks).
   - `use_bf16 = cuda_capability >= 8` — T4 is capability 7.5, so fp16 fallback; on better GPUs, bf16.
7. `load_dataset(DATASET_NAME)` → train/val/test — the pairs from step 3 arrive in the trainer's expected format.
8. Quantize: the familiar 4-bit `BitsAndBytesConfig` from Week 3 Day 4 → base model ≈1.7 GB on the GPU.
9. `LoraConfig(r, lora_alpha, lora_dropout=0.1, task_type="CAUSAL_LM", target_modules)` — the adapter blueprint: rank, scale, which layers to surgically modify.
10. `SFTConfig(...)` — epochs, batch, LR schedule (`warmup_ratio=0.01` = a few hundred tiny steps before trusting big ones — cold engines stall — then a `cosine` decay of the learning rate), `max_grad_norm=0.3` (a leash against explosive updates), `logging_steps`, `eval_steps`, `save_steps`, and `push_to_hub=True` — **checkpoints upload as it trains**, so a Colab crash costs minutes, not hours.
11. `SFTTrainer(model=base_model, train_dataset=train, eval_dataset=val, peft_config=lora_parameters, args=train_parameters)` — wire the four things together. **Only the LoRA matrices will train**; the 3B base stays frozen. (SFT = Supervised Fine-Tuning: ordinary examples with known right answers.)
12. `fine_tuning.train()` — the industrial version of Week 6's heartbeat (`zero_grad → forward → loss → backward → step`), plus checkpointing, logging, LR scheduling, and batching handled for you. 🖥️ tqdm progress bars; wandb shows train loss and val loss falling epoch by epoch. ⚠️ Train loss falls but **val loss rises** = overfitting alarm. Colab died mid-run? The companion notebook resumes from the last checkpoint (`PeftModel.from_pretrained(..., is_trainable=True)`).
13. `fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)` — **only the adapter** (~85 MB lite / ~300 MB full — compare: the base model is ~6 GB) lands on your HF account. You trained the cheat-sheet, not the surgeon.
14. **Day 5 eval:** 4-bit base + `PeftModel.from_pretrained(base, "you/price-...")` — the cheat-sheet clipped on → generate on 200 test prompts → `evaluate(predictor, test)` → 🖥️ **base-only $110.72 (worse than guessing the average!) → lite $65.40 → full $39.85**. `results.ipynb` charts all 17 course contenders. The model now answers *only* "149.00" — a specialist, not a chatterbox.

### ✅ Checkpoint
1. What's frozen, what trains? → *All 3B base weights: frozen (and 4-bit). The injected LoRA adapters: trainable.*
2. Why does the final artifact weigh ~85–300 MB? → *It contains only the adapter matrices — the base model stays a public download, combined at load time.*
3. What signals overfitting during training? → *Validation loss rising while training loss keeps falling.*

---
<a id="w8"></a>
# 1️⃣4️⃣ Week 8 — One full autonomous agent run, message by message

**What you're running:** `price_is_right.py` (or day 4's `agent.plan()`). Everything you learned — messages, tools, streaming, RAG, fine-tunes, neural nets — converges into one system that watches deal sites and texts your phone when it finds a bargain.
**What you'll learn:** what "autonomous agent" concretely means: **one long conversation with a frontier model, where the model chooses which of your functions to run next, in a loop, until the job is done.**

### 🗺️ The whole system on one diagram
```
RSS feeds → Scanner → deals → GPT-5.1 (planner) says "estimate this one"
                                  ├─ Preprocessor (Ollama llama3.2) cleans the text
                                  ├─ FrontierAgent  = GPT-5.1 + RAG over similar deals
                                  ├─ SpecialistAgent= Modal GPU → your Week-7 fine-tune
                                  └─ NN Agent       = Week-6 neural network forward pass
                               → weighted blend → "notify!" → Pushover → 📱 phone buzzes
```
Every box is a class ending in `Agent`, and every agent exposes the same single method: `price(description) -> float`. That interface discipline is what lets the ensemble mix a rented model, your own fine-tune, and your own network as equals.

---

### Before the loop
```python
messages = [{system: "You find great deals on bargain products using your tools, and notify the user of the best bargain."},
            {user: "First, use your tool to scan the internet for bargain deals. Then for each deal, use your tool to estimate its true value. Then pick the single most compelling deal ... notify the user. Then just reply OK."}]
tools = [scan_the_internet_for_bargains, estimate_true_value, notify_user_of_deal]   # JSON descriptions
```
🧠 Look at the user message: **it's a to-do list written in English.** The plan lives in the *prompt* (AutonomousPlanningAgent), not in your code's control flow — that's what makes it "autonomous." (Day 4's deterministic `PlanningAgent` hardcodes the same plan in Python: `for deal in deals: estimate...` — same outcome, zero autonomy, a useful comparison. Determinism is a feature when you can afford it.)

### The run, iteration by iteration

| # | finish_reason | Model asks to run | Your code actually runs | Messages list grows by |
|---|---|---|---|---|
| 1 | `tool_calls` | `scan_the_internet_for_bargains` | Scanner: RSS → filter → GPT-5-mini picks 5 | +2 (request, result) |
| 2–5 | `tool_calls` ×4 | `estimate_true_value(description=…)` ×4 | Preprocessor + 3 experts → blend | +2 each |
| 6 | `tool_calls` | `notify_user_of_deal(…)` | Claude writes blurb → Pushover → 📱 | +2 |
| 7 | `stop` | — (replies "OK") | loop exits; opportunity returned | +1 |

**Loop iteration 1 — the model asks to scan**
```python
response.choices[0].finish_reason == "tool_calls"
message.tool_calls[0].function.name == "scan_the_internet_for_bargains"   # no arguments
```
Your code runs `scanner.scan(memory=memory)`:
- `ScrapedDeal.fetch()` — `feedparser` pulls 3 **RSS feeds** from dealnews.com (predefined category URLs — RSS is the site's machine-readable article wire) → up to 10 entries each → `requests.get` each deal page → BeautifulSoup extracts details/features → truncated (title≤100, details/features≤500 chars).
- URLs already in `memory` (the agent's notepad, `memory.json`) are filtered out — so re-runs see only **new** deals.
- GPT-5-mini picks the 5 best via **structured outputs** (`.parse(response_format=DealSelection)` — Pydantic fields whose descriptions explicitly warn against the "$XXX off" phrasing trap, since a "discount" is not a price).
- Returned to the planner as `results.model_dump_json()`.

Messages appended: `{assistant: <tool_call scan>}`, `{tool: '{"deals":[{product_description, price, url} × 5]}'}`.

**Iterations 2–5 — the model prices each deal, one at a time**
Each iteration: the model emits `estimate_true_value(description="The Samsung Galaxy Watch Ultra...")` → your code runs the **EnsembleAgent**:
1. `Preprocessor.preprocess(description)` — llama3.2 (local Ollama on `localhost:11434`, or Groq via the `PRICER_PREPROCESSOR_MODEL` env var) rewrites the messy deal text into the clean `Title/Category/Brand/...` format the specialists were trained on. 🧠 Garbage-in protection: the experts were built on *clean* data, so someone must clean before pricing.
2. Three experts price the rewrite (parallel in concept, sequential in code):
   - **FrontierAgent** — MiniLM-embed the description → Chroma `products_vectorstore` (Week 8 day 2: 400k product summaries) top-5 similars *with their real prices in metadata* → GPT-5.1 with "Potentially related product: … Price is $…" context. RAG, final form.
   - **SpecialistAgent** — `modal.Cls.from_name("pricer-service","Pricer")` → a remote GPU ( Modal = rent containers/GPUs by the second, warm thanks to `scaledown_window=1200` and `MIN_CONTAINERS`) → **your Week-7 QLoRA model** generates "379.00".
   - **NeuralNetworkAgent** — Week 6's DNN: one forward pass, hardcoded `Y_MEAN/Y_STD` undo the log-normalization.
3. `frontier*0.8 + specialist*0.1 + neural_network*0.1` → one number back as the tool result. 🏨 Three doctors' opinions, weighted by track record, as a single verdict. (Why 80/10/10? The frontier model is most reliable — and the weights are a hypothesis you can change in one line.)

After 4 estimates, the message list has grown to ~10 entries — **full history resent every iteration** (statelessness, always, everywhere).

**Iteration 6 — the model notifies**
```python
tool_call: notify_user_of_deal(description="...", deal_price=350.0, estimated_true_value=773.81, url="https://dealnews...")
```
Your code: guard rail first (`if self.opportunity: ignore — only once` — a chatty model must not spam your phone), else `MessagingAgent.notify(...)` → **Claude writes 2–3 exciting sentences** (a different vendor for copywriting!) → `requests.post("https://api.pushover.net/1/messages.json")` (Pushover = a paid doorbell service for apps) → 📱 **your phone buzzes**: *"Deal Alert! Price=$350, Estimate=$773.81 — the Samsung Galaxy Watch Ultra… <url>"*. `self.opportunity = Opportunity(deal, estimate, discount=423.81)` is captured for the caller.

**Iteration 7 — done**
`finish_reason == "stop"`, content `= "OK"` → loop exits, `plan()` returns the opportunity → the framework appends it to `self.memory` → `write_memory()` persists to `memory.json` → the Gradio table updates. `gr.Timer(300)` does it all again in 5 minutes — the agent runs forever, unattended.

🖥️ The whole time, the dashboard's log panel streams color-coded lines (`[Scanner Agent]`, `[Frontier Agent]`, `[Specialist Agent]`…) via the queue→generator plumbing (QueueHandler → queue → generator yields HTML → Gradio renders; ANSI color codes converted to `<span>` styling by `log_utils.reformat`, last 18 lines shown).

### ✅ Checkpoint
1. Where does the agent's "plan" live? → *In the user prompt as an English to-do list — the model chooses tool order each iteration.*
2. Why can one weighted-average mix GPT-5.1, your fine-tune and a neural network? → *All three implement the identical `price(description) -> float` interface.*
3. What stops the phone from ringing twice for one deal? → *The `self.opportunity` guard + already-seen URLs in `memory.json`.*

---

<a id="appa"></a>
# 📎 Appendix A — The universal 6-step pattern (every project, one shape)

```
1. CONFIGURE   load .env, check keys, connect client        (every week, cell 1)
2. GATHER      scrape URL / RSS / audio / DB / dataset      (W1 scraper, W3 drive, W6 Amazon, W8 RSS)
3. PREPARE     build messages / chunks / prompts / JSONL    (system+user; one-shot examples; schemas)
4. CALL        chat.completions / pipeline / generate / train  (the model does its thing)
5. LOOPS & TOOLS  while tool_calls: run tool, append, recall   (W2D4 → W8 finale)
6. PRESENT     Markdown / Gradio / push notification / chart   (and EVALUATE — W4/W5/W6 discipline)
```

🧠 How to *use* this: when you open any new LLM codebase, hunt for the six. "Where's CONFIGURE?" (dotenv, clients). "What gets GATHERed?" (URLs? files? feeds?). "How is the prompt PREPARED?" (system+user? few-shot? JSON mode?). "What's the CALL?" (API? local generate? a trainer?). "Is there a LOOP?" (tools? epochs? batches?). "How is it PRESENTED and *evaluated*?" The answers form a mental map in ten minutes. If you can trace where each of the six happens in a new codebase, you can read any LLM project.

---

<a id="appb"></a>
# 📎 Appendix B — Complete run order for the whole course

| Order | What | Where | Needs |
|---|---|---|---|
| 0 | Setup: `uv sync`, `.env` keys | repo root | OpenAI key |
| 1 | `week1/day1.ipynb` | local | OpenAI |
| 2 | `week1/day2.ipynb` (+ Ollama install) | local | OpenAI, optional Ollama |
| 3 | `week1/day4.ipynb` → `week1/day5.ipynb` → `week1 EXERCISE` | local | OpenAI |
| 4 | `week2/day1.ipynb` (providers, debate) | local | OpenAI (+optional others) |
| 5 | `week2/day2.ipynb` → `day3.ipynb` → `day4.ipynb` → `day5.ipynb` → `extra.ipynb` | local | OpenAI (Gradio UIs open in browser) |
| 6 | `week3/day1–5.ipynb` → Colabs | Colab T4 | Google + HF account/token |
| 7 | `week4/day3.ipynb` → `day4` → `day5` | local | OpenAI (+compiler install) |
| 8 | `week5/day1` → **`day2`** (builds vector_db) → `day3` → `day4` → `day5`; then `implementation/ingest.py` → `app.py` → `evaluator.py` | local | OpenAI |
| 9 | `week6`: day1 → day2 (or skip via Hub) → day3 → day4 → day5 → redemption | local | OpenAI, HF, optional Groq |
| 10 | `week7/day2.ipynb` → Colabs (train, eval) | local + Colab | HF, wandb; A100 for full run |
| 11 | `week8`: day1 (Modal setup + deploy) → day2 (builds products_vectorstore) → day3 → day4 → day5 → `uv run price_is_right.py` | local + Modal cloud | OpenAI, HF, Modal, Pushover |

**Golden rules that repeat every week:** run notebooks top-to-bottom; re-run `load_dotenv(override=True)` after editing `.env`; check you're on the right kernel/runtime; evaluate before you celebrate; delete the generated folders (`vector_db/`, `products_vectorstore/`, `batches/`) to force clean rebuilds.

**Dependencies between weeks (why the order matters):** Week 6's datasets feed Week 7's training and Week 8's specialists; Week 7's adapter is served by Week 8's Modal service; Week 8's RAG store (day 2) must exist before the FrontierAgent runs; Week 2's agent loop is the skeleton of Week 8's planner. Skip a week and a later one will literally fail to find a file or a model.

---

<a id="appc"></a>
# 📎 Appendix C — Plain-English glossary (every term used in this guide)

**Term → what it actually means** (with the week where it first bites)

| Term | Plain English |
|---|---|
| **Agent** | Code where an LLM decides which of your functions to run next, in a loop, until done (W2D4 → W8) |
| **API key** | Secret billing/password string (`sk-proj-…`) proving you may call a service (W1D1) |
| **Attention** | The transformer mechanism: each token "looks at" every other token and weighs what matters (W3) |
| **Autoregressive** | Generating one token at a time, each prediction fed back in (W3D4) |
| **Backpropagation** | The chain rule applied through the whole network: assigns blame for the loss to every weight (W6) |
| **Bag of words** | Treating text as just its words + counts — every word gets a learned weight (W6) |
| **Batch API** | File-based async inference: results within 24h, half price (W6) |
| **Batch size** | Examples per training update (W6/W7) |
| **bfloat16/fp16** | Half-precision number formats — halve memory, keep the math sane (W3) |
| **Chroma** | An open-source vector database: store (vector+text+metadata), query by similarity (W5/W8) |
| **Chunking** | Splitting documents into ~1000-char pieces so retrieval can find small passages (W5) |
| **Colab / T4** | Google's free rented notebook + NVIDIA T4 GPU (16 GB) (W3/W7) |
| **Cosine similarity** | Angle-between-vectors measure of meaning-similarity, ignoring length (W5) |
| **Data leakage** | Test information sneaking into training — inflates scores dishonestly (W6) |
| **Dataset (HF)** | A downloadable table of examples hosted on the HuggingFace Hub (W6) |
| **Deduplication** | Removing exact/near copies — a leakage defense (W6) |
| **Embedding** | A list of numbers where similar meanings are nearby points (W5/W8) |
| **Ensemble** | Combining several models' outputs (weighted average) into one (W8) |
| **Epoch** | One complete pass through all training data (W6/W7) |
| **Eval harness** | The same fixed test given to every contender, scored identically (W4/W5/W6) |
| **finish_reason** | The model's status word: `"stop"` = answered; `"tool_calls"` = work order attached (W2D4) |
| **Fine-tuning** | Continued training on narrow examples to install a skill or style (W6D5/W7) |
| **Forward pass** | Pushing inputs through the network to get outputs (W3/W6) |
| **Generator (`yield`)** | A function that pauses at each yield, handing back partial results (W2) |
| **Gradient descent** | Nudging all weights slightly downhill on the loss, millions of times (W6/W7) |
| **Gradio** | Python library that turns functions into shareable web UIs (W2 → W8) |
| **JSON mode** | API constraint forcing the model to emit valid JSON (W1D5) |
| **JSONL** | JSON per line — the format of batch jobs and training files (W6/W7) |
| **Learning rate** | Step size of each weight nudge — too big thrashes, too small crawls (W6/W7) |
| **LoRA** | Train tiny adapter matrices beside frozen attention layers (W7) |
| **Loss** | One number: how wrong the model was. Training exists to shrink it (W6/W7) |
| **MRR** | Mean Reciprocal Rank — retrieval scoring that rewards finding truth early (W5) |
| **Memory illusion** | "Remembering" = resending the whole message list; nothing persists server-side (W1D4→) |
| **Messages list** | The `[{role, content}, …]` structure every chat call sends (everywhere) |
| **Modal** | Cloud platform to run containers/GPUs on demand, billed by the second (W8) |
| **One-shot / few-shot** | Including 1/several worked examples in the prompt to set format (W1D5) |
| **Overfitting** | Memorizing training data; symptom: training loss ↓ but validation loss ↑ (W7) |
| **PEFT / adapter** | Parameter-Efficient Fine-Tuning; the small trainable add-on modules (W7) |
| **Pipeline (HF)** | The 6-step wrapper: tokenize → GPU → forward → softmax → label (W3D2) |
| **Pushover** | Paid push-notification service; your agent's doorbell (W8) |
| **Quantization** | Storing weights in fewer bits (16→4): ~4× smaller, slightly blurrier (W3D4, W7) |
| **RAG** | Retrieval-Augmented Generation: fetch relevant passages, paste into prompt, answer from them (W5/W8) |
| **Residual connection** | Layer shortcut that lets very deep networks train (W6 DNN) |
| **RSS feed** | A site's machine-readable article wire; robots read it instead of scraping HTML (W8) |
| **Softmax** | Converts raw scores (logits) into probabilities summing to 1 (W3) |
| **Stateless** | The API remembers nothing between calls — you own all state (everywhere) |
| **Structured outputs** | API-enforced schema: model must fill the Pydantic form exactly (W8) |
| **System prompt** | The `system` role: persona, rules, output format — set by code, not the user (everywhere) |
| **Tensor** | A multi-dimensional array of numbers on CPU/GPU (W3) |
| **t-SNE** | Squashes high-dim vectors to 2D for humans, preserving neighborhoods (W5) |
| **Tokenizer** | Model-specific text ↔ token-id translator; must match the model (W3/W7) |
| **Tool calling** | The model requests a function run via a JSON work order; YOUR code executes it (W2D4) |
| **Transformer** | The attention-based architecture underlying all models in this course (W3) |
| **Vector store** | Database of embeddings supporting nearest-neighbor queries (W5/W8) |
| **wandb** | Weights & Biases — live training dashboards/telemetry (W7) |
| **Zero-shot** | No examples in the prompt — pure instruction (everywhere) |

---

*You now hold the complete map — concepts in the week guides, execution here, and every term in your own language. Build something of your own with it.* 🎓
