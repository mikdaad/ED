# 📘 Week 5 — Complete Beginner's Guide (RAG: Retrieval-Augmented Generation)

This guide explains **every file** in `week5/` of the *LLM Engineering* repo — what each piece does, why it exists, and how it all fits together. (Companions to the Weeks 1–2 and 3–4 guides.)

> **The one-sentence summary:** Until now the model only knew what was in its prompt. **Week 5 teaches it to "look things up"**: build a searchable knowledge base of vectors, retrieve the right documents for each question, stuff them into the prompt, and — critically — *measure* how well that works. This is **RAG**, the most commercially used LLM technique in industry.

---

# 🧠 PART 0 — The Big Picture: What is RAG and why?

## The problem

Ask GPT *"Who won the Insurellm Innovator of the Year award in 2023?"* — it has no idea. Insurellm is a **fictional insurance company created for this course**; its data exists only in `week5/knowledge-base/`. Options to fix this:

| Approach | Idea | Downside |
|---|---|---|
| **Fine-tuning** | Retrain the model on your documents | Expensive, slow, bad at pure *facts* (it learns style, not knowledge) |
| **Long context** | Paste ALL documents into every prompt | This knowledge base is ~175K tokens — costly and slow per question, and models degrade over huge contexts |
| **RAG** ✅ | Keep documents in a **searchable index**; for each question, **retrieve** only the relevant bits and put those in the prompt | You must build the index — which is exactly this week |

**The exam analogy:** RAG turns the LLM from a closed-book student into an **open-book** one. It still answers in its own words, but it gets to consult the relevant pages first.

## The universal RAG pipeline (memorize this)

```
ONCE ("ingestion"):
  documents → split into CHUNKS → embed each chunk into a VECTOR → store in a VECTOR DATABASE

EVERY QUESTION ("query time"):
  question → embed the question → find the NEAREST chunk vectors (semantic search)
           → paste those chunks into the system prompt → LLM answers with the facts
```

## New vocabulary

- **Embedding** — a list of numbers (a vector) representing a text's *meaning*, produced by an embedding model. Texts with similar meanings get **nearby vectors** — so "Who is Avery?" sits close to text about *Avery Lancaster, founder* even though the words differ. (Week 3's "embeddings" mention now pays off.)
- **Vector database (Chroma)** — a database specialized in storing vectors and finding the nearest ones fast (`collection.query(...)`).
- **Chunk** — a piece of a document (~200–1000 characters). We retrieve *chunks*, not whole files: small enough to be precise, big enough to be self-contained.
- **Retriever** — the component that turns a question into the top-k most relevant chunks.
- **MRR / nDCG** — scores for "did the right documents come back, near the top?" (explained in Day 4).
- **LLM-as-a-judge** — using one LLM to grade another LLM's answers against reference answers.

## What's in the folder (and the arc of the week)

```
knowledge-base/            The corpus: 4 folders of markdown
  company/                 about, careers, culture, overview
  employees/               ~20 HR records ("Avery Lancaster.md", ...)
  products/                8 products: Markellm, Carllm, Homellm, Rellm,
                           Lifellm, Healthllm, Bizllm, Claimllm
  contracts/               30 client contracts

day1.ipynb   Naive RAG: dictionary keyword lookup (see why it's not enough)
day2.ipynb   THE FOUNDATIONS: chunk → embed → Chroma → visualize in 2D/3D
day3.ipynb   Classic RAG in ~15 lines with LangChain + Gradio chatbot
day4.ipynb   EVALUATION: 150 test questions, MRR/nDCG, LLM-as-a-judge
day5.ipynb   "GO PRO": LLM-smart chunking, reranking, query rewriting

implementation/            The day2/3 pipeline packaged as importable modules
pro_implementation/        The day5 pipeline packaged (parallel, resilient)
app.py                     The PRODUCT: Gradio chat UI + "retrieved context" panel
evaluator.py               The QUALITY DASHBOARD: run all evals from a browser
evaluation/                test.py (loads tests) + eval.py (metrics) + tests.jsonl (150 Qs)
```

The story: **build it naively (day 1) → build it properly (days 2–3) → measure it (day 4) → make it state-of-the-art (day 5)** — then package, ship, and score it (`implementation/`, `pro_implementation/`, `app.py`, `evaluator.py`).

---

# 🟦 PART 1 — The notebooks, day by day

## 1.1 `day1.ipynb` — Naive RAG: keyword lookup (and why it fails)

**Goal:** a Q&A assistant for Insurellm employees. Day 1's "simplistic, brute-force" version needs no vectors at all:

### Loading the knowledge into a dictionary

```python
knowledge = {}

filenames = glob.glob("knowledge-base/employees/*")
for filename in filenames:
    name = Path(filename).stem.split(' ')[-1]     # "Avery Lancaster.md" -> "Lancaster"
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[name.lower()] = f.read()        # knowledge["lancaster"] = the file text

filenames = glob.glob("knowledge-base/products/*")
for filename in filenames:
    name = Path(filename).stem                    # "Carllm.md" -> "Carllm"
    knowledge[name.lower()] = f.read()
```

`glob` lists files matching a pattern; `Path(...).stem` is the filename without extension. Result: `{"lancaster": "...", "carllm": "...", ...}` — ~28 keys.

### Retrieval by exact word match

```python
def get_relevant_context(message):
    text = ''.join(ch for ch in message if ch.isalpha() or ch.isspace())  # strip punctuation
    words = text.lower().split()
    return [knowledge[word] for word in words if word in knowledge]       # dict lookups!
```

Ask *"Who is Lancaster and what is carllm?"* → the words `lancaster` and `carllm` are dictionary keys → those two full documents come back. The helper `additional_context(message)` joins them (or returns *"There is no additional context…"* if nothing matched), and the chat function stuffs the result into the system prompt — the same Week 2 "dynamic system prompt" trick:

```python
def chat(message, history):
    system_message = SYSTEM_PREFIX + additional_context(message)
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages)   # MODEL = gpt-4.1-nano (cheap!)
    return response.choices[0].message.content

gr.ChatInterface(chat, type="messages").launch(inbrowser=True)
```

### Why this can't be the real solution

It only works when the user *literally names* a file key. *"Who won the IIOTY award in 2023?"* matches nothing (the answer lives inside an employee file). *"Who went to Manchester University?"* matches nothing. There's no notion of *meaning*. Day 1 exists to make you feel this limitation — the fix is semantics, which needs **vectors**.

---

## 1.2 `day2.ipynb` — The foundations: CHUNKS → VECTORS → CHROMA → visualize

The most important notebook of the week. Three parts:

### PART A — Chunking the documents (LangChain loaders + splitter)

```python
folders = glob.glob("knowledge-base/*")
documents = []
for folder in folders:
    doc_type = os.path.basename(folder)                     # 'employees', 'products', ...
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader,
                             loader_kwargs={'encoding': 'utf-8'})
    for doc in loader.load():
        doc.metadata["doc_type"] = doc_type                 # tag each doc with its folder
        documents.append(doc)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
```

- LangChain `Document` objects = `{page_content, metadata}`; each gets a `doc_type` tag (used later for coloring and filtering).
- **`RecursiveCharacterTextSplitter`** tries to split on paragraph breaks first (`\n\n`), then sentences, then words, then characters — keeping chunks readable. `chunk_size=1000` = max characters per chunk; **`chunk_overlap=200`** = neighbors share 200 characters, so a fact sitting at a boundary appears complete in *both* chunks (this tiny detail dramatically improves retrieval).
- The notebook also counts the corpus with `tiktoken` (Week 1 Day 4 again): ~175K tokens — far too much to paste into every prompt, proving the need for retrieval.

### PART B — Embeddings + the vector store

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")   # FREE, runs locally
#embeddings = OpenAIEmbeddings(model="text-embedding-3-large")      # better, costs $

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings,
                                    persist_directory=db_name)

sample_embedding = vectorstore._collection.get(limit=1, include=["embeddings"])["embeddings"][0]
# -> "There are ~1,7xx vectors with 384 dimensions in the vector store"
```

Reading that closely:
- `all-MiniLM-L6-v2` is the small sentence-embedding model from Week 3's world (HuggingFace) — each chunk becomes **384 numbers**. The commented alternative, OpenAI's `text-embedding-3-large`, makes **3,072-number** vectors that capture meaning better (the `implementation/` modules use it).
- `Chroma.from_documents(...)` = embed every chunk + store, persisted to disk in `vector_db/` so later days just reopen it.
- `vectorstore._collection` — peeking inside Chroma to count vectors and inspect one. **Embeddings are not magic text compression — they are coordinates in "meaning space."**

### PART C — *Seeing* the vector space (t-SNE + Plotly)

```python
result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])            # every chunk's 384-dim vector

tsne = TSNE(n_components=2, random_state=42)        # 384-D -> 2-D for human eyes
reduced_vectors = tsne.fit_transform(vectors)

fig = go.Figure(data=[go.Scatter(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),      # color = doc_type
    text=[...], hoverinfo='text')])
fig.show()   # then the same again with n_components=3 for a 3D plot
```

**t-SNE** is a dimensionality-reduction algorithm: it squeezes 384 dimensions into 2 while keeping similar points near each other. The payoff plot is beautiful: **product chunks cluster with products, employee chunks with employees, contracts with contracts** — pure meaning, no human labeled anything. This cluster picture *is* why vector search works: a question lands near the chunks that answer it.

---

## 1.3 `day3.ipynb` — Classic RAG in ~15 lines (LangChain + Gradio)

With the store built on disk, the actual RAG application is astonishingly small:

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)   # reopen, no re-embed

retriever = vectorstore.as_retriever()               # the search component
llm = ChatOpenAI(temperature=0, model_name=MODEL)    # the answer component

SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

def answer_question(question: str, history):
    docs = retriever.invoke(question)                                # 1. RETRIEVE
    context = "\n\n".join(doc.page_content for doc in docs)          # 2. ASSEMBLE
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response = llm.invoke([SystemMessage(content=system_prompt),
                           HumanMessage(content=question)])          # 3. GENERATE
    return response.content

gr.ChatInterface(answer_question).launch()
```

That's the entire industry-standard pattern — retrieve → augment → generate. Now *"Who won the IIOTY award?"* retrieves the right chunk even though no filename was mentioned, because the **question vector is semantically near** the award text.

**The temperature sidebar** (worth keeping): temperature controls which tokens get picked during generation. `temperature=0` = always pick the top-probability token → predictable answers (right default for factual RAG); higher = more variety. It is *not* exactly "creativity" — if you want creativity, use the system prompt. (And temperature 0 alone isn't perfectly reproducible — you need a seed too.)

Try `retriever.invoke("Who is Avery?")` in the notebook — you can inspect *which* chunks come back and in what order, which sets up Day 4's question: *how good is that ordering, actually?*

---

## 1.4 `day4.ipynb` — Evaluation: the most business-critical day

> Notebook's own emphasis: *"People are often focused on RAG architecture and frameworks… But even more important: evaluations!"*

You can't improve what you don't measure. So the course ships a **test suite of 150 questions** and two evaluation families.

### The test suite (`evaluation/test.py` + `tests.jsonl`)

```python
class TestQuestion(BaseModel):
    question: str
    keywords: list[str]          # words that MUST appear in good retrieved context
    reference_answer: str        # a gold answer to compare against
    category: str

def load_tests():                # reads one JSON object per line (JSONL format)
    ...
```

```
{"question": "Who won the prestigious IIOTY award in 2023?",
 "keywords": ["Maxine", "Thompson", "IIOTY"],
 "reference_answer": "Maxine Thompson won the prestigious Insurellm Innovator of the Year (IIOTY) award in 2023.",
 "category": "direct_fact"}
```

The 150 tests span **7 difficulty categories**: `direct_fact` (70), `temporal` (20), `spanning` (20 — info across docs), `comparative` (10), `numerical` (10), `relationship` (10), `holistic` (10 — whole-corpus questions). Per-category results later reveal *which kinds of questions your RAG is bad at* — far more useful than one overall score.

### Retrieval metrics (`evaluation/eval.py`)

```python
def calculate_mrr(keyword, retrieved_docs):
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword.lower() in doc.page_content.lower():
            return 1.0 / rank          # found at rank 1 -> 1.0, rank 3 -> 0.33
    return 0.0                         # not found -> 0
```

- **MRR (Mean Reciprocal Rank)** — for each keyword, score `1/rank` of the *first* retrieved chunk containing it; average over keywords and tests. Beginner example: keyword in chunk #1 → 1.0; in #4 → 0.25; absent → 0. High MRR = "the right stuff is at the **top**", which matters because only the top chunks fit in the prompt.
- **nDCG (normalized Discounted Cumulative Gain)** — a fairer cousin: it credits **every** relevant chunk but discounts each by position (`relevance / log2(rank+1)`), then divides by the *ideal* ordering's score so results land between 0 and 1.

```python
class RetrievalEval(BaseModel):
    mrr: float
    ndcg: float
    keywords_found: int
    total_keywords: int
    keyword_coverage: float      # % of expected keywords present anywhere in top-k
```

`evaluate_retrieval(test)` calls the *same* `fetch_context()` your app uses (imported from `implementation/answer.py` — you're evaluating the real thing, not a copy) and returns those scores. Day 4's example run: `mrr=0.17, keyword_coverage=66.7%` for test #0 — the point is you now *know*, instead of *hoping*.

### Answer quality: LLM-as-a-judge

```python
class AnswerEval(BaseModel):
    feedback: str
    accuracy: float       # 1 (wrong — wrong answers MUST score 1) to 5 (perfect)
    completeness: float   # does it cover everything in the reference answer?
    relevance: float      # does it answer exactly the question, no rambling?

judge_messages = [
    {"role": "system", "content": "You are an expert evaluator assessing the quality of answers.
      Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores
      for perfect answers."},
    {"role": "user", "content": f"Question: {test.question}\nGenerated Answer: {generated_answer}
      \nReference Answer: {test.reference_answer}\n ... score accuracy/completeness/relevance 1-5"}]

judge_response = completion(model=MODEL, messages=judge_messages, response_format=AnswerEval)
```

**An LLM grades the RAG system's answer** against the reference, forced (via structured output) to return numbers — so evaluation is as scalable as the system itself. The strict "wrong ⇒ accuracy 1" and "only 5/5 for perfect" rules fight grade inflation. `evaluate_answer` returns `(AnswerEval, generated_answer, retrieved_chunks)`.

---

## 1.5 `day5.ipynb` — "Let's go PRO!" Advanced RAG

Four upgrades, dropping LangChain for raw `openai`/`chromadb`/`litellm`:

### Upgrade 1 — Let an LLM do the chunking (document pre-processing!)

Instead of blindly cutting every 1000 characters, **GPT-4.1-nano reads each document and decides the chunks itself**, writing a *headline* and *summary* for each (this is Week 2 Day 4's structured-output tool pattern, applied to ingestion):

```python
class Chunk(BaseModel):
    headline: str = Field(description="A brief heading for this chunk ... most likely to be surfaced in a query")
    summary: str = Field(description="A few sentences summarizing the content of this chunk to answer common questions")
    original_text: str = Field(description="The original text of this chunk ... exactly as is, not changed in any way")

    def as_result(self, document):           # what actually gets embedded:
        return Result(page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text, ...)

class Chunks(BaseModel):
    chunks: list[Chunk]

def process_document(document):
    response = completion(model=MODEL, messages=make_messages(document), response_format=Chunks)
    return [chunk.as_result(document) for chunk in Chunks.model_validate_json(
        response.choices[0].message.content).chunks]
```

The prompt tells the model roughly how many chunks to aim for (`len(text) // AVERAGE_CHUNK_SIZE + 1`) and to overlap ~25%. Note `as_result`: the **embedded text = headline + summary + original** — so a chunk about Alex Lancaster is *also* findable by asking about "the founder" or "CTO", because the headline/summary added those retrieval hooks. Genius-level upgrade, one LLM call per document (with `tqdm` progress; the module version parallelizes it).

### Storing with OpenAI embeddings (raw Chroma this time)

```python
def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    emb = openai.embeddings.create(model="text-embedding-3-large", input=texts).data
    vectors = [e.embedding for e in emb]

    collection = chroma.get_or_create_collection(collection_name)
    collection.add(ids=[str(i) for i in range(len(chunks))],
                   embeddings=vectors, documents=texts, metadatas=[c.metadata for c in chunks])
```

Then the same t-SNE 2D/3D visualizations as Day 2 — run them on both stores and compare the clustering!

### Upgrade 2 — Reranking: a second opinion on ordering

Vector search ranks chunks by *embedding* distance, which is fast but crude. Fix: retrieve generously (`RETRIEVAL_K = 10`... 20), then ask an LLM to **re-order them by actual relevance**:

```python
class RankOrder(BaseModel):
    order: list[int] = Field(description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number")

def rerank(question, chunks):
    # system: "You are a document re-ranker ... Reply only with the list of ranked chunk ids"
    # user:   the question + "# CHUNK ID: 1 ... 2 ... 3 ..." texts
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    order = RankOrder.model_validate_json(response.choices[0].message.content).order
    return [chunks[i - 1] for i in order]
```

The notebook's demo is delicious: for *"Who went to Manchester University?"* the right chunk sits at position **#6+** of 20 raw results; after `rerank` it's **position 0**. Since only the top chunks go into the prompt, reranking directly buys accuracy.

```python
def fetch_context(question):
    chunks = fetch_context_unranked(question)   # vector search
    return rerank(question, chunks)             # LLM re-orders
```

### Upgrade 3 — Query rewriting (fixing follow-up questions)

Embedding a bare follow-up like *"and his salary?"* retrieves garbage. So a tiny LLM call first **rewrites the question into a standalone search query**, using the conversation history:

```python
def rewrite_query(question, history=[]):
    """Rewrite the user's question to be a more specific question that is more likely
       to surface relevant content in the Knowledge Base."""
    # prompt includes the history + "Respond only with a single, refined question ..."
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content
```

### The full pro pipeline

```python
def answer_question(question, history=[]):
    query = rewrite_query(question, history)     # 1. sharpen the question
    chunks = fetch_context(query)                # 2. vector search + rerank
    messages = make_rag_messages(question, history, chunks)  # 3. context into system prompt
    response = completion(model=MODEL, messages=messages)    # 4. answer
    return response.choices[0].message.content, chunks
```

(Plus `make_rag_messages` now labels each extract with its source file, and the system prompt reminds the model it "will be evaluated for accuracy, relevance and completeness".)

---

# 🟩 PART 2 — The packaged product & the evaluation dashboard

The notebooks were experiments. These files are the same work organized as **software** — and they're what actually gets run.

## 2.1 `implementation/` — the standard pipeline as modules

**`ingest.py`** (run once: `python ingest.py`): the Day 2/3 pipeline — LangChain `DirectoryLoader` → `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)` → **`OpenAIEmbeddings("text-embedding-3-large")`** → Chroma at `week5/vector_db`. Ends with the familiar "There are N vectors with 3,072 dimensions" print.

**`answer.py`** (imported by the app and the evaluator):

```python
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model_name=MODEL)
RETRIEVAL_K = 10

def fetch_context(question):                    # shared by app AND evaluator
    return retriever.invoke(question, k=RETRIEVAL_K)

def combined_question(question, history=[]):
    """Combine all the user's messages into a single string."""
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question

def answer_question(question, history=[]):
    combined = combined_question(question, history)   # retrieval sees earlier turns too
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
    messages.extend(convert_to_messages(history))     # LangChain-format the history
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs                     # answer AND the evidence
```

Two lessons: (1) **`combined_question` is a poor-man's query rewriting** — concatenating prior user turns so retrieval handles follow-ups (Day 5's LLM version is the fancier fix); (2) `answer_question` returns `(answer, docs)` so the UI can *show its work*.

## 2.2 `pro_implementation/` — the Day 5 pipeline, productionized

Same architecture as the notebook, plus three production-grade hardening tricks:

```python
MODEL = "groq/openai/gpt-oss-120b"      # note the provider-prefixed LiteLLM naming!
AVERAGE_CHUNK_SIZE = 100                # finer-grained chunks than the notebook's 500
WORKERS = 3
wait = wait_exponential(multiplier=1, min=10, max=240)

@retry(wait=wait)                       # tenacity: on rate-limit errors, wait and retry
def process_document(document): ...

def create_chunks(documents):
    with Pool(processes=WORKERS) as pool:      # 3 parallel processes (LLM calls are slow)
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
```

- **`@retry(wait=wait_exponential(...))`** (library `tenacity`) — when the API rate-limits you, back off (10s → 20s → 40s… capped at 240s) and retry automatically. The difference between a demo and a tool.
- **`multiprocessing.Pool`** — chunk 60+ documents concurrently.
- In **`pro_answer.py`**: `RETRIEVAL_K = 20, FINAL_K = 10`, and **dual retrieval** — search with the original question *and* the LLM-rewritten one, merge & de-duplicate (`merge_chunks`), rerank the combined pool, keep the top 10. Two nets are better than one.

## 2.3 `app.py` — the product (chat + "show your sources" panel)

```python
def chat(history):
    last_message = history[-1]["content"]
    prior = history[:-1]
    answer, context = answer_question(last_message, prior)   # ← the implementation module
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)

with gr.Blocks(title="Insurellm Expert Assistant", theme=gr.themes.Soft(...)) as ui:
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(label="💬 Conversation", height=600, type="messages")
            message = gr.Textbox(placeholder="Ask anything about Insurellm...")
        with gr.Column(scale=1):
            context_markdown = gr.Markdown(label="📚 Retrieved Context", height=600)

    message.submit(put_message_in_chatbot, inputs=[message, chatbot],
                   outputs=[message, chatbot]).then(chat, inputs=[chatbot],
                   outputs=[chatbot, context_markdown])
```

The Week 2 Blocks pattern (`.submit(...).then(...)`) with one great addition: the right-hand panel renders every retrieved chunk with its **source filename** in orange (`format_context`). Users can see *why* the AI answered as it did — the single most trust-building feature in real RAG products.

## 2.4 `evaluator.py` — the quality dashboard

A Gradio admin app wrapping `evaluate_all_retrieval()` / `evaluate_all_answers()`:

- Two big buttons: **Retrieval Evaluation** (all 150 tests → avg **MRR**, **nDCG**, **keyword coverage**) and **Answer Evaluation** (avg **accuracy / completeness / relevance** on a 1–5 scale).
- Metric "cards" are **color-coded by thresholds** (`get_color`: ≥0.9 green, ≥0.75 amber, else red — and ≥4.5 / ≥4.0 for the 1–5 scales), so quality regressions jump out visually.
- `gr.BarPlot` shows each metric **broken down by question category** — e.g. discovering that `holistic` and `spanning` questions score lower is your to-do list.
- `progress=gr.Progress()` gives the evaluation a live progress bar while the generators (`yield test, result, progress`) stream results in.

This closes the loop the course keeps emphasizing: **build → evaluate → improve → re-evaluate** (e.g. run the dashboard on `implementation/`, then on `pro_implementation/`, and see the numbers move).

---

# 🎓 PART 3 — Consolidation

## Concepts introduced, week by day

| Concept | Where | One-liner |
|---|---|---|
| RAG = retrieval + generation | D1–5 | Open-book exam for LLMs |
| Naive keyword retrieval | D1 | Dict lookup by exact word — fast but blind to meaning |
| LangChain loaders & metadata | D2 | `DirectoryLoader`, `Document{page_content, metadata}` |
| **Chunking** & overlap | D2 | `RecursiveCharacterTextSplitter(1000/200)`; overlap preserves boundary facts |
| **Embeddings** | D2 | Meaning as coordinates; MiniLM 384-D vs OpenAI 3072-D |
| **Chroma vector DB** | D2–5 | Persist, query nearest, peek via `._collection` |
| t-SNE visualization | D2, D5 | 384-D → 2-D; clusters prove semantics |
| `as_retriever()` RAG loop | D3 | retrieve → stuff context → `llm.invoke` |
| Temperature | D3 | Token-selection dial; 0 for factual RAG |
| **MRR / nDCG / coverage** | D4 | Rank-aware retrieval scores |
| **LLM-as-a-judge** | D4 | Structured 1–5 accuracy/completeness/relevance |
| JSONL test suites | D4 | 150 gold questions in 7 categories |
| **LLM chunking** + headline/summary | D5 | Structured outputs at ingestion time |
| **Reranking** | D5 | Retrieve 20, LLM re-orders, keep top 10 |
| **Query rewriting** | D5 | Turn follow-ups into standalone search queries |
| Retry/backoff, multiprocessing | pro | `tenacity` + `Pool` = production ingestion |
| Sources in the UI | app | Show retrieved chunks → user trust |

## The whole week in one diagram

```
                 ┌─────────────────── INGESTION (once) ───────────────────┐
knowledge-base/  │  LangChain split (D2)   or   LLM chunks + summaries (D5)│
 .md files ─────►│  chunks ──► embeddings (MiniLM / 3-large) ──► CHROMA   │
                 └─────────────────────────────────────────────────────────┘
                                                                      │ nearest vectors
                 ┌──────────────────── QUERY TIME ────────────────────┘
question ─► (rewrite for context, D5) ─► embed ─► top-k chunks ─► (LLM rerank, D5)
        ─► system prompt = persona + chunks ─► gpt-4.1-nano ─► answer (+ sources shown in UI)
                                 │
                 150 tests ─► MRR/nDCG/coverage + LLM judge ─► evaluator dashboard
```

## Glossary additions

- **RAG** — Retrieval-Augmented Generation; answering from documents fetched at question time.
- **Ingestion** — the one-time indexing pipeline (load → chunk → embed → store).
- **Chunk overlap** — duplicated text at chunk boundaries so facts aren't cut in half.
- **Vector store / collection** — the Chroma database / its named table of vectors+text+metadata.
- **t-SNE** — algorithm that squashes high-dim vectors to 2D/3D for visualization.
- **MRR** — average of 1/rank for the first chunk containing each expected keyword.
- **nDCG** — position-discounted retrieval score normalized to 0–1.
- **JSONL** — one JSON object per line; the test-suite format.
- **Structured output** — forcing an LLM response to match a Pydantic schema (`response_format=`).
- **Reranking** — LLM re-orders retrieved chunks by true relevance.
- **Query rewriting** — LLM turns contextual follow-ups into standalone search queries.
- **Tenacity / exponential backoff** — auto-retry with growing waits on rate limits.

## Running it yourself

1. From the repo root: `uv sync` (gets `langchain`, `chromadb`, `litellm`, `tenacity`, etc.), with your `OPENAI_API_KEY` in `.env`. Costs are pennies — everything uses `gpt-4.1-nano` by default.
2. `cd week5`, then run the notebooks in order — day2 builds `vector_db/`, day3 needs it. (Delete the folder to force a rebuild.)
3. Product: `python ingest.py` (in `implementation/`), then `python app.py` from `week5/` → chat with the Insurellm assistant and watch the sources panel. Good test questions: *"Who won the IIOTY award in 2023?"*, *"Who went to Manchester University?"*, *"What does Carllm cost?"*.
4. Advanced: run `pro_implementation/ingest.py` (uses an LLM per document — slower; WORKERS=3 in parallel, auto-retries), point `app.py` at `pro_implementation.answer` to A/B them.
5. Quality: `python evaluator.py` → click Run Evaluation (retrieval eval is fast; answer eval makes 150 LLM calls + 150 judge calls, so minutes and a few cents).
6. Beginner exercises the notebooks suggest implicitly: change chunk sizes and re-run the evaluator; swap MiniLM ↔ `text-embedding-3-large`; try `RETRIEVAL_K` of 5 vs 20 with and without reranking; ask a question that isn't in the KB and check the bot admits it.

## Where this leaves the course

You can now build the **most requested enterprise LLM system** end-to-end — and prove it works. Week 6 turns to fine-tuning (teaching models *style and skill* rather than feeding them facts — now you understand why RAG handles the facts), Week 7 trains from scratch, and Week 8 assembles autonomous agents — which will lean on everything here: tools (Week 2), local models (Week 3), code generation & evaluation (Week 4), and RAG (Week 5).
