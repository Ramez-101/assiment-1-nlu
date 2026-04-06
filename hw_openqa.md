# hw_openqa.ipynb — Full Documentation

**Course:** CS224u, Stanford, Spring 2022  
**Authors:** Christopher Potts and Omar Khattab  
**Topic:** Few-shot Open-Domain Question Answering (OpenQA) with ColBERT Retrieval

---

## What This Notebook Does

This notebook builds and evaluates **few-shot open-domain question answering (OpenQA)** systems. The core idea is: given only a question (no pre-provided passage), can a general-purpose language model answer it correctly? The notebook progressively improves on this, going from a naive baseline all the way to an original system with BM25-guided few-shot examples and beam search decoding.

There are four main approaches explored:

| Approach | Passage Given | Few-shot Examples | Retrieval |
|---|---|---|---|
| Open QA with no context | No | No | No |
| Few-shot QA | Yes (gold) | Yes | No |
| Zero-shot OpenQA with ColBERT | No | No | ColBERT |
| Few-shot OpenQA (homework) | No | Yes | ColBERT |

---

## Setup & Dependencies

### Libraries Required
- `torch` — Deep learning framework (GPU recommended)
- `transformers` — Hugging Face tokenizers and causal LMs (GPT-Neo family)
- `datasets` — Loads SQuAD from Hugging Face
- `faiss-cpu` or `faiss-gpu` — Vector similarity search (used by ColBERT)
- `openai` — For GPT-3 API (optional, requires an API key)
- `scipy`, `numpy` — Math utilities
- `spacy`, `ujson`, `gitpython` — Miscellaneous utilities
- `rank_bm25` — BM25 retrieval (used in the original system)

### External Components
- **ColBERT** — Cloned from GitHub (`cgpotts/cs224u` branch), provides neural passage retrieval
- **ColBERTv2 checkpoint** — Pre-trained on MS MARCO, downloaded from Stanford (~388 MB)
- **cs224u collection index** — Pre-indexed 125,563 passages aligned with SQuAD/bake-off data

### Reproducibility
Seeds are set for `numpy`, `random`, and `torch` at the start. GPT-3 results are not reproducible since the API does not accept a seed.

---

## Data: SQuAD

The notebook uses the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset, loaded via Hugging Face `datasets`.

### Key Objects

**`SquadExample`** — A named tuple representing one QA example:
```
id, title, context, question, answers
```

**`squad_dev`** — Full SQuAD validation split (~10,570 examples)  
**`dev_exs`** — A deterministic 200-example sample of `squad_dev` (used for evaluation throughout)  
**`squad_train`** — Full SQuAD training split (~87,599 examples) used to build few-shot prompts

---

## Language Models

Two interchangeable LM interfaces are provided. Both return the same output format.

### `run_eleuther(prompts, temperature=0.1, top_p=0.95, **generate_kwargs)`

Runs one of the [EleutherAI GPT-Neo models](https://www.eleuther.ai/) locally via Hugging Face.  
Default model: `gpt-neo-125m` (configurable via `eleuther_model_name`).

**What it does:**
1. Tokenizes the input prompts (left-padded for batch processing)
2. Calls `model.generate()` with sampling enabled
3. Extracts per-token probabilities for the generated tokens
4. Isolates the "answer" — the first line of generated text after any leading newlines
5. Returns a list of result dicts

**Returns:** List of dicts, one per prompt:
```python
{
  "prompt": str,
  "generated_text": str,
  "generated_tokens": list[str],
  "generated_probs": list[float],
  "generated_answer": str,        # The extracted answer (first line)
  "generated_answer_tokens": list[str],
  "generated_answer_probs": list[float]
}
```

**Note on tokenization:** GPT-Neo uses `Ġ` for leading spaces and `Ċ` for newlines. These are decoded back to normal characters in the answer string.

---

### `run_gpt3(prompts, engine="text-curie-001", temperature=0.1, top_p=0.95, **gpt3_kwargs)`

Calls the OpenAI GPT-3 completions API.

**Important:** Set `openai.api_key` to your OpenAI API key before using. The `engine` must start with `"text"` (e.g., `text-curie-001`, `text-davinci-001`). Instruct-tuned models are excluded.

**Returns:** Same dict format as `run_eleuther`, making the two functions drop-in interchangeable.

---

### `_find_generated_answer(tokens, newline="\n")`

Internal helper. Scans through generated tokens to find the "answer" — the first non-newline line of text. Leading newlines are included for token-probability alignment but skipped in the final answer string.

---

## Evaluation

### Metric Functions

**`normalize_answer(s)`** — Lowercases, removes articles (a/an/the), punctuation, and extra whitespace. Used for both EM and F1.

**`compute_exact(a_gold, a_pred)`** — Returns 1 if normalized strings match exactly, else 0.

**`compute_f1(a_gold, a_pred)`** — Token-level F1 score between normalized gold and predicted answers. Handles empty answers gracefully.

**`compute_f1_from_tokens(gold_toks, pred_toks)`** — Same as above but accepts pre-tokenized lists.

### Main Evaluation Function

**`evaluate(examples, prompts, gens)`**

Evaluates a list of model generations against SQuAD gold answers.

- For each example, picks the **best** EM and F1 across all gold answer strings (SQuAD sometimes has multiple valid answers)
- Returns a dict with:
  - `"macro_f1"` — Mean F1 across all examples (primary bake-off metric)
  - `"em_per"` — Exact match percentage
  - `"examples"` — Full list of result dicts with per-example scores

---

## Approaches

### 1. Open QA with No Context

**`evaluate_no_context(examples, gen_func=run_eleuther, batch_size=20)`**

The simplest baseline: feed the raw question to the LM and see what it generates. No context, no examples. The LM is not told what kind of output is expected.

**Result (gpt-neo-125m):** macro_f1 ≈ 0.044 — unsurprisingly low, since the model has no cue to answer concisely.

---

### 2. Few-shot QA (with gold passages)

**`build_few_shot_qa_prompt(ex, squad_train, n_context=2, joiner="\n\n")`**

Builds a prompt by prepending `n_context` randomly-sampled SQuAD training examples. Each example is formatted as:
```
Title: <title>

Background: <passage>

Q: <question>

A: <answer>
```
The target example ends with `A:` (no answer), so the LM completes it.

**`evaluate_few_shot_qa(examples, squad_train, gen_func, batch_size=20, n_context=2)`**

Evaluates the few-shot QA approach across all examples in batches.

**Result (gpt-neo-125m, n_context=1):** macro_f1 ≈ 0.077 — improved over no-context, because the prompt format makes the task clearer.

---

### 3. ColBERT Retrieval

The ColBERT neural retriever is used to find relevant passages from a pre-indexed corpus of 125,563 passages, without relying on gold passages.

**Setup:**
```python
with Run().context(RunConfig(experiment='notebook')):
    searcher = Searcher(index=index_name)
collection = Collection(path=collection_path)
```

**`searcher.search(query, k=3)`** — Returns `(passage_ids, ranks, scores)` for the top-k passages.

**`success_at_k(examples, k=20)`** — Retrieval evaluation metric. Checks whether any of the top-k retrieved passages contains one of the gold answer strings as a substring (using DPR normalization). Returns the fraction of examples with at least one hit.

---

### 4. Zero-shot OpenQA with ColBERT

**`build_zero_shot_openqa_prompt(question, passage, joiner="\n\n")`**

Builds a prompt using a retrieved passage (no training examples):
```
Title: <title from passage>

Background: <passage body>

Q: <question>

A:
```
The passage string is formatted as `"Title | Background"` in the collection.

**`evaluate_zero_shot_openqa(examples, joiner, gen_func, batch_size=20)`**

Retrieves top-1 passage for each question and builds prompts from them. No few-shot examples are used.

**Result (gpt-neo-125m):** macro_f1 ≈ 0.055

---

## Homework Questions

### HW1: Few-shot OpenQA with No Context (2 pts)

**`build_few_shot_no_context_prompt(question, train_exs, joiner="\n\n")`**

Builds prompts like:
```
Q: <train question 1>

A: <train answer 1>

...

Q: <target question>

A:
```
No passage is provided — this tests how well the LM performs with just format cues and no context.

**`evaluate_few_shot_no_context(examples, squad_train, batch_size, n_context, joiner, gen_func)`**

Samples `n_context` random training examples per test question, builds prompts, and evaluates.

**Test:** `test_build_few_shot_no_context_prompt` and `test_evaluator` verify correctness.

---

### HW2: Few-shot OpenQA (2 pts)

**`build_few_shot_open_qa_prompt(question, passage, train_exs, joiner="\n\n")`**

Combines retrieved passage + few-shot training examples in a single prompt:
```
Title: <train title 1>

Background: <train context 1>

Q: <train question 1>

A: <train answer 1>

...

Title: <retrieved title>

Background: <retrieved passage>

Q: <target question>

A:
```

**`evaluate_few_shot_open_qa(examples, squad_train, batch_size, n_context, joiner, gen_func)`**

Retrieves top-1 passage per question and builds few-shot prompts.

**Test:** `test_build_few_shot_open_qa_prompt` and `test_evaluator` verify correctness.

---

### HW3: Answer Scoring (2 pts)

This section implements a joint scoring function that combines passage relevance and answer probability.

**Scoring formula:**

```
score(answer, passage, question) = P(passage | question) × P(answer | prompt(question, passage))
```

Where:
- `P(passage | question)` = softmax over the top-k ColBERT scores
- `P(answer | prompt)` = product of per-token probabilities from the LM

**`get_passages_with_scores(question, k=5)`**

Retrieves top-k passages and returns:
- `passages` — list of passage strings
- `passage_probs` — softmax-normalized scores (numpy array)

**`answer_scoring(passages, passage_probs, prompts, gen_func=run_eleuther)`**

For each passage/prompt pair, runs `gen_func` and computes:
```
score = product(answer_token_probs) × passage_prob
```
Returns a list of `(score, gen_dict)` pairs **sorted by descending score** — the best answer first.

**`answer_scoring_demo(question)`**

Example usage: retrieves passages, builds zero-shot prompts, scores answers, returns the top result.

---

### HW4: Original System (3 pts)

The original system (implemented inside an `if 'IS_GRADESCOPE_ENV' not in os.environ` guard) significantly improves on the baselines. It introduces:

#### Key Design Decisions:

1. **BM25 for few-shot example selection:** Instead of randomly sampling training examples, a BM25 index over `context + question` strings is used to find the most semantically related training examples for each retrieved passage.

2. **Multi-passage retrieval + answer scoring:** Top-k passages are retrieved (default k=2). A separate LM call is made for each passage, and the scored answer from all passages is used to pick the best prediction.

3. **Length-normalized scoring:** The answer probability is computed as the geometric mean of per-token probabilities (`prob^(1/n_tokens)`) rather than the raw product. This prevents long answers from being unfairly penalized.

4. **Beam search decoding:** `num_beams=4` (no temperature sampling) was found to outperform temperature-based sampling across a grid search.

5. **Hyperparameter search:** Both `temperature` and `num_beams` are swept over a grid; `tune_temperature` and `tune_n_beams` manage this search and display a summary table.

#### Key Functions (Original System):

**`squad_bm25`** — Class that builds a BM25 index from SQuAD training data and retrieves top-k relevant examples for a query.

**`build_few_shot_open_qa_prompt(question, passage, squad_index, k=2, joiner)`** *(redefined)* — Uses BM25 to select relevant training examples instead of random sampling.

> ⚠️ **Bug note:** The original code uses `query` (undefined) instead of `question` in the BM25 lookup line. The correct call should be:  
> `squad_index.select_relevant_docs(passage + " " + question, k)`

**`run_eleuther_v2(prompts, **generate_kwargs)`** — Variant of `run_eleuther` that passes all generation kwargs through, allowing both `temperature`+`do_sample=True` or `num_beams`+`do_sample=False` modes.

**`answer_scoring(passages, passage_probs, prompts, gen_func, num_beams, temperature)`** *(redefined)* — Extended scoring with length normalization and support for both decoding strategies.

**`original_system(question, gen_func, temperature, num_beams, k_passages, k_train_qa, joiner)`** — End-to-end function: retrieves passages → selects BM25 few-shot examples → scores answers → returns best result dict.

**`tune_temperature(examples, system, temperature_list)`** — Grid search over temperature values; displays a DataFrame of macro_f1 scores.

**`tune_n_beams(examples, system, num_beams_list)`** — Grid search over num_beams values; displays a DataFrame of macro_f1 scores.

**`final_system(question)`** — Wrapper that calls `original_system` with the best found parameters: `num_beams=4`, `k_passages=2`, `k_train_qa=2`.

#### Results Table:

| num_beams | temperature | macro_f1 |
|---|---|---|
| 2 | n/a | 0.139352 |
| 3 | n/a | 0.138919 |
| **4** | **n/a** | **0.143387** ← best |
| n/a | 0.1 | 0.124275 |
| n/a | 0.6 | 0.135449 |

---

## Bake-off

**`create_bakeoff_submission()`**

Loads unlabeled questions from `data/openqa/cs224u-openqa-test-unlabeled.txt` (400 questions), runs `final_system` on each, and writes results to `cs224u-openqa-bakeoff-entry.json`.

The output is a dict mapping each question string to the full generation dict (must contain `"generated_answer"`).

---

## Overall Results Summary

| System | macro_F1 |
|---|---|
| No context (gpt-neo-125m) | 0.044 |
| Zero-shot OpenQA + ColBERT (gpt-neo-125m) | 0.055 |
| Few-shot QA with gold passage (gpt-neo-125m) | 0.077 |
| Original system: BM25 few-shot + beam search (gpt-neo-125m) | **0.143** |

---

## Known Issues / Notes

1. **Bug in original system:** `build_few_shot_open_qa_prompt` (the redefined version) uses undefined variable `query` — should be `question`.
2. The `answer_scoring` function in the original system requires **exactly one** of `temperature` or `num_beams` to be set (enforced via `np.logical_xor`).
3. The notebook is designed to run on Colab with GPU. CPU-only is supported but very slow for the 1.3B+ parameter models.
4. GPT-3's `openai.api_key` is set to `None` in the code — users must fill in their own key.
5. `np.product` is deprecated in newer NumPy versions — use `np.prod` instead.
