# OpenQA Assignment Notes

This workspace contains one notebook file saved with a `.py` extension:

- `assiment 1.py` (Jupyter notebook JSON content)

## What This Notebook Does

The notebook builds and evaluates **few-shot open-domain question answering (OpenQA)** systems using:

1. A retriever (ColBERT) to find candidate passages.
2. An autoregressive language model (Eleuther or GPT-3 style API) to generate answers.
3. Evaluation with exact match (EM) and macro F1 on SQuAD-style examples.

It walks through:

- Zero-shot QA (question only)
- Few-shot QA (with training examples)
- Few-shot OpenQA (retrieved passages + few-shot prompting)
- Answer scoring that combines retrieval probability and answer probability
- An "original system" section for bake-off submission

## Fixes Applied

The following correctness fixes were applied in `assiment 1.py`:

1. **Undefined variable fix** in the original-system `build_few_shot_open_qa_prompt`:
   - Replaced `query` with `question` when selecting relevant SQuAD examples.
2. **Safer passage parsing** in both `build_few_shot_open_qa_prompt` functions:
   - Added fallback behavior when a retrieved passage does not contain the `" | "` separator.
   - Uses a default title (`"Retrieved passage"`) and full passage text as background in that case.

## How To Run

Because the file is notebook JSON, open and run it as a notebook in your IDE/Jupyter environment.

Typical run order:

1. Run setup/import cells.
2. Run model setup cells (`run_eleuther` / optional GPT-3 setup).
3. Run retrieval setup (ColBERT index + searcher).
4. Run evaluation and homework sections.
5. Run `create_bakeoff_submission()` to produce:
   - `cs224u-openqa-bakeoff-entry.json`

## Important Runtime Notes

- Some cells require large model downloads and/or GPU.
- The ColBERT retrieval components depend on external packages and index files.
- If running locally (not Colab), make sure notebook shell commands are adapted as needed.

## Quick Function Guide

- `build_few_shot_no_context_prompt`: few-shot prompt without retrieval context.
- `evaluate_few_shot_no_context`: evaluates no-context few-shot prompts.
- `build_few_shot_open_qa_prompt`: prompt with retrieved passage + few-shot examples.
- `evaluate_few_shot_open_qa`: evaluates retrieved few-shot OpenQA.
- `get_passages_with_scores`: gets top-k passages and normalized retrieval scores.
- `answer_scoring`: ranks generated answers using retrieval and generation probabilities.
- `final_system`: final original-system entry point used for bake-off generation.

