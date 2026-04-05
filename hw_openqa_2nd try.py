"""
hw_openqa.ipynb — Assignment Solutions
CS224u, Stanford, Spring 2022
Few-shot Open-Domain Question Answering with ColBERT Retrieval

This file contains clean solution code for all 4 homework tasks.
Run this alongside the notebook setup cells (imports, model loading, etc.)
"""

# ============================================================
# IMPORTS (already done in notebook setup cells)
# ============================================================
# collections, random, numpy, scipy.special.softmax,
# run_eleuther, run_gpt3, searcher, squad_train, dev_exs,
# evaluate, get_tokens  — all defined earlier in the notebook


# ============================================================
# HW QUESTION 1: Few-shot OpenQA with NO context [2 points]
# ============================================================
# TASK 1: Build prompts that look like:
#
#   Q: What is pragmatics?
#
#   A: The study of language use
#
#   Q: Who is Bert?
#
#   A: Bert is one of the Muppets.
#
#   Q: What was Stanford University founded?
#
#   A:
#
# TASK 2: Evaluate this approach using sst.experiment-style batching.


def build_few_shot_no_context_prompt(question, train_exs, joiner="\n\n"):
    """
    Build a few-shot prompt with NO background passage — question/answer pairs only.

    Parameters
    ----------
    question : str
        The target question to answer.
    train_exs : iterable of SquadExample
        Few-shot examples sampled from squad_train.
        Each has .question and .answers attributes.
    joiner : str
        String used to join segments of the prompt (default: double newline).

    Returns
    -------
    str
        The full prompt string ending with "A:" ready for the LM to complete.
    """
    segs = []

    # Add each training example as a Q/A pair (no title or background)
    for ex in train_exs:
        segs.append(f"Q: {ex.question}")
        segs.append(f"A: {ex.answers[0]}")

    # Add the target question with an open "A:" for the LM to complete
    segs.append(f"Q: {question}")
    segs.append("A:")

    return joiner.join(segs)


def evaluate_few_shot_no_context(
        examples,
        squad_train,
        batch_size=20,
        n_context=2,
        joiner="\n\n",
        gen_func=run_eleuther):
    """
    Evaluate few-shot OpenQA with no context (no retrieved passage).

    For each example, samples n_context random SQuAD training examples,
    builds a prompt using build_few_shot_no_context_prompt, runs it through
    gen_func, and collects results.

    Parameters
    ----------
    examples : list of SquadExample
        Test examples to evaluate (e.g. dev_exs).
    squad_train : list of SquadExample
        Source of few-shot training examples.
    batch_size : int
        Number of examples per batch sent to gen_func.
    n_context : int
        Number of few-shot Q/A examples to include in each prompt.
    joiner : str
        Passed to build_few_shot_no_context_prompt.
    gen_func : callable
        Either run_eleuther or run_gpt3.

    Returns
    -------
    dict
        Result dict from evaluate(), containing macro_f1, em_per, examples.
    """
    prompts = []
    gens    = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]

        # For each example in the batch, sample n_context training examples
        # and build the prompt
        ps = [
            build_few_shot_no_context_prompt(
                ex.question,
                random.sample(squad_train, k=n_context),
                joiner=joiner
            )
            for ex in batch
        ]

        # Run all prompts in the batch through the language model
        gs = gen_func(ps)

        prompts += ps
        gens    += gs

    return evaluate(examples, prompts, gens)


# ============================================================
# HW QUESTION 2: Few-shot OpenQA WITH retrieved passage [2 points]
# ============================================================
# TASK 1: Build prompts that look like:
#
#   Title: <train title 1>
#   Background: <train passage 1>
#   Q: <train question 1>
#   A: <train answer 1>
#
#   ...
#
#   Title: <retrieved title>
#   Background: <retrieved passage>
#   Q: <target question>
#   A:
#
# TASK 2: Evaluate this approach with ColBERT retrieval.


def build_few_shot_open_qa_prompt(question, passage, train_exs, joiner="\n\n"):
    """
    Build a few-shot OpenQA prompt using a retrieved passage + training examples.

    Parameters
    ----------
    question : str
        The target question to answer.
    passage : str
        A ColBERT-retrieved passage formatted as "Title | Background text".
    train_exs : iterable of SquadExample
        Few-shot examples from squad_train.
        Each has .title, .context, .question, .answers attributes.
    joiner : str
        String used to join segments (default: double newline).

    Returns
    -------
    str
        The full prompt string ending with "A:" ready for the LM to complete.
    """
    segs = []

    # Add each training example as a full Title/Background/Q/A block
    for ex in train_exs:
        segs.append(f"Title: {ex.title}")
        segs.append(f"Background: {ex.context}")
        segs.append(f"Q: {ex.question}")
        segs.append(f"A: {ex.answers[0]}")

    # Parse the retrieved passage — format is "Title | Background text"
    title, background = passage.split(" | ", 1)

    # Add the target question block with open "A:"
    segs.append(f"Title: {title}")
    segs.append(f"Background: {background}")
    segs.append(f"Q: {question}")
    segs.append("A:")

    return joiner.join(segs)


def evaluate_few_shot_open_qa(
        examples,
        squad_train,
        batch_size=20,
        n_context=2,
        joiner="\n\n",
        gen_func=run_eleuther):
    """
    Evaluate few-shot OpenQA with ColBERT-retrieved passages.

    For each example, retrieves the top-1 passage using ColBERT,
    samples n_context random SQuAD training examples, builds a prompt,
    runs it through gen_func, and collects results.

    Parameters
    ----------
    examples : list of SquadExample
        Test examples to evaluate.
    squad_train : list of SquadExample
        Source of few-shot training examples.
    batch_size : int
        Number of examples per batch sent to gen_func.
    n_context : int
        Number of few-shot examples to include in each prompt.
    joiner : str
        Passed to build_few_shot_open_qa_prompt.
    gen_func : callable
        Either run_eleuther or run_gpt3.

    Returns
    -------
    dict
        Result dict from evaluate(), containing macro_f1, em_per, examples.
    """
    prompts = []
    gens    = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]

        # Retrieve top-1 passage for each question using ColBERT
        retrieval_results = [searcher.search(ex.question, k=1) for ex in batch]
        passages = [searcher.collection[r[0][0]] for r in retrieval_results]

        # Build prompts: retrieved passage + random few-shot examples
        ps = [
            build_few_shot_open_qa_prompt(
                ex.question,
                psg,
                random.sample(squad_train, k=n_context),
                joiner=joiner
            )
            for ex, psg in zip(batch, passages)
        ]

        # Run all prompts through the language model
        gs = gen_func(ps)

        prompts += ps
        gens    += gs

    return evaluate(examples, prompts, gens)


# ============================================================
# HW QUESTION 3: Answer Scoring [2 points]
# ============================================================
# TASK: Implement joint scoring of answers using:
#
#   score(answer, passage, question) =
#       P(passage | question) x P(answer | prompt(question, passage))
#
# Where:
#   - P(passage | question) = softmax over ColBERT retrieval scores
#   - P(answer | prompt)    = product of per-token LM probabilities


def get_passages_with_scores(question, k=5):
    """
    Retrieve top-k passages and compute softmax-normalized passage probabilities.

    Parameters
    ----------
    question : str
        The question to retrieve passages for.
    k : int
        Number of passages to retrieve.

    Returns
    -------
    passages : list of str
        The k retrieved passage strings.
    passage_probs : np.array of float
        Softmax-normalized retrieval scores (sum to 1).
    """
    # Retrieve top-k passages using ColBERT
    # results = (passage_ids, ranks, scores)
    results = searcher.search(question, k=k)

    # Softmax-normalize the retrieval scores to get pseudo-probabilities
    passage_probs = softmax(results[2])

    # Get the actual passage text strings
    passages = [searcher.collection[pid] for pid in results[0]]

    return passages, passage_probs


def answer_scoring(passages, passage_probs, prompts, gen_func=run_eleuther):
    """
    Score each (passage, answer) pair and return sorted results.

    For each passage/prompt pair:
        score = product(answer_token_probs) x passage_prob

    Parameters
    ----------
    passages : list of str
        Retrieved passages.
    passage_probs : list of float or np.array
        Softmax-normalized retrieval scores, one per passage.
    prompts : list of str
        Pre-built prompts, one per passage (same order as passages).
    gen_func : callable
        Either run_eleuther or run_gpt3.

    Returns
    -------
    list of [score, gen_dict]
        Sorted descending by score — best answer first.
        gen_dict is the dict returned by gen_func for that example.
    """
    data = []

    for passage, passage_prob, prompt in zip(passages, passage_probs, prompts):
        # Run the LM on this single prompt (must be a list of 1)
        gen = gen_func([prompt])[0]

        # Score = product of answer token probabilities x passage probability
        answer_prob = np.prod(gen["generated_answer_probs"])
        score = answer_prob * passage_prob

        data.append([score, gen])

    # Sort descending — highest scoring answer first
    data = sorted(data, key=lambda x: -x[0])

    return data


# ============================================================
# HW QUESTION 4: Original System [3 points]
# ============================================================
# SYSTEM DESCRIPTION:
#
# Key improvements over the baselines:
#
# 1. BM25-guided few-shot example selection:
#    Instead of random sampling from squad_train, use BM25 to find
#    the most semantically relevant training examples for each
#    (question, retrieved passage) pair. This makes the few-shot
#    examples more informative for the specific question being asked.
#
# 2. Multi-passage retrieval + answer scoring:
#    Retrieve top-k passages (k=2) instead of just 1. Score all
#    (passage, answer) pairs jointly and pick the best-scoring answer.
#
# 3. Length-normalized scoring:
#    Use geometric mean of token probabilities (prob^(1/n)) instead
#    of raw product. This prevents longer answers from being
#    unfairly penalized regardless of per-token quality.
#
# 4. Beam search decoding:
#    Use num_beams=4 with greedy decoding (do_sample=False) instead
#    of temperature sampling. Grid search showed num_beams=4 gives
#    the best macro_f1 on dev_exs.
#
# Peak score: 0.143387 macro_f1 on dev_exs


from rank_bm25 import BM25Okapi


class SquadBM25:
    """
    BM25 index over SQuAD training examples for few-shot example retrieval.

    Builds an index over (context + question) strings so that given
    a new (passage, question) query, the most relevant training
    examples can be retrieved for few-shot prompting.
    """

    def __init__(self, squad_data, tokenizer):
        """
        Parameters
        ----------
        squad_data : list of SquadExample
        tokenizer : callable
            A function mapping str -> list of str tokens.
            Use get_tokens (already defined in notebook).
        """
        self.squad_data = squad_data
        self.tokenizer  = tokenizer
        self._build_index()

    def _build_index(self):
        """Build BM25 index from context + question strings."""
        corpus = [
            ex.context + " " + ex.question
            for ex in self.squad_data
        ]
        tokenized_corpus = [self.tokenizer(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_relevant_examples(self, query, k):
        """
        Retrieve the k most relevant SQuAD training examples for a query.

        Parameters
        ----------
        query : str
            Combined passage + question string.
        k : int
            Number of examples to return.

        Returns
        -------
        list of SquadExample
        """
        tokenized_query = self.tokenizer(query)
        scores  = self.bm25.get_scores(tokenized_query)
        # argpartition is faster than full sort for top-k
        top_indices = np.argpartition(scores, -k)[-k:]
        return [self.squad_data[idx] for idx in top_indices]


def build_few_shot_open_qa_prompt_bm25(question, passage, squad_index, k=2, joiner="\n\n"):
    """
    Build a few-shot OpenQA prompt using BM25-selected training examples.

    Unlike the random-sampling version, this selects the most relevant
    SQuAD training examples based on similarity to the (passage, question).

    Parameters
    ----------
    question : str
        The target question.
    passage : str
        ColBERT-retrieved passage formatted as "Title | Background text".
    squad_index : SquadBM25
        Pre-built BM25 index over squad_train.
    k : int
        Number of few-shot examples to include.
    joiner : str
        String used to join prompt segments.

    Returns
    -------
    str
        The full prompt string ending with "A:".
    """
    # Use BM25 to find the most relevant few-shot examples
    # Query = retrieved passage + target question (combined context)
    query     = passage + " " + question
    train_exs = squad_index.get_relevant_examples(query, k)

    segs = []

    # Add BM25-selected training examples as Title/Background/Q/A blocks
    for ex in train_exs:
        segs.append(f"Title: {ex.title}")
        segs.append(f"Background: {ex.context}")
        segs.append(f"Q: {ex.question}")
        segs.append(f"A: {ex.answers[0]}")

    # Parse the retrieved passage
    title, background = passage.split(" | ", 1)

    # Add the target question with open "A:"
    segs.append(f"Title: {title}")
    segs.append(f"Background: {background}")
    segs.append(f"Q: {question}")
    segs.append("A:")

    return joiner.join(segs)


def run_eleuther_v2(prompts, **generate_kwargs):
    """
    Variant of run_eleuther that passes all generation kwargs through.

    This allows using either temperature-based sampling or beam search
    by passing the appropriate kwargs (e.g. num_beams=4, do_sample=False).

    Parameters
    ----------
    prompts : list of str
    **generate_kwargs : passed directly to model.generate()

    Returns
    -------
    list of dicts (same format as run_eleuther)
    """
    prompt_ids = eleuther_tokenizer(
        prompts, return_tensors="pt", padding=True
    ).input_ids.to(device)

    with torch.inference_mode():
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
            model_output = eleuther_model.generate(
                prompt_ids,
                max_new_tokens=16,
                num_return_sequences=1,
                pad_token_id=eleuther_tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                **generate_kwargs
            )

    gen_ids   = model_output.sequences[:, prompt_ids.shape[-1]:]
    gen_probs = torch.stack(model_output.scores, dim=1).softmax(-1)
    gen_probs = torch.gather(gen_probs, 2, gen_ids[:, :, None]).squeeze(-1)

    gen_texts = eleuther_tokenizer.batch_decode(
        model_output.sequences, skip_special_tokens=True
    )

    data = []
    for prompt, gen_id, gen_text, gen_prob in zip(prompts, gen_ids, gen_texts, gen_probs):
        gen_tokens     = eleuther_tokenizer.convert_ids_to_tokens(gen_id)
        generated_text = gen_text[len(prompt):]
        gen_prob       = [float(x) for x in gen_prob.cpu().numpy()]
        ans_indices    = _find_generated_answer(gen_tokens, newline="Ċ")
        answer_tokens  = [gen_tokens[i] for i in ans_indices]
        answer_probs   = [gen_prob[i]   for i in ans_indices]
        answer         = "".join(answer_tokens).replace("Ġ", " ").replace("Ċ", "\n")

        data.append({
            "prompt":                 prompt,
            "generated_text":         generated_text,
            "generated_tokens":       gen_tokens,
            "generated_probs":        gen_prob,
            "generated_answer":       answer,
            "generated_answer_probs": answer_probs,
            "generated_answer_tokens": answer_tokens
        })

    return data


def answer_scoring_normalized(passages, passage_probs, prompts, gen_func,
                               num_beams=None, temperature=None):
    """
    Score (passage, answer) pairs with length normalization.

    Scores answers using geometric mean of token probabilities
    (prob^(1/n_tokens)) instead of raw product, preventing long
    answers from being unfairly penalized.

    Parameters
    ----------
    passages : list of str
    passage_probs : list of float
    prompts : list of str
    gen_func : callable (use run_eleuther_v2)
    num_beams : int or None
        If set, uses beam search (do_sample=False).
    temperature : float or None
        If set, uses temperature sampling (do_sample=True).
        Exactly one of num_beams or temperature must be set.

    Returns
    -------
    list of [score, gen_dict], sorted descending by score.
    """
    if not np.logical_xor(temperature is None, num_beams is None):
        raise ValueError("Exactly one of 'num_beams' or 'temperature' must be set.")

    data = []

    for passage, passage_prob, prompt in zip(passages, passage_probs, prompts):
        # Generate answer using the specified decoding strategy
        if temperature is not None:
            gen = gen_func([prompt], temperature=temperature, do_sample=True, top_p=0.95)[0]
        else:
            gen = gen_func([prompt], num_beams=num_beams, do_sample=False)[0]

        answer_probs = gen["generated_answer_probs"]

        # Length-normalized score: geometric mean x passage probability
        # Avoids penalizing longer (potentially better) answers
        n           = len(answer_probs) if len(answer_probs) > 0 else 1
        norm_score  = (np.prod(answer_probs) ** (1.0 / n)) * passage_prob

        data.append([norm_score, gen])

    return sorted(data, key=lambda x: -x[0])


def original_system(question, squad_index,
                    gen_func=run_eleuther_v2,
                    temperature=None,
                    num_beams=4,
                    k_passages=2,
                    k_train_qa=2,
                    joiner="\n\n"):
    """
    Full original few-shot OpenQA system.

    Given a question:
    1. Retrieves k_passages passages using ColBERT.
    2. For each passage, finds k_train_qa relevant SQuAD examples via BM25.
    3. Builds a few-shot prompt per passage.
    4. Scores all (passage, answer) pairs using length-normalized scoring.
    5. Returns the highest-scoring answer.

    Parameters
    ----------
    question : str
    squad_index : SquadBM25
        Pre-built BM25 index over squad_train.
    gen_func : callable
        Use run_eleuther_v2.
    temperature : float or None
    num_beams : int or None
        Default is 4 (best from grid search).
    k_passages : int
        Number of passages to retrieve and score (default: 2).
    k_train_qa : int
        Number of BM25 few-shot examples per prompt (default: 2).
    joiner : str

    Returns
    -------
    dict
        The gen_dict for the highest-scoring (passage, answer) pair.
    """
    # Step 1: retrieve passages + softmax passage probabilities
    passages, passage_probs = get_passages_with_scores(question, k=k_passages)

    # Step 2: build one prompt per passage using BM25 few-shot examples
    prompts = [
        build_few_shot_open_qa_prompt_bm25(
            question, passage, squad_index, k=k_train_qa, joiner=joiner
        )
        for passage in passages
    ]

    # Step 3: score all (passage, answer) pairs and return the best
    scored = answer_scoring_normalized(
        passages, passage_probs, prompts,
        gen_func=gen_func,
        num_beams=num_beams,
        temperature=temperature
    )

    return scored[0][1]  # return the gen_dict of the best answer


# ---- Hyperparameter tuning helpers ----

def tune_temperature(examples, squad_index, temperature_list):
    """
    Grid search over temperature values. Prints macro_f1 for each.

    Returns
    -------
    best_temperature : float
    scores : dict mapping temperature -> experiment result dict
    """
    scores = {}
    for temp in temperature_list:
        prompts, gens = [], []
        for ex in tqdm(examples, desc=f"temperature={temp}"):
            gen = original_system(
                ex.question, squad_index,
                temperature=temp, num_beams=None
            )
            prompts.append(" ")
            gens.append(gen)
        result = evaluate(examples, prompts, gens)
        scores[temp] = result
        print(f"temperature={temp}  macro_f1={result['macro_f1']:.4f}")

    best_temp = max(scores, key=lambda t: scores[t]["macro_f1"])
    return best_temp, scores


def tune_num_beams(examples, squad_index, num_beams_list):
    """
    Grid search over num_beams values. Prints macro_f1 for each.

    Returns
    -------
    best_num_beams : int
    scores : dict mapping num_beams -> experiment result dict
    """
    scores = {}
    for nb in num_beams_list:
        prompts, gens = [], []
        for ex in tqdm(examples, desc=f"num_beams={nb}"):
            gen = original_system(
                ex.question, squad_index,
                temperature=None, num_beams=nb
            )
            prompts.append(" ")
            gens.append(gen)
        result = evaluate(examples, prompts, gens)
        scores[nb] = result
        print(f"num_beams={nb}  macro_f1={result['macro_f1']:.4f}")

    best_nb = max(scores, key=lambda nb: scores[nb]["macro_f1"])
    return best_nb, scores


# ---- Final system with best hyperparameters ----

def final_system(question, squad_index):
    """
    Final system using best hyperparameters from grid search:
    num_beams=4, k_passages=2, k_train_qa=2.
    """
    return original_system(
        question,
        squad_index,
        gen_func=run_eleuther_v2,
        temperature=None,
        num_beams=4,
        k_passages=2,
        k_train_qa=2
    )


# ============================================================
# BAKEOFF SUBMISSION
# ============================================================

def create_bakeoff_submission(squad_index,
                               filename="cs224u-openqa-bakeoff-entry.json"):
    """
    Generate bakeoff predictions and save to JSON.

    Parameters
    ----------
    squad_index : SquadBM25
        Pre-built BM25 index over squad_train.
    filename : str
        Output file path.
    """
    import json

    test_file = os.path.join("data", "openqa", "cs224u-openqa-test-unlabeled.txt")
    with open(test_file) as f:
        questions = f.read().splitlines()

    gens = {}
    for question in tqdm(questions):
        gens[question] = final_system(question, squad_index)

    # Sanity checks
    assert all(q in gens for q in questions), "Missing questions in output!"
    assert all(
        isinstance(d, dict) and "generated_answer" in d
        for d in gens.values()
    ), "Output dicts missing 'generated_answer' key!"

    with open(filename, "wt") as f:
        json.dump(gens, f, indent=4)

    print(f"Saved {len(gens)} predictions to {filename}")


# ============================================================
# HOW TO RUN (in the notebook, after setup cells):
# ============================================================
#
# # Build BM25 index once:
# squad_index = SquadBM25(squad_train, get_tokens)
#
# # HW1:
# few_shot_no_context_results = evaluate_few_shot_no_context(dev_exs, squad_train)
# print(few_shot_no_context_results['macro_f1'])
#
# # HW2:
# few_shot_openqa_results = evaluate_few_shot_open_qa(dev_exs, squad_train)
# print(few_shot_openqa_results['macro_f1'])
#
# # HW3:
# passages, probs = get_passages_with_scores("How long is Moby Dick?")
# prompts = [build_zero_shot_openqa_prompt("How long is Moby Dick?", p) for p in passages]
# scored  = answer_scoring(passages, probs, prompts)
# print(scored[0][1]['generated_answer'])
#
# # HW4 (original system):
# result = original_system("How long is Moby Dick?", squad_index)
# print(result['generated_answer'])
#
# # Bakeoff:
# create_bakeoff_submission(squad_index)
