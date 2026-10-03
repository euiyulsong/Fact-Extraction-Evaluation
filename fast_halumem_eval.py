"""
fast_halumem_eval.py

Fast HaluMem binary judge evaluation.

Dataset:
    HaluMem-Medium

Sample:
    50 positive
        memory_source in {"primary", "secondary"}

    50 negative
        memory_source == "interference"

Total:
    100 examples

Judge:
    gpt-6-luna
    reasoning_effort="none"
    max_completion_tokens=16

Output:
    1 = valid grounded fact
    0 = invalid / unsupported / hallucinated / interference

Install:
    pip install -U requests huggingface_hub pandas tqdm

Run:
    export OPENAI_API_KEY="sk-..."
    python3 fast_halumem_eval.py

Optional:
    JUDGE_MODEL=gpt-6-luna python3 fast_halumem_eval.py
"""

import os
import json
import random
import time
import threading

import requests
import pandas as pd

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

N_POSITIVE = 50
N_NEGATIVE = 50

assert N_POSITIVE + N_NEGATIVE == 100

MODEL = os.getenv(
    "JUDGE_MODEL",
    "gpt-6-luna",
)

MAX_COMPLETION_TOKENS = 16

MAX_WORKERS = 8

MAX_RETRIES = 3

REQUEST_TIMEOUT = 60


# ============================================================
# OPENAI
# ============================================================

OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY"
)

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )

OPENAI_URL = (
    "https://api.openai.com/v1/chat/completions"
)


# ============================================================
# HALUMEM
# ============================================================

HF_REPO = "IAAR-Shanghai/HaluMem"

HF_FILENAME = "HaluMem-Medium.jsonl"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_CSV = "halumem_fast_100.csv"

OUTPUT_METRICS = "halumem_fast_metrics.json"

OUTPUT_ERRORS = "halumem_fast_errors.jsonl"


# ============================================================
# JUDGE PROMPT
# ============================================================

JUDGE_PROMPT = """
You are evaluating whether a candidate is a VALID long-term memory
derived from a user-assistant conversation.

Return exactly one character:

1 = VALID MEMORY
0 = INVALID / FALSE MEMORY

A candidate is VALID if it is a faithful semantic representation
of information established by the conversation.

IMPORTANT:
The candidate does NOT need to repeat the user's exact words.

Return 1 for:
- faithful paraphrases
- faithful summaries
- information clearly implied by the user's statements
- reasonable high-level abstractions of the user's expressed
  preferences, goals, plans, experiences, state, or relationships
- conclusions supported by combining multiple dialogue turns
- rewriting "I/my/me" into the user's name
- summaries that preserve the overall meaning even if the exact
  wording does not appear in the conversation

Do NOT require every phrase of the candidate to appear literally.

Example:

USER:
"I've been reading a lot more books about AI lately because
I want to move my career toward machine learning."

CANDIDATE:
"The user has shifted their reading preferences toward AI and
machine-learning literature to support their career goals."

Answer:
1


Example:

USER:
"I've been really stressed at work lately."
USER:
"I'm thinking I may need another job because this role is
affecting my health."

CANDIDATE:
"The user is considering a career change because their current
work is negatively affecting their health."

Answer:
1


The main reason to return 0 is FALSE MEMORY.

Return 0 when:
- the candidate contradicts the conversation
- the candidate introduces a genuinely new fact not supported
  or reasonably implied by the conversation
- an important entity, event, relationship, date, number,
  preference, or polarity is wrong
- the candidate is based only on an assistant assumption,
  suggestion, or invented statement that the user did not
  confirm or independently support

INTERFERENCE EXAMPLE:

USER:
"I've been trying to exercise more."

ASSISTANT:
"Since you love rock climbing, perhaps you could join a climbing gym."

The user never said they love rock climbing.

CANDIDATE:
"The user loves rock climbing."

Answer:
0


IMPORTANT CALIBRATION:

Do NOT reject a candidate merely because it:
- is more abstract than the original utterance
- summarizes several turns
- describes a motivation inferred directly from what the user said
- uses a third-person description
- uses the user's name
- combines mutually consistent information across turns

Ask yourself:

"Is this a faithful memory of what we learn about the user from
the conversation?"

If yes, output 1.

Only output 0 when there is a concrete factual mismatch,
unsupported invention, contradiction, or assistant-only interference.

Use no outside knowledge.

Output ONLY:
0
or
1
"""
# ============================================================
# DOWNLOAD
# ============================================================

def download_dataset():

    print(
        "Downloading / loading HaluMem-Medium..."
    )

    return hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILENAME,
        repo_type="dataset",
    )


# ============================================================
# LOAD
# ============================================================

def load_dataset(path):

    data = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            data.append(
                json.loads(line)
            )

    return data


# ============================================================
# ROLE NORMALIZATION
# ============================================================

def normalize_role(role):

    role = str(
        role
    ).strip().lower()

    if role in {
        "human",
        "person",
    }:
        return "user"

    if role in {
        "ai",
        "bot",
    }:
        return "assistant"

    return role


# ============================================================
# FORMAT DIALOGUE
# ============================================================

def format_dialogue(session):

    lines = []

    dialogue = session.get(
        "dialogue",
        []
    )

    for i, turn in enumerate(
        dialogue
    ):

        role = normalize_role(
            turn.get(
                "role",
                "unknown",
            )
        )

        content = str(
            turn.get(
                "content",
                "",
            )
        ).strip()

        if not content:
            continue

        turn_id = turn.get(
            "dialogue_turn",
            i,
        )

        timestamp = turn.get(
            "timestamp",
            "",
        )

        header = (
            f"[TURN={turn_id}]"
            f"[ROLE={role.upper()}]"
        )

        if timestamp:
            header += (
                f"[TIME={timestamp}]"
            )

        lines.append(
            f"{header}\n{content}"
        )

    return "\n\n".join(
        lines
    )


# ============================================================
# BUILD POSITIVE / NEGATIVE POOL
# ============================================================

def build_candidate_pool(dataset):

    positive = []
    negative = []

    for user_idx, user in enumerate(
        dataset
    ):

        uuid = user.get(
            "uuid",
            f"user_{user_idx}",
        )

        sessions = user.get(
            "sessions",
            [],
        )

        for session_idx, session in enumerate(
            sessions
        ):

            dialogue = format_dialogue(
                session
            )

            if not dialogue:
                continue

            memory_points = session.get(
                "memory_points",
                [],
            )

            for memory in memory_points:

                fact = str(
                    memory.get(
                        "memory_content",
                        "",
                    )
                ).strip()

                if not fact:
                    continue

                source = str(
                    memory.get(
                        "memory_source",
                        "",
                    )
                ).strip().lower()

                row = {
                    "uuid": uuid,
                    "session_idx": session_idx,
                    "memory_index": memory.get(
                        "index",
                        "",
                    ),
                    "memory_source": source,
                    "memory_type": memory.get(
                        "memory_type",
                        "",
                    ),
                    "is_update": memory.get(
                        "is_update",
                        "",
                    ),
                    "importance": memory.get(
                        "importance",
                        "",
                    ),
                    "fact": fact,
                    "dialogue": dialogue,
                }

                # Positive / reference memory
                if source in {
                    "primary", "secondary"
                }:

                    row["expected"] = 1

                    positive.append(
                        row
                    )

                # Negative / interference memory
                elif source == "interference":

                    row["expected"] = 0

                    negative.append(
                        row
                    )

    return (
        positive,
        negative,
    )


# ============================================================
# SAMPLE 100
# ============================================================

def sample_examples(
    positive,
    negative,
):

    if len(positive) < N_POSITIVE:

        raise RuntimeError(
            f"Need {N_POSITIVE} positives, "
            f"found {len(positive)}"
        )

    if len(negative) < N_NEGATIVE:

        raise RuntimeError(
            f"Need {N_NEGATIVE} negatives, "
            f"found {len(negative)}"
        )

    rng = random.Random(
        SEED
    )

    pos = rng.sample(
        positive,
        N_POSITIVE,
    )

    neg = rng.sample(
        negative,
        N_NEGATIVE,
    )

    examples = pos + neg

    rng.shuffle(
        examples
    )

    return examples


# ============================================================
# PARSE 0 / 1
# ============================================================

def parse_binary_output(output):

    if output is None:
        return None

    output = str(
        output
    ).strip()

    if output == "1":
        return 1

    if output == "0":
        return 0

    if output.startswith("1"):
        return 1

    if output.startswith("0"):
        return 0

    return None


# ============================================================
# JUDGE
# ============================================================

def judge_fact(
    dialogue,
    fact,
):

    headers = {
        "Authorization":
            f"Bearer {OPENAI_API_KEY}",

        "Content-Type":
            "application/json",
    }

    user_message = (
        "CONVERSATION:\n"
        "====================\n"
        f"{dialogue}\n"
        "====================\n\n"
        "CANDIDATE FACT:\n"
        f"{fact}\n\n"
        "OUTPUT ONLY 0 OR 1."
    )

    payload = {

        "model":
            MODEL,

        "messages": [
            {
                "role":
                    "developer",

                "content":
                    JUDGE_PROMPT,
            },
            {
                "role":
                    "user",

                "content":
                    user_message,
            },
        ],

        # ----------------------------------------------------
        # IMPORTANT FOR GPT-6 LUNA
        #
        # Without this Luna defaults to reasoning and can spend
        # all 16 completion tokens on hidden reasoning.
        # ----------------------------------------------------

        "reasoning_effort":
            "none",

        "max_completion_tokens":
            MAX_COMPLETION_TOKENS,

        # IMPORTANT:
        # NO temperature.
        #
        # Luna rejects temperature=0.
        # ----------------------------------------------------

        "store":
            False,
    }

    last_error = None

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            response = requests.post(
                OPENAI_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if not response.ok:

                raise RuntimeError(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:3000]}"
                )

            data = response.json()

            choices = data.get(
                "choices",
                [],
            )

            if not choices:

                raise RuntimeError(
                    "No choices returned:\n"
                    f"{json.dumps(data, ensure_ascii=False)[:3000]}"
                )

            choice = choices[0]

            message = choice.get(
                "message",
                {},
            )

            output = message.get(
                "content",
                "",
            )

            if output is None:
                output = ""

            output = str(
                output
            ).strip()

            label = parse_binary_output(
                output
            )

            if label is None:

                usage = data.get(
                    "usage",
                    {},
                )

                details = usage.get(
                    "completion_tokens_details",
                    {},
                )

                raise ValueError(
                    "\nJudge did not return 0/1."
                    f"\nmodel={MODEL}"
                    f"\nfinish_reason="
                    f"{choice.get('finish_reason')}"
                    f"\nraw_output={output!r}"
                    f"\ncompletion_tokens="
                    f"{usage.get('completion_tokens')}"
                    f"\nreasoning_tokens="
                    f"{details.get('reasoning_tokens')}"
                    "\nraw_response="
                    f"{json.dumps(data, ensure_ascii=False)[:4000]}"
                )

            usage = data.get(
                "usage",
                {},
            )

            details = usage.get(
                "completion_tokens_details",
                {},
            )

            return {
                "label":
                    label,

                "raw_output":
                    output,

                "finish_reason":
                    choice.get(
                        "finish_reason",
                        "",
                    ),

                "input_tokens":
                    usage.get(
                        "prompt_tokens",
                        0,
                    )
                    or 0,

                "output_tokens":
                    usage.get(
                        "completion_tokens",
                        0,
                    )
                    or 0,

                "reasoning_tokens":
                    details.get(
                        "reasoning_tokens",
                        0,
                    )
                    or 0,

                "total_tokens":
                    usage.get(
                        "total_tokens",
                        0,
                    )
                    or 0,
            }

        except Exception as e:

            last_error = e

            if (
                attempt
                == MAX_RETRIES - 1
            ):
                break

            time.sleep(
                2 ** attempt
            )

    raise last_error


# ============================================================
# EVALUATE ONE
# ============================================================

def evaluate_one(
    idx,
    row,
):

    start = time.time()

    result = judge_fact(
        dialogue=row["dialogue"],
        fact=row["fact"],
    )

    latency = (
        time.time()
        - start
    )

    expected = int(
        row["expected"]
    )

    judge = int(
        result["label"]
    )

    return {

        "id":
            idx,

        "uuid":
            row["uuid"],

        "session_idx":
            row["session_idx"],

        "memory_index":
            row["memory_index"],

        "memory_source":
            row["memory_source"],

        "memory_type":
            row["memory_type"],

        "fact":
            row["fact"],

        "expected":
            expected,

        "judge":
            judge,

        "correct":
            int(
                expected == judge
            ),

        "raw_output":
            result[
                "raw_output"
            ],

        "input_tokens":
            result[
                "input_tokens"
            ],

        "output_tokens":
            result[
                "output_tokens"
            ],

        "reasoning_tokens":
            result[
                "reasoning_tokens"
            ],

        "total_tokens":
            result[
                "total_tokens"
            ],

        "latency_sec":
            round(
                latency,
                3,
            ),
    }


# ============================================================
# ERROR OUTPUT
# ============================================================

ERROR_LOCK = threading.Lock()


def save_error(
    idx,
    error,
):

    with ERROR_LOCK:

        with open(
            OUTPUT_ERRORS,
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    {
                        "id": idx,
                        "error": str(error),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# EVALUATE ALL
# ============================================================

def evaluate_all(examples):

    results = []

    if os.path.exists(
        OUTPUT_ERRORS
    ):

        os.remove(
            OUTPUT_ERRORS
        )

    # --------------------------------------------------------
    # Smoke test first
    # --------------------------------------------------------

    print()
    print(
        "Running 1-example API smoke test..."
    )

    first = evaluate_one(
        0,
        examples[0],
    )

    results.append(
        first
    )

    print(
        "Smoke test OK"
    )

    print(
        f"  expected         : "
        f"{first['expected']}"
    )

    print(
        f"  judge            : "
        f"{first['judge']}"
    )

    print(
        f"  raw output       : "
        f"{first['raw_output']!r}"
    )

    print(
        f"  output tokens    : "
        f"{first['output_tokens']}"
    )

    print(
        f"  reasoning tokens : "
        f"{first['reasoning_tokens']}"
    )

    print(
        f"  latency          : "
        f"{first['latency_sec']:.2f}s"
    )

    print()

    # --------------------------------------------------------
    # Remaining 99
    # --------------------------------------------------------

    executor = ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    )

    futures = {}

    for idx in range(
        1,
        len(examples),
    ):

        f = executor.submit(
            evaluate_one,
            idx,
            examples[idx],
        )

        futures[f] = idx

    try:

        for future in tqdm(
            as_completed(
                futures
            ),
            total=len(
                futures
            ),
            desc="GPT judge",
        ):

            idx = futures[
                future
            ]

            try:

                results.append(
                    future.result()
                )

            except Exception as e:

                print(
                    f"\nERROR example "
                    f"{idx}: {e}"
                )

                save_error(
                    idx,
                    e,
                )

    except KeyboardInterrupt:

        print(
            "\nInterrupted. "
            "Cancelling pending calls..."
        )

        for future in futures:
            future.cancel()

        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

        raise

    else:

        executor.shutdown(
            wait=True
        )

    results.sort(
        key=lambda x: x["id"]
    )

    return results


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(results):

    tp = sum(
        r["expected"] == 1
        and r["judge"] == 1
        for r in results
    )

    tn = sum(
        r["expected"] == 0
        and r["judge"] == 0
        for r in results
    )

    fp = sum(
        r["expected"] == 0
        and r["judge"] == 1
        for r in results
    )

    fn = sum(
        r["expected"] == 1
        and r["judge"] == 0
        for r in results
    )

    n = len(
        results
    )

    accuracy = (
        (tp + tn) / n
        if n
        else 0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        )
        else 0
    )

    negative_rejection_rate = (
        tn / (tn + fp)
        if tn + fp
        else 0
    )

    false_accept_rate = (
        fp / (fp + tn)
        if fp + tn
        else 0
    )

    false_reject_rate = (
        fn / (fn + tp)
        if fn + tp
        else 0
    )

    return {

        "n":
            n,

        "tp":
            tp,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "negative_rejection_rate":
            negative_rejection_rate,

        "false_accept_rate":
            false_accept_rate,

        "false_reject_rate":
            false_reject_rate,

        "total_input_tokens":
            sum(
                r["input_tokens"]
                for r in results
            ),

        "total_output_tokens":
            sum(
                r["output_tokens"]
                for r in results
            ),

        "total_reasoning_tokens":
            sum(
                r["reasoning_tokens"]
                for r in results
            ),

        "mean_latency_sec":
            (
                sum(
                    r["latency_sec"]
                    for r in results
                )
                / n
                if n
                else 0
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(
        SEED
    )

    print(
        f"Judge model: {MODEL}"
    )

    print(
        "reasoning_effort: none"
    )

    print(
        f"max_completion_tokens: "
        f"{MAX_COMPLETION_TOKENS}"
    )

    print(
        f"workers: "
        f"{MAX_WORKERS}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset_path = (
        download_dataset()
    )

    dataset = load_dataset(
        dataset_path
    )

    print(
        f"Users loaded: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # Pools
    # --------------------------------------------------------

    positive, negative = (
        build_candidate_pool(
            dataset
        )
    )

    print(
        f"Positive pool: "
        f"{len(positive)}"
    )

    print(
        f"Negative pool: "
        f"{len(negative)}"
    )

    # --------------------------------------------------------
    # Sample 100
    # --------------------------------------------------------

    examples = sample_examples(
        positive,
        negative,
    )

    print(
        f"Evaluation set: "
        f"{len(examples)}"
    )

    print(
        f"  positive: "
        f"{sum(x['expected'] == 1 for x in examples)}"
    )

    print(
        f"  negative: "
        f"{sum(x['expected'] == 0 for x in examples)}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = evaluate_all(
        examples
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    metrics = calculate_metrics(
        results
    )

    with open(
        OUTPUT_METRICS,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "HaluMem Fast Binary Judge"
    )

    print(
        "=" * 60
    )

    print(
        f"N                        : "
        f"{metrics['n']}"
    )

    print(
        f"TP                       : "
        f"{metrics['tp']}"
    )

    print(
        f"TN                       : "
        f"{metrics['tn']}"
    )

    print(
        f"FP                       : "
        f"{metrics['fp']}"
    )

    print(
        f"FN                       : "
        f"{metrics['fn']}"
    )

    print()

    print(
        f"Accuracy                 : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision                : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall                   : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1                       : "
        f"{metrics['f1']:.4f}"
    )

    print()

    print(
        f"Negative rejection rate  : "
        f"{metrics['negative_rejection_rate']:.4f}"
    )

    print(
        f"False accept rate        : "
        f"{metrics['false_accept_rate']:.4f}"
    )

    print(
        f"False reject rate        : "
        f"{metrics['false_reject_rate']:.4f}"
    )

    print()

    print(
        f"Reasoning tokens         : "
        f"{metrics['total_reasoning_tokens']}"
    )

    print(
        f"Mean latency             : "
        f"{metrics['mean_latency_sec']:.3f}s"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Misclassified
    # --------------------------------------------------------

    bad = df[
        df["correct"] == 0
    ]

    if len(bad):

        print()
        print(
            "MISCLASSIFIED"
        )

        print(
            bad[
                [
                    "id",
                    "memory_source",
                    "expected",
                    "judge",
                    "fact",
                ]
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    print()
    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print(
        f"Saved: {OUTPUT_METRICS}"
    )


if __name__ == "__main__":
    main()
