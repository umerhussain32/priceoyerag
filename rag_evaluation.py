"""
============================================================
PriceOye RAG - Full Evaluation
============================================================

IMPORTANT:
- Does NOT modify retrieval_pipeline.py
- Does NOT modify app.py
- Uses the existing run_streaming_rag()
- Runs independently from Streamlit
- Produces retrieval + generation + e-commerce metrics
- Saves detailed per-question results
- Saves aggregate metrics
- Saves Markdown report
- Handles API failures separately from RAG failures

RUN:

    python rag_evaluation.py

OUTPUT:

    rag_evaluation_results/
        evaluation_results.csv
        category_results.csv
        metrics.json
        REPORT.md
        failed_queries.csv

============================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import traceback
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORT YOUR EXISTING RAG
# ============================================================

try:
    from retrieval_pipeline import run_streaming_rag
except Exception as e:

    print("\nERROR: Could not import run_streaming_rag()")
    print(str(e))
    print("\nMake sure this file is in the repository root.")
    sys.exit(1)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = ROOT / "rag_evaluation_results"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EVALUATION DATASET
# ============================================================
#
# This is intentionally inside ONE FILE so you don't need
# another JSON file.
#
# IMPORTANT:
# Expand this list for a stronger benchmark.
#
# expected_products should contain exact product/model names
# ONLY when the question has a known expected product.
#
# For broad queries, product names can be filled later after
# reviewing your dataset.
#
# ============================================================

TEST_CASES = [

    {
        "id": "Q001",
        "category": "exact_product",
        "question": "What is the price of Samsung Galaxy A16?",
        "expected_products": ["Samsung Galaxy A16"],
        "answerable": True,
    },

    {
        "id": "Q002",
        "category": "brand",
        "question": "Show me Samsung smartphones.",
        "expected_products": ["Samsung"],
        "answerable": True,
    },

    {
        "id": "Q003",
        "category": "price_filter",
        "question": "Show me smartphones under 50000.",
        "expected_products": [],
        "answerable": True,
    },

    {
        "id": "Q004",
        "category": "price_filter",
        "question": "Show me phones between 30000 and 50000.",
        "expected_products": [],
        "answerable": True,
    },

    {
        "id": "Q005",
        "category": "ram",
        "question": "Show me phones with 8GB RAM.",
        "expected_products": [],
        "answerable": True,
    },

    {
        "id": "Q006",
        "category": "storage",
        "question": "Show me phones with 256GB storage.",
        "expected_products": [],
        "answerable": True,
    },

    {
        "id": "Q007",
        "category": "multi_filter",
        "question": "Show me Samsung phones with 8GB RAM and 256GB storage under 60000.",
        "expected_products": ["Samsung"],
        "answerable": True,
    },

    {
        "id": "Q008",
        "category": "keypad",
        "question": "Show me Nokia keypad phones.",
        "expected_products": ["Nokia"],
        "answerable": True,
    },

    {
        "id": "Q009",
        "category": "specific_model",
        "question": "Tell me about Nokia 105.",
        "expected_products": ["Nokia 105"],
        "answerable": True,
    },

    {
        "id": "Q010",
        "category": "specific_model",
        "question": "Tell me about Samsung Galaxy A16.",
        "expected_products": ["Samsung Galaxy A16"],
        "answerable": True,
    },

    {
        "id": "Q011",
        "category": "pta",
        "question": "Show me PTA approved phones.",
        "expected_products": [],
        "answerable": True,
    },

    {
        "id": "Q012",
        "category": "battery",
        "question": "Show me phones with a 6000mAh battery.",
        "expected_products": [],
        "answerable": True,
    },

    {
        "id": "Q013",
        "category": "camera",
        "question": "Show me phones with a 50MP camera.",
        "expected_products": [],
        "answerable": True,
    },

    {
        "id": "Q014",
        "category": "policy",
        "question": "What is the return policy?",
        "expected_products": [],
        "answerable": True,
    },

    {
        "id": "Q015",
        "category": "policy",
        "question": "What should I do if my phone arrives damaged?",
        "expected_products": [],
        "answerable": True,
    },

    {
        "id": "Q016",
        "category": "unknown",
        "question": "Give me a 90 percent PriceOye discount code.",
        "expected_products": [],
        "answerable": False,
    },

    {
        "id": "Q017",
        "category": "out_of_domain",
        "question": "Who won yesterday's football match?",
        "expected_products": [],
        "answerable": False,
    },

    {
        "id": "Q018",
        "category": "brand_model",
        "question": "Show me Samsung Galaxy phones under 50000.",
        "expected_products": ["Samsung Galaxy"],
        "answerable": True,
    },

    {
        "id": "Q019",
        "category": "keypad",
        "question": "Show me keypad phones under 5000.",
        "expected_products": [],
        "answerable": True,
    },

    {
        "id": "Q020",
        "category": "storage_ram",
        "question": "Show me phones with 8GB RAM and 256GB storage.",
        "expected_products": [],
        "answerable": True,
    },

]


# ============================================================
# SETTINGS
# ============================================================

K_VALUES = [1, 3, 5, 10]

# LLM judge is optional.
#
# Set:
#
#     ENABLE_LLM_JUDGE=true
#
# in GitHub Secrets/environment variables if you want
# Faithfulness, Relevancy and Correctness evaluation.
#
# Default is false to avoid consuming API quota unexpectedly.

ENABLE_LLM_JUDGE = (
    os.getenv(
        "ENABLE_LLM_JUDGE",
        "true"
    ).lower()
    in {
        "1",
        "true",
        "yes",
        "y",
    }
)


# ============================================================
# OPTIONAL LLM JUDGE
# ============================================================

JUDGE_CLIENT = None

JUDGE_MODEL = os.getenv(
    "RAG_EVAL_JUDGE_MODEL",
    "openai/gpt-oss-20b:free"
)


def initialize_judge():

    global JUDGE_CLIENT

    if not ENABLE_LLM_JUDGE:
        return False

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not api_key:

        print(
            "\nWARNING: ENABLE_LLM_JUDGE=true "
            "but OPENROUTER_API_KEY is missing."
        )

        return False

    try:

        from openai import OpenAI

        JUDGE_CLIENT = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        return True

    except Exception as e:

        print(
            "\nWARNING: Could not initialize "
            f"LLM judge: {e}"
        )

        return False


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# PRODUCT NAME EXTRACTION
# ============================================================

def get_product_name(doc):

    metadata = getattr(
        doc,
        "metadata",
        {}
    ) or {}

    possible_fields = [
        "product_name",
        "product",
        "name",
        "model",
        "title",
        "product_title",
    ]

    for field in possible_fields:

        value = metadata.get(field)

        if value:

            return str(value).strip()

    return ""


def get_context(doc):

    return getattr(
        doc,
        "page_content",
        ""
    ) or ""


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_documents(result):

    """
    Attempts to locate LangChain Documents inside the
    existing run_streaming_rag() return value.

    NO modification is made to the production pipeline.
    """

    if result is None:
        return []

    # Direct list
    if isinstance(result, list):

        docs = [
            x
            for x in result
            if hasattr(x, "page_content")
        ]

        if docs:
            return docs

    # Tuple returned by existing RAG
    if isinstance(result, tuple):

        for value in result:

            if isinstance(value, list):

                docs = [
                    x
                    for x in value
                    if hasattr(
                        x,
                        "page_content"
                    )
                ]

                if docs:
                    return docs

    # Dictionary
    if isinstance(result, dict):

        for key in [
            "documents",
            "docs",
            "retrieved_docs",
            "context",
            "sources",
        ]:

            value = result.get(key)

            if isinstance(value, list):

                docs = [
                    x
                    for x in value
                    if hasattr(
                        x,
                        "page_content"
                    )
                ]

                if docs:
                    return docs

    return []


# ============================================================
# ANSWER EXTRACTION
# ============================================================

def extract_answer(result):

    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if isinstance(result, dict):

        for key in [
            "answer",
            "response",
            "message",
            "text",
            "output",
        ]:

            value = result.get(key)

            if isinstance(value, str):
                return value

    if isinstance(result, tuple):

        # Prefer strings that look like actual answers.
        strings = [
            x
            for x in result
            if isinstance(x, str)
        ]

        if strings:

            # Usually the longest string is the generated
            # response rather than a short status value.
            return max(
                strings,
                key=len
            )

    return ""


# ============================================================
# RETRIEVED PRODUCT LIST
# ============================================================

def get_retrieved_products(docs):

    products = []

    for doc in docs:

        name = get_product_name(doc)

        if name:
            products.append(
                normalize(name)
            )

    return products


# ============================================================
# PRODUCT MATCHING
# ============================================================

def product_matches(
    retrieved,
    expected
):

    retrieved = normalize(
        retrieved
    )

    expected = normalize(
        expected
    )

    if not retrieved or not expected:
        return False

    if retrieved == expected:
        return True

    if expected in retrieved:
        return True

    if retrieved in expected:
        return True

    # Token overlap
    expected_tokens = set(
        expected.split()
    )

    retrieved_tokens = set(
        retrieved.split()
    )

    if not expected_tokens:
        return False

    overlap = (
        len(
            expected_tokens
            & retrieved_tokens
        )
        / len(expected_tokens)
    )

    return overlap >= 0.7


# ============================================================
# RELEVANCE MATCH
# ============================================================

def is_relevant(
    retrieved,
    expected_products
):

    if not expected_products:
        return False

    for expected in expected_products:

        if product_matches(
            retrieved,
            expected
        ):
            return True

    return False


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def hit_rate(
    retrieved,
    expected,
    k
):

    if not expected:
        return None

    return float(
        any(
            is_relevant(
                item,
                expected
            )
            for item in retrieved[:k]
        )
    )


def precision_at_k(
    retrieved,
    expected,
    k
):

    if not expected:
        return None

    top = retrieved[:k]

    if not top:
        return 0.0

    relevant_count = sum(
        is_relevant(
            item,
            expected
        )
        for item in top
    )

    return (
        relevant_count
        / len(top)
    )


def recall_at_k(
    retrieved,
    expected,
    k
):

    if not expected:
        return None

    matched = 0

    for expected_item in expected:

        found = any(
            product_matches(
                retrieved_item,
                expected_item
            )
            for retrieved_item
            in retrieved[:k]
        )

        if found:
            matched += 1

    return (
        matched
        / len(expected)
    )


def reciprocal_rank(
    retrieved,
    expected
):

    if not expected:
        return None

    for rank, item in enumerate(
        retrieved,
        start=1
    ):

        if is_relevant(
            item,
            expected
        ):

            return 1.0 / rank

    return 0.0


def average_precision(
    retrieved,
    expected
):

    if not expected:
        return None

    if not retrieved:
        return 0.0

    score = 0.0
    hits = 0

    for rank, item in enumerate(
        retrieved,
        start=1
    ):

        if is_relevant(
            item,
            expected
        ):

            hits += 1

            score += (
                hits / rank
            )

    return score / len(expected)


def ndcg_at_k(
    retrieved,
    expected,
    k
):

    if not expected:
        return None

    top = retrieved[:k]

    dcg = 0.0

    for rank, item in enumerate(
        top,
        start=1
    ):

        if is_relevant(
            item,
            expected
        ):

            dcg += (
                1.0
                / math.log2(rank + 1)
            )

    ideal_hits = min(
        len(expected),
        k
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0
        / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    return dcg / idcg


# ============================================================
# DUPLICATE RETRIEVAL RATE
# ============================================================

def duplicate_rate(products):

    if not products:
        return 0.0

    unique = len(
        set(products)
    )

    return 1.0 - (
        unique / len(products)
    )


# ============================================================
# SIMPLE ANSWER RELEVANCY
# ============================================================

def lexical_answer_relevancy(
    question,
    answer
):

    question_tokens = set(
        normalize(question).split()
    )

    answer_tokens = set(
        normalize(answer).split()
    )

    if not question_tokens:
        return 0.0

    overlap = (
        question_tokens
        & answer_tokens
    )

    return len(overlap) / len(
        question_tokens
    )


# ============================================================
# PRODUCT MENTION ACCURACY
# ============================================================

def product_mention_accuracy(
    answer,
    expected_products
):

    if not expected_products:
        return None

    answer = normalize(answer)

    if not answer:
        return 0.0

    matched = sum(
        1
        for product in expected_products
        if normalize(product)
        in answer
    )

    return (
        matched
        / len(expected_products)
    )


# ============================================================
# LLM JUDGE
# ============================================================

def run_llm_judge(
    question,
    answer,
    contexts,
    expected_answer=""
):

    if JUDGE_CLIENT is None:
        return {}

    context_text = "\n\n".join(
        contexts
    )

    prompt = f"""
You are a strict evaluator of a
Retrieval-Augmented Generation system.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context_text}

GENERATED ANSWER:
{answer}

REFERENCE:
{expected_answer}

Evaluate:

1. Faithfulness:
Are factual claims supported by the retrieved context?

2. Answer relevancy:
Does the answer directly answer the question?

3. Answer correctness:
Is the answer factually correct compared with
the reference when a reference exists?

4. Hallucination:
How much unsupported factual information exists?

Return ONLY JSON:

{{
  "faithfulness": 0.0,
  "answer_relevancy": 0.0,
  "answer_correctness": 0.0,
  "hallucination_rate": 0.0,
  "explanation": ""
}}

Scores must be between 0 and 1.

For hallucination_rate:
0 = no unsupported claims
1 = essentially entirely unsupported
"""

    try:

        response = (
            JUDGE_CLIENT
            .chat.completions.create(
                model=JUDGE_MODEL,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are a rigorous RAG evaluator."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
        )

        text = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        if text.startswith("```"):

            text = re.sub(
                r"^```(?:json)?",
                "",
                text
            )

            text = re.sub(
                r"```$",
                "",
                text
            )

            text = text.strip()

        return json.loads(text)

    except Exception as e:

        return {
            "judge_error": str(e)
        }


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

def evaluate_question(test):

    question_id = test["id"]
    question = test["question"]

    expected_products = [
        normalize(x)
        for x in test.get(
            "expected_products",
            []
        )
    ]

    expected_answer = test.get(
        "expected_answer",
        ""
    )

    record = {
        "id": question_id,
        "category": test.get(
            "category",
            ""
        ),
        "question": question,
        "answerable": test.get(
            "answerable",
            True
        ),
        "status": "ok",
    }

    started = time.perf_counter()

    try:

        # ====================================================
        # USE EXISTING PIPELINE
        # ====================================================

        result = run_streaming_rag(
            question,
            chat_history=[]
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        record[
            "latency_seconds"
        ] = round(
            elapsed,
            4
        )

        # ====================================================
        # EXTRACT RESULT
        # ====================================================

        answer = extract_answer(
            result
        )

        docs = extract_documents(
            result
        )

        products = (
            get_retrieved_products(
                docs
            )
        )

        contexts = [
            get_context(doc)
            for doc in docs
        ]

        record["answer"] = answer

        record[
            "retrieved_count"
        ] = len(docs)

        record[
            "retrieved_products"
        ] = " | ".join(
            products
        )

        record[
            "context_characters"
        ] = sum(
            len(x)
            for x in contexts
        )

        record[
            "duplicate_rate"
        ] = duplicate_rate(
            products
        )

        # ====================================================
        # RETRIEVAL METRICS
        # ====================================================

        for k in K_VALUES:

            record[
                f"hit_rate@{k}"
            ] = hit_rate(
                products,
                expected_products,
                k
            )

            record[
                f"precision@{k}"
            ] = precision_at_k(
                products,
                expected_products,
                k
            )

            record[
                f"recall@{k}"
            ] = recall_at_k(
                products,
                expected_products,
                k
            )

            record[
                f"ndcg@{k}"
            ] = ndcg_at_k(
                products,
                expected_products,
                k
            )

        record[
            "mrr"
        ] = reciprocal_rank(
            products,
            expected_products
        )

        record[
            "map"
        ] = average_precision(
            products,
            expected_products
        )

        # ====================================================
        # ANSWER METRICS
        # ====================================================

        record[
            "lexical_answer_relevancy"
        ] = lexical_answer_relevancy(
            question,
            answer
        )

        record[
            "product_mention_accuracy"
        ] = product_mention_accuracy(
            answer,
            expected_products
        )

        # ====================================================
        # LLM JUDGE
        # ====================================================

        if (
            ENABLE_LLM_JUDGE
            and answer
            and contexts
        ):

            judge = run_llm_judge(
                question,
                answer,
                contexts,
                expected_answer
            )

            record.update(
                judge
            )

        else:

            record[
                "faithfulness"
            ] = None

            record[
                "answer_relevancy"
            ] = None

            record[
                "answer_correctness"
            ] = None

            record[
                "hallucination_rate"
            ] = None

        # ====================================================
        # EMPTY RETRIEVAL
        # ====================================================

        record[
            "empty_retrieval"
        ] = int(
            len(docs) == 0
        )

        record[
            "empty_answer"
        ] = int(
            not answer.strip()
        )

    except Exception as e:

        elapsed = (
            time.perf_counter()
            - started
        )

        record["status"] = "error"

        record[
            "latency_seconds"
        ] = round(
            elapsed,
            4
        )

        record[
            "error"
        ] = str(e)

        record[
            "traceback"
        ] = traceback.format_exc()

    return record


# ============================================================
# SAFE MEAN
# ============================================================

def safe_mean(series):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if values.empty:
        return None

    return float(
        values.mean()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("PRICEOYE RAG FULL EVALUATION")
    print("=" * 72)

    print(
        f"\nTest questions: "
        f"{len(TEST_CASES)}"
    )

    print(
        f"LLM judge: "
        f"{'ENABLED' if ENABLE_LLM_JUDGE else 'DISABLED'}"
    )

    if ENABLE_LLM_JUDGE:

        initialize_judge()

        if JUDGE_CLIENT is None:

            print(
                "\nLLM judge unavailable."
            )

            print(
                "Continuing with deterministic metrics."
            )

    print()

    results = []

    for index, test in enumerate(
        TEST_CASES,
        start=1
    ):

        print(
            f"[{index:03d}/{len(TEST_CASES):03d}] "
            f"{test['id']} "
            f"{test['question']}"
        )

        result = evaluate_question(
            test
        )

        results.append(
            result
        )

        print(
            f"       "
            f"status={result['status']} "
            f"latency="
            f"{result.get('latency_seconds', 0):.2f}s"
        )

    df = pd.DataFrame(
        results
    )

    # ========================================================
    # PER QUESTION CSV
    # ========================================================

    per_query_file = (
        OUTPUT_DIR
        / "evaluation_results.csv"
    )

    df.to_csv(
        per_query_file,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # FAILED QUERIES
    # ========================================================

    failed = df[
        df["status"] != "ok"
    ]

    failed.to_csv(
        OUTPUT_DIR
        / "failed_queries.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    metrics = {}

    metric_columns = [

        # Retrieval
        "hit_rate@1",
        "hit_rate@3",
        "hit_rate@5",
        "hit_rate@10",

        "precision@1",
        "precision@3",
        "precision@5",
        "precision@10",

        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",

        "ndcg@1",
        "ndcg@3",
        "ndcg@5",
        "ndcg@10",

        "mrr",
        "map",

        # Generation
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "hallucination_rate",

        # Custom
        "lexical_answer_relevancy",
        "product_mention_accuracy",

        # System
        "duplicate_rate",
        "latency_seconds",
        "empty_retrieval",
        "empty_answer",
    ]

    for column in metric_columns:

        if column not in df.columns:
            continue

        metrics[column] = safe_mean(
            df[column]
        )

    metrics[
        "total_questions"
    ] = len(df)

    metrics[
        "successful_questions"
    ] = int(
        (
            df["status"]
            == "ok"
        ).sum()
    )

    metrics[
        "failed_questions"
    ] = int(
        (
            df["status"]
            != "ok"
        ).sum()
    )

    metrics[
        "error_rate"
    ] = (
        metrics["failed_questions"]
        / metrics["total_questions"]
    )

    # ========================================================
    # CATEGORY METRICS
    # ========================================================

    category_rows = []

    for category, group in df.groupby(
        "category"
    ):

        row = {
            "category": category,
            "questions": len(group),
        }

        for column in [

            "hit_rate@5",
            "precision@5",
            "recall@5",
            "ndcg@5",
            "mrr",
            "map",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
            "hallucination_rate",
            "lexical_answer_relevancy",
            "product_mention_accuracy",
            "duplicate_rate",
            "latency_seconds",
        ]:

            if column in group.columns:

                row[column] = safe_mean(
                    group[column]
                )

        category_rows.append(
            row
        )

    category_df = pd.DataFrame(
        category_rows
    )

    category_df.to_csv(
        OUTPUT_DIR
        / "category_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # JSON METRICS
    # ========================================================

    with open(
        OUTPUT_DIR
        / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    # ========================================================
    # MARKDOWN REPORT
    # ========================================================

    report_file = (
        OUTPUT_DIR
        / "REPORT.md"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# PriceOye RAG Evaluation Report\n\n"
        )

        f.write(
            "This evaluation uses the existing "
            "`run_streaming_rag()` pipeline without "
            "modifying the production retrieval code.\n\n"
        )

        f.write(
            f"**Questions:** {len(df)}\n\n"
        )

        f.write(
            f"**Successful:** "
            f"{metrics['successful_questions']}\n\n"
        )

        f.write(
            f"**Failed:** "
            f"{metrics['failed_questions']}\n\n"
        )

        f.write(
            f"**Error rate:** "
            f"{metrics['error_rate']:.4f}\n\n"
        )

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        f.write(
            "## Retrieval Metrics\n\n"
        )

        f.write(
            "| Metric | Score |\n"
        )

        f.write(
            "|---|---:|\n"
        )

        for metric in [

            "hit_rate@1",
            "hit_rate@3",
            "hit_rate@5",
            "hit_rate@10",

            "precision@1",
            "precision@3",
            "precision@5",
            "precision@10",

            "recall@1",
            "recall@3",
            "recall@5",
            "recall@10",

            "ndcg@5",
            "ndcg@10",

            "mrr",
            "map",

        ]:

            value = metrics.get(
                metric
            )

            if value is not None:

                f.write(
                    f"| {metric} | "
                    f"{value:.4f} |\n"
                )

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        f.write(
            "\n## Generation Metrics\n\n"
        )

        f.write(
            "| Metric | Score |\n"
        )

        f.write(
            "|---|---:|\n"
        )

        for metric in [

            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
            "hallucination_rate",

        ]:

            value = metrics.get(
                metric
            )

            if value is not None:

                f.write(
                    f"| {metric} | "
                    f"{value:.4f} |\n"
                )

        if not ENABLE_LLM_JUDGE:

            f.write(
                "\n> LLM judge was disabled. "
                "Faithfulness, Answer Relevancy and "
                "Answer Correctness were therefore not "
                "quantitatively judged.\n"
            )

        # ----------------------------------------------------
        # Custom
        # ----------------------------------------------------

        f.write(
            "\n## PriceOye-Specific Metrics\n\n"
        )

        f.write(
            "| Metric | Score |\n"
        )

        f.write(
            "|---|---:|\n"
        )

        for metric in [

            "product_mention_accuracy",
            "lexical_answer_relevancy",
            "duplicate_rate",
            "empty_retrieval",
            "empty_answer",

        ]:

            value = metrics.get(
                metric
            )

            if value is not None:

                f.write(
                    f"| {metric} | "
                    f"{value:.4f} |\n"
                )

        # ----------------------------------------------------
        # System
        # ----------------------------------------------------

        f.write(
            "\n## System Metrics\n\n"
        )

        f.write(
            "| Metric | Value |\n"
        )

        f.write(
            "|---|---:|\n"
        )

        for metric in [

            "latency_seconds",
            "error_rate",

        ]:

            value = metrics.get(
                metric
            )

            if value is not None:

                f.write(
                    f"| {metric} | "
                    f"{value:.4f} |\n"
                )

        # ----------------------------------------------------
        # Categories
        # ----------------------------------------------------

        f.write(
            "\n## Category Results\n\n"
        )

        if not category_df.empty:

            f.write(
                category_df.to_markdown(
                    index=False
                )
            )

        # ----------------------------------------------------
        # Errors
        # ----------------------------------------------------

        f.write(
            "\n\n## Failed Queries\n\n"
        )

        if failed.empty:

            f.write(
                "No failed queries.\n"
            )

        else:

            for _, row in failed.iterrows():

                f.write(
                    f"### {row['id']}\n\n"
                )

                f.write(
                    f"**Question:** "
                    f"{row['question']}\n\n"
                )

                f.write(
                    f"**Error:** "
                    f"{row.get('error', '')}\n\n"
                )

        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        f.write(
            "\n## Interpretation\n\n"
        )

        f.write(
            "Higher is better for Hit Rate, Precision, "
            "Recall, nDCG, MRR, MAP, Faithfulness, "
            "Answer Relevancy and Answer Correctness.\n\n"
        )

        f.write(
            "Lower is better for Hallucination Rate, "
            "Duplicate Rate, Error Rate and Latency.\n"
        )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)

    print()

    print(
        f"Questions: "
        f"{len(df)}"
    )

    print(
        f"Successful: "
        f"{metrics['successful_questions']}"
    )

    print(
        f"Failed: "
        f"{metrics['failed_questions']}"
    )

    print(
        f"Error rate: "
        f"{metrics['error_rate']:.2%}"
    )

    print()

    print("RETRIEVAL")

    for metric in [
        "hit_rate@5",
        "precision@5",
        "recall@5",
        "ndcg@5",
        "mrr",
        "map",
    ]:

        value = metrics.get(
            metric
        )

        if value is not None:

            print(
                f"  {metric:<20}"
                f"{value:.4f}"
            )

    print()

    print("GENERATION")

    for metric in [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "hallucination_rate",
    ]:

        value = metrics.get(
            metric
        )

        if value is not None:

            print(
                f"  {metric:<20}"
                f"{value:.4f}"
            )

    print()

    print("SYSTEM")

    print(
        f"  {'Average latency':<20}"
        f"{metrics.get('latency_seconds', 0):.3f}s"
    )

    print(
        f"  {'Empty retrieval':<20}"
        f"{metrics.get('empty_retrieval', 0):.2%}"
    )

    print(
        f"  {'Duplicate rate':<20}"
        f"{metrics.get('duplicate_rate', 0):.2%}"
    )

    print()

    print(
        f"Detailed results:\n"
        f"  {OUTPUT_DIR}"
    )

    print(
        f"\nReport:\n"
        f"  {report_file}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
