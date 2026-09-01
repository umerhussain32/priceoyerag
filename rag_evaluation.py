````python
"""
PriceOye RAG - 10 Question Evaluation
=====================================

IMPORTANT:
- Does NOT modify retrieval_pipeline.py
- Does NOT modify app.py
- Uses the EXISTING run_streaming_rag()
- Only 10 evaluation questions
- Does not invent retrieval results
- Does not calculate invalid metrics
- Separates API errors from RAG quality
- Produces CSV + JSON + Markdown report

Run:

    python rag_evaluation.py

Optional LLM judge:

    ENABLE_LLM_JUDGE=true python rag_evaluation.py

Files created:

    rag_evaluation_results/
        evaluation_results.csv
        metrics.json
        category_results.csv
        failed_queries.csv
        REPORT.md
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import re
import traceback
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORT YOUR EXISTING PIPELINE
# ============================================================

try:
    from retrieval_pipeline import run_streaming_rag
except Exception as exc:
    print("\nERROR: Could not import run_streaming_rag().")
    print(str(exc))
    print("\nMake sure rag_evaluation.py is in the repository root.")
    sys.exit(1)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = ROOT / "rag_evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ONLY 10 QUESTIONS
# ============================================================
#
# IMPORTANT:
# These are deliberately mixed:
#
# 1 exact model
# 2 brand
# 3 price
# 4 RAM
# 5 storage
# 6 combined filter
# 7 keypad
# 8 battery
# 9 policy
# 10 unanswerable
#
# For retrieval ground-truth metrics, expected_products
# should ideally be verified against your actual dataset.
#
# Empty expected_products means retrieval precision/recall
# is NOT calculated for that question.
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
        "expected_products": [],
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
        "category": "ram",
        "question": "Show me phones with 8GB RAM.",
        "expected_products": [],
        "answerable": True,
    },

    {
        "id": "Q005",
        "category": "storage",
        "question": "Show me phones with 256GB storage.",
        "expected_products": [],
        "answerable": True,
    },

    {
        "id": "Q006",
        "category": "multi_filter",
        "question": "Show me Samsung phones with 8GB RAM and 256GB storage under 60000.",
        "expected_products": [],
        "answerable": True,
    },

    {
        "id": "Q007",
        "category": "keypad",
        "question": "Show me Nokia keypad phones.",
        "expected_products": [],
        "answerable": True,
    },

    {
        "id": "Q008",
        "category": "battery",
        "question": "Show me phones with a 6000mAh battery.",
        "expected_products": [],
        "answerable": True,
    },

    {
        "id": "Q009",
        "category": "policy",
        "question": "What is the return policy?",
        "expected_products": [],
        "answerable": True,
    },

    {
        "id": "Q010",
        "category": "unanswerable",
        "question": "Give me a guaranteed 90 percent PriceOye discount code.",
        "expected_products": [],
        "answerable": False,
    },
]


# ============================================================
# SETTINGS
# ============================================================

ENABLE_LLM_JUDGE = (
    os.getenv(
        "ENABLE_LLM_JUDGE",
        "false"
    ).lower()
    in ("1", "true", "yes", "y")
)

K_VALUES = [1, 3, 5, 10]


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
            "\nWARNING: ENABLE_LLM_JUDGE=true but "
            "OPENROUTER_API_KEY is not set."
        )
        return False

    try:

        from openai import OpenAI

        JUDGE_CLIENT = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        return True

    except Exception as exc:

        print(
            "\nWARNING: Could not initialize LLM judge:"
        )
        print(str(exc))

        return False


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize(value):

    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).lower()
    ).strip()


def is_document(obj):

    return hasattr(
        obj,
        "page_content"
    )


# ============================================================
# DEEP SEARCH FOR DOCUMENTS
# ============================================================

def find_documents(obj, found=None, visited=None):

    """
    Recursively searches the returned object for LangChain
    Document objects.

    This does NOT change your RAG.

    It only inspects the return value.
    """

    if found is None:
        found = []

    if visited is None:
        visited = set()

    # Avoid recursive structures
    try:
        object_id = id(obj)

        if object_id in visited:
            return found

        visited.add(object_id)

    except Exception:
        pass

    # Direct Document
    if is_document(obj):

        if obj not in found:
            found.append(obj)

        return found

    # List / tuple / set
    if isinstance(
        obj,
        (list, tuple, set)
    ):

        for item in obj:

            find_documents(
                item,
                found,
                visited
            )

        return found

    # Dictionary
    if isinstance(obj, dict):

        for value in obj.values():

            find_documents(
                value,
                found,
                visited
            )

        return found

    # Objects that may contain useful attributes
    for attr in [
        "documents",
        "docs",
        "retrieved_docs",
        "context",
        "sources",
    ]:

        try:

            value = getattr(
                obj,
                attr,
                None
            )

            if value is not None:

                find_documents(
                    value,
                    found,
                    visited
                )

        except Exception:
            pass

    return found


# ============================================================
# DEEP STRING EXTRACTION
# ============================================================

def find_strings(obj, found=None, visited=None):

    """
    Recursively finds strings in the returned value.
    """

    if found is None:
        found = []

    if visited is None:
        visited = set()

    try:

        object_id = id(obj)

        if object_id in visited:
            return found

        visited.add(object_id)

    except Exception:
        pass

    if isinstance(obj, str):

        text = obj.strip()

        if text:
            found.append(text)

        return found

    if isinstance(
        obj,
        (list, tuple, set)
    ):

        for item in obj:

            find_strings(
                item,
                found,
                visited
            )

        return found

    if isinstance(obj, dict):

        for key, value in obj.items():

            # Prefer obvious answer fields
            if str(key).lower() in {
                "answer",
                "response",
                "message",
                "text",
                "output",
                "response_text",
                "answer_text",
            }:

                if isinstance(value, str):

                    if value.strip():
                        found.append(
                            value.strip()
                        )

            find_strings(
                value,
                found,
                visited
            )

    return found


# ============================================================
# ANSWER EXTRACTION
# ============================================================

def extract_answer(result):

    """
    Attempts to extract the generated answer.

    We prioritize obvious dictionary fields and then inspect
    returned strings.

    If nothing can safely be identified, returns "" rather
    than pretending a result is an answer.
    """

    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):

        priority_keys = [
            "answer",
            "response",
            "message",
            "text",
            "output",
            "response_text",
            "answer_text",
        ]

        for key in priority_keys:

            value = result.get(key)

            if isinstance(value, str):

                if value.strip():

                    return value.strip()

    strings = find_strings(result)

    if not strings:
        return ""

    # Remove obvious metadata/status strings
    candidates = []

    for text in strings:

        lowered = normalize(text)

        if lowered in {
            "ok",
            "success",
            "error",
            "true",
            "false",
            "none",
            "null",
        }:
            continue

        candidates.append(text)

    if not candidates:
        return ""

    # The generated response is generally the longest
    # natural-language string.
    return max(
        candidates,
        key=len
    )


# ============================================================
# METADATA
# ============================================================

def get_metadata(doc):

    try:

        metadata = getattr(
            doc,
            "metadata",
            {}
        )

        if isinstance(
            metadata,
            dict
        ):
            return metadata

    except Exception:
        pass

    return {}


def get_product_name(doc):

    metadata = get_metadata(doc)

    fields = [
        "product_name",
        "product",
        "name",
        "model",
        "title",
        "product_title",
    ]

    for field in fields:

        value = metadata.get(
            field
        )

        if value:
            return str(value).strip()

    # Fallback: do NOT assume arbitrary page text is a
    # product name.
    return ""


def get_document_text(doc):

    try:

        return str(
            getattr(
                doc,
                "page_content",
                ""
            )
            or ""
        )

    except Exception:

        return ""


# ============================================================
# RETRIEVED PRODUCT NAMES
# ============================================================

def retrieved_products(docs):

    output = []

    for doc in docs:

        name = get_product_name(
            doc
        )

        if name:

            output.append(
                name
            )

    return output


# ============================================================
# PRODUCT MATCH
# ============================================================

def product_matches(
    retrieved,
    expected
):

    r = normalize(
        retrieved
    )

    e = normalize(
        expected
    )

    if not r or not e:
        return False

    if r == e:
        return True

    if e in r:
        return True

    if r in e:
        return True

    r_tokens = set(
        r.split()
    )

    e_tokens = set(
        e.split()
    )

    if not e_tokens:
        return False

    overlap = len(
        r_tokens & e_tokens
    ) / len(e_tokens)

    return overlap >= 0.70


def relevant(
    retrieved,
    expected_products
):

    return any(
        product_matches(
            retrieved_item,
            expected_item
        )
        for expected_item
        in expected_products
    )


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def hit_rate_at_k(
    products,
    expected,
    k
):

    if not expected:
        return None

    return float(
        any(
            relevant(
                item,
                expected
            )
            for item in products[:k]
        )
    )


def precision_at_k(
    products,
    expected,
    k
):

    if not expected:
        return None

    top = products[:k]

    if not top:
        return 0.0

    hits = sum(
        relevant(
            item,
            expected
        )
        for item in top
    )

    return hits / len(top)


def recall_at_k(
    products,
    expected,
    k
):

    if not expected:
        return None

    if not products:
        return 0.0

    hits = 0

    for expected_item in expected:

        if any(
            product_matches(
                item,
                expected_item
            )
            for item in products[:k]
        ):
            hits += 1

    return hits / len(
        expected
    )


def reciprocal_rank(
    products,
    expected
):

    if not expected:
        return None

    for rank, item in enumerate(
        products,
        start=1
    ):

        if relevant(
            item,
            expected
        ):

            return 1.0 / rank

    return 0.0


def average_precision(
    products,
    expected
):

    if not expected:
        return None

    if not products:
        return 0.0

    score = 0.0
    hits = 0

    for rank, item in enumerate(
        products,
        start=1
    ):

        if relevant(
            item,
            expected
        ):

            hits += 1

            score += (
                hits / rank
            )

    # Correct AP denominator is the number of relevant
    # ground-truth items, capped by what can be retrieved.
    denominator = min(
        len(expected),
        len(products)
    )

    if denominator == 0:
        return 0.0

    return min(
        1.0,
        score / denominator
    )


def ndcg_at_k(
    products,
    expected,
    k
):

    """
    Proper bounded nDCG.

    Always returns 0..1.
    """

    if not expected:
        return None

    top = products[:k]

    if not top:
        return 0.0

    # Binary relevance
    dcg = 0.0

    for rank, item in enumerate(
        top,
        start=1
    ):

        if relevant(
            item,
            expected
        ):

            dcg += (
                1.0
                / math.log2(rank + 1)
            )

    ideal_count = min(
        len(expected),
        k
    )

    if ideal_count <= 0:
        return 0.0

    idcg = sum(
        1.0
        / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_count + 1
        )
    )

    if idcg == 0:
        return 0.0

    return min(
        1.0,
        dcg / idcg
    )


# ============================================================
# DUPLICATE RATE
# ============================================================

def duplicate_rate(products):

    if not products:
        return 0.0

    normalized = [
        normalize(x)
        for x in products
        if normalize(x)
    ]

    if not normalized:
        return 0.0

    return (
        1.0
        -
        len(set(normalized))
        /
        len(normalized)
    )


# ============================================================
# SIMPLE ANSWER RELEVANCY
# ============================================================

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "what",
    "which",
    "show",
    "me",
    "of",
    "for",
    "with",
    "and",
    "in",
    "on",
    "to",
    "my",
    "do",
    "i",
    "you",
    "tell",
}


def lexical_answer_relevancy(
    question,
    answer
):

    if not answer:
        return 0.0

    question_tokens = {
        token
        for token in re.findall(
            r"\b\w+\b",
            normalize(question)
        )
        if token not in STOPWORDS
    }

    answer_tokens = {
        token
        for token in re.findall(
            r"\b\w+\b",
            normalize(answer)
        )
        if token not in STOPWORDS
    }

    if not question_tokens:
        return 0.0

    return len(
        question_tokens
        &
        answer_tokens
    ) / len(
        question_tokens
    )


# ============================================================
# PRODUCT MENTION
# ============================================================

def product_mention_accuracy(
    answer,
    expected_products
):

    if not expected_products:
        return None

    if not answer:
        return 0.0

    answer_norm = normalize(
        answer
    )

    hits = sum(
        normalize(product)
        in answer_norm
        for product
        in expected_products
    )

    return hits / len(
        expected_products
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
        contexts[:10]
    )

    prompt = f"""
You are evaluating a RAG system.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context_text}

GENERATED ANSWER:
{answer}

REFERENCE ANSWER:
{expected_answer}

Evaluate the generated answer.

faithfulness:
Are the factual claims supported by the retrieved context?

answer_relevancy:
Does the answer directly address the question?

answer_correctness:
Is the answer correct according to the reference/context?

hallucination_rate:
What proportion of factual claims are unsupported?

Return ONLY valid JSON:

{{
  "faithfulness": 0.0,
  "answer_relevancy": 0.0,
  "answer_correctness": 0.0,
  "hallucination_rate": 0.0
}}

All values must be between 0 and 1.
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
                            "You are a strict RAG evaluator."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
            )
        )

        text = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        text = re.sub(
            r"^```json\s*",
            "",
            text,
            flags=re.I
        )

        text = re.sub(
            r"^```\s*",
            "",
            text
        )

        text = re.sub(
            r"\s*```$",
            "",
            text
        )

        data = json.loads(
            text.strip()
        )

        # Clamp judge values safely to 0..1
        for key in [
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
            "hallucination_rate",
        ]:

            if key in data:

                try:

                    data[key] = max(
                        0.0,
                        min(
                            1.0,
                            float(
                                data[key]
                            )
                        )
                    )

                except Exception:

                    data[key] = None

        return data

    except Exception as exc:

        return {
            "judge_error": str(exc)
        }


# ============================================================
# API ERROR DETECTION
# ============================================================

def classify_error(exc):

    text = normalize(
        str(exc)
    )

    if (
        "429" in text
        or "rate limit" in text
        or "free-models-per-day" in text
    ):
        return "rate_limit"

    if (
        "401" in text
        or "unauthorized" in text
        or "invalid api key" in text
    ):
        return "authentication"

    if (
        "timeout" in text
        or "timed out" in text
    ):
        return "timeout"

    if "pinecone" in text:
        return "pinecone"

    return "pipeline_error"


# ============================================================
# ONE QUERY
# ============================================================

def evaluate_question(test):

    question = test["question"]

    expected = [
        normalize(x)
        for x in test.get(
            "expected_products",
            []
        )
        if normalize(x)
    ]

    record = {
        "id": test["id"],
        "category": test["category"],
        "question": question,
        "answerable": test.get(
            "answerable",
            True
        ),
        "status": "ok",
        "error_type": "",
        "error": "",
        "answer": "",
        "retrieved_document_count": 0,
        "retrieved_product_count": 0,
        "retrieved_products": "",
    }

    start = time.perf_counter()

    try:

        # ====================================================
        # CALL YOUR EXISTING PRODUCTION RAG
        # ====================================================

        result = run_streaming_rag(
            question,
            chat_history=[]
        )

        record[
            "latency_seconds"
        ] = round(
            time.perf_counter()
            - start,
            4
        )

        # ====================================================
        # FIND DOCUMENTS
        # ====================================================

        docs = find_documents(
            result
        )

        # ====================================================
        # FIND ANSWER
        # ====================================================

        answer = extract_answer(
            result
        )

        products = retrieved_products(
            docs
        )

        contexts = [
            get_document_text(doc)
            for doc in docs
            if get_document_text(doc)
        ]

        # ====================================================
        # BASIC OUTPUT
        # ====================================================

        record["answer"] = answer

        record[
            "retrieved_document_count"
        ] = len(docs)

        record[
            "retrieved_product_count"
        ] = len(products)

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
            "empty_retrieval"
        ] = int(
            len(docs) == 0
        )

        record[
            "empty_answer"
        ] = int(
            not answer.strip()
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
            ] = hit_rate_at_k(
                products,
                expected,
                k
            )

            record[
                f"precision@{k}"
            ] = precision_at_k(
                products,
                expected,
                k
            )

            record[
                f"recall@{k}"
            ] = recall_at_k(
                products,
                expected,
                k
            )

            record[
                f"ndcg@{k}"
            ] = ndcg_at_k(
                products,
                expected,
                k
            )

        record["mrr"] = reciprocal_rank(
            products,
            expected
        )

        record["map"] = average_precision(
            products,
            expected
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
            expected
        )

        # ====================================================
        # LLM JUDGE
        # ====================================================

        if (
            ENABLE_LLM_JUDGE
            and JUDGE_CLIENT is not None
            and answer
            and contexts
        ):

            judge = run_llm_judge(
                question,
                answer,
                contexts,
                test.get(
                    "expected_answer",
                    ""
                )
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

    except Exception as exc:

        record[
            "status"
        ] = "error"

        record[
            "error_type"
        ] = classify_error(
            exc
        )

        record[
            "error"
        ] = str(exc)

        record[
            "traceback"
        ] = traceback.format_exc()

        record[
            "latency_seconds"
        ] = round(
            time.perf_counter()
            - start,
            4
        )

    return record


# ============================================================
# SAFE MEAN
# ============================================================

def safe_mean(
    series
):

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
    print("PRICEOYE RAG - 10 QUESTION EVALUATION")
    print("=" * 72)

    print(
        f"\nQuestions: {len(TEST_CASES)}"
    )

    print(
        "Production retrieval modified: NO"
    )

    print(
        "LLM judge: "
        + (
            "ENABLED"
            if ENABLE_LLM_JUDGE
            else "DISABLED"
        )
    )

    if ENABLE_LLM_JUDGE:

        initialize_judge()

    print()

    results = []

    for index, test in enumerate(
        TEST_CASES,
        start=1
    ):

        print(
            f"[{index:02d}/10] "
            f"{test['id']} - "
            f"{test['question']}"
        )

        result = evaluate_question(
            test
        )

        results.append(
            result
        )

        print(
            f"      status: "
            f"{result['status']}"
        )

        print(
            f"      latency: "
            f"{result.get('latency_seconds', 0):.2f}s"
        )

        if result["status"] == "error":

            print(
                f"      error: "
                f"{result['error_type']}"
            )

        else:

            print(
                f"      documents: "
                f"{result['retrieved_document_count']}"
            )

            print(
                f"      answer chars: "
                f"{len(result.get('answer', ''))}"
            )

        print()

        # Small delay to avoid hammering APIs
        if index < len(TEST_CASES):

            time.sleep(1.0)

    df = pd.DataFrame(
        results
    )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    df.to_csv(
        OUTPUT_DIR
        / "evaluation_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # FAILED
    # ========================================================

    failed = df[
        df["status"] != "ok"
    ].copy()

    failed.to_csv(
        OUTPUT_DIR
        / "failed_queries.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # IMPORTANT:
    # Quality metrics exclude API/pipeline failures.
    # ========================================================

    successful = df[
        df["status"] == "ok"
    ].copy()

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    metrics = {}

    metrics[
        "total_questions"
    ] = int(len(df))

    metrics[
        "successful_questions"
    ] = int(len(successful))

    metrics[
        "failed_questions"
    ] = int(len(failed))

    metrics[
        "error_rate"
    ] = (
        len(failed)
        /
        len(df)
        if len(df)
        else 0
    )

    # --------------------------------------------------------
    # API errors
    # --------------------------------------------------------

    if not failed.empty:

        counts = (
            failed[
                "error_type"
            ]
            .value_counts()
            .to_dict()
        )

    else:

        counts = {}

    metrics[
        "rate_limit_errors"
    ] = int(
        counts.get(
            "rate_limit",
            0
        )
    )

    metrics[
        "authentication_errors"
    ] = int(
        counts.get(
            "authentication",
            0
        )
    )

    metrics[
        "timeout_errors"
    ] = int(
        counts.get(
            "timeout",
            0
        )
    )

    # ========================================================
    # QUALITY METRICS
    # ========================================================

    quality_columns = [

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

        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "hallucination_rate",

        "lexical_answer_relevancy",
        "product_mention_accuracy",

        "duplicate_rate",
        "empty_retrieval",
        "empty_answer",

        "latency_seconds",
    ]

    for column in quality_columns:

        if column in successful.columns:

            metrics[column] = safe_mean(
                successful[column]
            )

    # ========================================================
    # ANSWER GENERATION RATE
    # ========================================================

    if len(successful):

        metrics[
            "answer_generation_rate"
        ] = 1.0 - safe_mean(
            successful[
                "empty_answer"
            ]
        )

        metrics[
            "retrieval_success_rate"
        ] = 1.0 - safe_mean(
            successful[
                "empty_retrieval"
            ]
        )

    else:

        metrics[
            "answer_generation_rate"
        ] = None

        metrics[
            "retrieval_success_rate"
        ] = None

    # ========================================================
    # CATEGORY RESULTS
    # ========================================================

    category_rows = []

    for category, group in df.groupby(
        "category"
    ):

        successful_group = group[
            group["status"] == "ok"
        ]

        row = {
            "category": category,
            "questions": len(group),
            "successful": len(
                successful_group
            ),
            "failed": (
                len(group)
                -
                len(successful_group)
            ),
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
            "empty_retrieval",
            "empty_answer",
            "latency_seconds",
        ]:

            if (
                column in
                successful_group.columns
                and not successful_group.empty
            ):

                row[column] = safe_mean(
                    successful_group[column]
                )

            else:

                row[column] = None

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
    # JSON
    # ========================================================

    with open(
        OUTPUT_DIR
        / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
            allow_nan=False
        )

    # ========================================================
    # MARKDOWN REPORT
    # ========================================================

    report = []

    report.append(
        "# PriceOye RAG Evaluation Report"
    )

    report.append("")

    report.append(
        "This evaluation calls the existing "
        "`run_streaming_rag()` function without "
        "modifying the production retrieval pipeline."
    )

    report.append("")

    report.append(
        f"**Questions:** {len(df)}"
    )

    report.append(
        f"**Successful:** {len(successful)}"
    )

    report.append(
        f"**Failed:** {len(failed)}"
    )

    report.append(
        f"**Error rate:** "
        f"{metrics['error_rate']:.4f}"
    )

    report.append("")

    # ========================================================
    # RETRIEVAL
    # ========================================================

    report.append(
        "## Retrieval Metrics"
    )

    report.append("")

    report.append(
        "| Metric | Score |"
    )

    report.append(
        "|---|---:|"
    )

    for column in [
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
    ]:

        value = metrics.get(
            column
        )

        if value is not None:

            report.append(
                f"| {column} | "
                f"{value:.4f} |"
            )

    report.append("")

    report.append(
        "> Retrieval precision/recall metrics are only "
        "calculated for questions that have verified "
        "expected product names."
    )

    # ========================================================
    # GENERATION
    # ========================================================

    report.append("")

    report.append(
        "## Generation Metrics"
    )

    report.append("")

    report.append(
        "| Metric | Score |"
    )

    report.append(
        "|---|---:|"
    )

    for column in [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "hallucination_rate",
    ]:

        value = metrics.get(
            column
        )

        if value is not None:

            report.append(
                f"| {column} | "
                f"{value:.4f} |"
            )

    if not ENABLE_LLM_JUDGE:

        report.append("")

        report.append(
            "> LLM judge disabled. Faithfulness, "
            "answer relevancy and answer correctness "
            "were not assigned artificial scores."
        )

    # ========================================================
    # SYSTEM
    # ========================================================

    report.append("")

    report.append(
        "## System Metrics"
    )

    report.append("")

    report.append(
        "| Metric | Value |"
    )

    report.append(
        "|---|---:|"
    )

    for column in [
        "answer_generation_rate",
        "retrieval_success_rate",
        "duplicate_rate",
        "latency_seconds",
        "error_rate",
        "rate_limit_errors",
    ]:

        value = metrics.get(
            column
        )

        if value is not None:

            if column in {
                "answer_generation_rate",
                "retrieval_success_rate",
                "duplicate_rate",
                "error_rate",
            }:

                formatted = (
                    f"{value:.2%}"
                )

            elif column == "latency_seconds":

                formatted = (
                    f"{value:.3f}s"
                )

            else:

                formatted = str(
                    value
                )

            report.append(
                f"| {column} | "
                f"{formatted} |"
            )

    # ========================================================
    # FAILED QUERIES
    # ========================================================

    report.append("")

    report.append(
        "## Failed Queries"
    )

    report.append("")

    if failed.empty:

        report.append(
            "No failed queries."
        )

    else:

        for _, row in failed.iterrows():

            report.append(
                f"### {row['id']}"
            )

            report.append("")

            report.append(
                f"**Question:** "
                f"{row['question']}"
            )

            report.append("")

            report.append(
                f"**Error type:** "
                f"{row['error_type']}"
            )

            report.append("")

            report.append(
                f"**Error:** "
                f"`{row['error']}`"
            )

            report.append("")

    # ========================================================
    # CATEGORY
    # ========================================================

    report.append(
        "## Category Results"
    )

    report.append("")

    if not category_df.empty:

        display_columns = [
            "category",
            "questions",
            "successful",
            "failed",
            "hit_rate@5",
            "precision@5",
            "recall@5",
            "ndcg@5",
            "mrr",
            "map",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
            "latency_seconds",
        ]

        available = [
            x
            for x in display_columns
            if x in category_df.columns
        ]

        temp = category_df[
            available
        ].copy()

        report.append(
            temp.to_markdown(
                index=False
            )
        )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    report.append("")

    report.append(
        "## Important Interpretation Notes"
    )

    report.append("")

    report.append(
        "1. API rate-limit failures are reported "
        "separately and are not treated as retrieval "
        "quality failures."
    )

    report.append("")

    report.append(
        "2. Retrieval precision/recall/MRR/MAP/nDCG "
        "require ground-truth product names. "
        "Questions without expected products are "
        "excluded from those metrics."
    )

    report.append("")

    report.append(
        "3. nDCG is mathematically bounded between "
        "0 and 1 in this evaluator."
    )

    report.append("")

    report.append(
        "4. Faithfulness and correctness require "
        "an independent evaluation judge and are "
        "not fabricated when the judge is disabled."
    )

    report.append("")

    report.append(
        "5. Ten questions are useful for a smoke test "
        "but are not sufficient for a statistically "
        "strong benchmark."
    )

    with open(
        OUTPUT_DIR / "REPORT.md",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(report)
        )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("EVALUATION FINISHED")
    print("=" * 72)

    print(
        f"\nSuccessful: "
        f"{len(successful)}/10"
    )

    print(
        f"Failed: "
        f"{len(failed)}/10"
    )

    print(
        f"Error rate: "
        f"{metrics['error_rate']:.2%}"
    )

    print()

    print("RETRIEVAL:")

    for column in [
        "hit_rate@5",
        "precision@5",
        "recall@5",
        "ndcg@5",
        "mrr",
        "map",
    ]:

        value = metrics.get(
            column
        )

        if value is None:

            print(
                f"  {column:<20} N/A"
            )

        else:

            print(
                f"  {column:<20} "
                f"{value:.4f}"
            )

    print()

    print("GENERATION:")

    for column in [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "hallucination_rate",
    ]:

        value = metrics.get(
            column
        )

        if value is None:

            print(
                f"  {column:<20} N/A"
            )

        else:

            print(
                f"  {column:<20} "
                f"{value:.4f}"
            )

    print()

    print("SYSTEM:")

    print(
        f"  {'Answer generation':<20}"
        f"{metrics.get('answer_generation_rate', 0):.2%}"
    )

    print(
        f"  {'Retrieval success':<20}"
        f"{metrics.get('retrieval_success_rate', 0):.2%}"
    )

    print(
        f"  {'Average latency':<20}"
        f"{metrics.get('latency_seconds', 0):.3f}s"
    )

    print(
        f"  {'Rate-limit errors':<20}"
        f"{metrics.get('rate_limit_errors', 0)}"
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "Open:"
    )

    print(
        OUTPUT_DIR / "REPORT.md"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
````
