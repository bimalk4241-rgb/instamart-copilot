
import re
import json
import time
import random
from pathlib import Path
from collections import Counter

import pandas as pd
import mock_llm

Path("outputs").mkdir(exist_ok=True)

tickets = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

segments = pd.read_csv("outputs/customer_segments.csv")

ALLOWED = {"billing", "delivery", "technical", "fraud"}
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05

PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"\b[6-9]\d{9}\b"),
    "order_id": re.compile(r"ORD-\d{6}")
}

PLACEHOLDERS = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "order_id": "[ORDER_ID]"
}


# TASK 1 & 2: INPUT GUARDRAILS AND PII REDACTION

def redact(text):
    counts = {}
    for name, pattern in PATTERNS.items():
        text, count = pattern.subn(
            PLACEHOLDERS[name], text
        )
        counts[name] = count

    # Verify that no original PII pattern remains
    assert all(
        pattern.search(text) is None
        for pattern in PATTERNS.values()
    )

    return text, counts


print("\nPII REDACTION AUDIT")

total_counts = Counter()

for text in tickets["text"]:
    _, counts = redact(text)
    total_counts.update(counts)

print(dict(total_counts))
print("Total substitutions:", sum(total_counts.values()))


# TASK 3: PROMPTS

ZERO_SHOT = """
CLASSIFY customer support tickets.
Allowed categories: billing, delivery, technical, fraud.
Reply with JSON only:
{"category": "...", "reason": "..."}
"""

FEW_SHOT = """
CLASSIFY customer support tickets.
Allowed categories: billing, delivery, technical, fraud.
Reply with JSON only:
{"category": "...", "reason": "..."}

EXAMPLES:
Ticket: Unauthorized transaction. I was charged for an order
I did not place.
Answer: {"category": "fraud", "reason": "unauthorized charge"}

Ticket: The app crashes when I make a payment.
Answer: {"category": "technical", "reason": "app problem"}

Ticket: My invoice has the wrong amount.
Answer: {"category": "billing", "reason": "billing issue"}

Ticket: The rider arrived very late.
Answer: {"category": "delivery", "reason": "late delivery"}
"""


# TASK 4: RETRIES WITH EXPONENTIAL BACKOFF


def call_with_retry(messages, ticket_id, token_budget=None,
                    current_tokens=0):

    for retry in range(4):

        # Check the token budget before each model call
        if token_budget is not None:

            estimated_prompt = sum(
                mock_llm.est_tokens(m["content"])
                for m in messages
            )

            if current_tokens + estimated_prompt + 100 > token_budget:
                return None, "budget_exhausted"

        try:
            response = mock_llm.chat(messages)
            return response, None

        except mock_llm.TransientLLMError:

            if retry == 3:
                return None, "model_error"

            wait = (
                BASE_DELAY
                * (2 ** retry)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"Retry: {ticket_id}, "
                f"attempt={retry + 1}, "
                f"wait={wait:.4f}s"
            )

            time.sleep(wait)

    return None, "model_error"



# TASK 5: VALIDATE MODEL OUTPUT

def validate_answer(content):
    try:
        data = json.loads(content)

        if (
            isinstance(data, dict)
            and data.get("category") in ALLOWED
        ):
            return "classified_json", data["category"]

    except (json.JSONDecodeError, TypeError):
        pass

    # Keyword recovery for non-JSON responses
    words = set(
        re.findall(r"[a-z]+", (content or "").lower())
    )

    matches = words.intersection(ALLOWED)

    if len(matches) == 1:
        return "classified_keyword", next(iter(matches))

    return "flagged_invalid_output", ""


# TASK 6: CLASSIFICATION PIPELINE

def run_pipeline(prompt, token_budget=None):

    mock_llm.reset()
    random.seed(42)

    results = []
    total_tokens = 0
    classified_count = 0
    correct_count = 0

    for _, ticket in tickets.iterrows():

        ticket_id = ticket["ticket_id"]
        text = str(ticket["text"])

        status = ""
        category = ""
        tokens_used = 0

        if not text.strip():
            status = "skipped_empty"

        elif mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"

        elif (
            token_budget is not None
            and total_tokens >= token_budget
        ):
            status = "budget_exhausted"

        else:
            clean_text, _ = redact(text)

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": clean_text}
            ]

            response, error = call_with_retry(
                messages,
                ticket_id,
                token_budget,
                total_tokens
            )

            if error:
                status = error
            else:
                tokens_used = response.usage.total_tokens
                total_tokens += tokens_used

                content = response.choices[0].message.content

                status, category = validate_answer(content)

                if status in {
                    "classified_json",
                    "classified_keyword"
                }:
                    classified_count += 1

                    if category == ticket["true_category"]:
                        correct_count += 1

        results.append({
            "ticket_id": ticket_id,
            "customer_id": ticket["customer_id"],
            "status": status,
            "category": category,
            "tokens_used": tokens_used
        })

    output = pd.DataFrame(results)

    accuracy = (
        correct_count / classified_count
        if classified_count else 0
    )

    print("\nSTATUS COUNTS")
    print(output["status"].value_counts())

    print("Classified:", classified_count)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Total tokens:", total_tokens)

    return output


print("\nZERO-SHOT EVALUATION")
zero_results = run_pipeline(ZERO_SHOT)

print("\nFEW-SHOT EVALUATION")
few_results = run_pipeline(FEW_SHOT)


# TASK 7A: TOKEN BUDGET DEMONSTRATION

print("\nTOKEN BUDGET DEMONSTRATION")

budget_results = run_pipeline(
    ZERO_SHOT,
    token_budget=6000
)

budget_results.to_csv(
    "outputs/triage_budget_demo.csv",
    index=False
)


# TASK 7B: FINAL ROUTING

print("\nFINAL FEW-SHOT ROUTING")

final_results = run_pipeline(
    FEW_SHOT,
    token_budget=100000
)

segment_lookup = segments.set_index("customer_id")

def route_ticket(row):

    status = row["status"]
    category = row["category"]
    customer_id = row["customer_id"]

    if customer_id in segment_lookup.index:
        segment = segment_lookup.loc[customer_id]
        segment_name = segment["segment_name"]
        premium = bool(segment["is_premium"])
    else:
        segment_name = ""
        premium = False

    if status not in {
        "classified_json",
        "classified_keyword"
    }:
        return pd.Series({
            "segment_name": "",
            "priority": "NORMAL",
            "route": "human_review"
        })

    if category == "fraud":
        priority = "HIGH"
        route = "security_team"

    elif premium:
        priority = "HIGH"
        route = "premium_desk"

    else:
        priority = "NORMAL"
        route = "support_queue"

    return pd.Series({
        "segment_name": segment_name,
        "priority": priority,
        "route": route
    })


routing = final_results.apply(
    route_ticket,
    axis=1
)

final_results = pd.concat(
    [final_results, routing],
    axis=1
)

final_results = final_results[
    [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used"
    ]
]

final_results.to_csv(
    "outputs/triage_results.csv",
    index=False
)

print("\nFINAL ROUTING COUNTS")
print(final_results["route"].value_counts())

print("\nPART C COMPLETED")
