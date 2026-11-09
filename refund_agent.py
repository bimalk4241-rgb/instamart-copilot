
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import mock_llm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT = Path("outputs")
STATE = Path("state")
OUTPUT.mkdir(exist_ok=True)
STATE.mkdir(exist_ok=True)

DB = "data/instamart.db"
LEDGER = OUTPUT / "refund_ledger.jsonl"
RUNS = OUTPUT / "agent_runs.jsonl"

# Build the same policy retrieval approach used in Part D
chunks = []
for file in sorted(Path("data/policies").glob("*.txt")):
    words = file.read_text(encoding="utf-8").split()
    for start in range(0, len(words), 30):
        part = words[start:start + 40]
        if part:
            chunks.append({
                "source": file.name,
                "text": " ".join(part)
            })
        if start + 40 >= len(words):
            break

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform([c["text"] for c in chunks])


def search_policy(query):
    scores = cosine_similarity(
        vectorizer.transform([query]), matrix
    )[0]
    indices = scores.argsort()[::-1][:3]
    return {
        "results": [chunks[i] for i in indices]
    }


# TASK 5: AGENT TOOLS

def get_order(order_id):
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """SELECT order_id, customer_id, status,
                      order_amount, delivery_minutes, order_ts
               FROM orders WHERE order_id = ?""",
            (order_id,)
        ).fetchone()

    if row is None:
        return {"error": "order not found"}

    return dict(row)


def calculate_refund(order_amount, delivery_minutes):
    amount = float(order_amount)
    minutes = (
        float(delivery_minutes)
        if delivery_minutes is not None else 0
    )
    refund = amount if minutes > 45 else 0.0
    return {"refund_inr": round(refund, 2)}


def write_ledger(action, order_id, refund_inr):
    entry = {
        "action": action,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def issue_refund(order_id, refund_inr):
    order = get_order(order_id)

    if "error" in order:
        return order

    expected = calculate_refund(
        order["order_amount"], order["delivery_minutes"]
    )["refund_inr"]

    if (order["status"] != "completed"
            or expected <= 0
            or abs(expected - float(refund_inr)) > 0.01
            or expected > 1000):
        return {"error": "refund not permitted"}

    # Avoid issuing the same simulated refund twice
    if LEDGER.exists():
        for line in LEDGER.read_text(
            encoding="utf-8"
        ).splitlines():
            entry = json.loads(line)
            if (entry["order_id"] == order_id
                    and entry["action"] == "issue_refund"):
                return {"refund_inr": expected,
                        "already_issued": True}

    write_ledger("issue_refund", order_id, expected)
    return {"refund_inr": expected}


def escalate_to_human(order_id, refund_inr):
    order = get_order(order_id)

    if "error" in order:
        return order

    expected = calculate_refund(
        order["order_amount"], order["delivery_minutes"]
    )["refund_inr"]

    if (order["status"] != "completed"
            or expected <= 1000
            or abs(expected - float(refund_inr)) > 0.01):
        return {"error": "escalation not permitted"}

    if LEDGER.exists():
        for line in LEDGER.read_text(
            encoding="utf-8"
        ).splitlines():
            entry = json.loads(line)
            if (entry["order_id"] == order_id
                    and entry["action"] == "escalate_to_human"):
                return {"refund_inr": expected,
                        "already_escalated": True}

    write_ledger("escalate_to_human", order_id, expected)
    return {"refund_inr": expected}


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

TOOL_SPEC = [
    {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                key: {"type": typ}
                for key, typ in parameters.items()
            },
            "required": list(parameters)
        }
    }
    for name, description, parameters in [
        ("get_order", "Look up an order in SQLite.",
         {"order_id": "string"}),
        ("search_policy", "Retrieve relevant policy text.",
         {"query": "string"}),
        ("calculate_refund", "Calculate eligible late refund.",
         {"order_amount": "number",
          "delivery_minutes": "number"}),
        ("issue_refund", "Record an approved simulated refund.",
         {"order_id": "string", "refund_inr": "number"}),
        ("escalate_to_human", "Refer a large refund to a human.",
         {"order_id": "string", "refund_inr": "number"})
    ]
]


# TASK 6-8: REACT LOOP, FALLBACKS, CHECKPOINTS

def run_agent(user_message, scenario_id,
              max_iterations=6, token_limit=6000,
              time_limit_seconds=10,
              resume=False, crash_after=None):

    checkpoint = STATE / f"{scenario_id}.json"

    if resume:
        saved = json.loads(
            checkpoint.read_text(encoding="utf-8")
        )
        messages = saved["messages"]
        tokens_used = saved["tokens_used"]
        steps = saved["steps"]
        executed = saved["executed_tools"]
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "AGENT. Handle simulated refund requests "
                    "using tools. Follow the refund policy."
                )
            },
            {"role": "user", "content": user_message}
        ]
        tokens_used = 0
        steps = []
        executed = []

    start_time = time.monotonic()
    llm_calls = len([
        s for s in steps if s.get("type") == "llm"
    ])

    def finish(status, final_answer):
        return {
            "scenario_id": scenario_id,
            "status": status,
            "final_answer": final_answer,
            "steps": steps,
            "llm_calls": llm_calls,
            "tokens_used": tokens_used,
            "executed_tools": executed
        }

    def fallback(status):
        return finish(
            status,
            "Human handoff required. "
            f"Reason: {status}. "
            f"Completed tools: {', '.join(executed) or 'none'}. "
            f"Steps completed: {len(steps)}."
        )

    def save_checkpoint():
        checkpoint.write_text(
            json.dumps({
                "run_id": scenario_id,
                "scenario_id": scenario_id,
                "messages": messages,
                "tokens_used": tokens_used,
                "steps": steps,
                "executed_tools": executed,
                "step_number": len(steps)
            }, indent=2),
            encoding="utf-8"
        )

    while True:
        if time.monotonic() - start_time >= time_limit_seconds:
            return fallback("FALLBACK_TIMEOUT")

        if llm_calls >= max_iterations:
            return fallback("FALLBACK_MAX_ITERATIONS")

        if tokens_used >= token_limit:
            return fallback("FALLBACK_TOKEN_LIMIT")

        response = mock_llm.chat(messages, tools=TOOL_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens

        message = response.choices[0].message

        steps.append({
            "type": "llm",
            "content": message.content,
            "tool_calls": [
                {"id": tc.id, "name": tc.name,
                 "arguments": tc.arguments}
                for tc in message.tool_calls
            ],
            "tokens": response.usage.total_tokens
        })

        if tokens_used > token_limit:
            return fallback("FALLBACK_TOKEN_LIMIT")

        if time.monotonic() - start_time >= time_limit_seconds:
            return fallback("FALLBACK_TIMEOUT")

        if not message.tool_calls:
            text = message.content or ""
            status = text.split(":", 1)[0].strip()
            return finish(status, text)

        calls = [
            {"id": tc.id, "name": tc.name,
             "arguments": tc.arguments}
            for tc in message.tool_calls
        ]

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": calls
        })

        for tc in message.tool_calls:
            name = tc.name
            arguments = tc.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if name not in TOOL_MAP:
                observation = {"error": "unknown tool"}
            else:
                observation = TOOL_MAP[name](**arguments)

            executed.append(name)

            messages.append({
                "role": "tool",
                "name": name,
                "tool_call_id": tc.id,
                "content": json.dumps(observation)
            })

            steps.append({
                "type": "tool",
                "name": name,
                "arguments": arguments,
                "observation": observation
            })

            save_checkpoint()

            if (crash_after is not None
                    and len(executed) == crash_after):
                raise RuntimeError("SIMULATED CRASH")


# RUN ALL SEVEN SCENARIOS

scenarios = pd.read_csv(
    "data/raw/agent_scenarios.csv"
)

# Start each independent evaluation with fresh simulated state
for path in [RUNS, LEDGER]:
    if path.exists():
        path.unlink()

for checkpoint in STATE.glob("S*.json"):
    checkpoint.unlink()

print("\nREFUND AGENT SCENARIOS")
print("=" * 60)

results = []

for _, scenario in scenarios.iterrows():
    result = run_agent(
        scenario["message"],
        scenario["scenario_id"]
    )

    results.append(result)

    print(
        scenario["scenario_id"],
        "| Expected:", scenario["expected_outcome"],
        "| Actual:", result["status"],
        "| Calls:", result["llm_calls"],
        "| Tokens:", result["tokens_used"]
    )

    with RUNS.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result) + "\n")


# TOKEN LIMIT AND TIMEOUT DEMOS

s1 = scenarios.iloc[0]

token_demo = run_agent(
    s1["message"],
    "token_limit_demo",
    token_limit=200
)

timeout_demo = run_agent(
    s1["message"],
    "timeout_demo",
    time_limit_seconds=0
)

print("\nTOKEN LIMIT DEMO:", token_demo["status"])
print("TIMEOUT DEMO:", timeout_demo["status"])


# CRASH AND RESUME DEMONSTRATION

print("\nCRASH AND RESUME DEMO")

try:
    run_agent(
        s1["message"],
        "resume_demo",
        crash_after=2
    )
except RuntimeError as error:
    print("Simulated crash:", error)

resumed = run_agent(
    s1["message"],
    "resume_demo",
    resume=True
)

print("Resumed status:", resumed["status"])
print("Tools executed:", resumed["executed_tools"])
print("LLM calls:", resumed["llm_calls"])

print("\nPART D AGENT COMPLETED")
