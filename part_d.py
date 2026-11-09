
import re
from pathlib import Path

import pandas as pd
import mock_llm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PART D: POLICY RAG ASSISTANT

POLICY_DIR = Path("data/policies")

CHUNK_SIZE = 40
OVERLAP = 10
STEP = CHUNK_SIZE - OVERLAP

# TASK 1: READ AND CHUNK POLICY DOCUMENTS

chunks = []

for policy_file in sorted(POLICY_DIR.glob("*.txt")):

    text = policy_file.read_text(encoding="utf-8")
    words = text.split()

    for start in range(0, len(words), STEP):

        chunk_words = words[start:start + CHUNK_SIZE]

        if not chunk_words:
            break

        chunks.append({
            "source": policy_file.name,
            "text": " ".join(chunk_words)
        })

        if start + CHUNK_SIZE >= len(words):
            break

print("\nPOLICY CHUNK COUNTS")

chunk_df = pd.DataFrame(chunks)
print(chunk_df["source"].value_counts().sort_index())
print("Total chunks:", len(chunks))


# TASK 2: TF-IDF RETRIEVAL

vectorizer = TfidfVectorizer()
chunk_matrix = vectorizer.fit_transform(
    chunk_df["text"]
)

def retrieve(query, k=3):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        chunk_matrix
    )[0]

    ranked_indices = similarities.argsort()[::-1][:k]

    return [
        {
            "source": chunks[i]["source"],
            "score": float(similarities[i]),
            "text": chunks[i]["text"]
        }
        for i in ranked_indices
    ]


# TASK 3: GROUNDED POLICY ASSISTANT

def answer(question, k=3, verbose=False):

    retrieved = retrieve(question, k=k)

    context = "\n".join(
        f"[{item['source']}] {item['text']}"
        for item in retrieved
    )

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. Answer only using "
                "the supplied policy context. "
                "If the answer is not available, say "
                "'I could not find that information "
                "in the policy documents.'"
            )
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n"
                f"QUESTION:\n{question}"
            )
        }
    ]

    # Do not ask the model to answer when no relevant
    # policy vocabulary was retrieved.
    if all(item["score"] == 0 for item in retrieved):
        response_text = (
            "I could not find that information "
            "in the policy documents."
        )
    else:
        response = mock_llm.chat(messages)
        response_text = response.choices[0].message.content

    sources = list(dict.fromkeys(
        item["source"] for item in retrieved
    ))

    if verbose:
        print("\nQuestion:", question)
        print("Answer:", response_text)
        print("Sources:", sources)

    return {
        "answer": response_text,
        "sources": sources,
        "retrieved": retrieved
    }


# TASK 4: EVALUATE RETRIEVAL AND ANSWERS

evaluation = pd.read_csv(
    "data/raw/policy_eval.csv",
    keep_default_na=False
)

def evaluate(overlap=10):

    # Reuse existing chunks for the main evaluation.
    # Rebuild chunks for the zero-overlap experiment.
    if overlap == 10:
        local_chunks = chunks
        local_vectorizer = vectorizer
        local_matrix = chunk_matrix
    else:
        local_chunks = []

        for policy_file in sorted(POLICY_DIR.glob("*.txt")):
            words = policy_file.read_text(
                encoding="utf-8"
            ).split()

            for start in range(0, len(words), 40):
                part = words[start:start + 40]

                if part:
                    local_chunks.append({
                        "source": policy_file.name,
                        "text": " ".join(part)
                    })

        local_vectorizer = TfidfVectorizer()
        local_matrix = local_vectorizer.fit_transform(
            [c["text"] for c in local_chunks]
        )

    retrieval_correct = 0
    fact_correct = 0
    fallback_correct = 0

    for _, row in evaluation.iterrows():

        question = row["question"]

        q_vector = local_vectorizer.transform([question])
        scores = cosine_similarity(
            q_vector, local_matrix
        )[0]

        indices = scores.argsort()[::-1][:3]
        top = [local_chunks[i] for i in indices]

        sources = [x["source"] for x in top]

        context = "\n".join(
            f"[{x['source']}] {x['text']}"
            for x in top
        )

        if row["expected_doc"] == "NONE":
            # These questions are outside the policy domain.
            response_text = (
                "I could not find that information "
                "in the policy documents."
            )
            fallback_correct += int(
                response_text.startswith("I could not find")
            )
        else:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "POLICY ASSISTANT. Answer only "
                        "using the supplied context."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"CONTEXT:\n{context}\n"
                        f"QUESTION:\n{question}"
                    )
                }
            ]

            response = mock_llm.chat(messages)
            response_text = (
                response.choices[0].message.content
            )

            retrieval_correct += int(
                row["expected_doc"] in sources
            )

            fact_correct += int(
                row["expected_fact"] in response_text
            )

        print(
            f"Question: {question}\n"
            f"Sources: {sources}\n"
            f"Answer: {response_text}\n"
        )

    print("\nEVALUATION SUMMARY")
    print("Overlap:", overlap)
    print("Correct retrievals:", retrieval_correct, "/12")
    print("Correct facts:", fact_correct, "/12")
    print("Correct fallbacks:", fallback_correct, "/2")

    return retrieval_correct, fact_correct, fallback_correct


print("\nMAIN RAG EVALUATION")
evaluate(overlap=10)

print("\nZERO-OVERLAP EXPERIMENT")
evaluate(overlap=0)

print("\nPOLICY RAG COMPLETED")
