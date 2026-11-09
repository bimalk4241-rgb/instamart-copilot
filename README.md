
# Instamart Customer Intelligence & Support Copilot

An offline, reproducible Python capstone project covering data
cleaning, SQL analytics, customer segmentation, lapse prediction,
support ticket triage, policy retrieval and refund automation.

All data is synthetic and fictional.

## Overview

The project integrates four components:

- Part A: Data Foundation
- Part B: Customer Segmentation and Lapse Prediction
- Part C: Support Ticket Triage with Guardrails
- Part D: Policy RAG Assistant and Refund Agent

## Reproduce

Run the following commands in order from the repository root.

1. Create a virtual environment:

   `python -m venv venv`

2. Activate it on Windows:

   `.\venv\Scripts\Activate.ps1`

3. Install dependencies:

   `python -m pip install -r requirements.txt`

4. Generate the synthetic data:

   `python generate_data.py`

5. Clean the datasets:

   `python clean_customers.py`

   `python clean_orders.py`

6. Create and validate the database:

   `python create_database.py`

   `python validate_data.py`

7. Run SQL and exploratory analysis:

   `python run_sql.py`

   `python eda_analysis.py`

8. Run customer segmentation and prediction:

   `python part_b.py`

9. Run support ticket triage:

   `python part_c.py`

10. Run policy retrieval and refund agent:

    `python part_d.py`

    `python refund_agent.py`

## Part A

The cleaned dataset contains 12,309 orders, of which 11,110
are completed and 1,199 are cancelled.

Completed-order revenue: INR 2,998,668.80.

The SQLite database contains customers, products and orders.

## Part B

RFM analysis covers 1,175 customers.

K-means clustering uses four customer segments.

PCA explained variance:
- Component 1: 0.634
- Component 2: 0.218

Lapse prediction:
- Logistic Regression F1: 0.648
- Random Forest F1: 0.648
- Balanced Logistic Regression F1: 0.689

The selected Random Forest was saved and successfully reloaded.

## Part C

The support pipeline processes 164 tickets.

- PII substitutions: 278
- Zero-shot accuracy: 85.71%
- Few-shot accuracy: 100%
- JSON classifications: 135
- Keyword recoveries: 12
- Invalid responses: 9
- Temporary failures recovered: 19
- Budget demonstration: 5,899 of 6,000 tokens

## Part D

Policy retrieval:
- Chunks: 15
- Correct document retrievals: 12/12
- Correct policy facts: 12/12
- Correct fallbacks: 2/2
- Zero-overlap facts: 10/12

Refund agent:
- Seven scenarios tested successfully
- Automatic refunds and human escalation
- Token-limit and timeout fallbacks
- Crash recovery and checkpoint resumption

## Limitations and Next Steps

The project uses synthetic data and an offline simulated language
model, so production performance is not established.

The Premium segment uses the full April-September window.
Therefore, the retention list is illustrative and not a
leakage-free prospective evaluation.

Future improvements include more robust input validation,
additional model evaluation and production security controls.
