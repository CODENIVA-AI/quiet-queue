# CODENIVA AI — Quiet Queue 🦥

**Hacktoberfest 2026 / DEV Week 3 — Slow & Steady**

BY DIRECTOR RASHED ALMAYYAS · POWERED BY CODENIVA AI

Quiet Queue is a **new, standalone experimental prototype (Oct 24, 2026)** under the CODENIVA AI vision. It does not claim the larger platform is finished. It turns a messy local list of messages/tasks into a calm daily action brief — **without sending private data to external AI APIs**.

## What it does

1. Classifies short English or Arabic items with a **locally trained scikit-learn TF–IDF + logistic regression classifier** into `deadline`, `reply`, `review`, and `later`.
2. Elevates **explicit ISO due dates** that are within two days or overdue.
3. Detects **probable duplicate items** using local character n-gram TF–IDF vectors and cosine similarity.
4. Sorts actions by priority; exports a human-readable Markdown brief and optional machine-readable JSON.
5. Takes **no automated action**. Human review is required. It never sends email, archives, deletes, or reads Gmail.

This is **open-source AI / local ML inference**, not an LLM. It is a deliberately small interpretable model using the open-source `scikit-learn` framework. Its tiny synthetic training set cannot guarantee accurate classification; predictions are suggestions, not decisions. Never paste secrets into a public demo.

## Fast start (free; Python 3.10+)

```bash
python -m venv .venv
# Activate .venv as appropriate for your operating system
pip install scikit-learn
python cli.py examples/sample_inbox.json --today 2026-10-24 --out brief.md --json-out brief.json
python -m unittest discover -s tests -v
```

Optional local browser interface:

```bash
pip install gradio
python web_demo.py
```

Open the displayed `http://127.0.0.1:7860` or whichever local port is reported. The Gradio demo uses `share=False`, so it does not create a public link.

## Input

A local JSON array of up to 200 items:

```json
[{"id":"1", "title":"Deadline", "body":"Submit before 2026-10-25", "due_date":"2026-10-25"}]
```

`due_date` is optional. For safety, dates are interpreted only in `YYYY-MM-DD` ISO format, not from ambiguous natural-language dates.

## Privacy, fairness and limits

- Completely local predictions and embeddings. No cloud API keys, no paid subscriptions, no outbound requests from the app.
- Uses *synthetic* demo data; does **not** include real emails or customer data.
- Never takes destructive actions. Duplicate matches are suggestions, not merges.
- Small hand-authored training data causes calibration and generalization limits. Human approval is mandatory.
- Does not impersonate a human, claim external account access, or guarantee action correctness.
- `today` must be set intentionally. Timezones are not auto-inferred.

## Why open innovation matters

The core model can be inspected, retrained, and run without per-request cost. Users can add training examples in their own language or swap in a larger permitted local model later, without surrendering personal inbox content to a proprietary cloud service. The same local triage component could one day be integrated into a broader CODENIVA AI orchestration platform after privacy/security testing.

## Testable example

```bash
python cli.py examples/sample_inbox.json --today 2026-10-24 --out brief.md
```

The included sample contains six synthetic records, including one identical pair and one near-term deadline. See `assets/` for a short offline demo animation and `examples/expected_brief.md` for the captured real output from the CLI.

## Attribution

- Original project code and synthetic examples: **BY DIRECTOR RASHED ALMAYYAS**, developed with ChatGPT assistance, Oct 24, 2026.
- Open-source library: scikit-learn (BSD-3-Clause) with its transitive dependencies.
- No third-party code was copied into this repository.

## License

MIT for the original project code. Third-party dependencies retain their own licenses.

## Contest-status warning

This folder/repository alone **is not a submitted DEV Challenge entry**. A valid entry also requires publishing the official Week 3 DEV template post including `#devchallenge` and `#hf26challenge`, linking source code and an accessible demo, during Oct 19–25 2026 PDT. Eligibility and final receipt are subject to the contest organizer.
