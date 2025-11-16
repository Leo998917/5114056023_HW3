# Proposal 001 — Add basic message classifier

## Summary
Add a simple rule-based spam classifier inside hw3.py.

## Motivation
We want hw3.py to have basic functionality as a starting point.

## Technical Plan
- Add function: classify_message(text)
- Rules:
  - If text contains: "buy now", "free", "win", return "spam"
  - Otherwise return "ham"
- Add a demo test in __main__
