# ML Challenge 2026: Business Entity Resolution

**Team Name:** Code Red  
**Team Members:** [Add names]  
**Submission Date:** 2026-09-25

## 1. Executive Summary

Code Red uses disk-backed blocking followed by supervised pair classification.
Names and addresses are normalized without country-specific assumptions, candidate
pairs are generated from exact normalized-name, name-prefix, and numeric-address
indexes, and a gradient-boosted classifier predicts whether each pair is a match.

## 2. Methodology

### 2.1 Problem Analysis

The training set contains 2,206,821 S1 entities. S2 and S3 contain missing
addresses, while S1 addresses and names are complete in the initial audit. Train
contains US and India; test additionally contains France, so country is treated
as a normalized string feature rather than a fixed category.

### 2.2 Solution Strategy

**Approach Type:** Blocking + supervised pair classifier  
**Core Innovation:** A SQLite index keeps blocking feasible at multi-million-row
scale while preserving the exact candidate set used for inference.

## 3. Candidate Generation (Blocking)

Normalized S2/S3 records are indexed by normalized name plus country, the first
four characters of the first normalized name token plus country, and extracted
numeric address tokens plus country. The union of these lookups is capped at 30
candidates per S1 entity by default. Training blocking recall must be measured
before final submission; the cap should be increased if true matches are lost.

## 4. Matching Model

Features include character edit similarity, token-sort name similarity, name token
Jaccard, address edit similarity, address token overlap, numeric-token equality,
country equality, missing-address flags, and name-token count difference.

The model is LightGBM with a scikit-learn histogram gradient-boosting fallback.
Pairs are sampled by stable S1 hash, and the validation split is by S1 entity.
The threshold is selected by sweeping probabilities to maximize macro $F_{0.5}$,
including singleton entities.

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Filled by the pipeline's validation output.
- **Common false positives:** To be measured from validation hard negatives.
- **Common false negatives:** To be measured from blocking-recall and threshold reports.

## 6. Conclusion

The pipeline separates scalable candidate generation from learned matching and
keeps the submission outputs reproducible. Validation metrics and error buckets
should be recorded after the first full run before increasing candidate breadth.

## Appendix

The entry point is `code/business_entity_resolution/src/pipeline.py`.
Dependencies and exact reproduction instructions are in its sibling
`requirements.txt` and `README.md`.