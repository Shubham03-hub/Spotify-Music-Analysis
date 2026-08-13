# Business Understanding

## Business Problem
A&R teams, playlist curators, and label marketing teams currently decide
which tracks to push, promote, or shelve largely on gut feel and
after-the-fact streaming numbers. By the time a track's popularity score is
known, the promotional window has often already closed. There's no
systematic way to estimate a new or recent track's commercial potential
before it accumulates a full listening history, or to tell which genres are
gaining vs. losing momentum.

## Problem Statement
Given a track's audio characteristics (danceability, energy, tempo,
valence, etc.), metadata (genre, duration, explicit flag, release timing),
and artist context (career stage, follower count, star power), predict:
1. A continuous **Popularity Score** (0-100) — regression
2. A discrete **Success Tier** (Flop / Niche / Mid / Hit / Viral Hit) — classification
3. **Genre-level trend direction** (rising / stable / declining)

## Current Challenges
- Popularity is observed only after release and real listener exposure.
- Genre performance is analyzed qualitatively, with no repeatable process
  for spotting a rising micro-genre before it peaks.
- Marketing spend is allocated retroactively to already-popular tracks.
- Small dataset per genre/artist makes naive modeling prone to overfitting.

## Why Organizations Care
Labels, distributors, and playlist curators all compete for a shrinking
pool of listener attention. A system that flags high-potential tracks
earlier — and genres on the rise — directly informs A&R signing decisions,
marketing budget allocation, and playlist placement strategy.

## Use Cases
- A&R: score unreleased/pre-release tracks to prioritize promotional spend.
- Playlist curation: rank candidate tracks by predicted tier.
- Catalog strategy: identify which genres to invest in based on trend direction.
- Marketing: explain *why* a track is predicted to perform well.

## KPIs / Success Metrics
- Regression: RMSE/MAE, target R² >= 0.55 given a modest dataset. Achieved: 0.70 (CV).
- Classification: macro-F1 (rewards not ignoring rare Hit/Viral Hit classes). Achieved: 0.65 (CV).
- Trend analysis: validated against genre_dictionary's human-labeled ground truth — 87% agreement.
- Adoption (business-side): % of A&R shortlist decisions referencing the tool's score.

## Assumptions
- `popularity` is a fair, comparable proxy for commercial success across genres and time.
- Audio features are computed consistently across the catalog.
- The 1,702-track sample is representative enough for genre-level trend inference.

## Constraints
- Small dataset (1,702 tracks, 130 artists) → higher variance risk; mitigated
  with cross-validation and regularized/simple models.
- Severe class imbalance on Success Tier (Viral Hit = 4 tracks pre-cleaning)
  → macro-F1 and Viral Hit merged into Hit for CV validity (see `docs/architecture.md`).
- No per-track time series (only a static popularity score) → genre trend
  is inferred cross-sectionally by release year, not longitudinally per track.

## Risks
- Overfitting given dataset size — mitigated with k-fold CV and simple baselines.
- Popularity is influenced by factors not in this data (marketing spend,
  playlist placement, virality events) — framed as a prioritization aid,
  not a ground-truth oracle.
- Class imbalance risk of defaulting to majority-class predictions — evaluated explicitly per class.

## Expected ROI / Cost-Benefit
Low build cost (open-source stack, no paid infra) against the upside of
shifting even a small percentage of marketing spend toward higher-
probability-of-success tracks — high leverage given the high fixed
promotional cost per track in the music industry.

## Executive Summary
This project builds a track-and-genre intelligence system on Spotify-style
catalog data. It delivers a popularity score, a success tier classification,
and genre trend signal through a governed ML pipeline (validated data →
engineered features → tracked experiments → registered models) surfaced in
an interactive Streamlit dashboard for A&R and marketing stakeholders.
