# Guarantees and trade-offs — RVG 0.1.0.dev53

This document states what RVG upholds and what it gives up in return. It is written for 0.1.0.dev53. The code is the same as 0.1.0.dev46 (the decision path is the same as dev44), and the policy is sha256 `24777fc8…`. In dev47 the measurements of EVID-1 (2026-10-22) filled in the same-day latency ratios, the error rates under adjudicated reading, and the long-answer values of the current version. In dev53 correction notes were added to the accuracy comparison with plain RAG, from the same-coverage selective baselines of EVID-2 (2026-10-23) (code and policy are the same as dev47). The second commit of the same dev53 documentation version added correction notes from EVID-3, 3b, 4, 5 and 5b (2026-10-23) (code and policy unchanged).

The Korean original is [GUARANTEES.ko.md](GUARANTEES.ko.md). This English text is canonical; both carry the same numbers (`scripts/check_doc_numbers.py`).

**How to read it.**
- **There are two kinds of statements.**
  - **Structural guarantees** (§1) are properties the code is built to have; the test suite keeps them.
  - **Measured properties** (§2, §3) are values measured on particular data pools with a particular model combination. They do not certify that other KBs, questions or models give the same values.
- **Every number carries its experiment ID and date.** The experiment reports are in `docs/…` of the private research repository in which RVG was developed; experiment IDs name those reports, which are not public. The evidence table with denominators, intervals, graders and runs is the private research repository report `docs/RVG_PRODUCT_CONTRACT.md`. This document only carries numbers from existing reports; nothing was measured anew.
- **The baseline is plain RAG with the same model.** Claude Haiku answers directly from the top 10 (BH10) or 3 (BH3) retrieval results, with no controller. RVG uses the same Haiku as its writer and releases only what the controller (Jev judgement) confirmed.
- **Measured model combination.** Writer and extractor: the subscription Claude Code CLI `claude-haiku-5-5`; judge: OpenRouter Jev `typesafe/jev-1.14`; embeddings: OpenRouter Gemini. With other models, §2 and §3 must be measured again.

> **Correction (D-DIM-DECL, 0.1.0.dev56, PROD-1, 2026-10-24).** The existing sentences stay; this is added.
> - Every measurement in this document uses 1,536-dimensional embeddings. `models.embedding_dimensions: 3072` in the policy file is a declared field that the code does not read. The embedding space is the provider space of the settings and of the deployment's store.
> - From dev56, `RuntimeSession.open` and the CLI only log a warning when the declaration and the actual space dimensions differ. Results, settings hash, policy hash (`24777fc8…`) and calls are unchanged. From 0.1.0 this note is logged once per process at INFO (not visible unless logging is configured).
> - A fix that ties the declaration to actual behaviour changes the policy hash, so it is a candidate for a separate contract (CHANGELOG dev56).
> - **0.1.0 (PUB-2):** the default of `Settings.embedding_dimensions` changed from 3,072 to 1,536. The measured configuration is the default configuration. The policy file's 3,072 declaration stays, to keep the hash.

> **First public release 0.1.0 (2026-10-27).** The policy and the measured values of this document are those of 0.1.0.dev58. The code differs in the retrieval score computation (addition below) and in the fixes from the pre-release developer test (PUB-2: no draft values in held results §1, the decision cache records by default §6, default embedding dimensions 1,536 and more — CHANGELOG 0.1.0). The reports this document cites ("private research repository report `docs/…`") belong to the private research repository; the measurements will be released with the paper.
>
> **Addition (0.1.0, 2026-10-27) — platform-independent ranking.** Dense search scans with BLAS, then recomputes without BLAS, in a fixed order, only the rows that can enter the top k (within twice the proven error bound of the k-th scan score); those values decide the ranking and the scores. The fixed order rounds each product and adds the products with numpy's pairwise sum; the query is normalised in the same order. The same stored vectors and the same query give the same ranking and scores, to the bit, on any CPU and BLAS. Ties are broken in catalog order (`(id, revision)` ascending). Before, the ranking was taken directly from BLAS scores, so on another machine the last bit, and the order of rows with nearly equal scores, could differ. The measured values of this document do not change. Scores differ only in the last few digits, and the N0 replay gate and the EVID-4 audit are as before (CHANGELOG 0.1.0).
>
> **Addition (0.1.0.dev58 documentation version, code dev57, 2026-10-27).** The decision cache was adopted opt-in by the user's decision. §6 gains that section and §8 a rollback line. Defaults, policy (`24777fc8…`) and the other sections are unchanged. The other numbers of this document are those of the default path with the cache off.

## Summary

| | RVG (current defaults) | Same-model plain RAG (BH10) | Evidence |
|---|---|---|---|
| Superseded value released as the current one on real revision data | **0** (150 released of fresh 200) | 12 (v3.1-J grading, 48 released) | U31, 2026-10-21 |
| Harmful releases on 500 unanswerable questions | **32** | about 73.6 (estimate) | U20, U32 / U31 |
| Release error rate on static KBs (same consensus reading) | **4.6%** (14/304) | 7.8% (29/370) | EVID-1, difference −3.2 pp (95% −5.8 to −0.8) |
| Correct releases on static KBs (per 100 answerable questions, reading) | 72.5 | 85.25 | EVID-1, distance −12.75 |
| Correct releases of long answers (per 100) | ClapNQ 59.4 (current version) | 91.5 | EVID-1 / U27-b, distance −32.1 |
| Request latency (same day, same requests) | p50 about 2×, p95 about 3× | 1 | EVID-1 (that day 16.6 s / 40.8 s vs 8.4 s / 13.9 s) |

In one line: **RVG almost never releases a superseded value or an unsupported assertion; in exchange it holds answerable questions more often and is slower.** Most of the loss is holding (not releasing), not wrong answers. As the table shows, RVG releases fewer correct answers (72.5 vs 85.25 on static KBs).

> **Correction (EVID-2, 2026-10-23).** The existing sentences stay; this is added. The evidence is the private research repository report `docs/EVID_2_REPORT.md`.
> - **(a) This compares two points of different coverage.** In the error-rate comparison above (RVG 4.61% vs BH10 7.84%), RVG released 306 (76.5%) of the 400 static gold questions and BH10 released 372 (93%).
> - **(b) At the same coverage they cannot be told apart.** At the same coverage (76.5%), the error rates of the same-model plain RAG (BH10) with post-hoc selection, and the 95% intervals of the difference, are as follows. All three intervals include 0.
>   - RVG's claim judge applied after the fact: 4.28% (−0.33 pp [−2.34, +1.98])
>   - A single Jev judgement: 4.93% (+0.33 pp [−1.65, +2.44])
>   - The writer's (Haiku) own confidence: 3.44% (−1.17 pp [−3.02, +0.71])
>   - At the same error rate (4.61%) the baseline answers 82–83%.
> - **(c) On long-answer ClapNQ the baseline is better.** At the same coverage (75/200), the baseline's releases on answerable questions have 0 wrong; RVG has 63 correct and 3 wrong. Unfiltered BH10 also has a lower error rate, 2.0%, than RVG's 4.5%.
> - **(c′) The harmful releases on unanswerable questions (32 vs about 73.6) are also a comparison at different coverage.** At the same coverage (101/500), the harm of the post-hoc selected baselines is roughly 18–23 (sample reading, correction in §2.2).
> - **(d) What RVG guarantees is not an accuracy advantage at the same coverage.**
>   - Temporal validity: 0 superseded values on real revision data. Plain RAG has 12 of 48 releases, and post-hoc selection does not reach RVG's coverage (75%) (§2.1).
>   - Pre-release checks: permission, revocation, scoped fence (§1)
>   - Replayable decision records
>
> **Further correction (EVID-3, 3b, 4, 5, 5b, 2026-10-23).** The existing sentences and the correction above stay; this is added. The evidence is the private research repository reports `docs/EVID_3_REPORT.md`, `EVID_3B_REPORT.md`, `EVID_4_REPORT.md`, `EVID_5_REPORT.md` and `EVID_5B_REPORT.md`.
> - **(e) The 0 superseded values on real revision data are the effect of version and validity metadata and their rules.** Read the first row of the table above, and "plain RAG has 12 of 48 releases" in (d), as follows.
>   - Adjudicated by reading, the superseded values of plain RAG without dates (BH10) are 5/48. Of the 12 that v3.1-J counted, 8 are other wordings of the same fact (EVID-3b).
>   - Given the version dates, sources and a validity rule as instructions, the same model reaches, on real Wikipedia revisions (fresh 200), 0 superseded values by reading, coverage 88–89% (RVG 75%), and 172–173 fully correct answers (RVG 148) (EVID-3).
>   - So "RVG blocks superseded values that plain RAG cannot" is not a guarantee. The advantage of the enforcement showed only in the conflict and condition cases of the synthetic holdout (fully correct answers at the same coverage, RVG +19.5/100, EVID-3).
> - **(f) Error rates are given on two bases** (EVID-5 table 2, EVID-5b). The reference-answer basis keeps the reference answers and original labels as they are; the KB-faithfulness basis removes reference-answer and label defects and errors of the KB itself.
>
>   | | Reference-answer basis | KB-faithfulness basis |
>   |---|---:|---:|
>   | Static KB release error rate | 4.61% (14/304) | 1.64% (5/304) |
>   | ClapNQ release error rate | 4.55% (3/66) | 1.52% (1/66) |
>   | TechQA release error rate | 15.6% (5/32) | 12.5% (4/32) |
>   | U19 harmful release rate on unanswerable questions | 31.7% (32/101) | 16.8% (17/101) |
>
>   - The difference of 15 on U19 is 13 errors in the original data's "no answer" marks (KLUE and SQuAD impossible questions whose original passage answers the question as is) and 2 earlier reading defects. The id list is table 1 of the private research repository report `docs/EVID_5B_REPORT.md`.
>   - Among the remaining errors, 0 releases gave a value different from the cited evidence. The main causes are premise acceptance (U19: 15), completeness misjudgement and target confusion (EVID-5).
> - **(g) The exact meaning of the replay wording is in the correction under item 5 of §1.** The evidence of an exhaustive audit of the structural guarantees over records (EVID-4) is added at the end of §1.

## 1. Structural guarantees (code and tests)

1. **Only what was checked is released.** An answer uses as evidence only KB documents that are permitted and valid at that time. Only the parts the controller confirmed against the evidence are released, as `FULL` or `PARTIAL`. The rest is `ABSTAIN` (or `UNRESOLVED_SCOPE`, `ERROR`). No path releases a sentence that was not checked.
   - **A result that is not released carries no draft value (0.1.0, PUB-2).** The held draft's short answer, its claim sentences (`claims` of `assessment` and `assessment_before_recovery`), the paired draft's short answer (`trace.consistency`), and the text and sentences are removed from the public surface (`UniversalEngine.answer`, the `session.answer` envelope, the CLI). The draft is only in the opt-in handoff packet (`result.handoff`). Before, the `short_answer` of a held result kept an unchecked draft value (found by the pre-release developer test). Evaluation records (`rvg.testing.records.answer_one`) keep the removed values outside the result, in `held_draft`.
2. **One policy.** Every KB and language uses one policy file (`rvg/resources/policies/universal.json`). There are no per-KB or per-language profiles, thresholds, retrieval settings, top-k values or human calibration. Every answer is stamped with the policy hash (`config_hash`).
3. **Revocation and permission are checked right before release.** In every release mode, revocation and the principal's permission are checked again right before release (`release_authorize`).
4. **The scope of the release check (scoped, the default since dev27) is `authority + watermark + cutoff`.**
   - If every change is judged CLEAR (unrelated to what the answer read), then a fresh run from the fence state (the knowledge number and ACL epoch the release check read), with the same principal, question and selection, gets the same seeds and documents in every retrieval.
   - Not guaranteed: group-membership changes, changes outside the state store (writes that bypass the outbox), reproduction of model outputs.
   - A change that cannot be judged (UNKNOWN) falls to holding (the global rule). The detailed definition is in the module description of `rvg.universal.scoped`.
   - If the answer's retrieval read documents that have no vector yet (pending), those documents were ranked by BM25 and there is no exact watermark. Then even a change unrelated to the answer becomes UNKNOWN (`no_exact_watermark`) and the answer is held. This is a hold on the safe side; it appears where a process keeps writing, inserting documents and writing their vectors later (PUB-2: 2 of 10 repeated PostgreSQL runs).
5. **Recording and replay reproduce decisions.** Replaying the same tape (a recording per HTTP attempt) gives byte-identical decisions, released answers and release records.
   - Every version passes two sets of the N0 replay gate (120 requests each) with strict equality.
   - dev44 replayed the 150 requests of dev43's live tape, 150/150 the same (PERF-1, 2026-10-21).
   - **Correction (2026-10-23, evidence map (C) of the paper, EVID-4 table 3).** The exact meaning of "byte-identical" is:
     - Against the reference replay, decisions, released answers, release records and traces are strictly equal.
     - Against the original records, they are equal with the audit keys `provenance` and `tentative` excluded by an allowlist. Compared with the original records as they are, no request is equal, because of differences in audit keys.
     - Reproduction of the model outputs themselves is still outside the guarantee.
   - **Replay across the 0.1.0 result contract (PUB-2).** Records made before 0.1.0 are read through `rvg.testing.records.contract_view`, which applies the same rule (no draft values in held results) to the old results, and are compared in that view; the removed draft values are compared byte for byte with `held_draft` of the new records (n0b 120/120 with held drafts 12/12, s0jev114 120/120 with 19/19).

- **Evidence (EVID-4, 2026-10-23): items 1, 3, 4 and 5 above were audited exhaustively over records.** The checks are code, with no judge.
  - In 1,589 releases (2,182 sentences) under the current policy, the five checks find 0 violations. The five checks: every sentence has a citation; the citation is in the store at release time; it is valid at the reference time; it is visible to the asker; the release record matches the trace and the manifest.
  - Releases made while changes were happening (the current release path, 92 releases, 6 of them with an unrelated change inside the window) also have 0 violations.
  - In the strict replay of 1,152 dev47 requests, every release decision is the same.
  - Limits of the audit: the stores of the change experiments were deleted, so for documents inserted during a run only their presence in the original KB range was checked. Repeated revisions and churn events are not in the request records, so only single-event ranges were checked. There was no static pool with ACLs.

## 2. Measured properties — safety

### 2.1 Time and revisions: no superseded value as the current one

- **Real revision data, fresh 200** (EvolvingQA edited questions + Wikipedia revision history, U31, 2026-10-21).
  - RVG released 150, 148 of them fully correct, and released 0 superseded values (0 under both grader v3.1-J and v2).
  - On the same day BH10 released 48, with 12 superseded values (v3.1-J; 46 under v2).
- **Reading-corrected figures** (U21-C, 2026-10-19, RVG dev17). Superseded-value releases per 100 questions: RVG 0, BH10 0.8 (fresh) and 1.4 (real_full).
  - The 95% intervals of the release-error-rate difference are fresh [−5.9, −0.7] pp and real_full [−9.5, −1.2] pp, lower for RVG.
- **RVG also releases more correct answers here** (same U21-C, per 100 questions). fresh: RVG 61.0 vs BH10 2.5 (rule) to 31.7 (reading-corrected upper bound); holdout: 81.5 vs 17.3 to 63.6. Plain RAG cannot tell the versions apart, so it holds or errs between the old and the new version.
- **The same holds in the current version.**
  - Under the current policy (U32, 2026-10-21) and the dev27 release block (U24, 2026-10-20), the decisions replayed on fresh 200 are the same as the original records, 200/200.
  - In dev44 live (PERF-1, 2026-10-21, fresh 50) superseded values are 0 too (36 released, rule grading v2).
- **Conditions.** Version relations between documents are handled by the policy's prior (document-range overlap θ′ = 0.8, U10, U16) and by showing both versions alternately (U10-c). In synthetic conflict cases where the revision relation is uncertain, 1 assertion remains (N2, a known limit).
- **Correction (EVID-3, 3b, 2026-10-23).** The BH10 comparison above is against plain RAG that did not receive version dates or validity information.
  - Adjudicated by reading, the superseded values of BH10 without dates are 5/48 (8 of the 12 counted by v3.1-J are wording differences, EVID-3b).
  - Given version dates, sources and a validity rule, the same model reaches on fresh 0 superseded values by reading, coverage 88–89% and 172–173 fully correct answers (EVID-3). Read this section as "applying validity metadata and rules deterministically gives 0 superseded values".
  - In the conflict and condition cases of the synthetic holdout, fully correct answers at the same coverage are +19.5/100 on RVG's side. The RAG that received only instructions answered conflicts in which both values are valid with a single value.

### 2.2 Unanswerable questions: fewer unsupported assertions

- **Development set of 500 unanswerable questions** (150 with the gold removed + 350 impossible questions with twisted premises, KLUE and SQuAD 2.0).
  - RVG released 101, with **32** harmful releases (U20, 2026-10-19). Under the current policy it is still 101/32 (U30, U32).
  - BH10 (collected 2026-10-20) released 230, with about **73.6** harmful releases (U31, estimated by sample reading; 95% interval of the impossible part 69: [54.5, 83.5]).
  - BH3 was not measured on this pool.
- **Static nogold 360** (static KB questions with the gold document removed). Harmful releases: RVG 2 (83 released, 1 undecided), BH10 10 (146 released, 10 undecided), BH3 12 (137 released, 11 undecided) (U23, 2026-10-19). Undecided items remain, so the baseline values are lower bounds.
- **ClapNQ, 94 unanswerable questions** (U27-b, 2026-10-20). Release rate: RVG 10.6%, BH10 28.7%. Difference −18.1 pp, 95% interval [−26.6, −10.6]. No reading was done.
- **Correction (EVID-2, 2026-10-23, auxiliary, rough sample estimate).** The harmful-release comparison above also has different coverage (on the 500 unanswerable questions RVG released 101, BH10 230).
  - With post-hoc selection on plain RAG at the same coverage (101), harmful releases are roughly 18–23 (harmful share 18–24% of the 38–44 samples read × 101). Not more than RVG's 32.
  - On static nogold, at the same coverage (83), 0 of the 22–25 samples read were harmful (RVG 2).
  - The baseline harm is a rough sample-reading estimate, a descriptive value not used for verdicts (the private research repository report `docs/EVID_2_REPORT.md` table 4).
- **Correction (EVID-5, 5b, 2026-10-23).** RVG's 32 harmful releases are on the original-label basis (31.7%, 32/101). On the KB-faithfulness basis they are 17 (16.8%).
  - The difference of 15 is 13 errors in the original data's "no answer" marks (the original passage answers the question as is) and 2 earlier reading defects (the private research repository report `docs/EVID_5B_REPORT.md` table 1).
  - Of the remaining 17, 15 are releases that accepted the question's premise (EVID-5): the evidence matches the citation, but the question's premise does not stand on the evidence.
- **Meaning of a harmful release.** A release that accepts the question's premise and asserts an answer, or asserts a value that the remaining documents do not support. Two readers (Opus, Codex gpt-6-astra) read; when they disagree, a 3rd reader decides.

### 2.3 Error rate of released answers

- **Static KB gold 400** (N0 80 + 8 domain KBs 320). All releases of both sides were read with the same consensus reading (Opus → Codex → 3rd reader, blind to the arm) (EVID-1, 2026-10-22).
  - The release error rate of the current policy is **4.61%** (14/304, 2 undecided).
  - BH10 (collected 2026-10-20) has **7.84%** (29/370, 2 undecided).
  - The difference is −3.23 pp, and its 95% interval [−5.75, −0.83] does not include 0 (paired bootstrap over the 400 questions).
- **J1 grading counts about 3 pp more errors on both sides.** By J1 (`jev-1.14` same-answer judgement ≥ 0.55, or the same string) it is 7.52% vs 10.75% (difference −3.24 pp [−6.35, −0.15]). The 7.19% (U32) and 10.8% (U31) of the earlier document version are also J1-based.
  - Among the releases J1 called wrong, the reading judged 8 correct for RVG and 11 for BH10, mostly correct answers worded differently.
- **The earlier "not established" (U21-C, dev17, collected 2026-10-18) is a value of another collection and grading.** Its intervals included 0 (N0 [−10.8, +7.2] pp, domain KBs [−5.7, +0.03] pp).
- **Live sample of the current version.** The release error rate of dev44 on 100 domain-KB gold questions run live is 11.3% (9/80). dev43 had 10.8% (9/83) (PERF-1, 2026-10-21, J1 only, reported only). It is a single collection of 100, so wider than the 400 values above.
- **Correction (EVID-2, 2026-10-23).** The difference above (−3.23 pp) is between two points of different coverage.
  - At the same coverage (76.5%), the error rates of same-model plain RAG + post-hoc selection are 4.28%, 4.93% and 3.44%. They cannot be told apart from RVG's 4.61% (every interval includes 0).
  - So read this section as "at this coverage, this error rate", not "RVG is more accurate". Details are in the correction under the summary and in the private research repository report `docs/EVID_2_REPORT.md`.
- **Correction (EVID-5, 2026-10-23).** The error rates above are on the reference-answer basis. On the KB-faithfulness basis (reference-answer and label defects and errors of the KB itself removed) they are static 1.64% (5/304), ClapNQ 1.52% (1/66), TechQA 12.5% (4/32) (table (f) under the summary).

## 3. Measured properties — trade-offs (the price)

### 3.1 RVG holds answerable questions more often

- **Static KB gold 400.**
  - By consensus reading, correct releases per 100 are **72.5** for the current policy and 85.25 for BH10 (collected 2026-10-20). The distance is **−12.75**, 95% interval [−16.25, −9.5] (EVID-1, 2026-10-22).
  - By J1, the current policy's 284 (71.0) vs 84.0 for the BH10 of the U21-C collection, a distance of −13.0 (U32, `bh10_distance_correct_per_100_gold` in the private research repository's `docs/results/u32.json`).
  - Before the U30 default (`scope_top1`) was added, the policy had −14.0, 95% interval [−17.5, −10.75] (U23, 2026-10-19).
  - Against a BH10 collected again on the same day (332/400) it is also −13.0 (U31, 2026-10-21, pre-U30 policy).
  - **Correction (EVID-2, 2026-10-23).** This distance comes from the coverage difference.
    - At the same coverage (306/400), the post-hoc selected baselines release as many correct answers as RVG (RVG judge applied after the fact 291, writer self-confidence 293.6, RVG 290).
    - At the same error rate, the baseline answers 5–6 more per 100 (the private research repository report `docs/EVID_2_REPORT.md`).
- **The shape of the loss** (U23, pre-U30 policy, per 100).
  - Correct 70.0 vs 84.0, wrong releases 5.25 vs 9.0, holds 24.75 vs 7.0.
  - Most of the remaining distance is first-stage controller holds (U22, U25).
- **The loss is large on long answers.**
  - **ClapNQ** (confirmation sample 200, 106 answerable).
    - The current version (dev46, EVID-1, 2026-10-22) has **59.4** per 100 questions. BH10 (2026-10-20) has 91.5, a distance of **−32.1** (95% interval [−41.5, −23.6]).
    - No difference from the 57.5 of dev27 (U27-b) (+1.9 [−2.8, +7.5]). The default changes since then (U24, U30, U32, PERF-1) did not change the long-answer loss.
    - The main cause of missed correct answers is controller holds (28 questions in dev27). The causes narrowed down to the slot direct-answer judgement and the `answered` threshold (U28).
    - **Correction (EVID-2, 2026-10-23).** At the same coverage (75 of 200 requests), the post-hoc selected baselines answer 66–70 answerable questions correctly with 0 wrong (RVG 63 and 3). On long answers plain RAG is better in accuracy too.
  - **TechQA.**
    - The current version has **21.6** correct releases per 100 questions on confirmation 181 (125 answerable) (32 released, error rate 15.6%, EVID-1).
    - More than the 14.4 of the dev17 configuration (N6-2) (+7.2 [+2.4, +12.8]). Releases on the 56 unanswerable questions also rose from 6 to 12 (a stress indicator, not read).
    - This sample has no same-day BH10. On final 474 it was dev17 15.4 vs BH10 73.2 (−57.8) (U21-C, sample-reading estimate, no interval).

### 3.2 RVG is slower

- **On the same day and the same requests, about 2× plain RAG at p50 and about 3× at p95.**
  - EVID-1 (2026-10-22) ran static gold 100 + real revision fresh 50 alternately through RVG (dev46) and BH10, with 16 concurrent requests, 8 CLI slots and Claude Code 2.1.308.
  - That day RVG had p50 16.57 s / p95 40.84 s, BH10 8.43 s / 13.88 s.
  - The ratios are p50 **1.97** (95% interval [1.81, 2.06]) and p95 **2.94** ([2.37, 3.28]).
- **Absolute values vary with the day and the CLI version.**
  - The day before (PERF-1, 2026-10-21, Claude Code 2.1.306), dev44 on the same requests had p50 11.75 s / p95 26.94 s.
  - On the EVID-1 day the CLI's own time grew, so both arms were slower. API time and judgement time were the same.
  - So this document uses the ratio as the unit of its promise.
- **Plain RAG records of other days.** U27-b ClapNQ 6.1 s / 7.9 s, U31 p50 6.1 s, U21-C 7.4–13.4 s by pool.
- **Most latency is subscription CLI generation.** In PERF-1 stage A (U22, U23 live records, dev18) it was 84–85% of request time.
  - The writer's schema carries evidence ids that differ per request, so the writer cannot be started in advance.
  - A fixed writer schema is a candidate of the private research repository's ROADMAP item 22.

### 3.3 Calls and cost

- **Jev judgements**: 7.6–8.3 per request on average, p50 6, p95 19 (PERF-1 stage A, records of N0 120 and static gold 100). In dev44 live, 807 for 100 static gold requests.
- **Generation (subscription CLI Haiku)**: usually 2 per request (1 writing + 1 extraction), more when the second stage runs. The distribution is N0 {2: 91, 3: 4, 4: 18, 5: 5, 1: 2}/120 (PERF-1 stage A).
  - A request where `scope_top1` fires (24 of 1,652 requests, 1.5%, U30) needs 1 more writing and its judgement.
  - If a document the answer read changes during the release check, the answer is written once more. In the related-change scenario of U24 live, Haiku calls rose by +97 per 100 requests.
- **Paid API (judgement + embeddings)**: about $0.0005 per request (U27-b, 2026-10-20). Plain RAG uses only embeddings, $0.000004. Subscription CLI calls are separate and not in this amount.

## 4. Defaults (current)

| Item | Value | Adopted | Evidence (private research repository) |
|---|---|---|---|
| Release mode `release.mode` | `scoped_dependency`, `retry: 1`, `retry_rule: slack`, `revalidate: false` | dev27 | U24 (2026-10-20), RVI-03 F3 |
| `release.vectors` / `release.vector_moves` | `scoped` / `after_check` | dev27 | RVI-03 F2, FIX-V9-CHURN |
| `release.acl_moves` | `after_check` | dev42 (documented in dev43) | U32 (2026-10-21), first FIX-ACL-WINDOW |
| `cascade.scope_top1` | `dual-draft-v1` | dev42 (documented in dev43) | U32, first U30 |
| `cascade.evidence_window` | `neighbors-v1` | dev17 | N6-2 (2026-10-19) |
| `cascade.consistency` | paired draft `widened`, same-answer judgement θ 0.7 | dev18 | U22 (2026-10-19) |
| Controller `answered` threshold | 0.84 | dev20 (reverted from 0.6) | U20 §8, U23 rollback |
| Writer and extractor | subscription Claude Code CLI `claude-haiku-5-5` | dev16 | U21 (2026-10-18) |
| local exact index layout (with a matrix directory) | `base_delta` (`auto`) | dev41 | FIX-MATRIX-GEN (2026-10-21) |
| Return CLI generation results at once | always | dev44 | PERF-1 (2026-10-21) |
| Start CLI processes in advance, `JsonTransport(claude_code_prewarm=…)` | 2 | dev44 | PERF-1 |

The policy keys are in `universal.json`. Changing the policy changes the hash. Experiment arms (`rvg.universal.experiment.Arm`) are for the evaluation runner only and are not public API. The answer path of `open_session` uses only the policy values, with no arm.

## 5. Storage and operating requirements

- **What the default release check (scoped) reads.** The state store's ACL epoch, documents, revocations, hiding (`hidden`) and release authorization (`authorize_release`) are required features of the port. Beyond them it needs:
  - **An outbox**: document `insert`, `revoke`, `acl` and `vector` rows and a start mark (`outbox_mark`). The SQLite and PostgreSQL state stores have one. A custom store may not.
  - **`Release.hidden`** (U32): the visibility at the fence of the release receipt. Without it, `acl_moves = after_check` behaves like `hold`: if the ACL moves during the check, the answer is held.
  - **An exact-score index**: local exact (`EXACT_CANONICAL`), pgvector and Qdrant (`EXACT_NATIVE`). Approximate indexes and lexical retrieval make the watermark step UNKNOWN.
- **A missing feature falls to the safe side.** The corresponding change becomes UNKNOWN and the answer is held (an availability loss, not a wrong release). Such a store under a heavy concurrent-write load can release far less. It is the same shape as the releases of `current` mode falling to almost 0 in the research repository's N3L.
- **Matrix directory (file stores).** The default layout `base_delta` writes to `v3/base_delta/`. The first search in a new directory does one full build. The base is rewritten whenever the delta exceeds θ (0.10) of the base, so the write amplification is O(θN). At 1 write/s × 4 reads, the peak footprint of the directory was 1.015× (FIX-MATRIX-GEN).
- **Idle CLI processes.** Up to 2 processes started in advance per event loop (pool), about 270 MB each on average, p95 353 MB (PERF-1 A′, Claude Code 2.1.306).
  - A process older than 300 s is reaped at the next call. Without calls it stays until the loop ends (`shutdown_asyncgens`) or the program ends (`atexit`).
  - A process that is ending after returning its result is reaped in the background and killed if it has not ended within 30 s. So the momentary number of processes can briefly exceed the number of CLI slots.
- **CLI version.** Returning results at once and starting processes in advance were checked against the output format of Claude Code 2.1.306 (result JSON, stream-json input). If the format changes, it surfaces as an error, like a failed result (the meaning of timeouts and cancellation is unchanged).

## 6. Adopted but off by default

- **N1F-4 TurboQuant 4-bit candidate stage** (approximate candidates in front of local exact + exact reranking). In dev15, after N1F-4 ADOPT (2026-10-18).
  - **How to turn it on.** Storage setting `"vectors": {"candidate_stage": "turboquant_mse4"}` (local exact only). Build the codes after bulk indexing with `rvg.build_candidate_codes(storage, tenant, space)` (0.1.0 public name; `space` is an `EmbeddingSpace`, or a fingerprint together with `dims`). Without codes: `quantized_codes_missing`.
    - The old internal path `rvg.storage.quantized.build_codes(storage, tenant, fingerprint, dims)` still works, but is outside the stability promise.
  - **Effect.**
    - The top 20 were bit-identical to the exact path (17,114/17,114).
    - Four sets of N0 replay were strict 120/120.
    - Memory drops to 1/16.
    - At 12k rows the search is slower (p50 15.5 ms vs exact 1.05 ms). The gain exists only at fullwiki scale.
  - **Caution.** This index declares its precision `APPROXIMATE`. So with the default release check (since dev27, which became the default after N1F-4), even a new revision the answer did not read becomes UNKNOWN and the answer is held. The code behaves so; it was never measured.
    - Incremental layouts, remote indexes and PostgreSQL in combination with it were not tested either.

- **Handoff packet** (`handoff_packet=True`, dev56). PROD-1 ADOPT (2026-10-24), off by default.
  - **How to turn it on.** `open_session(…, handoff_packet=True)`, `UniversalEngine(…, handoff_packet=True)`, or the CLI `universal-query --handoff-packet`. It is not in the settings or the policy file.
  - **What it does.** It attaches `result.handoff` to results that are not released (`ABSTAIN`, `UNRESOLVED_SCOPE`, `ERROR`). Contents: the held draft and its citations, reason codes, claim→sentence mapping and labels, the evidence text and its validity state, the paired-draft comparison, the writer's refusal conditions and raw output. It is not a release (`released: false`). It does not enter released results, release records or lineage.
  - **What was checked.**
    - With it on, 3,080 recorded requests were replayed; the released results and release records were byte-identical, and the calls were the same too.
    - In 1,489 packets there were 0 unit texts the asker could not see (auditor P1–P4).
    - A unit the asker can no longer see (revocation, permission, exclusion list, selection, validity) carries only its id and state. Everything a stage wrote is withheld as well (tests).
  - **Limits.**
    - The reading precision of the automatic flags (`claim_not_supported`, `answered_below`) is 14.5% [10.3, 20.1], so they carry the note `"automatic flag, low confidence"`. A flag only points to where a person should look.
    - A proxy reading of the packet alone decided 93% (36% with the current result), but it accepted most of the J1-wrong drafts. The packet does not guarantee the correctness of the decision.
    - The size (median) of handed-off results grows 1.5–4.4× depending on the pool. Static N0: 4.8 KB → 20.9 KB, at most 68 KB (TechQA). Replay showed no latency difference, but that measurement ran alongside another collection and was not controlled.

- **Decision cache** (`decision_cache=CacheMode(...)`, code dev57, documented in dev58). Adopted opt-in by the user's decision (2026-10-27); off by default.
  - **The contract verdict stays REJECT.** PROD-2 (the private research repository report `docs/PROD_2_REPORT.md`) missed rule ④ (hit decision agreement ≥ 1 − the pairwise disagreement of the same question asked again) on techqa, fullwiki and tydi-te. The confirmation experiment PROD-3 (the private research repository report `docs/PROD_3_REPORT.md`, a new sample of 3,636) missed ④′(b) on techqa (78.71% < 80.19%) and ⑤ in total (cache J1 7.47% > 7.43%).
  - **How to turn it on.** `open_session(…, decision_cache=CacheMode(DecisionCache(path), …))` or `UniversalEngine(…, decision_cache=…)`. It is not in the settings or the policy file.
    - Store: `rvg.universal.cache.DecisionCache(path)` (a local SQLite file; read-only with `readonly=True`).
    - With `record` (default True), finished decisions (released and held) are written as entries; with `lookup` (default True), they are looked up. For a lookup-only cache pass `record=False`; to only fill one, `lookup=False`. In 0.1.0 the `record` default changed from False to True. Before, `CacheMode(DecisionCache(path))` alone recorded nothing, so nothing was reused (pre-release developer test, PUB-2). The research runner passes `record` and `lookup` explicitly, so the PROD-2 and PROD-3 results reproduce as they were.
    - `theta` (default 0.9): the Jev equivalence-probability threshold. `match` (default `top1_or_overlap`) and `tau` (default 0.5): the candidate key (the same top-1 document, or a first-stage window Jaccard ≥ τ). `max_candidates` (default 5): the number of candidates judged. `generation` (default `same_window`; with `same_snapshot` the knowledge number must be the same too).
    - **How to turn it off.** Do not pass `decision_cache`. Calls, traces and results are then the same as dev56 (N0 replay gate strict 120/120, three versions).
  - **Guaranteed (code and tests).**
    - A release from the cache reuses a recorded decision (draft and citations). The writer and the extractor are not called.
    - Such a release goes out only after passing the checks again for the new request. It may cite only inside the evidence window that passed the current retrieval, validity, ACL, exclusion list and selection (an entry whose cited unit is outside the window is not reused, including withdrawals and permission changes). The controller judges again with the new question (answered, citations, qualifiers). Then the release check (scoped fence) runs.
    - The release record keeps `source: {"cache": {entry, p, theta, match}}`, and the trace `cache` keeps p and the key per candidate, the chosen entry, and the evidence units checked again (`units`). These are subject to the EVID-4 audit C1–C5 (PROD-2 and PROD-3, 26+26 runs, 0 violations; the first PROD-2 run produced 469 C5 findings because the trace did not record the units, and was run again after `units` recording was added).
    - Equivalence judgements and re-check calls go into the tape and replay (200 replayed, 200/200, twice).
    - A held entry returns a hold with the same reason only when the first-stage window is the same.
  - **Not guaranteed.**
    - **Agreement with the fresh path.** The cache reproduces the original decision (hit decisions agree with the original decision 99.5%). So it differs from the decision a fresh generation would make for another sentence of the same meaning by that path's own variation (cache vs fresh 92.7–93.2%). The preregistered rules ④ and ④′(b) required this per KB and were missed.
    - **Entry choice in KBs with many repeated threads.** In a KB like TechQA, where nearly the same question comes in several requests, another request's entry can be chosen (the 2 released↔held flips in PROD-3 were both such cases).
    - **A better correctness rate than fresh generation.** The J1 error rate of cached released answers is 6.43% in PROD-2 (fresh 6.94%) and 7.47% in PROD-3 (fresh 6.43%). The cache inherits the original question's release, and with it its wrong answers.
  - **Operating points (PROD-2, PROD-3, key top1_or_overlap τ 0.2, 5 candidates).** The default τ 0.5 hits within 0.1 pp of the measured τ 0.2 (PROD-2 grid).

    | θ | Hits on paraphrased questions | False hits on other questions (strict) | Of those, releases on another question | Hit-rate variation when Jev is asked again |
    |---|---:|---:|---|---:|
    | 0.95 | 56.9% (PROD-2) · 57.0% (PROD-3) | 0.37% (48/13,000) · 0.31% (20/6,500) | 0 — every released false hit was the same question (the same for Sonnet, or the same on reading) | 4.7% · 4.8% |
    | 0.9 | 71.6% · 72.2% | 0.58% (75/13,000) · 0.51% (33/6,500) | 1 (PROD-2 science: both questions have the answer "Homeostasis", but Sonnet saw different scopes) | 2.0% · 2.0% |

    - A "strict false hit" gives a cached result (released or held) to a question with a different sentence. It includes natural duplicates in the data (two requests that word the same question differently). Byte-identical sentences (data duplicates) were counted separately.
    - The Jev re-asking variation is the hit↔miss variation, estimated from the rate at which the same pair crosses θ when asked again (EVID-8 re-asking sample). A miss takes the usual path, so it is not a wrong answer.
  - **Effect.** 1 hit saves 2.3 writer calls on average. The median request time on the hit path is about 1 s (retrieval embeddings excluded; fresh records of the same requests 6–25 s). A miss adds the lookup judgement (median 0.1–0.5 s).
  - **Measured scope.** 13 KBs (6 languages); the cache was built from the decisions of one run (EVID-7 r1, 12,972 entries). Not measured: the storage rule when the same question is recorded many times, measurements when the KB changes after loading, and the live writer cost of misses.

## 7. Not guaranteed

The following are in the code, but turn on only as arms of the evaluation runner (EXPERIMENTAL_ARM) or are experimental (EXPERIMENTAL). They are not promised as product behaviour.

- Release modes `as_of_snapshot` and `latest_at_release` (switching to latest: N3L-2 REJECT). `current` is for rollback (IMPLEMENTED).
- N2 `unknown_revision` keep_both/tentative, N4 `conflict_partial=rewrite`, N5 `writer_schema` W1/W2 (all REJECT)
- `cascade.writer_contract=form-v1` (dropped in N6-2 development), `cascade.scope_top1=entity-tie-v1` (U30 not met)
- `cascade.answered_threshold` and `write_top` (U23 arms), `release.revalidate: true` (RVI-03 F3 A3c passed its gate; the default is `false`)
- local exact incremental layouts `segments` and `paged` (N1F-2)

Not in this version (only on branches):
- Upstream conflict detection `cascade.upstream_conflict` (RVI-06 line REJECT, used to generate the U27-a grader)
- U26 premise and target gates (INCONCLUSIVE), U25 writer instruction (REJECT), U20 scope anchor (REJECT)

## 8. Rollback

| Change | Version | Rollback |
|---|---|---|
| CLI immediate return and processes started in advance | dev44 | dev43 wheel (`72d8ef37…`). To turn off only the processes started in advance: `JsonTransport(claude_code_prewarm=0)`. There is no switch that turns off only the immediate return |
| `acl_moves` and `scope_top1` defaults | dev42, dev43 | dev41 wheel (`a2a3e14f…`, policy `b85003cd…`) |
| matrix layout `base_delta` | dev41 | storage setting `vectors.layout: rebuild`, or the dev40 wheel (`e1394f8b…`, reads only `v2/`) |
| scoped release check | dev27 | dev26 wheel (`a77625fc…`, policy `5b7d666b…`) |
| decision cache (opt-in) | dev57, dev58 | not passing `decision_cache` turns it off (the same as dev56). The whole package: the dev56 wheel (`31235a0b…`) |

Wheels are kept as reproducible builds in the research repository's `.artifacts/rvi/rvg-runtime-<version>-<commit>/`. This version (dev53) changed only documents from dev47, so rolling back to the dev47 wheel (`d33850c5…`) or the dev46 wheel (`6cf7c884…`) behaves the same. Apart from the `rvg.capabilities()` extension of dev46 (FIX-CAPS) it is also the same as dev44.

## 9. What this document does not say

- **Other model combinations.** For other writers, judges or embeddings there is no evidence that §2 and §3 hold.
- **New holdouts.** Real revision fresh 200 and the static KBs were already used in many experiments. They are not new holdouts.
- **Live collected once.** Most measurements are a single collection per arm. Resampling the writer alone moves static gold correct answers by a few each time (U30 CTL24, PERF-1).
- **Stress load.** In U32 ACL churn live (472 requests), both arms had about 33% deadline-exceeded ERRORs, and the cause has not been confirmed. Availability under continuous concurrent writes is not promised by the numbers of this document.
- **Certification of correctness, security or local execution.** RVG is a runtime separated from the research repository with its behaviour preserved.
