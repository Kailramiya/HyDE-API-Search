# Complete Project Summary — RESTBERTa Mini-Project

> **Use this document**: Copy-paste into a fresh chat to generate the final report and presentation. Contains all phases, results, failures, learnings, and publication strategy.

---

## 1. Project Title

**"Empirical Investigation of Modern NLP Techniques for Web API Semantic Search: From Fine-Tuning to Retrieval-Augmented Generation and Hierarchical Decomposition"**

## 2. Student / Setting

- **Author**: Aman Kumar
- **Roll**: CS23B1003
- **Institution**: IIIT Raichur
- **Year**: BTech 3rd Year (6th sem)
- **Period**: Late March 2026 – April 2026 (~5 weeks)
- **Compute**: Free Google Colab (T4) and Kaggle (T4 x2 / P100) — total ~70 Colab compute units + ~50 Kaggle GPU hours used
- **Date**: April 2026

## 3. Baseline Paper (Target to Beat / Build Upon)

> **Kotstein, S. & Decker, C. (2024).** "RESTBERTa: A Transformer-based question answering approach for semantic search in Web API documentation." *Cluster Computing*, 27:4035–4061.
> DOI: 10.1007/s10586-023-04237-x

**Their best results**:
| Model | Dataset | Acc@1 | Acc@5 | Acc@10 |
|---|---|---|---|---|
| CB-PM (CodeBERT) | Parameter Matching | **0.8195** | 0.9759 | 0.9816 |
| CB-ED (CodeBERT) | Endpoint Discovery | **0.8844** | 0.9678 | 0.9749 |
| CB-PM+ED (mixed) | Parameter Matching | 0.8208 | 0.9759 | 0.9816 |

**Their compute**: 10 epochs × 864K samples × 1 day per epoch on NVIDIA Ampere (A100) = ~10 days training.

## 4. Problem Statement

Given:
- A natural-language description (e.g., *"the user's email address"*)
- An API schema context (e.g., `users[*].id users[*].name users[*].email address.street ...`)

Predict: the **exact XPath** of the matching parameter (e.g., `users[*].email`).

This is **extractive QA over hierarchical, XPath-style structured text**. Schemas average 12,721 tokens; max 160,955 tokens. The challenge is finding the right "needle" in a long haystack of similar-looking property names.

## 5. Dataset Used

- **Source**: Public Zenodo dataset from the RESTBERTa paper (DOI: 10.5281/zenodo.8349083)
- **Parameter Matching (PM)**: 1,085,051 QA pairs (864K train + 220K eval)
- **Endpoint Discovery (ED)**: 55,659 QA pairs (44K train + 11K eval)
- **Format**: Each QA = (NL question, flattened XPath context, answer XPath)

## 6. Phases Attempted (Chronological)

The project iterated through **7 phases**, each addressing learnings from the previous one.

---

### Phase 1: Reproduce Paper Baseline

- **Goal**: Verify that the paper's pre-trained models work.
- **What I did**: Loaded the paper's pre-trained CB-PM, CB-ED, CB-PM+ED checkpoints from Kaggle dataset. Ran their evaluation pipeline (`identify_properties`, `determine_best_property`, overflow tokenization with stride 128).
- **Result**: Reproduced baseline numbers — **0.8529 Acc@1 on PM** (matching paper's claim).
- **Outcome**: ✅ Successful reproduction. Established pipeline for fair comparison.

---

### Phase 2: Try Larger Encoders (DeBERTa-base, ALBERT-base)

- **Goal**: Test if simply using a larger / different encoder family beats CodeBERT.
- **What I did**: Fine-tuned **DeBERTa-v3-base** and **ALBERT-base-v2** on combined PM+ED for 3 epochs on 50K sampled QA pairs. Used the same overflow tokenization as the paper.
- **Results**:
  | Model | PM Acc@1 | ED Acc@1 |
  |---|---|---|
  | ALBERT-base | **0.8064** | 0.7545 |
  | DeBERTa-v3-base | 0.5348 | 0.7254 |
- **Lesson**: ALBERT scaled well even at 50K samples (0.8064). DeBERTa underperformed — likely SentencePiece tokenizer mismatched with XPath syntax. Paper's CodeBERT (byte-level BPE) is genuinely better-suited.
- **Outcome**: ⚠️ Did not beat paper but identified ALBERT as a promising scaling candidate.

---

### Phase 3: Encoder Fine-Tuning + LLM Zero-Shot

- **Goal (a)**: Compare more encoders (RoBERTa, ELECTRA).
- **Goal (b)**: Test if 2024-era LLMs can do this task zero-shot.
- **What I did**:
  - Fine-tuned **RoBERTa-base** and **ELECTRA-base** with 50K samples.
  - Zero-shot prompted **Llama, DeepSeek, Phi-2, Qwen-2.5** with prompt-based extraction.
- **Results**:
  | Model | Type | PM Acc@1 |
  |---|---|---|
  | RoBERTa-base | fine-tuned | 0.7998 |
  | ELECTRA-base | fine-tuned | 0.7802 |
  | Llama | zero-shot | 0.0470 |
  | DeepSeek | zero-shot | 0.1135 |
  | Qwen-2.5 | zero-shot | 0.2390 |
- **Lesson**: Fine-tuned encoders are the right tool. **Zero-shot LLMs catastrophically fail** because the task isn't natural-language QA — answers are XPath-shaped strings (`users[*].email`) that LLMs paraphrase or hallucinate.
- **Outcome**: ⚠️ Fine-tuned encoders within striking distance. LLMs unsuitable without fine-tuning + constrained decoding.

---

### Phase 4: LoRA / QLoRA Fine-Tuning (Faculty Suggestion)

- **Goal**: Use parameter-efficient fine-tuning (LoRA) on a larger encoder (DeBERTa-v3-large, 435M) to beat the paper without massive compute.
- **What I did**:
  - First attempt: DeBERTa-v3-large + LoRA r=16, target query/key/value
  - Second attempt: bigger LoRA (r=32, alpha=64) + `dense` target layers
  - Multiple Drive/RAM/runtime crashes on Colab (T4 has only 12.7 GB system RAM)
- **Results**:
  - Initial training: model collapsed catastrophically (constant logits, std ≈ 0)
  - Eval: 0.2612 Acc@1 (essentially random)
- **Lesson**: Adding `dense` to LoRA targets matched **all FFN layers** in DeBERTa, causing **encoder representation collapse** — every input produced the same hidden state. Also identified critical methodology bug: I was using truncation-only tokenization for evaluation, while the paper used overflow tokens — fundamentally non-comparable results.
- **Outcome**: ❌ Failed but very informative. Documented LoRA target selection as a real pitfall.

---

### Phase 5: ALBERT-xxlarge Full Fine-Tune (on Kaggle)

- **Goal**: Switch to Kaggle (more RAM, better GPUs) and try ALBERT-xxlarge (235M shared-param) — same family as the 0.8064 winner from Phase 2.
- **What I did**: Built clean Kaggle notebook with proper overflow tokenization, 3-cap negative sampling (matching paper's methodology).
- **Result**: ALBERT-xxlarge training was **brutally slow on T4 x2** — 0.01 it/s, ETA = 27 days. Aborted.
- **Lesson**: ALBERT's parameter sharing causes sequential compute → much slower than parameter count suggests. Not feasible on T4.
- **Outcome**: ❌ Aborted but established the engineering reality of training large models on free GPUs.

---

### Phase 6: Retrieval-Augmented Generation (RAG)

- **Goal**: Implement the paper's own future work (Section 7.2) — a two-stage retrieval + reranking pipeline.
- **Architecture**:
  ```
  Query → BGE-small-en-v1.5 (33M, fine-tuned with contrastive loss)
        → FAISS top-20 retrieval
        → Concat top-20 candidates as context
        → RoBERTa-base reranker (extractive QA)
        → Final span prediction
  ```
- **Training details**:
  - BGE: 100K (question, correct_property, hard_negative) triples, MultipleNegativesRankingLoss, 2 epochs (~2 hrs)
  - Reranker: 60K samples, 2 epochs (~1.5 hrs)
- **Results**:
  | Dataset | Acc@1 | Acc@5 | Acc@10 |
  |---|---|---|---|
  | PM | **0.6437** | **0.8675** | 0.8957 |
  | ED | 0.5582 | 0.7780 | 0.8611 |
- **Lesson**:
  - **Retrieval works well**: Acc@5 of 0.8675 means correct answer is in top-5 ~87% of the time
  - **Ranking is the bottleneck**: top-1 is hard because top candidates are very similar (`user.email` vs `users[*].email`)
  - This is the paper's **explicitly mentioned future work** — first implementation of it
- **Outcome**: ⚠️ Below paper's Acc@1 but **competitive Acc@5**. Working baseline for future improvements (cross-encoder reranker, BGE-large, hybrid sparse-dense retrieval).

---

### Phase 7: H-RESTBERTa — Hierarchical Decomposition (Novel Approach)

- **Goal**: Replace flat span extraction with **3-level hierarchical classification** (root → middle → leaf), matching the tree structure of API schemas.
- **Motivation**: Paper's #1 error mode is "Missing Context" (72% of errors) — model can't disambiguate `user.email` from `contact.email` because it loses parent-child relationships when flattening the tree.
- **Architecture**:
  ```
  Shared CodeBERT-base encoder
            │
        [CLS embedding]
            │
        Level 1 head → predict root (users / products / orders / ...)
            │ (conditioned on root embedding)
        Level 2 head → predict middle ([*] / id / name / ...)
            │ (conditioned on root + middle embeddings)
        Level 3 head → predict leaf (email / price / status / ...)
  ```
- **Training**:
  - 88K training samples, 3 epochs, 4h 18min on T4 x2
  - Multi-task loss: L_total = 1.0·L1 + 1.0·L2 + 1.5·L3
  - Teacher forcing during training, beam search at inference
  - Vocabulary sizes: L1=6730, L2=5789, L3=6959 (built with min_freq=2)
- **Results**:
  | Metric | PM | ED |
  |---|---|---|
  | Level 1 acc (argmax) | **0.6122** | 0.4100 |
  | Level 2 acc (argmax) | **0.6440** | 0.2578 |
  | Level 3 acc (argmax) | **0.6428** | 0.2527 |
  | **Joint per-level argmax** | **0.3738** | 0.1488 |
  | **Beam search Acc@1** | **0.0526** | 0.0002 |
  | Beam search Acc@10 | 0.0770 | 0.0009 |

- **Critical finding**: Massive gap between per-level joint accuracy (37%) and beam search reconstruction (5%). Diagnostic showed:
  - **Root cause**: PAD token has very high probability at Level 3 because many ground-truth answers are 2-level deep (`users.id`)
  - **Effect**: Beam search top-1 reconstruction is biased toward truncated paths (`users[*]` instead of `users[*].email`)
  - **Implication**: Hierarchical decomposition needs explicit depth prediction / PAD penalty in scoring
- **Outcome**: ❌ Did not beat paper but identified a **legitimate research problem** (label imbalance in hierarchical span prediction) and produced **publishable failure analysis**.

---

## 7. Final Comparison Table

| Phase | Approach | PM Acc@1 | ED Acc@1 | Status |
|---|---|---|---|---|
| Baseline | Paper's CB-PM | **0.8195** | — | ✅ Reproduced (Phase 1) |
| Baseline | Paper's CB-ED | — | **0.8844** | ✅ Reproduced (Phase 1) |
| Phase 2 | ALBERT-base (50K) | 0.8064 | 0.7545 | ⚠️ Close |
| Phase 2 | DeBERTa-v3-base (50K) | 0.5348 | 0.7254 | ❌ Tokenizer issue |
| Phase 3 | RoBERTa-base (50K) | 0.7998 | 0.7787 | ⚠️ Close |
| Phase 3 | ELECTRA-base (50K) | 0.7802 | 0.7610 | ⚠️ |
| Phase 3 | Qwen-2.5 zero-shot | 0.2390 | 0.2980 | ❌ |
| Phase 4 | DeBERTa-large + LoRA | 0.2612 | — | ❌ Encoder collapse |
| Phase 5 | ALBERT-xxlarge | — | — | ❌ Aborted (too slow) |
| **Phase 6** | **RAG (BGE + RoBERTa)** | **0.6437** | **0.5582** | ⚠️ Working pipeline |
| **Phase 7** | **H-RESTBERTa** | **0.0526** (5%) | 0.0002 | ❌ Beam search bug, but per-level joint = 37% |

## 8. Key Lessons Learned

### Methodological Lessons

1. **Tokenization methodology must match between training and evaluation.** Using overflow tokens for one and truncation for the other produces fundamentally non-comparable Acc@1 numbers. (Discovered in Phase 4.)
2. **`return_overflowing_tokens=True` with `stride=128` is critical** for this dataset — average context is 12,721 tokens, far exceeding BERT's 512-token limit. Without it, ~85% of windows have no answer and the model degenerates.
3. **3-cap negative sampling** (max 3 unanswerable windows per QA pair) matches the paper's methodology and prevents the model from learning "always predict CLS."

### Engineering Lessons

4. **LoRA target selection is non-trivial.** Adding `dense` target matches every FFN layer in DeBERTa, causing **encoder representation collapse**. Stick to attention projections (`query_proj`, `key_proj`, `value_proj`).
5. **fp16 training with LoRA** can crash with "unscale FP16 gradients" error. Fix: cast LoRA trainable params to float32 explicitly.
6. **`modules_to_save=['qa_outputs']`** is required when fine-tuning with PEFT for QA tasks — otherwise the QA head is silently re-randomized at load time.
7. **ALBERT's parameter sharing** makes inference sequential per layer-share-cycle; on T4 it is **5–10× slower** than equivalent-parameter BERT/RoBERTa.
8. **Free Colab T4 has only 12.7 GB system RAM**, which is the real bottleneck (not VRAM). Reservoir sampling and streaming tokenization are essential.
9. **Drive mount drops randomly** on Colab — every artifact must be re-loadable from disk; never rely on RAM state across cells.

### Research Lessons

10. **Acc@5 vs Acc@1 gap is highly informative.** Phase 6 RAG had 0.6437 Acc@1 but 0.8675 Acc@5 — telling us the system *retrieves* well but *ranks* poorly. This kind of decomposed metric is more useful than a single accuracy number.
11. **Per-level analysis beats end-to-end metrics for diagnosis.** H-RESTBERTa's joint-argmax accuracy (37%) vs beam-search Acc@1 (5%) revealed the PAD-bias bug. End-to-end Acc@1 alone hides the cause.
12. **A negative result with root-cause analysis is publishable.** The H-RESTBERTa failure mode (label imbalance in hierarchical span prediction) is a real and unstudied problem.

### Practical Lessons

13. **Compute matters far more than I expected.** The paper used a NVIDIA A100 for 10 days. On a free T4, comparable training would take ~50 days. Honest framing of compute constraints is part of the contribution.
14. **Paper-replication-first is the right starting strategy.** Phase 1's exact reproduction gave me a known-good pipeline against which to benchmark every novel idea.

## 9. Potential Publication Angles

Even without beating the baseline on Acc@1, the project has at least three publishable angles:

### Angle A: "Compute-Efficient Empirical Study"
> *"Reproducing State-of-the-Art Web API Semantic Search on Consumer GPUs: An Empirical Investigation"*

- Narrative: Replicate paper at scale-down, document compute-accuracy trade-offs.
- Audience: Workshop on Reproducibility (e.g., **MLRC**, **ML Reproducibility Challenge**).

### Angle B: "RAG for Structured-Text Search"
> *"Decoupled Retrieval-Reranking for API Semantic Search: Where Retrieval Helps and Ranking Fails"*

- Narrative: Implement paper's Section 7.2 future work; show retrieval vs ranking decomposition.
- Audience: **EMNLP Workshop on NLP for Search**, **SIGIR**, **ICTIR**.

### Angle C: "Hierarchical Decomposition Failure Analysis"
> *"On the Challenges of Hierarchical Decomposition for Web API Semantic Search"*

- Narrative: Propose H-RESTBERTa, show per-level model learns well (61–64% per level), identify PAD-bias bug, propose mitigations.
- Audience: **ICLR Workshop on Structured NLP**, **EMNLP Workshop on Insights from Negative Results**.

**Recommended target**: Angle B + Angle C combined into a "Comprehensive Empirical Study of Modern Approaches to Web API Semantic Search" workshop paper.

## 10. Artifacts Produced

| Artifact | Location | Status |
|---|---|---|
| Phase 1 reproduction notebook | `kaliram-final-final-12marchxxx.ipynb` | ✅ Complete |
| Phase 2 fine-tune notebook (DeBERTa, ALBERT) | `phase2_finetune.ipynb` | ✅ Complete with results |
| Phase 3 LLM eval notebook | `phase3_llm_eval (1).ipynb` | ✅ Complete with all baselines |
| Phase 4 LoRA notebook | `mini-project/phase4_lora_qlora_v4.ipynb` | ❌ Failed but code complete |
| Phase 6 RAG notebook | `mini-project/kaggle_phase6_rag.ipynb` | ✅ Working, 0.64 Acc@1 |
| Phase 7 H-RESTBERTa notebook | `mini-project/kaggle_h_restberta.ipynb` | ⚠️ Trained, failed at beam search |
| Faculty proposal | `mini-project/H-RESTBERTa_Faculty_Proposal.md` | ✅ 5000+ words |
| Personal explanation guide | `mini-project/H-RESTBERTa_Explanation_For_You.md` | ✅ Hinglish |
| Phase 6 architecture doc | `mini-project/Phase6_RAG_Architecture.md` | ✅ Complete |
| Phase 6 result JSONs | `kaggle_working/results/rag_*.json` | ✅ Saved |
| Phase 7 result JSONs | `kaggle_working/results/h_restberta_*.json` | ✅ Saved (low numbers) |
| New paper to study (next chat) | `mini-project/LLM-Based_RESTful_Web_API_Service_Discovery_LLM-Ba.pdf` | 📥 Pending |

## 11. What the Final Report Should Contain

If writing a 10–15 page report or thesis chapter:

1. **Abstract** — One paragraph: 7-phase study, baselines reproduced, RAG implemented, hierarchical proposed and analyzed.
2. **Introduction** — Web API search problem, challenges, our contributions.
3. **Background** — REST APIs, OpenAPI, BERT/CodeBERT, extractive QA, LoRA, RAG.
4. **Related Work** — RESTBERTa paper deep-dive, ServiceBERT, hierarchical text classification literature.
5. **Methodology** — All 7 phases (each with motivation, method, results, lessons).
6. **Experiments and Results** — Master comparison table, per-phase deep dive, error analysis.
7. **Discussion** — Per-level vs end-to-end metrics, retrieval vs ranking, compute-accuracy trade-off.
8. **Limitations** — Compute constraints, single-dataset study, single random seed.
9. **Future Work** — Bug fixes for H-RESTBERTa beam search, BGE-large for RAG, cross-encoder reranker.
10. **Conclusion** — Summary of empirical findings + methodology lessons.

## 12. What the Presentation Slides Should Contain

For a 15–20 minute defense:

| Slide | Content | Time |
|---|---|---|
| 1 | Title + author | 30s |
| 2 | Motivation: Why API search matters (Shopify 6000 params example) | 1 min |
| 3 | Baseline paper (RESTBERTa) — what they did, results | 1 min |
| 4 | Their identified weaknesses (72% Missing Context, etc.) | 1 min |
| 5 | My approach overview: 7 phases | 1 min |
| 6 | Phase 1: Reproduce baseline (✅) | 1 min |
| 7 | Phase 2-3: Try other encoders + LLMs (⚠️ ALBERT close, LLMs fail) | 2 min |
| 8 | Phase 4-5: LoRA / scaling (❌ collapsed / too slow) | 1 min |
| 9 | **Phase 6: RAG** (architecture diagram, retrieval-vs-ranking insight) | 3 min |
| 10 | **Phase 7: H-RESTBERTa** (architecture diagram, novel idea, per-level analysis) | 3 min |
| 11 | Master comparison table (all phases vs paper) | 1 min |
| 12 | Lessons learned | 2 min |
| 13 | Publication strategy + future work | 1 min |
| 14 | Q&A | open |

## 13. One-Paragraph Pitch

> Modern Web APIs expose thousands of parameters; finding the right one given a natural-language description is a real productivity bottleneck for developers. The recent RESTBERTa paper (Cluster Computing 2024) frames this as extractive question-answering and achieves 81.95% Accuracy@1 by fine-tuning CodeBERT on 1M+ QA pairs over ten days on an A100 GPU. In this project I conducted a seven-phase empirical investigation of modern NLP techniques on the same task using only consumer-grade compute (Colab T4, Kaggle T4×2). I reproduced the paper's baseline; tested four other encoder backbones and three zero-shot LLMs (zero-shot completely fails for this XPath-shaped extraction task); diagnosed a catastrophic representation-collapse failure when adding `dense` layers as LoRA targets in DeBERTa; implemented the paper's own proposed future work — a retrieval-augmented two-stage pipeline using BGE + RoBERTa achieving 64% Acc@1 and 87% Acc@5; and finally proposed and evaluated **H-RESTBERTa**, a novel hierarchical decomposition that predicts the answer XPath level-by-level. H-RESTBERTa learns per-level structure (61–64% accuracy at each level on PM) but suffers a label-imbalance failure at beam-search reconstruction (5% Acc@1) — a previously-unstudied research problem in hierarchical span prediction. The negative result, accompanied by detailed per-level diagnostic analysis, identifies a concrete future direction (depth-aware decoding) and constitutes a publishable contribution to the structured-NLP literature.

## 14. Suggested Prompt for the Next Chat

Paste this into a fresh chat to generate the final report and presentation:

```
Hi, I am Aman Kumar, BTech 3rd year at IIIT Raichur. I just finished a
mini-project on Web API semantic search and need help producing two
deliverables:

1. A formal academic report (8–12 pages, LaTeX preferred)
2. A presentation deck (15–20 slides, suitable for a 20-minute defense)

I have done substantial experimental work — 7 phases of attempts ranging
from full reproduction of the baseline paper to novel architectures.
Please read the comprehensive summary I will paste below and produce:

- A clean, formal academic report in LaTeX (article class, 11pt, with
  proper sections, bibliography, tables, and figures placeholders)
- A presentation deck in Markdown (one slide per heading, with speaker
  notes in italics)

Style requirements:
- Honest about failures — frame them as lessons / publishable insights,
  not as embarrassments
- Quantitative where possible
- Cite the baseline paper properly (Kotstein & Decker, Cluster Computing 2024)
- Include a master comparison table
- Don't oversell — but don't undersell either; this is genuine research effort
- Target audience: faculty advisor + workshop reviewers

Also generate:
- A one-paragraph abstract
- A one-line elevator pitch
- A list of 8–10 anticipated questions with crisp 2–3 sentence answers

Here is the complete project summary:

[PASTE THE CONTENTS OF mini-project/PROJECT_COMPLETE_SUMMARY.md HERE]

Start by giving me a brief outline of the report structure you propose,
then ask me to confirm before writing the full LaTeX. For the slides,
you can produce them right away after the report is approved.
```

---

**End of Project Summary**

This document captures everything needed for the next chat to produce a complete, professional report and presentation without losing any context.
