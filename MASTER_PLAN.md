# Master Plan

> The curriculum for *AI Engineering: From First Principles to Production*.
> This is the design document for the whole book. Chapters are written against it.
> Volume numbers, slugs and titles are mirrored in `book.toml`; the validator checks
> that both files agree.

---

## 1. Purpose

Teach an experienced software and machine learning engineer to design, implement, train,
evaluate, deploy, monitor and improve modern AI systems, from a from-scratch Transformer to a
production AI platform.

**End state.** The reader can build an LLM-based AI system from first principles and understands
the whole lifecycle: data, model development, evaluation, inference, deployment, production
monitoring and continuous improvement.

## 2. Reader contract

| Assumed (not taught)                           | Taught in depth                                        |
|------------------------------------------------|--------------------------------------------------------|
| Python, testing, packaging, typing             | Transformer internals, implemented from scratch        |
| Classical ML, statistics, optimisation basics  | LLM data, pretraining, post-training and reasoning RL  |
| Backprop, CNNs/RNNs, PyTorch basics            | Evaluation as an engineering discipline                |
| APIs, databases, queues, caching               | Inference economics, serving and GPU memory            |
| Distributed systems and cloud concepts         | RAG, agents, harnesses, guardrails, routing            |
| Production operations                          | AI security, observability, cost and platform design   |

Basic material is revisited only when an advanced concept depends on a detail the reader may
not have needed before, for example the numerics of softmax, the memory layout of tensors or
the precise definition of cross-entropy in nats per token.

## 3. Pedagogical model

Every major topic follows the same arc.

```text
Concept -> Problem -> Architecture -> Implementation -> Python Code
        -> Test -> Evaluate -> Debug -> Optimise -> Production Version
```

Important concepts are built in this order.

```text
Understand concept
  -> Implement simplified version from scratch (NumPy or plain PyTorch)
  -> Test it (unit tests, numerical comparison against a reference)
  -> Understand its limitations
  -> Use the production library
  -> Inspect what the library abstracts
  -> Compare both implementations (correctness, speed, memory)
```

Three recurring patterns are mandatory where they apply:

- **Problem first.** Every important concept opens with the engineering problem and competing
  solutions, then explains what each solution does and does not solve, with cost, latency,
  maintenance and failure modes.
- **Simple, then production, then why production is more complicated.** Complexity is never
  hidden; simplified systems are labelled as simplified.
- **Documented versus inferred.** Real products are described with explicit evidence labels.
  Private implementation details are never presented as fact.

## 4. Volume map

The request specified about 20 volume areas. The plan uses **30 volumes** (29 teaching volumes and
1 reference volume). Early splits are justified in section 5; Volumes 26–30 extend practice and
specialised delivery after the core curriculum.

| #  | Volume                                                   | Source area(s) | Forge increment           |
|----|----------------------------------------------------------|----------------|---------------------------|
| 1  | Modern AI Engineering                                    | 1, 25 (intro)  | `forge.core`, `forge.compute` |
| 2  | Transformers from First Principles                       | 2              | `forge.nn`                |
| 3  | Tokenisation and Representations                         | 3              | `forge.tokenizers`        |
| 4  | Building a Small LLM                                     | 4              | `forge.model`, `forge.train`, `forge.generate` |
| 5  | LLM Data Engineering                                     | 5              | `forge.data`              |
| 6  | LLM Pretraining and Distributed Training                 | 6              | `forge.train.distributed`, `forge.scaling` |
| 7  | Modern LLM Architectures                                 | 2, 6, 30 (new) | `forge.nn` (GQA, MoE, SWA) |
| 8  | Fine-Tuning and Post-Training                            | 7              | `forge.posttrain`         |
| 9  | Reasoning Models and Test-Time Compute                   | 7, 30 (new)    | `forge.reasoning`         |
| 10 | Evaluation Engineering                                   | 8, 18          | `forge.eval`              |
| 11 | Inference Optimisation                                   | 9, 31          | `forge.inference`         |
| 12 | Model Serving and Deployment                             | 10, 24         | `forge.serve`             |
| 13 | Prompt and Context Engineering                           | 12             | `forge.context`           |
| 14 | Retrieval-Augmented Generation                           | 11             | `forge.rag`               |
| 15 | AI Agents                                                | 13, 21         | `forge.agents`            |
| 16 | Agent Harnesses                                          | 14, 22         | `forge.harness`           |
| 17 | Guardrails and AI Security                               | 15, 23         | `forge.guard`             |
| 18 | Model Routing and Decision Systems                       | 16, 20         | `forge.router`            |
| 19 | Multimodal AI I: Understanding                           | 17             | `forge.multimodal`        |
| 20 | Multimodal AI II: Generation                             | 17             | `forge.multimodal.gen`    |
| 21 | AI Product Architecture and Case Studies                 | 9, 10, 18      | `services/` (HLD/LLD)     |
| 22 | AI Infrastructure, Operations and Cost                   | 19, 24         | `forge.ops`, `deploy/`    |
| 23 | AI Engineering Tooling and AI-Assisted Development       | 25, 26 (new)   | repo tooling, eval of generated code |
| 24 | Capstone: The Complete AI Platform                       | 20, 29         | integration of everything |
| 25 | Reference: Research Roadmap, Glossary and Interview Bank | 30, 37         | —                         |
| 26 | AI Coding Agents in Practice                             | 25, 26 (deep)  | `forge.devex` (workspace, rules, gates) |
| 27 | Forward Deployed AI Engineering                          | 18, 21 (new)   | `forge.fde` (discovery → PoC → pilot) |
| 28 | Enterprise Integration and Compliance                    | 17, 22, 27     | `forge.enterprise` (SSO, tenancy, residency) |
| 29 | Realtime and Voice Agents                                | 12, 15, 16     | `forge.realtime` (duplex, barge-in, telephony) |
| 30 | Synthetic Data and Eval Data Factories                   | 5, 8, 10       | `forge.synth` (generators, contamination control) |

**Ordering decisions.**

- Prompt and Context Engineering (13) comes **before** RAG (14). RAG is one kind of context
  assembly and relies on structured outputs, templates and context budgets taught in 13.
- Evaluation gets its own volume at 10, but evaluation is **not deferred**. Every volume from 4
  onwards has an evaluation chapter using the harness that Volume 10 later generalises.
- Security appears in every systems volume (12–18). Volume 17 consolidates it into one
  architecture.
- Volume 23 surveys the tool ecosystem. Volume **26** teaches how to *operate* Claude, ChatGPT,
  Cursor and Kiro day to day (files, features, learn / vibe / ship workflows). Volume **27**
  teaches Forward Deployed delivery on top of Volumes 21–24.
- Volumes **28–30** specialise enterprise integration, realtime/voice agents and synthetic/eval
  data factories after the FDE and coding-agent practice layers.

## 5. Justification for additional volumes

| Added / split volume | Why it cannot be absorbed |
|----------------------|---------------------------|
| 7 Modern Architectures | GQA/MQA/MLA, MoE, long context, SSMs and hybrids change inference cost, training dynamics and serving design. Squeezing them into Volume 2 would overload the first-principles volume. |
| 9 Reasoning Models | RL with verifiable rewards, process rewards, test-time search and thinking budgets form a distinct training and serving regime with their own evaluation and cost profile. |
| 19/20 Multimodal split | Understanding (encoders, VLMs, ASR) and generation (diffusion, flow matching, TTS, video) share little implementation. One volume would be too shallow. |
| 23 Tooling and AI-assisted development | Areas 25 and 26 require per-tool analysis and disciplined AI-assisted engineering practice. Scattering this across volumes loses the comparative view. |
| 25 Reference | A consolidated paper roadmap, glossary (area 37 terminology) and interview bank serve as the book's index layer. |
| 26 AI Coding Agents in Practice | Volume 23 compares tools; engineers still need a canonical workspace module (`AGENTS.md`, Cursor rules, Claude/Kiro skills, MCP), feature matrices, and distinct workflows for learning, vibe prototyping and production delivery with verification gates. |
| 27 Forward Deployed AI Engineering | Product architecture (21) and the platform capstone (24) do not teach customer discovery, PoC scoping, pilot success metrics, on-site integration or handoff — the FDE loop that turns Forge components into customer outcomes. |
| 28 Enterprise Integration and Compliance | Guardrails (17) and ops (22) do not cover SSO/OIDC, private networking, residency, tenancy contracts and procurement constraints that block real deployments. |
| 29 Realtime and Voice Agents | Serving (12) and agents (15–16) assume turn-based HTTP/SSE; duplex audio, barge-in, telephony and sub-second SLOs need a dedicated stack. |
| 30 Synthetic Data and Eval Data Factories | Data engineering (5) and evaluation (10) mention synthetic data; building generators, contamination control and preference/SFT factories at scale needs its own volume. |

## 6. The continuous project: Forge

One Python package, `forge`, grows through the whole book. It lives in `project/src/forge/` with
tests in `project/tests/`. Chapter code in `volumes/NN-*/code/` is pedagogical and self-contained.
Forge is the integrated system the reader keeps.

```text
V1  forge.core, forge.compute     config, logging, seeds, FLOP/memory calculators
V2  forge.nn                      attention, MHA, RoPE, RMSNorm, SwiGLU, blocks
V3  forge.tokenizers              byte-level BPE trainer and encoder, special tokens
V4  forge.model/train/generate    GPT model, training loop, checkpoints, sampling
V5  forge.data                    ingestion, filters, MinHash dedup, PII, shards
V6  forge.train.distributed       DDP/FSDP (CPU gloo), mixed precision, scaling fits
V7  forge.nn (extended)           GQA, MoE, sliding window, long-context RoPE scaling
V8  forge.posttrain               SFT, LoRA (from scratch), DPO, reward model
V9  forge.reasoning               GRPO on verifiable tasks, best-of-n, verifiers
V10 forge.eval                    eval harness, golden sets, judges, regression gates
V11 forge.inference               KV cache, paged KV, batching, INT8/INT4, speculation
V12 forge.serve                   model server, continuous batching, OpenAI-style API
V13 forge.context                 templates, structured outputs, context builder, memory
V14 forge.rag                     chunkers, BM25, vector index, hybrid, reranker
V15 forge.agents                  agent loop, tools, planner, reflection, multi-agent
V16 forge.harness                 runtime, permissions, budgets, sandbox, tracing
V17 forge.guard                   guardrail pipeline, injection defences, policy engine
V18 forge.router                  rules, cost/latency/quality routers, cascades, bandits
V19 forge.multimodal              ViT, CLIP-style contrastive training, VLM adapter
V20 forge.multimodal.gen          tiny DDPM/flow-matching, latent diffusion pipeline
V21 services/                     HLD/LLD of product services built on Forge
V22 forge.ops, deploy/            OpenTelemetry, Prometheus metrics, cost ledger, K8s
V23 tooling/                      agent instructions, generated-code validation gates
V24 platform/                     the complete integrated AI platform
V26 forge.devex                   coding-agent workspace: rules, skills, MCP, gates
V27 forge.fde                     discovery briefs, PoC harness, pilot scorecards
V28 forge.enterprise              SSO/OIDC stubs, tenancy, residency policy helpers
V29 forge.realtime                session state, barge-in, latency budgets, call graphs
V30 forge.synth                   generators, filters, contamination checks, factories
```

**Model access is vendor-neutral.** `forge.models` defines one `ChatModel` protocol with adapters:

- `ForgeModel`: the reader's own trained model.
- `HFModel`: small open-weight models through Hugging Face Transformers.
- `OpenAICompatibleModel`: any OpenAI-compatible HTTP endpoint, for example vLLM, SGLang, Ollama or hosted APIs.
- `AnthropicModel` and other hosted adapters, behind the same protocol.
- `ScriptedModel`: a deterministic fake for tests, so no test needs an API key or network.

## 7. Compute tiers

Every runnable example declares its tier.

| Tier | Hardware | Budget | Used for |
|------|----------|--------|----------|
| **L** Laptop | CPU or Apple MPS, 16 GB RAM | seconds to ~30 min | All from-scratch code, tests, tiny models (≤ 30M params) |
| **G** Single GPU | 1× 24–80 GB GPU (local or rented) | minutes to hours | 100M–1B training, LoRA on 1–8B models, vLLM serving |
| **C** Cluster | multi-GPU / multi-node | described, not required | Distributed training at scale, simulated locally with `gloo` |

All tests run at Tier L. Tier G and C examples include cost estimates.

## 8. Standard volume ending

Every volume ends with: *What You Learned · What You Built · Important Code · Important
Architecture (flow chart) · Failure Modes · Evaluation · Practical Project · Exercises ·
Research Papers · Interview Questions · Next Volume*. There is no generic conclusion.

## 9. Real-product case studies

Case studies are distributed and then consolidated in Volume 21. Each follows the template:
problem, user workflow, model requirements, HLD, LLD, data flow, model flow, retrieval, tools,
memory, routing, guardrails, evaluation, inference, scaling, cloud and cost. Every claim is marked
**Documented** (with a citation to a public source) or **Inferred** (reasoned architecture).

| Product class                | Primary volume | Also appears in |
|------------------------------|----------------|-----------------|
| Chat assistants (ChatGPT, Claude, Gemini) | 21 | 1, 12, 13, 17, 18 |
| Code completion (GitHub Copilot) | 21 | 11, 23 |
| Coding agents (Claude Code, Codex-style, SWE-agent) | 21 | 15, 16, 23 |
| Enterprise assistants        | 21 | 14, 17 |
| Research assistants          | 21 | 14, 15 |
| Customer support agents      | 21 | 15, 18, 10 |

---

# Curriculum

## Volume 1 — Modern AI Engineering

*Landscape, Lifecycles and Engineering Foundations*

**Objective.** Build a precise mental model of the modern AI stack and its vocabulary. Establish
the engineering foundations the rest of the book depends on: compute arithmetic, the model and
product lifecycles, the development environment and the Forge repository.

**Depends on.** Reader profile only.

**Chapters**

1. **The Modern AI Landscape**
   - How we got here: statistical NLP → deep learning → pretraining → foundation models → assistants → agents.
   - What changed in engineering: from task-specific models to general models plus systems around them.
   - The capability frontier and how it is measured; why benchmarks saturate.
   - Open-weight versus closed models; the model supply chain.
2. **Precise Terminology**
   - AI, ML, DL, Generative AI, Foundation Model, LLM, SLM, Multimodal Model.
   - AI Application, AI Workflow, AI Agent, Agentic System.
   - Agent Framework, Agent Harness, Orchestrator, Tool Runtime.
   - Guardrail, Evaluation System, AI Platform.
   - A layered reference model showing where each term sits, with counter-examples of misuse.
3. **How an LLM Works: an Engineer's Overview**
   - Text → tokens → embeddings → Transformer layers → logits → sampling.
   - Pretraining, post-training and inference as three distinct phases.
   - What the model is and is not: next-token predictor, not a database, not a program executor.
   - The first runnable demo: load a small open model, inspect tokens, logits and sampling.
4. **Compute, Memory and Cost Arithmetic**
   - FLOPs per token (≈ 2N forward, ≈ 6N training); tokens, parameters and bytes.
   - GPU memory anatomy: weights, gradients, optimiser states, activations and KV cache.
   - Arithmetic intensity and the roofline model: why decode is memory-bound.
   - Converting everything to money: GPU-hours, $/1M tokens, utilisation.
   - Implementation: `forge.compute` calculators with tests.
5. **The AI Model Lifecycle**
   - Data → pretraining → post-training → evaluation → release → monitoring → retraining.
   - Artefacts: datasets, checkpoints, model cards, eval reports, registries.
   - Reproducibility: seeds, versions, configs, environments.
6. **The AI Product Lifecycle**
   - Problem framing: is AI the right tool? Deterministic baselines first.
   - Prototype → evaluation set → pilot → production → continuous improvement.
   - The build/buy/fine-tune/prompt decision framework.
   - Success metrics: task success, quality, latency, cost, safety, business metrics.
7. **Anatomy of a Production AI System**
   - Reference HLD: gateway, orchestrator, model service, retrieval, tools, guardrails, eval, observability.
   - Reference LLD: request lifecycle, adapters, retries, timeouts, caching.
   - Systematic improvement: classify failures as model problems or system problems.
8. **Engineering Environment and the Forge Repository**
   - Repository layout, packaging, configuration, logging, seeding and determinism.
   - Testing strategy for stochastic systems: numerical tolerances, fakes, golden files.
   - Experiment tracking basics.
   - First principles of AI-assisted development: agent instructions, verification and review (expanded in Volume 23).

**Key concepts.** Foundation model; emergent capability (and its measurement caveats); compute-bound
versus memory-bound; lifecycle artefacts; model problem versus system problem; the terminology stack.

**Python implementations.**
- `forge.compute`: parameter counter, FLOPs estimator, training-memory and KV-cache calculators, cost model.
- A logit-inspection tool for a small open model (Transformers).
- `forge.core`: typed config, structured logging, seeding utilities.
- `ScriptedModel` and the `ChatModel` protocol, the first version of the vendor-neutral adapter.

**Practical project.** *Capacity planner.* Given a model size, context length, batch and hardware,
estimate training time, memory feasibility and serving cost per million tokens. Validate the
estimates against measured numbers from a small model on the reader's machine.

**Research papers.** Bommasani et al. 2021 (Foundation Models); Brown et al. 2020 (GPT-3);
Kaplan et al. 2020 (preview); Wei et al. 2022 (Emergent Abilities) with Schaeffer et al. 2023
(counterpoint); Ouyang et al. 2022 (InstructGPT); Sculley et al. 2015 (Hidden Technical Debt);
Williams et al. 2009 (Roofline); Sutton 2019 (The Bitter Lesson, essay).

---

## Volume 2 — Transformers from First Principles

*Attention, Blocks and Causal Language Modelling*

**Objective.** Implement every component of a Transformer from scratch in NumPy, verify it
numerically against PyTorch, then build the production PyTorch version and understand what
`scaled_dot_product_attention` and `nn.Transformer*` abstract.

**Depends on.** Volume 1 (compute arithmetic, Forge repo).

**Chapters**

1. **The Sequence Modelling Problem**
   - Why RNNs bottleneck: sequential computation, vanishing gradients, fixed-size state.
   - Attention as differentiable lookup; Bahdanau attention to self-attention.
2. **Tensor Mechanics for Attention**
   - Batched matmul, broadcasting, `einsum`, memory layout, views versus copies.
   - Numerically stable softmax and log-softmax; overflow and underflow in practice.
3. **Scaled Dot-Product Attention from Scratch**
   - Q, K, V as learned projections; attention scores; why scale by √d<sub>k</sub> (variance argument, proved empirically).
   - Softmax over keys; weighted sum of values; complexity O(n²d).
   - NumPy implementation, gradient check, comparison with PyTorch.
4. **Masking**
   - Causal masks, padding masks, combining masks; −inf versus large negative numbers.
   - Failure mode: information leakage from future tokens, and a test that catches it.
5. **Multi-Head Attention**
   - Head splitting and merging with reshape and transpose; output projection.
   - What heads learn; head redundancy; parameter and FLOP accounting.
6. **Position Information**
   - Permutation equivariance of attention (demonstrated).
   - Sinusoidal, learned absolute, relative, ALiBi, RoPE (derivation, complex-number view, implementation).
   - Length extrapolation behaviour of each.
7. **Feed-Forward Networks and Gating**
   - MLP as per-token computation; expansion ratio; GELU.
   - GLU variants and SwiGLU; parameter-matched comparisons.
8. **Residuals and Normalisation**
   - Residual stream view; post-LN versus pre-LN stability; LayerNorm versus RMSNorm.
   - Initialisation and its effect on training stability.
9. **The Transformer Block and Architectures**
   - Assembling the block; encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5).
   - Cross-attention; which architecture suits which task and why decoder-only won for LLMs.
10. **Causal Language Modelling**
    - The objective, teacher forcing, shifted targets, cross-entropy in nats and bits, perplexity.
    - Training a tiny Transformer on a toy language to verify the whole stack end to end.
11. **Production Transformers in PyTorch**
    - `F.scaled_dot_product_attention` and its backends; `torch.compile`.
    - Reading a production model implementation (Llama-style) line by line against our version.
    - Mechanistic view: residual stream, attention patterns, induction heads (visualised).

**Key concepts.** Q/K/V; attention scaling; masking; multi-head; RoPE; SwiGLU; pre-norm; RMSNorm;
residual stream; encoder/decoder/encoder-decoder; causal LM loss; perplexity.

**Python implementations.** NumPy: softmax, SDPA, masks, MHA, sinusoidal PE, RoPE, LayerNorm,
RMSNorm, FFN, SwiGLU and block (forward pass plus gradient checks). PyTorch: `forge.nn` modules with
parity tests against NumPy and `torch.nn.functional`. A tiny causal Transformer trained on
synthetic data at Tier L.

**Practical project.** *Transformer parity suite.* Implement the block twice (NumPy, PyTorch), prove
parity to 1e-5, benchmark against `F.scaled_dot_product_attention`, and visualise attention patterns
of a trained tiny model on a copy task, including an induction head.

**Research papers.** Bahdanau et al. 2014; Vaswani et al. 2017; He et al. 2015 (ResNet); Ba et al.
2016 (LayerNorm); Xiong et al. 2020 (Pre-LN); Zhang & Sennrich 2019 (RMSNorm); Shazeer 2020 (GLU
Variants); Su et al. 2021 (RoFormer); Press et al. 2021 (ALiBi); Devlin et al. 2018 (BERT);
Radford et al. 2018/2019 (GPT, GPT-2); Raffel et al. 2019 (T5); Elhage et al. 2021 (Transformer
Circuits); Olsson et al. 2022 (Induction Heads).

---

## Volume 3 — Tokenisation and Representations

*From Bytes to Contextual Vectors*

**Objective.** Build byte-level BPE, WordPiece and Unigram tokenisers from scratch; understand
how tokenisation shapes cost, multilingual fairness, arithmetic, code and security; understand
embeddings from static to contextual.

**Depends on.** Volume 2.

**Chapters**

1. **The Tokenisation Problem**
   - Characters versus words versus subwords versus bytes; open vocabulary; sequence length trade-off.
   - Unicode, UTF-8, normalisation (NFC/NFKC), grapheme clusters.
2. **Byte Pair Encoding from Scratch**
   - Training: pair counting, merges, efficient incremental updates.
   - Encoding: merge-rank application; correctness and speed.
   - Byte-level BPE and pre-tokenisation regexes (GPT-2 style).
3. **WordPiece and Unigram**
   - WordPiece likelihood-based merges and `##` continuation.
   - Unigram LM: EM training, Viterbi segmentation, subword regularisation.
   - SentencePiece: whitespace as a symbol, language independence.
4. **Special Tokens, Chat Templates and Vocabulary Design**
   - BOS, EOS, PAD, UNK, role and tool tokens; chat templates as a contract.
   - Vocabulary size trade-offs: embedding memory, softmax cost, compression, rare-token training.
5. **Production Tokenisers**
   - Hugging Face `tokenizers` (Rust), `tiktoken`, SentencePiece; what they add: speed, offsets, normalisers.
   - Parity tests: our BPE against `tiktoken`-style encodings on a corpus.
6. **Tokenisation Failure Modes**
   - Arithmetic and digit splitting; code whitespace; multilingual token tax.
   - Glitch and under-trained tokens; token boundary artefacts in prompting.
   - Security: tokenisation-based filter evasion; homoglyphs; invisible characters.
7. **Embeddings**
   - One-hot to dense embeddings; word2vec and GloVe intuition; weight tying.
   - Geometry: cosine similarity, anisotropy, norm effects.
8. **Contextual Representations**
   - From static to contextual (ELMo → BERT → decoder hidden states).
   - Probing representations; layer-wise behaviour; pooling strategies for sentence embeddings.
   - Positional representations revisited as part of the representation, not the tokeniser.
9. **Evaluating Tokenisers**
   - Compression (bytes per token), fertility, coverage, round-trip exactness, speed, fairness across languages.

**Key concepts.** BPE merges; byte fallback; pre-tokenisation; Unigram EM; special tokens; chat
templates; fertility; glitch tokens; weight tying; anisotropy; contextual embeddings.

**Python implementations.** `forge.tokenizers.bpe` (trainer and encoder), WordPiece, Unigram
(EM plus Viterbi), tokenizer evaluation suite, embedding analysis tools; parity and property-based
tests (round-trip `decode(encode(x)) == x`).

**Practical project.** *Train the Forge tokeniser.* Train a byte-level BPE on the Volume 4 corpus,
evaluate compression and fairness against `tiktoken` and SentencePiece, and ship it with a versioned
tokenizer artefact and chat template.

**Research papers.** Sennrich et al. 2015 (BPE); Schuster & Nakajima 2012 and Wu et al. 2016
(WordPiece); Kudo 2018 (Subword Regularization); Kudo & Richardson 2018 (SentencePiece); Mikolov et
al. 2013 (word2vec); Pennington et al. 2014 (GloVe); Peters et al. 2018 (ELMo); Ethayarajh 2019;
Xue et al. 2021 (ByT5); Petrov et al. 2023 (tokeniser unfairness); Land & Bartolo 2024 (under-trained
tokens); Pagnoni et al. 2024 (Byte Latent Transformer).

---

## Volume 4 — Building a Small LLM

*Tokeniser, Model, Training Loop, Generation and Evaluation*

**Objective.** Assemble a complete, trainable GPT-style model from Forge components; train it on
a laptop; implement checkpointing, generation and evaluation; learn to debug training.

**Depends on.** Volumes 2 and 3.

**Chapters**

1. **System Design of a Small LLM Project**
   - Components, configs, artefacts and the experiment loop; choosing a Tier L target (~10–30M params).
2. **The Dataset Pipeline**
   - Corpus selection, tokenisation to memory-mapped shards, sequence packing, train/validation split.
   - Data loaders that are deterministic and resumable.
3. **The Model**
   - `GPTConfig`, embedding, blocks, final norm, LM head, weight tying, initialisation.
   - Parameter counting against `forge.compute`.
4. **Loss, Optimiser and Schedule**
   - Cross-entropy details (ignore index, label smoothing), AdamW internals, weight-decay groups.
   - Warmup plus cosine and warmup-stable-decay schedules; gradient clipping.
5. **The Training Loop**
   - From scratch: forward, backward, step, accumulation, logging, evaluation cadence.
   - Mixed precision on MPS/CUDA; throughput measurement (tokens/s, MFU).
6. **Checkpointing and Resumption**
   - What must be saved (model, optimiser, scheduler, RNG, data position); atomic writes; exact resumption test.
7. **Generation**
   - Greedy, temperature, top-k, top-p, min-p, repetition penalty; stop conditions.
   - A naive generator, then a KV-cached generator as a first taste of Volume 11.
8. **Evaluation of a Small LLM**
   - Validation loss and perplexity; held-out domains; sample-based qualitative review.
   - Simple capability probes; the first `forge.eval` harness.
9. **Debugging Training**
   - Overfit-one-batch, loss curves taxonomy, gradient norms, dead units, NaNs, data bugs, tokenizer mismatches.
10. **From Scratch to Production Libraries**
    - The same model through Hugging Face `transformers` (`PreTrainedModel`), `accelerate` and `safetensors`.
    - What production trainers add: sharding, fault tolerance, logging integrations.

**Key concepts.** Packing; AdamW; LR schedules; gradient accumulation; MFU; exact resumption;
sampling strategies; perplexity; training debugging workflow.

**Python implementations.** `forge.model.GPT`, `forge.train.Trainer`, `forge.data.shards`,
`forge.generate` (naive and KV-cached), `forge.eval.perplexity`, checkpoint manager, HF export.

**Practical project.** *Mini LLM v1.* Train a ~20M parameter model on a curated small corpus
(TinyStories-scale) at Tier L, reach a target validation loss, export to `safetensors` and HF
format, and publish a model card with evaluation results.

**Research papers.** Radford et al. 2019 (GPT-2); Kingma & Ba 2014 (Adam); Loshchilov & Hutter 2017
(AdamW); Press & Wolf 2016 (weight tying); Goyal et al. 2017 (warmup); Holtzman et al. 2019 (nucleus
sampling); Eldan & Li 2023 (TinyStories).

---

## Volume 5 — LLM Data Engineering

*Collection, Cleaning, Deduplication, Mixtures and Contamination*

**Objective.** Build an LLM data pipeline that turns raw web and document data into versioned,
deduplicated, filtered, PII-scrubbed training mixtures with contamination checks and data cards.

**Depends on.** Volumes 3 and 4. Strong database knowledge is assumed.

**Chapters**

1. **Data as the Primary Lever**
   - Why data quality dominates at fixed compute; the data flywheel; legal and licensing landscape.
2. **Collection**
   - Common Crawl (WARC/WET), APIs, documents, code; robots.txt and terms; provenance metadata.
3. **Extraction and Normalisation**
   - HTML to text, boilerplate removal, PDF extraction, language identification, Unicode normalisation.
4. **Heuristic Quality Filtering**
   - Gopher/C4-style rules, repetition metrics, perplexity filtering, and why each rule exists.
5. **Model-Based Quality Filtering**
   - fastText-style classifiers, LLM-annotated quality labels (FineWeb-Edu pattern), threshold selection.
6. **Deduplication**
   - Exact (hashing), near-duplicate (MinHash + LSH from scratch), suffix-array substring dedup; scale trade-offs.
7. **PII and Safety Filtering**
   - Regex and checksum validators, NER-based detection, redaction versus removal, toxicity filtering, audit trails.
8. **Dataset Mixtures**
   - Domain weights, upsampling and epochs, curriculum, DoReMi-style mixture optimisation.
9. **Synthetic Data**
   - Generation strategies, diversity control, verification, model-collapse risk, licensing of generated data.
10. **Contamination**
    - n-gram overlap detection, canaries, benchmark decontamination, reporting.
11. **Versioning, Lineage and Data Cards**
    - Immutable shards, manifests, hashes, lineage graphs, datasheets; pipeline orchestration and cost.

**Key concepts.** WARC/WET; quality heuristics; perplexity filters; MinHash/LSH; Jaccard; PII
detection; mixtures; synthetic data; contamination; lineage.

**Python implementations.** `forge.data` pipeline stages: extractors, filters, MinHash LSH dedup
(from scratch, then `datasketch`), PII scrubber, mixture sampler, contamination checker, manifest
and versioning, all tested with fixture corpora and measured precision and recall.

**Practical project.** *Forge corpus v1.* Process a Common Crawl sample plus curated sources into
a deduplicated, filtered, versioned mixture; measure each stage's effect by training small models on
ablations (Volume 4 trainer).

**Research papers.** Raffel et al. 2019 (C4); Gao et al. 2020 (The Pile); Wenzek et al. 2019
(CCNet); Lee et al. 2021 (Deduplicating Training Data); Broder 1997 (MinHash); Penedo et al. 2023
(RefinedWeb) and 2024 (FineWeb); Soldaini et al. 2024 (Dolma); Li et al. 2024 (DataComp-LM); Xie
et al. 2023 (DoReMi); Gunasekar et al. 2023 (Textbooks Are All You Need); Shumailov et al. 2024
(model collapse); Carlini et al. 2021 (training-data extraction); Longpre et al. 2023 (Pretrainer's
Guide); Gebru et al. 2018 (Datasheets for Datasets).

---

## Volume 6 — LLM Pretraining and Distributed Training

*Scaling Laws, Precision, Parallelism and Checkpointing*

**Objective.** Understand large-scale pretraining quantitatively and mechanically: scaling laws,
compute budgeting, numerical precision, every major parallelism strategy (implemented at toy scale),
and the operational reality of long training runs.

**Depends on.** Volumes 1, 4 and 5; distributed-systems background.

**Chapters**

1. **The Economics of Pretraining**
   - Compute budget C ≈ 6ND; tokens versus parameters; inference-aware training (over-training small models).
2. **Scaling Laws**
   - Kaplan versus Chinchilla; fitting power laws to our own small runs; limits and extrapolation risk.
3. **Hyperparameters at Scale**
   - Batch size and critical batch size; learning-rate scaling; μP; warmup-stable-decay.
4. **Numerical Precision**
   - FP32, TF32, FP16, BF16, FP8; loss scaling; where precision matters (norms, softmax, optimiser state).
5. **Memory and Activation Management**
   - Activation checkpointing, recomputation trade-offs, CPU offload.
6. **Data Parallelism**
   - DDP from scratch using `torch.distributed` (gloo on CPU); all-reduce algorithms; gradient bucketing.
7. **Sharded Data Parallelism**
   - ZeRO stages 1–3 explained and implemented in miniature; PyTorch FSDP.
8. **Tensor and Sequence Parallelism**
   - Megatron column/row parallel linear layers implemented on CPU processes; communication cost.
9. **Pipeline Parallelism**
   - GPipe and 1F1B schedules, bubbles, micro-batching; a simulated pipeline.
10. **3D/4D Parallelism and Topology**
    - Combining DP/TP/PP/CP; NVLink versus InfiniBand; choosing a layout for a given model and cluster.
11. **Running Long Training Jobs**
    - Distributed checkpointing, failure recovery, loss spikes, divergence mitigation, monitoring and evaluation during training.

**Key concepts.** 6ND; Chinchilla optimality; critical batch size; μP; BF16/FP8; loss scaling;
all-reduce; ZeRO; FSDP; tensor/pipeline/sequence/context parallelism; pipeline bubbles; MFU.

**Python implementations.** `forge.scaling` (law fitting, compute planner); mixed-precision
trainer; mini-DDP; mini-ZeRO; column/row-parallel linear; simulated pipeline scheduler; distributed
checkpoint save and restore; loss-spike detector. All run with multiple CPU processes at Tier L.

**Practical project.** *Scaling study.* Train a family of 5–7 small models at fixed compute budgets,
fit a scaling law, predict a larger model's loss, verify at Tier G (optional), and write a compute plan
for a 1B model with parallelism layout and cost.

**Research papers.** Kaplan et al. 2020; Hoffmann et al. 2022 (Chinchilla); Sardana & Frankle 2023;
McCandlish et al. 2018; Yang et al. 2022 (μP); Micikevicius et al. 2017 (Mixed Precision) and 2022
(FP8); Chen et al. 2016 (activation checkpointing); Rajbhandari et al. 2019 (ZeRO); Zhao et al. 2023
(PyTorch FSDP); Shoeybi et al. 2019 and Narayanan et al. 2021 (Megatron-LM); Huang et al. 2018
(GPipe); Korthikanti et al. 2022 (sequence parallelism); Liu et al. 2023 (Ring Attention); Chowdhery
et al. 2022 (PaLM); Wortsman et al. 2023 (small-scale proxies for instabilities); Hägele et al. 2024
(WSD schedules); Dubey et al. 2024 (Llama 3).

---

## Volume 7 — Modern LLM Architectures

*Efficient Attention, Mixture of Experts and Long Context*

**Objective.** Understand and implement the architectural changes that distinguish current LLMs
from the 2017 Transformer, with their effects on training, inference memory, throughput and quality.

**Depends on.** Volumes 2, 4 and 6.

**Chapters**

1. **Why Architectures Keep Changing**: inference cost as the design driver; KV-cache pressure.
2. **Multi-Query, Grouped-Query and Multi-Head Latent Attention**: implementation, KV memory maths, quality trade-offs, uptraining MHA to GQA.
3. **Mixture of Experts**: routers, top-k gating, load-balancing losses, capacity factors, token dropping, expert parallelism, fine-grained and shared experts.
4. **Sparse and Local Attention**: sliding window, dilated and block-sparse patterns, attention sinks.
5. **Long Context**: RoPE scaling (position interpolation, NTK, YaRN), long-context training data, evaluation (needle tests, RULER) and "lost in the middle".
6. **Linear Attention and State Space Models**: linear attention as an RNN, S4/Mamba intuition and a minimal selective scan, hybrids.
7. **Multi-Token Prediction and Other Objectives**: MTP heads, fill-in-the-middle for code, prefix LM.
8. **Reading Modern Model Reports**: a structured method for reading Llama, Mixtral, DeepSeek and Qwen technical reports, separating documented facts from marketing.

**Key concepts.** MQA/GQA/MLA; KV-cache size formula; MoE routing and balance; expert parallelism;
sliding window; attention sinks; RoPE scaling; SSMs; hybrid architectures; MTP; FIM.

**Python implementations.** GQA and MLA modules with KV-memory benchmarks; MoE layer with top-k
router, auxiliary loss and capacity; sliding-window attention mask; RoPE interpolation and YaRN;
a minimal selective-scan SSM block; needle-in-a-haystack evaluation.

**Practical project.** *Architecture ablation.* Train parameter-matched dense, GQA and MoE variants of
the Volume 4 model; compare loss, throughput and KV-cache memory; extend context by 4× with RoPE
scaling and measure retrieval accuracy by position.

**Research papers.** Shazeer 2019 (MQA); Ainslie et al. 2023 (GQA); DeepSeek-AI 2024 (DeepSeek-V2,
MLA; DeepSeek-V3); Shazeer et al. 2017 (Sparsely-Gated MoE); Lepikhin et al. 2020 (GShard); Fedus et
al. 2021 (Switch Transformer); Jiang et al. 2024 (Mixtral); Dai et al. 2024 (DeepSeekMoE); Child et
al. 2019; Beltagy et al. 2020 (Longformer); Xiao et al. 2023 (Attention Sinks); Chen et al. 2023
(Position Interpolation); Peng et al. 2023 (YaRN); Liu et al. 2023 (Lost in the Middle); Hsieh et al.
2024 (RULER); Katharopoulos et al. 2020 (linear attention); Gu et al. 2021 (S4); Gu & Dao 2023
(Mamba); Lieber et al. 2024 (Jamba); Gloeckle et al. 2024 (Multi-token Prediction).

---

## Volume 8 — Fine-Tuning and Post-Training

*SFT, PEFT, LoRA, RLHF, DPO and Domain Adaptation*

**Objective.** Turn a base model into a useful, aligned, domain-adapted model. Implement SFT, LoRA,
reward modelling, PPO-style RLHF (miniature) and DPO from scratch, then with production libraries,
and learn when fine-tuning is the wrong answer.

**Depends on.** Volumes 4 and 6; Volume 10 concepts are previewed.

**Chapters**

1. **When to Fine-Tune**: problem-first comparison of prompting, RAG, tools, fine-tuning and continued pretraining; cost and maintenance; what fine-tuning cannot fix.
2. **Supervised Fine-Tuning and Instruction Tuning**: chat formatting, loss masking, packing, data quality over quantity, catastrophic forgetting.
3. **Instruction Data**: human-written, Self-Instruct and synthetic pipelines, filtering, diversity, licence hygiene.
4. **Parameter-Efficient Fine-Tuning**: adapters, prefix and prompt tuning, LoRA derivation and implementation from scratch, rank and alpha, target modules, merging.
5. **QLoRA and Memory-Efficient Training**: NF4, double quantisation, paged optimisers; memory arithmetic for fine-tuning 1–70B models.
6. **Continued Pretraining and Domain Adaptation**: data mixing with replay, learning-rate re-warming, forgetting measurement.
7. **Reward Models**: Bradley–Terry preference modelling, training and evaluation, reward hacking and over-optimisation.
8. **RLHF with PPO**: policy, reference, value and reward models; KL penalty; a miniature PPO loop on a toy task; why it is operationally hard.
9. **Direct Preference Optimisation and Variants**: DPO derivation, implementation, IPO, KTO, ORPO, SimPO; offline versus online preference learning.
10. **AI Feedback and Constitutional Methods**: RLAIF, principles, critique-and-revise data.
11. **Distillation and Model Merging**: logit and sequence distillation, imitation limits, model soups, TIES merging.
12. **Production Post-Training**: Hugging Face PEFT and TRL, experiment management, evaluation gates, model cards, release.

**Key concepts.** Loss masking; LoRA (ΔW = BA); QLoRA/NF4; forgetting; Bradley–Terry; KL-regularised
RL; PPO; DPO implicit reward; reward hacking; RLAIF; distillation; merging.

**Python implementations.** `forge.posttrain`: SFT trainer with masking; `LoRALinear` from scratch
with merge and unmerge; reward model; mini-PPO; DPO loss and trainer; distillation loss; PEFT/TRL
equivalents with parity checks on small models.

**Practical project.** *Forge-Instruct.* Instruction-tune a small open model with LoRA on a curated
dataset, train a reward model, run DPO, and compare base, SFT and DPO models with the Volume 10
harness (win-rate, safety and regression suites).

**Research papers.** Howard & Ruder 2018 (ULMFiT); Wei et al. 2021 (FLAN); Chung et al. 2022
(Scaling Instruction-Finetuned LMs); Wang et al. 2022 (Self-Instruct); Zhou et al. 2023 (LIMA);
Houlsby et al. 2019 (Adapters); Li & Liang 2021 (Prefix-Tuning); Lester et al. 2021 (Prompt Tuning);
Hu et al. 2021 (LoRA); Dettmers et al. 2023 (QLoRA); Liu et al. 2024 (DoRA); Gururangan et al. 2020
(Don't Stop Pretraining); Kirkpatrick et al. 2017 (EWC); Christiano et al. 2017; Stiennon et al.
2020; Ouyang et al. 2022 (InstructGPT); Schulman et al. 2017 (PPO); Bai et al. 2022 (HH-RLHF;
Constitutional AI); Gao et al. 2022 (reward over-optimisation); Lee et al. 2023 (RLAIF); Rafailov et
al. 2023 (DPO); Azar et al. 2023 (IPO); Ethayarajh et al. 2024 (KTO); Hong et al. 2024 (ORPO); Meng
et al. 2024 (SimPO); Hinton et al. 2015 (Distillation); Gudibande et al. 2023 (False Promise of
Imitation); Wortsman et al. 2022 (Model Soups); Yadav et al. 2023 (TIES-Merging).

---

## Volume 9 — Reasoning Models and Test-Time Compute

*Verifiable Rewards, Search, Distillation and Thinking Budgets*

**Objective.** Understand how reasoning models are trained and served: chain-of-thought,
sampling-based test-time compute, verifiers and process rewards, RL with verifiable rewards (GRPO),
distillation of reasoning, and the cost and latency implications of thinking tokens.

**Depends on.** Volumes 8 and 10 (concepts); Volume 11 is previewed for cost.

**Chapters**

1. **What "Reasoning" Means Operationally**: multi-step problems, verifiability, the difference between answer quality and trace quality.
2. **Prompted Reasoning**: chain-of-thought, zero-shot CoT, self-consistency, Tree of Thoughts; measured gains and costs.
3. **Test-Time Compute Scaling**: best-of-N, majority voting, verifier reranking, beam and tree search; compute-optimal allocation.
4. **Verifiers and Reward Models for Reasoning**: outcome versus process reward models; programmatic verifiers (unit tests, math checkers).
5. **Bootstrapped Reasoning**: STaR and rejection-sampling fine-tuning.
6. **Reinforcement Learning with Verifiable Rewards**: GRPO derivation and implementation, group baselines, KL, length bias, entropy collapse, DAPO-style fixes.
7. **Reasoning Model Training Recipes**: cold-start SFT, RL stages, distillation into smaller models, publicly documented recipes (DeepSeek-R1, Tülu 3, Kimi k1.5).
8. **Serving Reasoning Models**: thinking budgets, streaming, early exit, cost per solved task, latency SLOs, when not to reason.
9. **Faithfulness and Safety of Reasoning Traces**: unfaithful CoT, reward hacking in RL, monitoring traces.

**Key concepts.** CoT; self-consistency; best-of-N; ORM/PRM; RLVR; GRPO; rejection sampling;
reasoning distillation; thinking budget; cost per solved task; CoT faithfulness.

**Python implementations.** `forge.reasoning`: self-consistency sampler; best-of-N with verifier;
programmatic verifiers for arithmetic and code; a GRPO trainer on a toy arithmetic task at Tier L
(and a small open model at Tier G); budgeted-thinking controller; cost-per-solved-task evaluator.

**Practical project.** *Teach a small model to add.* Use GRPO with a verifiable reward to improve
multi-digit arithmetic accuracy of a small model; compare against SFT on traces and against
best-of-N at equal compute.

**Research papers.** Wei et al. 2022 (CoT); Kojima et al. 2022; Wang et al. 2022 (Self-Consistency);
Yao et al. 2023 (Tree of Thoughts); Cobbe et al. 2021 (Verifiers, GSM8K); Uesato et al. 2022;
Lightman et al. 2023 (Let's Verify Step by Step); Zelikman et al. 2022 (STaR); Shao et al. 2024
(DeepSeekMath, GRPO); DeepSeek-AI 2025 (DeepSeek-R1); Lambert et al. 2024 (Tülu 3); Kimi Team 2025
(k1.5); Yu et al. 2025 (DAPO); Snell et al. 2024 (test-time compute scaling); Muennighoff et al. 2025
(s1); Madaan et al. 2023 (Self-Refine); Huang et al. 2023 (LLMs cannot self-correct reasoning yet);
Turpin et al. 2023 (unfaithful CoT); OpenAI 2024 (Learning to Reason with LLMs, public post).

---

## Volume 10 — Evaluation Engineering

*Metrics, Golden Sets, Judges, Regression and Online Evaluation*

**Objective.** Treat evaluation as a core engineering discipline: define success, build datasets,
choose metrics, quantify uncertainty, build automated judges and calibrate them, run regression
gates in CI, and connect offline evaluation to online experiments.

**Depends on.** Volume 4 (first harness). Used by every later volume.

**Chapters**

1. **What Success Means**: task success versus proxy metrics; the evaluation pyramid (unit, component, system, online, business).
2. **Evaluation Datasets**: golden sets, stratification, edge-case and adversarial sets, synthetic generation, labelling guidelines, versioning.
3. **Metrics**: exact match, F1, BLEU/ROUGE (and their failure), pass@k (unbiased estimator), calibration (ECE), ranking metrics.
4. **Statistics for Evaluation**: confidence intervals, bootstrap, paired tests, power and sample size, multiple comparisons.
5. **Human Evaluation**: rubric design, inter-annotator agreement, pairwise versus absolute, cost.
6. **LLM-as-Judge**: pointwise and pairwise judges, rubric prompts, position and verbosity bias, judge calibration against humans, judge drift.
7. **Benchmarks**: MMLU, GPQA, HumanEval, SWE-bench, IFEval and others; how benchmarks are gamed, saturated and contaminated.
8. **Factuality, Hallucination and Faithfulness**: taxonomy, FActScore, SelfCheckGPT, citation verification.
9. **Evaluating Systems**: RAG evaluation (retrieval and generation), tool-call evaluation, agent trajectory evaluation, safety evaluation.
10. **Performance and Cost Evaluation**: TTFT, TPOT, throughput, tail latency, cost per task, load testing.
11. **Regression Testing and CI**: eval-driven development, gating thresholds, flaky evals, snapshot tests, model-comparison reports.
12. **Online Evaluation**: logging for evaluation, implicit feedback, A/B tests, interleaving, guardrail metrics, closing the loop to datasets.

**Key concepts.** Golden set; stratification; pass@k; bootstrap CI; paired comparison; Cohen's
kappa; judge bias; FActScore; faithfulness; TTFT/TPOT; regression gate; A/B testing.

**Python implementations.** `forge.eval`: dataset registry, metric library, bootstrap and paired
tests, judge framework with bias controls and human-agreement calibration, RAG/agent/tool evaluators,
load-test harness, CI regression gate producing an HTML report.

**Practical project.** *Evaluation platform v1.* A CI-integrated evaluation suite for Forge models
and systems with dashboards, significance testing and a regression gate that blocks bad releases.

**Research papers.** Papineni et al. 2002 (BLEU); Lin 2004 (ROUGE); Ribeiro et al. 2020
(CheckList); Hendrycks et al. 2020 (MMLU); Chen et al. 2021 (HumanEval/Codex); Liang et al. 2022
(HELM); Lin et al. 2021 (TruthfulQA); Zheng et al. 2023 (LLM-as-a-Judge, MT-Bench); Chiang et al.
2024 (Chatbot Arena); Wang et al. 2023 (LLMs are not fair evaluators); Liu et al. 2023 (G-Eval);
Min et al. 2023 (FActScore); Manakul et al. 2023 (SelfCheckGPT); Es et al. 2023 (RAGAS); Saad-Falcon
et al. 2023 (ARES); Jimenez et al. 2023 (SWE-bench); Rein et al. 2023 (GPQA); Zhou et al. 2023
(IFEval); Yao et al. 2024 (τ-bench); Mazeika et al. 2024 (HarmBench); Guo et al. 2017
(calibration); Miller 2024 (Adding Error Bars to Evals); Kohavi et al. 2020 (Trustworthy Online
Controlled Experiments, book).

---

## Volume 11 — Inference Optimisation

*KV Cache, Batching, Quantisation, Speculation and Kernels*

**Objective.** Understand why LLM inference is expensive and make it cheaper and faster: implement
the KV cache, paged KV, continuous batching, quantisation and speculative decoding from scratch;
benchmark them; relate every optimisation to latency, throughput and cloud cost.

**Depends on.** Volumes 1 (roofline), 4 and 7.

**Chapters**

1. **Anatomy of Autoregressive Inference**: prefill versus decode; compute-bound versus memory-bound; TTFT, TPOT and throughput.
2. **The KV Cache**: what it stores, memory formula, implementation, benchmarks, effect on batch size and cost.
3. **Memory Management for KV**: fragmentation, PagedAttention (block tables) from scratch, prefix caching and radix trees, KV quantisation and eviction.
4. **Batching**: static, dynamic, continuous (iteration-level) batching; chunked prefill; scheduler design and fairness.
5. **Quantisation Fundamentals**: number formats, symmetric and asymmetric, per-tensor/channel/group scales, calibration.
6. **Weight Quantisation Methods**: INT8 (LLM.int8(), SmoothQuant), INT4 (GPTQ, AWQ) implemented in miniature; FP8; quality evaluation of quantised models.
7. **Speculative Decoding**: draft-verify algorithm with exact acceptance sampling (implemented and proven); Medusa, EAGLE and n-gram drafting.
8. **Attention Kernels and GPU Optimisation**: GPU memory hierarchy, tiling, online softmax, FlashAttention algorithm (reference implementation), fused kernels, CUDA graphs, `torch.compile`, Triton introduction.
9. **Distributed Inference**: tensor parallel serving, disaggregated prefill and decode, expert parallel serving for MoE.
10. **CPU and Edge Inference**: llama.cpp/GGUF, ONNX Runtime, Apple MLX; quantisation for CPU; when CPU inference is right.
11. **Benchmarking Inference**: methodology, workloads, SLO-constrained throughput (goodput), cost per million tokens.

**Key concepts.** Prefill/decode; KV cache; PagedAttention; prefix caching; continuous batching;
chunked prefill; INT8/INT4/FP8; GPTQ; AWQ; SmoothQuant; speculative decoding; online softmax;
FlashAttention; disaggregation; goodput.

**Python implementations.** `forge.inference`: KV cache, paged KV allocator with block tables,
continuous-batching scheduler, per-channel and group quantisers, mini-GPTQ, mini-AWQ, speculative
decoder with rejection sampling, tiled attention with online softmax (NumPy reference plus Triton
at Tier G), inference benchmark suite.

**Practical project.** *Forge inference engine.* Serve the Volume 4 and 8 models with KV caching,
continuous batching, INT8 weights and speculative decoding; produce a benchmark report comparing
against Hugging Face `generate` and vLLM (Tier G).

**Research papers.** Pope et al. 2022 (Efficiently Scaling Transformer Inference); Yu et al. 2022
(Orca); Kwon et al. 2023 (PagedAttention/vLLM); Zheng et al. 2023 (SGLang); Agrawal et al. 2024
(Sarathi-Serve); Zhong et al. 2024 (DistServe); Patel et al. 2023 (Splitwise); Dettmers et al. 2022
(LLM.int8()); Xiao et al. 2022 (SmoothQuant); Frantar et al. 2022 (GPTQ); Lin et al. 2023 (AWQ);
Dettmers & Zettlemoyer 2022 (4-bit precision); Zhang et al. 2023 (H2O); Leviathan et al. 2022 and
Chen et al. 2023 (speculative decoding); Cai et al. 2024 (Medusa); Li et al. 2024 (EAGLE); Dao et al.
2022 (FlashAttention); Dao 2023 (FlashAttention-2); Shah et al. 2024 (FlashAttention-3).

---

## Volume 12 — Model Serving and Deployment

*Model Servers, APIs, Containers, GPUs and Release Strategies*

**Objective.** Take a model from a checkpoint to a reliable, observable, secure, autoscaled
production API, first with a hand-built FastAPI server and then with production serving engines on
containers and Kubernetes.

**Depends on.** Volume 11; backend and distributed-systems background.

**Chapters**

1. **Serving Requirements**: SLOs, traffic shapes, streaming, multi-tenancy, model versions and adapters.
2. **API Design for LLMs**: chat and completions contracts, streaming (SSE), structured outputs, tool calls, idempotency, errors and rate-limit semantics.
3. **Building a Model Server**: FastAPI plus the Forge engine; async request queue; backpressure; cancellation; health and readiness.
4. **Production Serving Engines**: vLLM, SGLang, TGI, TensorRT-LLM, Triton, Ollama; what each abstracts; selection matrix.
5. **Containers for GPU Workloads**: CUDA images, drivers and toolkit, image size, model weight loading strategies, cold starts.
6. **Kubernetes for Model Serving**: GPU scheduling, device plugins, node pools, KServe-style controllers, autoscaling on queue depth and KV utilisation.
7. **Load Balancing and Gateways**: LLM-aware routing (prefix-cache affinity, least-KV), rate limiting, API gateways, multi-region.
8. **Model Registry and Release Management**: artefact versioning, promotion, shadow, canary, blue-green, rollback, multi-LoRA serving.
9. **Reliability**: timeouts, retries with idempotency, circuit breakers, graceful degradation, fallbacks, load shedding.
10. **Cloud Architecture Patterns**: managed endpoints versus self-hosted, serverless GPUs, reserved versus spot, cost comparison; AWS/GCP/Azure equivalents at concept level.

**Key concepts.** SSE streaming; backpressure; cold start; GPU scheduling; queue-based autoscaling;
prefix-affinity routing; canary; shadow; multi-LoRA; circuit breaker; load shedding.

**Python implementations.** `forge.serve`: OpenAI-compatible FastAPI server with streaming,
continuous-batching backend, cancellation, rate limiter, idempotency keys, health checks; client SDK
with retries and timeouts; load generator; Dockerfile and Kubernetes manifests (validated locally
with kind where possible).

**Practical project.** *Forge serving stack.* Containerised model server with streaming API,
canary release of a new model version, autoscaling policy, and an SLO report under load.

**Research papers.** Crankshaw et al. 2017 (Clipper); Olston et al. 2017 (TF-Serving); Gujarati et
al. 2020 (Clockwork); Kwon et al. 2023 (vLLM); Sheng et al. 2023 (S-LoRA); Dean & Barroso 2013 (The
Tail at Scale); Burns et al. 2016 (Borg, Omega and Kubernetes); Zaharia et al. 2018 (MLflow); Beyer
et al. 2016 (SRE, book).

---

## Volume 13 — Prompt and Context Engineering

*Instructions, Structured Outputs, Context Assembly and Memory*

**Objective.** Engineer the model's input as a system: prompts as versioned, tested artefacts;
structured outputs with constrained decoding; context selection and compression under token
budgets; memory; and injection-aware context construction.

**Depends on.** Volumes 3, 4 and 10.

**Chapters**

1. **The Context Window as an Interface**: roles, system instructions, instruction hierarchy, chat templates revisited.
2. **Prompt Design**: specification-style prompts, decomposition, examples, output contracts, sensitivity to formatting.
3. **Few-Shot and In-Context Learning**: example selection, ordering effects, calibration, retrieval of examples.
4. **Structured Outputs**: JSON schema, function calling, grammar-constrained decoding implemented from scratch (logit masking with a finite-state machine), validation and repair.
5. **Prompt Templates and Management**: templating, versioning, prompt registries, testing, rollout.
6. **Prompt Evaluation and Optimisation**: prompt test suites, automatic prompt optimisation (APE, DSPy-style), overfitting to eval sets.
7. **Context Engineering**: context budgets, selection, ordering, position effects, deduplication, provenance tagging.
8. **Context Compression**: summarisation, extractive compression, token-level pruning, caching-aware layout (prompt caching).
9. **Memory**: short-term, episodic, semantic and procedural memory; write and retrieval policies; forgetting; memory evaluation.
10. **Prompt Injection in Context Construction**: direct versus indirect injection, delimiting and data-marking, instruction hierarchy limits (preview of Volume 17).

**Key concepts.** Instruction hierarchy; in-context learning; constrained decoding; schema
validation; prompt registry; context budget; lost-in-the-middle; prompt caching; memory types;
indirect injection.

**Python implementations.** `forge.context`: prompt template engine with versioning; FSM-based
JSON-constrained decoder for Forge models; schema validation and repair loop; context builder with
budgeting and provenance; compressors; memory store with write policies; prompt regression tests.

**Practical project.** *Structured extraction service.* Extract typed records from messy documents
with schema-constrained decoding, a prompt registry, evaluation suite and injection test cases.

**Research papers.** Brown et al. 2020; Reynolds & McDonell 2021; Zhao et al. 2021 (Calibrate Before
Use); Lu et al. 2021 (Fantastically Ordered Prompts); Min et al. 2022 (Rethinking Demonstrations);
Sclar et al. 2023 (prompt formatting sensitivity); Zhou et al. 2022 (APE); Khattab et al. 2023
(DSPy); Willard & Louf 2023 (Outlines); Liu et al. 2023 (Lost in the Middle); Jiang et al. 2023
(LLMLingua); Packer et al. 2023 (MemGPT); Park et al. 2023 (Generative Agents); Perez & Ribeiro
2022; Greshake et al. 2023 (indirect prompt injection); Wallace et al. 2024 (Instruction Hierarchy).

---

## Volume 14 — Retrieval-Augmented Generation

*Chunking, Hybrid Search, Reranking, Graphs and Agentic Retrieval*

**Objective.** Build RAG systems that are measurably correct: from a naive pipeline to hybrid
retrieval, reranking, query transformation, graph-based and agentic retrieval, with rigorous
evaluation, freshness, access control and cost management.

**Depends on.** Volumes 3, 10 and 13; database knowledge.

**Chapters**

1. **The Knowledge Problem**: problem-first comparison of fine-tuning, RAG, tools and long context; what RAG does not solve.
2. **Naive RAG End to End**: ingest, chunk, embed, index, retrieve, generate, cite; the baseline everything is compared against.
3. **Document Processing and Chunking**: parsing (PDF, HTML, tables), chunking strategies (fixed, recursive, semantic, structural), metadata, contextual chunk headers.
4. **Embedding Models**: bi-encoders, contrastive training, MTEB, domain fine-tuning of embedders, Matryoshka dimensions.
5. **Vector Search**: exact search, IVF, PQ, HNSW (implemented from scratch); recall/latency/memory trade-offs; FAISS; pgvector and dedicated vector databases.
6. **Lexical and Hybrid Search**: BM25 from scratch, hybrid fusion (RRF and weighted), when lexical wins.
7. **Reranking**: cross-encoders, late interaction (ColBERT), LLM rerankers; latency budgets.
8. **Query Understanding**: rewriting, decomposition, HyDE, multi-query, routing across indexes.
9. **Context Construction for RAG**: selection, ordering, compression, citation formatting, conflicting sources.
10. **Graph RAG and Structured Retrieval**: knowledge-graph construction, entity linking, community summaries, text-to-SQL.
11. **Agentic RAG**: iterative retrieval, self-reflection, corrective retrieval, tool-based search.
12. **RAG Evaluation**: retrieval metrics (recall@k, MRR, nDCG), generation faithfulness and answer correctness, end-to-end datasets, component attribution of failures.
13. **Production RAG**: ingestion pipelines, freshness, deletion, document-level access control, multi-tenancy, caching, cost and latency budgets, security (poisoned documents).

**Key concepts.** Chunking; bi-encoder versus cross-encoder; HNSW; IVF-PQ; BM25; RRF; reranking;
HyDE; graph RAG; agentic RAG; recall@k; nDCG; faithfulness; ACL-aware retrieval; index freshness.

**Python implementations.** `forge.rag`: parsers and chunkers; embedding wrapper; brute-force and
HNSW index from scratch; BM25 from scratch; hybrid retriever with RRF; cross-encoder reranker; query
rewriter; context builder with citations; graph extraction; agentic retriever; RAG eval suite;
pgvector-backed production retriever.

**Practical project.** *Enterprise document assistant.* A permission-aware RAG system over a document
corpus with hybrid search, reranking and citations, an evaluation dataset, a failure-attribution
report, and a latency and cost budget.

**Research papers.** Lewis et al. 2020 (RAG); Guu et al. 2020 (REALM); Karpukhin et al. 2020 (DPR);
Izacard & Grave 2020 (FiD); Borgeaud et al. 2021 (RETRO); Robertson & Zaragoza 2009 (BM25); Cormack
et al. 2009 (RRF); Malkov & Yashunin 2016 (HNSW); Jégou et al. 2011 (Product Quantization); Johnson
et al. 2017 (FAISS); Khattab & Zaharia 2020 (ColBERT); Nogueira & Cho 2019 (BERT reranking); Wang et
al. 2022 (E5); Kusupati et al. 2022 (Matryoshka); Thakur et al. 2021 (BEIR); Muennighoff et al. 2022
(MTEB); Gao et al. 2022 (HyDE); Trivedi et al. 2022 (IRCoT); Asai et al. 2023 (Self-RAG); Yan et al.
2024 (CRAG); Sarthi et al. 2024 (RAPTOR); Edge et al. 2024 (GraphRAG); Es et al. 2023 (RAGAS);
Anthropic 2024 (Contextual Retrieval, public post).

---

## Volume 15 — AI Agents

*Loops, Tools, Planning, Memory and Multi-Agent Systems*

**Objective.** Build agents from first principles, starting with the raw loop and then adding
state, memory, planning, tool selection, reflection, replanning and multiple agents. Only after that
are frameworks introduced, with an explanation of what they abstract.

**Depends on.** Volumes 10, 13 and 14.

**Chapters**

1. **Workflows versus Agents**: when autonomy is justified; deterministic workflows first; the cost of agency.
2. **The Minimal Agent Loop**: goal → LLM → action → tool → observation → stop or continue, implemented in ~100 lines with `ScriptedModel` tests.
3. **Tool Design**: tool schemas, descriptions as interfaces, argument validation, idempotency, error messages written for models, the agent-computer interface.
4. **State and Control Flow**: explicit state machines, step limits, termination conditions, checkpointing agent state.
5. **Memory for Agents**: scratchpads, episodic logs, retrieval of past trajectories, summarisation.
6. **Planning**: plan-then-execute, ReAct interleaving, hierarchical plans, plan validation.
7. **Reflection and Replanning**: self-critique, verifier-driven retries, failure recovery; when reflection does not help.
8. **Code as Action**: code-executing agents, sandboxed interpreters (preview of Volume 16).
9. **Multi-Agent Systems**: orchestrator-worker, pipelines, debate, communication protocols (MCP and agent-to-agent), shared state, failure modes of multi-agent systems.
10. **Long-Running Agents and Human Approval**: durable execution, resumability, human-in-the-loop checkpoints, escalation.
11. **Agent Frameworks**: LangGraph and similar frameworks; mapping framework concepts to our implementation; when not to use a framework.
12. **Evaluating Agents**: task success, trajectory quality, tool-call accuracy, cost and steps, reliability (pass^k), benchmark design.

**Key concepts.** Agent loop; tool schema; ACI; state machine; planning; ReAct; reflection; CodeAct;
orchestrator-worker; MCP; durable execution; human approval; pass^k; trajectory evaluation.

**Python implementations.** `forge.agents`: minimal loop; tool registry with JSON-schema validation;
state machine agent; memory; planner and executor; reflection; code-action agent; multi-agent
orchestrator; MCP client and server (minimal); human-approval gate; agent evaluation harness.
Framework comparison implemented in LangGraph.

**Practical project.** *Research agent.* An agent that plans, searches a local corpus and the web
(optional), reads, takes notes, and writes a cited report; evaluated for task success, citation
accuracy, cost and reliability across repeated runs.

**Research papers.** Yao et al. 2022 (ReAct); Schick et al. 2023 (Toolformer); Shinn et al. 2023
(Reflexion); Wang et al. 2023 (Plan-and-Solve; Voyager); Park et al. 2023 (Generative Agents); Qin et
al. 2023 (ToolLLM); Patil et al. 2023 (Gorilla); Wang et al. 2024 (CodeAct); Yang et al. 2024
(SWE-agent); Wu et al. 2023 (AutoGen); Hong et al. 2023 (MetaGPT); Du et al. 2023 (Multiagent
Debate); Liu et al. 2023 (AgentBench); Mialon et al. 2023 (GAIA); Kapoor et al. 2024 (AI Agents That
Matter); Cemri et al. 2025 (Why Multi-Agent Systems Fail); Kwa et al. 2025 (Measuring AI Ability to
Complete Long Tasks); Anthropic 2024 (Building Effective Agents, public post).

---

## Volume 16 — Agent Harnesses

*Runtimes, Permissions, Budgets, Sandboxes and Traces*

**Objective.** Build the runtime that makes agents safe and operable in production: lifecycle,
context management, tool runtime, permissions, budgets, retries, timeouts, sandboxing, execution
policies, tracing and evaluation. The result is a simplified but real agent harness.

**Depends on.** Volumes 15 and 12.

**Chapters**

1. **What a Harness Is**: framework versus harness versus orchestrator versus tool runtime; responsibilities and boundaries.
2. **Agent Lifecycle**: sessions, turns, steps; start, pause, resume, cancel; persistence.
3. **Context Management**: token accounting, compaction, tool-result truncation, context-window overflow handling.
4. **Tool Runtime**: registration, discovery, invocation, streaming results, concurrency, MCP integration.
5. **Permissions and Execution Policies**: capability-based permissions, allow/ask/deny policies, scoped credentials, least privilege.
6. **Budgets and Limits**: token, cost, time and step budgets; rate limits; graceful exhaustion.
7. **Reliability**: retries, timeouts, idempotency, checkpointing, crash recovery.
8. **Sandboxing**: process isolation, containers, gVisor and Firecracker microVMs, filesystem and network policies, resource limits.
9. **Observability**: structured events, traces and spans for LLM and tool calls (OpenTelemetry GenAI conventions), replay.
10. **Human-in-the-Loop Interfaces**: approval prompts, diff review, escalation and audit logs.
11. **Evaluating Harnesses**: simulated environments, adversarial tool outputs, regression suites, safety cases.

**Key concepts.** Session/turn/step; compaction; tool runtime; capability permissions; execution
policy; budgets; idempotent retries; sandbox isolation levels; trace replay; approval gates.

**Python implementations.** `forge.harness`: `AgentRuntime` with lifecycle state machine; context
manager with compaction; tool runtime with MCP adapter; policy engine (allow/ask/deny rules);
budget ledger; retry and timeout decorators; subprocess and container sandbox; OpenTelemetry tracing;
replay tool; approval interface; harness test-bench with adversarial tools.

**Practical project.** *Forge harness v1.* Run the Volume 15 research agent and a coding agent inside
the harness with permissions, budgets, sandboxed code execution, full tracing, and a red-team test
suite.

**Research papers.** Saltzer & Schroeder 1975 (protection principles); Sigelman et al. 2010 (Dapper);
Agache et al. 2020 (Firecracker); Young et al. 2019 (gVisor case study); Yang et al. 2024 (SWE-agent
ACI); Ruan et al. 2023 (ToolEmu); Debenedetti et al. 2024 (AgentDojo); Debenedetti et al. 2025
(CaMeL); Beurer-Kellner et al. 2025 (Design Patterns for Securing LLM Agents); Model Context
Protocol specification (documentation); OpenTelemetry GenAI semantic conventions (documentation).

---

## Volume 17 — Guardrails and AI Security

*Threat Models, Injection, Leakage, Policy and Defence in Depth*

**Objective.** Design and implement defence in depth for AI systems: threat modelling, deterministic
guardrails before model-based ones, injection and jailbreak defences, data-leakage prevention, tool
abuse prevention, supply-chain and training-time threats, and a coherent AI security architecture.

**Depends on.** Volumes 13–16.

**Chapters**

1. **Threat Modelling AI Systems**: assets, trust boundaries, attacker capabilities; OWASP LLM Top 10; MITRE ATLAS; NIST AI RMF.
2. **Input Guardrails**: validation, size and rate limits, encoding normalisation, topic and intent classifiers.
3. **Prompt Injection**: direct and indirect; why it is hard (no data/instruction separation); detection, isolation, dual-LLM and CaMeL-style designs.
4. **Jailbreaking**: attack families (role-play, many-shot, optimisation-based GCG, PAIR); defences and their limits.
5. **Output Guardrails**: schema validation, policy classifiers, groundedness checks, PII redaction, code and URL safety.
6. **PII and Data Leakage**: detection, redaction, tokenisation of secrets, memorisation and extraction, membership inference, cross-tenant leakage.
7. **Tool and Agent Security**: tool permissions, confused-deputy problems, exfiltration channels, sandbox escapes.
8. **Model-Level Threats**: data poisoning, backdoors, model extraction and stealing, supply-chain (weights, pickles, dependencies).
9. **Policy Enforcement**: policy-as-code, a policy engine, human approval, rate and cost limits.
10. **Model-Based Guardrails**: safety classifiers (Llama Guard pattern), LLM judges as guards, latency and cost trade-offs, false-positive management.
11. **AI Security Architecture**: defence-in-depth reference architecture, security testing (red teaming, automated attack suites), incident response.

**Key concepts.** Trust boundary; indirect injection; instruction/data separation; jailbreak; GCG;
output validation; PII; memorisation; membership inference; confused deputy; poisoning; backdoor;
model extraction; policy-as-code; red teaming.

**Python implementations.** `forge.guard`: guardrail pipeline (pre, mid and post); deterministic
validators (size, encoding, regex/checksum PII, schema); injection heuristics and a trained
classifier; output groundedness checker; policy engine; secret scanner; pickle-safety checker;
automated red-team harness with attack corpora; security evaluation metrics.

**Practical project.** *Secure the platform.* Threat-model the Forge RAG agent, implement layered
guardrails, run an automated red-team suite, and report attack success rate against false-positive
rate and latency overhead.

**Research papers.** Perez & Ribeiro 2022; Greshake et al. 2023; Wei et al. 2023 (Jailbroken); Zou et
al. 2023 (GCG); Chao et al. 2023 (PAIR); Anil et al. 2024 (Many-shot Jailbreaking); Perez et al. 2022
(Red Teaming LMs with LMs); Ganguli et al. 2022 (Red Teaming); Inan et al. 2023 (Llama Guard);
Rebedea et al. 2023 (NeMo Guardrails); Wallace et al. 2024 (Instruction Hierarchy); Sharma et al.
2025 (Constitutional Classifiers); Debenedetti et al. 2025 (CaMeL); Carlini et al. 2021 and Nasr et
al. 2023 (training-data extraction); Shokri et al. 2017 (membership inference); Tramèr et al. 2016
and Carlini et al. 2024 (model stealing); Carlini et al. 2023 (poisoning web-scale datasets); Wan et
al. 2023 (instruction-tuning poisoning); Hubinger et al. 2024 (Sleeper Agents); Abadi et al. 2016
(DP-SGD); Kirchenbauer et al. 2023 (Watermarking); OWASP Top 10 for LLM Applications; MITRE ATLAS;
NIST AI RMF 1.0 (documentation).

---

## Volume 18 — Model Routing and Decision Systems

*Capability, Cost, Latency, Confidence, Cascades and Ensembles*

**Objective.** Build routing systems progressively, from a three-line rule router to learned,
confidence-aware, cost-constrained and adaptive routers, and treat routing as a decision-making
problem with measurable quality, cost and latency trade-offs.

**Depends on.** Volumes 10, 12 and 13.

**Chapters**

1. **The Routing Problem**: no single model is optimal; the quality–cost–latency frontier; routing versus cascading versus ensembling.
2. **Rule-Based Routing**: the three-line router, request features, policy tables, and their failure modes.
3. **Capability Routing**: task classification, model capability profiles, modality and context-length constraints.
4. **Cost- and Latency-Aware Routing**: budgets, SLO-constrained selection, queue-aware latency prediction.
5. **Quality-Based and Learned Routing**: preference-data routers, matrix-factorisation and classifier routers, router training data and evaluation.
6. **Confidence Estimation**: log-probability, self-consistency, semantic entropy, verbalised confidence, calibration, conformal methods.
7. **Cascades**: sequential escalation with deferral thresholds, threshold optimisation, expected cost.
8. **Fallbacks and Resilience**: provider failure, rate limits, degraded modes, circuit breakers across providers.
9. **Ensembles**: voting, ranking fusion, mixture-of-agents, generate-then-rank; when ensembles pay off.
10. **Dynamic and Adaptive Routing**: contextual bandits, online learning from feedback, exploration safety.
11. **Decision Systems Beyond Models**: selective prediction, abstention, human escalation, decision-theoretic framing with utility and cost.
12. **Evaluating Routers**: cost–quality curves, AUC over budgets, regret, offline replay evaluation.

**Key concepts.** Pareto frontier; capability profile; deferral; calibration; semantic entropy;
conformal prediction; cascade threshold; fallback; mixture-of-agents; contextual bandit; regret;
selective prediction.

**Python implementations.** `forge.router`: the rule router, then capability, cost and latency
routers, a learned classifier router, confidence estimators, a cascade with threshold optimiser,
a fallback chain with circuit breakers, an ensemble combiner, a LinUCB bandit router and an offline
replay evaluator.

**Practical project.** *Cost-optimal assistant.* Route a mixed workload across three model tiers
(small local, mid open-weight, frontier API) to hit a quality target at minimum cost. Prove the saving
with cost–quality curves and significance tests.

**Research papers.** Viola & Jones 2001 (cascades); Dietterich 2000 (ensembles); Geifman & El-Yaniv
2017 (selective classification); Guo et al. 2017 (calibration); Angelopoulos & Bates 2021 (conformal
prediction); Li et al. 2010 (LinUCB); Kadavath et al. 2022 (LMs mostly know what they know); Kuhn et
al. 2023 (semantic entropy); Chen et al. 2023 (FrugalGPT); Jiang et al. 2023 (LLM-Blender); Shnitzer
et al. 2023 (routing with benchmark data); Ding et al. 2024 (Hybrid LLM); Ong et al. 2024 (RouteLLM);
Hu et al. 2024 (RouterBench); Wang et al. 2024 (Mixture-of-Agents).

---

## Volume 19 — Multimodal AI I: Understanding

*Vision Transformers, CLIP, VLMs, OCR, Speech and Video Understanding*

**Objective.** Extend the Transformer to images, audio and video for understanding tasks. Implement
ViT and contrastive image–text training, build a small vision–language model adapter, and cover
OCR, detection, speech recognition and video understanding with production patterns.

**Depends on.** Volumes 2, 4 and 8.

**Chapters**

1. **Multimodal Problem Framing**: modalities, alignment, fusion strategies (early, late, cross-attention, projection).
2. **Vision Transformers**: patch embedding, class token, position embeddings, ViT from scratch; MAE pretraining.
3. **Image Encoders and Self-Supervision**: DINOv2, SigLIP, resolution handling and dynamic tiling.
4. **Contrastive Image–Text Learning**: CLIP loss from scratch, zero-shot classification, image embeddings and retrieval.
5. **Vision–Language Models**: Flamingo, BLIP-2, LLaVA-style projection; building a mini-VLM adapter on Forge; visual instruction tuning.
6. **Document AI and OCR**: classic OCR pipelines, OCR-free models, layout, tables, visual document retrieval.
7. **Detection and Segmentation**: DETR-style set prediction, YOLO family, SAM; integrating detectors with LLMs.
8. **Visual Reasoning and Evaluation**: VQA, chart and diagram understanding, hallucination in VLMs, benchmarks.
9. **Speech Recognition and Audio Understanding**: spectrograms, CTC, Whisper-style encoder–decoder, audio embeddings, diarisation.
10. **Video Understanding**: frame sampling, spatio-temporal attention, video embeddings, long-video strategies, cost.
11. **Production Multimodal Systems**: preprocessing pipelines, token cost of images, latency, storage, privacy of media.

**Key concepts.** Patch embedding; ViT; MAE; contrastive loss; zero-shot transfer; projection
adapters; visual instruction tuning; OCR-free parsing; set prediction; CTC; log-mel spectrogram;
spatio-temporal attention; image token cost.

**Python implementations.** `forge.multimodal`: ViT from scratch; CLIP-style training on a small
dataset; image retrieval index; mini-VLM (frozen encoder, projection, Forge LM); OCR pipeline
comparison; log-mel and CTC decoder; Whisper inference and evaluation (WER); video frame sampler
and embedding pipeline.

**Practical project.** *Multimodal document assistant.* Answer questions over scanned documents and
charts using OCR, visual retrieval and a VLM, with an evaluation set and a cost analysis per page.

**Research papers.** Dosovitskiy et al. 2020 (ViT); He et al. 2021 (MAE); Oquab et al. 2023
(DINOv2); Radford et al. 2021 (CLIP); Zhai et al. 2023 (SigLIP); Alayrac et al. 2022 (Flamingo); Li
et al. 2023 (BLIP-2); Liu et al. 2023 (LLaVA); Wang et al. 2024 (Qwen2-VL); Carion et al. 2020 (DETR);
Redmon et al. 2015 (YOLO); Kirillov et al. 2023 (SAM); Smith 2007 (Tesseract); Kim et al. 2021
(Donut); Blecher et al. 2023 (Nougat); Faysse et al. 2024 (ColPali); Graves et al. 2006 (CTC);
Baevski et al. 2020 (wav2vec 2.0); Gong et al. 2021 (AST); Radford et al. 2022 (Whisper); Bertasius
et al. 2021 (TimeSformer); Arnab et al. 2021 (ViViT); Tong et al. 2022 (VideoMAE).

---

## Volume 20 — Multimodal AI II: Generation

*Diffusion, Latent Diffusion, Speech Synthesis and Video Generation*

**Objective.** Understand and implement generative models for images, audio and video: diffusion
and flow matching from scratch, latent diffusion, conditioning and guidance, editing, neural codecs
and speech synthesis, video generation and temporal consistency, and the evaluation and safety of
generated media.

**Depends on.** Volumes 2, 6 and 19.

**Chapters**

1. **Generative Modelling Landscape**: VAEs, GANs, autoregressive, diffusion and flow models; trade-offs.
2. **Diffusion from First Principles**: forward noising, reverse process, noise prediction, DDPM from scratch on 2D data and small images.
3. **Sampling**: DDPM versus DDIM, schedulers, step count trade-offs, distillation and consistency models.
4. **Flow Matching and Rectified Flow**: derivation and implementation; why current systems use it.
5. **Latent Diffusion**: autoencoders, latent spaces, U-Net versus DiT backbones.
6. **Conditioning and Guidance**: text conditioning with cross-attention, classifier-free guidance, ControlNet-style control.
7. **Image Editing**: SDEdit, inpainting, attention-based editing, personalisation.
8. **Audio Generation**: neural codecs (EnCodec, SoundStream), token-based audio LMs, TTS pipelines, vocoders, voice cloning risks.
9. **Video Generation**: spatio-temporal diffusion, temporal consistency, compute cost, publicly documented large systems.
10. **Evaluating Generative Media**: FID, CLIPScore, human preference, WER/MOS for speech, temporal consistency metrics.
11. **Serving and Safety**: GPU cost per image and per second, step and resolution trade-offs, content provenance (C2PA), watermarking, misuse.

**Key concepts.** Forward/reverse process; ε-prediction; DDIM; flow matching; latent space; DiT;
cross-attention conditioning; CFG; ControlNet; neural audio codec; temporal consistency; FID;
CLIPScore; provenance.

**Python implementations.** `forge.multimodal.gen`: DDPM and DDIM on 2D toy data and MNIST-scale
images; flow-matching trainer; tiny latent diffusion with a small autoencoder; CFG sampler; Diffusers
pipeline comparison; TTS pipeline integration; evaluation metrics (FID via library, CLIPScore).

**Practical project.** *Tiny text-to-image.* Train a small class- or caption-conditioned latent
flow-matching model at Tier L/G, add CFG, evaluate with FID and CLIPScore, and compare with a
Diffusers pipeline for quality, latency and cost.

**Research papers.** Kingma & Welling 2013 (VAE); Goodfellow et al. 2014 (GAN); van den Oord et al.
2016 (WaveNet) and 2017 (VQ-VAE); Sohl-Dickstein et al. 2015; Ho et al. 2020 (DDPM); Song et al. 2020
(DDIM; Score SDE); Nichol & Dhariwal 2021; Dhariwal & Nichol 2021; Ho & Salimans 2022 (CFG);
Rombach et al. 2021 (Latent Diffusion); Ramesh et al. 2022 (unCLIP); Saharia et al. 2022 (Imagen);
Peebles & Xie 2022 (DiT); Lipman et al. 2022 (Flow Matching); Liu et al. 2022 (Rectified Flow);
Esser et al. 2024 (SD3); Zhang et al. 2023 (ControlNet); Meng et al. 2021 (SDEdit); Hertz et al. 2022
(Prompt-to-Prompt); Song et al. 2023 (Consistency Models); Esser et al. 2020 (VQGAN); Chang et al.
2022 (MaskGIT); Zeghidour et al. 2021 (SoundStream); Défossez et al. 2022 (EnCodec); Borsos et al.
2022 (AudioLM); Wang et al. 2023 (VALL-E); Copet et al. 2023 (MusicGen); Kim et al. 2021 (VITS); Kong
et al. 2020 (HiFi-GAN); Ho et al. 2022 (Video Diffusion Models); Blattmann et al. 2023 (Stable Video
Diffusion); OpenAI 2024 (Video generation models as world simulators, public report); Heusel et al.
2017 (FID); Hessel et al. 2021 (CLIPScore).

---

## Volume 21 — AI Product Architecture and Case Studies

*HLD, LLD and Real-World AI Products*

**Objective.** Design AI products end to end. Produce HLD and LLD for real product classes,
implement their key LLD components in Python, and analyse publicly documented systems with strict
separation of documented facts and inferred architecture.

**Depends on.** Volumes 10–18.

**Chapters**

1. **From Product Problem to AI Architecture**: requirements, SLOs, risk, the architecture decision record for AI.
2. **Reference Architecture**: gateway, orchestration, model services, retrieval services, agent services, tool services, evaluation services, guardrail services, data stores, queues, caches.
3. **LLD Patterns**: model adapters, request context, streaming pipelines, retries and timeouts, idempotency, caching (exact, semantic and prompt caches), error taxonomy.
4. **Data Architecture for AI Products**: conversation stores, feedback stores, vector stores, event logs, analytics, retention and privacy.
5. **Case Study: General Chat Assistants** (ChatGPT, Claude, Gemini): documented features and public reports; inferred HLD; memory, tools, routing, safety layers.
6. **Case Study: Code Completion** (GitHub Copilot): latency constraints, context gathering, FIM, telemetry, documented evaluation.
7. **Case Study: Coding Agents** (Claude Code, Codex-style agents, SWE-agent): harness, tools, permissions, sandboxing, evaluation on SWE-bench.
8. **Case Study: Enterprise Assistants**: permission-aware RAG, connectors, tenancy, compliance.
9. **Case Study: Research Assistants**: deep-research agents, source quality, citation verification, long-running jobs.
10. **Case Study: Customer Support Agents**: workflows versus agents, escalation, policy compliance, deflection metrics, the documented productivity evidence.
11. **Cost Architecture**: unit economics per conversation, caching strategy, routing savings, capacity planning.
12. **Design Review Method**: a checklist-driven review of an AI system design covering quality, safety, cost and operability.

**Key concepts.** HLD/LLD; service boundaries; semantic cache; prompt cache; conversation store;
evidence labels (documented/inferred); unit economics; design review.

**Python implementations.** `services/`: gateway, orchestrator, model-adapter layer, cache layer,
feedback service, and one simplified end-to-end implementation per case study built on Forge
components, with contract tests.

**Practical project.** *Design and build a support agent.* Produce the HLD, LLD, ADRs, evaluation plan
and cost model, then implement it on Forge with guardrails, routing and escalation.

**Sources.** OpenAI GPT-4 Technical Report and System Card 2023; Gemini Team 2023 (Gemini report);
Anthropic model cards and public documentation; Chen et al. 2021 (Codex); Ziegler et al. 2022
(Copilot productivity measurement); Peng et al. 2023 (Copilot productivity experiment); Brynjolfsson
et al. 2023 (Generative AI at Work); Nakano et al. 2021 (WebGPT); Yang et al. 2024 (SWE-agent);
Amershi et al. 2019 (Guidelines for Human-AI Interaction); Parnin et al. 2023 (Building Your Own
Product Copilot); Shankar et al. 2024 (Who Validates the Validators?); Kleppmann 2017 (Designing
Data-Intensive Applications, book).

---

## Volume 22 — AI Infrastructure, Operations and Cost

*GPUs, Cloud, Kubernetes, Observability, Reliability and FinOps*

**Objective.** Operate AI systems in production: GPU infrastructure, cloud platforms, Kubernetes,
observability with OpenTelemetry, Prometheus and Grafana, reliability engineering for model services,
and cost management.

**Depends on.** Volumes 6, 11 and 12.

**Chapters**

1. **GPU Hardware for Engineers**: GPU generations, HBM, NVLink, InfiniBand, MIG; choosing hardware for training versus inference.
2. **Cloud AI Platforms**: AWS, GCP and Azure building blocks at concept level; managed model APIs versus managed endpoints versus raw GPUs; neoclouds.
3. **Kubernetes for AI**: GPU operators, scheduling (gang scheduling, queues), storage for checkpoints and weights, multi-tenant clusters.
4. **Infrastructure as Code**: Terraform concepts, environment promotion, secrets.
5. **Observability Foundations**: metrics, logs, traces; OpenTelemetry instrumentation of LLM and tool calls; semantic conventions.
6. **Metrics and Dashboards**: Prometheus metrics for model services (TTFT, TPOT, KV utilisation, queue depth), Grafana dashboards, alerting on SLOs.
7. **LLM-Specific Monitoring**: quality drift, input drift, guardrail triggers, feedback signals, cost anomalies.
8. **Reliability Engineering**: SLOs and error budgets, incident response, GPU failure modes, capacity planning, chaos testing.
9. **Cost Management**: cost attribution per tenant and feature, utilisation, reserved/spot/on-demand strategy, token budgets, FinOps reviews.
10. **Scaling**: horizontal and vertical scaling, multi-region, disaster recovery, capacity reservations.
11. **Security Operations**: IAM, network isolation, secrets, audit, supply-chain scanning for images and models.

**Key concepts.** HBM bandwidth; MIG; gang scheduling; OpenTelemetry; spans for LLM calls;
Prometheus histograms; SLO/error budget; drift; cost attribution; spot interruption handling.

**Python implementations.** `forge.ops`: OpenTelemetry instrumentation for Forge services;
Prometheus exporters; cost ledger with per-request attribution; drift monitors; SLO calculator;
synthetic canaries. `deploy/`: Docker Compose observability stack (Prometheus, Grafana, OTel
collector, Tempo or Jaeger) runnable locally; Terraform and Kubernetes examples (validated, not
applied).

**Practical project.** *Operate Forge.* Instrument the full Forge stack, run it locally with the
observability stack, define SLOs, create dashboards and alerts, inject failures, and produce a monthly
cost report with attribution.

**Research papers.** Barroso et al. (The Datacenter as a Computer, book); Verma et al. 2015 (Borg);
Sigelman et al. 2010 (Dapper); Dean & Barroso 2013; Beyer et al. 2016 (SRE); Jeon et al. 2019
(multi-tenant GPU clusters); Weng et al. 2022 (MLaaS in the Wild); Hu et al. 2024 (LLM development
in the datacenter); Jiang et al. 2024 (MegaScale); Kokolis et al. 2024 (reliability in ML research
clusters); Dubey et al. 2024 (Llama 3 infrastructure); Patterson et al. 2021 (carbon emissions).

---

## Volume 23 — AI Engineering Tooling and AI-Assisted Development

*The Tool Ecosystem and Working Effectively with Coding Agents*

**Objective.** Build a vendor-neutral, critical understanding of the AI engineering tool ecosystem,
and learn to use AI coding tools to become a better engineer, with verification rather than blind
acceptance.

**Depends on.** Volumes 1–22 (tools are analysed with full context).

**Chapters**

1. **A Framework for Evaluating Tools**: problem solved, mechanism, abstraction, when to use, when not to use, lock-in, total cost.
2. **Model Development Tools**: PyTorch, Hugging Face (Transformers, Datasets, PEFT, TRL, Accelerate), JAX (overview), Lightning.
3. **Experiment Tracking and Registries**: MLflow, Weights & Biases; what to log; reproducibility.
4. **Serving Tools**: vLLM, SGLang, TGI, Ollama, llama.cpp, Triton; comparison matrix.
5. **RAG and Data Tools**: vector databases (pgvector, dedicated engines), search engines, document parsers.
6. **Agent and Orchestration Tools**: LangGraph and alternatives, MCP ecosystem; framework lock-in analysis.
7. **Evaluation and Observability Tools**: eval frameworks, tracing platforms, OpenTelemetry, Prometheus, Grafana.
8. **Deployment Tools**: Docker, Kubernetes, managed platforms.
9. **AI Coding Tools**: Claude Code, Cursor, GitHub Copilot and others; completion versus chat versus agentic workflows.
10. **Repository Context and Agent Instructions**: project documentation for agents (CLAUDE.md/AGENTS.md patterns), context curation, conventions.
11. **AI-Assisted Engineering Workflows**: test-driven development with agents, debugging, refactoring, code review, architecture review, security review.
12. **Validating Generated Code**: dependency verification (package hallucination), static analysis, tests, security scanning, licence checks, review checklists; measured productivity evidence.

**Key concepts.** Abstraction cost; lock-in; experiment lineage; agent instructions; context
curation; TDD with agents; package hallucination; verification gates; measured productivity.

**Python implementations.** Tool comparison benchmarks (the same task in several tools); a generated
code validation gate (ruff, mypy, pytest, bandit-style checks, dependency existence and licence
check); a repository-context generator; an evaluation of AI-generated patches on a local task set.

**Practical project.** *AI-assisted feature delivery.* Implement a non-trivial Forge feature using an
AI coding agent under a documented workflow (spec, tests first, agent instructions, review gates) and
report defects caught by each gate.

**Research papers.** Chen et al. 2021 (Codex); Pearce et al. 2022 (Asleep at the Keyboard); Perry et
al. 2023 (Do Users Write More Insecure Code with AI Assistants?); Ziegler et al. 2022; Peng et al.
2023; Spracklen et al. 2024 (package hallucinations); Becker et al. 2025 (METR developer productivity
study); Jimenez et al. 2023 (SWE-bench); Yang et al. 2024 (SWE-agent); Paszke et al. 2019 (PyTorch);
Wolf et al. 2020 (Transformers); Zaharia et al. 2018 (MLflow); Kwon et al. 2023 (vLLM).

---

## Volume 24 — Capstone: The Complete AI Platform

*Integrating Models, Retrieval, Agents, Guardrails, Routing and Operations*

**Objective.** Integrate every Forge component into one production-grade AI platform, deploy it,
evaluate it end to end, operate it, and run a full continuous-improvement cycle.

**Depends on.** All previous volumes.

**Chapters**

1. **Platform Requirements and Architecture**: capabilities, tenants, SLOs, security, cost targets; final HLD.
2. **Platform LLD**: service contracts, shared libraries, configuration, data model.
3. **Model Layer**: Forge models, fine-tuned adapters, external providers behind one adapter protocol; model registry.
4. **Inference and Serving Layer**: engine, batching, quantised variants, autoscaling.
5. **Knowledge Layer**: ingestion, hybrid retrieval, graph, freshness, ACLs.
6. **Agent Layer**: harness, tools, MCP, approvals.
7. **Control Layer**: guardrails, policy engine, router, budgets.
8. **Evaluation Layer**: offline suites, CI gates, online experiments, feedback loop.
9. **Operations Layer**: observability, SLOs, cost ledger, incident runbooks.
10. **Deployment**: local (Compose), single GPU VM, Kubernetes; cloud reference deployment (documented, applied only with explicit authorisation).
11. **The Continuous Improvement Loop**: failure classification (model versus system), targeted fixes (data, fine-tuning, retrieval, prompts, routing), re-evaluation, release.
12. **Platform Review**: end-to-end design review, threat model, cost review, reliability review, and what to build next.

**Key concepts.** Platform boundaries; contracts; configuration as code; end-to-end evaluation;
continuous improvement; operability.

**Python implementations.** `platform/`: the integrated platform with end-to-end tests, load tests,
red-team suite, and deployment manifests.

**Practical project.** *Forge Platform 1.0.* Deploy locally and optionally to one cloud GPU VM; pass
the evaluation, security and SLO gates; run one improvement cycle from production-like feedback to a
measured release.

**Research papers.** Sculley et al. 2015; Breck et al. 2017 (The ML Test Score); Amershi et al. 2019
(Software Engineering for ML: A Case Study); Paleyes et al. 2022 (Challenges in Deploying ML);
Shankar et al. 2022 (Operationalizing ML: An Interview Study).

---

## Volume 25 — Reference: Research Roadmap, Glossary and Interview Bank

**Objective.** Provide the index layer of the book: every paper card collected in chronological and
thematic roadmaps, the full terminology glossary, and the interview question bank with answer
outlines.

**Chapters**

1. **Research Roadmap by Theme**: Transformers → pretraining → scaling → alignment → efficiency → retrieval → agents → multimodal → reasoning.
2. **Research Roadmap by Year**: 2013–present timeline with dependency links between papers.
3. **Paper Cards**: problem, previous limitation, approach, architecture, main contribution, practical impact, limitations.
4. **Glossary**: precise definitions with cross-references to the volume that teaches each term.
5. **Interview Question Bank**: advanced questions per volume with answer outlines and follow-ups.
6. **Formula and Numbers Reference**: FLOPs, memory, KV cache, throughput and cost formulas; latency numbers every AI engineer should know.

**Note.** Volumes 26–27 are practice volumes that sit after the reference index. Readers may study
them immediately after Volume 23 (tooling) and Volume 21 (product architecture) respectively; the
canonical book order places them after Volume 25 so the reference volume still closes the core
curriculum.

---

## Volume 26 — AI Coding Agents in Practice

*Claude, ChatGPT, Cursor, Kiro: Files, Features and Optimised Workflows*

**Objective.** Teach engineers to use modern AI coding products as a *system*: which tool for which
job; which repo files each tool reads; how to structure rules, skills and MCP; and how to run three
disciplined workflows — learning, vibe prototyping and production delivery with verification —
without confusing them.

**Depends on.** Volumes 1, 15–16 and 23 (tool survey). Complements Volume 27 for customer delivery.

**Chapters**

1. **The Coding-Agent Stack**: chat versus IDE agent versus CLI agent versus cloud agent; where ChatGPT, Claude, Cursor and Kiro sit; when *not* to use an agent.
2. **Feature Matrix**: completion, chat, composer/multi-file edit, terminal/tools, MCP, rules/skills, memory, cloud/background agents, review; cost and context limits as engineering constraints.
3. **The Workspace Module (`forge.devex`)**: canonical layout — `AGENTS.md` / `CLAUDE.md`, `.cursor/rules`, skills, hooks, MCP config, local fakes; one module that every tool can share.
4. **Claude (chat, projects, Claude Code)**: Projects/knowledge, Artifacts, Computer use (where relevant), Claude Code files and permissions, CLAUDE.md patterns, when to prefer Claude for architecture and long reasoning.
5. **ChatGPT (and Codex-style coding)**: Custom GPTs, Projects, memory, Advanced Data Analysis, coding agent modes; using ChatGPT for learning plans, rubber-ducking and light coding without repo coupling.
6. **Cursor**: Tab, Chat, Composer/Agent, `@` context, Rules (user/project), Memories, MCP, Bugbot/Cloud Agents where applicable; optimising context and avoiding infinite edit loops.
7. **Kiro**: Spec-driven / agentic IDE patterns, steering files, hooks and autonomy levels; how it differs from Cursor/Claude Code and when to choose it.
8. **Workflow A — Learning**: spaced prompts, Socratic mode, building mental models, turning chats into notes and tests; anti-patterns (copying without running).
9. **Workflow B — Vibe Coding**: fast prototypes, throwaway branches, time-boxes, when vibe coding is allowed and when it must stop.
10. **Workflow C — Ship with Gates**: spec → tests → agent → review → static analysis → security → PR; measuring defect catch rate per gate (extends Volume 23).
11. **Multi-Tool Playbooks**: handoffs (ChatGPT plan → Cursor implement → Claude review); parallel exploration; conflict resolution when tools disagree.
12. **Team and Enterprise Practice**: shared rules repos, secrets hygiene, customer-code policies, eval of agent productivity without vanity metrics.

**Key concepts.** Coding-agent workspace; rules versus skills versus MCP; context curation; learning
versus vibe versus ship workflows; verification gates; package hallucination; tool lock-in.

**Python implementations.** `forge.devex`: workspace scaffold generator; rules/AGENTS.md linter;
MCP allowlist checker; generated-code gate runner (reuse Volume 23 validators); session checklist
CLI; toy “agent transcript” metrics (files touched, tests run, gate failures).

**Practical project.** *Instrument one Forge feature with three workflows.* Deliver the same small
feature once in learning mode (notes + tests), once as a timed vibe prototype, and once through the
full ship gate; compare defect rates and time.

**Sources / papers.** Pearce et al. 2022; Perry et al. 2023; Spracklen et al. 2024 (package
hallucinations); Peng et al. 2023; Becker et al. 2025 (METR); Yang et al. 2024 (SWE-agent); vendor
docs for Claude Code, Cursor, ChatGPT and Kiro (labelled Documented).

---

## Volume 27 — Forward Deployed AI Engineering

*Discovery, PoC, Pilot and Customer Delivery*

**Objective.** Train Forward Deployed / customer engineers to turn AI platform capabilities into
measured customer outcomes: discovery and scoping, constrained PoCs, production pilots, integration
into customer environments, success metrics, risk and handoff — without confusing demos with
deployments.

**Depends on.** Volumes 10, 14–18, 21–24 and 26.

**Chapters**

1. **What an FDE Is**: role boundaries versus product engineering, solutions consulting and support; engagement lifecycle.
2. **Discovery and Problem Framing**: jobs-to-be-done, decision owners, data reality, hard constraints (latency, residency, SSO); writing an engagement brief.
3. **Scoping and ADRs for Customers**: build versus buy versus configure; PoC exit criteria; killing bad engagements early.
4. **Environment Reality**: customer VPCs, identity, networking, secrets, data samples and synthetic substitutes; security review checklist.
5. **PoC Engineering**: thin vertical slices on Forge; fake versus real connectors; demo hygiene; reproducibility packs.
6. **Evaluation with the Customer**: golden sets co-owned with the customer; offline gates; human review loops; no vanity leaderboards.
7. **Pilot Design**: SLOs, traffic shape, shadow/canary in customer context; escalation and human-in-the-loop.
8. **Integration Patterns**: RAG over customer corpora with ACLs; tool calling into customer APIs; routing and cost caps; guardrails mapped to policy.
9. **Change Management**: training champions, runbooks, deflection metrics for support use cases; documenting productivity claims carefully (Documented vs Inferred).
10. **Handoff and Operability**: runbooks, on-call expectations, cost reports, model/prompt change control; what product engineering owns next.
11. **Failure Modes of FDE Work**: demo-driven development, unbounded scope, unowned data quality, shadow IT models, security theatre.
12. **Engagement Retrospective**: scorecards, reusable assets back into Forge/`services/`, updating the playbook.

**Key concepts.** Engagement brief; PoC exit criteria; customer golden set; shadow pilot; ACL-aware
integration; handoff packet; Documented versus Inferred claims in customer settings.

**Python implementations.** `forge.fde`: engagement brief schema; PoC checklist and scorecard;
synthetic customer corpus generator; pilot SLO dashboard stub; handoff packet builder; connector
contract tests against fakes.

**Practical project.** *Customer support pilot pack.* Run a fictional enterprise support engagement
from brief → PoC → scored pilot plan on the Forge platform, with golden-set eval, guardrails, cost
cap and a handoff packet.

**Sources.** Amershi et al. 2019; Brynjolfsson et al. 2023; Peng et al. 2023; Parnin et al. 2023;
Shankar et al. 2024; Breck et al. 2017; public FDE / solutions-engineering practice notes labelled
Inferred where not documentary.

---

## Volume 28 — Enterprise Integration and Compliance

*SSO, Tenancy, Residency, Private Networking and Procurement Reality*

**Objective.** Make AI platforms deployable inside enterprises: identity (SSO/OIDC), tenancy and
isolation, data residency, private networking and VPC endpoints, secrets and audit, compliance
mappings (SOC 2 / ISO 27001 / HIPAA / GDPR patterns at engineering level), and procurement
constraints that shape architecture — without confusing checklists with legal advice.

**Depends on.** Volumes 12, 14, 17, 22 and 27.

**Chapters**

1. **Enterprise Constraints as Requirements**: latency, residency, identity, audit, change control; ADRs that survive security review.
2. **Identity and Access**: OIDC/SAML SSO, SCIM provisioning, service accounts, least privilege for tools and agents.
3. **Tenancy Models**: shared vs siloed; row-level vs project-level isolation; cross-tenant leakage tests.
4. **Data Residency and Retention**: region pinning, deletion/export, training-data policies, BYOK and CMEK patterns.
5. **Private Networking**: VPC peering / Private Link style patterns, egress allowlists, no public model endpoints.
6. **Secrets, Keys and Audit Trails**: vault integration, short-lived credentials, immutable audit logs for prompts/tools.
7. **Compliance Engineering**: mapping controls to SOC 2 / ISO / HIPAA / GDPR *engineering artefacts* (Documented vs Inferred); DPIA-style checklists.
8. **Procurement and Vendor Risk**: DPAs, subprocessors, model-provider terms, open-weight vs hosted trade-offs.
9. **Enterprise Connectors**: CRM/ITSM/IdP adapters; rate limits; sandbox tenants; contract tests.
10. **Change Management for Regulated Deploys**: CAB, model/prompt change tickets, rollback evidence.
11. **Security Testing for Enterprise Deploys**: tenancy fuzzing, ACL regression, red-team against connectors.
12. **Reference Deploy Pack**: Compose/K8s manifests and runbooks for a private Forge tenant (documented, not applied).

**Key concepts.** OIDC; SCIM; tenancy isolation; residency; private endpoint; BYOK; control mapping;
subprocessor; connector contract.

**Python implementations.** `forge.enterprise`: OIDC claim validator (toy), tenancy context middleware,
residency policy engine, ACL regression suite, connector contract tests, audit-log schema.

**Practical project.** *Private tenant pack.* Configure a Forge RAG+agent stack with SSO claims,
tenant ACLs, residency pins and an audit trail; prove cross-tenant denial with tests.

**Sources.** NIST AI RMF; OWASP LLM Top 10; vendor SSO/OIDC docs (Documented); enterprise security
guidance labelled Inferred where not documentary. Not legal advice.

---

## Volume 29 — Realtime and Voice Agents

*Duplex Audio, Barge-in, Telephony and Sub-second SLOs*

**Objective.** Build voice and realtime agents: full-duplex sessions, barge-in, turn-taking,
telephony integration patterns, streaming ASR/TTS and LLM tool loops under hard latency budgets —
and know when text chat is the better product.

**Depends on.** Volumes 11–12, 15–16 and 19 (speech understanding overview).

**Chapters**

1. **Realtime Problem Framing**: half-duplex vs full-duplex; conversational latency budgets; when voice fails.
2. **Session and Media Graph**: capture → ASR → LLM/agent → TTS → playback; jitter buffers; clock domains.
3. **Streaming ASR and Partial Hypotheses**: interim results, endpointing, confidence, language codes.
4. **Streaming TTS and Prosody**: chunking, barge-in cancellation, voice cloning risk/ethics.
5. **Barge-in and Turn-taking**: VAD, interruption policies, state machines that do not double-speak.
6. **LLM Loop under Latency**: speculative replies, tool call deferral, filler strategies (honesty constraints).
7. **Telephony Patterns**: SIP/WebRTC concepts, PSTN gateways, call recording retention, DTMF.
8. **Realtime Safety**: prompt injection via audio, social engineering, PII in transcripts, kill switches.
9. **Observability for Calls**: TTFT analogues (time-to-first-audio), barge-in rate, talk ratio, cost per minute.
10. **Evaluation**: scenario scripts, interruption stress tests, ASR WER vs task success.
11. **Architecture Variants**: cascaded ASR-LLM-TTS vs speech-to-speech models; trade-offs (Documented/Inferred).
12. **Production Hardening**: concurrency, regional failover, codec choices, load shedding mid-call.

**Key concepts.** Full-duplex; barge-in; VAD; endpointing; time-to-first-audio; media graph;
speech-to-speech vs cascade.

**Python implementations.** `forge.realtime`: session state machine, barge-in simulator, latency
budget calculator, synthetic call graph, evaluation harness for interruption scenarios (audio I/O
mocked at Tier L).

**Practical project.** *Voice support slice.* Simulate a duplex support agent with barge-in and tool
calls under a 800 ms time-to-first-audio budget; report breach rates.

**Sources.** Whisper / speech papers from Volume 19; WebRTC docs; vendor realtime APIs labelled
Documented; telephony patterns labelled Inferred where simplified.

---

## Volume 30 — Synthetic Data and Eval Data Factories

*Generators, Filters, Contamination Control and Preference Factories*

**Objective.** Build factories that produce training and evaluation data safely: generation
pipelines, quality filters, diversity and coverage, contamination control against eval sets,
preference/SFT/synthetic reasoning data, and governance so synthetic data does not silently poison
benchmarks or customer evals.

**Depends on.** Volumes 5, 8–10 and 14.

**Chapters**

1. **Why Synthetic Data**: coverage holes, privacy, rare classes; what synthetic cannot invent.
2. **Generator Designs**: prompt templates, multi-agent debate, retrieval-augmented synthesis, verifiers.
3. **Quality Filters**: heuristics, model judges, cross-model agreement, human spot checks.
4. **Diversity and Dedup**: near-duplicate detection, topic coverage, taxonomy-driven sampling.
5. **Contamination Control**: n-gram / embedding overlap with held-out evals; quarantine workflows.
6. **SFT and Instruction Factories**: Self-Instruct style pipelines with licence and PII scrubbing.
7. **Preference and Critique Data**: pairwise generation, RLAIF-style critiques, length bias traps.
8. **Reasoning Trace Factories**: verified maths/code traces; rejection sampling; faithfulness risks.
9. **RAG and Domain Corpora**: synthetic docs vs real docs; ACL-safe substitutes for FDE PoCs.
10. **Eval Set Factories**: adversarial and stratified sets; versioning; customer co-ownership (ties to V27).
11. **Governance and Lineage**: manifests, seeds, model versions used to generate, release gates.
12. **Failure Modes**: model collapse, echo chambers, benchmark hacking, licence laundering.

**Key concepts.** Synthetic factory; contamination; Self-Instruct; RLAIF data; verified traces;
lineage manifest; model collapse.

**Python implementations.** `forge.synth`: template generators, filter stack, MinHash/eval overlap
checker (reuse `forge.data`), preference pair builder, lineage manifest, factory CLI.

**Practical project.** *Eval-safe instruction factory.* Generate 1k synthetic instruction pairs with
filters, prove &lt;τ overlap with a frozen eval set, and ship a lineage manifest.

**Sources.** Wang et al. 2022 (Self-Instruct); Gunasekar et al. 2023 (phi / textbook quality);
Shumailov et al. 2023 (model collapse); contamination literature from Volumes 5 and 10; Bai et al.
2022 (Constitutional AI / RLAIF).

---

# Cross-cutting maps

## Tool coverage map (Area 25)

| Tool | Introduced | Analysed in depth |
|------|-----------|-------------------|
| PyTorch | V2 | V4, V6, V11, V23 |
| Hugging Face (Transformers, Tokenizers, Datasets, PEFT, TRL, Accelerate) | V3 | V4, V8, V23 |
| vLLM / SGLang / TGI / llama.cpp / Ollama | V11 | V12, V23 |
| FastAPI | V12 | V21, V24 |
| Docker / Kubernetes | V12 | V22, V24 |
| MLflow / Weights & Biases | V4 | V23 |
| LangGraph and agent frameworks | V15 | V23 |
| Vector databases, pgvector, FAISS | V14 | V23 |
| OpenTelemetry / Prometheus / Grafana | V16 | V22 |
| Claude Code / Cursor / GitHub Copilot | V1 | V23, **V26** |
| ChatGPT (coding / learning workflows) | V23 | **V26** |
| Kiro | V26 | **V26** |
| MCP (as coding-agent context) | V15, V23 | **V26** |

## Terminology map (Area 37)

Defined in Volume 1 Chapter 2, refined in the volume that implements each concept, and collected
in Volume 25: AI · Machine Learning · Deep Learning · Generative AI · Foundation Model · LLM · SLM ·
Multimodal Model · AI Application · AI Workflow · AI Agent · Agentic System · Agent Framework ·
Agent Harness · Orchestrator · Tool Runtime · Guardrail · Evaluation System · AI Platform.

## Research paper policy

- Paper lists above are the **plan**. A paper is cited in a chapter only after its entry is added
  to `shared/references.json` with full bibliographic data and a URL.
- `validate_book.py` fails the build if a chapter cites an unknown key.
- Every major paper gets a paper card (problem, previous limitation, approach, architecture, main
  contribution, practical impact, limitations) in the volume where it matters most. Volume 25
  collects them.

## Build order

```text
Stage 0  Architecture, curriculum, style, design system, build, validation   <- current
Stage 1  Volume 1 (then validate, review, refine the system)
Stage 2  Volumes 2-4   (model foundations; Forge becomes a trainable LLM)
Stage 3  Volumes 5-9   (data, scale, architectures, post-training, reasoning)
Stage 4  Volumes 10-12 (evaluation, inference, serving)
Stage 5  Volumes 13-18 (context, RAG, agents, harness, guardrails, routing)
Stage 6  Volumes 19-20 (multimodal)
Stage 7  Volumes 21-24 (products, operations, tooling, capstone)
Stage 8  Volume 25 (reference index) and the combined edition
Stage 9  Volumes 26-27 (coding-agent practice, Forward Deployed delivery)
Stage 10 Volumes 28-30 (enterprise integration, realtime/voice, synthetic data factories)
```
