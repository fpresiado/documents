# TITANS BIBLE: A Claude-Code-Ready Reference for Building a Local Titans-Style Memory AI (Sept 2026)

Short answer: you can build a working Titans-style system on a Framework Desktop (Ryzen AI MAX+ 395, gfx1151, 128 GB). Google has published no official Titans code or weights, so everything here is a reimplementation or an inference from the papers. The plan most likely to work has two tracks. First, train a small MAG/MAC model from scratch (about 30M–170M parameters) to validate the memory mechanism. Second, add a Titans neural-memory adapter to a frozen open-weight LLM (Qwen/Llama/Gemma) for the agent you actually use. Do not try to reproduce the paper's 760M/30B-token results at home.

## TL;DR

- **What Titans is.** A small MLP whose weights are trained *while the model runs*. On every chunk of tokens it takes a gradient step on ‖M(k)−v‖², using momentum ("past surprise") and a learned forgetting gate. That memory is combined with sliding-window/segment attention as short-term memory, plus learned persistent tokens. In the paper's tables MAC is the best variant for long context. The 16K-token S-NIAH, BABILong and 2M-token results are author-reported and have not been independently reproduced.
- **Code.** No official code exists: the paper says only "we intend to make the code ... available soon". Use `lucidrains/titans-pytorch` (latest v0.5.5, 13 Jul 2026, torch ≥ 2.8) as a reference, pinned. For your own system, prefer the from-scratch PyTorch implementation in §7 so you control every gate, chunk size and state-saving path. `flash-linear-attention` provides fast kernels for Gated DeltaNet and similar models, but its model list has no Titans entry.
- **Hardware.** On gfx1151, use TheRock nightly PyTorch wheels (`rocm.nightlies.amd.com/v2/gfx1151/`). Set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` before importing torch. Do not set `HSA_OVERRIDE_GFX_VERSION` on native builds. Expect bf16 bugs and occasional GPU hangs, and keep memory weights and momentum in fp32. Use the Threadripper for data prep, evals and CI.

---

# PART 0 — CLAUDE.md (paste at repo root)

```markdown
# CLAUDE.md — titan-local
## Mission
Local, zero-cloud AI with a Titans-style neural long-term memory (test-time-trained MLP): first a small
from-scratch LM (MAG, then MAC), then an adapter on a frozen open-weight LLM, with memory persisted to disk.

## Ground rules (NEVER violate)
1. No network calls at runtime. Assets are fetched once by scripts/fetch_assets.py only.
2. Memory weights, momentum, gates and associative loss run in fp32 even when the backbone is bf16.
3. Every memory update is causal: chunk c reads only memory produced by chunks < c.
4. Pin every dependency; never upgrade without updating requirements.lock and running tests.
5. Before `import torch` on Strix Halo: export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1.
   Do NOT set HSA_OVERRIDE_GFX_VERSION with native gfx1151 wheels.
6. Never wrap NeuralMemory in activation checkpointing / saved-tensor hooks (torch.func.grad incompatibility).
7. "Done" = `pytest -q` passes AND the phase gate in ROADMAP.md passes.
8. Memory state files are bound to a checkpoint hash; refuse to load on mismatch.
9. Log every eval to runs/<name>/metrics.jsonl; never hand-edit results.

## Build order (each phase has a gate; see Part 12)
P0 env → P1 NeuralMemory tests → P2 toy recall → P3 MAG LM → P4 passkey/NIAH → P5 MAC → P6 persistence
→ P7 frozen-LLM adapter → P8 agent integration → P9 optimizations.
```

---

# PART 1 — Core Theory (exact math, arXiv 2501.00663)

## 1.1 Memory perspective

Titans splits memory into three roles.

- **Attention is short-term memory.** It keeps the full KV history uncompressed. That makes it accurate, but its cost is quadratic and it only covers the current window.\[1\]\[2\]\[3\]
- **The deep neural memory is long-term memory.**\[1\] Linear RNNs and linear attention squeeze history into a vector or matrix. The paper argues that "a very long context cannot be properly compressed in a small vector-valued or matrix-valued states."\[3\]
- **Learnable, data-independent tokens are persistent (task) memory.**\[3\]

## 1.2 Associative loss and projections

Keys and values are projections of the input: k_t = x_t W_K, v_t = x_t W_V (Eq. 11).\[3\]

The memory is trained on the loss ℓ(M_{t−1}; x_t) = ‖M_{t−1}(k_t) − v_t‖²₂ (Eq. 12).\[3\]

Retrieval is a plain forward pass with no weight update: q_t = x_t W_Q, y_t = M*(q_t) (Eq. 15).\[3\]

W_K, W_V and W_Q are outer-loop parameters, learned in normal training. The memory weights are the inner loop, learned at test time.\[3\] Implementation details (§4.4):
- SiLU on q, k and v, with ℓ2-normalized q and k.\[4\]
- A 1D depthwise-separable convolution after each q/k/v projection.\[4\]
- Residual connections everywhere.\[4\]
- Normalization plus a linear gate before the output projection.\[4\]

## 1.3 Surprise, momentum, forgetting

The simplest update is plain gradient descent: M_t = M_{t−1} − θ_t ∇ℓ(M_{t−1}; x_t) (Eq. 8).\[3\]

The problem is that after a burst of surprising tokens the gradients can shrink toward zero, and whatever comes next is missed.\[3\] Titans therefore adds momentum:

- S_t = η_t S_{t−1} − θ_t ∇ℓ(M_{t−1}; x_t) (Eq. 10). The first term is **past surprise**, the second **momentary surprise**.\[3\]
- M_t = (1 − α_t) M_{t−1} + S_t (Eq. 13), where α_t ∈ [0,1] is the **adaptive forgetting (weight-decay) gate**.\[3\]\[5\]

All three gates are data-dependent functions of x_t:
- **η_t** controls surprise decay. η → 0 ignores past surprise (useful at a context switch); η → 1 carries it forward.\[3\]
- **θ_t** is the inner learning rate.
- **α_t** controls forgetting. α → 0 keeps the memory; α → 1 wipes it.\[3\]

The paper notes that this is exactly mini-batch gradient descent with momentum and weight decay on a meta-network.\[6\] The α gate generalizes the forget gates of Mamba-2 and Gated DeltaNet.\[3\]

## 1.4 Why deep memory (L_M ≥ 2) beats linear memory

With a matrix memory, the objective ‖W k_t − v_t‖² is online linear regression, so it can only capture linear dependencies. An MLP with two or more layers is strictly more expressive (Hornik et al.).\[3\]\[7\] The paper's evidence:

- **Table 5:** replacing deep memory with linear memory raises perplexity from 27.01 to 28.49 and drops long-context accuracy from 92.68 to 85.34.\[4\]
- **Figure 7:** perplexity improves with depth (L_M = 1→4) at every sequence length.\[8\]
- **Figure 8:** throughput falls linearly with depth.\[4\]

**Default: L_M = 2; go to 3 at most.**

## 1.5 Persistent memory

N_p learnable tokens are prepended to the input: x_new = [p_1 … p_{N_p}] ‖ x (Eq. 19).\[4\]

They serve three purposes: they store task knowledge that does not depend on the input, they act like data-independent FFN "attention" (Sukhbaatar et al.), and they redistribute the causal-attention bias that otherwise piles onto the first tokens. Removing them costs 27.01 → 27.63 perplexity.\[4\]

## 1.6 Ablation (Table 5)

| Variant | ppl ↓ | reasoning ↑ | long-ctx ↑ |
|---|---|---|---|
| LMM | 27.01 | 47.83 | 92.68 |
| +Attn (MAC) | 26.67 | 48.65 | 97.95 |
| +Attn (MAG) | 25.70 | 48.60 | 96.70 |
| +Attn (MAL) | 25.91 | 47.87 | 96.91 |
| Linear memory | 28.49 | 46.97 | 85.34 |
| w/o convolution | 28.73 | 45.82 | 90.28 |
| w/o momentum | 28.98 | 45.49 | 87.12 |
| w/o weight decay | 29.04 | 45.11 | 85.60 |
| w/o persistent memory | 27.63 | 46.35 | 92.49 |

**What this means:** forgetting and momentum contribute the most. If you have to cut features, keep those two and give up memory depth beyond 2 first. The paper also proves (Theorem 4.1) that Titans can solve problems beyond TC⁰, which makes them more expressive than Transformers on state-tracking tasks.\[4\]

---

# PART 2 — Variants: MAC, MAG, MAL, LMM

## 2.1 Memory as Context (MAC): best for long context

Split the sequence into segments S^(t) and, for each segment:

1. Retrieve history: h_t = M*_{t−1}(q_t) (Eq. 21).\[4\]
2. Build the attention input: S̃ = [P ‖ h_t ‖ S^(t)] (Eq. 22).\[4\]
3. Attend: y_t = Attn(S̃) (Eq. 23).\[4\]
4. Write to memory: M_t = M_{t−1}(y_t) (Eq. 24).\[4\]
5. Output: o_t = y_t ⊗ M*_t(y_t) (Eq. 25).\[4\]

```
S(t) ─► q ─► M*_{t-1} ─► h_t ─┐
P ────────────────────────────┼─► [P|h_t|S(t)] ─► causal Attn ─► y_t ─► write M ; o_t = y_t ⊗ M*_t(y_t)
```

Attention decides whether the retrieved history is useful, and it also filters what gets stored. At test time P stays fixed, attention learns in context, and M keeps learning.\[4\]

## 2.2 Memory as Gate (MAG)

x̃ = P ‖ x; y = SW-Attn*(x̃); o = y ⊗ M(x̃) (Eqs. 26–28).\[4\]

There is no segmentation. The two branches run in parallel, and both outputs are normalized with learnable vector weights and then combined through a nonlinearity. SWA acts as precise short-term memory and the neural memory as a fading long-term memory.\[4\]

## 2.3 Memory as Layer (MAL)

x̃ = P ‖ x; y = M(x̃); o = SW-Attn(y) (Eqs. 29–31).\[4\]

This is the standard hybrid stacking pattern. It is the fastest variant, since it can use FlashAttention, but "the power of the model is limited by each of the layers."\[4\]

## 2.4 LMM

The neural memory used alone, with no attention.\[4\]

## 2.5 Reported numbers (author-reported)

760M params / 30B tokens (Table 1):

| Model | Wiki ppl ↓ | LMB ppl ↓ | Avg acc ↑ |
|---|---|---|---|
| Transformer++ | 25.21 | 27.64 | 48.69 |
| Mamba2 | 22.94 | 28.37 | 48.34 |
| Gated DeltaNet | 21.18 | 22.09 | 49.69 |
| Gated DeltaNet-H2* | 19.88 | 20.83 | 51.49 |
| Titans (LMM) | 20.04 | 21.96 | 51.56 |
| Titans (MAC) | 19.93 | 20.12 | 52.51 |
| Titans (MAG) | 18.61 | 19.86 | 52.50 |
| Titans (MAL) | 19.07 | 20.33 | 50.97 |

The MAL row reports SIQA = 30.98, against about 40 for its sibling variants.\[4\] That looks like an outlier or a typo.

S-NIAH (RULER) at 16K (PK / N / W):

| Model | PK | N | W |
|---|---|---|---|
| MAC | 98.4 | 97.4 | 95.2 |
| MAG | 97.4 | 98.6 | 88.2 |
| MAL | 97.8 | 96.4 | 90.4 |
| LMM | 96.2 | 80.2 | 80.6 |
| TTT | 88.4 | 4.4 | 0.0 |
| Mamba2 | 5.4 | 0.0 | 0.0 |
| DeltaNet | 71.4 | 5.4 | 0.0 |

**Decision:** MAG ≈ MAC on language modeling, MAC is better at long-context recall, and MAL is fastest.\[4\] **Build MAG first, because it is simplest, then MAC.**

---

# PART 3 — Training and Parallelization

## 3.1 Chunked (tensorized) mini-batch gradient descent

Split the sequence into chunks of size b. Every gradient inside a chunk is taken at the weights from the start of the chunk (t′ = t − mod(t, b)):\[3\]

M_t = β_t M_0 − Σ_{i≤t} θ_i (β_t/β_i) ∇ℓ(M_{t′}; x_i), β_i = Π_{j≤i}(1 − α_j) (Eq. 16).\[3\]

For linear memory the chunk sum reduces to matmuls, Θ_b B_b (W_0 X − X) Xᵀ (Eq. 17), and the same idea extends to MLPs. This builds on the TTT dual-form trick (Sun et al., arXiv 2407.04620).\[3\]

## 3.2 Momentum via parallel associative scan

Once the per-token gradients u_t are known, S_t = η_t S_{t−1} − θ_t u_t is a linear recurrence, so an associative scan can compute it. If the gates are constant within a chunk, the system is linear time-invariant and a global convolution works. The paper's experiments used per-token gates.\[3\]

The net effect: the recurrence is **non-linear across chunks and linear within a chunk**. That is what separates it from Gated DeltaNet.\[4\]

## 3.3 Chunk-size tradeoff

- **Small chunks (4–16):** finer updates and better quality, but lower GPU utilization.
- **Large chunks (64–512):** faster, but the updates get blurred.\[9\]
- TNT (arXiv 2511.07343, ICLR 2026) describes this directly: "large chunks boost speed but degrade performance, necessitating a fixed, suboptimal compromise."\[10\]
- MIRAS reports that b "usually is 16 or 64".\[11\]
- lucidrains' `train_mac.py` uses `NEURAL_MEM_SEGMENT_LEN = 4` and `NEURAL_MEM_BATCH_SIZE = 128`.\[12\]

Local default: 16, then sweep {8, 16, 32, 64}.

## 3.4 TNT (Google, 2025)

TNT trains in two stages.

- **Stage 1:** a hierarchical memory. A global memory processes large chunks, and several parallel local memories are periodically reset, which breaks the sequential dependency.\[13\]
- **Stage 2:** a short fine-tune.

It also projects queries onto the span of previously seen keys.\[9\] Evaluated on Titans and TTT, the TNT abstract (arXiv 2511.07343) says it "achieves a substantial acceleration in training speed-up to 17 times faster than the most accurate baseline configuration - while simultaneously improving model accuracy." Borrow this if throughput becomes your bottleneck.

## 3.5 Relatives (Appendix C)

- **Gated DeltaNet** is the special case of LMM with linear memory and η = 0. Titans adds momentum, deep memory and non-linear inter-chunk recurrence.\[4\]
- **TTT** has no forgetting and no momentum.\[4\]
- **RWKV-7** uses the same loss formulation.\[4\]

## 3.6 Paper setup

- **Model sizes:** 170M/340M/400M trained on **15B** FineWeb-Edu tokens, 760M on **30B**. The OpenReview version adds a **1.3B model trained on 100B** tokens.\[3\]\[8\]
- **Recipe:** Llama-2 tokenizer (32K vocab), 4K training length, AdamW at lr 4e-4 with a cosine schedule, 0.5M-token batches, weight decay 0.1. The depth ablation used a subset of the Pile.\[4\]
- **Evaluations:** Wikitext, LAMBADA, PIQA, HellaSwag, WinoGrande, ARC-e/c, SIQA, BoolQ.\[4\]

---

# PART 4 — Benchmarks: Claims vs Reproduction

| Claim | Source | Status |
|---|---|---|
| Beats Transformer++ and linear RNNs at 340M–760M | Table 1 | Author-reported; 400M baselines taken from the Gated DeltaNet paper |
| S-NIAH 16K ≥ 95% (MAC) | Table 2 | Author-reported |
| BABILong: fine-tuned MAC beats GPT-4, Qwen2.5-72B, Llama3.1-8B+RAG ("about ×70 less parameters") | Fig. 6 | Author-reported; baselines "reported by (Yuri Kuratov et al. 2024)" |
| >2M context | Abstract | Author-reported; no public model |
| Time series (e.g., ETTm1 MSE 0.358 vs Simba 0.383) | Table 3 | Author-reported |
| DNA (Enhancer Cohn 75.2 vs HyenaDNA 74.2) | Table 4 | Baselines reused from Arora et al. |
| Independent reimplementation | "Titans Revisited" (arXiv 2510.09551; Di Nepi, Siciliano and Silvestri, 10 Oct 2025; evaluated on masked language modeling, time-series forecasting and recommendation, citing "the lack of publicly available code and ambiguities in the original description") | "Titans does not always outperform established baselines due to chunking. However, its Neural Memory component consistently improves performance compared to attention-only models." |
| Fact storage vs recall | Knowledge Objects (arXiv 2603.17781) | MAC at dim 128: 100% memorization, but only 0–40% free-form completion |\[14\]

**Interpretation:** the mechanism itself holds up. The headline gains are not verified at scale. Trust your own gates (§12).

---

# PART 5 — Follow-up Work (and what to borrow)

## 5.1 MIRAS: "It's All Connected" (arXiv 2504.13173)

MIRAS describes any sequence model through four choices:\[15\]
1. **Memory architecture**
2. **Attentional bias** (the inner objective)
3. **Retention gate** (forgetting, recast as regularization)
4. **Memory learning algorithm** (the optimizer)\[16\]

| Variant | Memory | Attentional bias | Retention |
|---|---|---|---|
| **Moneta** | 2-layer MLP | ℓp | ℓq + ℓ2 |
| **Yaad** | MLP | Huber | local + global (Titans-like) |
| **Memora** | MLP | ℓ2 | KL divergence (simplex memory) |

These were reported at 340M, 760M and 1.3B.\[17\] The paper's observation is that "most existing sequence models leverage either (1) dot-product similarity, or (2) L2 regression objectives."\[18\] **Borrow:** Yaad's Huber loss is a one-line change that makes writes robust to outlier tokens.

## 5.2 ATLAS (arXiv 2505.23735)

- **Omega rule:** optimize memory over a sliding window of the last c tokens instead of only the current one.\[19\]
- **Polynomial feature maps** on q and k.\[20\]
- **Muon** as the inner optimizer, for "locally optimal" memory updates.\[19\]\[21\]
- Introduces DeepTransformers/SWDT (strict generalizations of Transformers), Dot and OmegaNet.\[21\]\[22\]
- Claims "+80% accuracy in 10M context length" on BABILong.\[21\] An independent review (Pith) notes that the main-table baselines were imported, not rerun.\[23\]

**Borrow:** an Omega window with c = 4–16. It is cheap.

## 5.3 Google blog, "Titans + MIRAS: Helping AI have long-term memory" (Dec 2025)

The official overview, written by Ali Behrouz, Meisam Razaviyayn and Vahab Mirrokni (VP and Google Fellow). It covers surprise-gated memory with momentum and adaptive weight decay, and defines MIRAS "through four key design choices." Secondary coverage reports wins over Transformer++, Mamba-2 and Gated DeltaNet, and a daily.dev summary says Titans/MIRAS let models "handle massive contexts (2+ million tokens) through dynamic memory updates during runtime." No code was released.

## 5.4 Nested Learning / HOPE (arXiv 2512.24695)

- **Continuum Memory System:** MLP blocks updated at different frequencies.\[24\]
- **HOPE:** "a variant of the Titans architecture" that modifies itself.\[24\]
- **Borrow:** multi-frequency memory. Update one memory per chunk, one per turn, and one nightly.

## 5.5 2026 work

- **Memory Caching** (arXiv 2602.24281, ICML 2026): caches memory-state checkpoints per segment, so capacity grows with length, sitting between O(L) and O(L²).\[25\]\[26\] **Borrow:** snapshot memory per session or topic and retrieve over the snapshots.
- **Proteus** (arXiv 2608.16844, 17 Aug 2026): incremental memory activation, which unlocks memory blocks as the context grows. It adds no parameters and improves SWLA, Comba, Titans and Hope-Attention.\[5\]
- **In-Place TTT** (arXiv 2604.06169): uses the Transformer's own MLPs as fast weights. Relevant for retrofitting.\[27\]

## 5.6 Baselines

**Gated DeltaNet** (arXiv 2412.06464) is in Qwen3-Next via fla and is the strongest cheap alternative.\[28\] Also compare against Mamba-2 and TTT. **Rule:** your Titans layer must beat fla Gated DeltaNet on your NIAH gate before you invest further.

---

# PART 6 — Open-Source Implementations and Tools

## 6.1 lucidrains/titans-pytorch (unofficial, MIT)

**Versions.** Latest is **v0.5.5, released 13 Jul 2026**. The earlier 2026 releases were 0.5.0 (7 Jan), 0.5.1 (27 Jan) and 0.5.3 (9 Feb).\[29\]\[30\] The repo has about 2k stars and 208 forks.\[31\]\[32\]

**Dependencies (v0.5.0):** `torch>=2.8`, `assoc-scan>=0.0.4`, `axial_positional_embedding>=0.3.10`, `einops>=0.8.0`, `einx>=0.3.0`, `hyper-connections>=0.3.11`, `Ninja`, `rotary-embedding-torch`, `tensordict`, `tqdm`, `x-transformers`.\[33\]

**Exports:** `NeuralMemory`, `NeuralMemState`, `mem_state_detach`, `MemoryMLP`, `MemoryAttention`, `FactorizedMemoryMLP`, `MemorySwiGluMLP`, `GatedResidualMemoryMLP`, `MemoryAsContextTransformer`.\[34\]

```python
# pip install titans-pytorch==0.5.5
import torch
from titans_pytorch import NeuralMemory, MemoryAsContextTransformer, MemoryMLP

mem = NeuralMemory(dim=384, chunk_size=64).cuda()      # 'cuda' == HIP device on ROCm
seq = torch.randn(2, 1024, 384).cuda()
retrieved, mem_state = mem(seq)                        # (retrieved, NeuralMemState)

model = MemoryAsContextTransformer(
    num_tokens=256, dim=384, depth=8, segment_len=32,
    num_persist_mem_tokens=4, num_longterm_mem_tokens=4,
    neural_memory_layers=(2, 4, 6), neural_memory_segment_len=4, neural_memory_batch_size=128,
    sliding_window_attn=True, use_flex_attn=False,     # flex attention is CUDA-first
    neural_memory_model=MemoryMLP(dim=64, depth=2),
    neural_memory_kwargs=dict(dim_head=64, heads=1, qk_rmsnorm=True, momentum=True, momentum_order=1,
                              default_step_transform_max_lr=1e-1, per_parameter_lr_modulation=True,
                              spectral_norm_surprises=True),
).cuda()
loss = model(torch.randint(0, 256, (1, 513)).cuda(), return_loss=True); loss.backward()
```

**`train_mac.py` defaults:** enwik8 with the first 90M bytes as the training split, SEQ_LEN 512, batch 4 × grad-accum 4, lr 2e-4 with `AdoptAtan2`, grad clip 0.5, dim 384, depth 8, window 32.\[12\]

**Known issues:**
- **#18:** `torch.func.{grad, vjp, ...} don't yet support saved tensor hooks` under activation checkpointing.\[35\]
- **#19:** fails inside JIT-scripted modules.\[36\]
- **#64 (Sep 2026):** per-head memory parameters alias a single tensor, so optimizer steps and `load_state_dict` fail for `heads > 1`.\[31\]
- **#51:** no pretrained weights.\[31\]

**Rule:** pin the version, use `heads=1`, and verify the state-passing kwarg in `neural_memory.py` yourself. I could not confirm it from source.

## 6.2 Other implementations

- **Aedelon/titans-pytorch-mlx:** PyTorch and MLX versions of MAC/MAG/MAL, with FineWeb-Edu streaming pretraining.\[6\] `NeuralLongTermMemory(config)` returns `(output, state)`, and `state=` can be passed back in.\[6\]
- **danielquintas8/atlas-torch:** an ATLAS implementation.\[37\]
- **javilima01/Titans:** a Titans adapter on a **frozen Qwen**. It keeps memory parameters, fast weights and optimizer state in fp32 with a bf16 backbone, and uses per-user state files "bound to the exact adapter weights". Training data is synthetic plus BABILong/QASPER, with a text-retrieval control.\[38\] **This is the closest template for Part 10.**
- **ddidacus/llama-titans:** `TitanModel.from_pretrained("meta-llama/Llama-3.2-1B", gated=True, segment_size=4096)`.\[39\]
- **kolejnyy/titans-lmm:** a readable proof-of-concept MAC layer.\[7\]
- **TPTT** (arXiv 2506.17671): "Transforming Pretrained Transformers into Titans". It uses MaG plus LiZA linearized attention on Llama, OpenELM, Qwen, Gemma and Mistral.\[40\]
- **langchain-nmret:** a LangChain retriever that combines titans-pytorch memory ("abstract guidance vectors") with a vector store and an LLM compressor.\[41\]
- **Official JAX:** none. The paper says the model was "implemented in Pytorch and JAX", but that code was never released.\[4\]

## 6.3 flash-linear-attention (fla-org)

Triton kernels and layers, "verified on NVIDIA, AMD, and Intel hardware". Covered models include DeltaNet, **Gated DeltaNet** (with a FlashQLA backend), RWKV7, Mamba2/3, KDA, MesaNet, Comba, Log-Linear Attention and GDN-2, plus the `flame` trainer.\[28\] **The model table has no Titans or ATLAS.**\[28\] On gfx1151, a community guide reports that the PyPI wheel crashes because `num_warps > 4` trips Triton LinearLayoutConversions assertions.\[42\] Patch the kernels to use ≤ 4 warps.

## 6.4 Checkpoints

There are **no official weights** (issue #51).\[31\] The only ones available are toys, such as `thebnbrkr/titans-marv-enwik8` (0.37M params, val loss 1.863 nats), whose memory "accumulates writes rather than forgetting them".\[43\] You will need to train your own.

---

# PART 7 — From-Scratch Reference Implementation (PyTorch)

Design:
- Heads are folded into the batch dimension.
- Gradients are taken at the weights from the start of each chunk (Eq. 16).
- Momentum and decay use an exact sequential loop over the tokens in a chunk. Replace it with a scan later.
- Chunk c reads the weights produced after chunk c−1.
- The outer loop backpropagates through the inner updates.

## 7.1 `titan/memory.py`

```python
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call, grad, vmap

class MemoryMLP(nn.Module):
    """Deep memory M(x) = x + W2·silu(W1·x); L_M = 2."""
    def __init__(self, dim, expansion=2):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(dim, dim * expansion) / math.sqrt(dim))
        self.w2 = nn.Parameter(torch.zeros(dim * expansion, dim))   # zero-init: starts as identity
    def forward(self, x):
        return x + F.silu(x @ self.w1) @ self.w2

def _assoc_loss(params, model, k, v):
    pred = functional_call(model, params, (k.unsqueeze(0),)).squeeze(0)
    return (pred - v).pow(2).sum()                                  # ||M(k)-v||^2 (Eq. 12)

class NeuralMemory(nn.Module):
    def __init__(self, dim, heads=1, dim_head=None, chunk_size=16, max_lr=0.1, expansion=2, conv_kernel=4):
        super().__init__()
        self.h, self.dh, self.chunk, self.max_lr = heads, dim_head or dim // heads, chunk_size, max_lr
        inner = self.h * self.dh
        self.to_qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.conv = nn.Conv1d(3 * inner, 3 * inner, conv_kernel, groups=3 * inner, padding=conv_kernel - 1)
        self.to_gates = nn.Linear(dim, 3 * heads)                   # theta, eta, alpha
        self.mem = MemoryMLP(self.dh, expansion)                    # meta-learned M_0
        self.out, self.out_gate, self.norm = nn.Linear(inner, dim, bias=False), nn.Linear(dim, inner), nn.RMSNorm(inner)
        g = grad(_assoc_loss)
        self._per_tok_grad = vmap(vmap(g, in_dims=(None, None, 0, 0)), in_dims=(0, None, 0, 0))

    def init_state(self, B, device):
        W = {n: p.unsqueeze(0).expand(B * self.h, *p.shape).clone() for n, p in self.mem.named_parameters()}
        return {"W": W, "S": {n: torch.zeros_like(w) for n, w in W.items()}}

    @staticmethod
    def detach_state(state):
        return {k: {n: t.detach() for n, t in d.items()} for k, d in state.items()}

    def _retrieve(self, W, q):
        return vmap(lambda p, x: functional_call(self.mem, p, (x,)))(W, q)

    def forward(self, x, state=None, update=True):
        B, T, _ = x.shape
        qkv = self.conv(self.to_qkv(x).transpose(1, 2))[..., :T].transpose(1, 2)   # causal depthwise conv
        q, k, v = F.silu(qkv).chunk(3, dim=-1)
        split = lambda t: t.view(B, T, self.h, self.dh).transpose(1, 2).reshape(B * self.h, T, self.dh)
        q, k, v = map(split, (q, k, v))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        g = torch.sigmoid(self.to_gates(x).float()).view(B, T, 3, self.h).permute(2, 0, 3, 1).reshape(3, B * self.h, T)
        theta, eta, alpha = g[0] * self.max_lr, g[1], g[2]
        state = state or self.init_state(B, x.device)
        W, S, outs = state["W"], state["S"], []
        for s in range(0, T, self.chunk):
            e = min(s + self.chunk, T)
            outs.append(self._retrieve(W, q[:, s:e].float()))       # read with M_{c-1}
            if not update:
                continue
            G = self._per_tok_grad(W, self.mem, k[:, s:e].float(), v[:, s:e].float())
            for t in range(e - s):                                  # Eqs. 13-14
                th, et, al = theta[:, s + t], eta[:, s + t], alpha[:, s + t]
                for n in W:
                    shp = (-1,) + (1,) * (W[n].dim() - 1)
                    S[n] = et.view(shp) * S[n] - th.view(shp) * G[n][:, t]
                    W[n] = (1 - al.view(shp)) * W[n] + S[n]
        y = torch.cat(outs, 1).to(x.dtype).view(B, self.h, T, self.dh).transpose(1, 2).reshape(B, T, -1)
        y = self.norm(y) * torch.sigmoid(self.out_gate(x))
        return self.out(y), {"W": W, "S": S}
```

Swaps:
- **Omega (ATLAS):** sum the loss over the last c keys and values.
- **Huber (Yaad):** replace `.pow(2)` with `F.huber_loss(..., reduction='sum')`.

## 7.2 `titan/attention.py`

```python
import torch, torch.nn as nn, torch.nn.functional as F

def sliding_window_mask(T, window, n_prefix, device):
    i = torch.arange(T, device=device)[:, None]; j = torch.arange(T, device=device)[None, :]
    return (j <= i) & (((i - j) < window) | (j < n_prefix))       # causal SWA + persistent prefix

class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__(); self.h = heads
        self.qkv, self.o = nn.Linear(dim, 3 * dim, bias=False), nn.Linear(dim, dim, bias=False)
    def forward(self, x, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)   # AOTriton path on ROCm
        return self.o(y.transpose(1, 2).reshape(B, T, D))

class MLP(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__(); self.a, self.b = nn.Linear(dim, 2 * mult * dim, bias=False), nn.Linear(mult * dim, dim, bias=False)
    def forward(self, x):
        u, g = self.a(x).chunk(2, -1); return self.b(u * F.silu(g))
```

Add RoPE for real runs.

## 7.3 `titan/model.py`: MAG and MAC

```python
import torch, torch.nn as nn
from .memory import NeuralMemory
from .attention import Attention, MLP, sliding_window_mask

class MAGBlock(nn.Module):
    def __init__(self, dim, heads, window, n_persist, chunk=16):
        super().__init__()
        self.window, self.np = window, n_persist
        self.n1, self.n3, self.gy, self.gm = (nn.RMSNorm(dim) for _ in range(4))
        self.attn, self.mem, self.mlp = Attention(dim, heads), NeuralMemory(dim, 1, chunk_size=chunk), MLP(dim)
    def forward(self, x, state):
        h = self.n1(x)
        y = self.attn(h, sliding_window_mask(h.size(1), self.window, self.np, h.device))
        m, state = self.mem(h, state)
        x = x + self.gy(y) * torch.sigmoid(self.gm(m))                # o = y ⊗ M(x~) (Eq. 28)
        return x + self.mlp(self.n3(x)), state

class TitanMAG(nn.Module):
    def __init__(self, vocab, dim=512, depth=8, heads=8, window=256, n_persist=16, chunk=16):
        super().__init__()
        self.np, self.emb = n_persist, nn.Embedding(vocab, dim)
        self.persist = nn.Parameter(torch.randn(n_persist, dim) * 0.02)
        self.blocks = nn.ModuleList(MAGBlock(dim, heads, window, n_persist, chunk) for _ in range(depth))
        self.norm, self.head = nn.RMSNorm(dim), nn.Linear(dim, vocab, bias=False)
    def forward(self, ids, states=None):
        x = self.emb(ids); x = torch.cat([self.persist.expand(x.size(0), -1, -1), x], 1)
        states, new = states or [None] * len(self.blocks), []
        for blk, st in zip(self.blocks, states):
            x, st = blk(x, st); new.append(st)
        return self.head(self.norm(x[:, self.np:])), new

class TitanMAC(nn.Module):
    """Per segment: [P | h_t | S_t] -> causal attn -> write y_t -> o_t = y_t ⊗ M*_t(y_t)."""
    def __init__(self, vocab, dim=512, depth=8, heads=8, seg=128, n_persist=16, chunk=16):
        super().__init__()
        self.seg, self.np, self.emb = seg, n_persist, nn.Embedding(vocab, dim)
        self.persist = nn.Parameter(torch.randn(n_persist, dim) * 0.02)
        self.attn = nn.ModuleList(Attention(dim, heads) for _ in range(depth))
        self.mlp = nn.ModuleList(MLP(dim) for _ in range(depth))
        self.mem = nn.ModuleList(NeuralMemory(dim, 1, chunk_size=chunk) for _ in range(depth))
        self.n1 = nn.ModuleList(nn.RMSNorm(dim) for _ in range(depth))
        self.n2 = nn.ModuleList(nn.RMSNorm(dim) for _ in range(depth))
        self.norm, self.head = nn.RMSNorm(dim), nn.Linear(dim, vocab, bias=False)
    def forward(self, ids, states=None):
        B, T = ids.shape; states = states or [None] * len(self.attn); outs = []
        for s in range(0, T, self.seg):
            x = self.emb(ids[:, s:s + self.seg]); L = x.size(1)
            for i in range(len(self.attn)):
                h = self.n1[i](x)
                hist, _ = self.mem[i](h, states[i], update=False)     # h_t = M*_{t-1}(q_t) (Eq. 21)
                z = torch.cat([self.persist.expand(B, -1, -1), hist, h], 1)
                mask = torch.ones(z.size(1), z.size(1), dtype=torch.bool, device=z.device).tril()
                mask[:, :self.np] = True
                y = self.attn[i](z, mask)[:, self.np + L:]           # Eq. 23
                _, states[i] = self.mem[i](y, states[i], update=True) # Eq. 24
                r, _ = self.mem[i](y, states[i], update=False)
                x = x + y * torch.sigmoid(r)                          # Eq. 25
                x = x + self.mlp[i](self.n2[i](x))
            outs.append(x)
        return self.head(self.norm(torch.cat(outs, 1))), states
```

Simplifications:
- `hist` has one token per segment token. You can pool it down to N_l tokens, as lucidrains does with `num_longterm_mem_tokens`.
- Causality holds because every `hist` token comes from M_{t−1}.

## 7.4 `titan/state_io.py`: persistent cross-session memory

```python
import hashlib, json, os, torch
from safetensors.torch import save_file, load_file
from safetensors import safe_open

def ckpt_hash(model):
    h = hashlib.sha256()
    for n, p in sorted(model.state_dict().items()):
        h.update(n.encode()); h.update(p.detach().float().cpu().numpy().tobytes()[:4096])
    return h.hexdigest()[:16]

def save_states(path, states, model, meta):
    flat = {f"{li}.{kd}.{n}": t.detach().float().contiguous().cpu()
            for li, st in enumerate(states) for kd in ("W", "S") for n, t in st[kd].items()}
    save_file(flat, path + ".tmp", metadata={"meta": json.dumps({**meta, "ckpt": ckpt_hash(model)})})
    os.replace(path + ".tmp", path)                                   # atomic write

def load_states(path, model, device):
    with safe_open(path, "pt") as f:
        meta = json.loads(f.metadata()["meta"])
    if meta["ckpt"] != ckpt_hash(model):
        raise RuntimeError("memory state bound to a different checkpoint")
    flat = load_file(path, device=str(device))
    states = [{"W": {}, "S": {}} for _ in range(1 + max(int(k.split(".")[0]) for k in flat))]
    for k, t in flat.items():
        li, kd, n = k.split(".", 2); states[int(li)][kd][n] = t
    return states, meta
```

Keep one file per user or agent, with rolling snapshots. The snapshots double as a Memory-Caching archive (§5.5).

## 7.5 `scripts/train.py` (bf16, clipping, truncated BPTT over memory)

```python
import os; os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")   # before torch import
import math, json, time, torch, numpy as np
from titan.model import TitanMAG
from titan.memory import NeuralMemory

cfg = dict(vocab=256, dim=384, depth=6, heads=6, window=128, n_persist=8, chunk=16,
           seq=512, bptt=4, batch=8, lr=3e-4, wd=0.1, steps=20000, clip=1.0)
data = np.memmap("data/enwik8.bin", dtype=np.uint8, mode="r"); train = data[:90_000_000]
def batch():
    L = cfg["seq"] * cfg["bptt"] + 1; ix = np.random.randint(0, len(train) - L, cfg["batch"])
    return torch.from_numpy(np.stack([train[i:i + L] for i in ix]).astype(np.int64)).cuda()

model = TitanMAG(cfg["vocab"], cfg["dim"], cfg["depth"], cfg["heads"], cfg["window"], cfg["n_persist"], cfg["chunk"]).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, cfg["steps"]) / cfg["steps"])))
os.makedirs("runs/mag", exist_ok=True); log = open("runs/mag/metrics.jsonl", "a")
for step in range(cfg["steps"]):
    x, states, tot = batch(), None, 0.0
    for sg in range(cfg["bptt"]):                                     # truncated BPTT across segments
        s = sg * cfg["seq"]; inp, tgt = x[:, s:s + cfg["seq"]], x[:, s + 1:s + cfg["seq"] + 1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, states = model(inp, states)
        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, cfg["vocab"]), tgt.reshape(-1))
        (loss / cfg["bptt"]).backward()
        states = [NeuralMemory.detach_state(st) for st in states]    # cut graph, keep contents
        tot += loss.item() / cfg["bptt"]
    if not math.isfinite(tot): raise SystemExit(f"non-finite loss at {step}: check bf16 bugs / max_lr")
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
    opt.step(); opt.zero_grad(set_to_none=True); sched.step()
    if step % 100 == 0: log.write(json.dumps({"step": step, "bpc": tot / math.log(2), "t": time.time()}) + "\n"); log.flush()
    if step % 2000 == 0: torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, "runs/mag/ckpt.pt")
```

## 7.6 `tests/test_memory.py` (gate P1)

```python
import torch
from titan.memory import NeuralMemory
def test_causality():
    torch.manual_seed(0); m = NeuralMemory(64, heads=2, chunk_size=8); x = torch.randn(2, 32, 64)
    y, _ = m(x); x2 = x.clone(); x2[:, 24:] += 5.0; y2, _ = m(x2)
    assert y.shape == x.shape and torch.allclose(y[:, :24], y2[:, :24], atol=1e-5)
def test_state_carries_info():
    torch.manual_seed(0); m = NeuralMemory(32, chunk_size=4, max_lr=0.5); x = torch.randn(1, 64, 32)
    _, st = m(x); a, _ = m(x, st, update=False); b, _ = m(x, None, update=False)
    assert not torch.allclose(a, b)
def test_outer_grads():
    m = NeuralMemory(32, chunk_size=4); y, _ = m(torch.randn(1, 16, 32)); y.sum().backward()
    assert m.to_gates.weight.grad is not None and m.mem.w1.grad is not None
```

## 7.7 `scripts/eval_niah.py`: passkey eval

```python
import torch, random
def prompt(n, key):
    f = "The grass is green. The sky is blue. The sun is yellow. " * n; p = random.randint(0, len(f))
    return f[:p] + f" The pass key is {key}. Remember it. " + f[p:] + " The pass key is"
@torch.no_grad()
def eval_passkey(model, lengths=(1_000, 4_000, 16_000, 64_000), trials=20, seg=512):
    res = {}
    for L in lengths:
        ok = 0
        for _ in range(trials):
            key = str(random.randint(10000, 99999)); ids = torch.tensor([list(prompt(L // 56, key).encode())]).cuda()
            st = None
            for s in range(0, ids.size(1) - 1, seg): _, st = model(ids[:, s:min(s + seg, ids.size(1) - 1)], st)
            out, cur = "", ids[:, -1:]
            for _ in range(6):
                lg, st = model(cur, st); cur = lg[:, -1].argmax(-1, keepdim=True); out += chr(cur.item())
            ok += key in out
        res[L] = ok / trials
    return res
```

Under `no_grad`, `torch.func.grad` still computes its own gradients, but verify this on your build. If it fails, wrap `NeuralMemory.forward` in `torch.enable_grad()`. Also run RULER S-NIAH and BABILong qa1–qa5 so your numbers compare with the paper.

---

# PART 8 — Optimizations

1. **Cost.** Each memory layer needs roughly 3× the memory-MLP FLOPs per token (forward plus per-token backward), in both training and inference. Put memory in only some layers; lucidrains uses layers 2, 4 and 6.\[12\]
2. **Remove the Python loop.** Compute the momentum and decay with a log-space cumulative-product scan or `assoc-scan`, or use per-chunk gates (§3.2) so each chunk is one matmul-sum.
3. **Chunk sweep.** For b ∈ {8, 16, 32, 64}, log tokens/s, bpc and the passkey score, and pick the knee. If b ≥ 64 is needed for speed, move to TNT-style global and local memory.
4. **torch.compile.** Compile the attention and MLP blocks. The functorch transforms plus dynamic chunk counts cause graph breaks, so pad T to a multiple of b. MLX users report that "mx.compile cannot compile full Titans models".\[6\]
5. **Kernels.** Attention through SDPA with AOTriton. fla needs `num_warps ≤ 4` on gfx1151. The first fused kernel worth writing is the chunk update: grad, then scan, then the decay-weighted sum.
6. **Precision.** bf16 backbone, **fp32** for W, S, gates and the loss. Cap θ with `max_lr` (0.1), and optionally clip surprise by spectral norm (lucidrains' `spectral_norm_surprises`).\[12\]
7. **Quantization.** Quantize only the frozen backbone (GGUF, AWQ or bnb at 4–8 bit). Never quantize W or S.
8. **KV cache vs memory state.** MAG only needs a window-sized KV cache plus a fixed state of layers × params × 2 (W and S) × 4 bytes. For example, 8 layers with d_h = 512 and e = 2 is about 1M parameters per layer, or roughly **64 MB** in total. A full KV cache for 1M tokens is many GB.\[44\]
9. **Throughput.** No gfx1151 Titans benchmarks exist. As an *illustrative estimate, not a measurement*: 170M parameters × 3.4B tokens is about 3.5e18 FLOPs, which is roughly 4 days at an assumed 10 TFLOPS. Add 1.5–3× for the memory layers. Measure MFU in P3 and re-plan.

---

# PART 9 — Local Hardware

## 9.1 Strix Halo gfx1151 status (Sept 2026)

**Support.** As of February 2026, gfx1151 was not on AMD's official ROCm support matrix. It still works, helped by ROCm 7.2's gfx11-generic target.\[45\] One user reports that ROCm 7.2.4 ships native hipBLASLt gfx1151 kernels.\[46\]

**Wheels.** Install with `pip install --pre torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/`. \[47\] ROCm issue #6034 notes that "Stable ROCm does not ship gfx1151 kernels".\[48\] Builds seen in the wild:
- January 2026: `2.11.0a0+rocm7.11.0a20260106`\[48\]
- September 2026: TheRock `torch==2.12.0+rocm10.2.0a20260921` (seen on a different gfx target)\[49\]
- A fine-tuning guide pairs a ROCm 7.13 nightly with PyTorch 2.11 and fla.\[42\]

Container options are `kyuz0/amd-strix-halo-pytorch-gfx1151-aotriton` and `rocm/pytorch-nightly`.\[50\]\[51\]

**HSA_OVERRIDE_GFX_VERSION: the reports conflict.** Some 2026 guides still recommend 11.0.0 or 11.5.1.\[52\] #6034 and the fine-tuning guide say it "actively breaks native gfx1151 kernels".\[42\]\[48\] **Decision:** don't set it with native wheels.

**Attention.**
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` must be set before `import torch`. One report measured 4096-token attention at 135 ms without it and 12 ms with it; #6034 describes a 19× speedup.\[48\]\[53\]
- The ROCm flash-attention fork works through Triton with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`.\[54\]
- Early TheRock builds lacked flash SDPA on gfx1151 (TheRock #1364).\[55\]\[56\]

**Bugs.**
- #6034 reports five critical bf16 bugs and NaNs at some batch sizes, and advises unsetting `PYTORCH_HIP_ALLOC_CONF`.\[48\]
- Silent hangs and PERMISSION_FAULTs are reported in #6530, #6186 and #6165.\[57\]
- Mitigations: checkpoint at least every 30 minutes, run a step-time watchdog, keep memory in fp32, and pin a nightly you have validated.

**Big-model inference.** llama.cpp with Vulkan (RADV) or ROCm/HIP. Vulkan is easiest; ROCm with flash attention holds up better at long context.\[58\] On Windows, AMD recommends a 96 GB VGM setting for best performance, although AMD's VGM FAQ says the iGPU "can technically access a total graphics memory size of 112GB" on 128 GB systems. AMD's blog (Adrenalin 25.8.1 WHQL) says the 128 GB Ryzen AI Max+ 395 "can not only run Meta's Llama 4 Scout but do it at a context length of 256,000 (Flash Attention ON, KV Cache Q8)." Theoretical memory bandwidth is 256 GB/s (256-bit LPDDR5x-8000); the hogeheer499 strix-halo-guide measured "~215 GB/s measured, 256 GB/s theoretical". As a result, MoE models run much faster than dense 70B models.

## 9.2 Realistic sizes on 128 GB

| Workload | Feasible? | Notes |
|---|---|---|
| From-scratch MAG/MAC, 30–170M | Yes | Main target; days to weeks |
| From-scratch 340–400M | Marginal | Weeks |
| 760M/30B reproduction | No | Compute-bound |
| Frozen 4–8B bf16 + adapter | Yes | Backbone ~8–16 GB |
| Frozen 14–32B 4–8-bit + adapter | Yes, slower | bnb/ROCm path |
| 70–120B MoE GGUF inference | Yes | llama.cpp; Titans adapter needs a PyTorch backbone |

## 9.3 Threadripper 2970WX

24 Zen+ cores, no AVX-512. Use it for downloading and tokenizing data, dedup, building RULER/BABILong sets, CPU tests and CI, and llama.cpp CPU fallback. Not for training.

---

# PART 10 — Strategy

## 10.1 From scratch (validation track)

Train a 30–170M byte-level model on enwik8 or TinyStories. It is cheap, it exercises every gate, and it proves your causal and persistence code works.

## 10.2 Retrofit a frozen LLM (product track)

The recipe follows javilima01/Titans, TPTT and Prometheus Mind:
- Take a frozen Qwen/Llama at 1–8B in bf16.
- Add residual `NeuralMemory` branches at 1–3 mid-depth layers, with the output gate initialized near 0.\[38\]
- Train only the adapter, on synthetic "fact early, question later" episodes plus BABILong.\[38\]
- Compare against a text-retrieval control.\[38\] **If the adapter doesn't beat RAG on your facts, ship RAG.**

Risks:
- **Hidden-state collapse.** Prometheus Mind found representations of "wife" and "brother" at 0.98+ similarity.\[59\]
- **Training collapse.** Prometheus Mind needed stage-wise training.\[59\]
- **Storage without recall** (Knowledge Objects).\[14\]
- **MMLU drops** ("Language Model Memory" paper).\[60\]
- **Checkpointing conflicts** (#18).\[35\]

```python
# titan/adapter.py (hook-based; no backbone edits)
import torch, torch.nn as nn
from .memory import NeuralMemory
class MemoryAdapter(nn.Module):
    def __init__(self, hidden, mem_dim=512, chunk=32):
        super().__init__()
        self.down, self.up = nn.Linear(hidden, mem_dim), nn.Linear(mem_dim, hidden)
        self.mem, self.state = NeuralMemory(mem_dim, chunk_size=chunk), None
        self.gate = nn.Parameter(torch.full((hidden,), -4.0))       # sigmoid(-4)≈0.018 at init
    def forward(self, h):
        y, self.state = self.mem(self.down(h.float()), self.state)
        return h + torch.sigmoid(self.gate) * self.up(y).to(h.dtype)
def attach(model, layers=(12,), **kw):
    for p in model.parameters(): p.requires_grad_(False)
    ads = nn.ModuleDict()
    for i in layers:
        ad = ads[str(i)] = MemoryAdapter(model.config.hidden_size, **kw).to(model.device)
        model.model.layers[i].register_forward_hook(
            lambda m, inp, out, ad=ad: (ad(out[0]),) + tuple(out[1:]) if isinstance(out, tuple) else ad(out))
    return ads
```

## 10.3 Titans-inspired agent memory (zero cloud)

The agent combines five pieces:
- **(a)** a frozen local LLM;
- **(b)** the neural adapter, with a state file per user;
- **(c)** a Memory-Caching snapshot archive, per session or topic;
- **(d)** a local vector store as ground truth (the langchain-nmret pattern);\[41\]
- **(e)** Nested-Learning update frequencies: fast memory per chunk, session memory per turn, and nightly consolidation by replaying the day's transcripts.

Use the surprise norm ‖∇ℓ‖ as an importance score, and write high-surprise turns to the vector store.

---

# PART 11 — Honest Limitations

1. **No official code or weights.** Only third-party implementations exist (see issue #51).\[31\]
2. **Reproducibility.** "Titans Revisited" (Di Nepi, Siciliano and Silvestri, Oct 2025) cites "the lack of publicly available code and ambiguities in the original description" and finds that chunking can erase the gains. Baselines in both Titans and ATLAS are imported rather than rerun,\[23\] and the MAL SIQA number looks anomalous.
3. **Speed.** The paper itself says the memory is "slightly slower than Mamba2 and Gated DeltaNet",\[4\] and test-time gradients cost real compute.
4. **Instability.** Inner learning rates, bf16 and the gfx1151 bugs combine badly.
5. **Storage vs recall.** Parametric memory can store a fact and still fail to recall it.\[14\]
6. **Scale.** Nothing has been reported above 1.3B.
7. **Hardware.** gfx1151 is not on AMD's official support matrix,\[45\] and hangs are reported.\[57\]

---

# PART 12 — Roadmap for Claude Code

| Phase | Task | Gate |
|---|---|---|
| P0 | TheRock torch, env vars, requirements | `torch.cuda.is_available()` and the device name shows Radeon 8060S; bf16 2048² matmul OK; SDPA at 4096 tokens < 50 ms |
| P1 | memory.py + tests | `pytest tests/test_memory.py` passes |
| P2 | Toy k→v recall (64 pairs) | Cosine > 0.9 at depth 2; beats linear memory |
| P3 | MAG on enwik8 (dim 384, depth 6) | val bpc < 1.6 by 20k steps; no NaNs; MFU logged |
| P4 | Passkey | ≥ 90% at 4K and ≥ 70% at 16K with window 128; memory-disabled ≤ 20% at 16K |
| P5 | MAC | bpc within 3% of MAG; better passkey at the longest length |
| P6 | Persistence | Fact from session 1 recalled in a new process; hash mismatch raises |
| P7 | Frozen-LLM adapter | Init: perplexity change < 1%; trained: beats the retrieval control on held-out episodes; MMLU-subset drop < 1 pt |
| P8 | Offline agent | Network disabled; ≥ 80% recall of 20 facts told across 5 sessions |
| P9 | Optimizations | ≥ 2× tokens/s vs P3 with ≤ 2% bpc regression |

## 12.1 `requirements.txt`

```text
# torch: pip install --pre torch==<validated-nightly> --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
numpy==2.1.3
einops==0.8.1
safetensors==0.5.3
transformers==4.57.1
accelerate==1.10.1
datasets==5.0.1
tokenizers==0.22.1
pytest==8.3.5
tqdm==4.67.1
titans-pytorch==0.5.5          # reference only; heads=1 until issue #64 is fixed
# optional: flash-linear-attention==<pin>  (patch num_warps<=4 on gfx1151)
```

Only `titans-pytorch==0.5.5` and `datasets==5.0.1` (the version javilima01 tested) come from sources.\[38\] The other pins are suggestions: validate them in P0, then run `pip freeze > requirements.lock`.

## 12.2 Repo layout

```
titan-local/
├── CLAUDE.md  ROADMAP.md  requirements.txt  requirements.lock
├── titan/ {memory.py, attention.py, model.py, state_io.py, adapter.py}
├── scripts/ {fetch_assets.py, train.py, eval_niah.py, eval_babilong.py, chunk_sweep.py, chat.py}
├── tests/ {test_memory.py, test_model.py, test_state_io.py, test_offline.py}
├── configs/ {mag_small.yaml, mac_small.yaml, adapter_qwen.yaml}
├── data/ runs/ memstate/   (gitignored)
└── env/strix_halo.sh   # exports TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1; unsets PYTORCH_HIP_ALLOC_CONF
```

---

## Caveats

- All benchmark numbers in Parts 2 and 4 are author-reported. The only independent checks are "Titans Revisited" and small probes.
- ROCm details change weekly, so pin whatever passes P0.
- I could not confirm the exact NeuralMemory state-passing kwarg in lucidrains' source. Read `neural_memory.py` at your pinned version.
- The throughput figures in Part 8 are illustrative estimates, not measurements.

## Sources

1. [Paper page - Titans: Learning to Memorize at Test Time](https://huggingface.co/papers/2501.00663)
2. [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
3. [Titans: Learning to Memorize at Test Time](https://arxiv.org/html/2501.00663v1)
4. <https://arxiv.org/pdf/2501.00663>
5. [Proteus: Incremental Memory Activation for Long-Context Sequence Modeling](https://arxiv.org/pdf/2608.16844)
6. [GitHub - Aedelon/titans-pytorch-mlx: PyTorch implementation of Google Titans: Learning to Memorize at Test Time (arXiv:2501.00663) · GitHub](https://github.com/Aedelon/titans-pytorch-mlx)
7. [GitHub - kolejnyy/titans-lmm: A proof-of-concept implementation of Titans: models mixing long-term, short-term and persistent memories · GitHub](https://github.com/kolejnyy/titans-lmm)
8. [Titans: Learning to Memorize at Test Time Ali Behrouz Google Research USA](https://openreview.net/pdf/8cd62cb08869de77128e63a849345dea7da48b60.pdf)
9. [TNT: Improving Chunkwise Training for Test-Time Memorization — Lacuna](https://lacuna.tiptreesystems.com/work/tnt-improving-chunkwise-training-for-test-time-memorization/wrk_755f6bbd704e33eeb7f15a43b36ddada)
10. [TNT: Improving Chunkwise Training for Test-Time Memorization | Cool Papers - Immersive Paper Discovery](https://papers.cool/arxiv/2511.07343)
11. [It’s All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization](https://arxiv.org/html/2504.13173)
12. [titans\_NPC / train\_mac.py](https://huggingface.co/datasets/ChipYTY/titans_NPC/blob/main/train_mac.py)
13. [TNT: Improving Chunkwise Training for Test-Time Memorization | ML Anthology](https://mlanthology.org/iclr/2026/li2026iclr-tnt/)
14. [Facts as First Class Objects: Knowledge Objects for Persistent LLM Memory](https://arxiv.org/pdf/2603.17781)
15. [(PDF) It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization](https://www.researchgate.net/publication/390893044_It's_All_Connected_A_Journey_Through_Test-Time_Memorization_Attentional_Bias_Retention_and_Online_Optimization)
16. [Google's Titans Long-Context AI Uses Surprise Memory System To Outperform GPT-4 On BABILong Tests](https://www.etavrian.com/news/google-titans-miras-long-context-memory)
17. [\[2504.13173\] It’s All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization](https://ar5iv.labs.arxiv.org/html/2504.13173)
18. [\[2504.13173\] It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization](https://arxiv.org/abs/2504.13173)
19. [Introducing DeepTransformers: ATLAS | by Sai Krishna Reddy Mudhiganti | Medium](https://medium.com/@saimudhiganti/introducing-deeptransformers-atlas-64d8b96d5e90)
20. [Rohan Paul on X: "Existing models struggle with long contexts due to quadratic cost or limited memory and online updates. Atlas improves long-context understanding and memory, outperforming Transformers and Recurrent Neural Networks on key benchmarks. It learns context memorization using a new https://t.co/3LJ7KbzwPT" / X](https://x.com/rohanpaul_ai/status/1929361576982700474)
21. [Atlas: Learning to Optimally Memorize the Context at Test Time](https://arxiv.org/html/2505.23735v1)
22. [FOD#103: The Paper You Missed, and maybe the one we’ll be quoting a year from now](https://www.turingpost.com/p/fod103)
23. [ATLAS: Learning to Optimally Memorize the Context at Test Time · Pith Review](https://pith.science/paper/2505.23735)
24. [Introducing Nested Learning: A new ML paradigm for continual learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)
25. [Memory Caching: RNNs with Growing Memory — Large Language Models](https://awesomepapers.io/llm-papers/papers/hf2602.24281)
26. [Memory Caching: RNNs with Growing Memory](https://arxiv.org/html/2602.24281v1)
27. [In-Place Test-Time Training](https://arxiv.org/html/2604.06169v1)
28. [GitHub - fla-org/flash-linear-attention: 🚀 Efficient implementations for emerging model architectures](https://github.com/fla-org/flash-linear-attention)
29. [titans-pytorch review: NeuralMemory and MemoryAsContextTransformer](https://hysenlabs.com/en/projects/lucidrains-titans-pytorch)
30. [titans-pytorch](https://pypi.org/project/titans-pytorch)
31. [Issues · lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch/issues)
32. [GitHub - lucidrains/titans-pytorch: Unofficial implementation of Titans, SOTA memory for transformers, in Pytorch](https://github.com/lucidrains/titans-pytorch)
33. [pyproject.toml · ChipYTY/titans\_NPC at main](https://huggingface.co/datasets/ChipYTY/titans_NPC/blob/main/pyproject.toml)
34. [titans-pytorch/titans\_pytorch/\_\_init\_\_.py at main · lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch/blob/main/titans_pytorch/__init__.py)
35. [Including NeuralMemory in Llama](https://github.com/lucidrains/titans-pytorch/issues/18)
36. [The stateless API can't be used with Jitted modules · Issue #19 · lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch/issues/19)
37. [GitHub - danielquintas8/atlas-torch: Atlas: Learning to Optimally Memorize at Test Time - PyTorch Implementation · GitHub](https://github.com/danielquintas8/atlas-torch)
38. [GitHub - javilima01/Titans · GitHub](https://github.com/javilima01/Titans)
39. [llama-titans/README.md at main · ddidacus/llama-titans](https://github.com/ddidacus/llama-titans/blob/main/README.md)
40. [TPTT: Transforming Pretrained Transformers into Titans](https://arxiv.org/html/2506.17671v2)
41. [langchain-nmret · PyPI](https://pypi.org/project/langchain-nmret/)
42. [GitHub - h34v3nzc0dex/strix-halo-llm-finetune-guide: Home-enthusiast's guide to fine-tuning 27B+ LLMs on AMD Strix Halo (gfx1151, Ryzen AI MAX+ 395) — the patches and tuning to make Linux mainline + ROCm 7.13 nightly + PyTorch 2.11 + bitsandbytes + flash-linear-attention all work together for multi-day LoRA training with out-of-process eval orchestration.](https://github.com/h34v3nzc0dex/strix-halo-llm-finetune-guide)
43. [thebnbrkr/titans-marv-enwik8 · Hugging Face](https://huggingface.co/thebnbrkr/titans-marv-enwik8)
44. [Hardware requirements and memory planning | Huggingface Transformers Advanced Course | The Neural Base](https://theneuralbase.com/huggingface-transformers/learn/advanced/hardware-requirements/)
45. [Upgrading ROCm 7.0 to 7.2 on AMD Strix Halo (gfx1151) | TinyComputers.io](https://tinycomputers.io/posts/upgrading-rocm-7.0-to-7.2-on-amd-strix-halo-gfx1151.html)
46. [I tuned llama.cpp on a Strix Halo mini-PC and it beats Ollama on the same weights — here’s the recipe (and why “Vulkan beats ROCm” is a myth) | by Bkpaine | Medium](https://medium.com/@bkpaine1/i-tuned-llama-cpp-33d68b1c851d)
47. [\[Issue\]: Strix Halo (gfx1151) gives segfault on any VRAM access with torch nightly package · Issue #5853 · ROCm/ROCm](https://github.com/ROCm/ROCm/issues/5853)
48. [Strix Halo gfx1151: 93 ML experiments, 5 critical bf16 bugs, AOTriton 19x speedup undocumented · Issue #6034 · ROCm/ROCm](https://github.com/ROCm/ROCm/issues/6034)
49. [ROCm: validate native gfx1151 serving on Strix Halo by dbourdea · Pull Request #260 · FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken/pull/260)
50. [GitHub - kyuz0/amd-strix-halo-pytorch-gfx1151-aotriton · GitHub](https://github.com/kyuz0/amd-strix-halo-pytorch-gfx1151-aotriton)
51. [hub.docker.com](https://hub.docker.com/r/rocm/pytorch-nightly/tags)
52. [AMD Ryzen AI Max+ 395 (Strix Halo) for Local LLMs in 2026: 128GB Unified Memory, 100 t/s on 30B Models, and Whether It Beats a Discrete GPU](https://runaihome.com/blog/ryzen-ai-max-395-strix-halo-local-llm-2026/)
53. [GitHub - kroqueta-s/trellis-strix-halo: TRELLIS runner for AMD Strix Halo (gfx1151) on Windows ROCm. Pure-torch replacements for spconv, flash-attn and nvdiffrast. · GitHub](https://github.com/kroqueta-s/trellis-strix-halo)
54. [Running vLLM on Strix Halo - epheo - personal how-to, technical notes and insights](https://blog.epheo.eu/notes/strix-halo/index.html)
55. [\[Issue\]: PyTorch Flash Attention with gfx1151 · Issue #1364 · ROCm/TheRock](https://github.com/ROCm/TheRock/issues/1364?timeline_page=1)
56. [PyTorch w/ Flash Attention + vLLM for Strix Halo - Framework Desktop - Framework Community](https://community.frame.work/t/pytorch-w-flash-attention-vllm-for-strix-halo/74736)
57. [\[Issue\]: gfx1151 (Strix Halo) — silent hangs in PyTorch diffusion workloads on kernel 6.14 and 7.0.0, plus \~300x mmap weight-loading regression on 7.0.0 · Issue #6530 · ROCm/legacy-rocm-build](https://github.com/ROCm/legacy-rocm-build/issues/6530)
58. [AMD Ryzen AI Max+ 395 For Local LLMs (2026): Is 128GB Unified Memory Worth It?](https://www.hashtechwave.com/amd-strix-halo-review/)
59. [Prometheus Mind: Retrofitting Memory to Frozen Language Models — Large Language Models](https://awesomepapers.io/llm-papers/papers/2601.15324)
60. [Language Model Memory and Memory Models for Language](https://arxiv.org/pdf/2602.13466)
