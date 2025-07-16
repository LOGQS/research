### High-Level Specification for an **Expressivity Benchmark**

---

#### 1 | Why the benchmark exists

Provide a **quick, architecture-agnostic litmus test** that answers two questions for *any* sequence encoder:

1. **How well does it keep essential information when squeezed through a tight bottleneck?**
2. **How easily can it form content-dependent interactions among tokens?**

The test must run entirely on synthetic data and finish in seconds to minutes, so researchers can iterate without large datasets or long training cycles.

---

#### 2 | Core goals

| Goal                              | Explanation                                                                                                |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **G1 — Data independence**        | Benchmark must rely on data that is generated on-the-fly, never downloaded or stored. (efficiency, generalization)                      |
| **G2 — Architectural neutrality** | It must treat the encoder as a black box; no assumptions about attention, convolutions, recurrence, etc. (support for most architectures, coded in a certain way)  |
| **G3 — Dual evaluation**          | Score both *raw compression fidelity* and *ability to create dynamic, content-based relationships*. (expressivity capability)        |
| **G4 — Minimal compute**          | Default run completes in seconds to minutes on a CPU; optional “longer” mode still minutes. (GPU still acceptable)                    |
| **G5 — Parameter awareness**      | Report model size so scores can be interpreted relative to capacity.                                       |
| **G6 — Reproducibility**          | All randomness seeded; same inputs produce same scores across machines.                                    |
| **G7 — Extensibility**            | Users can swap in alternative synthetic patterns or additional metrics without rewriting the core harness. |

---

#### 3 | Key constraints

1. **No external datasets** — everything must be self-contained.
2. **Tiny default models** — demo encoders use only a few thousand parameters. (for speed)
3. **Forward-pass first** — the primary compression probe cannot rely on training the encoder; it should reveal what information is already preserved. (as soon as you train a decoder, there are different factors that affect your result)
4. **Optional quick training** — if enabled, the encoder may train for at most a few hundred mini-batches, keeping runtime short.
5. **Single-file delivery** — the entire benchmark (config, data generator, metrics, CLI, demo models) lives in one script for copy-paste ease.
6. **Metric fairness** — equations must not privilege any particular mechanism (e.g., softmax attention) over another.
7. **Information-theoretic grounding** — scores are expressed as variance explained (R²) or channel-capacity proxies (e.g., Jacobian entropy) or as something else, just not task accuracy on arbitrary labels.

---

#### 4 | Conceptual structure of the test

1. **Synthetic sequence generator**

2. **Compression probe**

3. **Interaction probe**
   * **Metric:** how close the head gets to the theoretical optimum within that tiny budget; higher implies the encoder facilitates content-dependent reasoning.

4. **Capacity probe**
   * **Metric:** effective number of independent channels the encoder can modulate locally.

5. **Report**


---

#### 5 | Intended use cases

* Rapid architecture ablations before large-scale training.
* To gain surface understanding on why some architecture components perform better than others

---

#### 6 | Non-goals & limitations

* The benchmark is **not** meant to replace domain-specific evaluation (e.g., language modelling, image classification).
* To get useful feedback, you need to change the model size or the difficulty of the task manually (which is for flexibility and simplicity)

---