# Expressivity Benchmark 🧪

A tiny, **single-file** harness that stress-tests any sequence encoder (functionally) on two fronts:

1. **Compression Fidelity** – can the model squeeze a long sequence through a tight latent and still reconstruct the global context?
2. **Content-Dependent Interaction** – can it pick out the right token-to-token relationships on demand?

Everything is synthetic, deterministic, and CPU-friendly so you can answer those questions in **seconds**, not hours (at least to some degree).

---

## ✨ Key Properties

| Goal                         | What it means in practice                                                                          |
| ---------------------------- | -------------------------------------------------------------------------------------------------- |
| **Data-free** 📦             | All samples are generated on the fly—no datasets to download.                                      |
| **Architecture-agnostic** 🧩 | Treats the encoder as a black box; works for attention, convolutions, recurrence or anything else. |
| **Dual metrics** 🎯          | Reports both bottleneck R² scores and retrieval / mutual-information probes.                       |
| **Reproducible** 🔒          | Global seed, deterministic PyTorch ops, fixed random matrices.                                     |
| **Extensible** 🔧            | Register your own model with a one-line `@register("Name")` decorator.                             |

---

## 📦 Installation

```bash
# Python ≥3.8 with PyTorch ≥2.1 assumed
pip install torch  # plus CUDA if you have it
```

No other dependencies are required.

---

## 🚀 Quick-start

```bash
python measure.py
```

Example output:

```
Model         k-params      sec/run       R2            R2_0.75       R2_0.50       R2_0.25       NCE(bits)     RetrieveR2    entropy       sv_min        sv_median     Gini        
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
SelfAttn         x            a           b                c             d             e             f             g             h            i               j          k           
StaticMLP        y            l           m                n             o             p             q             r             s            t               u          v           
```

---

## ⚙️ CLI Flags

| Flag                                         | Purpose                                                             |
| -------------------------------------------- | ------------------------------------------------------------------- |
| `--train_steps N`                            | Add a lightweight decoder fine-tune probe for N steps.              |
| `--fast`                                     | Skip Jacobian metrics & shorten retrieval training (handy on CPUs). |
| `--cpu-only`                                 | Force CPU even if CUDA is visible.                                  |
| `--L`, `--ar_coeff`, `--noise`, `--retr_hid` | Control sequence length and generator hyper-params.                 |
| `--self-test`                                | Run a couple of asserts and exit; good for CI.                      |

---

## 📊 What the Metrics Mean

* **R2**, **R2\_0.75/0.50/0.25** – variance explained when you randomly drop 0 %, 25 %, 50 %, 75 % of latent dimensions.
  *High drop tolerance → strong compression.*
* **NCE(bits)** – mutual-information proxy between latent and global factors.
  *Higher = better.*
* **RetrieveR2** – after a tiny MLP head learns, how well can the encoder fetch a value given its key anywhere in the sequence?
  *Measures practical token-token interaction.*
* **Jacobian stats** (`entropy`, `sv_min`, `sv_median`, `gini`) – local channel-capacity estimates.
  *Flat spectra and high gini = richer representations.*

---

## 🔧 Adding Your Own Model

```python
from measure import register, Config, nn, F

@register("MyCoolEncoder")
class MyEnc(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = ...
    def forward(self, x):
        z = self.net(x)          # (B, L, D_z) or (B, D_z)
        return F.relu(z[:, -1])  # ExpressivityTester expects (B, D_z)
```

That’s it—re-run `python measure.py` and your model will appear in the table.

---

## 🛠  Extending the Benchmark

* **Synthetic patterns** – tweak `HybridGenerator` (global vs. local factors, noise, AR dynamics).
* **Extra probes** – add a method inside `ExpressivityTester` and pack its result into the dict.
* **Longer runs** – increase `batch_eval` / `batch_train` or remove `--fast`.

---

## ❗ Limitations

* Not a replacement for domain tasks (language modelling, vision, …).
* Scores are most informative when **comparing** alternatives of similar parameter budgets.
* You may need to dial sequence length or noise to match very large encoders.

---

## 📝 License

MIT

Contributions welcome.
