from __future__ import annotations
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import argparse, dataclasses, math, time, logging, sys, random
from typing import Dict, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class Config:
    L: int = 32
    K_keys: int = 8
    V_dim: int = 4
    F_feat: int = 12
    D_in: int = K_keys + V_dim + F_feat
    D_z: int = 32

    r_global: int = 4
    r_local: int = 8

    ar_coeff: float = 0.8
    noise: float = 0.03

    batch_eval: int = 2048
    batch_train: int = 256

    train_steps: int = 0
    lr: float = 3e-3
    fast: bool = False
    cpu_only: bool = False
    retr_hid: int = 64

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

cfg_default = Config()

def setup_reproducibility(cfg: Config):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def setup_matmul_precision():
    if torch.cuda.is_available():
        prec = os.getenv("MM_PRECISION", "high")
        torch.set_float32_matmul_precision(prec)

print(f"Using device: {cfg_default.device}")
print(f"Using seed: {cfg_default.seed}")

setup_matmul_precision()

class HybridGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        assert cfg.F_feat % 2 == 0, "F_feat must be even (global/local split)"
        assert cfg.L > 2*cfg.K_keys, "Increase L so each key fits once"
        
        g = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
        self.rng = g                     
        self.Wg = torch.randn(cfg.r_global,
                             cfg.F_feat // 2,
                             generator=g, device=cfg.device)
        self.Wl = torch.randn(cfg.r_local,
                             cfg.F_feat - cfg.F_feat // 2,
                             generator=g, device=cfg.device)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.rng, device=self.cfg.device)

    @torch.no_grad()
    def sample_features(self, batch: int, context_id: int):
        c = self.cfg
        B, L = batch, c.L

        g_lat = self._randn(B, c.r_global)
        l0    = self._randn(B, c.r_local)
        eps   = self._randn(B, L - 1, c.r_local)

        ls = [l0]
        α = c.ar_coeff
        for t in range(L - 1):
            ls.append(α * ls[-1] + (1 - α**2)**0.5 * eps[:, t])
        l_lat = torch.stack(ls, 1)

        feat = torch.zeros(B, L, c.F_feat, device=c.device)
        scale = 0.35                           
        feat[:, :, :c.F_feat//2]  = torch.tanh(scale * (g_lat @ self.Wg)).unsqueeze(1).expand_as(feat[:, :, :c.F_feat//2])
        feat[:, :, c.F_feat//2:]  = torch.tanh(scale * torch.einsum('blr,rf->blf', l_lat, self.Wl))

        x = torch.zeros(B, L, c.D_in, device=c.device)

        x[:, 0, context_id % c.D_in] = 1.0

        if context_id == 0:
            x[:, :, c.K_keys+c.V_dim:] = feat
        elif context_id == 1:
            x[:, 1::2, c.K_keys+c.V_dim:] = feat[:, 1::2]
        else:
            half = L // 2
            x[:, 1:half, c.K_keys+c.V_dim:] = feat[:, 1:half]

        x += c.noise * self._randn(*x.shape)
        return x, g_lat

    @torch.no_grad()
    def sample_retrieval(self, batch: int):
        c = self.cfg
        B = batch
        keys   = torch.randint(0, c.K_keys, (B, c.L-1), generator=self.rng,
                               device=c.device)
        for b in range(B):
            keys[b, :c.K_keys] = torch.randperm(c.K_keys, generator=self.rng,
                                                device=c.device)
        values = self._randn(B, c.L-1, c.V_dim)
        q_idx  = torch.randint(0, c.L-1, (B,), generator=self.rng,
                               device=c.device)
        q_keys = keys[torch.arange(B), q_idx]
        target = values[torch.arange(B), q_idx]

        x = torch.zeros(B, c.L, c.D_in, device=c.device)
        x[:, :c.L-1, :c.K_keys]       = F.one_hot(keys, c.K_keys).float()
        x[:, :c.L-1, c.K_keys:c.K_keys+c.V_dim] = values

        query_left = torch.rand(B, generator=self.rng,
                                device=c.device) < 0.5
        for b in range(B):
            pos = 0 if query_left[b] else c.L-1
            x[b, pos, :c.K_keys] = F.one_hot(q_keys[b], c.K_keys).float()

        return x, target

class ExpressivityTester:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data = HybridGenerator(cfg)

    @staticmethod
    def _safe_r2(pred, target):
        var = target.var()
        if var < 1e-8: return 0.0
        return float(1 - F.mse_loss(pred, target) / var)

    @torch.no_grad()
    def _compression(self, enc: nn.Module) -> Dict[str, float]:
        results = {}
        for ctx in (0, 1, 2):
            X, g = self.data.sample_features(self.cfg.batch_eval, ctx)
            Z = enc(X).view(X.size(0), -1)

            for ratio in (1.0, 0.75, 0.5, 0.25):
                k  = int(Z.size(1) * ratio)
                idx = torch.randperm(Z.size(1), generator=self.data.rng,
                                   device=Z.device)[:k]
                Zc = Z[:, idx]
                Zc = (Zc - Zc.mean(0, keepdim=True)) / (Zc.std(0, unbiased=False, keepdim=True) + 1e-8)

                try:
                    W = torch.linalg.lstsq(Zc, g).solution
                except RuntimeError:
                    W = torch.linalg.pinv(Zc) @ g

                r2 = self._safe_r2(Zc @ W, g)
                results[f"R2_{ratio:.2f}_ctx{ctx}"] = r2

        for ratio in ("1.00", "0.75", "0.50", "0.25"):
            results[f"R2_{ratio}"] = sum(results[f"R2_{ratio}_ctx{i}"]
                                         for i in (0,1,2)) / 3
        results["R2"] = results["R2_1.00"]
        return results

    @torch.no_grad()
    def _info_nce(self, enc: nn.Module) -> float:
        X, g = self.data.sample_features(self.cfg.batch_eval, 0)
        Z = enc(X).view(X.size(0), -1)
        
        mid = Z.size(0) // 2
        try:
            W = torch.linalg.lstsq(Z[:mid], g[:mid]).solution
        except RuntimeError:
            W = torch.linalg.pinv(Z[:mid]) @ g[:mid]
        z_proj = Z[mid:] @ W
        logits = z_proj @ g[mid:].T
        ce  = F.cross_entropy(logits, torch.arange(mid, device=Z.device))
        mi_bits = (math.log(mid) - ce) / math.log(2)
        return float(mi_bits)

    def _retrieval_r2(self, enc: nn.Module) -> float:
        head = nn.Sequential(
            nn.Linear(self.cfg.D_z, self.cfg.retr_hid * 2, device=self.cfg.device),
            nn.ReLU(),
            nn.Linear(self.cfg.retr_hid * 2, self.cfg.V_dim, device=self.cfg.device)
        )
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        opt  = torch.optim.Adam(head.parameters(), lr=1e-3)
        steps = 200 if self.cfg.fast else 1200
        for _ in range(steps):
            X, v = self.data.sample_retrieval(self.cfg.batch_train)
            z = enc(X).view(X.size(0), -1)
            loss = F.mse_loss(head(z), v)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            X, v = self.data.sample_retrieval(self.cfg.batch_eval)
            z = enc(X).view(X.size(0), -1)
            return self._safe_r2(head(z), v)

    @torch.no_grad()
    def _jacobian_metrics(self, enc: nn.Module) -> Dict[str, float]:
        if self.cfg.fast and self.cfg.device == "cpu":
            return {"entropy": float("nan"), "sv_min": float("nan"), "sv_median": float("nan"), "gini": float("nan")}
        
        N_J = 8
        s_vals = []
        for _ in range(N_J):
            X, _ = self.data.sample_features(1, 0)
            X.requires_grad_(True)
            J_full = torch.autograd.functional.jacobian(
                         lambda inp: enc(inp).view(-1), X)
            J = J_full.view(self.cfg.D_z, -1)
            
            s  = torch.linalg.svdvals(J.to(torch.float64))
            s_safe = s + 1e-12
            p = s_safe / s_safe.sum()
            ent    = -(p * p.log()).sum()
            gini_i = 1 - (p * p).sum()
            s_vals.append(torch.tensor([ent, s.min(), s.median(), gini_i]))
        
        stats = torch.stack(s_vals)
        return {
            "entropy":   float(stats[:,0].mean()),
            "sv_min":    float(stats[:,1].mean()),
            "sv_median": float(stats[:,2].mean()),
            "gini":      float(stats[:,3].mean()),
        }

    def _after_train_r2(self, enc: nn.Module) -> float:
        dec = nn.Linear(self.cfg.D_z, self.cfg.r_global, device=self.cfg.device)
        opt = torch.optim.Adam((*enc.parameters(), *dec.parameters()), lr=self.cfg.lr)
        for _ in range(self.cfg.train_steps):
            X, g = self.data.sample_features(self.cfg.batch_train, 0)
            z = enc(X).view(X.size(0), -1)
            loss = F.mse_loss(dec(z), g)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            X, g = self.data.sample_features(self.cfg.batch_eval, 0)
            z = enc(X).view(X.size(0), -1)
            return self._safe_r2(dec(z), g)

    def evaluate(self, enc: nn.Module) -> Dict[str, Any]:
        enc = enc.to(self.cfg.device).eval()
        t0  = time.perf_counter()

        out = {"enc_params": sum(p.numel() for p in enc.parameters())}
        out.update(self._compression(enc))
        out["NCE_bits"]     = self._info_nce(enc)
        out["RetrieveR2"]   = self._retrieval_r2(enc)
        out.update(self._jacobian_metrics(enc))

        if self.cfg.train_steps:
            enc.train()
            out["AfterTrainR2"] = self._after_train_r2(enc)

        out["sec/run"] = time.perf_counter() - t0
        return out

registry = OrderedDict()

def register(name):
    def decorator(cls):
        registry[name] = cls
        return cls
    return decorator

@register("SelfAttn")
class TinySelfAttn(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d_model = 72
        self.proj  = nn.Linear(cfg.D_in, d_model)
        self.attn  = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.norm  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, cfg.D_z)
    def forward(self, x):
        qkv = self.proj(x)               
        h, _ = self.attn(qkv, qkv, qkv)  
        return F.relu(self.head(self.norm(h[:, -1])))

@register("StaticMLP")
class FlattenMLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(cfg.L * cfg.D_in, cfg.D_z), nn.ReLU())
    def forward(self, x): return self.net(x)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_steps", type=int, default=0,
                    help="enable quick fine-tune probe")
    ap.add_argument("--fast", action="store_true",
                    help="skip Jacobian & cut retrieval steps for CPU demo")
    ap.add_argument("--cpu-only", action="store_true",
                    help="force CPU usage even if CUDA is available")
    ap.add_argument("--self-test", action="store_true",
                    help="run automated self-test")
    ap.add_argument("--L", type=int, default=32,
                    help="sequence length")
    ap.add_argument("--ar_coeff", type=float, default=0.8,
                    help="AR coefficient for local features")
    ap.add_argument("--noise", type=float, default=0.03,
                    help="noise level")
    ap.add_argument("--retr_hid", type=int, default=64,
                    help="retrieval head hidden size")
    args = ap.parse_args()
    
    device = "cpu" if args.cpu_only else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dataclasses.replace(cfg_default, 
                              train_steps=args.train_steps, 
                              fast=args.fast,
                              cpu_only=args.cpu_only,
                              device=device,
                              L=args.L,
                              ar_coeff=args.ar_coeff,
                              noise=args.noise,
                              retr_hid=args.retr_hid)

    setup_reproducibility(cfg)
    
    if args.self_test:
        tester = ExpressivityTester(cfg)
        z = TinySelfAttn(cfg).to(cfg.device)(torch.zeros(2, cfg.L, cfg.D_in, device=cfg.device))
        assert z.shape == (2, cfg.D_z), f"Expected shape (2, {cfg.D_z}), got {z.shape}"
        print("✓ quick sanity checks passed")
        sys.exit(0)

    tester = ExpressivityTester(cfg)
    models = {name: cls(cfg) for name, cls in registry.items()}

    cols = ["Model","k-params","sec/run","R2","R2_0.75","R2_0.50","R2_0.25",
            "NCE(bits)","RetrieveR2","entropy","sv_min","sv_median","Gini"]
    if cfg.train_steps: cols.insert(4, "AfterTrainR2")
    print("  ".join(f"{c:<12}" for c in cols))
    print("-"*13*len(cols))

    for name, m in models.items():
        res = tester.evaluate(m)
        row = [
            name,
            f"{res['enc_params']/1e3:.1f}",
            f"{res['sec/run']:.2f}",
            f"{res['R2']:.2f}", f"{res['R2_0.75']:.2f}",
            f"{res['R2_0.50']:.2f}", f"{res['R2_0.25']:.2f}",
            f"{res['NCE_bits']:.2f}",
            f"{res['RetrieveR2']:.2f}",
            f"{res['entropy']:.2f}",
            f"{res['sv_min']:.2e}",
            f"{res['sv_median']:.2e}",
            f"{res['gini']:.2f}",
        ]
        if cfg.train_steps:
            row.insert(4, f"{res['AfterTrainR2']:.2f}")
        print("  ".join(f"{x:<12}" for x in row))
