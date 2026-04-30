"""
═══════════════════════════════════════════════════════════════════════════════
PROJECT ECHO — Calibración de α para φ = [φ_content; α·φ_modal]
═══════════════════════════════════════════════════════════════════════════════

PROBLEMA:
  φ_modal con InfoNCE resolvió A3 (γ_mod=3.433) pero rompió A1 porque las 
  64 dimensiones modales tienen varianza mucho mayor que las 1024 de BGE.
  d_hedge_modal=0.1907 vs d_hedge_content≈0.02 → φ_modal domina la distancia.

SOLUCIÓN:
  Escalar φ_modal por factor α antes de concatenar:
    φ(y) = [φ_content(y); α · φ_modal(y)]
  
  Buscar α tal que A1, A2, A3 se cumplan simultáneamente.

  α grande → φ_modal domina → A3 ✓ pero A1 ✗ (paráfrasis separadas)
  α pequeño → φ_content domina → A1 ✓ pero A3 ✗ (volvemos al problema original)
  α óptimo → equilibrio → A1 ✓, A2 ✓, A3 ✓

USO:
  python phi_alpha_calibration.py
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine as cosine_dist

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = Path(__file__).parent / "reports"

# ═══════════════════════════════════════════════════════════════════════════
# φ_modal architecture (must match training)
# ═══════════════════════════════════════════════════════════════════════════

class PhiModalProjection(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        z = self.projection(x)
        return F.normalize(z, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# AXIOM CORPUS (completo — 45 tripletas del corpus expandido)
# ═══════════════════════════════════════════════════════════════════════════

AXIOM_CORPUS = [
    # Trading
    {"id": "T01", "domain": "trading", "anchor": "The market will rise sharply tomorrow due to strong earnings reports.", "paraphrase": "Prices are expected to surge tomorrow following robust corporate results.", "contradiction": "The market will decline significantly tomorrow despite earnings season.", "hedge": "The market might rise tomorrow, though there is considerable uncertainty about the magnitude."},
    {"id": "T02", "domain": "trading", "anchor": "Bitcoin's volatility is driven primarily by retail speculation.", "paraphrase": "Individual investor speculation is the main driver of Bitcoin price swings.", "contradiction": "Bitcoin's volatility stems mainly from institutional trading and macro hedging.", "hedge": "Bitcoin's volatility could be driven by retail speculation, but institutional factors may also play a significant role."},
    {"id": "T03", "domain": "trading", "anchor": "The trailing stop should be set at 1.5% for this regime.", "paraphrase": "A 1.5% trailing stop is the optimal setting in the current market conditions.", "contradiction": "The trailing stop should be widened to at least 3% in this regime.", "hedge": "A trailing stop around 1.5% seems reasonable, but the optimal value depends on factors I'm not fully certain about."},
    {"id": "T04", "domain": "trading", "anchor": "The current regime is high-volatility with no clear trend direction.", "paraphrase": "We're in a volatile, directionless market environment right now.", "contradiction": "The current regime shows low volatility with a strong bullish trend.", "hedge": "The regime appears to be high-volatility, though I'm not entirely confident in this classification."},
    {"id": "T05", "domain": "trading", "anchor": "The RSI divergence signals an imminent reversal in the short timeframe.", "paraphrase": "Short-term RSI divergence indicates a price reversal is about to happen.", "contradiction": "The RSI divergence is a lagging artifact and has no predictive value for short-term reversals.", "hedge": "The RSI divergence might signal a reversal, but such signals have mixed reliability historically."},
    {"id": "T06", "domain": "trading", "anchor": "Liquidity in the order book is concentrated at the 50,000 level.", "paraphrase": "The main pool of resting orders sits around the 50,000 price point.", "contradiction": "Liquidity is evenly distributed across the order book with no significant concentration.", "hedge": "Liquidity appears concentrated near 50,000, though the order book can shift rapidly."},
    {"id": "T07", "domain": "trading", "anchor": "The toxicity score for this pair indicates unsafe trading conditions.", "paraphrase": "Current toxicity metrics flag this trading pair as too risky to trade.", "contradiction": "The toxicity score is within normal bounds and conditions are safe for trading.", "hedge": "The toxicity score looks elevated, though the orderbook data feeding it might have a lag that makes me less sure."},
    {"id": "T08", "domain": "trading", "anchor": "The funding rate is extremely negative, indicating a short squeeze is likely.", "paraphrase": "Highly negative funding rates suggest shorts will be forced to cover soon.", "contradiction": "The negative funding rate reflects genuine bearish sentiment and shorts will maintain their positions.", "hedge": "The funding rate is quite negative, which sometimes precedes a squeeze, but the signal isn't always reliable."},
    {"id": "T09", "domain": "trading", "anchor": "Dollar bars eliminate the need for time-based resampling in this strategy.", "paraphrase": "Using dollar-denominated bars removes the requirement for fixed-interval time sampling.", "contradiction": "Dollar bars introduce their own artifacts and time-based resampling remains necessary.", "hedge": "Dollar bars probably reduce the need for time resampling, though they may introduce other sampling biases I haven't fully characterized."},
    {"id": "T10", "domain": "trading", "anchor": "The conflict guard correctly blocked this contradictory signal.", "paraphrase": "The signal conflict detection system properly rejected this inconsistent trade signal.", "contradiction": "The conflict guard incorrectly blocked a valid signal due to a false positive.", "hedge": "The conflict guard seems to have blocked the signal appropriately, but I'd want to review the specific conditions that triggered it."},
    # Software
    {"id": "S01", "domain": "software", "anchor": "The bug is in the database connection pooling logic.", "paraphrase": "The defect originates from the DB connection pool management code.", "contradiction": "The bug is in the API serialization layer, not the database connection.", "hedge": "The bug is probably in the connection pooling logic, but I'd want to verify with more debugging."},
    {"id": "S02", "domain": "software", "anchor": "Using PostgreSQL for this workload will outperform MongoDB significantly.", "paraphrase": "PostgreSQL is a much better choice than MongoDB for this specific use case.", "contradiction": "MongoDB's document model makes it far superior to PostgreSQL for this workload.", "hedge": "PostgreSQL might outperform MongoDB here, though it depends on query patterns I haven't fully analyzed."},
    {"id": "S03", "domain": "software", "anchor": "The memory leak is caused by unclosed file handles in the data pipeline.", "paraphrase": "File handles not being properly released in the data pipeline are creating the memory leak.", "contradiction": "The memory leak stems from the caching layer retaining stale objects, not file handles.", "hedge": "The memory leak could be caused by unclosed file handles, but there might be other contributing factors."},
    {"id": "S04", "domain": "software", "anchor": "The NSSM service configuration is correct and the service will auto-restart on failure.", "paraphrase": "NSSM has been properly configured to automatically recover the service after any crash.", "contradiction": "The NSSM configuration has a critical error that will prevent service auto-recovery.", "hedge": "The NSSM setup looks correct for auto-restart, but I haven't tested every edge case failure mode."},
    {"id": "S05", "domain": "software", "anchor": "The race condition occurs because the exit tracker and executor share mutable state without locking.", "paraphrase": "Shared mutable state between the exit tracker and executor without synchronization causes the race condition.", "contradiction": "There is no race condition; the exit tracker and executor operate on separate state copies.", "hedge": "There might be a race condition in the shared state between exit tracker and executor, but I'd need to trace the exact execution order to confirm."},
    {"id": "S06", "domain": "software", "anchor": "Migrating to async IO will reduce the API latency by at least 40%.", "paraphrase": "Switching to asynchronous input/output will cut API response times by 40% or more.", "contradiction": "Async IO will add complexity without meaningful latency improvement for this workload.", "hedge": "Async IO could potentially reduce latency, though the actual improvement depends on the ratio of IO-bound to CPU-bound work in the pipeline."},
    {"id": "S07", "domain": "software", "anchor": "The ALTER TABLE command will fix the missing btc_change_lag column without data loss.", "paraphrase": "Adding the btc_change_lag column via ALTER TABLE is safe and preserves all existing data.", "contradiction": "The ALTER TABLE will cause a full table rewrite and risk data corruption on large tables.", "hedge": "ALTER TABLE should safely add the column, though on very large tables there could be locking issues I haven't accounted for."},
    {"id": "S08", "domain": "software", "anchor": "The WebSocket connection drops because the heartbeat interval exceeds the server timeout.", "paraphrase": "Server timeout is shorter than the client heartbeat interval, causing WebSocket disconnections.", "contradiction": "The WebSocket drops are caused by network-level packet loss, not heartbeat timing.", "hedge": "The heartbeat interval might be causing the disconnections, but network issues could also be a contributing factor."},
    {"id": "S09", "domain": "software", "anchor": "The cron job fails silently because stderr is not redirected to the log file.", "paraphrase": "Error output from the cron job is lost because stderr isn't captured in the logging configuration.", "contradiction": "The cron job fails because of a permissions issue on the target directory, not logging configuration.", "hedge": "The silent failure is probably a stderr redirection issue, though there could be other factors suppressing the error output."},
    {"id": "S10", "domain": "software", "anchor": "Indexing the timestamp column will speed up the query from 12 seconds to under 500 milliseconds.", "paraphrase": "Adding a database index on the timestamp field will reduce query time from 12s to below half a second.", "contradiction": "The timestamp index won't help because the bottleneck is the JOIN operation, not the WHERE clause.", "hedge": "An index on timestamp should help significantly, though the actual speedup depends on table statistics and the query planner's choices."},
    # ML Config
    {"id": "M01", "domain": "ml_config", "anchor": "Increasing the learning rate to 3e-4 will accelerate convergence without overfitting.", "paraphrase": "A learning rate of 3e-4 will speed up training while maintaining generalization.", "contradiction": "A learning rate of 3e-4 is too high and will cause training instability and overfitting.", "hedge": "A learning rate around 3e-4 might work well, but the optimal value depends on the batch size and architecture specifics."},
    {"id": "M02", "domain": "ml_config", "anchor": "The model is overfitting because the training set lacks diversity in regime transitions.", "paraphrase": "Overfitting occurs because the training data doesn't contain enough varied market regime changes.", "contradiction": "The model is underfitting due to excessive regularization, not a data diversity issue.", "hedge": "The overfitting might be related to insufficient regime diversity in training, though I'd want to check other factors too."},
    {"id": "M03", "domain": "ml_config", "anchor": "Removing cyclic time features eliminated the spurious correlation with calendar patterns.", "paraphrase": "Dropping the cyclical temporal features fixed the false correlation to calendar-based patterns.", "contradiction": "The cyclic time features were providing genuine signal and removing them degraded model performance.", "hedge": "Removing cyclic features seems to have reduced spurious correlations, though I'm not completely sure no useful signal was lost."},
    {"id": "M04", "domain": "ml_config", "anchor": "The dollar bar transformation eliminates the non-stationarity in the price series.", "paraphrase": "Converting to dollar bars successfully removes the time-varying properties of raw price data.", "contradiction": "Dollar bars still exhibit significant non-stationarity and require additional transforms.", "hedge": "Dollar bars should reduce non-stationarity, though some residual time-dependence might remain."},
    {"id": "M05", "domain": "ml_config", "anchor": "Purged k-fold cross-validation completely eliminates look-ahead bias in the backtest.", "paraphrase": "Using purged k-fold CV removes all temporal leakage from the backtesting evaluation.", "contradiction": "Purged k-fold still leaks information through shared feature engineering and normalization statistics.", "hedge": "Purged k-fold should eliminate most look-ahead bias, though subtle leakage through preprocessing steps might remain."},
    {"id": "M06", "domain": "ml_config", "anchor": "The OOB precision of 0.89 confirms the model generalizes well to unseen data.", "paraphrase": "An out-of-bag precision score of 0.89 validates strong generalization performance.", "contradiction": "The 0.89 OOB precision is misleading because the OOB samples are not truly independent of the training distribution.", "hedge": "The OOB precision looks promising at 0.89, though OOB estimates can sometimes be optimistic depending on the correlation structure in the data."},
    {"id": "M07", "domain": "ml_config", "anchor": "Fractional differentiation preserves memory while achieving stationarity.", "paraphrase": "Using fractional differencing maintains the long-memory properties of the series while making it stationary.", "contradiction": "Fractional differentiation destroys the predictive signal by removing too much of the original series structure.", "hedge": "Fractional differentiation should balance memory preservation and stationarity, though finding the right differentiation order requires careful empirical testing."},
    # Technical
    {"id": "C01", "domain": "technical", "anchor": "Transformer attention mechanisms compute pairwise token relationships in parallel.", "paraphrase": "The attention layers in transformers calculate interactions between all token pairs simultaneously.", "contradiction": "Transformer attention operates sequentially, processing one token relationship at a time.", "hedge": "Transformers likely compute attention in parallel, though the exact implementation details may vary across frameworks."},
    {"id": "C02", "domain": "technical", "anchor": "Conformal prediction provides distribution-free coverage guarantees under exchangeability.", "paraphrase": "Under the assumption of exchangeable data, conformal methods guarantee coverage without distributional assumptions.", "contradiction": "Conformal prediction requires strong distributional assumptions and fails under exchangeability alone.", "hedge": "Conformal prediction should provide coverage guarantees under exchangeability, though I'm less certain about the tightness of those guarantees in practice."},
    {"id": "C03", "domain": "technical", "anchor": "The PAC-Bayes bound tightens as the posterior concentrates around a single mode.", "paraphrase": "PAC-Bayes generalization bounds improve when the learned posterior has low entropy around one optimum.", "contradiction": "PAC-Bayes bounds are independent of posterior concentration and depend only on the prior.", "hedge": "The PAC-Bayes bound probably tightens with posterior concentration, though the relationship might not be monotonic in all cases."},
    {"id": "C04", "domain": "technical", "anchor": "The KL divergence between the approximate and true posterior is bounded by the ELBO gap.", "paraphrase": "The ELBO gap provides an upper bound on how far the variational approximation is from the true posterior.", "contradiction": "The ELBO gap is unrelated to the KL divergence between approximate and true posteriors.", "hedge": "The ELBO gap should bound the KL divergence, though in practice the bound can be very loose for complex posteriors."},
    {"id": "C05", "domain": "technical", "anchor": "Gradient clipping prevents exploding gradients in deep recurrent networks.", "paraphrase": "Clipping gradient norms stops the gradient explosion problem in deep RNN architectures.", "contradiction": "Gradient clipping masks the real problem and the network should be redesigned to avoid exploding gradients entirely.", "hedge": "Gradient clipping generally helps with exploding gradients, though it can sometimes interfere with optimization dynamics in ways that aren't fully understood."},
    {"id": "C06", "domain": "technical", "anchor": "The softmax bottleneck limits the expressiveness of language models with tied embeddings.", "paraphrase": "Tying input and output embeddings creates a softmax bottleneck that constrains what the model can represent.", "contradiction": "The softmax bottleneck is a theoretical concern with minimal practical impact on modern language model performance.", "hedge": "The softmax bottleneck might limit expressiveness in some settings, though its practical significance compared to other bottlenecks is debatable."},
    {"id": "C07", "domain": "technical", "anchor": "Layer normalization stabilizes training by reducing internal covariate shift.", "paraphrase": "Applying layer norm reduces the shifting distribution of activations and makes training more stable.", "contradiction": "Layer normalization works by smoothing the loss landscape, not by reducing covariate shift as commonly claimed.", "hedge": "Layer normalization appears to stabilize training, though the exact mechanism might differ from the originally proposed covariate shift explanation."},
    {"id": "C08", "domain": "technical", "anchor": "The Fisher information matrix approximates the curvature of the loss landscape at convergence.", "paraphrase": "At convergence, the loss surface curvature can be approximated using the Fisher information matrix.", "contradiction": "The Fisher information matrix is a poor approximation of loss curvature because it ignores higher-order terms.", "hedge": "The Fisher information matrix provides a reasonable curvature approximation, though its accuracy degrades in regions far from the optimum."},
    {"id": "C09", "domain": "technical", "anchor": "Sparse attention reduces the quadratic complexity of self-attention to linear.", "paraphrase": "Using sparse attention patterns brings the computational cost of self-attention from O(n²) down to O(n).", "contradiction": "Sparse attention sacrifices too much representational capacity and full quadratic attention remains necessary for high performance.", "hedge": "Sparse attention can reduce complexity substantially, though whether it achieves truly linear scaling depends on the specific sparsity pattern and task."},
    {"id": "C10", "domain": "technical", "anchor": "Mixture of experts routes tokens to specialized sub-networks based on learned gating.", "paraphrase": "MoE architectures use a learned gating mechanism to direct each token to the most relevant expert sub-network.", "contradiction": "MoE gating is largely random in practice and the experts do not develop meaningful specialization.", "hedge": "MoE routing seems to develop some specialization, though the degree of meaningful expert differentiation is still being studied."},
    # Epistemic
    {"id": "E01", "domain": "epistemic", "anchor": "I am certain this configuration will improve the model's performance.", "paraphrase": "This configuration change will definitely enhance the model's results.", "contradiction": "This configuration is likely to degrade performance based on similar past experiments.", "hedge": "I think this configuration might improve performance, but I'm drawing on limited evidence."},
    {"id": "E02", "domain": "epistemic", "anchor": "The correlation between these features is causal, not merely statistical.", "paraphrase": "There is a genuine causal link between these features, beyond simple correlation.", "contradiction": "The apparent relationship between these features is purely correlational with no causal basis.", "hedge": "The features appear correlated and might be causally linked, but I can't rule out confounding without further analysis."},
    {"id": "E03", "domain": "epistemic", "anchor": "Based on my analysis, the optimal position size is 2% of capital.", "paraphrase": "My calculations indicate that allocating 2% of capital per position is ideal.", "contradiction": "A 2% position size is far too conservative; 5% would be optimal given current conditions.", "hedge": "A position size around 2% seems reasonable, though the Kelly criterion calculation depends on parameters I'm estimating with uncertainty."},
    {"id": "E04", "domain": "epistemic", "anchor": "This approach has been validated across all relevant market regimes.", "paraphrase": "Testing confirms this method works in every significant type of market environment.", "contradiction": "This approach has only been tested in bull markets and may fail in bear or sideways regimes.", "hedge": "This approach has been tested in several regimes, though I'm not fully confident it generalizes to all possible market conditions."},
    {"id": "E05", "domain": "epistemic", "anchor": "I have high confidence in this recommendation because it is supported by extensive empirical evidence.", "paraphrase": "Strong empirical backing gives me great certainty about this recommendation.", "contradiction": "Despite appearing well-supported, the empirical evidence is cherry-picked and unreliable.", "hedge": "I'm fairly confident in this recommendation, though the empirical evidence, while extensive, comes from a limited set of conditions."},
    {"id": "E06", "domain": "epistemic", "anchor": "The backtest results prove this strategy is profitable in all conditions.", "paraphrase": "Backtesting demonstrates that this strategy generates positive returns regardless of market conditions.", "contradiction": "The backtest is overfitted to historical data and the strategy will fail in live trading.", "hedge": "The backtest results are encouraging, but I'm aware that historical performance doesn't guarantee future results and there might be overfitting I haven't detected."},
    {"id": "E07", "domain": "epistemic", "anchor": "I understand exactly why the SHORT model outperforms the LONG model in this regime.", "paraphrase": "The reason for SHORT's superior performance over LONG in this market environment is completely clear to me.", "contradiction": "Nobody fully understands why SHORT outperforms LONG here; the performance gap defies our current theoretical framework.", "hedge": "I have a hypothesis about why SHORT outperforms LONG, but the RSI paradox we documented suggests there are dynamics I don't fully grasp."},
    {"id": "E08", "domain": "epistemic", "anchor": "The data unambiguously supports rejecting the null hypothesis.", "paraphrase": "Statistical evidence clearly and definitively favors rejecting the null hypothesis.", "contradiction": "The data is inconclusive and does not provide sufficient evidence to reject the null.", "hedge": "The data leans toward rejecting the null, though the p-value is close to the threshold and the sample size leaves me somewhat uncertain."},
]


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_with_alpha(encoder, phi_modal, corpus, alpha, device="cpu"):
    """Evalúa axiomas A1-A3 con φ = [φ_content; α·φ_modal]."""
    
    phi_modal.eval()
    
    d_para, d_contra, d_hedge = [], [], []
    domain_data = {}
    
    with torch.no_grad():
        for item in corpus:
            texts = [item["anchor"], item["paraphrase"], item["contradiction"], item["hedge"]]
            
            content_embs = encoder.encode(
                texts, normalize_embeddings=True,
                show_progress_bar=False, convert_to_tensor=True
            ).to(device)
            
            modal_embs = phi_modal(content_embs)
            
            # Concatenar con escala α
            combined = torch.cat([content_embs, alpha * modal_embs], dim=-1)
            combined = F.normalize(combined, dim=-1)
            
            embs_np = combined.cpu().numpy()
            dp = cosine_dist(embs_np[0], embs_np[1])
            dc = cosine_dist(embs_np[0], embs_np[2])
            dh = cosine_dist(embs_np[0], embs_np[3])
            
            d_para.append(dp)
            d_contra.append(dc)
            d_hedge.append(dh)
            
            dom = item["domain"]
            if dom not in domain_data:
                domain_data[dom] = {"para": [], "contra": [], "hedge": []}
            domain_data[dom]["para"].append(dp)
            domain_data[dom]["contra"].append(dc)
            domain_data[dom]["hedge"].append(dh)
    
    d_para = np.array(d_para)
    d_contra = np.array(d_contra)
    d_hedge = np.array(d_hedge)
    
    eps_lex = np.percentile(d_contra, 10)
    mean_para = np.mean(d_para)
    mean_contra = np.mean(d_contra)
    mean_hedge = np.mean(d_hedge)
    
    gamma_prop = mean_contra / eps_lex if eps_lex > 0 else 0
    gamma_mod = mean_hedge / eps_lex if eps_lex > 0 else 0
    
    a1 = bool(mean_para < eps_lex)
    a2 = bool(gamma_prop >= 1.3)
    a3 = bool(gamma_mod >= 2.0)
    
    # Tasa individual A1
    a1_rate = float(np.mean(d_para <= eps_lex))
    
    # Domain breakdown
    dom_results = {}
    for dom, vals in sorted(domain_data.items()):
        mp = np.mean(vals["para"])
        mc = np.mean(vals["contra"])
        mh = np.mean(vals["hedge"])
        dom_results[dom] = {
            "n": len(vals["para"]),
            "gamma_prop": float(mc / mp) if mp > 0 else 0,
            "gamma_mod": float(mh / mp) if mp > 0 else 0,
        }
    
    return {
        "alpha": alpha,
        "mean_para": float(mean_para),
        "mean_contra": float(mean_contra),
        "mean_hedge": float(mean_hedge),
        "eps_lex": float(eps_lex),
        "gamma_prop": float(gamma_prop),
        "gamma_mod": float(gamma_mod),
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "a1_rate": a1_rate,
        "all_pass": a1 and a2 and a3,
        "domain_results": dom_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print()
    print("=" * 76)
    print("  PROYECTO ECHO — Calibración de α para φ = [φ_content; α·φ_modal]")
    print("=" * 76)
    print()
    
    # Cargar encoder
    from sentence_transformers import SentenceTransformer
    print("  Cargando BGE-large...", flush=True)
    encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    print("  OK")
    
    # Cargar φ_modal entrenado
    print("  Cargando φ_modal...", flush=True)
    checkpoint = torch.load(MODELS_DIR / "phi_modal_bge_large.pt", map_location=device, weights_only=False)
    phi_modal = PhiModalProjection(
        input_dim=1024,
        hidden_dim=checkpoint["config"]["hidden_dim"],
        output_dim=checkpoint["config"]["projection_dim"]
    ).to(device)
    phi_modal.load_state_dict(checkpoint["state_dict"])
    phi_modal.eval()
    print(f"  OK — mejor γ_mod en training: {checkpoint['best_gamma_mod_val']:.3f}")
    print()
    
    # ═══ FASE 1: Grid search grueso ═══
    print("━" * 76)
    print("  FASE 1: Grid Search Grueso (α ∈ [0.01, 2.0])")
    print("━" * 76)
    print()
    
    alphas_coarse = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 
                     0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    
    results_coarse = []
    
    print(f"  {'α':>6s} {'A1':>4s} {'A2':>4s} {'A3':>4s} {'γ_prop':>8s} {'γ_mod':>8s} {'d(para)':>8s} {'ε_lex':>8s} {'A1%':>6s} {'ALL':>5s}")
    print(f"  {'─'*6} {'─'*4} {'─'*4} {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*5}")
    
    for alpha in alphas_coarse:
        r = evaluate_with_alpha(encoder, phi_modal, AXIOM_CORPUS, alpha, device)
        results_coarse.append(r)
        
        all_mark = "✓✓✓" if r["all_pass"] else ""
        print(f"  {alpha:6.3f} {'✓' if r['a1'] else '✗':>4s} {'✓' if r['a2'] else '✗':>4s} {'✓' if r['a3'] else '✗':>4s} "
              f"{r['gamma_prop']:8.3f} {r['gamma_mod']:8.3f} {r['mean_para']:8.4f} {r['eps_lex']:8.4f} "
              f"{r['a1_rate']:5.1%} {all_mark:>5s}")
    
    print()
    
    # Encontrar rango donde todos pasan
    passing = [r for r in results_coarse if r["all_pass"]]
    
    if passing:
        alpha_min = min(r["alpha"] for r in passing)
        alpha_max = max(r["alpha"] for r in passing)
        print(f"  Rango donde A1+A2+A3 pasan: α ∈ [{alpha_min:.3f}, {alpha_max:.3f}]")
    else:
        # Encontrar la frontera de A1 y A3
        a1_max = max((r["alpha"] for r in results_coarse if r["a1"]), default=0)
        a3_min = min((r["alpha"] for r in results_coarse if r["a3"]), default=999)
        print(f"  ⚠ Ningún α satisface los 3 axiomas simultáneamente")
        print(f"    A1 pasa hasta α ≤ {a1_max:.3f}")
        print(f"    A3 pasa desde α ≥ {a3_min:.3f}")
        
        if a3_min <= a1_max:
            print(f"    Overlap existe: refinar en [{a3_min:.3f}, {a1_max:.3f}]")
            alpha_min = a3_min
            alpha_max = a1_max
        else:
            print(f"    Gap: [{a1_max:.3f}, {a3_min:.3f}] — no hay solución exacta")
            print(f"    Refinando en zona de transición...")
            alpha_min = a1_max * 0.8
            alpha_max = a3_min * 1.2
    
    print()
    
    # ═══ FASE 2: Grid search fino ═══
    print("━" * 76)
    print(f"  FASE 2: Grid Search Fino (α ∈ [{alpha_min:.3f}, {alpha_max:.3f}])")
    print("━" * 76)
    print()
    
    alphas_fine = np.linspace(
        max(0.001, alpha_min * 0.7), 
        alpha_max * 1.3, 
        50
    )
    
    results_fine = []
    
    print(f"  {'α':>6s} {'A1':>4s} {'A2':>4s} {'A3':>4s} {'γ_prop':>8s} {'γ_mod':>8s} {'d(para)':>8s} {'ε_lex':>8s} {'A1%':>6s} {'ALL':>5s}")
    print(f"  {'─'*6} {'─'*4} {'─'*4} {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*5}")
    
    for alpha in alphas_fine:
        alpha = round(float(alpha), 4)
        r = evaluate_with_alpha(encoder, phi_modal, AXIOM_CORPUS, alpha, device)
        results_fine.append(r)
        
        all_mark = "✓✓✓" if r["all_pass"] else ""
        print(f"  {alpha:6.4f} {'✓' if r['a1'] else '✗':>4s} {'✓' if r['a2'] else '✗':>4s} {'✓' if r['a3'] else '✗':>4s} "
              f"{r['gamma_prop']:8.3f} {r['gamma_mod']:8.3f} {r['mean_para']:8.4f} {r['eps_lex']:8.4f} "
              f"{r['a1_rate']:5.1%} {all_mark:>5s}")
    
    print()
    
    # ═══ SELECCIÓN ÓPTIMA ═══
    all_results = results_coarse + results_fine
    passing_all = [r for r in all_results if r["all_pass"]]
    
    print("=" * 76)
    print("  SELECCIÓN DE α ÓPTIMO")
    print("=" * 76)
    print()
    
    if passing_all:
        # Seleccionar α que maximiza min(margen_A1, margen_A3)
        # margen_A1 = eps_lex - mean_para (positivo = pasa)
        # margen_A3 = gamma_mod - 2.0 (positivo = pasa)
        
        def margin_score(r):
            margin_a1 = (r["eps_lex"] - r["mean_para"]) / r["eps_lex"] if r["eps_lex"] > 0 else 0
            margin_a3 = (r["gamma_mod"] - 2.0) / r["gamma_mod"] if r["gamma_mod"] > 0 else 0
            return min(margin_a1, margin_a3)
        
        best = max(passing_all, key=margin_score)
        best_margin = margin_score(best)
        
        print(f"  α ÓPTIMO: {best['alpha']:.4f}")
        print(f"  Margen equilibrado: {best_margin:.4f}")
        print()
        print(f"  Axiomas con α = {best['alpha']:.4f}:")
        print(f"    A1: ✓  media_para={best['mean_para']:.4f} < ε_lex={best['eps_lex']:.4f}  (margen: {best['eps_lex']-best['mean_para']:.4f})")
        print(f"    A2: ✓  γ_prop={best['gamma_prop']:.3f} ≥ 1.3")
        print(f"    A3: ✓  γ_mod={best['gamma_mod']:.3f} ≥ 2.0")
        print(f"    Tasa individual A1: {best['a1_rate']:.1%}")
        print()
        
        # Domain breakdown para α óptimo
        print("  Análisis por dominio con α óptimo:")
        print(f"  {'Dominio':15s} {'N':>4s} {'γ_prop':>8s} {'γ_mod':>8s}")
        print(f"  {'─'*15} {'─'*4} {'─'*8} {'─'*8}")
        for dom, vals in sorted(best["domain_results"].items()):
            print(f"  {dom:15s} {vals['n']:4d} {vals['gamma_prop']:8.2f} {vals['gamma_mod']:8.2f}")
        print()
        
        print("  ┌──────────────────────────────────────────────────────────────────────┐")
        print("  │  ✓ φ ES ADMISIBLE                                                    │")
        print("  │                                                                      │")
        print(f"  │  φ(y) = [BGE-large(y); {best['alpha']:.4f} · φ_modal(y)]" + " " * (38 - len(f"{best['alpha']:.4f}")) + "│")
        print(f"  │  Dimensión: 1024 + 64 = 1088                                       │")
        print("  │                                                                      │")
        print(f"  │  A1 ✓ (γ margin: {best['eps_lex']-best['mean_para']:.4f})    " +
              f"A2 ✓ (γ_prop: {best['gamma_prop']:.3f})    " +
              f"A3 ✓ (γ_mod: {best['gamma_mod']:.3f})" + "  │")
        print("  │                                                                      │")
        print("  │  PROCEDER A H1: Descomposición de incertidumbre epistémica           │")
        print("  └──────────────────────────────────────────────────────────────────────┘")
        
        # Guardar α óptimo en el checkpoint del modelo
        checkpoint["optimal_alpha"] = best["alpha"]
        checkpoint["axiom_results_at_optimal_alpha"] = {
            "a1": best["a1"], "a2": best["a2"], "a3": best["a3"],
            "gamma_prop": best["gamma_prop"], "gamma_mod": best["gamma_mod"],
        }
        torch.save(checkpoint, MODELS_DIR / "phi_modal_bge_large.pt")
        print(f"\n  α óptimo guardado en modelo: {MODELS_DIR / 'phi_modal_bge_large.pt'}")
        
    else:
        # Encontrar α más cercano a satisfacer todo
        def gap_score(r):
            gap_a1 = max(0, r["mean_para"] - r["eps_lex"])
            gap_a3 = max(0, 2.0 - r["gamma_mod"])
            return gap_a1 + gap_a3
        
        best = min(all_results, key=gap_score)
        
        print(f"  ⚠ No se encontró α que satisfaga A1+A2+A3 simultáneamente")
        print(f"  Mejor compromiso: α = {best['alpha']:.4f}")
        print(f"    A1: {'✓' if best['a1'] else '✗'}  mean_para={best['mean_para']:.4f}, ε_lex={best['eps_lex']:.4f}")
        print(f"    A2: {'✓' if best['a2'] else '✗'}  γ_prop={best['gamma_prop']:.3f}")
        print(f"    A3: {'✓' if best['a3'] else '✗'}  γ_mod={best['gamma_mod']:.3f}")
        print()
        print("  ACCIONES POSIBLES:")
        print("    1. Aceptar compromiso y proceder con el mejor α disponible")
        print("    2. Re-entrenar φ_modal con regularización más fuerte")
        print("    3. Usar ponderación asimétrica: φ = [w1·φ_content; w2·φ_modal]")
    
    print()
    
    # Guardar reporte
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "project": "ECHO",
        "module": "alpha_calibration",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "corpus_size": len(AXIOM_CORPUS),
        "coarse_results": [{k: v for k, v in r.items() if k != "domain_results"} for r in results_coarse],
        "fine_results": [{k: v for k, v in r.items() if k != "domain_results"} for r in results_fine],
        "optimal_alpha": best["alpha"] if passing_all else None,
        "best_result": {k: v for k, v in best.items()},
        "all_pass_found": bool(passing_all),
    }
    
    report_path = REPORTS_DIR / "alpha_calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Reporte guardado: {report_path}")
    print()


if __name__ == "__main__":
    main()