"""
═══════════════════════════════════════════════════════════════════════════════
PROJECT ECHO — PROTOCOLO DE VALIDACIÓN DE φ (Proyección Semántica)
═══════════════════════════════════════════════════════════════════════════════

Referencia: ECHO Marco Matemático v2.0, §2.1.1
Autores: Kent Valera Chirinos & Claude (Anthropic)
Fecha: Abril 2026

Este módulo implementa:
  1. Generación del corpus de validación (tripletas: paráfrasis, contradicción, hedge)
  2. Implementación de φ_content (encoder semántico de contenido)
  3. Implementación de φ_modal (proyección de modalidad epistémica)  
  4. Verificación de los 4 axiomas (A1–A4)
  5. Reporte de admisibilidad con estadísticas bootstrap

DECISIÓN DE DISEÑO:
  φ = [φ_content(y); φ_modal(y)]
  Esto separa "¿sobre qué habla?" de "¿con qué certeza lo dice?"

USO:
  python phi_validation_protocol.py [--model MODEL_NAME] [--n-bootstrap N]

RESULTADO:
  - Reporte de admisibilidad (JSON + texto)
  - Diagnóstico por axioma
  - Recomendación: ADMISIBLE / NO ADMISIBLE / PARCIALMENTE ADMISIBLE
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import time
import argparse
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cosine as cosine_dist

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: CORPUS DE VALIDACIÓN — TRIPLETAS
# ═══════════════════════════════════════════════════════════════════════════

# Cada tripleta contiene:
#   - anchor: afirmación base
#   - paraphrase: misma semántica, diferente léxico (para A1)
#   - contradiction: semántica opuesta/diferente (para A2)
#   - hedge: misma semántica + incertidumbre modal (para A3)
#
# Cubrimos 5 dominios relevantes para ECHO/KAIRI:
#   - Trading/mercados financieros
#   - Ingeniería de software / debugging
#   - Configuración de modelos ML
#   - Conceptos técnicos generales
#   - Meta-razonamiento / epistémico

VALIDATION_CORPUS = [
    # ─── DOMINIO: Trading / Mercados ───────────────────────────────────────
    {
        "domain": "trading",
        "anchor": "The market will rise sharply tomorrow due to strong earnings reports.",
        "paraphrase": "Prices are expected to surge tomorrow following robust corporate results.",
        "contradiction": "The market will decline significantly tomorrow despite earnings season.",
        "hedge": "The market might rise tomorrow, though there is considerable uncertainty about the magnitude."
    },
    {
        "domain": "trading",
        "anchor": "Bitcoin's volatility is driven primarily by retail speculation.",
        "paraphrase": "Individual investor speculation is the main driver of Bitcoin price swings.",
        "contradiction": "Bitcoin's volatility stems mainly from institutional trading and macro hedging.",
        "hedge": "Bitcoin's volatility could be driven by retail speculation, but institutional factors may also play a significant role."
    },
    {
        "domain": "trading",
        "anchor": "The trailing stop should be set at 1.5% for this regime.",
        "paraphrase": "A 1.5% trailing stop is the optimal setting in the current market conditions.",
        "contradiction": "The trailing stop should be widened to at least 3% in this regime.",
        "hedge": "A trailing stop around 1.5% seems reasonable, but the optimal value depends on factors I'm not fully certain about."
    },
    {
        "domain": "trading",
        "anchor": "The current regime is high-volatility with no clear trend direction.",
        "paraphrase": "We're in a volatile, directionless market environment right now.",
        "contradiction": "The current regime shows low volatility with a strong bullish trend.",
        "hedge": "The regime appears to be high-volatility, though I'm not entirely confident in this classification."
    },
    {
        "domain": "trading",
        "anchor": "The RSI divergence signals an imminent reversal in the short timeframe.",
        "paraphrase": "Short-term RSI divergence indicates a price reversal is about to happen.",
        "contradiction": "The RSI divergence is a lagging artifact and has no predictive value for short-term reversals.",
        "hedge": "The RSI divergence might signal a reversal, but such signals have mixed reliability historically."
    },
    {
        "domain": "trading",
        "anchor": "Liquidity in the order book is concentrated at the 50,000 level.",
        "paraphrase": "The main pool of resting orders sits around the 50,000 price point.",
        "contradiction": "Liquidity is evenly distributed across the order book with no significant concentration.",
        "hedge": "Liquidity appears concentrated near 50,000, though the order book can shift rapidly."
    },
    # ─── DOMINIO: Software / Debugging ─────────────────────────────────────
    {
        "domain": "software",
        "anchor": "The bug is in the database connection pooling logic.",
        "paraphrase": "The defect originates from the DB connection pool management code.",
        "contradiction": "The bug is in the API serialization layer, not the database connection.",
        "hedge": "The bug is probably in the connection pooling logic, but I'd want to verify with more debugging."
    },
    {
        "domain": "software",
        "anchor": "Using PostgreSQL for this workload will outperform MongoDB significantly.",
        "paraphrase": "PostgreSQL is a much better choice than MongoDB for this specific use case.",
        "contradiction": "MongoDB's document model makes it far superior to PostgreSQL for this workload.",
        "hedge": "PostgreSQL might outperform MongoDB here, though it depends on query patterns I haven't fully analyzed."
    },
    {
        "domain": "software",
        "anchor": "The memory leak is caused by unclosed file handles in the data pipeline.",
        "paraphrase": "File handles not being properly released in the data pipeline are creating the memory leak.",
        "contradiction": "The memory leak stems from the caching layer retaining stale objects, not file handles.",
        "hedge": "The memory leak could be caused by unclosed file handles, but there might be other contributing factors."
    },
    {
        "domain": "software",
        "anchor": "Implementing retry logic with exponential backoff will resolve the timeout issue.",
        "paraphrase": "Adding exponential backoff retries will fix the connection timeout problem.",
        "contradiction": "Retry logic will only mask the problem; the root cause is insufficient server capacity.",
        "hedge": "Exponential backoff retries might help with the timeouts, though I'm not sure they address the underlying cause."
    },
    # ─── DOMINIO: ML / Configuración de Modelos ───────────────────────────
    {
        "domain": "ml_config",
        "anchor": "Increasing the learning rate to 3e-4 will accelerate convergence without overfitting.",
        "paraphrase": "A learning rate of 3e-4 will speed up training while maintaining generalization.",
        "contradiction": "A learning rate of 3e-4 is too high and will cause training instability and overfitting.",
        "hedge": "A learning rate around 3e-4 might work well, but the optimal value depends on the batch size and architecture specifics."
    },
    {
        "domain": "ml_config",
        "anchor": "The model is overfitting because the training set lacks diversity in regime transitions.",
        "paraphrase": "Overfitting occurs because the training data doesn't contain enough varied market regime changes.",
        "contradiction": "The model is underfitting due to excessive regularization, not a data diversity issue.",
        "hedge": "The overfitting might be related to insufficient regime diversity in training, though I'd want to check other factors too."
    },
    {
        "domain": "ml_config",
        "anchor": "Removing cyclic time features eliminated the spurious correlation with calendar patterns.",
        "paraphrase": "Dropping the cyclical temporal features fixed the false correlation to calendar-based patterns.",
        "contradiction": "The cyclic time features were providing genuine signal and removing them degraded model performance.",
        "hedge": "Removing cyclic features seems to have reduced spurious correlations, though I'm not completely sure no useful signal was lost."
    },
    {
        "domain": "ml_config",
        "anchor": "Feature importance shows that order book imbalance is the strongest predictor.",
        "paraphrase": "The most predictive feature according to importance analysis is the order book imbalance ratio.",
        "contradiction": "Price momentum, not order book imbalance, is the dominant predictive feature in this model.",
        "hedge": "Order book imbalance appears to be the strongest feature, but importance measures can be unstable across different data splits."
    },
    # ─── DOMINIO: Conceptos Técnicos ───────────────────────────────────────
    {
        "domain": "technical",
        "anchor": "Transformer attention mechanisms compute pairwise token relationships in parallel.",
        "paraphrase": "The attention layers in transformers calculate interactions between all token pairs simultaneously.",
        "contradiction": "Transformer attention operates sequentially, processing one token relationship at a time.",
        "hedge": "Transformers likely compute attention in parallel, though the exact implementation details may vary across frameworks."
    },
    {
        "domain": "technical",
        "anchor": "Conformal prediction provides distribution-free coverage guarantees under exchangeability.",
        "paraphrase": "Under the assumption of exchangeable data, conformal methods guarantee coverage without distributional assumptions.",
        "contradiction": "Conformal prediction requires strong distributional assumptions and fails under exchangeability alone.",
        "hedge": "Conformal prediction should provide coverage guarantees under exchangeability, though I'm less certain about the tightness of those guarantees in practice."
    },
    {
        "domain": "technical",
        "anchor": "The PAC-Bayes bound tightens as the posterior concentrates around a single mode.",
        "paraphrase": "PAC-Bayes generalization bounds improve when the learned posterior has low entropy around one optimum.",
        "contradiction": "PAC-Bayes bounds are independent of posterior concentration and depend only on the prior.",
        "hedge": "The PAC-Bayes bound probably tightens with posterior concentration, though the relationship might not be monotonic in all cases."
    },
    {
        "domain": "technical",
        "anchor": "Batch normalization reduces internal covariate shift during training.",
        "paraphrase": "Using batch norm mitigates the problem of shifting activation distributions across layers.",
        "contradiction": "Batch normalization works by smoothing the loss landscape, not by reducing covariate shift.",
        "hedge": "Batch normalization is believed to reduce internal covariate shift, though recent evidence suggests the mechanism might be different than originally proposed."
    },
    # ─── DOMINIO: Meta-razonamiento / Epistémico ──────────────────────────
    {
        "domain": "epistemic",
        "anchor": "I am certain this configuration will improve the model's performance.",
        "paraphrase": "This configuration change will definitely enhance the model's results.",
        "contradiction": "This configuration is likely to degrade performance based on similar past experiments.",
        "hedge": "I think this configuration might improve performance, but I'm drawing on limited evidence."
    },
    {
        "domain": "epistemic",
        "anchor": "The correlation between these features is causal, not merely statistical.",
        "paraphrase": "There is a genuine causal link between these features, beyond simple correlation.",
        "contradiction": "The apparent relationship between these features is purely correlational with no causal basis.",
        "hedge": "The features appear correlated and might be causally linked, but I can't rule out confounding without further analysis."
    },
    {
        "domain": "epistemic",
        "anchor": "Based on my analysis, the optimal position size is 2% of capital.",
        "paraphrase": "My calculations indicate that allocating 2% of capital per position is ideal.",
        "contradiction": "A 2% position size is far too conservative; 5% would be optimal given current conditions.",
        "hedge": "A position size around 2% seems reasonable, though the Kelly criterion calculation depends on parameters I'm estimating with uncertainty."
    },
    {
        "domain": "epistemic",
        "anchor": "This approach has been validated across all relevant market regimes.",
        "paraphrase": "Testing confirms this method works in every significant type of market environment.",
        "contradiction": "This approach has only been tested in bull markets and may fail in bear or sideways regimes.",
        "hedge": "This approach has been tested in several regimes, though I'm not fully confident it generalizes to all possible market conditions."
    },
    {
        "domain": "epistemic",
        "anchor": "The model's prediction accuracy is 87% on the test set.",
        "paraphrase": "On held-out test data, the model achieves an accuracy of 87%.",
        "contradiction": "The model's true accuracy is closer to 72% when evaluated on properly out-of-sample data.",
        "hedge": "The test set accuracy appears to be around 87%, though this estimate could be optimistic if there's subtle data leakage."
    },
    # ─── TRIPLETAS ADICIONALES POR DOMINIO PARA ROBUSTEZ ──────────────────
    {
        "domain": "trading",
        "anchor": "The toxicity score for this pair indicates unsafe trading conditions.",
        "paraphrase": "Current toxicity metrics flag this trading pair as too risky to trade.",
        "contradiction": "The toxicity score is within normal bounds and conditions are safe for trading.",
        "hedge": "The toxicity score looks elevated, though the orderbook data feeding it might have a lag that makes me less sure."
    },
    {
        "domain": "software",
        "anchor": "The NSSM service configuration is correct and the service will auto-restart on failure.",
        "paraphrase": "NSSM has been properly configured to automatically recover the service after any crash.",
        "contradiction": "The NSSM configuration has a critical error that will prevent service auto-recovery.",
        "hedge": "The NSSM setup looks correct for auto-restart, but I haven't tested every edge case failure mode."
    },
    {
        "domain": "ml_config",
        "anchor": "The dollar bar transformation eliminates the non-stationarity in the price series.",
        "paraphrase": "Converting to dollar bars successfully removes the time-varying properties of raw price data.",
        "contradiction": "Dollar bars still exhibit significant non-stationarity and require additional transforms.",
        "hedge": "Dollar bars should reduce non-stationarity, though some residual time-dependence might remain."
    },
    {
        "domain": "technical",
        "anchor": "The KL divergence between the approximate and true posterior is bounded by the ELBO gap.",
        "paraphrase": "The ELBO gap provides an upper bound on how far the variational approximation is from the true posterior.",
        "contradiction": "The ELBO gap is unrelated to the KL divergence between approximate and true posteriors.",
        "hedge": "The ELBO gap should bound the KL divergence, though in practice the bound can be very loose for complex posteriors."
    },
    {
        "domain": "epistemic",
        "anchor": "I have high confidence in this recommendation because it is supported by extensive empirical evidence.",
        "paraphrase": "Strong empirical backing gives me great certainty about this recommendation.",
        "contradiction": "Despite appearing well-supported, the empirical evidence is cherry-picked and unreliable.",
        "hedge": "I'm fairly confident in this recommendation, though the empirical evidence, while extensive, comes from a limited set of conditions."
    },
]

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: DATACLASS PARA RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AxiomResult:
    """Resultado de verificación de un axioma individual."""
    axiom: str
    name: str
    passed: bool
    metric_value: float
    threshold: float
    margin: float  # ratio o diferencia respecto al umbral
    p_value: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    details: str = ""

@dataclass 
class PhiValidationReport:
    """Reporte completo de admisibilidad de φ."""
    timestamp: str = ""
    model_name: str = ""
    corpus_size: int = 0
    domains: list = field(default_factory=list)
    axiom_results: list = field(default_factory=list)
    admissibility: str = ""  # ADMISIBLE / NO ADMISIBLE / PARCIALMENTE ADMISIBLE
    recommendation: str = ""
    epsilon_lex: float = 0.0
    epsilon_prop: float = 0.0
    epsilon_mod: float = 0.0
    lipschitz_estimate: float = 0.0
    n_bootstrap: int = 0

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: IMPLEMENTACIÓN DE φ
# ═══════════════════════════════════════════════════════════════════════════

class PhiProjection:
    """
    Implementación de φ = [φ_content; φ_modal]
    
    En esta versión v1 (pre fine-tuning de φ_modal):
      - φ_content: sentence-transformer pre-entrenado
      - φ_modal: NO fine-tuned aún — usamos el mismo encoder
        para establecer la BASELINE de qué tan lejos estamos
        de satisfacer los axiomas con embeddings estándar.
    
    Esto nos permite medir:
      1. ¿Los axiomas A1 y A2 se cumplen con embeddings vanilla?
         (Probablemente sí — los encoders modernos separan contenido)
      2. ¿El axioma A3 se cumple? 
         (Probablemente NO — los encoders no separan modalidad epistémica)
      3. ¿Cuánto gap hay? 
         (Esto define el trabajo de fine-tuning necesario)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"  Cargando modelo: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"  Dimensión de embeddings: {self.dim}")
    
    def encode(self, texts: list[str]) -> np.ndarray:
        """Codifica textos a espacio Z."""
        return self.model.encode(texts, normalize_embeddings=True, 
                                  show_progress_bar=False)
    
    def distance(self, z1: np.ndarray, z2: np.ndarray) -> float:
        """Distancia coseno en Z (rango [0, 2], 0=idéntico)."""
        return float(cosine_dist(z1, z2))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: VERIFICACIÓN DE AXIOMAS
# ═══════════════════════════════════════════════════════════════════════════

def verify_axiom_a1(phi: PhiProjection, corpus: list[dict], 
                     n_bootstrap: int = 5000) -> AxiomResult:
    """
    AXIOMA A1 — Invariancia Léxica
    Si y₁ ≡_sem y₂ (paráfrasis), entonces d_Z(φ(y₁), φ(y₂)) ≤ ε_lex
    ε_lex definido como percentil 10 de distancias inter-clase (contradicciones)
    """
    # Calcular distancias paráfrasis (intra-clase)
    d_para = []
    for item in corpus:
        z_anchor = phi.encode([item["anchor"]])[0]
        z_para = phi.encode([item["paraphrase"]])[0]
        d_para.append(phi.distance(z_anchor, z_para))
    
    # Calcular distancias contradicción (inter-clase) para calibrar ε_lex
    d_contra = []
    for item in corpus:
        z_anchor = phi.encode([item["anchor"]])[0]
        z_contra = phi.encode([item["contradiction"]])[0]
        d_contra.append(phi.distance(z_anchor, z_contra))
    
    d_para = np.array(d_para)
    d_contra = np.array(d_contra)
    
    # ε_lex = percentil 10 de distancias inter-clase
    epsilon_lex = np.percentile(d_contra, 10)
    
    # Test: ¿media de distancias paráfrasis < ε_lex?
    mean_para = np.mean(d_para)
    pass_rate = np.mean(d_para <= epsilon_lex)
    
    # Bootstrap CI para mean_para
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(d_para, size=len(d_para), replace=True)
        boot_means.append(np.mean(sample))
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    
    passed = mean_para < epsilon_lex
    
    return AxiomResult(
        axiom="A1",
        name="Invariancia Léxica (Estabilidad bajo paráfrasis)",
        passed=passed,
        metric_value=mean_para,
        threshold=epsilon_lex,
        margin=epsilon_lex / mean_para if mean_para > 0 else float('inf'),
        ci_low=ci_low,
        ci_high=ci_high,
        details=(
            f"Media dist. paráfrasis: {mean_para:.4f} (CI95: [{ci_low:.4f}, {ci_high:.4f}])\n"
            f"    ε_lex (p10 inter-clase): {epsilon_lex:.4f}\n"
            f"    Tasa de cumplimiento individual: {pass_rate:.1%}\n"
            f"    Margen (ε_lex/media): {epsilon_lex/mean_para:.2f}x"
        )
    )


def verify_axiom_a2(phi: PhiProjection, corpus: list[dict],
                     epsilon_lex: float, gamma_min: float = 3.0,
                     n_bootstrap: int = 5000) -> AxiomResult:
    """
    AXIOMA A2 — Sensibilidad Proposicional
    Si y₁ ≢_sem y₂ (contradicción), entonces d_Z(φ(y₁), φ(y₂)) ≥ ε_prop
    Con ε_prop / ε_lex ≥ γ (γ_min = 3.0)
    """
    d_contra = []
    for item in corpus:
        z_anchor = phi.encode([item["anchor"]])[0]
        z_contra = phi.encode([item["contradiction"]])[0]
        d_contra.append(phi.distance(z_anchor, z_contra))
    
    d_contra = np.array(d_contra)
    epsilon_prop = np.mean(d_contra)
    
    gamma = epsilon_prop / epsilon_lex if epsilon_lex > 0 else float('inf')
    
    # Bootstrap CI para gamma
    d_para_all = []
    for item in corpus:
        z_a = phi.encode([item["anchor"]])[0]
        z_p = phi.encode([item["paraphrase"]])[0]
        d_para_all.append(phi.distance(z_a, z_p))
    d_para_all = np.array(d_para_all)
    
    boot_gammas = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(corpus), size=len(corpus), replace=True)
        boot_contra = d_contra[idx]
        boot_para = d_para_all[idx]
        boot_eps_lex = np.percentile(boot_contra, 10)
        boot_eps_prop = np.mean(boot_contra)
        if boot_eps_lex > 0:
            boot_gammas.append(boot_eps_prop / boot_eps_lex)
    ci_low, ci_high = np.percentile(boot_gammas, [2.5, 97.5])
    
    passed = gamma >= gamma_min
    
    return AxiomResult(
        axiom="A2",
        name="Sensibilidad Proposicional (Discriminación semántica)",
        passed=passed,
        metric_value=gamma,
        threshold=gamma_min,
        margin=gamma / gamma_min,
        ci_low=ci_low,
        ci_high=ci_high,
        details=(
            f"ε_prop (media dist. contradicción): {epsilon_prop:.4f}\n"
            f"    ε_lex: {epsilon_lex:.4f}\n"
            f"    γ = ε_prop / ε_lex: {gamma:.2f} (CI95: [{ci_low:.2f}, {ci_high:.2f}])\n"
            f"    Umbral mínimo γ_min: {gamma_min:.1f}\n"
            f"    Margen: {gamma/gamma_min:.2f}x"
        )
    )


def verify_axiom_a3(phi: PhiProjection, corpus: list[dict],
                     epsilon_lex: float, modal_ratio_min: float = 2.0,
                     n_bootstrap: int = 5000) -> AxiomResult:
    """
    AXIOMA A3 — Sensibilidad Modal (Preservación de certeza epistémica)
    Si y₁ expresa certeza e y₂ expresa incertidumbre (hedge), entonces:
    d_Z(φ(y₁), φ(y₂)) ≥ ε_mod > ε_lex
    Con ε_mod / ε_lex ≥ 2.0
    
    NOTA: Este es el axioma más difícil. Embeddings estándar NO fueron
    entrenados para capturar modalidad epistémica.
    """
    d_hedge = []
    for item in corpus:
        z_anchor = phi.encode([item["anchor"]])[0]
        z_hedge = phi.encode([item["hedge"]])[0]
        d_hedge.append(phi.distance(z_anchor, z_hedge))
    
    d_hedge = np.array(d_hedge)
    epsilon_mod = np.mean(d_hedge)
    
    modal_ratio = epsilon_mod / epsilon_lex if epsilon_lex > 0 else float('inf')
    
    # Bootstrap CI
    boot_ratios = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(d_hedge), size=len(d_hedge), replace=True)
        boot_mod = np.mean(d_hedge[idx])
        if epsilon_lex > 0:
            boot_ratios.append(boot_mod / epsilon_lex)
    ci_low, ci_high = np.percentile(boot_ratios, [2.5, 97.5])
    
    passed = modal_ratio >= modal_ratio_min
    
    return AxiomResult(
        axiom="A3",
        name="Sensibilidad Modal (Preservación de certeza epistémica)",
        passed=passed,
        metric_value=modal_ratio,
        threshold=modal_ratio_min,
        margin=modal_ratio / modal_ratio_min,
        ci_low=ci_low,
        ci_high=ci_high,
        details=(
            f"ε_mod (media dist. anchor↔hedge): {epsilon_mod:.4f}\n"
            f"    ε_lex: {epsilon_lex:.4f}\n"
            f"    Ratio ε_mod / ε_lex: {modal_ratio:.2f} (CI95: [{ci_low:.2f}, {ci_high:.2f}])\n"
            f"    Umbral mínimo: {modal_ratio_min:.1f}\n"
            f"    {'⚠ ESPERADO: embeddings estándar probablemente fallan aquí' if not passed else '✓ Sorprendentemente, el encoder captura modalidad'}"
        )
    )


def verify_axiom_a4(phi: PhiProjection, corpus: list[dict],
                     n_samples: int = 500,
                     n_bootstrap: int = 5000) -> AxiomResult:
    """
    AXIOMA A4 — Continuidad Lipschitz
    ∃ L > 0 tal que d_Z(φ(y₁), φ(y₂)) ≤ L · d_Y(y₁, y₂)
    
    Estimamos L empíricamente como max ratio d_Z/d_Y sobre pares muestreados.
    d_Y = distancia de edición normalizada (Levenshtein / max(len)).
    """
    
    def normalized_edit_distance(s1: str, s2: str) -> float:
        """Distancia de edición normalizada (Levenshtein)."""
        len1, len2 = len(s1), len(s2)
        if len1 == 0 and len2 == 0:
            return 0.0
        
        # Optimización: si son iguales
        if s1 == s2:
            return 0.0
        
        # DP simplificado
        if len1 > len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1
        
        prev_row = list(range(len2 + 1))
        for i in range(1, len1 + 1):
            curr_row = [i] + [0] * len2
            for j in range(1, len2 + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                curr_row[j] = min(
                    curr_row[j-1] + 1,
                    prev_row[j] + 1,
                    prev_row[j-1] + cost
                )
            prev_row = curr_row
        
        return prev_row[len2] / max(len1, len2)
    
    # Recolectar todos los textos
    all_texts = []
    for item in corpus:
        for key in ["anchor", "paraphrase", "contradiction", "hedge"]:
            all_texts.append(item[key])
    
    # Muestrear pares y calcular ratios
    ratios = []
    n_pairs = min(n_samples, len(all_texts) * (len(all_texts) - 1) // 2)
    
    # Precomputar embeddings
    all_embeddings = phi.encode(all_texts)
    
    indices = np.random.choice(len(all_texts), size=(n_pairs, 2), replace=True)
    for i, j in indices:
        if i == j:
            continue
        
        d_z = phi.distance(all_embeddings[i], all_embeddings[j])
        d_y = normalized_edit_distance(all_texts[i], all_texts[j])
        
        if d_y > 0.01:  # evitar divisiones por ~0
            ratios.append(d_z / d_y)
    
    ratios = np.array(ratios)
    
    # Estimación de L como percentil 95 (robusto a outliers)
    L_estimate = np.percentile(ratios, 95)
    L_max = np.max(ratios)
    L_mean = np.mean(ratios)
    
    # Bootstrap CI para L (p95)
    boot_L = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(ratios, size=len(ratios), replace=True)
        boot_L.append(np.percentile(sample, 95))
    ci_low, ci_high = np.percentile(boot_L, [2.5, 97.5])
    
    # Un L finito y razonable (< 10) indica continuidad
    # L > 10 sugiere que φ amplifica ruido léxico excesivamente
    L_threshold = 10.0
    passed = L_estimate < L_threshold
    
    return AxiomResult(
        axiom="A4",
        name="Continuidad Lipschitz",
        passed=passed,
        metric_value=L_estimate,
        threshold=L_threshold,
        margin=L_threshold / L_estimate if L_estimate > 0 else float('inf'),
        ci_low=ci_low,
        ci_high=ci_high,
        details=(
            f"L estimado (p95 de d_Z/d_Y): {L_estimate:.4f} (CI95: [{ci_low:.4f}, {ci_high:.4f}])\n"
            f"    L máximo observado: {L_max:.4f}\n"
            f"    L medio: {L_mean:.4f}\n"
            f"    Umbral: L < {L_threshold:.1f}\n"
            f"    Pares evaluados: {len(ratios)}"
        )
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: ANÁLISIS POR DOMINIO
# ═══════════════════════════════════════════════════════════════════════════

def domain_analysis(phi: PhiProjection, corpus: list[dict]) -> dict:
    """Calcula métricas por dominio para diagnóstico granular."""
    domains = {}
    for item in corpus:
        d = item["domain"]
        if d not in domains:
            domains[d] = {"d_para": [], "d_contra": [], "d_hedge": []}
        
        z_a = phi.encode([item["anchor"]])[0]
        z_p = phi.encode([item["paraphrase"]])[0]
        z_c = phi.encode([item["contradiction"]])[0]
        z_h = phi.encode([item["hedge"]])[0]
        
        domains[d]["d_para"].append(phi.distance(z_a, z_p))
        domains[d]["d_contra"].append(phi.distance(z_a, z_c))
        domains[d]["d_hedge"].append(phi.distance(z_a, z_h))
    
    results = {}
    for d, vals in domains.items():
        results[d] = {
            "n_triplets": len(vals["d_para"]),
            "mean_d_paraphrase": float(np.mean(vals["d_para"])),
            "mean_d_contradiction": float(np.mean(vals["d_contra"])),
            "mean_d_hedge": float(np.mean(vals["d_hedge"])),
            "separation_ratio_prop": float(np.mean(vals["d_contra"]) / np.mean(vals["d_para"])) if np.mean(vals["d_para"]) > 0 else float('inf'),
            "separation_ratio_modal": float(np.mean(vals["d_hedge"]) / np.mean(vals["d_para"])) if np.mean(vals["d_para"]) > 0 else float('inf'),
        }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: PROTOCOLO COMPLETO
# ═══════════════════════════════════════════════════════════════════════════

def run_validation_protocol(model_name: str = "all-MiniLM-L6-v2",
                             n_bootstrap: int = 5000) -> PhiValidationReport:
    """
    Ejecuta el protocolo completo de validación de φ.
    
    Returns:
        PhiValidationReport con todos los resultados y recomendación.
    """
    
    print("=" * 72)
    print("  PROYECTO ECHO — PROTOCOLO DE VALIDACIÓN DE φ")
    print("  Referencia: Marco Matemático v2.0, §2.1.1")
    print("=" * 72)
    print()
    
    report = PhiValidationReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        model_name=model_name,
        corpus_size=len(VALIDATION_CORPUS),
        domains=list(set(item["domain"] for item in VALIDATION_CORPUS)),
        n_bootstrap=n_bootstrap
    )
    
    # ─── Paso 1: Cargar modelo ──────────────────────────────────────────
    print("[1/6] Cargando modelo φ_content...")
    phi = PhiProjection(model_name)
    print()
    
    # ─── Paso 2: Verificar A1 ───────────────────────────────────────────
    print("[2/6] Verificando Axioma A1 (Invariancia Léxica)...")
    a1 = verify_axiom_a1(phi, VALIDATION_CORPUS, n_bootstrap)
    report.axiom_results.append(a1)
    report.epsilon_lex = a1.threshold  # ε_lex calibrado
    print(f"  {'✓ PASS' if a1.passed else '✗ FAIL'}")
    print(f"    {a1.details}")
    print()
    
    # ─── Paso 3: Verificar A2 ───────────────────────────────────────────
    print("[3/6] Verificando Axioma A2 (Sensibilidad Proposicional)...")
    a2 = verify_axiom_a2(phi, VALIDATION_CORPUS, report.epsilon_lex, 
                          gamma_min=3.0, n_bootstrap=n_bootstrap)
    report.axiom_results.append(a2)
    report.epsilon_prop = a2.metric_value * report.epsilon_lex  # reconstruir ε_prop
    print(f"  {'✓ PASS' if a2.passed else '✗ FAIL'}")
    print(f"    {a2.details}")
    print()
    
    # ─── Paso 4: Verificar A3 ───────────────────────────────────────────
    print("[4/6] Verificando Axioma A3 (Sensibilidad Modal)...")
    a3 = verify_axiom_a3(phi, VALIDATION_CORPUS, report.epsilon_lex,
                          modal_ratio_min=2.0, n_bootstrap=n_bootstrap)
    report.axiom_results.append(a3)
    report.epsilon_mod = a3.metric_value * report.epsilon_lex
    print(f"  {'✓ PASS' if a3.passed else '✗ FAIL'}")
    print(f"    {a3.details}")
    print()
    
    # ─── Paso 5: Verificar A4 ───────────────────────────────────────────
    print("[5/6] Verificando Axioma A4 (Continuidad Lipschitz)...")
    a4 = verify_axiom_a4(phi, VALIDATION_CORPUS, n_bootstrap=n_bootstrap)
    report.axiom_results.append(a4)
    report.lipschitz_estimate = a4.metric_value
    print(f"  {'✓ PASS' if a4.passed else '✗ FAIL'}")
    print(f"    {a4.details}")
    print()
    
    # ─── Paso 6: Análisis por dominio ───────────────────────────────────
    print("[6/6] Análisis por dominio...")
    domain_results = domain_analysis(phi, VALIDATION_CORPUS)
    for d, metrics in domain_results.items():
        print(f"  {d:15s} | para={metrics['mean_d_paraphrase']:.4f} "
              f"contra={metrics['mean_d_contradiction']:.4f} "
              f"hedge={metrics['mean_d_hedge']:.4f} "
              f"| γ_prop={metrics['separation_ratio_prop']:.2f} "
              f"γ_mod={metrics['separation_ratio_modal']:.2f}")
    print()
    
    # ─── Determinación de admisibilidad ─────────────────────────────────
    n_passed = sum(1 for r in report.axiom_results if r.passed)
    
    if n_passed == 4:
        report.admissibility = "ADMISIBLE"
        report.recommendation = (
            "φ satisface los 4 axiomas. Proceder con H1 (descomposición "
            "de incertidumbre) usando este encoder como φ_content."
        )
    elif n_passed >= 2 and a1.passed and a2.passed:
        report.admissibility = "PARCIALMENTE ADMISIBLE"
        failed = [r.axiom for r in report.axiom_results if not r.passed]
        report.recommendation = (
            f"φ satisface A1+A2 (contenido semántico) pero falla en {failed}. "
            f"{'A3 (modalidad epistémica) requiere fine-tuning contrastivo de φ_modal. ' if not a3.passed else ''}"
            f"{'A4 (Lipschitz) indica amplificación de ruido léxico. ' if not a4.passed else ''}"
            f"ACCIÓN: Implementar φ_modal con entrenamiento contrastivo sobre "
            f"pares (certeza, hedge) antes de proceder con H1."
        )
    else:
        report.admissibility = "NO ADMISIBLE"
        report.recommendation = (
            "φ no satisface axiomas fundamentales de contenido (A1/A2). "
            "Considerar un encoder diferente o verificar que el corpus "
            "de validación es correcto. NO proceder con H1."
        )
    
    # ─── Reporte final ──────────────────────────────────────────────────
    print("=" * 72)
    print(f"  RESULTADO: {report.admissibility}")
    print(f"  Axiomas: {n_passed}/4 cumplidos")
    print(f"  {report.recommendation}")
    print("=" * 72)
    
    return report, domain_results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ECHO — Protocolo de Validación de φ (Proyección Semántica)"
    )
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Nombre del modelo sentence-transformer")
    parser.add_argument("--n-bootstrap", type=int, default=5000,
                        help="Número de iteraciones bootstrap")
    parser.add_argument("--output", type=str, default=None,
                        help="Ruta para guardar reporte JSON")
    
    args = parser.parse_args()
    
    np.random.seed(42)
    
    report, domain_results = run_validation_protocol(
        model_name=args.model,
        n_bootstrap=args.n_bootstrap
    )
    
    # Guardar reporte JSON
    output_path = args.output or "phi_validation_report.json"
    report_dict = asdict(report)
    report_dict["domain_analysis"] = domain_results
    report_dict["axiom_results"] = [asdict(r) for r in report.axiom_results]
    
    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)
    
    print(f"\nReporte guardado en: {output_path}")
