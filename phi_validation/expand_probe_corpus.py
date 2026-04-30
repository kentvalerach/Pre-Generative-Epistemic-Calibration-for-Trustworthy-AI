"""
═══════════════════════════════════════════════════════════════════════════════
PROJECT ECHO — Sprint 2A (Corrección): Expansión de Corpus para Probe
═══════════════════════════════════════════════════════════════════════════════

PROBLEMA:
  N=56 insuficiente para probe generalizable. OOD r=-0.221 (FAIL).
  Ningún probe generaliza con 56 muestras en 5 dominios.

SOLUCIÓN:
  Expandir a 200+ muestras mediante 4 estrategias de augmentación
  semántica controlada (Kent Valera, abril 2026):
  
  1. Paráfrasis controladas: preservar significado, variar sintaxis
  2. Inyección de hedges: agregar/remover marcadores epistémicos
  3. Negación condicional: "X es Y" vs "No hay base para afirmar Y"
  4. Variación de longitud: 15-80 tokens por muestra

  Etiquetado: semi-automático con ep_score ∈ [0, 1]
  Distribución objetivo: uniforme sobre [0, 1] × 5 dominios

USO:
  python expand_probe_corpus.py
  → Genera C:\\ECHO\\data\\probe_corpus_expanded.json
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
import time
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = Path(__file__).parent / "reports"

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# BASE CORPUS: 40 seed texts × 5 domains × 5 certainty levels
# Each seed has ep_score and will be augmented into ~5 variants
# ═══════════════════════════════════════════════════════════════════════════

SEED_CORPUS = [
    # ═══ TRADING (8 seeds) ════════════════════════════════════════════════
    # Level 0.0-0.2: Factual
    {"text": "The trailing stop follows price movement in one direction only.", "ep": 0.06, "domain": "trading"},
    {"text": "Bitcoin's block reward halves approximately every four years.", "ep": 0.08, "domain": "trading"},
    # Level 0.2-0.4: Confident claims
    {"text": "The trailing stop should be set at 1.5% for this regime.", "ep": 0.25, "domain": "trading"},
    {"text": "The funding rate indicates a short squeeze is imminent.", "ep": 0.35, "domain": "trading"},
    # Level 0.4-0.6: Moderate uncertainty
    {"text": "The toxicity score looks elevated, suggesting caution.", "ep": 0.50, "domain": "trading"},
    {"text": "The current market seems to be in a high-volatility regime.", "ep": 0.48, "domain": "trading"},
    # Level 0.6-0.8: Significant uncertainty
    {"text": "The regime classification is ambiguous right now.", "ep": 0.70, "domain": "trading"},
    {"text": "The backtest results are encouraging, but overfitting is a real concern.", "ep": 0.68, "domain": "trading"},
    # Level 0.8-1.0: High uncertainty
    {"text": "The market could go either way from here — it's genuinely unclear.", "ep": 0.90, "domain": "trading"},
    {"text": "I have a hypothesis about SHORT outperforming LONG, but the RSI paradox suggests dynamics I don't fully grasp.", "ep": 0.85, "domain": "trading"},

    # ═══ SOFTWARE (8 seeds) ═══════════════════════════════════════════════
    {"text": "WebSocket provides full-duplex communication over TCP.", "ep": 0.05, "domain": "software"},
    {"text": "NSSM is a Windows service manager for running executables as services.", "ep": 0.05, "domain": "software"},
    {"text": "The bug is in the connection pooling logic.", "ep": 0.28, "domain": "software"},
    {"text": "This index will reduce query time from 12 seconds to under 500ms.", "ep": 0.25, "domain": "software"},
    {"text": "The heartbeat interval might be causing the WebSocket disconnections.", "ep": 0.52, "domain": "software"},
    {"text": "The race condition is probably caused by unsynchronized shared state.", "ep": 0.48, "domain": "software"},
    {"text": "The memory leak could be from unclosed file handles, but there might be other factors.", "ep": 0.65, "domain": "software"},
    {"text": "ALTER TABLE should be safe, though on very large tables there could be locking issues.", "ep": 0.60, "domain": "software"},
    {"text": "The cron job failure could be many things — stderr, permissions, or something else entirely.", "ep": 0.82, "domain": "software"},
    {"text": "I'm not sure what's causing the silent failure in the pipeline.", "ep": 0.88, "domain": "software"},

    # ═══ ML_CONFIG (8 seeds) ══════════════════════════════════════════════
    {"text": "The learning rate controls the step size during gradient descent.", "ep": 0.05, "domain": "ml_config"},
    {"text": "Dollar bars sample by notional volume rather than time.", "ep": 0.08, "domain": "ml_config"},
    {"text": "A learning rate of 3e-4 will accelerate convergence without overfitting.", "ep": 0.30, "domain": "ml_config"},
    {"text": "Removing cyclic features eliminated the spurious correlation.", "ep": 0.30, "domain": "ml_config"},
    {"text": "The model is likely overfitting due to insufficient regime diversity.", "ep": 0.45, "domain": "ml_config"},
    {"text": "Increasing batch size should improve training stability.", "ep": 0.42, "domain": "ml_config"},
    {"text": "Purged k-fold should eliminate most bias, though subtle leakage might remain.", "ep": 0.62, "domain": "ml_config"},
    {"text": "Fractional differentiation should balance memory and stationarity, though the optimal d requires search.", "ep": 0.65, "domain": "ml_config"},
    {"text": "The ensemble configuration might be suboptimal — I'm not sure which base learners work best here.", "ep": 0.80, "domain": "ml_config"},
    {"text": "I can't tell if the feature importance is stable or if it's an artifact of the cross-validation scheme.", "ep": 0.85, "domain": "ml_config"},

    # ═══ TECHNICAL (8 seeds) ══════════════════════════════════════════════
    {"text": "The softmax function converts logits to probability distributions.", "ep": 0.05, "domain": "technical"},
    {"text": "The KL divergence is always non-negative.", "ep": 0.03, "domain": "technical"},
    {"text": "Gradient clipping prevents exploding gradients in deep recurrent networks.", "ep": 0.25, "domain": "technical"},
    {"text": "The ELBO gap equals the KL divergence to the true posterior.", "ep": 0.20, "domain": "technical"},
    {"text": "Layer normalization appears to stabilize training in most architectures.", "ep": 0.40, "domain": "technical"},
    {"text": "Sparse attention can likely reduce complexity to near-linear.", "ep": 0.45, "domain": "technical"},
    {"text": "The Fisher information matrix provides a reasonable approximation, though accuracy degrades far from the optimum.", "ep": 0.60, "domain": "technical"},
    {"text": "MoE routing seems to develop some specialization, though the degree is debated.", "ep": 0.65, "domain": "technical"},
    {"text": "The information bottleneck theory offers one explanation for generalization, but its applicability is disputed.", "ep": 0.78, "domain": "technical"},
    {"text": "The Lottery Ticket Hypothesis might apply here, though it's been shown to break down in some settings.", "ep": 0.75, "domain": "technical"},

    # ═══ EPISTEMIC (8 seeds) ══════════════════════════════════════════════
    {"text": "I am certain this configuration will improve performance.", "ep": 0.10, "domain": "epistemic"},
    {"text": "The data clearly supports our hypothesis.", "ep": 0.12, "domain": "epistemic"},
    {"text": "Based on my analysis, the optimal position size is 2% of capital.", "ep": 0.30, "domain": "epistemic"},
    {"text": "The correlation between these features is causal, not merely statistical.", "ep": 0.35, "domain": "epistemic"},
    {"text": "I'm fairly confident in this recommendation, though the evidence is limited.", "ep": 0.55, "domain": "epistemic"},
    {"text": "The experiment provides useful evidence, though additional experiments would help.", "ep": 0.50, "domain": "epistemic"},
    {"text": "This approach has been tested in several regimes, though I'm not confident it generalizes.", "ep": 0.72, "domain": "epistemic"},
    {"text": "The data leans toward rejecting the null, though the p-value is close and the sample is small.", "ep": 0.78, "domain": "epistemic"},
    {"text": "I think this might work, but I'm not entirely sure about the outcome.", "ep": 0.85, "domain": "epistemic"},
    {"text": "I'm not confident in this prediction because the evidence is mixed.", "ep": 0.88, "domain": "epistemic"},
]


# ═══════════════════════════════════════════════════════════════════════════
# AUGMENTATION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

# ─── Strategy 1: Paraphrasis controladas ──────────────────────────────────

def paraphrase(text, ep, domain):
    """Genera paráfrasis preservando significado y ep_score."""
    import re
    variants = []
    
    # Reordenamiento sintáctico
    if ", " in text:
        parts = text.split(", ", 1)
        if len(parts) == 2 and len(parts[1]) > 10:
            reordered = parts[1].rstrip(".") + ", " + parts[0].lower() + "."
            reordered = reordered[0].upper() + reordered[1:]
            variants.append({"text": reordered, "ep": ep, "domain": domain, "aug": "paraphrase_reorder"})
    
    # Sustitución de sinónimos comunes
    synonyms = [
        (r'\bshould\b', 'ought to'), (r'\bwill\b', 'is going to'),
        (r'\bis\b', "represents", 0.3), (r'\bcomputes\b', 'calculates'),
        (r'\bprovides\b', 'offers'), (r'\bcontrols\b', 'governs'),
        (r'\breduces\b', 'decreases'), (r'\bimproves\b', 'enhances'),
        (r'\bcauses\b', 'leads to'), (r'\bprevents\b', 'avoids'),
        (r'\bindicates\b', 'signals'), (r'\bsuggests\b', 'implies'),
    ]
    
    for syn in synonyms:
        pattern, replacement = syn[0], syn[1]
        if re.search(pattern, text):
            new_text = re.sub(pattern, replacement, text, count=1)
            if new_text != text:
                variants.append({"text": new_text, "ep": ep, "domain": domain, "aug": "paraphrase_synonym"})
                break  # Solo una sustitución por seed
    
    # Voz activa ↔ pasiva (simplificado)
    if " is " in text and not text.startswith("I "):
        # "X is Y" → "Y characterizes X"
        pass  # Demasiado frágil para regex; skip
    
    return variants[:2]  # Max 2 paráfrasis por seed


# ─── Strategy 2: Inyección/remoción de hedges ────────────────────────────

HEDGE_INJECTORS = [
    ("probably ", 0.15),
    ("possibly ", 0.15),
    ("it seems that ", 0.20),
    ("I believe ", 0.18),
    ("likely ", 0.12),
    ("in my assessment, ", 0.15),
    ("based on available evidence, ", 0.10),
]

HEDGE_REMOVERS = [
    (r'\bprobably\s+', '', -0.15),
    (r'\bpossibly\s+', '', -0.15),
    (r'\blikely\s+', '', -0.12),
    (r'\bI think\s+', '', -0.20),
    (r'\bI believe\s+', '', -0.18),
    (r'\bit seems that\s+', '', -0.20),
    (r'\bseems to\s+', '', -0.15),
    (r'\bmight\b', 'will', -0.20),
    (r'\bcould\b', 'will', -0.18),
    (r',\s*though.*$', '.', -0.15),
    (r',\s*but I.*$', '.', -0.15),
    (r',\s*although.*$', '.', -0.12),
]

def inject_hedge(text, ep, domain):
    """Añade hedge marker → incrementa ep_score."""
    import re
    variants = []
    
    if ep < 0.7:  # Solo inyectar si no es ya muy incierto
        prefix, delta = random.choice(HEDGE_INJECTORS)
        new_text = prefix + text[0].lower() + text[1:]
        new_ep = min(ep + delta, 0.95)
        variants.append({"text": new_text, "ep": round(new_ep, 2), "domain": domain, "aug": "hedge_inject"})
    
    # Sufijo de incertidumbre
    if ep < 0.6:
        suffixes = [
            (", though I'm not entirely certain.", 0.18),
            (", but I could be wrong about this.", 0.20),
            (", although more evidence would strengthen this claim.", 0.15),
            (", but the evidence is still preliminary.", 0.17),
            (", though this depends on factors I haven't fully analyzed.", 0.15),
        ]
        suffix, delta = random.choice(suffixes)
        new_text = text.rstrip(".") + suffix
        new_ep = min(ep + delta, 0.95)
        variants.append({"text": new_text, "ep": round(new_ep, 2), "domain": domain, "aug": "hedge_suffix"})
    
    return variants[:2]


def remove_hedge(text, ep, domain):
    """Remueve hedge markers → decrementa ep_score."""
    import re
    variants = []
    
    if ep > 0.3:  # Solo remover si tiene algo de incertidumbre
        for pattern, replacement, delta in HEDGE_REMOVERS:
            if re.search(pattern, text, re.IGNORECASE):
                new_text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE).strip()
                if new_text and new_text != text and len(new_text) > 15:
                    new_ep = max(ep + delta, 0.03)  # delta es negativo
                    # Limpiar dobles espacios y capitalización
                    new_text = re.sub(r'\s+', ' ', new_text).strip()
                    if not new_text[0].isupper():
                        new_text = new_text[0].upper() + new_text[1:]
                    variants.append({"text": new_text, "ep": round(new_ep, 2), "domain": domain, "aug": "hedge_remove"})
                    break
    
    return variants[:1]


# ─── Strategy 3: Negación condicional ─────────────────────────────────────

def negate_conditional(text, ep, domain):
    """Genera versión con negación condicional → cambia ep según dirección."""
    import re
    variants = []
    
    # "X is Y" → "There is no clear evidence that X is Y"
    if ep < 0.5 and len(text) < 100:
        neg_text = f"There is no clear evidence that {text[0].lower()}{text[1:]}"
        neg_text = neg_text.rstrip(".") + "."
        new_ep = min(ep + 0.35, 0.90)
        variants.append({"text": neg_text, "ep": round(new_ep, 2), "domain": domain, "aug": "negate_uncertain"})
    
    # "I'm not sure about X" → "X is well established"
    if ep > 0.6:
        # Extraer la proposición core y afirmarla
        certainty_prefixes = [
            "It is well established that ",
            "The evidence clearly shows that ",
            "Without question, ",
        ]
        # Simplificación: tomar la parte después de hedge y afirmar
        core = text
        for pattern in [r"^I think\s+", r"^I'm not sure\s+(if|whether|about)\s+",
                       r"^It's hard to say\s+(if|whether)\s+", r"^I'm not confident\s+that\s+",
                       r"^It seems that\s+"]:
            m = re.match(pattern, core, re.IGNORECASE)
            if m:
                core = core[m.end():]
                break
        
        if core != text and len(core) > 15:
            prefix = random.choice(certainty_prefixes)
            affirmed = prefix + core[0].lower() + core[1:]
            new_ep = max(ep - 0.40, 0.05)
            variants.append({"text": affirmed, "ep": round(new_ep, 2), "domain": domain, "aug": "negate_certain"})
    
    return variants[:1]


# ─── Strategy 4: Variación de longitud ────────────────────────────────────

def vary_length(text, ep, domain):
    """Genera versiones más cortas o más largas, preservando ep."""
    variants = []
    tokens = text.split()
    n = len(tokens)
    
    # Versión corta (si el texto es largo)
    if n > 20:
        # Tomar primeros ~60% de tokens y cerrar
        cut = int(n * 0.6)
        short = " ".join(tokens[:cut]).rstrip(",;:—-") + "."
        variants.append({"text": short, "ep": round(ep + 0.02, 2), "domain": domain, "aug": "length_short"})
    
    # Versión larga (agregar contexto)
    if n < 25 and ep < 0.7:
        extensions = [
            f" This applies specifically in the context of {domain} systems.",
            f" The reasoning follows from first principles in {domain}.",
            f" This has been validated in multiple empirical settings.",
        ]
        long_text = text.rstrip(".") + "." + random.choice(extensions)
        variants.append({"text": long_text, "ep": round(ep - 0.02, 2), "domain": domain, "aug": "length_long"})
    
    # Versión larga con incertidumbre (agregar caveats)
    if n < 30 and ep > 0.3:
        caveats = [
            f" However, there are several factors that could change this assessment, including data quality and regime shifts.",
            f" That said, the confidence interval is wide and more data would help narrow it down considerably.",
            f" The underlying assumptions have not been fully tested across all relevant conditions and edge cases.",
        ]
        long_uncertain = text.rstrip(".") + "." + random.choice(caveats)
        new_ep = min(ep + 0.10, 0.92)
        variants.append({"text": long_uncertain, "ep": round(new_ep, 2), "domain": domain, "aug": "length_caveat"})
    
    return variants[:2]


# ═══════════════════════════════════════════════════════════════════════════
# ADDITIONAL HAND-CRAFTED SAMPLES FOR BALANCE
# ═══════════════════════════════════════════════════════════════════════════

ADDITIONAL_SAMPLES = [
    # Extra high-certainty trading
    {"text": "A limit order sits in the order book until filled or cancelled.", "ep": 0.04, "domain": "trading", "aug": "manual"},
    {"text": "Leverage amplifies both gains and losses proportionally.", "ep": 0.05, "domain": "trading", "aug": "manual"},
    {"text": "The bid-ask spread represents the cost of immediate execution.", "ep": 0.05, "domain": "trading", "aug": "manual"},
    
    # Extra mid-certainty across domains
    {"text": "The API should handle this edge case, but I haven't tested it with large payloads.", "ep": 0.55, "domain": "software", "aug": "manual"},
    {"text": "Attention heads in later layers tend to specialize, though the pattern varies by model size.", "ep": 0.52, "domain": "technical", "aug": "manual"},
    {"text": "The validation loss suggests convergence, but the gap with training loss is concerning.", "ep": 0.55, "domain": "ml_config", "aug": "manual"},
    {"text": "My analysis suggests this is the right call, but I'd like a second opinion.", "ep": 0.58, "domain": "epistemic", "aug": "manual"},
    
    # Extra high-uncertainty
    {"text": "I really don't know if this architecture choice will scale to production loads.", "ep": 0.90, "domain": "software", "aug": "manual"},
    {"text": "The relationship between these variables is unclear and may be confounded by regime effects.", "ep": 0.82, "domain": "ml_config", "aug": "manual"},
    {"text": "Whether transformers truly learn compositionality remains an open question without consensus.", "ep": 0.80, "domain": "technical", "aug": "manual"},
    {"text": "The signals are contradictory and I can't form a reliable view on direction.", "ep": 0.92, "domain": "trading", "aug": "manual"},
    {"text": "My uncertainty about this recommendation is high because the precedents are ambiguous.", "ep": 0.87, "domain": "epistemic", "aug": "manual"},
    
    # Extra factual epistemic
    {"text": "This result has been independently replicated across three different laboratories.", "ep": 0.08, "domain": "epistemic", "aug": "manual"},
    {"text": "The mathematical proof is complete and verified by automated theorem checking.", "ep": 0.03, "domain": "epistemic", "aug": "manual"},
    {"text": "I have personally verified this configuration works on our production system.", "ep": 0.12, "domain": "epistemic", "aug": "manual"},
    
    # Extra factual ml_config
    {"text": "Cross-entropy loss measures the divergence between predicted and true distributions.", "ep": 0.05, "domain": "ml_config", "aug": "manual"},
    {"text": "Adam optimizer combines momentum with adaptive learning rates per parameter.", "ep": 0.06, "domain": "ml_config", "aug": "manual"},
    
    # Extra uncertain software
    {"text": "The deployment might fail if the container orchestration doesn't handle the rolling update correctly, but I'm not experienced enough with this stack to be sure.", "ep": 0.78, "domain": "software", "aug": "manual"},
    {"text": "I suspect the timeout is too aggressive but changing it could mask other issues.", "ep": 0.68, "domain": "software", "aug": "manual"},
]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 76)
    print("  ECHO — Expansión de Corpus para Probe H3")
    print("  Objetivo: 200+ muestras con augmentación semántica controlada")
    print("=" * 76)
    print()
    
    all_samples = []
    aug_stats = {}
    
    # ─── Seeds originales ───
    for seed in SEED_CORPUS:
        all_samples.append({
            "text": seed["text"],
            "ep": seed["ep"],
            "domain": seed["domain"],
            "aug": "original",
        })
    
    n_seeds = len(all_samples)
    print(f"  Seeds originales: {n_seeds}")
    
    # ─── Augmentación ───
    print("  Aplicando 4 estrategias de augmentación...")
    
    for seed in SEED_CORPUS:
        text, ep, domain = seed["text"], seed["ep"], seed["domain"]
        
        # Strategy 1: Paráfrasis
        for v in paraphrase(text, ep, domain):
            all_samples.append(v)
        
        # Strategy 2a: Inyección de hedges
        for v in inject_hedge(text, ep, domain):
            all_samples.append(v)
        
        # Strategy 2b: Remoción de hedges
        for v in remove_hedge(text, ep, domain):
            all_samples.append(v)
        
        # Strategy 3: Negación condicional
        for v in negate_conditional(text, ep, domain):
            all_samples.append(v)
        
        # Strategy 4: Variación de longitud
        for v in vary_length(text, ep, domain):
            all_samples.append(v)
    
    n_augmented = len(all_samples) - n_seeds
    print(f"  Muestras augmentadas: {n_augmented}")
    
    # ─── Adicionales manuales ───
    all_samples.extend(ADDITIONAL_SAMPLES)
    n_manual = len(ADDITIONAL_SAMPLES)
    print(f"  Muestras manuales adicionales: {n_manual}")
    
    # ─── Deduplicación ───
    seen = set()
    unique = []
    for s in all_samples:
        key = s["text"][:80].lower().strip()
        if key not in seen and len(s["text"]) >= 15:
            seen.add(key)
            unique.append(s)
    
    all_samples = unique
    print(f"  Después de dedup: {len(all_samples)}")
    print()
    
    # ─── Estadísticas ───
    print("━" * 76)
    print("  ESTADÍSTICAS DEL CORPUS EXPANDIDO")
    print("━" * 76)
    print()
    
    # Por dominio
    domain_counts = {}
    for s in all_samples:
        d = s["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
    
    print(f"  {'Dominio':15s} {'N':>6s} {'%':>7s}")
    print(f"  {'─'*15} {'─'*6} {'─'*7}")
    for d in sorted(domain_counts.keys()):
        n = domain_counts[d]
        print(f"  {d:15s} {n:6d} {n/len(all_samples)*100:6.1f}%")
    print(f"  {'─'*15} {'─'*6} {'─'*7}")
    print(f"  {'TOTAL':15s} {len(all_samples):6d}")
    print()
    
    # Por tipo de augmentación
    aug_counts = {}
    for s in all_samples:
        a = s.get("aug", "original")
        aug_counts[a] = aug_counts.get(a, 0) + 1
    
    print(f"  {'Augmentación':25s} {'N':>6s}")
    print(f"  {'─'*25} {'─'*6}")
    for a in sorted(aug_counts.keys()):
        print(f"  {a:25s} {aug_counts[a]:6d}")
    print()
    
    # Distribución de ep_scores
    ep_vals = np.array([s["ep"] for s in all_samples])
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    
    print(f"  Distribución de ep_score:")
    print(f"  {'Rango':12s} {'N':>6s} {'%':>7s} {'Histograma':>20s}")
    print(f"  {'─'*12} {'─'*6} {'─'*7} {'─'*20}")
    for lo, hi in bins:
        mask = (ep_vals >= lo) & (ep_vals < hi)
        n = np.sum(mask)
        pct = n / len(all_samples) * 100
        bar = "█" * int(pct / 2)
        print(f"  [{lo:.1f}, {hi:.1f})  {n:6d} {pct:6.1f}% {bar}")
    print()
    
    # Estadísticas de longitud
    lengths = [len(s["text"].split()) for s in all_samples]
    print(f"  Longitud (tokens): min={min(lengths)}, max={max(lengths)}, "
          f"media={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")
    print()
    
    # ─── Guardar ───
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = DATA_DIR / "probe_corpus_expanded.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"  Corpus guardado: {output_path}")
    print(f"  Tamaño: {os.path.getsize(output_path) / 1024:.1f} KB")
    print()
    
    # ─── Muestras ───
    print("  ─── Muestras (10 aleatorias) ───")
    print()
    samples = random.sample(all_samples, min(10, len(all_samples)))
    for i, s in enumerate(samples):
        short = s["text"][:70] + "..." if len(s["text"]) > 73 else s["text"]
        print(f"  [{i+1}] ep={s['ep']:.2f} | {s['domain']:10s} | {s.get('aug','?'):20s}")
        print(f"       {short}")
        print()
    
    # ─── Reporte ───
    report = {
        "project": "ECHO",
        "module": "corpus_expansion_probe",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_samples": len(all_samples),
        "n_seeds": n_seeds,
        "n_augmented": n_augmented,
        "n_manual": n_manual,
        "domain_distribution": domain_counts,
        "augmentation_distribution": aug_counts,
        "ep_score_stats": {
            "mean": float(np.mean(ep_vals)),
            "std": float(np.std(ep_vals)),
            "min": float(np.min(ep_vals)),
            "max": float(np.max(ep_vals)),
        },
        "length_stats": {
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "min": int(min(lengths)),
            "max": int(max(lengths)),
        },
        "output_file": str(output_path),
    }
    
    report_path = REPORTS_DIR / "corpus_expansion_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Reporte: {report_path}")
    print()
    
    # ─── Siguiente paso ───
    print("=" * 76)
    print("  SIGUIENTE: Re-ejecutar Sprint 2A con corpus expandido")
    print("  + capas 12-20 + atención entropía + normas")
    print("  + MLP probe con LODO-CV")
    print("=" * 76)
    print()


if __name__ == "__main__":
    main()