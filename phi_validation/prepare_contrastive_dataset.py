"""
═══════════════════════════════════════════════════════════════════════════════
PROJECT ECHO — Dataset Contrastivo para φ_modal
═══════════════════════════════════════════════════════════════════════════════

Referencia: ECHO Marco Matemático v2.0, §2.1.1

OBJETIVO:
  Construir un dataset de ~2000+ pares contrastivos (certeza ↔ hedge) para
  fine-tuning de φ_modal con objetivo InfoNCE.

FUENTES:
  1. CommitmentBank (de Marneffe et al. 2019)
     - 1,200 discursos con anotaciones de compromiso del hablante
     - Filtro: ModalType = EP (epistémica)
     - GitHub: https://github.com/mcdm/CommitmentBank

  2. MultiNLI / MNLI (Williams et al. 2018)
     - 433K pares entailment/neutral/contradiction
     - Filtro: neutral con hedge markers en hipótesis
     - HuggingFace: nyu-mll/multi_nli

  3. CalibratedMath (Lin et al. 2022)
     - Suite de tareas con confianza calibrada
     - Usado como referencia de calibración, no pares directos
     - GitHub: https://github.com/sylinrl/CalibratedMath

  4. ECHO Synthetic (generación asistida)
     - Pares certeza/hedge generados sobre dominios KAIRI
     - Trading, software, ML config, técnico, epistémico

SALIDA:
  C:\ECHO\data\contrastive_pairs\phi_modal_training.json
  Formato: [{"certain": str, "hedge": str, "source": str, "domain": str}, ...]

USO:
  pip install datasets requests
  python prepare_contrastive_dataset.py

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
import re
import csv
import sys
import time
import random
import io
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "contrastive_pairs"
REPORTS_DIR = Path(__file__).parent / "reports"

HEDGE_MARKERS = [
    "might", "could", "may", "possibly", "perhaps", "maybe",
    "probably", "likely", "unlikely", "seems", "appears",
    "I think", "I believe", "I suspect", "I guess",
    "not sure", "not certain", "uncertain", "unsure",
    "it's possible", "it is possible",
    "hard to say", "difficult to say", "difficult to tell",
    "to some extent", "in some ways", "somewhat",
    "tends to", "tend to",
    "suggest", "suggests", "suggesting",
    "indicate", "indicates",
    "would argue", "one could argue",
    "it depends", "depending on",
    "not necessarily", "not always",
    "roughly", "approximately", "around",
    "if I had to guess",
]

CERTAINTY_MARKERS = [
    "certainly", "definitely", "absolutely", "clearly",
    "obviously", "undoubtedly", "without doubt", "no question",
    "I am certain", "I am sure", "I am confident",
    "it is clear", "it is obvious", "it is evident",
    "must be", "has to be", "will definitely",
    "always", "never", "impossible", "guaranteed",
    "proven", "established", "known fact",
    "beyond doubt", "indisputable", "unquestionable",
]


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE 1: CommitmentBank
# ═══════════════════════════════════════════════════════════════════════════

def download_commitmentbank():
    """Descarga CommitmentBank desde GitHub."""
    import urllib.request
    
    url = "https://raw.githubusercontent.com/mcdm/CommitmentBank/master/CommitmentBank-items.csv"
    
    print("  Descargando CommitmentBank...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ECHO-Project/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        print(f"OK ({len(content)} bytes)")
        return content
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def process_commitmentbank(csv_content):
    """
    Extrae pares certeza/hedge del CommitmentBank.
    
    Estrategia:
    - Items con ModalType = EP (epistémico) contienen verbos modales epistémicos
    - El Target contiene la oración con el verbo de compromiso
    - Construimos par: (versión sin modal = certeza, versión con modal = hedge)
    """
    pairs = []
    
    if csv_content is None:
        print("  CommitmentBank no disponible, usando fallback...")
        return pairs
    
    reader = csv.DictReader(io.StringIO(csv_content))
    
    for row in reader:
        try:
            target = row.get("Target", "").strip()
            context = row.get("Context", "").strip()
            modal_type = row.get("ModalType", "").strip()
            verb = row.get("Verb", "").strip()
            
            if not target or len(target) < 20:
                continue
            
            # Filtrar por modalidad epistémica o items con verbos de compromiso
            is_epistemic = modal_type == "EP"
            has_hedge_verb = verb.lower() in [
                "think", "believe", "suppose", "guess", "imagine",
                "suspect", "assume", "feel", "seem", "appear",
                "suggest", "indicate", "doubt"
            ]
            has_certainty_verb = verb.lower() in [
                "know", "realize", "discover", "find", "prove",
                "confirm", "establish", "demonstrate", "show",
                "understand", "recognize", "acknowledge"
            ]
            
            if not (is_epistemic or has_hedge_verb or has_certainty_verb):
                continue
            
            # Extraer la proposición embebida (el complemento del verbo)
            # Patrón: "Subject VERB that COMPLEMENT"
            that_match = re.search(
                r'(?:that|if|whether)\s+(.+?)(?:\.|$)',
                target, re.IGNORECASE
            )
            
            if that_match:
                complement = that_match.group(1).strip().rstrip(".")
                if len(complement) < 10:
                    continue
                
                # Construir par certeza/hedge
                certain_version = complement[0].upper() + complement[1:] + "."
                
                # Crear versión hedge usando diferentes patrones
                hedge_patterns = [
                    f"It seems that {complement}.",
                    f"It is possible that {complement}.",
                    f"{complement}, though I'm not entirely certain.",
                    f"I think {complement}, but I could be wrong.",
                    f"It might be the case that {complement}.",
                ]
                
                hedge_version = random.choice(hedge_patterns)
                
                pairs.append({
                    "certain": certain_version,
                    "hedge": hedge_version,
                    "source": "commitmentbank",
                    "domain": "general",
                    "modal_type": modal_type,
                    "original_verb": verb,
                })
        except Exception:
            continue
    
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE 2: MultiNLI
# ═══════════════════════════════════════════════════════════════════════════

def load_mnli():
    """Carga MNLI desde HuggingFace datasets."""
    print("  Cargando MNLI desde HuggingFace...", end=" ", flush=True)
    
    try:
        from datasets import load_dataset
        dataset = load_dataset("nyu-mll/multi_nli", split="train")
        print(f"OK ({len(dataset)} examples)")
        return dataset
    except ImportError:
        print("ERROR: instala 'datasets' con: pip install datasets")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def has_hedge_marker(text):
    """Detecta si un texto contiene marcadores de hedging."""
    text_lower = text.lower()
    for marker in HEDGE_MARKERS:
        if marker in text_lower:
            return True
    return False


def has_certainty_marker(text):
    """Detecta si un texto contiene marcadores de certeza."""
    text_lower = text.lower()
    for marker in CERTAINTY_MARKERS:
        if marker in text_lower:
            return True
    return False


def process_mnli(dataset, max_pairs=800):
    """
    Extrae pares certeza/hedge de MNLI.
    
    Estrategia:
    1. Pares "neutral" donde la hipótesis tiene hedge markers
       → (premisa como certeza, hipótesis como hedge)
    2. Pares "entailment" donde premisa es segura e hipótesis tiene hedge
       → (premisa como certeza, hipótesis con hedge como hedge)
    3. Crear transformaciones certeza↔hedge de pares "entailment" claros
    """
    pairs = []
    
    if dataset is None:
        print("  MNLI no disponible, usando fallback...")
        return pairs
    
    # Filtrar datos útiles
    seen_premises = set()
    
    for item in dataset:
        if len(pairs) >= max_pairs:
            break
        
        premise = item.get("premise", "")
        hypothesis = item.get("hypothesis", "")
        label = item.get("label", -1)
        genre = item.get("genre", "")
        
        if not premise or not hypothesis:
            continue
        if len(premise) < 30 or len(hypothesis) < 20:
            continue
        if len(premise) > 200 or len(hypothesis) > 200:
            continue
        
        # Evitar duplicados
        premise_key = premise[:50]
        if premise_key in seen_premises:
            continue
        
        # Estrategia 1: Neutral con hedge en hipótesis
        if label == 1:  # neutral
            if has_hedge_marker(hypothesis) and not has_hedge_marker(premise):
                pairs.append({
                    "certain": premise,
                    "hedge": hypothesis,
                    "source": "mnli_neutral",
                    "domain": genre if genre else "general",
                })
                seen_premises.add(premise_key)
                continue
        
        # Estrategia 2: Entailment → crear hedge de la premisa
        if label == 0:  # entailment
            if not has_hedge_marker(premise) and len(premise) > 40:
                # Crear versión hedged de la premisa
                hedge_transforms = [
                    lambda p: f"It seems that {p[0].lower()}{p[1:]}".rstrip(".") + ", though this isn't entirely certain.",
                    lambda p: f"It's possible that {p[0].lower()}{p[1:]}",
                    lambda p: re.sub(r'\bis\b', 'might be', p, count=1) if ' is ' in p else None,
                    lambda p: re.sub(r'\bwill\b', 'could potentially', p, count=1) if ' will ' in p else None,
                    lambda p: re.sub(r'\bhas\b', 'seems to have', p, count=1) if ' has ' in p else None,
                    lambda p: re.sub(r'\bare\b', 'appear to be', p, count=1) if ' are ' in p else None,
                    lambda p: p.rstrip(".") + ", but I'm not completely sure." if len(p) < 120 else None,
                ]
                
                random.shuffle(hedge_transforms)
                for transform in hedge_transforms:
                    try:
                        hedged = transform(premise)
                        if hedged and hedged != premise and len(hedged) > 20:
                            pairs.append({
                                "certain": premise,
                                "hedge": hedged,
                                "source": "mnli_entailment_transform",
                                "domain": genre if genre else "general",
                            })
                            seen_premises.add(premise_key)
                            break
                    except Exception:
                        continue
    
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE 3: CalibratedMath Reference Pairs
# ═══════════════════════════════════════════════════════════════════════════

def generate_calibratedmath_pairs(n_pairs=100):
    """
    Genera pares certeza/hedge basados en el estilo CalibratedMath.
    
    Patrones: respuestas a preguntas con diferente nivel de confianza.
    """
    pairs = []
    
    templates = [
        {
            "certain": "The answer is {x}.",
            "hedge": "The answer is probably {x}, but I'm not fully confident in this calculation.",
        },
        {
            "certain": "The result is definitely {x}.",
            "hedge": "The result might be {x}, though I could have made an error.",
        },
        {
            "certain": "This equals {x}, no question about it.",
            "hedge": "I think this equals {x}, but I'd want to double-check.",
        },
        {
            "certain": "The correct value is {x}.",
            "hedge": "The value appears to be around {x}, though my estimate has some uncertainty.",
        },
        {
            "certain": "{x} is the right answer and I'm certain of it.",
            "hedge": "{x} seems like the right answer, but I'm working from limited information.",
        },
    ]
    
    # Generar con valores variados
    values = [
        "42", "3.14", "256", "0.95", "7", "1024", "2.5%", "150ms",
        "12 seconds", "0.001", "64 dimensions", "3 layers", "0.87",
        "negative", "positive", "increasing", "stable", "decreasing",
        "above average", "within normal range", "the first option",
        "both A and B", "neither", "approximately 500", "less than 1%",
    ]
    
    for _ in range(n_pairs):
        template = random.choice(templates)
        value = random.choice(values)
        pairs.append({
            "certain": template["certain"].format(x=value),
            "hedge": template["hedge"].format(x=value),
            "source": "calibratedmath_style",
            "domain": "quantitative",
        })
    
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE 4: ECHO Synthetic — Dominios KAIRI
# ═══════════════════════════════════════════════════════════════════════════

def generate_echo_synthetic():
    """
    Genera pares certeza/hedge específicos para los dominios de ECHO/KAIRI.
    Estos son de alta calidad porque están diseñados para nuestro caso de uso exacto.
    """
    pairs = []
    
    # ─── Trading ───
    trading_pairs = [
        ("The market will reverse at the 50,000 support level.", "The market might find support around 50,000, though the level could be breached."),
        ("This is a clear breakout pattern.", "This looks like it could be a breakout pattern, but false breakouts are common."),
        ("The trend will continue for at least another week.", "The trend seems likely to continue, though trend duration is inherently unpredictable."),
        ("Volume confirms the price movement.", "Volume appears to support the price movement, although volume signals can be misleading."),
        ("The correlation between BTC and ETH is breaking down.", "The BTC-ETH correlation seems weaker recently, but it's hard to tell if this is a regime change or noise."),
        ("Funding rates indicate a long squeeze is imminent.", "Funding rates are elevated, which sometimes precedes a squeeze, though the timing is uncertain."),
        ("The order book shows clear institutional accumulation.", "The order book pattern could indicate institutional accumulation, but I can't be sure without more data."),
        ("This pair has a toxicity score that makes it unsafe to trade.", "The toxicity score looks concerning, though there might be a data lag affecting the reading."),
        ("The Kelly criterion gives an optimal position size of 3%.", "The Kelly criterion suggests something around 3%, but the input parameters have estimation error."),
        ("The regime filter correctly identified the current state.", "The regime filter's classification seems reasonable, though edge cases between regimes are hard to classify definitively."),
        ("Volatility will decrease as we approach the monthly close.", "Volatility tends to decrease near monthly close, though this pattern doesn't hold every month."),
        ("The spread between the two exchanges indicates an arbitrage opportunity.", "There appears to be a spread between exchanges, but execution slippage and fees might eliminate the profit."),
        ("The trailing stop at 1.2% is optimal for this volatility level.", "A trailing stop around 1.2% seems appropriate given current volatility, though the optimal value depends on factors I haven't fully analyzed."),
        ("Dollar bars have eliminated all non-stationarity in this series.", "Dollar bars appear to have reduced non-stationarity, though some residual time-dependence likely remains."),
        ("The short model outperforms the long model in every regime.", "The short model seems to outperform in most regimes, though the RSI paradox suggests dynamics I don't fully understand."),
        ("This signal has a 90% win rate in backtesting.", "The backtest shows a ~90% win rate, but past performance may not reflect live trading conditions."),
        ("The circuit breaker will protect against tail risk.", "The circuit breaker should help with tail risk, though extreme market conditions can overwhelm any protection mechanism."),
        ("The conflict guard correctly blocked the contradictory signal.", "The conflict guard appears to have blocked the signal appropriately, but I'd want to review the specific trigger conditions."),
        ("OKX data quality is identical to Bitget for this pair.", "OKX data quality seems comparable to Bitget, though there could be differences in orderbook depth I haven't examined."),
        ("The toxicity model's 13-minute lag is the root cause of false signals.", "The toxicity model's lag is probably contributing to false signals, but there might be additional factors."),
    ]
    
    for certain, hedge in trading_pairs:
        pairs.append({"certain": certain, "hedge": hedge, "source": "echo_synthetic", "domain": "trading"})
    
    # ─── Software / Debugging ───
    software_pairs = [
        ("The null pointer exception is in the data pipeline's transform stage.", "The null pointer exception is likely in the transform stage, but I'd need to see the stack trace to confirm."),
        ("The query will run in under 100ms with the new index.", "The query should be significantly faster with the index, though the exact improvement depends on data distribution."),
        ("The race condition is caused by unsynchronized access to the shared state.", "There might be a race condition from shared state access, but reproducing it reliably would help confirm."),
        ("The service crashes because the heap size is too small.", "The heap size could be causing the crashes, though memory profiling would give a more definitive answer."),
        ("This SQL migration is safe to run on production.", "The migration looks safe, but I'd recommend running it on staging first given the table size."),
        ("The WebSocket disconnections are caused by the heartbeat interval.", "The heartbeat interval might be causing disconnections, but network-level issues could also contribute."),
        ("The NSSM service will auto-restart within 5 seconds of failure.", "The NSSM configuration should handle restarts, though I haven't tested every failure mode."),
        ("The cron job fails because stderr isn't redirected.", "The silent failure is probably a stderr issue, though there could be other factors suppressing the output."),
        ("Async IO will cut latency by 40%.", "Async IO could reduce latency, though the actual improvement depends on the IO-to-CPU ratio of the workload."),
        ("The connection pool leak is in the finally block.", "The leak might be in the finally block, but I'd want to trace the connection lifecycle to be sure."),
        ("The ALTER TABLE will complete without locking the table.", "ALTER TABLE should be safe, though on very large tables there could be locking implications."),
        ("The batch size of 1000 is optimal for this INSERT.", "A batch size around 1000 seems reasonable, but the optimal value depends on network latency and server config."),
        ("The deadlock is caused by inconsistent lock ordering.", "The deadlock could be from inconsistent lock ordering, but I'd need to analyze the lock graph to confirm."),
        ("This regex will match all valid email addresses.", "This regex should match most email addresses, though edge cases in the RFC spec might slip through."),
        ("The container will stay under 512MB of memory.", "Memory usage should stay under 512MB under normal load, though spikes during peak traffic could exceed that."),
    ]
    
    for certain, hedge in software_pairs:
        pairs.append({"certain": certain, "hedge": hedge, "source": "echo_synthetic", "domain": "software"})
    
    # ─── ML / Model Configuration ───
    ml_pairs = [
        ("The learning rate of 1e-3 is too high and will cause divergence.", "A learning rate of 1e-3 might be too aggressive, though it depends on the optimizer and batch size."),
        ("Dropout of 0.3 prevents overfitting in this architecture.", "Dropout around 0.3 should help with overfitting, though the optimal value requires tuning."),
        ("The feature importance ranking is stable across splits.", "The feature importance ranking seems fairly consistent, though there's some variation between splits."),
        ("Purged k-fold completely eliminates look-ahead bias.", "Purged k-fold should eliminate most look-ahead bias, though subtle leakage through preprocessing might remain."),
        ("The model converges in 50 epochs.", "The model appears to converge around 50 epochs, though the exact number varies with initialization."),
        ("Batch normalization is causing the training instability.", "Batch norm could be contributing to instability, but there might be other factors like learning rate schedule."),
        ("The ensemble of 5 models gives the best bias-variance tradeoff.", "An ensemble of 5 seems to work well, though I'm not sure if more models would help or if the marginal gain is negligible."),
        ("Fractional differentiation with d=0.4 preserves memory optimally.", "Fractional differentiation with d around 0.4 seems to balance memory and stationarity, though the optimal d requires empirical search."),
        ("The OOB precision of 0.89 means the model generalizes well.", "An OOB precision of 0.89 is encouraging, though OOB estimates can sometimes be optimistic."),
        ("Removing cyclic features fixed the spurious calendar correlation.", "Removing cyclic features seems to have reduced spurious correlations, but I can't be sure no useful signal was lost."),
        ("The triple barrier labels are correctly computed.", "The triple barrier labels look correct, though edge cases near barrier boundaries might have labeling ambiguity."),
        ("The class imbalance ratio of 3:1 requires SMOTE.", "The 3:1 imbalance might benefit from resampling, though it's not extreme enough to definitely require SMOTE."),
        ("The attention mechanism captures long-range dependencies in this data.", "Attention seems to capture some long-range patterns, though whether these are genuine dependencies or spurious correlations is unclear."),
        ("Weight decay of 0.01 provides sufficient regularization.", "Weight decay of 0.01 appears to help, but the optimal amount depends on model size and data volume."),
        ("The validation loss plateau indicates the model has converged.", "The validation loss seems to have plateaued, though it could also be a temporary saddle point."),
    ]
    
    for certain, hedge in ml_pairs:
        pairs.append({"certain": certain, "hedge": hedge, "source": "echo_synthetic", "domain": "ml_config"})
    
    # ─── Technical / Conceptual ───
    technical_pairs = [
        ("Gradient clipping prevents exploding gradients.", "Gradient clipping generally helps with gradient explosion, though it can interfere with optimization dynamics."),
        ("The softmax bottleneck limits model expressiveness.", "The softmax bottleneck might limit expressiveness in theory, though its practical impact is debatable."),
        ("Layer normalization works by reducing internal covariate shift.", "Layer norm is believed to reduce covariate shift, though recent evidence suggests the mechanism might be different."),
        ("The Fisher information matrix approximates loss curvature.", "The Fisher information provides a curvature approximation, though its accuracy degrades far from the optimum."),
        ("Sparse attention achieves linear complexity.", "Sparse attention can reduce complexity substantially, though whether it's truly linear depends on the sparsity pattern."),
        ("The ELBO gap equals the KL divergence to the true posterior.", "The ELBO gap should correspond to the KL divergence, though in practice the bound can be loose."),
        ("Conformal prediction provides distribution-free guarantees.", "Conformal prediction should provide coverage guarantees under exchangeability, though practical tightness varies."),
        ("The PAC-Bayes bound tightens with posterior concentration.", "The PAC-Bayes bound probably tightens with concentration, though the relationship might not be monotonic."),
        ("Residual connections solve the vanishing gradient problem.", "Residual connections help with gradient flow, though they don't completely eliminate the vanishing gradient issue in all architectures."),
        ("The transformer's quadratic attention is the computational bottleneck.", "Quadratic attention is often the bottleneck, though for shorter sequences other operations can dominate."),
        ("Knowledge distillation preserves the teacher's performance.", "Knowledge distillation typically retains most of the teacher's performance, though some capability loss is common."),
        ("MoE routing develops meaningful expert specialization.", "MoE routing seems to develop some specialization, though the degree of meaningful differentiation is still debated."),
        ("The Lottery Ticket Hypothesis holds for this architecture.", "The Lottery Ticket Hypothesis might apply here, though it's been shown to break down in some settings."),
        ("Contrastive learning produces linearly separable representations.", "Contrastive learning tends to produce well-structured representations, though linear separability isn't guaranteed for all downstream tasks."),
        ("The information bottleneck explains deep learning generalization.", "The information bottleneck theory offers one explanation for generalization, though its applicability to all architectures is disputed."),
    ]
    
    for certain, hedge in technical_pairs:
        pairs.append({"certain": certain, "hedge": hedge, "source": "echo_synthetic", "domain": "technical"})
    
    # ─── Epistemic / Meta-reasoning ───
    epistemic_pairs = [
        ("I am certain this approach will work.", "I think this approach might work, but I'm drawing on limited evidence."),
        ("The data clearly supports our hypothesis.", "The data leans toward supporting our hypothesis, though the evidence isn't conclusive."),
        ("This is the correct interpretation of the results.", "This seems like a reasonable interpretation, though other explanations are possible."),
        ("My analysis proves the causal relationship.", "My analysis suggests a correlation that might be causal, but I can't rule out confounding."),
        ("The experiment definitively answers the question.", "The experiment provides useful evidence, though additional experiments would strengthen the conclusion."),
        ("I understand exactly why the model behaves this way.", "I have a hypothesis about the model's behavior, but there are aspects I don't fully grasp."),
        ("This strategy has been validated across all regimes.", "This strategy has been tested in several regimes, though I'm not confident it generalizes to all conditions."),
        ("The error is reproducible and the fix is guaranteed to work.", "The error seems reproducible and the fix should work, but edge cases might still exist."),
        ("My confidence in this prediction is 95%.", "My confidence is moderate — maybe 60-70%, with significant uncertainty about the underlying assumptions."),
        ("The literature unanimously supports this conclusion.", "Most of the literature supports this conclusion, though there are dissenting views I haven't fully evaluated."),
        ("I have complete understanding of why SHORT outperforms LONG here.", "I have some intuition about SHORT's advantage, but the RSI paradox suggests dynamics I don't fully understand."),
        ("The backtest proves this strategy is profitable.", "The backtest results are encouraging, but historical performance doesn't guarantee future results."),
        ("This recommendation is based on extensive empirical evidence.", "This recommendation is based on available evidence, though the evidence comes from a limited set of conditions."),
        ("The calibration is perfect across all confidence levels.", "The calibration looks reasonable overall, though there might be systematic biases at extreme confidence levels."),
        ("I know the exact cause of this anomaly.", "I have a plausible explanation for this anomaly, but there could be factors I'm not considering."),
    ]
    
    for certain, hedge in epistemic_pairs:
        pairs.append({"certain": certain, "hedge": hedge, "source": "echo_synthetic", "domain": "epistemic"})
    
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# AUGMENTATION: Transformaciones sistemáticas
# ═══════════════════════════════════════════════════════════════════════════

def augment_certain_to_hedge(certain_text):
    """
    Genera variante hedged de un texto certero usando transformaciones.
    Retorna None si no se puede transformar limpiamente.
    """
    text = certain_text.strip()
    if len(text) < 15:
        return None
    
    transforms = [
        # Prefijos de incertidumbre
        (r'^(.)', lambda m: f"It seems that {m.group(1).lower()}", 0.3),
        (r'^(.)', lambda m: f"It's possible that {m.group(1).lower()}", 0.3),
        (r'^(.)', lambda m: f"I think {m.group(1).lower()}", 0.2),
        
        # Sustituciones modales
        (r'\bis\b', 'might be', 0.4),
        (r'\bare\b', 'could be', 0.4),
        (r'\bwill\b', 'might', 0.5),
        (r'\bwill\b', 'could potentially', 0.3),
        (r'\bmust\b', 'probably should', 0.4),
        (r'\balways\b', 'often', 0.5),
        (r'\bnever\b', 'rarely', 0.5),
        (r'\bclearly\b', 'apparently', 0.5),
        (r'\bcertainly\b', 'possibly', 0.5),
        (r'\bdefinitely\b', 'probably', 0.5),
        
        # Sufijos de incertidumbre
        (r'\.$', ', though I\'m not entirely certain.', 0.3),
        (r'\.$', ', but this could change.', 0.2),
        (r'\.$', ', although I\'d want to verify this.', 0.2),
    ]
    
    # Intentar una transformación aleatoria
    random.shuffle(transforms)
    for pattern, replacement, prob in transforms:
        if random.random() < prob:
            if callable(replacement):
                result = re.sub(pattern, replacement, text, count=1)
            else:
                result = re.sub(pattern, replacement, text, count=1)
            if result != text and len(result) > 20:
                return result
    
    return None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    random.seed(42)
    np.random.seed(42)
    
    print()
    print("=" * 76)
    print("  PROYECTO ECHO — Preparación de Dataset Contrastivo para φ_modal")
    print("=" * 76)
    print()
    
    all_pairs = []
    
    # ─── Source 1: CommitmentBank ──────────────────────────────────────
    print("━" * 76)
    print("  FUENTE 1: CommitmentBank (de Marneffe et al. 2019)")
    print("━" * 76)
    print()
    
    cb_content = download_commitmentbank()
    cb_pairs = process_commitmentbank(cb_content)
    print(f"  Pares extraídos: {len(cb_pairs)}")
    all_pairs.extend(cb_pairs)
    print()
    
    # ─── Source 2: MNLI ────────────────────────────────────────────────
    print("━" * 76)
    print("  FUENTE 2: MultiNLI (Williams et al. 2018)")
    print("━" * 76)
    print()
    
    mnli = load_mnli()
    mnli_pairs = process_mnli(mnli, max_pairs=800)
    print(f"  Pares extraídos: {len(mnli_pairs)}")
    all_pairs.extend(mnli_pairs)
    print()
    
    # ─── Source 3: CalibratedMath-style ────────────────────────────────
    print("━" * 76)
    print("  FUENTE 3: CalibratedMath-style (Lin et al. 2022)")
    print("━" * 76)
    print()
    
    cm_pairs = generate_calibratedmath_pairs(n_pairs=100)
    print(f"  Pares generados: {len(cm_pairs)}")
    all_pairs.extend(cm_pairs)
    print()
    
    # ─── Source 4: ECHO Synthetic ──────────────────────────────────────
    print("━" * 76)
    print("  FUENTE 4: ECHO Synthetic (dominios KAIRI)")
    print("━" * 76)
    print()
    
    echo_pairs = generate_echo_synthetic()
    print(f"  Pares generados: {len(echo_pairs)}")
    all_pairs.extend(echo_pairs)
    print()
    
    # ─── Augmentation ──────────────────────────────────────────────────
    print("━" * 76)
    print("  AUGMENTACIÓN: Transformaciones certeza → hedge")
    print("━" * 76)
    print()
    
    augmented = []
    for pair in all_pairs:
        # Intentar generar variante adicional del texto certero
        alt_hedge = augment_certain_to_hedge(pair["certain"])
        if alt_hedge and alt_hedge != pair["hedge"]:
            augmented.append({
                "certain": pair["certain"],
                "hedge": alt_hedge,
                "source": pair["source"] + "_augmented",
                "domain": pair.get("domain", "general"),
            })
    
    print(f"  Pares augmentados: {len(augmented)}")
    all_pairs.extend(augmented)
    print()
    
    # ─── Deduplication ─────────────────────────────────────────────────
    print("━" * 76)
    print("  DEDUPLICACIÓN Y LIMPIEZA")
    print("━" * 76)
    print()
    
    # Dedup por texto certero
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        key = pair["certain"][:80].lower().strip()
        if key not in seen:
            seen.add(key)
            # Limpiar
            pair["certain"] = pair["certain"].strip()
            pair["hedge"] = pair["hedge"].strip()
            # Filtrar pares demasiado cortos o idénticos
            if (len(pair["certain"]) >= 15 and 
                len(pair["hedge"]) >= 15 and
                pair["certain"].lower() != pair["hedge"].lower()):
                unique_pairs.append(pair)
    
    all_pairs = unique_pairs
    print(f"  Pares únicos después de dedup: {len(all_pairs)}")
    print()
    
    # ─── Statistics ────────────────────────────────────────────────────
    print("━" * 76)
    print("  ESTADÍSTICAS DEL DATASET")
    print("━" * 76)
    print()
    
    # Por fuente
    source_counts = {}
    for p in all_pairs:
        src = p["source"].replace("_augmented", " (aug)")
        source_counts[src] = source_counts.get(src, 0) + 1
    
    print(f"  {'Fuente':35s} {'N':>6s} {'%':>8s}")
    print(f"  {'─'*35} {'─'*6} {'─'*8}")
    for src, n in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = n / len(all_pairs) * 100
        print(f"  {src:35s} {n:6d} {pct:7.1f}%")
    print(f"  {'─'*35} {'─'*6} {'─'*8}")
    print(f"  {'TOTAL':35s} {len(all_pairs):6d}")
    print()
    
    # Por dominio
    domain_counts = {}
    for p in all_pairs:
        d = p.get("domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1
    
    print(f"  {'Dominio':25s} {'N':>6s}")
    print(f"  {'─'*25} {'─'*6}")
    for d, n in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {d:25s} {n:6d}")
    print()
    
    # Longitud promedio
    cert_lens = [len(p["certain"]) for p in all_pairs]
    hedge_lens = [len(p["hedge"]) for p in all_pairs]
    print(f"  Longitud media (certeza): {np.mean(cert_lens):.0f} chars")
    print(f"  Longitud media (hedge):   {np.mean(hedge_lens):.0f} chars")
    print(f"  Ratio hedge/certeza:      {np.mean(hedge_lens)/np.mean(cert_lens):.2f}x")
    print()
    
    # ─── Guardar ───────────────────────────────────────────────────────
    print("━" * 76)
    print("  GUARDANDO DATASET")
    print("━" * 76)
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Dataset principal
    output_path = OUTPUT_DIR / "phi_modal_training.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)
    print(f"  Dataset guardado: {output_path}")
    print(f"  Tamaño: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    # Train/val split (80/20)
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    train_path = OUTPUT_DIR / "phi_modal_train.json"
    val_path = OUTPUT_DIR / "phi_modal_val.json"
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_pairs, f, indent=2, ensure_ascii=False)
    
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"  Train split: {train_path} ({len(train_pairs)} pares)")
    print(f"  Val split:   {val_path} ({len(val_pairs)} pares)")
    print()
    
    # Reporte
    report = {
        "project": "ECHO",
        "module": "contrastive_dataset_preparation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_pairs": len(all_pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "sources": source_counts,
        "domains": domain_counts,
        "mean_certain_length": float(np.mean(cert_lens)),
        "mean_hedge_length": float(np.mean(hedge_lens)),
        "output_files": {
            "full": str(output_path),
            "train": str(train_path),
            "val": str(val_path),
        }
    }
    
    report_path = REPORTS_DIR / "contrastive_dataset_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Reporte: {report_path}")
    print()
    
    # ─── Samples ───────────────────────────────────────────────────────
    print("━" * 76)
    print("  MUESTRAS DEL DATASET (5 aleatorias)")
    print("━" * 76)
    print()
    
    samples = random.sample(all_pairs, min(5, len(all_pairs)))
    for i, s in enumerate(samples):
        print(f"  [{i+1}] Fuente: {s['source']} | Dominio: {s.get('domain', '?')}")
        print(f"      CERT: {s['certain'][:100]}{'...' if len(s['certain'])>100 else ''}")
        print(f"      HEDGE: {s['hedge'][:100]}{'...' if len(s['hedge'])>100 else ''}")
        print()
    
    # ─── Next steps ────────────────────────────────────────────────────
    print("=" * 76)
    print("  SIGUIENTE PASO: Fine-tuning de φ_modal")
    print("=" * 76)
    print()
    print(f"  Dataset listo: {len(all_pairs)} pares contrastivos")
    print(f"  Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    print()
    print("  Para fine-tuning:")
    print("    1. Cargar BGE-large como base encoder")
    print("    2. Añadir cabeza de proyección lineal (1024 → 64)")
    print("    3. Entrenar con InfoNCE loss sobre pares contrastivos")
    print("    4. Re-ejecutar protocolo de validación con φ = [φ_content; φ_modal]")
    print()


if __name__ == "__main__":
    main()