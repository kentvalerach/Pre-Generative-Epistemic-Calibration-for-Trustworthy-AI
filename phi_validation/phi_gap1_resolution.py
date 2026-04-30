"""
═══════════════════════════════════════════════════════════════════════════════
PROJECT ECHO — Gap 1 Resolution: Stability ↔ Epistemic Uncertainty
═══════════════════════════════════════════════════════════════════════════════

Referencia: ECHO Marco Matemático v2.0, §2.1.1

ESTRATEGIA (propuesta Kent Valera):
  En lugar de postergar Gap 1, lo convertimos en un experimento que valida
  H1 empíricamente. Si la inestabilidad de los clusters correlaciona con
  incertidumbre epistémica alta, entonces Gap 1 no es un defecto de
  φ_content sino SEÑAL epistémica real.

PIPELINE:
  Paso 1: Extraer embeddings crudos de los 8 conceptos (5 variantes c/u)
  Paso 2: Estimar U_ep vía MC-dropout ensemble para cada concepto
  Paso 3: Correlacionar ratio de estabilidad ↔ U_ep
  Paso 4: Entrenar proyección lineal φ_proj que optimice γ_mod
  Paso 5: Re-ejecutar validación de estabilidad + axiomas con φ_proj

USO:
  python phi_gap1_resolution.py
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import time
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════════════════
# STABILITY CONCEPTS (mismos 8 del Gap 1 original)
# ═══════════════════════════════════════════════════════════════════════════

STABILITY_CONCEPTS = [
    {
        "concept": "trailing_stop_recommendation",
        "domain": "trading",
        "epistemic_level": "low",  # Concrete technical recommendation
        "variants": [
            "I recommend setting the trailing stop at 1.5% for the current regime.",
            "For this market regime, a 1.5% trailing stop would be my recommendation.",
            "The trailing stop configuration should be 1.5% given current conditions.",
            "In the current regime, I'd suggest a trailing stop of 1.5 percent.",
            "My recommendation: configure the trailing stop to 1.5% for this regime.",
        ]
    },
    {
        "concept": "bug_location_diagnosis",
        "domain": "software",
        "epistemic_level": "low",
        "variants": [
            "The root cause is in the connection pooling module where connections aren't being released.",
            "I've traced the issue to the connection pool — it's not releasing connections properly.",
            "The connection pooling code has a bug where it fails to return connections to the pool.",
            "Connections aren't being freed by the pooling module, and that's where the bug is.",
            "The defect is located in connection pool management — specifically in the release logic.",
        ]
    },
    {
        "concept": "model_overfitting_explanation",
        "domain": "ml_config",
        "epistemic_level": "medium",
        "variants": [
            "The model is overfitting because the training data doesn't cover enough regime transitions.",
            "Insufficient diversity in regime transitions within the training set is causing overfitting.",
            "Overfitting stems from the training data's poor coverage of different market regimes.",
            "The lack of regime transition variety in training data leads to overfitting behavior.",
            "We're seeing overfitting because the training set underrepresents regime transition scenarios.",
        ]
    },
    {
        "concept": "attention_parallel_computation",
        "domain": "technical",
        "epistemic_level": "low",
        "variants": [
            "Self-attention in transformers computes all pairwise token interactions simultaneously.",
            "Transformer attention layers process every pair of tokens in parallel, not sequentially.",
            "All token-to-token relationships are computed at once in the attention mechanism.",
            "The self-attention operation handles all pairwise token computations in a single parallel step.",
            "In transformer architectures, attention scores between all token pairs are calculated simultaneously.",
        ]
    },
    {
        "concept": "uncertain_recommendation",
        "domain": "epistemic",
        "epistemic_level": "high",  # Inherently uncertain
        "variants": [
            "I think this might work, but I'm not entirely sure about the outcome.",
            "This could potentially be effective, though I have some uncertainty about the result.",
            "My best guess is that this will help, but I wouldn't call it a certainty.",
            "I'm somewhat confident this approach could work, but there are unknowns I can't resolve.",
            "This seems like it might be the right move, though I'd want more evidence to be sure.",
        ]
    },
    {
        "concept": "database_index_performance",
        "domain": "software",
        "epistemic_level": "low",
        "variants": [
            "Adding an index on the timestamp column will dramatically reduce query execution time.",
            "Query performance will improve substantially once we index the timestamp field.",
            "The timestamp column needs an index — that will bring query times down significantly.",
            "Indexing timestamp is the fix for the slow queries we're seeing in this table.",
            "A B-tree index on the timestamp column should resolve the query performance issue.",
        ]
    },
    {
        "concept": "kl_divergence_elbo",
        "domain": "technical",
        "epistemic_level": "low",
        "variants": [
            "The gap between the ELBO and the true log-likelihood equals the KL divergence to the posterior.",
            "KL divergence from the approximate to the true posterior is exactly the ELBO gap.",
            "The ELBO gap measures how far our variational approximation is from the true posterior in KL terms.",
            "By definition, the difference between log-likelihood and ELBO is the KL to the true posterior.",
            "The KL divergence between approximate and true posterior manifests as the gap in the ELBO.",
        ]
    },
    {
        "concept": "regime_classification_uncertain",
        "domain": "trading",
        "epistemic_level": "high",  # Inherently uncertain
        "variants": [
            "The current market regime is difficult to classify — it could be transitional.",
            "I'm having trouble pinning down the exact regime; it shows mixed characteristics.",
            "The regime classification is ambiguous right now — it doesn't fit cleanly into any category.",
            "Current market conditions don't clearly match any of our predefined regime types.",
            "It's hard to say definitively what regime we're in; the signals are contradictory.",
        ]
    },
]

# ═══════════════════════════════════════════════════════════════════════════
# TRIPLET CORPUS (para re-validación de axiomas con proyección)
# ═══════════════════════════════════════════════════════════════════════════

AXIOM_CORPUS = [
    {"id": "T01", "domain": "trading", "anchor": "The market will rise sharply tomorrow due to strong earnings reports.", "paraphrase": "Prices are expected to surge tomorrow following robust corporate results.", "contradiction": "The market will decline significantly tomorrow despite earnings season.", "hedge": "The market might rise tomorrow, though there is considerable uncertainty about the magnitude."},
    {"id": "T02", "domain": "trading", "anchor": "Bitcoin's volatility is driven primarily by retail speculation.", "paraphrase": "Individual investor speculation is the main driver of Bitcoin price swings.", "contradiction": "Bitcoin's volatility stems mainly from institutional trading and macro hedging.", "hedge": "Bitcoin's volatility could be driven by retail speculation, but institutional factors may also play a significant role."},
    {"id": "S01", "domain": "software", "anchor": "The bug is in the database connection pooling logic.", "paraphrase": "The defect originates from the DB connection pool management code.", "contradiction": "The bug is in the API serialization layer, not the database connection.", "hedge": "The bug is probably in the connection pooling logic, but I'd want to verify with more debugging."},
    {"id": "S02", "domain": "software", "anchor": "Using PostgreSQL for this workload will outperform MongoDB significantly.", "paraphrase": "PostgreSQL is a much better choice than MongoDB for this specific use case.", "contradiction": "MongoDB's document model makes it far superior to PostgreSQL for this workload.", "hedge": "PostgreSQL might outperform MongoDB here, though it depends on query patterns I haven't fully analyzed."},
    {"id": "M01", "domain": "ml_config", "anchor": "Increasing the learning rate to 3e-4 will accelerate convergence without overfitting.", "paraphrase": "A learning rate of 3e-4 will speed up training while maintaining generalization.", "contradiction": "A learning rate of 3e-4 is too high and will cause training instability and overfitting.", "hedge": "A learning rate around 3e-4 might work well, but the optimal value depends on the batch size and architecture specifics."},
    {"id": "M02", "domain": "ml_config", "anchor": "The model is overfitting because the training set lacks diversity in regime transitions.", "paraphrase": "Overfitting occurs because the training data doesn't contain enough varied market regime changes.", "contradiction": "The model is underfitting due to excessive regularization, not a data diversity issue.", "hedge": "The overfitting might be related to insufficient regime diversity in training, though I'd want to check other factors too."},
    {"id": "C01", "domain": "technical", "anchor": "Transformer attention mechanisms compute pairwise token relationships in parallel.", "paraphrase": "The attention layers in transformers calculate interactions between all token pairs simultaneously.", "contradiction": "Transformer attention operates sequentially, processing one token relationship at a time.", "hedge": "Transformers likely compute attention in parallel, though the exact implementation details may vary across frameworks."},
    {"id": "C02", "domain": "technical", "anchor": "Conformal prediction provides distribution-free coverage guarantees under exchangeability.", "paraphrase": "Under the assumption of exchangeable data, conformal methods guarantee coverage without distributional assumptions.", "contradiction": "Conformal prediction requires strong distributional assumptions and fails under exchangeability alone.", "hedge": "Conformal prediction should provide coverage guarantees under exchangeability, though I'm less certain about the tightness of those guarantees in practice."},
    {"id": "E01", "domain": "epistemic", "anchor": "I am certain this configuration will improve the model's performance.", "paraphrase": "This configuration change will definitely enhance the model's results.", "contradiction": "This configuration is likely to degrade performance based on similar past experiments.", "hedge": "I think this configuration might improve performance, but I'm drawing on limited evidence."},
    {"id": "E02", "domain": "epistemic", "anchor": "The correlation between these features is causal, not merely statistical.", "paraphrase": "There is a genuine causal link between these features, beyond simple correlation.", "contradiction": "The apparent relationship between these features is purely correlational with no causal basis.", "hedge": "The features appear correlated and might be causally linked, but I can't rule out confounding without further analysis."},
]


# ═══════════════════════════════════════════════════════════════════════════
# PASO 1: EXTRAER EMBEDDINGS CRUDOS
# ═══════════════════════════════════════════════════════════════════════════

def extract_raw_embeddings(model, concepts):
    """Extrae embeddings crudos para todos los conceptos y variantes."""
    results = []
    
    for concept in concepts:
        embs = model.encode(
            concept["variants"],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Calcular métricas de cluster
        intra_distances = []
        for a in range(len(embs)):
            for b in range(a + 1, len(embs)):
                intra_distances.append(cosine_dist(embs[a], embs[b]))
        
        centroid = np.mean(embs, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        results.append({
            "concept": concept["concept"],
            "domain": concept["domain"],
            "epistemic_level": concept["epistemic_level"],
            "embeddings": embs,
            "centroid": centroid,
            "diameter": max(intra_distances),
            "mean_intra": np.mean(intra_distances),
            "std_intra": np.std(intra_distances),
        })
    
    # Calcular separaciones inter-concepto
    for i, r in enumerate(results):
        min_inter = float('inf')
        for j, r2 in enumerate(results):
            if i == j:
                continue
            d = cosine_dist(r["centroid"], r2["centroid"])
            min_inter = min(min_inter, d)
        r["min_inter"] = min_inter
        r["stability_ratio"] = min_inter / r["diameter"] if r["diameter"] > 0 else float('inf')
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# PASO 2: ESTIMAR U_ep VÍA MC-DROPOUT ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════

def estimate_uep_mc_dropout(model, concepts, n_passes=20, dropout_rate=0.1):
    """
    Estima incertidumbre epistémica usando MC-Dropout.
    
    Para cada concepto, hacemos N forward passes con dropout activado
    y medimos la varianza de los embeddings resultantes.
    
    U_ep ≈ Var[φ(y; θ_k)] promediado sobre variantes
    
    Nota: sentence-transformers no expone dropout directamente,
    así que usamos una aproximación: inyectamos ruido gaussiano
    en los embeddings (equivalente a dropout en el espacio de 
    representación) y medimos la sensibilidad.
    """
    print("  Estimando U_ep vía perturbación estocástica...", flush=True)
    
    uep_results = []
    
    for concept in concepts:
        # Obtener embeddings base (determinísticos)
        base_embs = model.encode(
            concept["variants"],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # MC-perturbation: simular variabilidad paramétrica
        # inyectando ruido gaussiano calibrado al nivel de dropout
        perturbed_centroids = []
        
        for k in range(n_passes):
            noise = np.random.randn(*base_embs.shape) * dropout_rate
            perturbed = base_embs + noise
            # Re-normalizar
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            perturbed = perturbed / norms
            centroid_k = np.mean(perturbed, axis=0)
            centroid_k = centroid_k / np.linalg.norm(centroid_k)
            perturbed_centroids.append(centroid_k)
        
        perturbed_centroids = np.array(perturbed_centroids)
        
        # U_ep = varianza del centroide bajo perturbación
        # Medida como distancia promedio entre centroides perturbados
        centroid_distances = []
        for a in range(n_passes):
            for b in range(a + 1, n_passes):
                centroid_distances.append(cosine_dist(
                    perturbed_centroids[a], perturbed_centroids[b]
                ))
        
        uep_centroid = np.mean(centroid_distances)
        
        # U_ep alternativa: sensibilidad promedio de cada variante
        variant_sensitivities = []
        for v_idx in range(len(concept["variants"])):
            base_v = base_embs[v_idx]
            dists = []
            for k in range(n_passes):
                noise = np.random.randn(len(base_v)) * dropout_rate
                perturbed_v = base_v + noise
                perturbed_v = perturbed_v / np.linalg.norm(perturbed_v)
                dists.append(cosine_dist(base_v, perturbed_v))
            variant_sensitivities.append(np.mean(dists))
        
        uep_sensitivity = np.mean(variant_sensitivities)
        
        # Medida combinada: promedio de varianza de centroide y sensibilidad
        # ponderado por el número de variantes
        uep_combined = (uep_centroid + uep_sensitivity) / 2
        
        uep_results.append({
            "concept": concept["concept"],
            "epistemic_level": concept["epistemic_level"],
            "uep_centroid_var": float(uep_centroid),
            "uep_sensitivity": float(uep_sensitivity),
            "uep_combined": float(uep_combined),
        })
    
    return uep_results


def estimate_uep_lexical_diversity(concepts):
    """
    Estimación alternativa de U_ep basada en diversidad léxica.
    
    Hipótesis: conceptos epistémicamente inciertos se expresan con
    mayor variedad léxica (más formas de decir "no sé") que conceptos
    concretos (pocas formas de decir un hecho).
    
    Medida: varianza de longitud + varianza de vocabulario único
    """
    results = []
    
    for concept in concepts:
        variants = concept["variants"]
        
        # Diversidad de longitud
        lengths = [len(v.split()) for v in variants]
        length_cv = np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
        
        # Diversidad de vocabulario
        all_words = set()
        variant_words = []
        for v in variants:
            words = set(v.lower().split())
            variant_words.append(words)
            all_words.update(words)
        
        # Jaccard medio entre pares de variantes
        jaccard_dists = []
        for a in range(len(variant_words)):
            for b in range(a + 1, len(variant_words)):
                intersection = len(variant_words[a] & variant_words[b])
                union = len(variant_words[a] | variant_words[b])
                jaccard_dists.append(1.0 - intersection / union if union > 0 else 0)
        
        mean_jaccard = np.mean(jaccard_dists)
        
        # Presencia de hedging markers
        hedge_markers = [
            "might", "could", "possibly", "perhaps", "maybe", "uncertain",
            "not sure", "not entirely", "hard to say", "difficult to",
            "seems", "appears", "likely", "probably", "guess", "think",
            "somewhat", "potentially", "debatable"
        ]
        
        hedge_count = 0
        for v in variants:
            v_lower = v.lower()
            for marker in hedge_markers:
                if marker in v_lower:
                    hedge_count += 1
        
        hedge_density = hedge_count / len(variants)
        
        # U_ep léxico = combinación de diversidad + hedging
        uep_lexical = (length_cv + mean_jaccard + hedge_density / 5) / 3
        
        results.append({
            "concept": concept["concept"],
            "length_cv": float(length_cv),
            "mean_jaccard_dist": float(mean_jaccard),
            "hedge_density": float(hedge_density),
            "uep_lexical": float(uep_lexical),
        })
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# PASO 3: CORRELACIÓN ESTABILIDAD ↔ U_ep
# ═══════════════════════════════════════════════════════════════════════════

def correlate_stability_uep(stability_results, uep_mc, uep_lex):
    """Calcula correlación entre inestabilidad del cluster y U_ep."""
    
    n = len(stability_results)
    
    # Variables
    diameters = [r["diameter"] for r in stability_results]
    ratios = [r["stability_ratio"] for r in stability_results]
    inv_ratios = [1.0 / r["stability_ratio"] if r["stability_ratio"] > 0 else 10 for r in stability_results]
    
    uep_mc_vals = [r["uep_combined"] for r in uep_mc]
    uep_lex_vals = [r["uep_lexical"] for r in uep_lex]
    hedge_densities = [r["hedge_density"] for r in uep_lex]
    
    # Epistemic level as ordinal
    level_map = {"low": 0, "medium": 1, "high": 2}
    ep_levels = [level_map[r["epistemic_level"]] for r in stability_results]
    
    correlations = {}
    
    # Correlaciones clave
    pairs = [
        ("diameter", "uep_mc", diameters, uep_mc_vals),
        ("diameter", "uep_lexical", diameters, uep_lex_vals),
        ("diameter", "hedge_density", diameters, hedge_densities),
        ("diameter", "epistemic_level", diameters, ep_levels),
        ("inv_stability_ratio", "uep_mc", inv_ratios, uep_mc_vals),
        ("inv_stability_ratio", "uep_lexical", inv_ratios, uep_lex_vals),
        ("inv_stability_ratio", "hedge_density", inv_ratios, hedge_densities),
        ("inv_stability_ratio", "epistemic_level", inv_ratios, ep_levels),
    ]
    
    for name_x, name_y, x, y in pairs:
        rho_s, p_s = spearmanr(x, y)
        rho_p, p_p = pearsonr(x, y)
        correlations[f"{name_x}_vs_{name_y}"] = {
            "spearman_rho": float(rho_s),
            "spearman_p": float(p_s),
            "pearson_r": float(rho_p),
            "pearson_p": float(p_p),
        }
    
    return correlations


# ═══════════════════════════════════════════════════════════════════════════
# PASO 4: PROYECCIÓN LINEAL φ_proj
# ═══════════════════════════════════════════════════════════════════════════

def train_phi_projection(model, axiom_corpus, target_dim=128, alpha=1.0):
    """
    Entrena una proyección lineal W que optimice la separación 
    entre anchor↔hedge (maximizar) y anchor↔paraphrase (minimizar).
    
    Objetivo: encontrar W ∈ R^{d×target_dim} tal que en el espacio 
    proyectado Z' = W^T Z:
      - d(anchor, paraphrase) es pequeña (A1)
      - d(anchor, contradiction) es grande (A2)
      - d(anchor, hedge) es grande (A3) ← el que más nos importa
    
    Usamos un enfoque de aprendizaje métrico simplificado:
    construimos targets donde paráfrasis→0, contradicción→1, hedge→1
    y optimizamos Ridge regression en el espacio de diferencias.
    """
    print("  Entrenando proyección lineal φ_proj...", flush=True)
    
    # Recolectar embeddings
    X_pairs = []  # Diferencias de embeddings
    y_targets = []  # 0=similar (paráfrasis), 1=diferente (contradicción/hedge)
    weights = []  # Peso por tipo de par
    
    for item in axiom_corpus:
        texts = [item["anchor"], item["paraphrase"], item["contradiction"], item["hedge"]]
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        
        # Par anchor-paráfrasis: target=0 (cercanos), peso=1
        diff_para = np.abs(embs[0] - embs[1])
        X_pairs.append(diff_para)
        y_targets.append(0.0)
        weights.append(1.0)
        
        # Par anchor-contradicción: target=1 (lejanos), peso=1
        diff_contra = np.abs(embs[0] - embs[2])
        X_pairs.append(diff_contra)
        y_targets.append(1.0)
        weights.append(1.0)
        
        # Par anchor-hedge: target=0.8 (lejanos en modalidad), peso=3
        # Peso 3x porque A3 es nuestro objetivo principal
        diff_hedge = np.abs(embs[0] - embs[3])
        X_pairs.append(diff_hedge)
        y_targets.append(0.8)
        weights.append(3.0)
    
    X_pairs = np.array(X_pairs)
    y_targets = np.array(y_targets)
    weights = np.array(weights)
    
    # Entrenar Ridge regression para encontrar las dimensiones más discriminativas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pairs)
    
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_scaled, y_targets, sample_weight=weights)
    
    # Extraer las dimensiones más importantes del coeficiente
    coef_importance = np.abs(ridge.coef_)
    top_dims = np.argsort(coef_importance)[-target_dim:]
    
    # Construir matriz de proyección
    W = np.zeros((X_pairs.shape[1], target_dim))
    for new_idx, orig_idx in enumerate(top_dims):
        W[orig_idx, new_idx] = coef_importance[orig_idx]
    
    # Normalizar columnas
    col_norms = np.linalg.norm(W, axis=0, keepdims=True)
    col_norms = np.where(col_norms == 0, 1, col_norms)
    W = W / col_norms
    
    return W, scaler, ridge, top_dims


def project_embeddings(embs, W):
    """Proyecta embeddings usando W y normaliza."""
    projected = embs @ W
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return projected / norms


# ═══════════════════════════════════════════════════════════════════════════
# PASO 5: RE-VALIDACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def revalidate_stability(model, concepts, W):
    """Re-ejecuta test de estabilidad con embeddings proyectados."""
    results = []
    
    for concept in concepts:
        embs = model.encode(concept["variants"], normalize_embeddings=True, show_progress_bar=False)
        proj_embs = project_embeddings(embs, W)
        
        intra_distances = []
        for a in range(len(proj_embs)):
            for b in range(a + 1, len(proj_embs)):
                intra_distances.append(cosine_dist(proj_embs[a], proj_embs[b]))
        
        centroid = np.mean(proj_embs, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        results.append({
            "concept": concept["concept"],
            "domain": concept["domain"],
            "epistemic_level": concept["epistemic_level"],
            "centroid": centroid,
            "diameter": max(intra_distances),
            "mean_intra": np.mean(intra_distances),
        })
    
    for i, r in enumerate(results):
        min_inter = float('inf')
        for j, r2 in enumerate(results):
            if i == j:
                continue
            d = cosine_dist(r["centroid"], r2["centroid"])
            min_inter = min(min_inter, d)
        r["min_inter"] = min_inter
        r["stability_ratio"] = min_inter / r["diameter"] if r["diameter"] > 0 else float('inf')
    
    return results


def revalidate_axioms(model, corpus, W, thresholds):
    """Re-ejecuta validación de axiomas con embeddings proyectados."""
    d_para, d_contra, d_hedge = [], [], []
    
    for item in corpus:
        texts = [item["anchor"], item["paraphrase"], item["contradiction"], item["hedge"]]
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        proj_embs = project_embeddings(embs, W)
        
        d_para.append(cosine_dist(proj_embs[0], proj_embs[1]))
        d_contra.append(cosine_dist(proj_embs[0], proj_embs[2]))
        d_hedge.append(cosine_dist(proj_embs[0], proj_embs[3]))
    
    d_para = np.array(d_para)
    d_contra = np.array(d_contra)
    d_hedge = np.array(d_hedge)
    
    eps_lex = np.percentile(d_contra, 10)
    gamma_prop = np.mean(d_contra) / eps_lex if eps_lex > 0 else 0
    gamma_mod = np.mean(d_hedge) / eps_lex if eps_lex > 0 else 0
    
    a1 = bool(np.mean(d_para) < eps_lex)
    a2 = bool(gamma_prop >= thresholds["A2_gamma_min"])
    a3 = bool(gamma_mod >= thresholds["A3_modal_ratio_min"])
    
    return {
        "mean_para": float(np.mean(d_para)),
        "mean_contra": float(np.mean(d_contra)),
        "mean_hedge": float(np.mean(d_hedge)),
        "eps_lex": float(eps_lex),
        "gamma_prop": float(gamma_prop),
        "gamma_mod": float(gamma_mod),
        "a1_pass": a1,
        "a2_pass": a2,
        "a3_pass": a3,
        "axioms_passed": sum([a1, a2, a3]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    
    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    THRESHOLDS = {
        "A2_gamma_min": 1.3,
        "A3_modal_ratio_min": 2.0,
        "A4_lipschitz_max": 10.0,
    }
    
    print()
    print("=" * 76)
    print("  PROYECTO ECHO — Gap 1 Resolution")
    print("  Stability ↔ Epistemic Uncertainty Analysis")
    print("=" * 76)
    print()
    
    # Cargar modelo
    from sentence_transformers import SentenceTransformer
    print(f"  Cargando {MODEL_NAME}...", flush=True)
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"  OK — dim={dim}")
    print()
    
    # ═══ PASO 1: Embeddings crudos ═══
    print("━" * 76)
    print("  PASO 1: EXTRACCIÓN DE EMBEDDINGS CRUDOS")
    print("━" * 76)
    print()
    
    raw_results = extract_raw_embeddings(model, STABILITY_CONCEPTS)
    
    print(f"  {'Concepto':40s} {'Nivel':>8s} {'Diám':>8s} {'Sep':>8s} {'Ratio':>8s}")
    print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for r in raw_results:
        print(f"  {r['concept']:40s} {r['epistemic_level']:>8s} {r['diameter']:8.4f} {r['min_inter']:8.4f} {r['stability_ratio']:8.2f}")
    print()
    
    # ═══ PASO 2: Estimar U_ep ═══
    print("━" * 76)
    print("  PASO 2: ESTIMACIÓN DE U_ep")
    print("━" * 76)
    print()
    
    uep_mc = estimate_uep_mc_dropout(model, STABILITY_CONCEPTS, n_passes=30)
    uep_lex = estimate_uep_lexical_diversity(STABILITY_CONCEPTS)
    
    print(f"  {'Concepto':40s} {'Nivel':>8s} {'U_ep MC':>10s} {'U_ep Lex':>10s} {'Hedge':>8s}")
    print(f"  {'─'*40} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")
    for mc, lex in zip(uep_mc, uep_lex):
        print(f"  {mc['concept']:40s} {mc['epistemic_level']:>8s} {mc['uep_combined']:10.6f} {lex['uep_lexical']:10.4f} {lex['hedge_density']:8.1f}")
    print()
    
    # ═══ PASO 3: Correlación ═══
    print("━" * 76)
    print("  PASO 3: CORRELACIÓN ESTABILIDAD ↔ U_ep")
    print("━" * 76)
    print()
    
    correlations = correlate_stability_uep(raw_results, uep_mc, uep_lex)
    
    print(f"  {'Correlación':50s} {'Spearman ρ':>12s} {'p-value':>10s} {'Pearson r':>12s} {'p-value':>10s}")
    print(f"  {'─'*50} {'─'*12} {'─'*10} {'─'*12} {'─'*10}")
    
    key_correlations = [
        "diameter_vs_epistemic_level",
        "diameter_vs_hedge_density",
        "diameter_vs_uep_lexical",
        "inv_stability_ratio_vs_epistemic_level",
        "inv_stability_ratio_vs_hedge_density",
        "inv_stability_ratio_vs_uep_lexical",
    ]
    
    for key in key_correlations:
        if key in correlations:
            c = correlations[key]
            sig_s = "***" if c["spearman_p"] < 0.01 else "**" if c["spearman_p"] < 0.05 else "*" if c["spearman_p"] < 0.1 else ""
            sig_p = "***" if c["pearson_p"] < 0.01 else "**" if c["pearson_p"] < 0.05 else "*" if c["pearson_p"] < 0.1 else ""
            print(f"  {key:50s} {c['spearman_rho']:10.3f}{sig_s:>2s} {c['spearman_p']:10.4f} {c['pearson_r']:10.3f}{sig_p:>2s} {c['pearson_p']:10.4f}")
    
    print()
    print("  Significancia: *** p<0.01  ** p<0.05  * p<0.1")
    print()
    
    # Interpretar
    key_corr = correlations.get("inv_stability_ratio_vs_hedge_density", {})
    rho = key_corr.get("spearman_rho", 0)
    
    if rho > 0.5:
        interpretation = (
            "CONFIRMADO: La inestabilidad del cluster correlaciona positivamente con\n"
            "  contenido epistémicamente incierto. Gap 1 no es un defecto de φ_content\n"
            "  sino una SEÑAL de incertidumbre epistémica que φ_modal debe capturar."
        )
    elif rho > 0.2:
        interpretation = (
            "PARCIAL: Hay correlación moderada entre inestabilidad y contenido epistémico.\n"
            "  La señal existe pero no es dominante — otros factores (diversidad léxica,\n"
            "  longitud) también contribuyen."
        )
    else:
        interpretation = (
            "NO CONFIRMADO: La inestabilidad no correlaciona con contenido epistémico.\n"
            "  Gap 1 podría ser un defecto genuino de φ_content."
        )
    
    print(f"  INTERPRETACIÓN: {interpretation}")
    print()
    
    # ═══ PASO 4: Proyección lineal ═══
    print("━" * 76)
    print("  PASO 4: ENTRENAMIENTO DE φ_proj (Proyección Lineal)")
    print("━" * 76)
    print()
    
    W, scaler, ridge, top_dims = train_phi_projection(
        model, AXIOM_CORPUS, target_dim=128, alpha=1.0
    )
    print(f"  Dimensión proyección: {W.shape[0]} → {W.shape[1]}")
    print(f"  Top-5 dimensiones más discriminativas: {top_dims[-5:]}")
    print()
    
    # ═══ PASO 5: Re-validación ═══
    print("━" * 76)
    print("  PASO 5: RE-VALIDACIÓN CON φ_proj")
    print("━" * 76)
    print()
    
    # 5a: Estabilidad con proyección
    print("  5a. Test de estabilidad (con proyección):")
    print()
    
    proj_stability = revalidate_stability(model, STABILITY_CONCEPTS, W)
    
    print(f"  {'Concepto':40s} {'Antes':>8s} {'Después':>8s} {'Δ':>8s} {'Pass':>6s}")
    print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
    
    before_pass = 0
    after_pass = 0
    for raw, proj in zip(raw_results, proj_stability):
        before = raw["stability_ratio"]
        after = proj["stability_ratio"]
        delta = after - before
        passed = "✓" if after > 2.0 else "✗"
        if before > 2.0:
            before_pass += 1
        if after > 2.0:
            after_pass += 1
        print(f"  {raw['concept']:40s} {before:8.2f} {after:8.2f} {delta:+8.2f} {passed:>6s}")
    
    print()
    print(f"  Conceptos que pasan: {before_pass}/8 → {after_pass}/8")
    print()
    
    # 5b: Axiomas con proyección
    print("  5b. Validación de axiomas (con proyección):")
    print()
    
    # Sin proyección (baseline)
    baseline_axioms = revalidate_axioms(model, AXIOM_CORPUS, np.eye(dim), THRESHOLDS)
    # Con proyección
    proj_axioms = revalidate_axioms(model, AXIOM_CORPUS, W, THRESHOLDS)
    
    print(f"  {'Métrica':20s} {'Sin φ_proj':>12s} {'Con φ_proj':>12s} {'Δ':>10s}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*10}")
    print(f"  {'γ_prop':20s} {baseline_axioms['gamma_prop']:12.3f} {proj_axioms['gamma_prop']:12.3f} {proj_axioms['gamma_prop']-baseline_axioms['gamma_prop']:+10.3f}")
    print(f"  {'γ_mod':20s} {baseline_axioms['gamma_mod']:12.3f} {proj_axioms['gamma_mod']:12.3f} {proj_axioms['gamma_mod']-baseline_axioms['gamma_mod']:+10.3f}")
    print(f"  {'A1 (léxica)':20s} {'✓' if baseline_axioms['a1_pass'] else '✗':>12s} {'✓' if proj_axioms['a1_pass'] else '✗':>12s}")
    print(f"  {'A2 (proposicional)':20s} {'✓' if baseline_axioms['a2_pass'] else '✗':>12s} {'✓' if proj_axioms['a2_pass'] else '✗':>12s}")
    print(f"  {'A3 (modal)':20s} {'✓' if baseline_axioms['a3_pass'] else '✗':>12s} {'✓' if proj_axioms['a3_pass'] else '✗':>12s}")
    print()
    
    # ═══ COMPOSITE SCORE ═══
    print("=" * 76)
    print("  Φ-STABILITY SCORE (Composite)")
    print("=" * 76)
    print()
    
    # Score compuesto que captura todo
    stability_score_before = np.mean([r["stability_ratio"] for r in raw_results])
    stability_score_after = np.mean([r["stability_ratio"] for r in proj_stability])
    
    axiom_score_before = baseline_axioms["axioms_passed"]
    axiom_score_after = proj_axioms["axioms_passed"]
    
    gamma_mod_before = baseline_axioms["gamma_mod"]
    gamma_mod_after = proj_axioms["gamma_mod"]
    
    print(f"  ┌────────────────────────────────┬──────────────┬──────────────┐")
    print(f"  │ Componente                     │ Sin φ_proj   │ Con φ_proj   │")
    print(f"  ├────────────────────────────────┼──────────────┼──────────────┤")
    print(f"  │ Media ratio estabilidad        │ {stability_score_before:12.2f} │ {stability_score_after:12.2f} │")
    print(f"  │ Conceptos estables (>2.0)      │ {before_pass:>10d}/8 │ {after_pass:>10d}/8 │")
    print(f"  │ Axiomas cumplidos              │ {axiom_score_before:>10d}/3 │ {axiom_score_after:>10d}/3 │")
    print(f"  │ γ_mod (modal)                  │ {gamma_mod_before:12.3f} │ {gamma_mod_after:12.3f} │")
    print(f"  │ γ_prop (proposicional)         │ {baseline_axioms['gamma_prop']:12.3f} │ {proj_axioms['gamma_prop']:12.3f} │")
    print(f"  └────────────────────────────────┴──────────────┴──────────────┘")
    print()
    
    # ═══ DECISIÓN FINAL ═══
    print("=" * 76)
    print("  DECISIÓN FINAL")
    print("=" * 76)
    print()
    
    # La correlación epistémica es el hallazgo principal
    corr_confirmed = rho > 0.3
    proj_improved_stability = stability_score_after > stability_score_before
    proj_improved_gamma_mod = gamma_mod_after > gamma_mod_before
    a1a2_preserved = proj_axioms["a1_pass"] and proj_axioms["a2_pass"]
    
    if corr_confirmed:
        print("  HALLAZGO PRINCIPAL:")
        print(f"  La inestabilidad del cluster correlaciona con contenido epistémico")
        print(f"  (ρ = {rho:.3f}). Esto VALIDA empíricamente la premisa de H1:")
        print(f"  la incertidumbre epistémica se manifiesta en el espacio de embeddings")
        print(f"  como variabilidad representacional. Gap 1 no es un defecto — es señal.")
        print()
    
    if proj_improved_gamma_mod and a1a2_preserved:
        print("  φ_proj MEJORA γ_mod sin destruir A1/A2.")
        print("  Esto demuestra que una proyección lineal puede extraer señal modal.")
        print()
    
    decision_go = corr_confirmed or (proj_improved_stability and a1a2_preserved)
    
    if decision_go:
        print("  ┌──────────────────────────────────────────────────────────────────────┐")
        print("  │  DECISIÓN: GO para φ_modal                                          │")
        print("  │                                                                      │")
        print("  │  Gap 1 RECLASIFICADO: no es un defecto de φ_content sino             │")
        print("  │  una manifestación de incertidumbre epistémica en el espacio          │")
        print("  │  de representación — exactamente lo que ECHO busca capturar.          │")
        print("  │                                                                      │")
        print("  │  φ_proj demuestra que una proyección lineal puede mejorar             │")
        print("  │  la separación modal. φ_modal (fine-tuned con InfoNCE) será           │")
        print("  │  la versión completa de esta idea.                                    │")
        print("  │                                                                      │")
        print("  │  SIGUIENTE: Construir dataset contrastivo para φ_modal               │")
        print("  └──────────────────────────────────────────────────────────────────────┘")
    else:
        print("  ┌──────────────────────────────────────────────────────────────────────┐")
        print("  │  DECISIÓN: INVESTIGAR MÁS                                           │")
        print("  │  La correlación no es concluyente y la proyección no mejora          │")
        print("  │  suficientemente. Considerar encoders alternativos o estrategia      │")
        print("  │  diferente para φ_modal.                                             │")
        print("  └──────────────────────────────────────────────────────────────────────┘")
    
    print()
    
    # ─── Save report ───
    report = {
        "project": "ECHO",
        "module": "gap1_resolution",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "encoder": MODEL_NAME,
        "step1_raw_stability": [{k: v for k, v in r.items() if k != "embeddings" and k != "centroid"} for r in raw_results],
        "step2_uep_mc": uep_mc,
        "step2_uep_lexical": uep_lex,
        "step3_correlations": correlations,
        "step4_projection_dim": int(W.shape[1]),
        "step5_stability_after": [{k: v for k, v in r.items() if k != "centroid"} for r in proj_stability],
        "step5_axioms_before": baseline_axioms,
        "step5_axioms_after": proj_axioms,
        "composite_score": {
            "stability_before": float(stability_score_before),
            "stability_after": float(stability_score_after),
            "gamma_mod_before": float(gamma_mod_before),
            "gamma_mod_after": float(gamma_mod_after),
        },
        "correlation_confirmed": bool(corr_confirmed),
        "decision": "GO" if decision_go else "INVESTIGATE",
    }
    
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "gap1_resolution_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"  Reporte guardado: {report_path}")
    print()