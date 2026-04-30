"""
═══════════════════════════════════════════════════════════════════════════════
PROJECT ECHO — Fine-tuning de φ_modal con InfoNCE
═══════════════════════════════════════════════════════════════════════════════

Referencia: ECHO Marco Matemático v2.0, §2.1.1

ARQUITECTURA:
  φ(y) = [φ_content(y); φ_modal(y)]
  
  φ_content = BGE-large (congelado, 1024 dim)
  φ_modal   = Proyección lineal entrenable:
              MLP: 1024 → 256 → 64 con LayerNorm y ReLU
  
  Espacio final: 1024 + 64 = 1088 dim
  Pero para evaluar A3, usamos SOLO φ_modal (64 dim).

FUNCIÓN DE PÉRDIDA — InfoNCE modificado:
  Para cada (certain_i, hedge_i) en el batch:
    - Queremos: distancia(φ_modal(certain_i), φ_modal(hedge_i)) GRANDE
    - Queremos: distancia(φ_modal(certain_i), φ_modal(certain_j≠i)) pequeña
                si certain_j es similar en certeza
    
  Formulación: tratamos certain como "anchor" y sus paráfrasis certain 
  en otros pares como "positives"; hedge_i es "negative duro".
  
  Como no tenemos paráfrasis directas, usamos una formulación más simple:
    L = -log[ exp(-d(cert_i, hedge_i)/τ) / Σ_j exp(-d(cert_i, hedge_j)/τ) ]
    
  Es decir: queremos que hedge_i sea el MÁS LEJANO de cert_i entre todos 
  los hedges del batch. Esto fuerza a φ_modal a aprender la dirección 
  de modalidad.

OVERSAMPLING:
  Los pares de dominios KAIRI (trading, software, ml_config, technical, 
  epistemic) reciben peso 3x durante el sampling del DataLoader.

EARLY STOPPING:
  Monitor: γ_mod en validation set
  Paciencia: 3 épocas sin mejora

USO:
  python phi_modal_training.py
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
import time
import random
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.spatial.distance import cosine as cosine_dist

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "contrastive_pairs"
REPORTS_DIR = Path(__file__).parent / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = DATA_DIR / "phi_modal_train.json"
VAL_FILE = DATA_DIR / "phi_modal_val.json"

# Hyperparameters
CONFIG = {
    "base_encoder": "BAAI/bge-large-en-v1.5",
    "projection_dim": 64,
    "hidden_dim": 256,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "max_epochs": 50,
    "temperature": 0.1,          # Para InfoNCE
    "patience": 3,                # Early stopping
    "kairi_oversample_weight": 3.0,
    "random_seed": 42,
}

KAIRI_DOMAINS = {"trading", "software", "ml_config", "technical", "epistemic"}


# ═══════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════

class ContrastivePairsDataset(Dataset):
    """Dataset de pares (certain, hedge) para entrenar φ_modal."""
    
    def __init__(self, json_path: Path, encoder, device: str = "cpu"):
        with open(json_path, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)
        
        self.encoder = encoder
        self.device = device
        self._embeddings_cache = None
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Precomputa embeddings BGE para todos los pares (ahorra tiempo)."""
        print(f"  Precomputando embeddings para {len(self.pairs)} pares...", flush=True)
        t0 = time.time()
        
        all_certain = [p["certain"] for p in self.pairs]
        all_hedge = [p["hedge"] for p in self.pairs]
        
        with torch.no_grad():
            cert_embs = self.encoder.encode(
                all_certain, 
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64,
                convert_to_tensor=True
            )
            hedge_embs = self.encoder.encode(
                all_hedge,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64,
                convert_to_tensor=True
            )
        
        self._embeddings_cache = {
            "certain": cert_embs.cpu(),
            "hedge": hedge_embs.cpu(),
        }
        
        elapsed = time.time() - t0
        print(f"  Cache listo ({elapsed:.1f}s)")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return {
            "certain_emb": self._embeddings_cache["certain"][idx],
            "hedge_emb": self._embeddings_cache["hedge"][idx],
            "domain": self.pairs[idx].get("domain", "general"),
            "source": self.pairs[idx].get("source", "unknown"),
        }
    
    def get_sample_weights(self, oversample_weight: float = 3.0):
        """Pesos para WeightedRandomSampler: 3x para dominios KAIRI."""
        weights = []
        for p in self.pairs:
            domain = p.get("domain", "general")
            if domain in KAIRI_DOMAINS:
                weights.append(oversample_weight)
            else:
                weights.append(1.0)
        return torch.tensor(weights, dtype=torch.double)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL: φ_modal projection head
# ═══════════════════════════════════════════════════════════════════════════

class PhiModalProjection(nn.Module):
    """
    Cabeza de proyección para modalidad epistémica.
    
    Input:  embedding BGE (1024 dim, normalizado)
    Output: embedding en espacio modal (64 dim, normalizado)
    """
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, output_dim: int = 64):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Init xavier para estabilidad
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        z = self.projection(x)
        return F.normalize(z, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# LOSS: InfoNCE modificado para separar certainty/hedge
# ═══════════════════════════════════════════════════════════════════════════

def infonce_modal_loss(cert_proj: torch.Tensor, hedge_proj: torch.Tensor, 
                       temperature: float = 0.1) -> torch.Tensor:
    """
    InfoNCE que fuerza separación entre pares (certain, hedge).
    
    Para cada certain_i:
      - Su hedge_i es el "negativo duro" (debe estar lejos)
      - Los otros certain_j son "anchors relacionados en modalidad"
      
    Loss:
      L_i = -log[ exp(sim(cert_i, cert_j∼)) / (exp(sim(cert_i, cert_j∼)) + exp(sim(cert_i, hedge_i))) ]
    
    Equivalentemente, maximizamos la separación cert_i vs hedge_i relativa 
    a la similaridad cert_i vs cert_j.
    
    En la práctica usamos una formulación simétrica que trata ambas direcciones:
      - certains deben agruparse entre sí
      - hedges deben agruparse entre sí
      - pero certains deben estar LEJOS de hedges
    """
    batch_size = cert_proj.shape[0]
    
    # Similaridad coseno (embeddings ya normalizados)
    sim_cert_cert = cert_proj @ cert_proj.T      # (B, B): cert vs cert
    sim_hedge_hedge = hedge_proj @ hedge_proj.T  # (B, B): hedge vs hedge  
    sim_cert_hedge = cert_proj @ hedge_proj.T    # (B, B): cert vs hedge
    
    # Escalar por temperatura
    sim_cert_cert = sim_cert_cert / temperature
    sim_hedge_hedge = sim_hedge_hedge / temperature
    sim_cert_hedge = sim_cert_hedge / temperature
    
    # Máscara para excluir diagonal (cert_i vs cert_i = 1 trivialmente)
    mask = torch.eye(batch_size, device=cert_proj.device, dtype=torch.bool)
    
    # ─── Loss 1: Separación cert↔hedge ───
    # Para cada i: queremos que sim(cert_i, hedge_i) sea BAJA
    # comparada con sim(cert_i, cualquier_otro_cert_j)
    #
    # Log-softmax sobre fila de sim_cert_hedge:
    # el target "positivo" es NO ser hedge_i → 
    # equivalente a: maximizar log(1 - softmax(sim_cert_hedge_i))
    #
    # Formulación más limpia: InfoNCE donde los "positivos" de cert_i 
    # son los otros certs, y el "negativo" a separar es hedge_i.
    
    # Combinar matriz de similaridades: para cada anchor cert_i,
    # los candidatos son [cert_1, ..., cert_B, hedge_1, ..., hedge_B]
    # Target: los cert_j≠i son "positivos modales" (misma clase: certain)
    # hedge_i es "negativo duro" (misma proposición pero modalidad opuesta)
    
    # Para cada fila i:
    # - numerador: exp(sim(cert_i, cert_j)) sumando sobre j≠i (los positivos)
    # - denominador: numerador + exp(sim(cert_i, hedge_i)) (el duro) + 
    #                otros exp(sim(cert_i, hedge_k≠i)) (negativos suaves)
    
    # Simplificación: InfoNCE clásico con dos vistas donde 
    # cert[i] y hedge[i] son vistas OPUESTAS (no el mismo item)
    
    # Usamos la formulación "repulsion":
    # Queremos maximizar la diferencia: media(sim cert-cert) - media(sim cert-hedge)
    
    # 1. Certs se agrupan: media de sim_cert_cert fuera de la diagonal
    sim_cc_off = sim_cert_cert.masked_fill(mask, 0).sum() / (batch_size * (batch_size - 1))
    
    # 2. Hedges se agrupan: media de sim_hedge_hedge fuera de la diagonal
    sim_hh_off = sim_hedge_hedge.masked_fill(mask, 0).sum() / (batch_size * (batch_size - 1))
    
    # 3. Cert-hedge del MISMO par deben estar lejos: diagonal de sim_cert_hedge
    sim_ch_same = torch.diag(sim_cert_hedge).mean()
    
    # 4. InfoNCE formal: para cert_i, hedge_i es el negativo duro
    #    targets[i] = i significa "el verdadero positivo de cert_i es cert_i"
    #    pero como tenemos auto-similaridad = 1/τ, necesitamos formulación diferente
    
    # Formulación principal: cross-entropy donde para cada cert_i,
    # el objetivo es que hedge_i (posición i en sim_cert_hedge) 
    # tenga la MÍNIMA similaridad entre todos los targets
    
    # Equivalente: tratamos cada cert_i como query, y queremos que 
    # argmin de sim(cert_i, hedge_k) = i (hedge_i es el más lejano)
    
    # Implementación: cross-entropy con targets inversos
    neg_sim_ch = -sim_cert_hedge
    targets_ch = torch.arange(batch_size, device=cert_proj.device)
    loss_ch = F.cross_entropy(neg_sim_ch, targets_ch)
    
    # Versión simétrica: lo mismo desde hedge hacia cert
    neg_sim_hc = -sim_cert_hedge.T
    loss_hc = F.cross_entropy(neg_sim_hc, targets_ch)
    
    # Componente de agregación modal
    # (queremos que certs estén más juntos entre sí que con hedges)
    modal_gap = sim_cc_off + sim_hh_off - 2 * sim_ch_same
    
    # Loss total: combinación
    # - loss_ch + loss_hc: fuerza separación direccional
    # - -modal_gap: recompensa por agregación modal (restamos porque queremos maximizarlo)
    
    total_loss = 0.5 * (loss_ch + loss_hc) - 0.1 * modal_gap
    
    # Métricas auxiliares para monitoreo
    with torch.no_grad():
        # Separación media cert↔hedge (diagonal)
        mean_ch_dist = (1.0 - torch.diag(sim_cert_hedge * temperature)).mean().item()
        # Agregación cert↔cert (fuera de diagonal)
        mean_cc_sim = (sim_cert_cert.masked_fill(mask, 0) * temperature).sum().item() / (batch_size * (batch_size - 1))
    
    return total_loss, {
        "loss_total": total_loss.item(),
        "loss_ch": loss_ch.item(),
        "loss_hc": loss_hc.item(),
        "modal_gap": modal_gap.item(),
        "mean_ch_dist": mean_ch_dist,
        "mean_cc_sim": mean_cc_sim,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_gamma_mod(phi_modal: nn.Module, val_loader: DataLoader, 
                       device: str = "cpu") -> dict:
    """
    Evalúa γ_mod = distancia_media(cert, hedge) / percentil_10_inter-cert
    
    En el espacio proyectado φ_modal.
    """
    phi_modal.eval()
    
    all_cert = []
    all_hedge = []
    
    with torch.no_grad():
        for batch in val_loader:
            cert_embs = batch["certain_emb"].to(device)
            hedge_embs = batch["hedge_emb"].to(device)
            
            cert_proj = phi_modal(cert_embs)
            hedge_proj = phi_modal(hedge_embs)
            
            all_cert.append(cert_proj.cpu().numpy())
            all_hedge.append(hedge_proj.cpu().numpy())
    
    all_cert = np.concatenate(all_cert, axis=0)
    all_hedge = np.concatenate(all_hedge, axis=0)
    
    # Distancias cert-hedge (del mismo par)
    d_cert_hedge = []
    for i in range(len(all_cert)):
        d_cert_hedge.append(cosine_dist(all_cert[i], all_hedge[i]))
    
    # Distancias cert-cert (entre pares diferentes): proxy para ε_lex en modalidad
    n = len(all_cert)
    n_samples = min(500, n * (n - 1) // 2)
    d_cert_cert = []
    for _ in range(n_samples):
        i, j = np.random.randint(0, n, size=2)
        if i != j:
            d_cert_cert.append(cosine_dist(all_cert[i], all_cert[j]))
    
    # γ_mod = media(d_cert_hedge) / p10(d_cert_cert)
    # Interpretación: d_cert_hedge DEBE ser grande, d_cert_cert DEBE ser pequeña
    mean_ch = float(np.mean(d_cert_hedge))
    p10_cc = float(np.percentile(d_cert_cert, 10))
    
    gamma_mod = mean_ch / p10_cc if p10_cc > 0 else float('inf')
    
    return {
        "gamma_mod": gamma_mod,
        "mean_d_cert_hedge": mean_ch,
        "p10_d_cert_cert": p10_cc,
        "n_val_pairs": n,
    }


def evaluate_axioms_combined(phi_modal: nn.Module, encoder, axiom_corpus: list, 
                             device: str = "cpu") -> dict:
    """
    Re-ejecuta validación de A1-A4 con φ = [φ_content; φ_modal].
    
    Concatenamos embeddings BGE (φ_content) con proyección φ_modal.
    """
    phi_modal.eval()
    
    d_para, d_contra, d_hedge = [], [], []
    d_para_modal, d_contra_modal, d_hedge_modal = [], [], []
    
    with torch.no_grad():
        for item in axiom_corpus:
            texts = [item["anchor"], item["paraphrase"], item["contradiction"], item["hedge"]]
            
            # φ_content (BGE)
            content_embs = encoder.encode(
                texts, normalize_embeddings=True, 
                show_progress_bar=False, convert_to_tensor=True
            ).to(device)
            
            # φ_modal
            modal_embs = phi_modal(content_embs)
            
            # Espacio combinado: concatenación normalizada
            # Ponderamos igualmente content y modal (podría ajustarse)
            combined = torch.cat([content_embs, modal_embs], dim=-1)
            combined = F.normalize(combined, dim=-1)
            
            # Distancias en espacio combinado
            embs_np = combined.cpu().numpy()
            d_para.append(cosine_dist(embs_np[0], embs_np[1]))
            d_contra.append(cosine_dist(embs_np[0], embs_np[2]))
            d_hedge.append(cosine_dist(embs_np[0], embs_np[3]))
            
            # Distancias SOLO en espacio modal (para medir A3 específicamente)
            modal_np = modal_embs.cpu().numpy()
            d_para_modal.append(cosine_dist(modal_np[0], modal_np[1]))
            d_contra_modal.append(cosine_dist(modal_np[0], modal_np[2]))
            d_hedge_modal.append(cosine_dist(modal_np[0], modal_np[3]))
    
    d_para = np.array(d_para)
    d_contra = np.array(d_contra)
    d_hedge = np.array(d_hedge)
    
    d_para_m = np.array(d_para_modal)
    d_contra_m = np.array(d_contra_modal)
    d_hedge_m = np.array(d_hedge_modal)
    
    # Axiomas en espacio combinado
    eps_lex = np.percentile(d_contra, 10)
    gamma_prop = np.mean(d_contra) / eps_lex if eps_lex > 0 else 0
    gamma_mod = np.mean(d_hedge) / eps_lex if eps_lex > 0 else 0
    
    a1_combined = bool(np.mean(d_para) < eps_lex)
    a2_combined = bool(gamma_prop >= 1.3)
    a3_combined = bool(gamma_mod >= 2.0)
    
    # Axiomas en espacio modal solo
    eps_lex_m = np.percentile(d_contra_m, 10)
    gamma_mod_pure = np.mean(d_hedge_m) / eps_lex_m if eps_lex_m > 0 else 0
    
    return {
        "combined_space": {
            "mean_d_para": float(np.mean(d_para)),
            "mean_d_contra": float(np.mean(d_contra)),
            "mean_d_hedge": float(np.mean(d_hedge)),
            "eps_lex": float(eps_lex),
            "gamma_prop": float(gamma_prop),
            "gamma_mod": float(gamma_mod),
            "a1": a1_combined,
            "a2": a2_combined,
            "a3": a3_combined,
        },
        "modal_space_only": {
            "mean_d_para": float(np.mean(d_para_m)),
            "mean_d_contra": float(np.mean(d_contra_m)),
            "mean_d_hedge": float(np.mean(d_hedge_m)),
            "gamma_mod_pure": float(gamma_mod_pure),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# AXIOM CORPUS (para validación final)
# ═══════════════════════════════════════════════════════════════════════════

AXIOM_CORPUS = [
    {"id": "T01", "domain": "trading", "anchor": "The market will rise sharply tomorrow due to strong earnings reports.", "paraphrase": "Prices are expected to surge tomorrow following robust corporate results.", "contradiction": "The market will decline significantly tomorrow despite earnings season.", "hedge": "The market might rise tomorrow, though there is considerable uncertainty about the magnitude."},
    {"id": "T02", "domain": "trading", "anchor": "Bitcoin's volatility is driven primarily by retail speculation.", "paraphrase": "Individual investor speculation is the main driver of Bitcoin price swings.", "contradiction": "Bitcoin's volatility stems mainly from institutional trading and macro hedging.", "hedge": "Bitcoin's volatility could be driven by retail speculation, but institutional factors may also play a significant role."},
    {"id": "T03", "domain": "trading", "anchor": "The trailing stop should be set at 1.5% for this regime.", "paraphrase": "A 1.5% trailing stop is the optimal setting in the current market conditions.", "contradiction": "The trailing stop should be widened to at least 3% in this regime.", "hedge": "A trailing stop around 1.5% seems reasonable, but the optimal value depends on factors I'm not fully certain about."},
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
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train_phi_modal(train_dataset, val_dataset, encoder, config: dict, device: str):
    """Entrena φ_modal con InfoNCE + early stopping."""
    
    # Modelo
    phi_modal = PhiModalProjection(
        input_dim=1024,
        hidden_dim=config["hidden_dim"],
        output_dim=config["projection_dim"]
    ).to(device)
    
    n_params = sum(p.numel() for p in phi_modal.parameters() if p.requires_grad)
    print(f"  Parámetros entrenables: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        phi_modal.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["max_epochs"]
    )
    
    # DataLoaders con oversampling KAIRI en train
    train_weights = train_dataset.get_sample_weights(config["kairi_oversample_weight"])
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    
    # Training loop
    history = {
        "train_loss": [],
        "val_gamma_mod": [],
        "val_mean_d_ch": [],
        "val_p10_d_cc": [],
    }
    
    best_gamma_mod = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state_dict = None
    
    print()
    print(f"  {'Época':>5s} {'Train Loss':>12s} {'γ_mod':>8s} {'d(c,h)':>8s} {'p10(c,c)':>10s} {'LR':>10s}")
    print(f"  {'─'*5} {'─'*12} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")
    
    for epoch in range(1, config["max_epochs"] + 1):
        phi_modal.train()
        epoch_losses = []
        
        for batch in train_loader:
            cert_embs = batch["certain_emb"].to(device)
            hedge_embs = batch["hedge_emb"].to(device)
            
            cert_proj = phi_modal(cert_embs)
            hedge_proj = phi_modal(hedge_embs)
            
            loss, metrics = infonce_modal_loss(
                cert_proj, hedge_proj, 
                temperature=config["temperature"]
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(phi_modal.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        mean_train_loss = np.mean(epoch_losses)
        
        # Validación
        val_metrics = evaluate_gamma_mod(phi_modal, val_loader, device)
        
        history["train_loss"].append(mean_train_loss)
        history["val_gamma_mod"].append(val_metrics["gamma_mod"])
        history["val_mean_d_ch"].append(val_metrics["mean_d_cert_hedge"])
        history["val_p10_d_cc"].append(val_metrics["p10_d_cert_cert"])
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(f"  {epoch:5d} {mean_train_loss:12.4f} "
              f"{val_metrics['gamma_mod']:8.3f} "
              f"{val_metrics['mean_d_cert_hedge']:8.4f} "
              f"{val_metrics['p10_d_cert_cert']:10.4f} "
              f"{current_lr:10.6f}")
        
        # Early stopping
        if val_metrics["gamma_mod"] > best_gamma_mod:
            best_gamma_mod = val_metrics["gamma_mod"]
            best_epoch = epoch
            patience_counter = 0
            # Guardar mejor estado
            best_state_dict = {k: v.clone().cpu() for k, v in phi_modal.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n  Early stopping en época {epoch} (mejor: época {best_epoch}, γ_mod={best_gamma_mod:.3f})")
                break
    
    # Restaurar mejor modelo
    if best_state_dict is not None:
        phi_modal.load_state_dict(best_state_dict)
    
    return phi_modal, history, best_gamma_mod, best_epoch


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Setup
    random.seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])
    torch.manual_seed(CONFIG["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["random_seed"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print()
    print("=" * 76)
    print("  PROYECTO ECHO — Fine-tuning de φ_modal")
    print("  Objetivo: resolver A3 (sensibilidad modal epistémica)")
    print("=" * 76)
    print()
    print(f"  Dispositivo: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    print("  Configuración:")
    for k, v in CONFIG.items():
        print(f"    {k:25s} {v}")
    print()
    
    # ─── Cargar encoder base ───
    print("━" * 76)
    print("  CARGA DE ENCODER BASE (BGE-large)")
    print("━" * 76)
    print()
    
    from sentence_transformers import SentenceTransformer
    print(f"  Cargando {CONFIG['base_encoder']}...", flush=True)
    encoder = SentenceTransformer(CONFIG["base_encoder"], device=device)
    # Congelar encoder
    for param in encoder.parameters():
        param.requires_grad = False
    print(f"  OK — dim={encoder.get_sentence_embedding_dimension()}, congelado")
    print()
    
    # ─── Cargar datasets ───
    print("━" * 76)
    print("  CARGA DE DATASETS")
    print("━" * 76)
    print()
    
    print("  Train dataset:")
    train_dataset = ContrastivePairsDataset(TRAIN_FILE, encoder, device)
    print(f"    {len(train_dataset)} pares")
    
    print("  Val dataset:")
    val_dataset = ContrastivePairsDataset(VAL_FILE, encoder, device)
    print(f"    {len(val_dataset)} pares")
    print()
    
    # ─── Entrenamiento ───
    print("━" * 76)
    print("  ENTRENAMIENTO DE φ_modal")
    print("━" * 76)
    
    phi_modal, history, best_gamma, best_epoch = train_phi_modal(
        train_dataset, val_dataset, encoder, CONFIG, device
    )
    
    print()
    print(f"  Mejor γ_mod en validation: {best_gamma:.3f} (época {best_epoch})")
    print()
    
    # ─── Validación final en espacio combinado ───
    print("━" * 76)
    print("  VALIDACIÓN FINAL — AXIOMAS CON φ = [φ_content; φ_modal]")
    print("━" * 76)
    print()
    
    axiom_results = evaluate_axioms_combined(
        phi_modal, encoder, AXIOM_CORPUS, device
    )
    
    combined = axiom_results["combined_space"]
    modal_only = axiom_results["modal_space_only"]
    
    print("  Espacio combinado [φ_content; φ_modal]:")
    print(f"    Media d(a, paráfrasis):    {combined['mean_d_para']:.4f}")
    print(f"    Media d(a, contradicción): {combined['mean_d_contra']:.4f}")
    print(f"    Media d(a, hedge):         {combined['mean_d_hedge']:.4f}")
    print(f"    ε_lex (p10 contra):        {combined['eps_lex']:.4f}")
    print(f"    γ_prop:                    {combined['gamma_prop']:.3f} (umbral: 1.3)")
    print(f"    γ_mod:                     {combined['gamma_mod']:.3f} (umbral: 2.0)")
    print()
    
    print("  Axiomas:")
    print(f"    A1 (invariancia léxica):   {'✓ PASS' if combined['a1'] else '✗ FAIL'}")
    print(f"    A2 (sep. proposicional):   {'✓ PASS' if combined['a2'] else '✗ FAIL'}")
    print(f"    A3 (modalidad epistémica): {'✓ PASS' if combined['a3'] else '✗ FAIL'}")
    print()
    
    print("  Solo espacio modal φ_modal (64 dim):")
    print(f"    Media d(a, paráfrasis):    {modal_only['mean_d_para']:.4f}")
    print(f"    Media d(a, contradicción): {modal_only['mean_d_contra']:.4f}")
    print(f"    Media d(a, hedge):         {modal_only['mean_d_hedge']:.4f}")
    print(f"    γ_mod puro:                {modal_only['gamma_mod_pure']:.3f}")
    print()
    
    # ─── Guardar modelo y reporte ───
    print("━" * 76)
    print("  GUARDANDO RESULTADOS")
    print("━" * 76)
    print()
    
    model_path = MODELS_DIR / "phi_modal_bge_large.pt"
    torch.save({
        "state_dict": phi_modal.state_dict(),
        "config": CONFIG,
        "best_gamma_mod_val": best_gamma,
        "best_epoch": best_epoch,
    }, model_path)
    print(f"  Modelo guardado: {model_path}")
    
    report = {
        "project": "ECHO",
        "module": "phi_modal_training",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "config": CONFIG,
        "device": device,
        "train_pairs": len(train_dataset),
        "val_pairs": len(val_dataset),
        "best_gamma_mod_val": float(best_gamma),
        "best_epoch": best_epoch,
        "training_history": history,
        "axiom_validation": axiom_results,
    }
    
    report_path = REPORTS_DIR / "phi_modal_training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Reporte: {report_path}")
    print()
    
    # ─── Decisión final ───
    print("=" * 76)
    print("  DECISIÓN FINAL")
    print("=" * 76)
    print()
    
    all_pass = combined["a1"] and combined["a2"] and combined["a3"]
    
    if all_pass:
        print("  ┌──────────────────────────────────────────────────────────────────────┐")
        print("  │  ✓ φ ES ADMISIBLE                                                    │")
        print("  │                                                                      │")
        print("  │  φ = [φ_content (BGE-large); φ_modal (64 dim fine-tuned)]           │")
        print("  │                                                                      │")
        print("  │  Todos los axiomas A1-A3 se cumplen en el espacio combinado.        │")
        print("  │  PROCEDER a H1 (descomposición de incertidumbre).                   │")
        print("  └──────────────────────────────────────────────────────────────────────┘")
    else:
        failing = []
        if not combined["a1"]: failing.append("A1")
        if not combined["a2"]: failing.append("A2")
        if not combined["a3"]: failing.append("A3")
        
        print(f"  ⚠ Axiomas que fallan: {', '.join(failing)}")
        print()
        
        if "A3" in failing:
            print(f"    γ_mod actual: {combined['gamma_mod']:.3f}")
            print(f"    γ_mod puro (solo modal): {modal_only['gamma_mod_pure']:.3f}")
            print(f"    Gap al umbral 2.0: {2.0 - combined['gamma_mod']:.3f}")
            print()
            print("    POSIBLES ACCIONES:")
            print("      1. Aumentar tamaño del dataset (más pares KAIRI-específicos)")
            print("      2. Incrementar peso de oversampling KAIRI (actualmente 3x)")
            print("      3. Ajustar arquitectura (aumentar hidden_dim o capas)")
            print("      4. Usar hard negative mining durante entrenamiento")
    
    print()


if __name__ == "__main__":
    main()