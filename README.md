# ECHO: Pre-Generative Epistemic Calibration for Trustworthy AI

> Teaching AI to think before it speaks — a mathematical framework for separating and calibrating epistemic uncertainty in autoregressive generative models, with pre-generative metacognitive temperature modulation.

**Authors:** Kent Valera Chirinos 
**Status:** Phase 2 Complete — ECHO Not Refuted (R9.3)  
**Version:** 3.2 (April 2026)

---

## What is ECHO?

Current language models generate every token by optimizing next-token likelihood with a fixed temperature. This process is **epistemically blind** — it does not modulate behavior based on what the model *knows* vs. what it *interpolates plausibly*.

ECHO proposes a fundamental change: insert an epistemic uncertainty signal **inside** the generation process, creating a metacognitive feedback loop.

ECHO seeks to address a critical question for the next generation of generative AI: How can a system honestly estimate and communicate its own uncertainty when its architecture was not designed to represent epistemic uncertainty separately from random uncertainty?

```
Standard:     y_t ~ softmax(logits / T₀)           [fixed temperature]
ECHO:         y_t ~ softmax(logits / T(e_t))        [adaptive temperature]
```

Where `e_t = Û_ep(context)` is an epistemic uncertainty estimate computed **before** generating the next token, and `T(e_t)` modulates the temperature: more uncertainty → higher temperature → more cautious generation.

## Key Results

| Component | Status | Key Metric |
|-----------|--------|------------|
| φ (Semantic Projection) | ✅ Validated | A1-A4 axioms, α=1.1929 |
| H1 (Uncertainty Decomposition) | ✅ Validated | U_ep + U_al = H_total (0.00% error) |
| H3 (Epistemic Probe) | ⚠️ Partial | r=0.483 LODO, r=0.75 tails, OOD ✓ |
| T(e_t) (Temperature Control) | ✅ Validated | V1-V5, ACI convergent |
| L_meta (Optimization Functional) | ✅ Validated | V1-V4, equilibrium 56/18/26% |
| Metacognitive Metrics | ✅ Validated | MES=5.5pp, CAR=3.05, EIG=0.095 bits |
| R9.3 (Refutation) | ✅ Not Triggered | ECHO not refuted |
| H4 (Production Validation) | 🔜 Pending | Requires GPU + real outcomes |

### Metacognitive Efficiency

When the system abstains from the 20% most uncertain predictions:

- **Accuracy improves** from 81.0% to 86.5% (+5.5pp)
- **Errors concentrate** 3x more in abstentions than executions (CAR=3.05)
- **Signal is informative**: 0.095 bits of mutual information between e_t and outcomes
- **Generalizes across domains**: CAR ≥ 2.0 in 5/5 domains tested

## Scientific Contributions

1. **Axiomatization of φ** with verifiable validation protocol (4 axioms, empirical thresholds)
2. **Gaussian covariance decomposition** (Valera 2026) for exact U_ep: error = 0.00%
3. **Pre-generative temperature modulation** T(e_t) — original contribution, no precedent in literature
4. **L_meta trilateral functional** with generation quality, calibration, and abstention cost
5. **"U_ep is sensitivity, not state"** — epistemic uncertainty in frozen transformers lives in the mapping's fragility, not in activation values
6. **Geometric-lexical probe** combining Consensus_epistemic in φ_modal subspace with hedge features and isotonic calibration
7. **5-iteration H3 probe development** with documented misalignment diagnoses and corrections — reproducible methodology

## Project Structure

```
ECHO/
├── docs/
│   ├── ECHO_marco_matematico_v3.md          # Mathematical framework (ES, 1400+ lines)
│   ├── ECHO_mathematical_framework_v3_EN.md # English translation
│   ├── ECHO_mathematischer_rahmen_v3_DE.md  # German translation
│   ├── ECHO_Phase0_Milestone_Report.docx    # Phase 0 completion report
│   └── ECHO_Section2_Metacognitive_Formalization.md
│
├── phi_validation/                          # All experimental scripts
│   ├── ECHO_phi_validation_protocol.py      # Phase 0: φ axiom validation
│   ├── phi_validation_recalibration.py      # Encoder selection (BGE vs E5 vs GTE)
│   ├── phi_validation_gaps.py              # Gap analysis and stability
│   ├── phi_gap1_resolution.py              # Stability↔U_ep correlation
│   ├── prepare_contrastive_dataset.py       # CommitmentBank + MNLI + synthetic
│   ├── phi_modal_training.py               # φ_modal InfoNCE fine-tuning
│   ├── phi_alpha_calibration.py            # α grid search (optimal: 1.1929)
│   ├── h1_uncertainty_decomposition.py      # H1: Gaussian covariance decomposition
│   ├── expand_probe_corpus.py              # Corpus expansion: 56 → 279 samples
│   ├── sprint2a_v5_corrected.py            # H3 probe v5 (5 corrections)
│   ├── sprint2b_temperature_control.py      # T(e_t) verification V1-V5
│   ├── sprint2c_lmeta_functional.py        # L_meta trilateral equilibrium
│   ├── sprint2d_metrics.py                 # MES/CAR/EIG with bootstrap CIs
│   └── reports/                            # JSON reports for all experiments
│
├── data/
│   ├── probe_corpus_expanded.json          # 279 texts, 5 domains
│   ├── contrastive_pairs/                  # φ_modal training data (1,406 pairs)
│   └── calibration_logs/                   # H4 logging structure
│
├── models/
│   ├── phi_modal_bge_large.pt              # Trained φ_modal + α=1.1929
│   └── echo_probe_h3.json                 # Probe weights (Ridge + PCA)
│
└── echo_env/                               # Python virtual environment
```

## Mathematical Framework

The framework is structured as 4 sub-hypotheses:

- **H1 — Decomposability:** Total uncertainty H decomposes exactly into H = U_al + U_ep via Gaussian covariance decomposition
- **H2 — Context Sensitivity:** Miscalibration has exploitable structure conditioned on context features
- **H3 — Architectural Signal:** Epistemic uncertainty is extractable from internal activations in single-pass
- **H4 — KAIRI Testability:** Calibrated U_ep improves sequential decision-making in trading

### The Metacognitive Loop (Section 9)

```
Input → φ(y) → Ensemble → Û_ep → e_t → T(e_t) → π_meta → Output
                                                      ↓
                                              Confidence signal
                                                      ↓
                                                  Outcome
                                                      ↓
                                              L_meta backprop
```

**Temperature Control Law:**
```
T(e) = T₀ · (1 + β · σ(κ(e - τ)))
```
Properties: strictly monotonic, bounded, controllable sensitivity, minimal parameterization.

**Optimization Functional:**
```
L_meta = L_gen + λ₁ · L_cal + λ₂ · L_abs
```
Where L_gen preserves generation quality, L_cal ensures calibration, and L_abs penalizes excessive abstention.

## Key Finding: Sensitivity, Not State

> "Epistemic uncertainty in frozen transformers does not reside in activation values (state) but in the mapping's sensitivity (geometry)."

This was discovered through 5 iterations of probe development (v1-v5), each with documented failures and corrections:

| Version | Approach | Result | Root Cause |
|---------|----------|--------|------------|
| v1 | Static CLS, N=56 | r=1.000 (memorization) | D/N = 91x |
| v2 | PCA + Ridge + LOO | R² = -0.128 | N too small |
| v3 | Layers 12-20, MLP, N=279 | r = -0.098 | Static activations ≠ U_ep |
| v4 | KL attention + norms | r = 0.266 | Proxy ≠ Jacobian |
| v5 | JVP + Consensus + hedges | r = 0.483 ✓ | 5 corrections applied |

## Requirements

### Current (Phase 0-2)
- Python 3.10+
- PyTorch 2.0+
- sentence-transformers
- scikit-learn, scipy, numpy
- transformers (HuggingFace)
- ~8GB RAM (CPU-only, no GPU required)

### Future (H4 — Production Validation)
- GPU ≥24GB VRAM (RTX 4090 or A100)
- vLLM with LogitsProcessor support
- Llama-3.1-8B-Instruct or Mistral-7B
- PostgreSQL (KAIRI database)
- ≥2,000 trades with binary outcomes (TP/SL)

## Setup

```bash
# Create environment
python -m venv echo_env

# Activate
# Windows:
echo_env\Scripts\activate
# Linux/Mac:
source echo_env/bin/activate

# Install dependencies
pip install torch sentence-transformers scikit-learn scipy transformers

# Run Phase 0 validation
python phi_validation/ECHO_phi_validation_protocol.py

# Run full pipeline (Phase 0-2)
python phi_validation/phi_modal_training.py
python phi_validation/phi_alpha_calibration.py
python phi_validation/h1_uncertainty_decomposition.py
python phi_validation/sprint2a_v5_corrected.py
python phi_validation/sprint2b_temperature_control.py
python phi_validation/sprint2c_lmeta_functional.py
python phi_validation/sprint2d_metrics.py
```

## Refutation Conditions

ECHO is designed to be **falsifiable**. The following conditions would refute the framework:

| Condition | Description | Status |
|-----------|-------------|--------|
| R9.1 | Probe fails C4.1-C4.3 | Partial (C4.3 pending real data) |
| R9.2 | Loop Jacobian ≥ 1 in >10% inputs | ✅ |J|=0 (ensemble) |
| R9.3 | MES<5 AND CAR<2 AND EIG<0.05 | ✅ All three pass |
| R9.4 | L_meta does not converge | ✅ 79.5% reduction |
| R9.5 | τ(ACI) oscillates >0.3 | ✅ std=0.011 |

## Roadmap to Full Validation (H4)

1. **Dataset:** Extract ≥2,000 KAIRI trades with binary outcomes
2. **Model:** Deploy Llama-3.1-8B with eager attention + hidden states
3. **Probe Migration:** Adapt features to generative model prefill (layers 20-28)
4. **Shadow Mode:** Log e_t without intervening in decisions for 30 days
5. **Backtest:** Compare ECHO vs. baseline over 6 months
6. **Production:** Sustained ≥60 days without triggering R9.3
7. **Validation Criteria:** C4.3 ≥ 6pp with p<0.05 (McNemar), ΔSharpe ≥ 0.1

## Citation

If you use this work, please cite:

```
@misc{echo2026,
  title={ECHO: Epistemic Calibration for Honest Output — A Mathematical Framework 
         for Pre-Generative Metacognitive Temperature Modulation},
  author={Valera Chirinos, Kent},
  year={2026},
  note={Internal technical report, v3.2}
}
```

## License

This project is private research. Contact the authors for collaboration inquiries.

---

*"The system behaves as if it knew what it does not know — with explicit limits, statistical guarantees, and real decision utility."*
