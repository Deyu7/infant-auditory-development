# Computational Modeling of Infant Auditory Development

A computational study of how structured auditory representations emerge from exposure to
naturalistic acoustic environments. Pretrained audio embeddings (AST) are probed for
auditory skills, then decomposed with **Bayesian Non-negative Matrix Factorization (BNMF)**
to interpret the latent "sound components" that the model has internalized.

This is the capstone project for the **MS in Computational Social Science (CSS)** at
**UC San Diego**, by **Deyu Wang**, supervised by **Prof. Meenakshi Khosla**.

The work is inspired by Orhan et al. (2024, *Science*), *"Grounded language acquisition
through the eyes and ears of a single child,"* which showed that deep networks can acquire
structured representations from the naturalistic sensory stream of a single developing child.
Because the original SAYCam audio requires IRB approval and Databrary access, this project
uses **AudioCaps** (sourced from AudioSet / YouTube) as a publicly accessible proxy for
naturalistic auditory input.

---

## Research questions

1. **RQ1 — Capability.** Can deep neural networks develop basic auditory skills (speech
   detection, sound categorization) from naturalistic audio without explicit linguistic labels?
2. **RQ2 — Input sensitivity.** How does the composition of training sounds
   (speech-dominant vs. balanced vs. non-speech-dominant) shape learned representations?
3. **RQ3 — Developmental trajectory.** Do certain auditory capabilities emerge earlier than
   others — analogous to developmental milestones in infants?
4. **RQ4 — Alignment.** Do model representations align with known findings from infant
   auditory perception research (e.g., early speech sensitivity)?

---

## Data and methods

**Dataset.** AudioCaps test split (`AudioLLMs/audiocaps_test`, HuggingFace): 1,442 unique
10-second clips (16 kHz, mono) with human-written captions.

**Weak labels.** AudioCaps has no categorical sound labels, so they are derived from the
captions via keyword matching across five categories — *speech, music, nature, environment,
animal* — plus a binary *speech / non-speech* label.

**Feature extraction.** Three acoustic representations are compared:

| Feature | Dim | Description |
| --- | --- | --- |
| MFCC | 80 | 40 MFCCs, mean + std pooled — classical low-level fingerprint |
| Log-mel | 128 | 64 mel filter banks, mean + std pooled |
| **AST embeddings** | 768 | Hidden states from `MIT/ast-finetuned-audioset-10-10-0.4593` (Gong et al., 2021), mean-pooled |

**Evaluation.** An L2-regularized logistic-regression probe (`C=1.0`, standard scaling) is
trained with 5-fold stratified cross-validation on (1) binary speech detection and
(2) 5-class categorization.

**Interpretation (BNMF).** The AST embedding matrix is factorized with Bayesian NMF.
Model order is selected by BIC (K = 10), and the resulting components are characterized by
category purity, co-activation, weight-space similarity, TF-IDF caption keywords, and t-SNE
activation maps.

---

## Key findings

**Pretrained AST embeddings dominate hand-crafted features.** On the held-out partition the
AST probe reaches ~85% on speech detection and ~72% on 5-class categorization, well above
MFCC and log-mel baselines.

| Task | MFCC | Log-mel | AST |
| --- | --- | --- | --- |
| Speech vs. non-speech (acc) | 0.69 | 0.68 | **0.85** |
| 5-class categorization (acc) | 0.59 | 0.56 | **0.72** |

**Speech sensitivity emerges first.** Speech detection exceeds 83% accuracy with only 5% of
the training data (~57 clips), while fine-grained categorization needs far more data to
plateau — echoing the early specialization for speech seen in infant auditory development.

**Balanced acoustic environments generalize best.** Probes trained on a balanced
speech/non-speech mix categorize sounds better than either speech-dominant or
non-speech-dominant regimes.

**BNMF discovers interpretable sub-structure.** The 10 components subdivide the coarse
5-category labels into meaningful sound types — e.g. `environment` splits into Railway,
Machinery, Impact and Vehicles; `speech` splits into clean female speech, crowd/children
noise, and infant/animal vocalisations; `nature` splits into Water/Rain and Weather/Waves.

| Component | Dominant category | Purity | Characteristic words |
| --- | --- | --- | --- |
| C2 Machinery | environment | 94% | engine, revving, motor, idle |
| C4 Crowd noise | speech | 90% | crowd, cheer, children, applauding |
| C7 Female speech | speech | 88% | (clean speech cluster) |
| C1 Railway | environment | — | train, horn, railroad, tracks |
| C3 Water/Rain | nature | — | rain, water, thunder, splashing |

---

## Repository structure

```
.
├── README.md                                   # this file
├── requirements.txt                            # Python dependencies
├── CITATION.cff                                # how to cite this work
├── .gitignore
├── docs/
│   ├── 01_Complete_Report_with_BNMF_Interpretation.docx
│   └── 02_BNMF_Interpretation_Addendum.docx
├── notebooks/
│   ├── capstone_bnmf_analysis.ipynb            # extended BNMF component analysis
│   └── capstone_notebook_previous_context.ipynb# original feature/probe pipeline
├── figures/                                    # exported analysis figures (PNG)
├── tables/                                     # exported result tables (CSV)
└── context_previous_material/                  # earlier report, for background
```

> **Note on data and cached outputs.** This repository contains the analysis notebooks,
> figures, and result tables. The raw audio, extracted `data/features/`, and cached
> `results/bnmf/` arrays are **not** included (they are large and partly derived from
> third-party data). The notebooks document how each artifact is produced; see *Reproducing*
> below.

---

## Reproducing the analysis

```bash
# 1. Create an environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Download the dataset (AudioCaps test split) from HuggingFace
#    AudioLLMs/audiocaps_test  — 1,442 clips with captions

# 3. Run the pipeline notebook to extract features and probe representations
jupyter lab notebooks/capstone_notebook_previous_context.ipynb

# 4. Run the BNMF component-interpretation notebook
jupyter lab notebooks/capstone_bnmf_analysis.ipynb
```

The notebooks expect a project layout with `data/`, `data/features/`, and `results/bnmf/`
directories at the repository root; they create `results/figures/` as needed.

---

## Citation

If you use this work, please cite it (see `CITATION.cff`):

> Wang, D. (2026). *Computational Modeling of Infant Auditory Development.*
> Capstone project, MS in Computational Social Science, UC San Diego.
> Supervised by M. Khosla.

### References

- Orhan, A. E., et al. (2024). Grounded language acquisition through the eyes and ears of a
  single child. *Science*.
- Gong, Y., Chung, Y.-A., & Glass, J. (2021). AST: Audio Spectrogram Transformer.
  *Interspeech*.
- Kim, C. D., et al. (2019). AudioCaps: Generating Captions for Audios in The Wild. *NAACL*.

---

## Acknowledgements

Developed under the guidance of **Prof. Meenakshi Khosla** (UC San Diego). The Audio
Spectrogram Transformer weights are from MIT (`ast-finetuned-audioset-10-10-0.4593`), and the
dataset is AudioCaps, built on AudioSet.

---

## License

No license has been chosen yet. Until a license is added, this work is **"all rights
reserved"** by default — others may view the repository but do not have permission to reuse,
modify, or redistribute it. A license (e.g. MIT for code, CC BY 4.0 for the written report)
can be added later by dropping a `LICENSE` file in the repository root.
