# Paper §VI — Document Index

This index file summarises the v2 paper §VI deliverables. Each
subsection lives in its own markdown file under
`cloud_removal_v2/docs/`; tables live in
`cloud_removal_v2/docs/tables/` (twin .md + .tex per table).

## Subsection files

|  §     | File                                         | Lines | Commit    | Headline |
|:-------|:---------------------------------------------|:-----:|:---------:|:---------|
| §VI-A  | `paper_v2_section_6_A_setup.md`              | 158   | `5ff1b1e` | Setup, config parity, MultiSpike-4 / ANN T=4 disclosures |
| §VI-B  | `paper_v2_section_6_B_noniid.md`             | 150   | `13a1a9e` | Source-level Dirichlet vs FLSNN label-shift — our non-IID is strictly weaker |
| §VI-C  | `paper_v2_section_6_C_fedbn.md`              | 169   | `af78a4f` | **MAIN CLAIM.** FedBN below noise floor on this task; TDBN reduces Var(γ) by 42 % (mechanism vs effect layers) |
| §VI-D  | `paper_v2_section_6_D_aggregation.md`        | 194   | `5490ed7` | AllReduce Pareto-dominates; FLSNN ranking reversal via Theorem 2 boundary case |
| §VI-E  | `paper_v2_section_6_E_ann_vs_snn.md`         | 163   | `fcc8010` | ANN +0.75 dB / 1.61× on GPU; SNN energy NOT measured, deferred to v3 |
| §VI-F  | `paper_v2_section_6_F_qualitative.md`        | 146   | `093d1bb` | Qualitative panel — thin/medium cloud OK, thick cloud fails uniformly; 6-sample ranking is anti-correlated with full test ranking |
| §VI-G  | `paper_v2_section_6_G_per_plane.md`          | 169   | `d183773` | Per-plane PSNR-std vs Var(γ); A<>B direction flip explained by P2 data atypicality |
| §VI-H  | `paper_v2_section_6_H_discussion.md`         | 167   | `55ba9c6` | 4 takeaways + 10-row L-ledger of limitations + v3 work ledger |

**Total §VI draft: 1316 lines of markdown** (excluding tables).

## Supporting infrastructure

### Tables (under `cloud_removal_v2/docs/tables/`)

|  Paper label | Source files                                        | Commit    |
|:-------------|:----------------------------------------------------|:---------:|
| Table I      | `table_main.md`, `table_main.tex`                   | `081a26b` |
| Table II     | `table_fedbn_ablation.md`, `table_fedbn_ablation.tex` | `22f9045` |
| Table III    | `table_drift.md`, `table_drift.tex`                 | `a45a762` |
| Table IV     | `table_ann_vs_snn.md`, `table_ann_vs_snn.tex`       | `fe19bbe` |
| Index        | `tables/README.md`                                  | `20b4bd2` |

### Scripts

|  Script                                | Purpose                                              | Commit    |
|:---------------------------------------|:-----------------------------------------------------|:---------:|
| `analyze_bn_drift_posthoc.py`          | Isinstance-based BN detection, produces Table III    | `17cd881` |
| `plot_results.py`                      | Loss / PSNR / SSIM vs round, per cell (v1 inherited) | —         |
| `plot_comm_efficiency.py`              | (Comm, PSNR) Pareto frontier                         | —         |
| `plot_partition_heatmap.py`            | Fig. 1 — Dirichlet partition visualisation            | —         |
| `plot_per_plane.py`                    | Per-plane PSNR-std table (§VI-G)                     | —         |
| `visualize.py`                         | Qualitative de-clouding panels (§VI-F)               | —         |

## Reproduction recipe

End-to-end regeneration of every numeric claim in §VI-A..H from
the three summary.json files and 60 plane checkpoints:

```bash
cd <repo>/Decentralized-Satellite-FL-dev-main

# Tables I, II, IV — from summary.json (no GPU needed)
python3 -c "import json; [print(k, v['PSNR_final']) for k, v in \
  json.load(open('Outputs_v2/v2a_v2a_80ep_summary.json'))['final'].items()]"

# Table III — from plane ckpts (30 seconds, CPU)
python -m cloud_removal_v2.analyze_bn_drift_posthoc \
    --ckpt_dir Outputs_v2/ckpts \
    --out      Outputs_v2/v2_drift_report.md

# §VI-A..F figures (PDFs under Outputs_v2/)
python -m cloud_removal_v2.plot_results           --run_name v2a_80ep       --output_dir ./Outputs_v2
python -m cloud_removal_v2.plot_comm_efficiency   --run_name v2a_80ep       --output_dir ./Outputs_v2
python -m cloud_removal_v2.plot_partition_heatmap --run_name v2a_80ep       --output_dir ./Outputs_v2 --data_root /root/autodl-tmp/C-CUHK
python -m cloud_removal_v2.plot_per_plane         --run_name v2a_80ep       --output_dir ./Outputs_v2
python -m cloud_removal_v2.plot_per_plane         --run_name v2a_80ep_stdbn --output_dir ./Outputs_v2
python -m cloud_removal_v2.visualize              --run_name v2a_80ep       --output_dir ./Outputs_v2 --data_root /root/autodl-tmp/C-CUHK --patch_size 256 --n_samples 6
```

## Preceding docs (the v2 research dossier — not camera-ready)

These older files capture the v2 research process. They are
retained for paper-writing reference but are **not** part of the
camera-ready paper. If a reviewer asks "why did you do X",
pointers below.

|  File                              | Lines | Role |
|:-----------------------------------|:-----:|:-----|
| `v2_comprehensive_literature.md`   | 4096  | §1-§25 literature + claim ledger + §VI plan |
| `v2_theory_and_related.md`         | 941   | FedBN / TDBN / RelaySum theory derivations |
| `v2_formal_derivations.md`         | 415   | Dirichlet-to-ζ² closed form (§D1) |
| `v2_interpretation.md`             | 265   | Tier A/B/C hypotheses for observed results |
| `v2_remaining_issues.md`           | 241   | V1–V15 issue tracker |
| `v2_results_synthesis.md`          | 219   | Early 35-ep results synthesis |
| `v2a_setup.md`                     | 146   | Environment / training / reproduction guide |
| `v1/docs/paper_section_6_draft.md` | 321   | v1 paper §VI — `[v2 TBD]` placeholders now filled |

## Things NOT in §VI (deferred to v3 or out of scope)

Explicitly disclaimed in §VI-H.2 (L-ledger) and §VI-H.5. Summary:

* Multi-seed confidence intervals → v3 (V9)
* α-sweep → v3
* RelaySum lr_scale=1.0 ablation → v3 (D9-abl)
* SGD-vs-AdamW ablation → v3 (SGD-abl)
* MDST on 42/7/1 Walker Delta → v3 (MDST)
* Per-epoch BN-drift trajectory → v3 (V3 plumbing fix)
* Cosine-similarity between planes → v3 (V3)
* SNN-vs-ANN energy ratio → v3 (E1 + Table V)
* Centralised (non-FL) oracle → v3 (V8)
* Multi-temporal cloud removal → v3 / v4

## Commit-chain integrity

All §VI commits are contiguous on branch
`claude/setup-new-output-directory-8xvx9`:

```
55ba9c6  docs(v2): paper section VI-H — Discussion, Limitations, Future Work
d183773  docs(v2): paper section VI-G — Per-plane Spread and BN-drift Mechanism
093d1bb  docs(v2): paper section VI-F — Qualitative De-clouding
fcc8010  docs(v2): paper section VI-E — ANN vs SNN backbone, Pareto, energy
5490ed7  docs(v2): paper section VI-D — Inter-Plane Aggregation Schemes
af78a4f  docs(v2): paper section VI-C — FedBN BN-Local Aggregation (MAIN CLAIM)
13a1a9e  docs(v2): paper section VI-B — Federated Partition (source-shift Dirichlet)
5ff1b1e  docs(v2): paper section VI-A — Experimental Setup
20b4bd2  docs(v2): tables/README.md — index, conventions, parity guarantees
fe19bbe  docs(v2): Table IV — ANN vs SNN backbone (FedBN + AllReduce only)
a45a762  docs(v2): Table III — cross-plane BN-affine drift (51-layer, isinstance-based)
22f9045  docs(v2): Table II — FedBN-vs-FedAvg Δ(PSNR) per scheme
081a26b  docs(v2): Table I — main 13-cell results (md + tex twin)
17cd881  feat(v2): analyze_bn_drift_posthoc.py — isinstance-based BN detection
```
