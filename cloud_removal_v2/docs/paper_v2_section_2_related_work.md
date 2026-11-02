# §II. Related Work

We position our contribution along six literature axes. Every
citation in this section has been verified against the original
arXiv / venue listing (✓ in `v2_comprehensive_literature.md`
§20.4, commit history pre-dating paper drafting). References
without arXiv identifiers are labelled explicitly; all others
carry the stored arXiv ID.

## II.A  Federated learning on satellite constellations

The closest prior art is **FLSNN** [Yang *et al.*, arXiv 2501.15995,
2025], which introduced RelaySum-based decentralised FL for
50/5/1 Walker-Star LEO constellations with Spiking CNN / Spiking
ResNet backbones on EuroSAT classification. Our paper inherits
FLSNN's algorithm, constellation topology, and convergence theorem
verbatim and extends to pixel-regression (cloud removal) with
deeper networks, explicit non-IID source partition, and
controlled backbone / normalisation ablations.

The canonical survey anchor is **Matthiesen *et al.*,
"Federated Learning in Satellite Constellations,"** IEEE Network
2023 (arXiv 2305.13602). Ground-station-PS baselines:
**Razmi *et al.*** ICC 2022 (arXiv 2111.04953) and WCL 2022
(arXiv 2202.01267). Asynchronous variants:
**AsyncFLEO** (Elmahallawy & Luo, BigData 2022, arXiv 2212.11522).
Hierarchical / high-altitude platforms: **FedHAP**
(arXiv 2205.07216). Remote-sensing FL with Sentinel-2 imagery:
**Büyüktaş, Sumbul, Demir**, IGARSS 2023 (arXiv 2306.00792).

**Our positioning.** None of the six above uses SNN; all do
classification with ANN and — with the exception of FLSNN —
ground-station PS. We extend the FL-satellite line to
**pixel-level image regression** on a two-source visual-feature
partition, with explicit ablation of the aggregation scheme
(§VI-D), BN family (§VI-C) and backbone (§VI-E).

## II.B  Non-IID FL and the batch-normalisation family

The BN-localisation chain most relevant to our Claim C16 (§VI-C)
is the four-paper lineage **SiloBN** [Andreux *et al.*, DART
2020, arXiv 2008.07424] → **FedBN** [Li *et al.*, ICLR 2021,
arXiv 2102.07623] → **HarmoFL** [Jiang *et al.*, AAAI 2022,
arXiv 2112.10775] → **FedWon** [Zhuang & Lyu, ICLR 2024,
arXiv 2306.05879]. The underlying mechanism — "external
covariate shift" across clients — is formalised in
**Du *et al.*** (arXiv 2210.03277, 2022).

Orthogonal client-drift correction methods include
**FedProx** [Li *et al.*, arXiv 1812.06127], **SCAFFOLD**
[Karimireddy *et al.*, ICML 2020, arXiv 1910.06378],
**FedDyn** [Acar *et al.*, ICLR 2021], and **FedDC**
[Gao *et al.*, CVPR 2022]. These correct weight-space drift
without modifying BN layers and are complementary to the BN
family above.

**Our positioning.** No prior work combines SNN-specific BN
(TDBN, Zheng *et al.*, AAAI 2021, arXiv 2011.05280) with FL
normalisation strategies. Claim C16 — that TDBN's
$\alpha V_{\rm th}$-shared scaling renders FedBN-style
BN-local aggregation largely redundant on our regression task
(§VI-C with supporting drift evidence in §VI-G) — is novel
relative to the BN-family literature above.

## II.C  Decentralised optimisation and sharpness-aware minimisation

Decentralised-SGD foundations span **D-PSGD** [Lian *et al.*,
NeurIPS 2017, arXiv 1705.09056], **MATCHA** [Wang & Joshi,
arXiv 1905.09435], **Cooperative SGD** [Wang & Joshi,
arXiv 1808.07576], and the unified theory of
**Koloskova, Lin, Stich, Jaggi** [ICML 2020, arXiv 2003.10422].
RelaySum specifically: **Vogels *et al.*** [NeurIPS 2021,
arXiv 2110.04175], from which FLSNN inherits Algorithm 2 and
Theorem 2 (see §IV of our paper).

The sharpness-aware-minimisation (SAM) family — **SAM**
[Foret *et al.*, ICLR 2021, arXiv 2010.01412], **ASAM**
[Kwon *et al.*, arXiv 2102.11600], **ESAM** [Du *et al.*,
arXiv 2110.03141] — has been brought to FL by **FedSAM**
[Qu *et al.*, ICML 2022, arXiv 2206.02618] and
**Caldarola *et al.*** [ECCV 2022, arXiv 2203.11834].

**Our positioning.** Our observed scheme-ranking reversal
(§VI-D: AllReduce Pareto-dominates in our regime vs FLSNN
Fig. 5's RelaySum-leads classification result) is a new
empirical observation in the small-$\zeta^2$ regime that
existing decentralised-SGD bounds (Koloskova 2020, Vogels 2021)
address only asymptotically. Our §IV Corollary 1 formalises
the connection between $\zeta^2$ shrinkage and scheme-rank
collapse. We do *not* claim theoretical novelty for weight
perturbation; SAM methods are cited for completeness.

## II.D  Spiking neural networks and federated SNN

Surrogate-gradient foundations: **STBP** [Wu *et al.*,
arXiv 1706.02609], **STBP-tdBN** (TDBN) [Zheng *et al.*, AAAI
2021, arXiv 2011.05280], **TET** [Deng *et al.*, ICLR 2022,
arXiv 2202.11946], and the theoretical treatment of
**Zenke & Vogels** [Neural Computation 2021]. The `MultiSpike4`
5-level quantisation used in our VLIFNet backbone is a variant
of TDBN's training scheme; see §VI-A.4 for our specific design
and the explicit disclosure that it is not binary (§VI-E.4.2).

Federated SNN literature:
**Venkatesha *et al.*** [IEEE TSP 2021, arXiv 2106.06579] —
the first FL-SNN benchmark on standard CNN datasets;
**Skatchkovsky *et al.*** [ICASSP 2020, arXiv 1910.09594] — the
earliest FL-SNN study; **Yang *et al.*** [arXiv 2309.09219,
2023] — compressed-gradient FL-SNN; **FLSNN** [Yang 2025,
arXiv 2501.15995] — satellite-specific FL-SNN.

**Our positioning.** We extend the FL-SNN line from
classification to pixel-level regression on satellite
constellations — a regime not covered by the four above.

## II.E  Federated image restoration

The closest peer-reviewed pixel-regression FL systems are:

* **FedMRI** [Feng *et al.*, IEEE TMI 2022, arXiv 2112.05752] —
  shared encoder + client-specific decoders for MR reconstruction.
* **FedFTN** [Zhou *et al.*, MedIA 2023, arXiv 2304.00570] —
  personalised positron-emission-tomography denoising.
* **FedFDD** [Chen *et al.*, MIDL 2024, OpenReview Zg0mfl10o2] —
  DCT-frequency-split FL for low-dose CT.
* **FedNS** [Li *et al.*, arXiv 2409.02189, 2024] — noise-sifting
  aggregation for image denoising.

**Centralised cloud-removal SOTA** (cited in §VI-F as PSNR
reference, not in §II as positioning): DC4CR
[arXiv 2504.14785] 26.29 dB on CR1; DE-MemoryNet [Sui *et al.*,
TGRS 2024, arXiv 2401.15105] 26.18 dB; CVAE [Ding *et al.*,
ACCV 2022] 24.25 dB; SpA-GAN [Pan *et al.*, arXiv 2009.13015]
21.00 dB.

**Our positioning.** To our knowledge, no peer-reviewed prior
work has performed federated cloud removal with (i) orbit-plane
structured satellites and (ii) SNN backbones. The four systems
above treat other pixel-regression domains (MRI / PET / CT /
photographic denoising) with ANN backbones under different
partition schemes. Our paper is to the best of our knowledge
the first federated cloud-removal study with a controlled
TDBN / ANN-vs-SNN ablation. This positioning should be
re-verified against the most recent literature before final
submission; the field moves fast.

## II.F  Energy analysis methodology

**Horowitz, "Computing's Energy Problem,"** ISSCC 2014 — the
canonical 45-nm CMOS per-operation energy table (4.6 pJ/MAC,
0.9 pJ/AC for binary spikes) inherited by FLSNN §VI-B and by
our §VI-E. **Rueckauer *et al.***, Frontiers in Neuroscience
2017, provided the per-layer SNN-vs-ANN energy methodology
that FLSNN adapts. We disclose in §VI-E that the 0.9 pJ/AC
figure is a *binary*-spike bound; our MultiSpike-4 5-level
encoding requires a looser per-operation estimate, and we
defer a measured energy ratio to v3.

## II.G  What this paper does and does not claim relative to II.A–II.F

Explicit positioning, to pre-empt reviewer overreach:

1. **We do not claim** a new RelaySum algorithm. We inherit
   the FLSNN / Vogels 2021 implementation verbatim (§IV.B)
   and bound its constants by our Proposition 1 (§IV.C).
2. **We do not claim** a new TDBN algorithm. We use
   `spikingjelly.ThresholdDependentBatchNorm2d` [Zheng 2021]
   as-is (§VI-A.4) and apply it to a new FL setting.
3. **We do not claim** to refute FedBN's reported accuracy
   gains on classification-with-style-shift. We report a null
   effect in our regime (source-shift + pixel regression),
   and §VI-C argues this is consistent with — not a
   refutation of — the $\zeta^2$-dependent behaviour predicted
   by Corollary 1.
4. **We do not claim** an energy ratio for SNN vs ANN. §VI-E
   discloses that our MultiSpike-4 quantisation invalidates
   the 0.9 pJ/AC binary-spike bound and defers a proper
   measurement to v3.
5. **We do claim** the empirical observation (§VI-D) that
   AllReduce is Pareto-optimal on our specific task /
   topology / non-IID setting, the FedBN-redundancy result
   of §VI-C, and the Proposition 1 / Corollary 1 pair in §IV.
