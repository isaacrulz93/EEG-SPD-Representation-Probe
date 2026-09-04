# Literature Overlap Audit: Baseline-Referenced SPD Trajectory V0

Audit date: 2026-09-04. Only primary papers/proceedings and author repositories
were used for the scientific comparison.

| method | primary source | input object | time representation | SPD preserved until where | domain handling | conditional correspondence source | unseen-domain mechanism | overlap with this branch | remaining distinction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAtt | [Pan et al., 2022](https://arxiv.org/abs/2210.01986) | learned EEG patch covariances/SPD features | manifold attention over spatiotemporal patches | through manifold attention, then readout | no pre-cue subject reference in the method definition | supervised task labels | trained network representation | SPD temporal modeling | this branch first tests fixed C0-relative descriptors and zero-label semantic assignment, not attention |
| TSMNet / SPDDSMBN | [Kobler et al., 2022](https://arxiv.org/abs/2206.01323) | covariance-pooled SPD features | short labeled EEG segments; not a within-trial five-state path | SPD BiMap/ReEig and domain-specific SPD momentum normalization, then tangent readout | separate running Fréchet mean/variance per source/target domain | supervised source labels; unlabeled target domain statistics | multi-source/multi-target UDA using target running statistics | SPD congruence normalization and subject/session transfer | C0 is a per-trial pre-cue physical-time reference; primary LOSO is fully inductive and tests correspondence before domain normalization |
| Tensor-CSPNet | [Ju and Guan, 2022](https://arxiv.org/abs/2202.02472) | time-frequency tensor of window covariances | temporal convolution across fixed time-frequency windows | BiMap/ReEig stages; LogEig before Euclidean temporal/readout layers | holdout evaluation, no target-specific canonicalizer | supervised labels | shared learned time-frequency representation | ordered local covariance windows and BiMap | no pre-cue C0-relative path, no zero-label target semantic permutation/action gate |
| Graph-CSPNet | [Ju and Guan, 2023](https://arxiv.org/abs/2211.02641) | graph nodes are time-frequency SPD covariances | graph edges encode selected forward time/frequency relations | graph BiMap layers, then log map at identity | graph construction can be training-fold/individual specific | supervised labels and training-set lattice geometry | learned graph classifier | local SPD states and explicit time relations | no pre-cue baseline whitening or target-label-free class/action correspondence test |
| GeoDynamics | [Dan et al., 2026](https://arxiv.org/abs/2601.13570) | dynamic functional-connectivity SPD matrices | manifold-aware recurrent/state-space trajectory | latent transitions remain manifold-aware before task heads | dataset/task modeling, not the frozen EEG LOSO C0 contract | supervised task/disease labels | trained geometric state-space model | direct SPD trajectory/recurrent overlap | fMRI/FC and general action sequences; does not test trial-specific pre-cue partial canonicalization or zero-label target naming |
| SPD statistical recurrent model | [Chakraborty et al., NeurIPS 2018](https://papers.nips.cc/paper/2018/hash/7070f9088e456682f0f84f815ebda761-Abstract.html) | ordered manifold-valued observations | intrinsic statistical recurrent unit | recurrent state is defined on SPD/manifold objects | no EEG subject-domain correspondence mechanism | supervised/sequence objective of each application | shared recurrent parameters | establishes prior SPD recurrent networks | Phase 1, if unlocked, is only a small Euclidean GRU on audited baseline-relative latent logs and makes no novelty claim about SPD recurrence |
| covariance trajectory analysis | [Dai et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7164686/) and [Su et al., 2015](https://arxiv.org/abs/1503.06699) | sliding-window covariance/SPD trajectories | continuous/discrete trajectories, registration, TSRVF/rate invariance | trajectory geometry remains Riemannian | time-warp/rate nuisance, not EEG subject semantic permutation | known task labels for evaluation | trajectory registration/kernel/dimension reduction | covariance trajectories and nuisance removal | this branch uses cue-locked nonoverlapping EEG windows and asks whether C0-relative geometry identifies unseen-subject class correspondence |
| conditional-alignment DG | [Li et al., ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.html) and [EEG-DG](https://arxiv.org/abs/2311.05415) | learned Euclidean features | architecture-dependent, not intrinsically an SPD path | generally not SPD-preserving | aligns labeled source joint/class-conditional distributions for unseen-domain DG | source labels define conditions | source-domain invariant representation | same downstream motivation: preserve semantics across subjects | current Phase 0 performs no distribution alignment and tests whether a fixed observable makes target semantics identifiable at all |
| domain-specific/conditional normalization | [Chang et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_Domain-Specific_Batch_Normalization_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf) and [MetaNorm, ICLR 2021](https://openreview.net/forum?id=9z_dNsC4B5t) | network activations plus domain/support-set statistics | not specifically temporal | Euclidean in the cited general methods | per-domain BN or support-conditioned predicted moments | pseudo-labels, domain identity, or support context | target statistics or a learned moment hypernetwork | motivates a later context-to-canonicalizer | these methods assume/use contextual information; this branch first determines whether unlabeled C0-relative context contains class-correspondence information |

## Blocking-overlap decision

`NO_BLOCKING_OVERLAP`.

Prior work clearly covers SPD sequences, covariance trajectories, manifold
attention/recurrence, SPD domain normalization, and conditional domain
alignment. None of the audited primary sources combines all of the following:

- a trial-specific cue-pre baseline covariance C0;
- fixed C0-relative post-cue local SPD states with exact and partial invariant
  controls;
- fully inductive cross-subject linear headroom gates; and
- target-label-free clustering plus semantic-permutation/common-action gates.

Therefore implementation is not blocked. This conclusion is narrow: it is a
novelty candidate for the frozen diagnostic bridge, not a claim that
baseline-relative covariance, SPD trajectories, or recurrent SPD modeling are
new in general.
