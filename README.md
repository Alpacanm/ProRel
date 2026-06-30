# ProRel: Prototype-Guided Reliable Transfer for Cross-Domain Recommendation

Official implementation of **ProRel**, a prototype-guided reliable transfer framework for cross-domain recommendation.


## Abstract

Cross-domain recommendation (CDR) mitigates target-domain sparsity by transferring preference signals through overlapping users. However, shared users establish cross-domain connectivity, not guaranteed transferability. Overlap-based bridging indicates that two domains are connected, but it does not specify whether the carried signals are trustworthy for target-domain recommendation. Reliability gaps appear at two stages where overlap-induced connectivity is consumed. During graph propagation, uniformly treating observed edges can amplify weakly supported or domain-specific user-item relations. During representation fusion, transferred representations may retain redundant or domain-conflicting factors and increase negative transfer.

We propose ProRel, a Prototype-guided Reliable transfer framework that applies reliability criteria at these two stages. For graph propagation, Prototype-Guided Reliable Graph Learning (PGRL) estimates edge reliability by measuring the consistency between local user-item relations and prototype-level preference regularities. PGRL assigns larger propagation weights to edges with stronger node-prototype agreement, so unreliable relations are down-weighted before their signals spread through the shared graph.

For representation fusion, Reliability-Aware Selective Fusion (RASF) targets negative transfer after reliable propagation. RASF preserves recommendation-relevant user correspondence across shared, domain-specific, and fused representations, while suppressing redundant cross-domain dependence with HSIC-based regularization. By doing so, RASF reduces the influence of source-specific or conflicting factors before final prediction. Experiments on four Amazon CDR tasks show that ProRel consistently outperforms competitive baselines, achieving an average relative improvement of 7.03% and up to 12.20% over the strongest baseline across HR@10 and NDCG@10.

## Environment

- Python 3.9.0
- PyTorch 1.12.0
- NumPy 1.24.3
- SciPy 1.11.1

## Datasets

We use the datasets provided by [DisenCDR](https://github.com/cjx96/DisenCDR).

Place the processed data under `dataset/` with the expected domain-pair folder structure, for example:

```text
dataset/
  electronic_cloth/
  cloth_electronic/
  sport_phone/
  phone_sport/
```

## Running

Run commands from the `src/` directory. The shared hyperparameter across the four Amazon tasks is `lambda1=0.01`. The item prototype count is controlled by `k_item` for both domains.

Amazon Elec & Cloth:

```shell
CUDA_VISIBLE_DEVICES=1 python train_rec.py --dataset electronic_cloth --lambda1 0.01 --lambda2 0.6 --lambda_regularization 0.6 --k_user 4 --k_item 16
```

Amazon Sport & Phone:

```shell
CUDA_VISIBLE_DEVICES=0 python train_rec.py --dataset sport_phone --lambda1 0.01 --lambda2 0.4 --lambda_regularization 0.5 --k_user 4 --k_item 16
```

Amazon Sport & Cloth:

```shell
CUDA_VISIBLE_DEVICES=2 python train_rec.py --dataset sport_cloth --lambda1 0.01 --lambda2 0.6 --lambda_regularization 0.6 --k_user 16 --k_item 16
```

Amazon Elec & Phone:

```shell
CUDA_VISIBLE_DEVICES=2 python train_rec.py --dataset electronic_phone --lambda1 0.01 --lambda2 0.6 --lambda_regularization 0.6 --k_user 4 --k_item 32
```

## Main Arguments

- `--model_name`: model name, default `ProRel`
- `--use_prototype_denoise`: enable PGRL
- `--edge_reliability_mode`: edge reliability setting, one of `learned`, `uniform`, or `shuffle`
- `--k_user`: number of user prototypes
- `--k_item`: number of item prototypes for each domain
- `--lambda_regularization`: weight for HSIC-based reliability regularization in RASF
- `--fusion_mode`: user representation fusion strategy, one of `ours`, `mean`, or `sum`

## Acknowledgements

This repository builds on resources from:

- [LightGCL](https://github.com/HKUDS/LightGCL)
- [DisenCDR](https://github.com/cjx96/DisenCDR)
- [ETL](https://github.com/xuChenSJTU/ETL-master)

We thank the authors for making their code and datasets publicly available.
