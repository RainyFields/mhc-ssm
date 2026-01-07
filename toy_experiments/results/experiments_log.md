# mHC-SSM Experiment Log

| Date | ID | Vocab | Layers | Steps | Std Loss | mHC Loss | Std Acc | mHC Acc | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-01-06 01:39 | exp_20260106_005325 | 100 | 8 | 3000 | 4.607 | 4.607 | 0.013 | 0.005 | Synthetic Copy Task. Plot: [exp_20260106_005325_plot.png](exp_20260106_005325_plot.png) |

<details><summary>Run exp_20260106_005325 Details</summary>

**Command:** `CUDA_VISIBLE_DEVICES=7 python mhc_mamba2_test.py`

**Config:**
```json
{'D_MODEL': 256, 'N_LAYERS': 8, 'VOCAB': 100, 'STREAMS': 4, 'BS': 32, 'SEQ': 128, 'STEPS': 3000, 'LR': 0.001}
```
</details>
| 2026-01-06 02:58 | exp_20260106_024701 | 16 | 8 | 3000 | 2.775 | 2.770 | 0.067 | 0.081 | Synthetic Copy Task. Plot: [exp_20260106_024701_plot.png](exp_20260106_024701_plot.png) |

<details><summary>Run exp_20260106_024701 Details</summary>

**Command:** `CUDA_VISIBLE_DEVICES=7 python mhc_mamba2_test.py`

**Config:**
```json
{'D_MODEL': 256, 'N_LAYERS': 8, 'VOCAB': 16, 'STREAMS': 4, 'BS': 32, 'SEQ': 32, 'STEPS': 3000, 'LR': 0.005}
```
</details>
| exp_20260106_040256 | 1.776 | 1.742 | 0.393 | 0.407 | Fix: PreNorm+StickyInit |
| exp_20260106_040610 | 1.728 | 1.526 | 0.419 | 0.494 | Fix: PreNorm+StickyInit |
| exp_20260106_185041 | 0.174 | 0.145 | 0.946 | 0.952 | Fix: PreNorm+StickyInit |
| exp_20260106_231157 | 1.418 | 1.423 | 0.506 | 0.504 | Fix: PreNorm+StickyInit |
| exp_20260106_195704 | 2.770 | 0.999 | 0.199 | 0.689 | Fix: PreNorm+StickyInit |
| exp_20260106_231510 | 1.950 | 0.202 | 0.355 | 0.938 | Fix: PreNorm+StickyInit |
| H1_SEQ128 | copy | 0.6002 | 0.5848 | 16637 | 16522 | L=4, D=128, SEQ=128, TASK=copy |
| H1_SEQ256 | copy | 0.3319 | 0.3063 | 16119 | 16164 | L=4, D=128, SEQ=256, TASK=copy |
| H2_VOC1024 | copy | 0.3189 | 0.0032 | 16247 | 16085 | L=4, D=128, SEQ=256, TASK=copy |
| H5_MQAR | mqar | 0.1562 | 0.0625 | 15534 | 15340 | L=4, D=128, SEQ=256, TASK=mqar |
| H5_FUZZY | fuzzy_recall | 0.0312 | 0.0000 | 15403 | 15226 | L=4, D=128, SEQ=256, TASK=fuzzy_recall |
| H2_VOC4096 | copy | 0.0005 | 0.0002 | 16838 | 17058 | L=4, D=128, SEQ=256, TASK=copy |
| H1_SEQ128 | copy | 0.6458 | 0.5208 | 17927 | 17616 | L=4, D=128, SEQ=128, TASK=copy |
| H1_SEQ256 | copy | 0.2960 | 0.3145 | 15841 | 15907 | L=4, D=128, SEQ=256, TASK=copy |
| H2_VOC1024 | copy | 0.0025 | 0.3797 | 16099 | 16063 | L=4, D=128, SEQ=256, TASK=copy |
| H5_MQAR | mqar | 0.0938 | 0.0938 | 15818 | 15707 | L=4, D=128, SEQ=256, TASK=mqar |
| H5_FUZZY | fuzzy_recall | 0.0312 | 0.0625 | 15552 | 15519 | L=4, D=128, SEQ=256, TASK=fuzzy_recall |
| H2_VOC4096 | copy | 0.0000 | 0.0000 | 16512 | 16674 | L=4, D=128, SEQ=256, TASK=copy |
| H4_NOMIXER | copy | 0.0000 | 0.2593 | 0 | 16303 | L=4, D=128, SEQ=512, TASK=copy |
| H4_NOMIXER | copy | 0.0000 | 0.9348 | 0 | 13856 | L=4, D=128, SEQ=512, TASK=copy |
| H1_SEQ512 | copy | 0.2186 | 0.2093 | 15539 | 15400 | L=4, D=128, SEQ=512, TASK=copy |
| H3_LAYERS16 | copy | 0.6073 | 0.3989 | 3703 | 3681 | L=16, D=128, SEQ=256, TASK=copy |
| H1_SEQ128 | copy | 0.6265 | 0.5446 | 17215 | 16784 | L=4, D=128, SEQ=128, TASK=copy |
| H1_SEQ256 | copy | 0.3172 | 0.2891 | 16444 | 16665 | L=4, D=128, SEQ=256, TASK=copy |
| exp_20260107_172610 | 1.427 | 1.477 | 0.514 | 0.503 | Fix: PreNorm+StickyInit |
| exp_20260107_174602 | 1.473 | 1.436 | 0.517 | 0.520 | Fix: PreNorm+StickyInit |
| H3_LAYERS16 | copy | 0.7936 | 0.5458 | 3611 | 3721 | L=16, D=128, SEQ=256, TASK=copy |
| exp_20260107_182537 | 0.066 | 0.155 | 0.977 | 0.940 | Fix: PreNorm+StickyInit |
| ablation_20260107_140548 | 0.1533 | 0.1848 | D_MODEL: 128<br>N_LAYERS: 4<br>VOCAB: 64<br>STREAMS: 4<br>BS: 32<br>SEQ: 512<br>STEPS: 8000<br>LR: 0.001 |
