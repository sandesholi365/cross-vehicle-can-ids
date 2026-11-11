# Cross-vehicle CAN intrusion detection: code, audit and results

Code for the MSc thesis *Evaluating Cross-Dataset Generalization of CAN Bus Intrusion Detection Models under Federated Learning: Cross-vehicle evaluation, model averaging and threshold transfer* (Sandesh Oli, DTU Compute, Technical University of Denmark, 2027; supervisors Gaurav Choudhary and Maria Papaioannou).

The thesis trains CAN intrusion detectors on some vehicles and tests them on a vehicle they have never seen. It compares single-vehicle, pooled, FedAvg and FedProx training on CAN-Train-and-Test (leave-one-vehicle-out), studies what the averaging step of federated training does, tests alarms over several windows, and validates the frozen method on CrySyS and ROAD.

This repository contains **code, dataset-audit files and result tables only**. It does **not** contain any CAN traffic or windows derived from it. Download the datasets from their original sources (below).

## What is here

| Path | What it is |
| --- | --- |
| `src/data/` | conversion to one CSV format, duplicate audit, content-based vehicle check, CrySyS frame labels |
| `src/features/` | windows of 100 messages and their features, for CAN-Train-and-Test, ROAD and CrySyS |
| `src/experiments/` | all experiments, checks, analyses and figures of the thesis (see "How to reproduce") |
| `results/audit/` | the dataset audit: `file_audit.csv` (one row per raw CAN-Train-and-Test file, with its hash and identical files), `dataset_decisions.csv` (the decision for every one of the 236 files), the vehicle check, and `ctt_conflict_evidence.csv` (hashes and ID evidence for the 12 files listed under two vehicles) |
| `results/` (other folders) | the result tables behind every number in the thesis; per-window scores are not included because of their size |
| `results/frozen/` | the frozen method used for the external tests: every setting and choice, and the MD5 hash of every script and data file |

## The dataset audit in short

The 236 CSV files of CAN-Train-and-Test hold 174 different recordings; 62 files are exact copies of others. Fifty of the copies are the same vehicle's recording reused in another set, which the dataset's first author confirmed is by design. In 12 cases, identical files are listed under different vehicles in the README: all files in `set_01/test_04_unknown_vehicle_unknown_attack` are identical to files in `set_02/test_01_known_vehicle_known_attack`. Their message IDs match the Traverse, and the dataset's first author confirmed (personal communication, 31 October 2026) that `set_01/test_04` holds Traverse recordings, not Silverado ones. The dataset README could not be corrected because the repository is read-only, so this note is the record of the correction. The same files are also in `can-train-and-test-v1.5`. The thesis uses each of the 174 recordings once; `results/audit/dataset_decisions.csv` lists every decision.

## Datasets (not included)

| Dataset | Source | Licence / terms |
| --- | --- | --- |
| CAN-Train-and-Test | Lampe & Meng, *Computers & Security* 140 (2024), doi:10.1016/j.cose.2024.103777; dataset on DTU Data, doi:10.11583/DTU.24805533 | CC BY 4.0 |
| ROAD | Verma et al., *PLoS ONE* 19(1) (2024); Oak Ridge National Laboratory | No licence stated in the dataset's readme; it asks users to cite the paper |
| CrySyS | Gazdag, Ferenc & Buttyán, *Scientific Data* 10, 903 (2023); Figshare, doi:10.6084/m9.figshare.c.6726165 | CC0 |

Please cite the dataset papers when you use them.

## Folder layout

Place the downloaded data like this (paths are relative to the repository root):

```
data/raw/can_train_and_test/set_01 ... set_04/     (as downloaded)
data/raw/road/ambient/*.log
data/raw/road/attacks/*.log, capture_metadata.json
data/raw/crysys/                                   (as downloaded)
```

All intermediate data are written to `data/processed/` and all results to `results/`.

## Requirements

Python 3 with the packages in `requirements.txt` (`pip install -r requirements.txt`). The MLP runs use PyTorch on the CPU. Several scripts accept `QUICK=1` for a short pipeline check before the real run (on Windows PowerShell: `$env:QUICK="1"`).

## How to reproduce

Run every command from the repository root, in this order.

**1. Data and audit (CAN-Train-and-Test, ROAD)**

```
python src/data/audit_duplicates.py
python src/data/verify_vehicles.py
python src/data/ctt_conflict_evidence.py     # evidence table for the 12 conflicting files
python src/data/convert.py
```

`convert.py` also contains an OTIDS converter from an early stage of the project; OTIDS is not used in the thesis.

**2. Windows and features**

```
python src/features/build_windows.py
python src/features/build_road_windows.py
python -I src/data/crysys_find_missing.py --root data/raw/crysys --expected src/data/crysys_expected_files.csv
python -I src/data/crysys_labels.py --root data/raw/crysys
python -I src/data/crysys_label_truth_test.py data/raw/crysys S-1-1 10
python src/features/build_crysys_windows.py --root data/raw/crysys
```

The CrySyS steps first check that the download is complete, then derive and check the frame labels, and then test the labelling on simulated recordings built from the real benign traces and attack plans (no model is involved). Only after that are the windows built.

**3. Main experiments on CAN-Train-and-Test (Chapter 6, first part)**

```
python src/experiments/run_baseline.py           # within- vs cross-vehicle baseline
python src/experiments/run_baselines.py          # Random Forest, leave-one-vehicle-out
python src/experiments/run_federated.py          # MLP: single, pooled, FedAvg, FedProx
python src/experiments/check_arm_parity.py       # the comparability check
python src/experiments/check_calibration.py
python src/experiments/check_benign_windows.py
python src/experiments/validate_calibration.py
python src/experiments/calib_sensitivity.py run block
python src/experiments/calib_sensitivity.py run recording
python src/experiments/analyse_results.py
python src/experiments/make_figures.py
```

**4. Original external test on ROAD**

```
python src/experiments/run_road.py
python src/experiments/road_extra_analysis.py
python src/experiments/road_allseeds.py run
python src/experiments/road_allseeds.py analyse
python src/experiments/analyse_road.py
```

**5. Follow-up analyses on CAN-Train-and-Test**

```
python -u src/experiments/explain_arms.py run            # why FedAvg keeps its threshold
python -u src/experiments/explain_arms.py report
python -u src/experiments/explain_avg.py run
python -u src/experiments/explain_avg.py report
python -u src/experiments/direction1_averaging.py select  # averaging vehicle models
python -u src/experiments/direction1_averaging.py run
python -u src/experiments/direction1_control.py run       # random-partition control
python -u src/experiments/direction1_control.py report
python -u src/experiments/feasibility_alarms.py
python -u src/experiments/direction2_alarms.py inner      # alarms over several windows
python -u src/experiments/direction2_alarms.py select
python -u src/experiments/direction2_alarms.py run
```

**6. Freeze, then the external validation**

```
python -u src/experiments/frozen_method.py select
python -u src/experiments/frozen_method.py freeze
python -u src/experiments/run_external.py run crysys
python -u src/experiments/run_external.py report crysys
python -u src/experiments/run_external.py run road
python -u src/experiments/run_external.py report road
```

`run_external.py` checks the hashes in `results/frozen/` before it starts and stops if any script or data file has changed. Each dataset is meant to be run once.

## Use of AI

The code was written with the help of an AI assistant (Claude, Anthropic), as declared in the thesis.

## Licence

The code is released under the MIT Licence (see `LICENSE`). The licence covers the code, the audit files and the result tables, not the datasets, which keep their own terms.
