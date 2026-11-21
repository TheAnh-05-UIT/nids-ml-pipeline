# Full eligible-pool training

The operator selected a four-family pilot on 2026-11-22: Edge-IIoTset DDoS TCP
and ICMP; CICIoMT2024 MQTT Connect Flood and Publish Flood, together with Normal.
The output remains binary **Normal / DDoS**. HTTP and UDP are missing; this is
not all six requested families. Bootstrap requires `--allow-incomplete-families`
to record this choice. A fresh job/model identity starts from new initial weights.

## Data and scheduling

`scripts/run_full_swarm.py plan` compiles small references from the completed
mapped Edge catalog and MQTT temporal catalog, preserving each parent's splits
and training-client assignments. It excludes non-DDoS attacks and all tensor
patterns shared between the two parents. No held-out row moves into training.

Each reference uses nine bytes; parent SQLite files and HDF5 are reused. A pass
uses every eligible training sample exactly once on its assigned node. Enough
rounds are generated to respect 4,096 samples/client/round without wrapping,
padding or dropping a tail. Both labels appear each round; the smaller node can
contribute fewer samples. FedAvg uses the actual counts. Shard order is seeded
and rows shuffle inside each round. Each round need not contain every family.

This ML orchestration profile supports up to 4,096 rounds and ten passes.
Legacy ML commands keep their two-round guards. The new controller supplies a
validated schedule and dynamic receipt counts while reusing SWOP controller
logic. SN, SL, User-TF and SWOP source/images are unchanged. There is no uncertain
upload retry or automatic resume. Failures retain evidence for review.

Validation runs after each complete pass. Test evaluation is separate, against
a frozen checkpoint, with `--evaluate-test-approved`. Threshold defaults to 0.5;
choose it without tuning on test. Input preflight may hash test files for input
integrity; this does not evaluate the model or use test values for optimization.

Shared held-out rows are evaluated by client 0; local Edge rows by their original
owner. `aggregate-evaluation` sums disjoint confusion matrices and sample-weighted
loss for the same checkpoint, threshold, plan and split. It checks all local,
global and family sample counts. Do not average F1 or count shared test twice.

Edge provenance remains `DATASET_PROVIDED_MAPPED`; MQTT remains the approved
`DATASET_PROVIDED_SCENARIO`, a weak capture-level label which can include background
traffic. Independent packet ground truth is not verified. Edge connection holdout
and MQTT temporal holdout are different protocols; combined results do not prove
unseen-capture generalization.

## Next operator step on RPi5

Commit/push the ML changes and pull on both boards first. Then run this only on
RPi5, where both completed catalogs already exist:

```bash
(
  set -euo pipefail
  cd /home/ubuntu/nids-ml-pipeline
  git pull --ff-only origin main
  PIN="$(git rev-parse HEAD)"
  test -z "$(git status --porcelain -- src scripts pyproject.toml)"
  python3 -B scripts/run_full_swarm.py --help
  sudo -v
  python3 -u -B scripts/prepare_full_plan.py preflight --expected-commit "$PIN"
  python3 -u -B scripts/prepare_full_plan.py run --expected-commit "$PIN" --passes 1 --execute-approved
)
```

This prepares the **plan**, without training or starting SN/SL. It hashes both
existing indexes, requires 6 GiB available RAM, reserves 1 GiB disk plus at most
64 MiB for final files, and uses the pinned existing ARM64 TF image. The container
has no network, a 3 GiB private tmpfs and a 4 GiB memory limit. Runtime files are
pinned before/after compilation. The three final files are copied with SHA/fsync;
the new private temporary SQLite is removed after success. Existing files/jobs
are preserved. No PCAP export or completed catalog needs to run again.

The command prints `evidence_directory`. Follow its `plan.log`; then require
`exit-code.txt=0` and `PASS_COMPACT_FULL_PLAN_PERSISTED_SHA` in `publication.json`.
The plan is in `evidence_directory/plan`. Its exact post-dedup node counts and
source requirements determine native placement/bootstrap.

Orange Pi has about 1.24 GB free in the latest snapshot. Do not duplicate a full
catalog or all HDF5 there. Transfer its small plan reference file and manifest;
identify missing source shards before transfer. A private RAM directory may hold
missing training shards if capacity permits, keeping durable originals. RAM
staging disappears on reboot. Require `inspect --client 1 --source SHA=PATH` in
its actual runtime before training. Native placement/training has not run yet.

## Remaining CLI steps after placement

```bash
python3 -B scripts/run_full_swarm.py inspect --help
python3 -B scripts/run_full_swarm.py bootstrap --help
python3 -B scripts/run_full_swarm.py client --help
python3 -B scripts/run_full_swarm.py controller --help
python3 -B scripts/run_full_swarm.py verify --help
python3 -B scripts/run_full_swarm.py evaluate --help
python3 -B scripts/run_full_swarm.py aggregate-evaluation --help
```

Network roles default to dry-run and require `--execute-approved`. Use fresh
endpoints/SN epoch and the new bootstrap. `verify` independently replays every
weighted mean, binds each round to its exact reference segment, checks completion
barriers and chains the final checkpoint to the final global tensor digest.

Local verification uses tiny synthetic fixtures and a 12-round loopback run
(three complete passes). It is not ARM64 or real-dataset training evidence.
