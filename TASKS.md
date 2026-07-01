# Engineering Tasks — Restructuring & Shared Code

---

## 1. Create `models/` Package

Create `models/__init__.py` and the following files:

**`models/encoder.py`**
Extract the DenseNet-169 encoder forward pass (currently duplicated in every `model.py` across all technique folders) into a single `Encoder` class. Constructor accepts `pretrained: bool` read from config.

**`models/decoder_1ch.py`**
Extract the 1-channel decoder (depth output) from the existing `model.py` files into a single `Decoder1Ch` class with the four `UpSample` blocks as defined in the current implementation.

**`models/decoder_3ch.py`**
Extract the 3-channel decoder (residual / direct output) from the existing `model_3D.py` files into a single `Decoder3Ch` class.

**`models/weight_head.py`**
Extract both weight head variants into this file:
- `WeightHeadSigmoid(n=2)` — n-output sigmoid head (used in var1 of Techniques 1–3)
- `WeightHeadSoftmax(n=3)` — n-output softmax head (used in var2 and Technique 4)

**`models/model_builder.py`**
Implement `build_models(technique: int, variant: str) -> tuple` that assembles and returns the correct `(model_1, model_2, model_3)` combination for the given technique and variant using the components above. `model_3` is `None` for Techniques 1 and 2.

---

## 2. Create `utils/` Package

Create `utils/__init__.py` and the following files:

**`utils/physics.py`**
Move into here (single canonical copy, delete all duplicates):
- `compute_haze_image(...)`
- `compute_complex_image(...)`
- Everything currently in `ricardo_code_1.py` and `ricardo_underwater_image.py`

**`utils/loss.py`**
Move into here (single canonical copy):
- `ssim(...)` and the SSIM loss wrapper
- `VGGPerceptualLoss` (currently only in `Technique_4/loss.py`)

**`utils/metrics.py`**
Move into here:
- `compute_errors_nyu(...)`
- `add_results_1(...)`

**`utils/helpers.py`**
Move into here:
- `DepthNorm(...)`
- `AverageMeter`
- `colorize(...)`
- `simple_save_images(...)`

**`utils/transforms.py`**
Move into here:
- `RandomChannelSwap`
- `ToTensor`
- `Augmentation` and all custom transform classes from `custom_transforms.py`

---

## 3. Create `data/` Package

Create `data/__init__.py` and the following files:

**`data/nyu.py`**
Consolidate all NYU dataset and dataloader logic (currently spread across `data_2.py`, `data_3.py`, `data_4.py` in multiple technique folders) into a single file exposing:
- `get_train_loader(config) -> DataLoader`
- `get_test_loader(config) -> DataLoader`

**`data/make3d.py`**
Consolidate all Make3D dataset and dataloader logic (currently in `data_4_Make_3D.py`, `data_make3d.py`) into a single file exposing the same interface:
- `get_train_loader(config) -> DataLoader`
- `get_test_loader(config) -> DataLoader`

**`data/kitti.py`**
Empty placeholder file with the two function stubs (`get_train_loader`, `get_test_loader`) raising `NotImplementedError`.

---

## 4. Restructure Technique Folders

For each of `Technique_1/`, `Technique_2/`, `Technique_3/`, `Technique_4/`, create the following subfolder structure:

```
Technique_X/
├── NYU/
│   ├── base/
│   │   ├── train.py
│   │   └── test.py
│   ├── var1/
│   │   ├── train.py
│   │   └── test.py
│   └── var2/
│       ├── train.py
│       └── test.py
├── Make3D/
│   ├── base/
│   │   ├── train.py
│   │   └── test.py
│   ├── var1/
│   │   ├── train.py
│   │   └── test.py
│   └── var2/
│       ├── train.py
│       └── test.py
└── KITTI/
    └── .gitkeep
```

Migrate the existing train/test scripts into the correct locations. After migration, delete all old script files and the old per-technique `model.py`, `model_3D.py`, `model_weight*.py`, `loss.py`, `utils.py`, `custom_transforms.py`, `data_*.py`, `ricardo_*.py` copies — they are now replaced by the shared packages in tasks 1–3.

---

## 5. Slim Down Train / Test Scripts

After tasks 1–4 are done, each `train.py` and `test.py` must only contain:
1. `argparse` setup (`--config`, `--resume`).
2. Config loading.
3. A call to `build_models(technique, variant)` from `models/model_builder.py`.
4. A call to `get_train_loader(config)` / `get_test_loader(config)` from the appropriate `data/` module.
5. The training or evaluation loop (the only logic that is genuinely unique per technique/variant).

No physics helpers, no loss class definitions, no dataset classes, no model class definitions inside these files.

---

## 6. Verify All Imports and Run Paths

All scripts must be runnable from the repo root via:

```bash
python Technique_1/NYU/base/train.py --config config.yaml
python Technique_1/Make3D/var1/test.py --config config.yaml
```

Ensure `models/`, `utils/`, and `data/` are importable without path manipulation by adding a `pyproject.toml` or `setup.py` that installs the repo as a package in editable mode (`pip install -e .`), or by confirming that running from the repo root is sufficient for Python's module resolution.