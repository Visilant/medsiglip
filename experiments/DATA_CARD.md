# Data Card — Visilant Ophthalmology Dataset

---

## Dataset Identity

| Field | Value |
|-------|-------|
| **Name** | Visilant MVP Slit-Lamp Dataset |
| **Domain** | Ophthalmology (anterior segment) |
| **Modality** | Slit-lamp photographs (RGB JPEG) |
| **Resolution** | Variable (resized to 448x448 for MedSigLIP) |
| **Source CSV** | `/home/adi/Visilant_MVP.csv` |
| **Image Directory** | `/home/adi/visilant_data/` |
| **Test CSV** | `/home/adi/visilant_data/test_dataset_combined.csv` |
| **Total CSV rows** | 102,951 |
| **Images on disk** | 117,534 .jpg files |
| **Matched to CSV** | ~96,931 |
| **Clean (post-filter)** | 96,927 |
| **Corrupt removed** | 15,131 (12.9%) — listed in `experiments/data/bad_images.json` |
| **"Other" lens dropped** | 50 (ambiguous label) |

---

## Label Schema

### Lens Status (7 classes)

| Class | Count | % of Clean | Weight (clamped) |
|-------|-------|-----------|-----------------|
| clear_crystalline_lens | 44,319 | 45.7% | 0.50 |
| immature_cataract | 19,590 | 20.2% | 0.71 |
| early_lens_changes | 18,770 | 19.4% | 0.74 |
| PCIOL | 9,385 | 9.7% | 1.47 |
| not_able_to_visualize_lens | 8,873 | 9.2% | 1.56 |
| mature_cataract | 1,554 | 1.6% | 8.89 |
| aphakia | 410 | 0.4% | 10.00 (clamped) |
| ~~Other~~ | ~~50~~ | — | Dropped |

**Imbalance ratio:** 108:1 (clear_crystalline_lens : aphakia)

### Corneal Abnormality (4 binned classes)

| Class | Count | % of Clean | Original Labels Mapped |
|-------|-------|-----------|----------------------|
| Normal | 47,474 | 49.0% | Normal |
| Active corneal infection | 38,643 | 39.9% | Active corneal infection |
| Inactive corneal opacity | 12,200 | 12.6% | Inactive corneal opacity / scarring |
| Rare | 4,634 | 4.8% | Other, Keratoplasty/graft, Prosthetic/Phthisis, Conjunctivitis, PCIOL/Pseudophakia, Trauma/foreign body, Corneal degeneration/dystrophy, Pterygium, Subconjunctival hemorrhage |

**Binning rationale:** 9 rare conditions each had <500 samples. Collapsed into "Rare" to ensure sufficient training signal. Mapping defined in `experiments/config/base.yaml`.

---

## Data Splits

| Split | Count | % | Stratification |
|-------|-------|---|---------------|
| Train | 86,672 | 89.4% | lens_status \| corneal_binned composite key |
| Val | 9,631 | 9.9% | Same stratification |
| Test (excluded) | 628 | 0.6% | Pre-defined from `test_dataset_combined.csv` (317 patients x 2 eyes) |

**Split file:** `experiments/data/splits.json`
**Split generation:** `experiments/data/splits.py` using sklearn `train_test_split` with stratification on composite `lens_status|corneal_binned` key. Test images excluded before splitting. Rare composite combinations (<2 samples) handled via fallback stratification.

---

## Metadata Columns (used in captions)

| Column | Description | Used In |
|--------|-------------|---------|
| `image_name` | Filename without .jpg extension | Image loading |
| `lens_status` | 8 original classes (7 after dropping "Other") | Labels, captions |
| `corneal_abnormality` | 12 original classes → 4 binned | Labels, captions |
| `image_type` | Illumination type (diffuse, blue, slit, tabletop + variants) | Clinical caption |
| `age` | Patient age | Clinical caption |
| `sex` | Patient sex | Clinical caption |
| `visual_acuity_logmar` | LogMAR visual acuity | Clinical caption |
| `dilation` | Pupil dilation status | Clinical caption |

---

## Caption Styles

Three caption formats fit within MedSigLIP's 64-token limit via greedy clause truncation:

1. **clinical** (default): `"Slit lamp photo, {image_type} illumination, {dilation}. {age}yo {gender}. Lens: {lens}. Cornea: {cornea}. VA: {va}."`
2. **label_only** (ablation): `"Lens: {lens}. Cornea: {cornea}."`
3. **sentence** (ablation): `"Slit lamp photograph showing {lens} and {cornea}."`

**Tokenization:** SentencePiece via `SiglipTokenizer`. Truncation reserves 2 tokens for BOS/EOS, greedily drops trailing sentence-clauses until within 62-token budget.

---

## Preprocessing Pipeline

1. **Load:** `PIL.Image.open(path).convert("RGB")`
2. **Resize:** `torchvision.transforms.Resize(448)` (shorter edge to 448, aspect ratio preserved)
3. **ToTensor:** [0, 255] → [0.0, 1.0]
4. **Normalize:** `mean=0.5, std=0.5` → [-1.0, 1.0]

**Corrupt image handling:**
- Primary: 15,131 corrupt files filtered via `bad_images.json` during data loading
- Fallback: `PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True` zero-fills truncated JPEGs

---

## Known Issues & Caveats

| Issue | Impact | Mitigation |
|-------|--------|-----------|
| 15,131 corrupt JPEGs (12.9%) | Training crashes if encountered | Filtered in `load_and_filter_data()`; LOAD_TRUNCATED_IMAGES as backup |
| 108:1 class imbalance (lens) | Model biased toward majority class | Class weight clamping [0.5, 10.0]; focal loss option in Phase 2 |
| "Rare" corneal bin is heterogeneous | Mixed pathologies in single class | Accepted trade-off for sufficient sample size; may revisit if data grows |
| No demographic balance analysis | Potential bias in age/sex distribution | Not yet assessed; recommended before deployment |
| Test set is small (628 images) | Limited statistical power for rare classes | Confidence intervals should be reported; consider bootstrap |
| Image quality varies (illumination, focus) | Some images diagnostically useless | Not filtered beyond corruption check; quality scoring is future work |
