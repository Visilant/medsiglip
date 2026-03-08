# Visilant Data Library — Dataset Documentation

> Prepared for MedSigLIP fine-tuning

## 1. Image Store

| Property | Value |
|----------|-------|
| **Path** | `/home/adi/visilant_data/` |
| **Total size** | 191 GB |
| **Total images** | 117,473 JPGs (root level) |
| **Format** | JPEG, RGB, 3 channels |
| **Resolutions** | 4032x3024 (~40%), 2576x1934 (~38%), 2560x1920 (~19%), other (~3%) |
| **Dilation** | 98.7% undilated, 1.1% dilated, 0.2% unknown |

Additional images in subdirectories:
- `LEBWCO/` — 389 images (Lebanon WHO field collection)
- `storage/emulated/0/` — 246 images (Android device captures)

**Image name convention:** CSV `image_name` column stores filenames **without** `.jpg` extension. To resolve on disk: `{image_name}.jpg`.

---

## 2. Label Files

### 2a. Primary: `Visilant_MVP.csv`

| Property | Value |
|----------|-------|
| **Path** | `/home/adi/Visilant_MVP.csv` |
| **Rows** | 102,951 |
| **Matched to disk** | 96,979 (94.2%) |
| **Unmatched** | 5,972 (images not on disk) |

**Columns (19):**

```
image_name, dilation_status, image_type, age, gender, MEDICAL HISTORY,
DIAGNOSIS - Corneal Infection, Complaints, Visual Acuity, Lacrimal Duct,
Visual Acuity Type, Pinhole Acuity, DIAGNOSIS - LENS STATUS,
DIAGNOSIS - ADD PATH, DIAGNOSIS - CORNEAL ABNORMALITY,
mapped_diagnosis, mapped_category, mapped_corneal_abnormality, mapped_lens_status
```

### 2b. Extended: `cornea_classifier/Visilant_MVP_1.csv`

| Property | Value |
|----------|-------|
| **Path** | `/home/adi/cornea_classifier/Visilant_MVP_1.csv` |
| **Rows** | 112,114 |
| **Matched to disk** | 100,683 (89.8%) |
| **Same schema as MVP.csv** | Yes (18 columns) |

### 2c. Fine-tune subsets (MedGemma)

| File | Path | Rows |
|------|------|------|
| all_visilant_data | `/home/adi/FineTune_Medgemma/all_visilant_data.csv` | 13,435 |
| clean (Aug 2025) | `/home/adi/FineTune_Medgemma/clean_visilant_data_aug11_2025.csv` | 4,098 |

The clean subset uses a numeric `lens_status_labelled` column (0–3).

### 2d. Test set: `test_dataset_combined.csv`

| Property | Value |
|----------|-------|
| **Path** | `/home/adi/visilant_data/test_dataset_combined.csv` |
| **Rows** | 317 |
| **Structure** | Per-eye (left/right) columns with granular binary labels |

Per-eye labels: `Active corneal infection`, `Pterygium`, `Inactive corneal opacity`, `Subtle corneal infection`, `Subtle corneal opacity`, `corneal abnormality`, `needs confirmation`, `remove`, `s/p keratoplasty`, `brown cataract`, `cataract`, `normal`, `lens_status`, `Infection Type Suspected`, `Infection Type Confirmed`.

### 2e. Coverage summary

| Metric | Count |
|--------|-------|
| Images on disk | 117,473 |
| Images with labels (in either CSV) | 100,617 |
| Orphan images (no labels) | 16,856 |

---

## 3. Label Taxonomy

### 3a. `mapped_lens_status` — Cataract / lens grading (primary target)

| Label | Count (MVP.csv) |
|-------|-----------------|
| clear_crystalline_lens | 44,319 |
| immature_cataract | 19,590 |
| early_lens_changes | 18,770 |
| PCIOL | 9,385 |
| not_able_to_visualize_lens | 8,873 |
| mature_cataract | 1,554 |
| aphakia | 410 |
| Other | 50 |

### 3b. `mapped_corneal_abnormality` — Cornea condition

| Label | Count |
|-------|-------|
| Normal | 47,474 |
| Active corneal infection | 38,643 |
| Inactive corneal opacity / scarring | 12,200 |
| Other | 4,194 |
| Keratoplasty / graft | 206 |
| Prosthetic / Phthisis | 65 |
| Conjunctivitis | 48 |
| PCIOL / Pseudophakia | 44 |
| Trauma / foreign body | 35 |
| Corneal degeneration / dystrophy | 26 |
| Pterygium | 15 |
| Subconjunctival hemorrhage | 1 |

### 3c. `mapped_category` — High-level diagnosis bucket

| Label | Count |
|-------|-------|
| Normal | 53,806 |
| Other | 44,696 |
| Pterygium | 3,466 |
| Corneal scarring | 353 |
| Infectious keratitis | 287 |
| PCIOL | 171 |
| Trauma / foreign body | 66 |
| Conjunctivitis | 53 |
| Cataract | 47 |
| aphakia | 5 |
| Subconjunctival hemorrhage | 1 |

### 3d. `image_type` — Illumination modality

| Type | Count |
|------|-------|
| diffuse | 31,364 |
| blue | 23,984 |
| diffuse (tabletop slit lamp) | 23,754 |
| blue (tabletop slit lamp) | 22,822 |
| slit | 799 |
| Slit lamp blue light | 113 |
| Slit lamp diffuse illumination | 84 |

### 3e. Numeric cataract grade (clean subset only)

From `clean_visilant_data_aug11_2025.csv`, column `lens_status_labelled`:

| Grade | Meaning (inferred) | Count |
|-------|---------------------|-------|
| 0 | Clear / normal | 559 |
| 1 | Early lens changes | 442 |
| 2 | Immature cataract | 254 |
| 3 | Mature / advanced cataract | 509 |

---

## 4. Metadata Fields (for text encoder input)

These fields can be concatenated into natural-language captions for SigLIP contrastive training:

| Field | Example value | Notes |
|-------|---------------|-------|
| `age` | `62.0` | Float, some nulls |
| `gender` | `female` | male / female |
| `Complaints` | `Blurry Vision Far, Redness` | Comma-separated, free text |
| `Visual Acuity` | `5/60` | Snellen notation |
| `Pinhole Acuity` | `6/12` | Snellen, many blanks |
| `MEDICAL HISTORY` | `No medication` | Free text |
| `DIAGNOSIS - LENS STATUS` | `Immature cataract` | Raw clinician diagnosis |
| `DIAGNOSIS - ADD PATH` | `4mm` | Additional pathology size |
| `DIAGNOSIS - CORNEAL ABNORMALITY` | `Active corneal infection - Active Infiltrate` | Raw corneal dx |
| `dilation_status` | `undilated` | undilated / dilated |
| `image_type` | `diffuse` | Illumination modality |

---

## 5. MedSigLIP Fine-Tuning Considerations

### 5.1 Recommended label columns for training tasks

| Task | Label column | Classes |
|------|-------------|---------|
| Lens / cataract classification | `mapped_lens_status` | 7 clean classes + Other |
| Corneal abnormality detection | `mapped_corneal_abnormality` | 12 classes |
| High-level triage | `mapped_category` | 11 classes |
| Ordinal cataract grading | `lens_status_labelled` (clean subset) | 4 grades (0–3) |

### 5.2 Caption construction strategy

For contrastive (SigLIP) training, construct text pairs from metadata. Example template:

```
Slit lamp photograph, {image_type} illumination, {dilation_status}.
{age:.0f} year old {gender}.
Lens status: {mapped_lens_status}.
Corneal finding: {mapped_corneal_abnormality}.
Complaints: {Complaints}.
Visual acuity: {Visual Acuity}.
```

### 5.3 Image preprocessing

- **Native resolutions** are large (2560–4032 px). Resize/center-crop to model input (e.g., 384x384 for SigLIP).
- All images are RGB JPEG — no grayscale or RGBA conversion needed.
- Consider separate handling of `diffuse` vs `blue` illumination (fluorescein staining) — these are fundamentally different modalities.

### 5.4 Data splits

- **Train:** Use `Visilant_MVP.csv` or `Visilant_MVP_1.csv` (100K+ labeled images).
- **Held-out test:** Use `test_dataset_combined.csv` (317 images, per-eye granular labels).
- **Validation:** Carve ~10% from training set, stratified by `mapped_lens_status` and `mapped_corneal_abnormality`.
- The 16,856 orphan images (on disk, no labels) could be used for self-supervised pretraining or unlabeled contrastive augmentation.

### 5.5 Class imbalance

`mapped_lens_status` is heavily skewed toward `clear_crystalline_lens` (43%). Consider:
- Oversampling rare classes (mature_cataract: 1.5%, aphakia: 0.4%)
- Class-weighted contrastive loss
- Stratified sampling per batch

`mapped_corneal_abnormality` is bimodal — Normal (46%) vs Active corneal infection (38%), with a long tail. The tail classes (Keratoplasty, Prosthetic, Conjunctivitis, etc.) have <250 samples each.

### 5.6 Data quality notes

- ~5.9K rows in `Visilant_MVP.csv` reference images not on disk (likely from other devices/sites).
- The `mapped_diagnosis` column is low-entropy — 99%+ is either `['normal']` or `Other`. Prefer `mapped_corneal_abnormality` and `mapped_lens_status` instead.
- Some `mapped_category` values overlap with `mapped_lens_status` (e.g., `PCIOL`, `aphakia` appear in both). These are separate clinical axes — corneal vs lens pathology.
- `image_type` contains parenthesized variants (`diffuse /(tabletop slit lamp/)`) — normalize before use.

### 5.7 File references quick lookup

```
/home/adi/visilant_data/                              # 191 GB, 117K images
/home/adi/Visilant_MVP.csv                            # 103K rows, primary labels
/home/adi/cornea_classifier/Visilant_MVP_1.csv        # 112K rows, extended labels
/home/adi/FineTune_Medgemma/all_visilant_data.csv     # 13K rows, fine-tune subset
/home/adi/FineTune_Medgemma/clean_visilant_data_aug11_2025.csv  # 4K rows, numeric grades
/home/adi/visilant_data/test_dataset_combined.csv     # 317 rows, held-out test
```
