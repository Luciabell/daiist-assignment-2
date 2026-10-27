# Assignment 2 Report

- **Name**: Lucia Bell
- **Student ID**: 19177
- **Email**: lbellsicre.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

- **Decision supported**: at the entrance of a construction site, is each worker wearing a safety hard hat? Today a guard or the site safety manager decides by looking. The model turns that visual check into three outcomes: **pass**, **send to a person**, or **fail** (worker stopped at the gate).
- **Who acts on it**: the entrance gate system and the site safety manager, who handles every case marked "human review" and every "fail".
- **Positive class**: Head (a bare head, i.e. a worker with no hard hat). This is the safety violation. The other class, helmet, is the compliant worker.
- **Cost of each mistake**:
  - **False negative** (violation passes as compliant): **5,000 EUR**. Fines, liability and injury risk from a worker on site without protection.
  - **False positive** (compliant worker stopped): **5 EUR**. A short delay at the gate and a quick manual check.
  - **Human review**: **1.5 EUR** per case (about 2 minutes at 45 EUR/hour loaded cost).
  - **True positive and true negative**: 0 EUR.
- **Why the asymmetry drives everything else**: a missed violation costs 1,000 times more than a false alarm. So the headline metric is recall on head , the confidence threshold is pushed low, and a band of uncertain cases goes to a person instead of being decided automatically.

## Dataset

- **Source**: Roboflow Universe, public dataset (not forked), license CC BY 4.0.
- **Workspace**: `gbc-tdplg`
- **Project**: `helmet-no-helmet`
- **Version**: 1 (generated Aug 10, 2023, 1,801 images, 640x640)
- **Link**: https://universe.roboflow.com/gbc-tdplg/helmet-no-helmet
- **Task**: object detection, 3 classes: `head`, `helmet`, `person`.
- **What one image represents**: a photo of one or more people on or near a construction site, with a box around each head (with helmet if it wears a hard hat, head if it is bare) and around some whole people (person).
- **Why this dataset serves the problem**: it has an explicit class for the violation (head) next to the compliant class (helmet), so the model learns the exact distinction the gate has to make. A generic model such as COCO has no hard-hat class at all. The person class is not part of the decision: it is still trained on, but it is not counted as a case in the cost analysis.

## What the dataset's composition implies

- **Split sizes** : train 1,602 images, validation 149, test 50. The test split is only about 2.8% of the data.
- **Instances per class** :

|| Split | `head` | `helmet` | `person` |
| :--- | :---: | :---: | :---: |
| Train | 2,633 | 8,286 | 299 |
| Validation | 103 | 593 | 9 |
| Test | 23 | 192 | 30 |

- **Object size**: median box area is **0.57% of the image in train** (0.52% in validation, 0.63% in test), so about 0.6% overall (`median_box_area_pct_of_image` in the JSON). That is roughly 48x48 pixels on a 640x640 image, so the objects are small.
- **What this implies**:
  - **The test number for `head` is thin.** With only 23 head instances, one extra miss moves recall by about 4 points. A recall of 20/23 = 0.87 has a 95% Wilson interval of roughly 0.68 to 0.95, so I cannot say much more than "somewhere between 70% and 95%".
  - **`person` in the test set comes from 2 images** (30 boxes), so its numbers are close to meaningless. I do not use it for any decision.
  - **The test split does not match production.** head is 23/215 = 11% of worker instances in the test set, but I assume 5% in production, so the cost analysis re-weights cases to the assumed base rate. The pictures also look like low-resolution site photos, probably not the camera angle and lighting of a real entrance gate.
  - **What I cannot conclude**: that the cost figures will hold at a real gate, or that performance is stable for rare conditions (night, rain, heavy occlusion), which a 50-image test set cannot show.

## Finetuning and baselines

- **Model and task**: YOLO11n, object detection, finetuned from the pretrained `yolo11n.pt` checkpoint. I chose the nano size because I train on a laptop CPU, and because for a gate check the speed is more useful than a small accuracy gain.
- **Where and how I trained**: MacBook Air, Apple M4 CPU (no GPU), 30 epochs in about **2.5 hours**, image size 640, batch 16, seed 0, starting from the pretrained `yolo11n.pt` checkpoint, Ultralytics default settings (AdamW chosen automatically). I did no hyperparameter tuning.
- **Results on the same test split** (positive class: head):

| Model | `head` AP50 | `head` recall | `head` precision | Cost per 1,000 workers |
| :--- | :---: | :---: | :---: | :---: |
| **Finetuned YOLO11n (30 epochs)** | **0.865** | **0.870** | **0.986** | **32,632 EUR** (threshold 0.25) |
| Pretrained COCO, zero-shot | 0 | 0 | 0 | not defined |
| Trivial: predict nothing | 0 | 0 | n/a | 250,000 EUR |

  Overall for the finetuned model: mAP50 0.632, mAP50-95 0.406. Precision and recall per class are at the point of best F1 reported by Ultralytics, not at my chosen threshold.
- **COCO zero-shot**: COCO shares only `person` with this dataset. It has no class that separates a bare head from a hard hat, so for the positive class it scores 0 by construction. This gap is huge but not surprising; it only shows that finetuning is needed.
- **Predict nothing**: costs 5% x 1,000 x 5,000 = 250,000 EUR per 1,000 workers, because every violation passes. The finetuned model cuts that cost by about 87%.
- **What changed with training length**: a quick 2-epoch trial gave head recall of only 0.49 on the same test split. With 30 epochs it is 0.87. I did not test more than 30 epochs, so I do not know whether it has converged.

## Per-class evaluation and failure modes

| Class | Precision | Recall | AP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| `head` (positive) | 0.986 | 0.870 | 0.865 | 0.576 |
| `helmet` | 0.922 | 0.875 | 0.932 | 0.583 |
| `person` | 0.634 | 0.067 | 0.098 | 0.058 |

- **Which class carries the headline number**: `head`, because it is the violation and the one that costs money. Its AP50 is 0.865. `helmet` is stronger (0.932), which is expected: it is the majority class with a distinctive shape and colour.
- **Which class drags the average**: `person`. It pulls the overall mAP50 down to 0.632 even though it plays no role in the decision. That is why I do not use the overall number.
- **Where `head` fails**: recall of 0.870 means 3 of the 23 test heads are not detected (20/23). These are the expensive errors. Precision is very high (0.986), so false alarms are rare. I inspected the 3 misses at threshold 0.25: none is covered by a `helmet` box or gets even a weak `head` box (score 0.00 at a 0.01 cut-off). Two are the same head, seen from behind, boxed twice in a mirrored-looking image; the third is a tiny box at the top edge of a crowded aerial photo. These are unusual views rather than a general weakness on bare heads, but with only three cases I cannot say more.


## Cost matrix and operating point

- **Where each figure comes from** (all are my assumptions, not measured data):
  - **False negative, 5,000 EUR**: a regulatory fine, work stoppage and the expected cost of an injury from a worker on site without a hard hat. I have no source for the exact number.
  - **False positive, 5 EUR**: a short delay at the gate plus a quick manual check.
  - **Human review, 1.5 EUR**: 2 minutes at about 45 EUR/hour loaded cost.
  - **Base rate, 5%**: 5 of every 100 workers arrive without a hard hat. The test set has 11%; a real site would have to measure this.
- **How the cost is computed**: each worker is one case. For each threshold I count false negatives and false positives on the test set, re-weight them to the 5% base rate, and compute the expected cost per 1,000 workers.
- **Result of the sweep (0.01 to 0.95)**: the minimum is at threshold **0.25**, at **32,632 EUR per 1,000 workers**. A low threshold wins because a false negative costs 1,000 times more than a false positive. That cost is about 6.5 missed violations per 1,000 workers (roughly a 13% miss rate, consistent with the 0.87 recall).
- **Assumption that would most change my answer**: the cost of a false negative and the cost of a human check. A person checking every worker costs about 1,500 EUR per 1,000 workers (assuming no human errors), so the model alone only wins if a false negative costs less than roughly 230 EUR, or a human check costs more than about 32 EUR. These break-even points are approximate; I can verify them in the dashboard.

## What a person still has to check

- **Chosen thresholds**: pass below **0.25**, fail from **0.60**, human review in between.
- **Share sent to a person**: **1.6%** of cases. For 1,000 workers a day that is **16 cases per day**, at 2 minutes each: **about 0.5 hours per day**, which is **0.08 people** (assuming 6.5 productive hours per person per day). This is clearly realistic for a single site: it is a few minutes of one officer's time per day.
-- **The band does not help**: the cost with the band (32,632 EUR) is the same as with a single threshold. The 3 missed heads get no `head` box even at a 0.01 cut-off, so their score is 0 and they fall below any pass threshold. A review band can only catch cases that get a score. Almost all of the cost comes from these 3 misses; false alarms add only about 20 EUR.
- **Design priced against the alternatives** (per 1,000 workers, my assumptions):

| Strategy | Cost per 1,000 workers | People needed |
| :--- | :---: | :---: |
| Do nothing (everyone passes) | 250,000 EUR | 0 |
| A person checks everyone (assumes a perfect human) | 1,500 EUR | about 5.1 |
| Model alone, best single threshold (0.25) | 32,632 EUR | 0 |
| Model plus human-review band (0.25 to 0.60) | 32,632 EUR | 0.08 |

- **Honest conclusion**: under my own assumptions, the model alone is much better than doing nothing, but **it is not cheaper than having a person check everyone**, because a miss costs 5,000 EUR. The model should therefore not be the only decision maker. Its case rests on things my numbers do not capture: human checkers get tired and make mistakes (my "perfect human" assumption is generous to the human alternative), one model can cover many gates with fewer staff, and it leaves an audit trail. A fair recommendation is to use it as a second pair of eyes next to a person, not as a replacement, until recall on `head` improves.

## Limitations & next steps

- **Small test set, only 23 `head` instances.** All head numbers have wide uncertainty. With more time I would collect or label 200+ more head instances from the real camera angle and test on those.
- **Recall of 0.87 on the expensive class.** About 13% of bare heads are missed, and no threshold can recover a head the model does not detect. Next steps: train longer and check the training curves, try yolo11s, and use a larger image size, since small heads are probably the hardest case.
- **The `person` class is useless and drags the overall number.** I would remove it or train a model with only head and helmet.
- **Cost figures are assumptions.** The 5,000 EUR, 5% base rate and 1.5 EUR are not measured. With more time I would take real incident and fine data from a site and measure the true violation rate at the gate.
- **Data mismatch with a real gate.** The photos are not from a gate camera, so lighting, distance and angle differ. I would test on a few days of real footage before trusting any cost figure.
- **Privacy.** A gate camera records workers' faces, which in the EU falls under data protection rules (GDPR). A real deployment would need a legal basis, a retention limit, and probably to store only the decision, not the images.

## Generative AI use disclosure

I used Claude (Anthropic) as an assistant. Firstly, I did another training with a different dataset and it helped me see what I was doing wrong, so I had to change the entire project. 
It helped me debug the pipeline and write and correct the code for `train.py`, `app.py` and `costing.py`, and structure this report from my own results. 
I chose the problem and the cost assumptions, downloaded the dataset, ran the training on my own laptop, and checked the dashboard locally. 
All numbers in this report come from my own run, written into `weights/eval_artifacts.json` by `train.py`. 