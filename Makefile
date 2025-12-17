PY:=python3
DATASET?=dataset/melanoma_cancer_dataset
BACKBONE?=resnet18
EPOCHS?=12
BATCH?=32
IMG?=224
WEIGHTS?=backend/ml/weights/skin_$(BACKBONE).pth
REPORT_DIR?=reports/skin_$(BACKBONE)
THRESHOLD?=0.45

.PHONY: train eval setup-model train-eval explain

train:
	@mkdir -p $(dir $(WEIGHTS))
	$(PY) scripts/train_skin.py --data-dir $(DATASET) --split-subdirs --backbone $(BACKBONE) --epochs $(EPOCHS) --batch-size $(BATCH) --output $(WEIGHTS)

eval:
	@mkdir -p $(REPORT_DIR)
	$(PY) scripts/eval_skin.py --weights $(WEIGHTS) --data-dir $(DATASET)/test --backbone $(BACKBONE) --out $(REPORT_DIR)

setup-model:
	@echo "LESION_MODEL_WEIGHTS=$(WEIGHTS)" >> .env
	@echo "LESION_FORCE_THRESHOLD=1" >> .env
	@echo "LESION_MALIGNANT_THRESHOLD=$(THRESHOLD)" >> .env
	@echo "Wrote LESION_* settings to .env. Restart backend."

train-eval: train eval setup-model

explain:
	@echo "Run Grad-CAM by uploading an image via the app; the API will return an explanation image when a model is active."
