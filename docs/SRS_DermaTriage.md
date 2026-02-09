# Software Requirements Specification

**for**

## DermaTriage: A Clinically Aligned Framework for Dermatological Diagnostics

**Version 1.0**

**Prepared by**

**Team Number: C1**

Arjungopal Anilkumar (Team Lead)  
CB.SC.U4AIE23271

Suryansh Ram Menon  
CB.SC.U4AIE23255

Divagar  
CB.SC.U4AIE23223

**Project Manager:** Dr. Keerthika  
**Course:** 22AIE311 SOFTWARE ENGINEERING  
**Date:** 09/02/2026

---

## Contents

1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [Specific Requirements](#3-specific-requirements)
4. [Non-functional Requirements](#4-non-functional-requirements)
5. [Other Requirements](#5-other-requirements)
6. [Appendix A - Activity Log](#appendix-a---activity-log)

---

## Revisions

| Version | Primary Author(s) | Description of Version | Date Completed |
|---------|------------------|------------------------|----------------|
| 1.0 | Arjungopal Anilkumar, Suryansh Ram Menon, Divagar | Initial SRS document for DermaTriage system | 09/02/2026 |

---

## 1 Introduction

This Software Requirements Specification (SRS) document provides a comprehensive description of DermaTriage, an AI-powered dermatological diagnostic system designed to assist healthcare professionals in the early detection and classification of skin diseases. The system leverages state-of-the-art deep learning architectures and a carefully curated dataset of over 300,000 dermatological images to achieve clinical-grade accuracy in identifying 25 common skin conditions.

### 1.1 Document Purpose

This document specifies the complete software requirements for DermaTriage version 1.0. It is intended for use by the development team, project stakeholders, quality assurance personnel, and academic reviewers. The SRS describes all functional and non-functional requirements, external interface requirements, design constraints, and quality attributes expected from the system.

The primary objectives of this document are to:
- Establish a clear and unambiguous specification of all system requirements
- Provide a baseline for validation and verification activities
- Serve as a contractual agreement between the development team and project stakeholders
- Enable accurate project planning and resource estimation

### 1.2 Product Scope

DermaTriage is a comprehensive AI diagnostic framework designed to classify dermatological conditions with a target accuracy of 95-97%. The system addresses a critical gap in healthcare accessibility by providing accurate preliminary diagnoses for skin conditions, particularly in regions with limited access to specialized dermatological care.

**Key Benefits:**
- **Early Detection**: Identifies potentially malignant conditions (melanoma, BCC, SCC, AK) with 94-96% recall, minimizing false negatives
- **Accessibility**: Enables preliminary screening in remote or underserved areas
- **Efficiency**: Reduces diagnostic workload for dermatologists by triaging cases
- **Diversity**: Trained on Fitzpatrick skin tone scale (I-VI) to ensure equitable performance across all populations
- **Clinical Alignment**: 25-class taxonomy aligned with standard dermatological practice

The system encompasses the complete ML pipeline from data ingestion through model inference, including dataset preparation, two-stage training (pretrain + finetune), and production deployment on GPU clusters.

### 1.3 Intended Audience and Document Overview

This document is intended for multiple audiences:

**Developers**: Sections 2.3, 3.1, 3.2, and 3.3 provide detailed technical requirements and system architecture  
**Project Managers**: Sections 2.1, 2.2, and 4.1 outline scope, functionality, and performance metrics  
**QA/Testing Teams**: Sections 3.2, 3.3, and 4.3 specify testable requirements and quality attributes  
**Academic Reviewers**: All sections provide comprehensive documentation of the software engineering process  
**Medical Stakeholders**: Sections 2.2 and 4.2 describe clinical functionality and safety requirements

The remainder of this document is organized as follows: Section 2 provides an overall system description including product perspective, functionality, and constraints. Section 3 details specific functional requirements including external interfaces and use cases. Section 4 specifies non-functional requirements including performance, safety, and quality attributes.

### 1.4 Definitions, Acronyms and Abbreviations

| Term | Definition |
|------|------------|
| **AI** | Artificial Intelligence |
| **AK** | Actinic Keratosis (pre-cancerous lesion) |
| **AMP** | Automatic Mixed Precision |
| **API** | Application Programming Interface |
| **BCC** | Basal Cell Carcinoma |
| **BS** | Batch Size |
| **ConvNeXt** | Convolutional Neural Network (modernized architecture) |
| **DermNet** | Dermatology image dataset |
| **FN** | False Negative |
| **FP** | False Positive |
| **GPU** | Graphics Processing Unit |
| **HAM10000** | Human Against Machine 10000 (dataset) |
| **ISIC** | International Skin Imaging Collaboration |
| **LR** | Learning Rate |
| **MixUp** | Data augmentation technique blending image pairs |
| **PAD-UFES-20** | Public dataset of smartphone skin lesion images |
| **SCC** | Squamous Cell Carcinoma |
| **SLURM** | Simple Linux Utility for Resource Management |
| **SRS** | Software Requirements Specification |
| **TTA** | Test-Time Augmentation |
| **UI** | User Interface |

### 1.5 Document Conventions

This document follows IEEE SRS formatting standards:
- **Font**: Arial 11pt for body text, 12pt for headings
- **Margins**: 1" on all sides
- **Spacing**: Single-spaced paragraphs
- **Priority Levels**: High, Medium, Low (for requirements and use cases)
- **Requirement IDs**: Functional requirements prefixed with "F", Use cases with "U"
- **Italics**: Used for comments and emphasis
- **Tables**: Used for structured information presentation

### 1.6 References and Acknowledgments

1. IEEE Std 830-1998, IEEE Recommended Practice for Software Requirements Specifications
2. Liu, Z., Mao, H., Wu, C. Y., et al. (2022). "A ConvNet for the 2020s." CVPR.
3. ISIC Archive: https://www.isic-archive.com/
4. Fitzpatrick, T. B. (1988). "The validity and practicality of sun-reactive skin types I through VI."
5. Template adapted from GMU (Dr. Rob Pettit)

**Acknowledgments**: Dataset providers (ISIC, HAM10000, DermNet, Fitzpatrick17k), GPU cluster infrastructure team, Dr. Keerthika (Project Manager)

---

## 2 Overall Description

### 2.1 Product Overview

DermaTriage is a standalone AI diagnostic system designed to assist healthcare professionals in the classification of dermatological conditions. The system is positioned as a clinical decision support tool rather than a replacement for professional medical diagnosis.

**System Context:**  
The product operates within the broader healthcare technology ecosystem, interfacing with medical imaging devices, electronic health record (EHR) systems, and clinical workflows. It serves as a triage tool that processes dermatological images and provides preliminary diagnostic predictions with associated confidence scores.

**High-Level Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    DermaTriage System                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Data       │───>│   Training   │───>│  Inference   │  │
│  │ Preparation  │    │   Pipeline   │    │   Engine     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │          │
│         v                    v                    v          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        ConvNeXt-Large Model (95-97% accuracy)        │  │
│  │       25 Disease Classes | Fitzpatrick I-VI          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
         ↑                                          ↓
    Image Input                              Diagnosis Output
  (Smartphones,                          (Class + Confidence +
   Clinical Cameras)                      Attention Map)
```

**External Interfaces:**
- **Input**: Medical images (JPEG/PNG) from clinical cameras, smartphones, or dermatoscopes
- **Output**: Disease classification, confidence scores, visual attention maps
- **Training Infrastructure**: GPU cluster (RTX 6000 Ada, A100) via SLURM job scheduler
- **Storage**: Network-attached storage for datasets (~32GB total)

### 2.2 Product Functionality

The system provides the following major functions:

- **Multi-Dataset Consolidation**: Unify 6+ dermatological datasets (300k+ images) into standardized format
- **Two-Stage Training Pipeline**: Pretrain on general dermatology data, finetune on diverse skin tones
- **25-Class Disease Classification**: Melanoma, BCC, SCC, AK, nevus, eczema, psoriasis, candida, etc.
- **Cancer Prioritization**: 3x weight on cancer classes to minimize false negatives (target: 94-96% recall)
- **Test-Time Augmentation**: Average predictions over augmented versions for improved accuracy
- **Confidence Scoring**: Provide probability distributions across all 25 classes
- **Visual Explainability**: Generate attention maps showing diagnostic regions
- **Batch Inference**: Process multiple images efficiently on GPU
- **Model Checkpointing**: Save/resume training, version model artifacts
- **Performance Monitoring**: Track per-class accuracy, recall, and confusion matrices

### 2.3 Design and Implementation Constraints

**Hardware Constraints:**
- Requires GPU with ≥40GB VRAM for training (RTX 6000 Ada or A100)
- Training batch size limited to 32-64 based on GPU memory
- Inference requires GPU for real-time (<2s per image) performance
- Dataset storage requires ~32GB disk space

**Software Constraints:**
- PyTorch ≥2.0 (for torch.compile optimization)
- Python 3.9+
- CUDA 11.8+ for GPU acceleration
- Linux OS (for SLURM cluster integration)

**Regulatory Constraints:**
- Must comply with medical device software standards (IEC 62304)
- HIPAA compliance for any patient data handling
- Disclaimer required: "Not a substitute for professional medical diagnosis"

**Development Constraints:**
- ConvNeXt-Large architecture (no deviation from proven backbone)
- Focal Loss with label smoothing (for class imbalance)
- Cosine annealing learning rate schedule with warmup
- MixUp augmentation (α=0.2) for training robustness

**Deployment Constraints:**
- Model size: ~800MB (ConvNeXt-Large)
- Inference latency: <2 seconds per image on GPU, <10 seconds on CPU
- API response time: <3 seconds end-to-end

### 2.4 Assumptions and Dependencies

**Assumptions:**
- Input images are of sufficient quality (min 224x224 pixels, well-lit)
- Image capture follows standard dermatological protocols
- Users (healthcare professionals) understand AI limitations and will use clinical judgment
- Dataset labels are accurate (reliance on curated medical datasets)
- GPU infrastructure availability for training and inference

**Dependencies:**
- **External Datasets**: ISIC, HAM10000, DermNet, Fitzpatrick17k (licensed under CC BY-NC)
- **Third-Party Libraries**: PyTorch, TorchVision, NumPy, Pandas, Pillow, tqdm
- **Pretrained Weights**: ImageNet-22k pretrained ConvNeXt-Large (from Meta/Facebook Research)
- **Kaggle API**: For automated dataset downloads
- **SLURM Scheduler**: For cluster job management
- **Conda Environment Management**: For reproducible Python environments

**Critical Dependencies:**
- Continued availability of dataset sources
- PyTorch framework compatibility across versions
- GPU driver/CUDA library stability

---

## 3 Specific Requirements

### 3.1 External Interface Requirements

#### 3.1.1 Hardware Interfaces

**Training Hardware:**
- **GPU Requirements**: NVIDIA RTX 6000 Ada (48GB VRAM) or A100 (40GB VRAM)
  - Interface: PCIe Gen 4 x16
  - Minimum CUDA Compute Capability: 8.0
  - Driver: NVIDIA 525+ with CUDA 11.8+

**Inference Hardware:**
- **Clinical Deployment**: NVIDIA T4 (16GB) minimum for real-time inference
- **Image Capture Devices**: 
  - Standard clinical dermatology cameras (USB interface)
  - Consumer smartphones (iOS/Android) via network upload
  - Dermatoscopes with digital output (USB/Bluetooth)

**Storage:**
- Network-attached storage (NAS) for dataset (≥50GB capacity, ≥100MB/s throughput)

#### 3.1.2 Software Interfaces

**Operating System:**
- Linux (Ubuntu 20.04+ or CentOS 7+) for training cluster
- Cross-platform inference (Linux, macOS, Windows via PyTorch)

**SLURM Workload Manager:**
- Interface for job submission (`sbatch`), monitoring (`squeue`), and resource allocation
- Auto-configuration scripts detect GPU type and optimize parameters

**Kaggle API:**
- Python interface for automated dataset downloads
- Requires authentication via `~/.kaggle/kaggle.json`

**No External Application Integration Required**:  
This is a standalone system with file-based input/output. Future versions may integrate with PACS (Picture Archiving and Communication System) or EHR systems via HL7/FHIR APIs.

### 3.2 Functional Requirements

**F1:** The system shall consolidate images from ISIC, HAM10000, DermNet, Massive, PAD-UFES-20, and Fitzpatrick17k datasets into a unified directory structure organized by disease class.

**F2:** The system shall normalize all disease labels to a standardized 25-class taxonomy (melanoma, bcc, scc, ak, nevus, seborrheic_keratosis, angioma, eczema, psoriasis, acne, dermatitis, urticaria, bullous, candida, herpes, scabies, impetigo, nail_fungus, chickenpox, shingles, wart, alopecia, hyperpigmentation, lupus, healthy).

**F3:** The system shall perform stratified train/validation/test splits (80%/10%/10%) ensuring balanced representation of all classes.

**F4:** The system shall train a ConvNeXt-Large model in Stage 1 on all datasets except Fitzpatrick17k using focal loss with 3x weight for cancer classes.

**F5:** The system shall finetune the Stage 1 model on Fitzpatrick17k dataset in Stage 2 using a 10x lower learning rate (1e-5) to adapt to diverse skin tones.

**F6:** The system shall apply MixUp augmentation (α=0.2) during training to improve model robustness and calibration.

**F7:** The system shall use Cosine Annealing with Warm Restarts for learning rate scheduling to ensure stable convergence.

**F8:** The system shall implement Test-Time Augmentation (TTA) during inference by averaging predictions over 5 augmented versions of the input image.

**F9:** The system shall generate per-class probability distributions as output, not just the top prediction.

**F10:** The system shall produce visual attention maps (GradCAM or similar) highlighting diagnostically relevant image regions.

**F11:** The system shall save model checkpoints after each epoch, retaining only the best model based on mean recall.

**F12:** The system shall log training metrics (loss, accuracy, per-class recall, cancer recall) to enable progress monitoring.

**F13:** The system shall validate input images for minimum resolution (224x224) and supported formats (JPEG, PNG).

**F14:** The system shall handle corrupt or unreadable images gracefully by skipping them and logging errors.

**F15:** The system shall inference on GPU if available, otherwise fall back to CPU with a warning.

### 3.3 Use Case Model

![Use Case Diagram](use_case_diagram.png)

```
┌─────────────────────────────────────────────────────────────┐
│                   DermaTriage System                         │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                                                        │  │
│  │  (U1) Prepare Datasets ◄───────┐                     │  │
│  │                                   │                     │  │
│  │  (U2) Train Stage 1          │                     │  │
│  │        (Pretrain)               │   ML Engineer     │  │
│  │                                   │     (Actor)       │  │
│  │  (U3) Train Stage 2          │                     │  │
│  │        (Finetune)               │                     │  │
│  │                                   │                     │  │
│  │  (U4) Classify Skin Lesion  ◄───┘                     │  │
│  │         «extends»                                      │  │
│  │           └──> (U5) Generate Attention Map            │  │
│  │                                                        │  │
│  │  (U6) Batch Process Images ◄──── Clinician (Actor)   │  │
│  │                                                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### 3.3.1 Use Case #U1: Prepare Datasets

**Author:** Suryansh Ram Menon

**Purpose:** Consolidate and preprocess all dermatological datasets into a unified format suitable for two-stage training.

**Requirements Traceability:** F1, F2, F3

**Priority:** High (prerequisite for all downstream tasks)

**Preconditions:**
- All dataset ZIP files downloaded to `datasets/` directory
- Kaggle API configured for automated downloads
- Sufficient disk space available (≥50GB)

**Postconditions:**
- Two output directories created: `data/stage1/` and `data/stage2_finetune/`
- Each contains `train/`, `val/`, `test/` subdirectories with images organized by disease class
- Metadata files (`hierarchy.json`) saved with class lists and split statistics

**Actors:** ML Engineer

**Extends:** None

**Includes:** None

**Notes/Issues:** Must handle inconsistent labeling across datasets. Uses hardlink for space efficiency where possible.

**Flow of Events:**

**Basic Flow:**

| Actor's Action | System's Response |
|---------------|-------------------|
| 1. Runs `python scripts/prepare_unified_v4.py` | 2. Scans `datasets/` directory for known dataset folders |
| | 3. Loads ISIC metadata CSV, extracts image paths and labels |
| | 4. Normalizes labels using DISEASE_MAP dictionary |
| | 5. Repeats for HAM10000, DermNet, Massive, PAD-UFES datasets |
| | 6. Groups all samples by disease class |
| | 7. Performs stratified 80/10/10 split for each class |
| | 8. Creates output directory structure |
| | 9. Copies/hardlinks images to appropriate directories |
| | 10. Saves metadata JSON files |
| | 11. Displays summary: train/val/test counts per class |

**Alternative Flow:**

| Actor's Action | System's Response |
|---------------|-------------------|
| | If a dataset folder is missing |
| | - Logs warning message |
| | - Continues with available datasets |

**Exceptions:**

| Actor's Action | System's Response |
|---------------|-------------------|
| | If no datasets found |
| | - Raises error: "No valid datasets found in datasets/" |
| | - Suggests running download script |

#### 3.3.2 Use Case #U2: Train Stage 1 (Pretrain)

**Author:** Arjungopal Anilkumar

**Purpose:** Train ConvNeXt-Large model on all datasets except Fitzpatrick to learn core dermatological patterns.

**Requirements Traceability:** F4, F6, F7, F11, F12

**Priority:** High

**Preconditions:**
- `data/stage1/train/` and `data/stage1/val/` directories exist with prepared data
- GPU with ≥40GB VRAM available
- Conda environment `skindoc` activated

**Postconditions:**
- Trained model saved to `checkpoints/stage1_best.pth`
- Training logs saved to `logs/train_<timestamp>.log`
- Validation accuracy ≥93%, mean recall ≥91%

**Actors:** ML Engineer

**Extends:** None

**Includes:** None

**Flow of Events:**

**Basic Flow:**

| Actor's Action | System's Response |
|---------------|-------------------|
| 1. Runs `python scripts/train_stage1_optimized.py --variant convnext_large --batch_size 32 --epochs 50` | 2. Detects GPU type (A100/RTX6000) and sets mixed precision dtype |
| | 3. Loads ConvNeXt-Large with ImageNet pretrained weights |
| | 4. Creates DataLoaders with weighted sampling (3x for cancer) |
| | 5. Initializes Focal Loss, AdamW optimizer, Cosine scheduler |
| | 6. For each epoch: |
| | 7.   - Apply MixUp augmentation to batches |
| | 8.   - Forward pass with mixed precision |
| | 9.   - Compute focal loss + backprop |
| | 10.  - Update weights via AdamW |
| | 11.  - Log metrics (loss, accuracy, recall) |
| | 12.  - Validate on val set |
| | 13.  - Save checkpoint if mean recall improved |
| | 14. Early stopping after 10 epochs without improvement |
| | 15. Displays final metrics: Best recall = 93.x% |

**Alternative Flow:**

| Actor's Action | System's Response |
|---------------|-------------------|
| Actor provides `--resume checkpoints/latest.pth` | - Loads model state and resumes training from last epoch |

**Exceptions:**

| Actor's Action | System's Response |
|---------------|-------------------|
| | If CUDA Out of Memory error occurs |
| | - Logs error with suggestion to reduce batch size |
| | - Exits gracefully |

#### 3.3.3 Use Case #U3: Train Stage 2 (Finetune)

**Author:** Arjungopal Anilkumar

**Purpose:** Finetune Stage 1 model on Fitzpatrick17k to adapt to diverse skin tones (Fitzpatrick I-VI).

**Requirements Traceability:** F5, F6, F7, F11, F12

**Priority:** High

**Preconditions:**
- Stage 1 model exists at `checkpoints/stage1_best.pth`
- `data/stage2_finetune/train/` and `data/stage2_finetune/val/` prepared
- Fitzpatrick dataset rsynced to cluster

**Postconditions:**
- Final production model saved to `checkpoints/stage2_final.pth`
- Validation accuracy ≥95%, cancer recall ≥94%

**Actors:** ML Engineer

**Flow of Events:**

**Basic Flow:**

| Actor's Action | System's Response |
|---------------|-------------------|
| 1. Runs `python scripts/train_stage2_finetune_optimized.py --checkpoint checkpoints/stage1_best.pth --lr 1e-5 --epochs 20` | 2. Loads Stage 1 checkpoint |
| | 3. Reinitializes optimizer with 10x lower LR |
| | 4. Trains on Fitzpatrick dataset with same protocol |
| | 5. Saves final model to stage2_final.pth |

#### 3.3.4 Use Case #U4: Classify Skin Lesion

**Author:** Divagar

**Purpose:** Provide clinical diagnosis prediction for a single dermatological image.

**Requirements Traceability:** F8, F9, F13, F14, F15

**Priority:** High

**Preconditions:**
- Trained model exists (`checkpoints/stage2_final.pth`)
- Input image provided (minimum 224x224 pixels)

**Postconditions:**
- JSON output with class probabilities and top-3 predictions
- Inference completed in <2 seconds (GPU) or <10 seconds (CPU)

**Actors:** Clinician

**Flow of Events:**

**Basic Flow:**

| Actor's Action | System's Response |
|---------------|-------------------|
| 1. Uploads dermoscopic/clinical image via web interface | 2. Validates image format and resolution |
| | 3. Applies preprocessing (resize to 384x384, normalize) |
| | 4. Creates 5 augmented versions (original, flip, rotate) |
| | 5. Runs inference on all versions with mixed precision |
| | 6. Averages predictions (TTA) |
| | 7. Computes softmax probabilities |
| | 8. Returns JSON: `{"top_class": "melanoma", "confidence": 0.87, "probabilities": {...}}` |

**Exceptions:**

| Actor's Action | System's Response |
|---------------|-------------------|
| User uploads 100x100 image | - Returns error: "Image resolution too low (min 224x224)" |

#### 3.3.5 Use Case #U5: Generate Attention Map

**Author:** Divagar

**Purpose:** Visualize which regions of the image the model focused on for its prediction (explainability).

**Requirements Traceability:** F10

**Priority:** Medium

**Preconditions:**
- Use Case U4 completed successfully
- Model loaded in memory

**Postconditions:**
- Heatmap overlay image generated showing attention regions

**Actors:** Clinician

**Extends:** Use Case #U4

**Flow of Events:**

**Basic Flow:**

| Actor's Action | System's Response |
|---------------|-------------------|
| | 1. Computes GradCAM from final convolutional layer |
| | 2. Upsamples heatmap to original image size |
| | 3. Overlays heatmap (red = high attention) on original image |
| | 4. Returns image as PNG |

#### 3.3.6 Use Case #U6: Batch Process Images

**Author:** Suryansh Ram Menon

**Purpose:** Efficiently process multiple images in a single batch for clinical screening workflows.

**Requirements Traceability:** F9, F13, F15

**Priority:** Medium

**Actors:** Clinician

**Flow of Events:**

**Basic Flow:**

| Actor's Action | System's Response |
|---------------|-------------------|
| 1. Uploads folder with 50 dermoscopic images | 2. Validates all images |
| | 3. Batches into groups of 32 |
| | 4. Runs inference on each batch (parallelized on GPU) |
| | 5. Returns CSV with predictions for all images |
| | 6. Total time: ~1 second per image on GPU |

---

## 4 Non-functional Requirements

### 4.1 Performance Requirements

**P1:** Training shall complete Stage 1 (50 epochs on 280k images) within 48-72 hours on RTX 6000 Ada or A100 GPU.

**P2:** Training shall complete Stage 2 (20 epochs on 16k images) within 8-12 hours on the same hardware.

**P3:** Inference shall process a single image in <2 seconds on GPU (NVIDIA T4 or better), <10 seconds on CPU (Intel Xeon or equivalent).

**P4:** Batch inference shall achieve ≥20 images/second throughput on GPU for batch size 32.

**P5:** Model loading time shall be <5 seconds from checkpoint file to ready-for-inference state.

**P6:** Memory consumption during inference shall not exceed 8GB GPU VRAM, 16GB system RAM.

**P7:** The system shall support concurrent inference requests (up to 4 simultaneous batches on A100 via MPS).

**P8:** Dataset preparation script shall process 300k images in <30 minutes using hardlinks.

**Rationale:** These performance targets ensure the system is viable for both research (training) and clinical deployment (inference). GPU acceleration is essential for real-time diagnostic support.

### 4.2 Safety and Security Requirements

**S1:** The system shall display a prominent disclaimer on all outputs: *"This is a diagnostic aid. Not a substitute for professional medical evaluation."*

**S2:** The system shall not store any patient-identifiable information (PII) unless explicitly configured for clinical deployment with HIPAA-compliant storage.

**S3:** All image uploads shall be transmitted over HTTPS (TLS 1.3+) if deployed as a web service.

**S4:** The system shall validate uploaded images for malicious payloads (via magic number verification, not just extension checking).

**S5:** Cancer-class predictions with confidence >75% shall be flagged with a high-priority warning: *"Possible malignant condition detected. Urgent dermatological consultation recommended."*

**S6:** The system shall log all diagnostic predictions with timestamps and model version for auditability (without storing images unless consented).

**S7:** The system shall rate-limit API requests to prevent abuse (max 100 requests/hour per IP in production).

**S8:** Model weights and training data shall not be publicly accessible (stored on secure cluster with SSH key authentication).

**S9:** The system shall gracefully handle adversarial or out-of-distribution images by outputting low confidence across all classes (<30% max probability).

**Rationale:** Medical AI systems must prioritize patient safety through clear disclaimers, secure data handling, and conservative predictions (especially for cancer). Regulatory compliance (FDA, CE mark, HIPAA) may be required for clinical deployment.

### 4.3 Software Quality Attributes

#### 4.3.1 Reliability

**R1:** The system shall achieve ≥95% accuracy on held-out test set (not used during training/validation).

**R2:** Cancer recall (sensitivity) shall be ≥94% to minimize false negatives (missing melanoma/BCC/SCC/AK).

**R3:** The system shall recover from transient GPU errors (e.g., CUDA out of memory) by automatically reducing batch size and retrying.

**R4:** Training shall auto-save checkpoints every epoch to prevent data loss in case of cluster job preemption.

**Rationale:** Reliability in medical AI means both statistical performance (accuracy/recall) and robustness to failures. False negatives for cancer are unacceptable.

#### 4.3.2 Maintainability

**M1:** All hyperparameters shall be configurable via command-line arguments (no hardcoding).

**M2:** Code shall follow PEP 8 style guide with maximum line length 120 characters.

**M3:** Each script shall include docstrings for all functions/classes following Google style.

**M4:** Model architecture, training configuration, and class list shall be saved in checkpoint metadata for reproducibility.

**M5:** The system shall support PyTorch version upgrades with a maximum 2-week migration window for breaking changes.

**Rationale:** Medical AI systems have long lifespans and require continuous updates (new datasets, regulatory changes). Well-documented, modular code is essential.

#### 4.3.3 Portability

**PT1:** Inference code shall run on Linux, macOS, and Windows without modification (via PyTorch cross-platform support).

**PT2:** The system shall support deployment on any NVIDIA GPU with Compute Capability ≥7.0 (Volta or newer).

**PT3:** CPU-only inference shall be supported for edge deployment (albeit with reduced performance).

**PT4:** Model checkpoints shall be framework-agnostic (exportable to ONNX format for non-PyTorch deployment).

**Rationale:** Clinical deployment environments vary widely (cloud, on-premise, edge devices). Maximum portability reduces integration barriers.

---

## 5 Other Requirements

**Legal Requirement:** The system shall include an open-source license (MIT or Apache 2.0) for the codebase, with appropriate attribution for third-party datasets and libraries.

**Ethical Requirement:** The system shall be trained on diverse skin tones (Fitzpatrick I-VI) to avoid bias, as validated by per-skin-tone performance metrics in Stage 2.

**Documentation Requirement:** A user manual and API reference shall be provided in Markdown format in the `docs/` directory.

---

## Appendix A - Activity Log

### Meeting History

| Date | Duration | Attendees | Purpose | Outcomes |
|------|----------|-----------|---------|----------|
| 02/02/2026 | 2 hours | All team members | Project kickoff, dataset survey | Identified 6 datasets, decided on ConvNeXt architecture |
| 04/02/2026 | 3 hours | All team members | Dataset download and validation | Automated download scripts, validated 289k images |
| 05/02/2026 | 4 hours | Arjun, Suryansh | Training script development | Implemented focal loss, weighted sampling |
| 07/02/2026 | 2 hours | All team members | Two-stage strategy review | Approved pretrain+finetune approach |
| 08/02/2026 | 5 hours | Arjun, Suryansh | Optimization implementation | Added MixUp, TTA, Cosine scheduler |
| 09/02/2026 | 3 hours | All team members + Dr. Keerthika | SRS review and finalization | Approved SRS, clarified use cases |

### Individual Contributions

**Arjungopal Anilkumar (Team Lead):**
- Sections 1.1-1.6, 2.1-2.4 (Introduction and Overall Description)
- Use Cases U2, U3 (Training pipeline use cases)
- Section 4.1 (Performance Requirements)
- Coordination of team activities and document assembly

**Suryansh Ram Menon:**
- Section 3.1, 3.2 (External Interfaces and Functional Requirements)
- Use Cases U1, U6 (Dataset preparation, batch processing)
- Section 4.2 (Safety and Security Requirements)
- Implementation of download scripts and dataset consolidation

**Divagar:**
- Use Cases U4, U5 (Inference and explainability)
- Section 4.3 (Software Quality Attributes)
- Use case diagrams and activity flows
- Section 5 and Appendix A

**Collaborative Work:**
- Requirements brainstorming (all members)
- Technical architecture design (Arjun, Suryansh)
- Quality attribute definitions (all members)
- Document review and proofreading (all members)

### Client Meetings

**Meeting with Dr. Keerthika (Project Manager) - 09/02/2026:**
- **Duration:** 1 hour
- **Discussion Points:**
  - Reviewed two-stage training rationale
  - Clarified safety requirements for medical AI
  - Discussed performance targets (95-97% accuracy)
  - Approved 25-class disease taxonomy
- **Action Items:**
  - Finalize SRS document by 09/02/2026 ✓
  - Begin Stage 1 training upon dataset completion
  - Prepare progress presentation for mid-term review

---

**Document Prepared by Team C1**  
**Submitted: 09/02/2026**
