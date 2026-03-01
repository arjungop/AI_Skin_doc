# DermaTriage: AI-Powered Dermatological Diagnostic System

## Comprehensive Project Report

**Version 1.0 | March 2026**

**Team C1**
- Arjungopal Anilkumar (Team Lead) — CB.SC.U4AIE23271
- Suryansh Ram Menon — CB.SC.U4AIE23255
- Divagar — CB.SC.U4AIE23223

**Project Manager:** Dr. Keerthika
**Course:** 22AIE311 Software Engineering
**Institution:** Amrita Vishwa Vidyapeetham

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Backend — FastAPI Server](#4-backend--fastapi-server)
5. [Frontend — React Application](#5-frontend--react-application)
6. [Machine Learning Pipeline](#6-machine-learning-pipeline)
7. [Database Design](#7-database-design)
8. [Authentication & Security](#8-authentication--security)
9. [AI/LLM Integration](#9-aillm-integration)
10. [Training Results](#10-training-results)
11. [Deployment Architecture](#11-deployment-architecture)
12. [Screenshots & UI Showcase](#12-screenshots--ui-showcase)
13. [API Reference](#13-api-reference)
14. [Testing & Validation](#14-testing--validation)
15. [Future Enhancements](#15-future-enhancements)
16. [Appendix](#16-appendix)

---

## 1. Executive Summary

DermaTriage is a full-stack AI-powered dermatological diagnostic platform that combines deep learning–based skin disease classification with telemedicine features, enabling patients to receive instant AI-assisted skin analysis and connect with board-certified dermatologists for follow-up care.

### Key Achievements

| Metric | Value |
|--------|-------|
| **Model Accuracy** | **98.22%** validation accuracy (epoch 24/30) |
| **Disease Classes** | 20 skin conditions |
| **Training Dataset** | 259,154 images from 5 merged datasets |
| **Architecture** | ConvNeXt-Base (87.6M parameters) |
| **Backend API Endpoints** | 70+ RESTful routes |
| **Frontend Pages** | 28 distinct views |
| **Database Tables** | 25+ ORM models |
| **LLM Providers Supported** | 4 (Gemini, Azure OpenAI, OpenAI, Ollama) |

### Problem Statement

Access to dermatological expertise is severely limited globally—there are only ~0.35 dermatologists per 100,000 people in low- and middle-income countries. Skin cancer (melanoma, BCC, SCC) survival rates drop dramatically without early detection. DermaTriage addresses this by providing:

1. **Instant AI screening** — Upload a photo and receive a classification within seconds
2. **Doctor consultation** — Real-time chat with dermatologists for follow-up
3. **Personalized care** — Treatment plans, routine tracking, and skin journey logging
4. **Environmental awareness** — UV index and weather-based skin advisories

---

## 2. System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DermaTriage Platform                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────────────┐  │
│  │   React + Vite   │───>│  FastAPI Backend  │───>│   MySQL / SQLite DB   │  │
│  │  (Port 5173)     │    │   (Port 8000)     │    │                        │  │
│  │                   │    │                   │    └────────────────────────┘  │
│  │  • TailwindCSS    │    │  • 15 Route       │                               │
│  │  • Framer Motion  │    │    Modules         │    ┌────────────────────────┐ │
│  │  • Three.js       │    │  • JWT Auth        │───>│   LLM Service          │ │
│  │  • Leaflet Maps   │    │  • Rate Limiting   │    │  Gemini / Azure / GPT  │ │
│  │  • React Router   │    │  • CORS + Security │    │  / Ollama (Local)      │ │
│  └─────────────────┘    │  • WebSocket Ready  │    └────────────────────────┘ │
│                          │                     │                               │
│                          │    ┌─────────────┐  │    ┌────────────────────────┐ │
│                          │    │ ML Inference │  │───>│  Azure Blob Storage    │ │
│                          │    │ PyTorch +    │  │    │  (Image Storage)       │ │
│                          │    │ ConvNeXt     │  │    └────────────────────────┘ │
│                          │    │ + Grad-CAM   │  │                               │
│                          │    └─────────────┘  │                               │
│                          └──────────────────┘                                  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                     Jarvis Training Pipeline                             │  │
│  │  RTX A6000 (48 GB) │ ConvNeXt-Base │ 259K images │ 20 classes           │  │
│  │  FocalLoss + EMA + Mixup/CutMix + OneCycleLR + bfloat16 AMP            │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
Patient                     Frontend (React)                Backend (FastAPI)              ML / LLM
  │                              │                               │                          │
  │──── Upload Image ───────────>│                               │                          │
  │                              │──── POST /lesions/predict ───>│                          │
  │                              │                               │──── predict_image() ────>│
  │                              │                               │<─── {label, prob, cam} ──│
  │                              │                               │──── Save to DB           │
  │                              │<──── {prediction, risk} ─────│                          │
  │<──── Show Results ──────────│                               │                          │
  │                              │                               │                          │
  │──── Request Diagnosis ──────>│                               │                          │
  │                              │──── POST /lesions/{id}/report>│                          │
  │                              │                               │──── LLM chat() ─────────>│
  │                              │                               │<─── Detailed report ─────│
  │                              │<──── {diagnosis report} ─────│                          │
  │<──── Show Report ───────────│                               │                          │
```

---

## 3. Technology Stack

### Backend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Framework** | FastAPI | 0.124.4 | Async REST API with auto OpenAPI docs |
| **Language** | Python | 3.11+ | Backend logic |
| **ORM** | SQLAlchemy | 2.x | Database abstraction |
| **Auth** | python-jose + bcrypt | Latest | JWT token generation, password hashing |
| **ML Framework** | PyTorch | 2.x | Model inference + Grad-CAM |
| **Image Processing** | Pillow (PIL) | Latest | Image validation & manipulation |
| **Rate Limiting** | SlowAPI | Latest | DDoS/abuse prevention |
| **Cloud Storage** | Azure Blob SDK | 12.27.1 | Medical image storage |
| **Server** | Uvicorn (ASGI) | Latest | Production HTTP server |

### Frontend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Framework** | React | 18.3.1 | Component-based UI |
| **Build Tool** | Vite | 5.4.0 | Fast HMR dev server + production bundler |
| **Styling** | TailwindCSS | 3.4.10 | Utility-first CSS |
| **Animations** | Framer Motion | 11.3.28 | Page transitions & micro-interactions |
| **3D Graphics** | Three.js + R3F | 0.163 | 3D face map on dashboard |
| **Maps** | Leaflet + React Leaflet | 1.9.4 | Doctor geolocation |
| **Markdown** | react-markdown | 9.0.1 | AI chat response rendering |
| **Router** | React Router | 6.26.2 | SPA navigation |
| **Icons** | react-icons + Lucide | 5.5.0 | 1000+ icons |
| **Type Support** | TypeScript | 5.9.2 | Doctor portal pages |

### ML Training Pipeline

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Model** | ConvNeXt-Base (timm) | 87.6M param vision backbone |
| **Pretrained** | ImageNet-22k → 1k | Transfer learning base |
| **Training** | NVIDIA A6000 (48 GB) | GPU training on Jarvis Labs |
| **Loss Function** | Focal Loss + Label Smoothing | Class imbalance handling |
| **Optimizer** | AdamW (lr=6e-4, wd=0.05) | Decoupled weight decay |
| **Scheduler** | OneCycleLR | Super-convergence |
| **Augmentation** | Mixup + CutMix + Heavy Aug | Regularization |
| **Precision** | bfloat16 AMP | Memory efficiency |
| **EMA** | Decay 0.9998 | Better generalization |

### Databases

| Environment | Database | Purpose |
|-------------|----------|---------|
| **Development** | SQLite | Zero-config local dev |
| **Production** | MySQL 8.0 | Scalable relational DB |
| **Cloud** | Azure SQL (optional) | MSSQL via pyodbc |

---

## 4. Backend — FastAPI Server

### 4.1 Project Structure

```
backend/
├── main.py              # FastAPI app factory, middleware, router registration
├── database.py          # SQLAlchemy engine, session factory, migration runner
├── models.py            # 25+ ORM models (User, Patient, Doctor, Lesion, etc.)
├── schemas.py           # Pydantic input/output schemas with validation
├── auth.py              # Login, registration, JWT issue, password change
├── security.py          # JWT decode, role-based guards, token refresh
├── crud.py              # Password hashing, CRUD helpers
├── inference.py         # PyTorch model inference + Grad-CAM explainability
├── llm_service.py       # Multi-provider LLM abstraction (Gemini/Azure/OpenAI/Ollama)
├── azure_blob.py        # Azure Blob Storage upload helper
├── evaluate.py          # Model evaluation utilities
├── notify.py            # Notification service
├── Dockerfile           # Production container image
├── ml/
│   ├── model.py         # HierarchicalSkinClassifier (dual-head CNN)
│   ├── data.py          # Dataset utilities
│   ├── environment.py   # Environment detection
│   └── weights/         # Pretrained model weights
└── routes/
    ├── admin.py          # Admin dashboard, user management, doctor approvals
    ├── ai_chat.py        # Per-user AI chat sessions with history
    ├── appointments.py   # Appointment CRUD + status management
    ├── chat.py           # Real-time messaging (rooms, messages, typing, video link)
    ├── doctors.py        # Doctor listing, application, availability
    ├── journey.py        # Skin journey photo logging
    ├── lesions.py        # Image upload, ML prediction, diagnosis reports
    ├── llm.py            # LLM chat + streaming endpoints
    ├── notifications.py  # Push notification management
    ├── patients.py       # Patient registration and listing
    ├── profile.py        # User profile (skin type, Fitzpatrick, allergies, goals)
    ├── recommendations.py# UV risk assessment + doctor product suggestions
    ├── routine.py        # Treatment plans, medication steps, adherence tracking
    ├── support.py        # Support tickets + newsletter subscriptions
    └── transactions.py   # Billing/payment log with summaries
```

### 4.2 Middleware Pipeline

The FastAPI application applies security-hardened middleware in the following order:

```
Request → MaxBodySize (15MB) → SecurityHeaders → CORS → Rate Limiter → Route Handler → Response
```

| Middleware | Purpose |
|-----------|---------|
| **SecurityHeadersMiddleware** | Adds `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection`, `Referrer-Policy`, `Permissions-Policy` |
| **MaxBodySizeMiddleware** | Rejects uploads > 15 MB with HTTP 413 |
| **CORSMiddleware** | Allows configurable origins via `FRONTEND_ORIGINS` env |
| **SlowAPI Rate Limiter** | 10 req/min on login, 5 req/min on forgot password |

### 4.3 Key Components

#### Inference Engine (`inference.py`)

The inference engine implements a singleton pattern for model loading:

```python
class MelanomaInference:
    def __init__(self, weights, device, backbone):
        # Hierarchical dual-head model: 5 categories + 19 diseases
        self.model = HierarchicalSkinClassifier(
            num_categories=5, num_diseases=19, backbone=backbone
        )
        # Thread-safe model loading
        state = torch.load(weights, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict_image(self, image):
        # Returns: {label, probability, p_malignant}
        _, dis_logits = self.model(x)
        probs = torch.softmax(dis_logits, dim=1)
        # Cancer probability = sum(melanoma + bcc + scc + ak)
        p_malignant = probs[0:4].sum()
        return {"label": label, "probability": conf, "p_malignant": p_malignant}

    def gradcam_overlay(self, image):
        # Grad-CAM heatmap for visual explainability
        # Returns PIL image with red heatmap overlay
```

#### Input Validation (`routes/lesions.py`)

The lesion upload endpoint performs robust validation before ML inference:

1. **EXIF stripping** — Removes metadata for patient privacy
2. **Resolution check** — Minimum 128px, maximum 8000px
3. **Aspect ratio** — Must be between 0.5:1 and 2:1
4. **Skin detection** — Dual algorithm (YCbCr + HSV color space) rejects non-skin images below 12% skin pixel ratio
5. **Format validation** — JPEG, PNG, WebP, HEIC supported

---

## 5. Frontend — React Application

### 5.1 Application Structure

```
frontend-react/src/
├── App.jsx               # Router + Protected route guards
├── main.jsx              # React DOM entry point
├── index.css             # Tailwind directives + custom theme
├── components/
│   ├── AppShell.jsx      # Main layout (sidebar + header + nav)
│   ├── AuthLayout.jsx    # Login/register layout wrapper
│   ├── Card.jsx          # Reusable card system (glassmorphism)
│   ├── ConfirmModal.jsx  # Action confirmation dialogs
│   ├── DetailsDrawer.jsx # Slide-out detail panel
│   ├── ErrorBoundary.jsx # Crash recovery wrapper
│   ├── FaceMap.jsx       # 2D face zone selector
│   ├── FaceMap3D.jsx     # Three.js 3D face model
│   ├── Navbar.jsx        # Top navigation bar
│   ├── OfflineBanner.jsx # PWA offline indicator
│   ├── ProductSearch.jsx # AI-powered product search
│   ├── Skeleton.jsx      # Loading skeleton components
│   ├── Toast.jsx         # Toast notification system
│   ├── WeatherWidget.jsx # Weather display widget
│   └── dashboard/
│       ├── DoctorSuggestions.jsx
│       ├── PatientQueueList.jsx
│       ├── QuickPrescription.jsx
│       ├── UVIndexWidget.jsx
│       └── WeatherWidget.jsx
├── pages/ (28 pages)
│   ├── Landing.jsx           # Public homepage
│   ├── Login.jsx / Register.jsx / ForgotPassword.jsx
│   ├── Onboarding.jsx        # First-time patient profile setup
│   ├── Dashboard.jsx         # Patient home (3D background, weather, UV)
│   ├── LesionUpload.jsx      # AI skin scan with drag-and-drop
│   ├── Chat.jsx              # AI chatbot with streaming
│   ├── Messages.jsx          # Doctor-patient real-time chat
│   ├── SkinCoach.jsx         # Personalized skincare advisor
│   ├── SkinJourney.jsx       # Photo diary with face zone tagging
│   ├── Routine.jsx           # Treatment plan adherence tracker
│   ├── Appointments.jsx      # Appointment booking
│   ├── FindDermatologists.jsx# Doctor search with map
│   ├── Transactions.jsx      # Billing history
│   ├── Contact.jsx           # Support form
│   ├── DoctorDashboard.jsx   # Doctor home view
│   ├── DoctorAppointments.tsx # Doctor calendar management
│   ├── DoctorAvailability.tsx # Weekly schedule editor
│   ├── DoctorPatients.tsx    # Patient list for doctors
│   ├── DoctorProfile.tsx     # Doctor public profile editor
│   ├── DoctorSettings.tsx    # Doctor preferences
│   ├── DoctorTransactions.tsx# Doctor billing
│   ├── DoctorTreatmentPlans.jsx # Prescribe treatment plans
│   ├── ApplyDoctor.jsx       # Doctor application form
│   ├── AdminDashboard.jsx    # Admin overview & stats
│   ├── AdminUsers.jsx        # User management
│   └── AdminTransactions.jsx # Transaction oversight
├── services/
│   └── api.js            # Centralized API client (70+ methods)
├── hooks/
│   ├── useGeolocation.js   # Browser geolocation hook
│   ├── useNotifications.js # WebSocket notification handler
│   └── useOnlineStatus.js  # Network connectivity monitor
└── utils/
    └── imageUtils.js       # Client-side image compression
```

### 5.2 Route Architecture

The application uses role-based routing with three distinct portals:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Route Map                                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PUBLIC ROUTES                                                           │
│  ├── /login                 Login page                                   │
│  ├── /register              Patient registration                         │
│  ├── /apply-doctor          Doctor application                           │
│  └── /forgot-password       Password reset                               │
│                                                                          │
│  PATIENT PORTAL (role: PATIENT)                                          │
│  ├── /onboarding            First-time skin profile setup                │
│  ├── /dashboard             Home — weather, UV, 3D face, quick actions   │
│  ├── /lesions               AI skin scan — upload & predict              │
│  ├── /chat                  AI chatbot (Gemini/GPT streaming)            │
│  ├── /messages              Doctor-patient messaging                     │
│  ├── /appointments          Book & manage appointments                   │
│  ├── /coach                 Personalized skin advisor + product search   │
│  ├── /journey               Skin photo diary with tags                   │
│  ├── /routine               Treatment plan adherence tracker             │
│  ├── /find-doctors          Map-based doctor discovery                   │
│  ├── /transactions          Billing history                              │
│  └── /contact               Support tickets                              │
│                                                                          │
│  DOCTOR PORTAL (role: DOCTOR)                                            │
│  ├── /doctor                Dashboard — patient queue, quick Rx          │
│  ├── /doctor/appointments   Calendar management                          │
│  ├── /doctor/availability   Weekly schedule editor                       │
│  ├── /doctor/patients       Patient list with lesion history             │
│  ├── /doctor/treatment-plans Prescribe medications & steps               │
│  ├── /doctor/profile        Public profile editor                        │
│  ├── /doctor/transactions   Revenue tracking                             │
│  └── /doctor/settings       Account preferences                          │
│                                                                          │
│  ADMIN PORTAL (role: ADMIN)                                              │
│  ├── /admin                 System overview — stats & charts             │
│  ├── /admin/users           User management, role assignment, bans       │
│  └── /admin/transactions    Global transaction oversight                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Key UI Features

#### Landing Page
- **Hospital-grade design** with gradient hero section
- Framer Motion staggered animations
- Department cards (Dermatology, Cosmetology, Oncology, Pediatrics)
- AI diagnostic preview mockup showing 98.5% confidence card
- HIPAA-compliant floating badge
- Newsletter subscription
- Responsive navigation with sign-in/register CTAs

#### Patient Dashboard
- **3D Face Model** (Three.js) — Ambient background with R3F
- **Weather + UV Widget** — Real-time environmental data for skin protection
- **Smart Greeting** — Time-of-day aware ("Good morning, Arjun")
- **Quick Action Cards** — Scan skin, chat with AI, book appointment
- **Upcoming Appointments** — Next appointment at a glance
- **Skin Journey Preview** — Recent photo logs

#### AI Skin Scan (LesionUpload)
- **Drag-and-drop** image upload with animated drop zone
- **File validation** — 10MB limit, JPEG/PNG/WebP/HEIC only
- **High Sensitivity Mode** — Toggle for lower malignancy threshold (0.35 vs 0.5)
- **AI Classification Result** — Disease label + confidence percentage
- **Risk Score Badge** — Green (low) / Amber (medium) / Red (high risk)
- **Grad-CAM Heatmap** — Visual attention map overlay
- **One-Click Diagnosis Report** — LLM-generated detailed medical summary

#### AI Chat (Skin Coach)
- **Streaming responses** — Real-time LLM output via SSE
- **Session management** — Create, rename, delete chat sessions
- **Markdown rendering** — Formatted medical advice
- **Multi-provider** — Seamlessly switches between Gemini/GPT/Ollama
- **Persistent history** — Stored in localStorage + server-side per session

#### Skin Journey
- **Photo timeline** — Gallery and list view modes
- **Face zone tagging** — Interactive SVG face map for location marking
- **Client-side compression** — Images compressed before base64 encoding
- **Tag system** — Custom tags + anatomical location tags
- **Delete with confirmation** — Modal-based delete protection

#### Treatment Plans (Routine)
- **Doctor-prescribed plans** — Diagnosis, medications, dosage, frequency
- **Daily adherence tracker** — Bubble-based AM/PM step checkboxes
- **Progress ring** — Visual adherence percentage
- **Side effect reporting** — Severity rating (mild/moderate/severe) with text
- **Auto-loads active plan** — First active plan selected on mount

---

## 6. Machine Learning Pipeline

### 6.1 Jarvis Training Pipeline

The ML pipeline is housed in `jarvis-training/` — a self-contained 8-file codebase (3,070 LOC):

```
jarvis-training/
├── setup.sh           # One-command environment setup
├── download_data.py   # Kaggle API automated dataset download
├── prepare_data.py    # Multi-dataset unification + stratified splitting
├── train.py           # Production trainer (1,171 LOC)
├── evaluate.py        # Evaluation + confusion matrix + cancer sensitivity
├── requirements.txt   # Pinned dependencies
├── README.md          # Usage instructions
└── .gitignore         # Ignore data/, checkpoints/, runs/
```

### 6.2 Dataset Preparation

Five major dermatological datasets were unified:

| Dataset | Source | Raw Images | Contribution |
|---------|--------|-----------|--------------|
| **ISIC 2019** | International Skin Imaging Collaboration | 25,331 | Dermoscopic images |
| **HAM10000** | ViDIR Group, Medical University of Vienna | 10,015 | 7 lesion types |
| **DermNet** | DermNet NZ + Kaggle | ~23,000 | Clinical photos, 23 categories |
| **PAD-UFES-20** | Federal University of Espírito Santo | 2,298 | Smartphone images (Brazil) |
| **Massive Balanced** | Kaggle aggregation | ~198,000 | Large-scale balanced dataset |
| **Total** | | **259,154** | **20 unified classes** |

#### Unified Label Mapping

The `DISEASE_MAP` dictionary maps 100+ raw labels from all 5 datasets into 20 canonical classes:

```python
DISEASE_MAP = {
    # ISIC 2019 codes
    "MEL": "melanoma", "NV": "nevus", "BCC": "bcc", "AK": "ak",
    "BKL": "seborrheic_keratosis", "DF": "dermatofibroma", "VASC": "angioma", "SCC": "scc",

    # HAM10000 codes
    "mel": "melanoma", "nv": "nevus", "bcc": "bcc", "akiec": "ak",

    # DermNet folder names
    "Melanoma Skin Cancer Nevi and Moles": "melanoma",
    "Acne and Rosacea Photos": "acne",
    "Eczema Photos": "eczema",
    # ... 100+ total mappings
}
```

#### 20-Class Taxonomy

| # | Class | Category | Cancer? |
|---|-------|----------|---------|
| 1 | melanoma | Cancer | Yes |
| 2 | bcc (basal cell carcinoma) | Cancer | Yes |
| 3 | scc (squamous cell carcinoma) | Cancer | Yes |
| 4 | ak (actinic keratosis) | Cancer | Yes |
| 5 | nevus | Benign | No |
| 6 | seborrheic_keratosis | Benign | No |
| 7 | angioma | Benign | No |
| 8 | dermatofibroma | Benign | No |
| 9 | eczema | Inflammatory | No |
| 10 | psoriasis | Inflammatory | No |
| 11 | acne | Inflammatory | No |
| 12 | dermatitis | Inflammatory | No |
| 13 | urticaria | Inflammatory | No |
| 14 | bullous | Inflammatory | No |
| 15 | impetigo | Infectious | No |
| 16 | herpes | Infectious | No |
| 17 | fungal | Infectious | No |
| 18 | scabies | Infectious | No |
| 19 | wart | Infectious | No |
| 20 | hyperpigmentation | Pigmentary | No |

#### Data Split

| Split | Images | Percentage |
|-------|--------|-----------|
| Train | 207,345 | 80% |
| Validation | 25,905 | 10% |
| Test | 25,904 | 10% |

### 6.3 Model Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ConvNeXt-Base (timm)                               │
│                                                                       │
│  Input: 320×320×3 RGB Image                                         │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Stem: 4×4 Conv, stride 4 → 128 channels                       │ │
│  │  Stage 1: 3× ConvNeXt Block → 128 channels, 80×80              │ │
│  │  Downsample → 256 channels, 40×40                               │ │
│  │  Stage 2: 3× ConvNeXt Block → 256 channels                     │ │
│  │  Downsample → 512 channels, 20×20                               │ │
│  │  Stage 3: 27× ConvNeXt Block → 512 channels                    │ │
│  │  Downsample → 1024 channels, 10×10                              │ │
│  │  Stage 4: 3× ConvNeXt Block → 1024 channels                    │ │
│  │  Global Average Pooling → 1024-dim vector                       │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                        │
│                    ┌─────────┴──────────┐                            │
│                    │   Dropout (0.4)     │                            │
│                    │   Linear(1024, 20)  │                            │
│                    │   → 20 class logits │                            │
│                    └────────────────────┘                             │
│                                                                       │
│  Parameters: 87.6M  |  Pretrained: ImageNet-22k → ImageNet-1k       │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.4 Training Configuration

```yaml
# Model
backbone: convnext_base.fb_in22k_ft_in1k_384
img_size: 320
drop_rate: 0.4
drop_path_rate: 0.3

# Optimization
optimizer: AdamW
lr: 6e-4
min_lr: 1e-6
weight_decay: 0.05
scheduler: OneCycleLR (warmup 10% → peak → cosine decay)
grad_clip: 1.0
batch_size: 96
epochs: 30

# Loss
criterion: FocalLoss (gamma=2.0, label_smoothing=0.1)
cancer_weight_boost: 2.0x for melanoma, bcc, scc, ak
class_weights: Inverse-frequency, normalized

# Augmentation
train_augmentation:
  - RandomResizedCrop(320, scale=0.7-1.0)
  - RandomHorizontalFlip(0.5)
  - RandomVerticalFlip(0.5)
  - RandomRotation(30°)
  - ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08)
  - RandomAffine(translate=8%, shear=8°)
  - RandomErasing(p=0.25, scale=0.02-0.15)
  - Mixup(alpha=0.2) / CutMix(alpha=1.0) at 50% probability

val_augmentation:
  - Resize(366, bicubic)  # 320 * 1.143
  - CenterCrop(320)
  - Normalize(ImageNet stats)

# Regularization
ema: enabled (decay=0.9998)
weighted_sampling: enabled (cancer 2x boost)
precision: bfloat16 AMP (native A6000 support)

# Checkpointing
save: best_model.pth (best val acc), checkpoint_latest.pth (every epoch)
periodic: checkpoint_epoch_N.pth (every 5 epochs), keep last 3
signal_safe: SIGINT/SIGTERM → graceful save + exit
```

### 6.5 Training Techniques Explained

#### Focal Loss
Standard cross-entropy treats all samples equally. Focal Loss down-weights well-classified examples (easy negatives) and focuses learning on hard, misclassified samples — critical for rare cancer classes:

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

With $\gamma = 2.0$, an easy example with $p_t = 0.9$ has its loss reduced by 100×, while a hard example with $p_t = 0.1$ retains full loss magnitude.

#### Exponential Moving Average (EMA)
Maintains a shadow copy of model weights that smoothly tracks the training trajectory:

$$\theta_{EMA} = \alpha \cdot \theta_{EMA} + (1 - \alpha) \cdot \theta_{model}$$

With $\alpha = 0.9998$, EMA averages over ~5000 recent steps, producing more stable weights for evaluation.

#### Mixup / CutMix
At 50% probability per batch, either Mixup (pixel-level blending) or CutMix (patch replacement) is applied:

- **Mixup**: $\tilde{x} = \lambda x_i + (1-\lambda) x_j$, $\tilde{y} = \lambda y_i + (1-\lambda) y_j$
- **CutMix**: Replace a random rectangular patch of $x_i$ with the same region from $x_j$

This prevents overconfident predictions and improves calibration.

#### Weighted Random Sampling
Cancer classes receive 2× sampling weight, ensuring the model sees proportionally more cancer images per epoch:

$$w_i = \frac{1}{|C_i|} \times \begin{cases} 2.0 & \text{if } C_i \in \{\text{melanoma, bcc, scc, ak}\} \\ 1.0 & \text{otherwise} \end{cases}$$

---

## 7. Database Design

### 7.1 Entity-Relationship Diagram

```
┌─────────────────┐       ┌──────────────────┐
│     User         │       │   UserProfile     │
│  ─────────────   │       │  ──────────────   │
│  user_id (PK)    │──1:1──│  skin_type        │
│  username        │       │  fitzpatrick_type │
│  email           │       │  allergies        │
│  hashed_password │       │  goals            │
│  role            │       │  location_city    │
└────────┬────────┘       └──────────────────┘
         │
    ┌────┴────┐
    │         │
 ┌──▼──┐  ┌──▼───┐
 │Patient│  │Doctor │
 │──────│  │──────│
 │name  │  │spec. │
 │age   │  │bio   │
 │gender│  │avail.│
 └──┬───┘  └──┬───┘
    │         │
    │    ┌────┴────────┐
    │    │              │
 ┌──▼───▼──┐  ┌───────▼────────┐
 │Appointment│  │ ChatRoom        │
 │──────────│  │ ────────        │
 │date      │  │ messages[]      │
 │reason    │  │ video_link      │
 │status    │  │ unread_counts   │
 └──────────┘  └────────────────┘

 ┌──────────────┐      ┌───────────────────┐
 │   Lesion      │───>  │ DiagnosisReport    │
 │  ──────────   │      │ ──────────────     │
 │  image_path   │      │ prediction         │
 │  prediction   │      │ summary            │
 │  created_at   │      │ details (LLM)      │
 └───────┬──────┘      └────────┬──────────┘
         │                       │
 ┌───────▼──────┐       ┌───────▼────────────┐
 │ LesionReview  │       │ DoctorSuggestion    │
 │ ────────────  │       │ ──────────────────  │
 │ decision      │       │ product_name        │
 │ override_label│       │ product_link        │
 │ comment       │       │ notes               │
 └──────────────┘       └────────────────────┘

 ┌──────────────────┐    ┌───────────────────┐
 │  TreatmentPlan    │───>│ TreatmentStep      │
 │  ──────────────   │    │ ──────────────     │
 │  diagnosis        │    │ medication_name    │
 │  status           │    │ dosage, frequency  │
 │  notes            │    │ time_of_day        │
 └──────────────────┘    └────────┬──────────┘
                                   │
                          ┌────────▼──────────┐
                          │ TreatmentAdherence │
                          │ ──────────────────│
                          │ date, taken, notes│
                          │ side_effects      │
                          └───────────────────┘
```

### 7.2 Core Tables

| Table | Columns | Purpose |
|-------|---------|---------|
| `users` | user_id, username, email, hashed_password, role | Central user identity |
| `patients` | patient_id, user_id, first_name, last_name, age, gender | Patient demographics |
| `doctors` | doctor_id, user_id, specialization | Doctor credentials |
| `doctor_profiles` | doctor_id, bio, visibility | Public doctor profiles |
| `doctor_applications` | application_id, user_id, license_no, hospital, status | Application workflow (PENDING → APPROVED/REJECTED) |
| `doctor_availability` | availability_id, doctor_id, weekday, start_time, end_time | Weekly schedule |
| `appointments` | appointment_id, patient_id, doctor_id, date, status | Booking system |
| `lesions` | lesion_id, patient_id, image_path, prediction, created_at | AI scan results |
| `lesion_reviews` | review_id, lesion_id, doctor_id, decision, override_label | Doctor confirmation/override |
| `diagnosis_reports` | report_id, lesion_id, patient_id, prediction, details | LLM diagnostic reports |
| `chat_rooms` | room_id, patient_id, doctor_id, video_link, unread_counts | Doctor-patient messaging |
| `messages` | message_id, room_id, sender_user_id, content, type, status | Chat messages |
| `message_reactions` | reaction_id, message_id, user_id, emoji | Emoji reactions |
| `ai_chat_sessions` | session_id, user_id, title | Per-user AI chat history |
| `ai_chat_messages` | message_id, session_id, role, content | AI chat turns |
| `transactions` | transaction_id, user_id, amount, status, category | Billing records |
| `transaction_meta` | id, transaction_id, method, reference, note | Payment metadata |
| `user_profiles` | profile_id, user_id, skin_type, fitzpatrick_type, allergies, goals | Personalization |
| `skin_logs` | log_id, user_id, image_path, notes, tags | Skin journey entries |
| `routine_items` | item_id, user_id, product_name, time_of_day, step_order | Skincare routine |
| `routine_completions` | completion_id, routine_item_id, date, status | Routine adherence |
| `products` | product_id, name, brand, ingredients_text, embedding | Product catalog |
| `support_tickets` | ticket_id, user_id, subject, message | Customer support |
| `newsletter_subscribers` | id, email | Marketing list |
| `audit_logs` | id, user_id, action, meta | Admin audit trail |
| `settings` | key, value | System configuration KV store |
| `user_token_versions` | user_id, version | JWT invalidation (logout all) |
| `user_status` | user_id, status (ACTIVE/SUSPENDED/TERMINATED) | Account lifecycle |

---

## 8. Authentication & Security

### 8.1 Authentication Flow

```
┌──────────┐    POST /auth/login     ┌──────────┐     Verify Password      ┌─────────┐
│  Client   │ ────────────────────> │  Backend  │ ──────────────────────> │ Database │
│           │ (email + password)     │           │    bcrypt.verify()      │          │
│           │                        │           │ <────────────────────── │          │
│           │                        │           │     Check UserStatus    │          │
│           │                        │           │ ──────────────────────> │          │
│           │                        │           │ <────────────────────── │          │
│           │  JWT + user payload    │           │     Issue JWT           │          │
│           │ <────────────────────  │           │                        │          │
└──────────┘                        └──────────┘                        └─────────┘
```

### 8.2 JWT Structure

```json
{
  "sub": "42",              // user_id as string
  "role": "PATIENT",        // PATIENT | DOCTOR | ADMIN
  "tv": 1,                  // Token version (incremented on "logout all")
  "patient_id": 15,         // Included for PATIENT role
  "doctor_id": 7,           // Included for DOCTOR role
  "iat": 1709141234,        // Issued at timestamp
  "exp": 1709148434         // Expires in 120 minutes (configurable)
}
```

### 8.3 Security Features

| Feature | Implementation |
|---------|---------------|
| **Password Hashing** | bcrypt with salt rounds |
| **Password Policy** | Min 8 chars, 1 uppercase, 1 lowercase, 1 digit, 1 special char |
| **JWT Signing** | HS256 with `JWT_SECRET` env var |
| **Token Refresh** | Auto-refresh when < 10 min to expiry on `/auth/me` |
| **Token Invalidation** | Token version (`tv`) — increment to invalidate all sessions |
| **Rate Limiting** | 10 login attempts/min, 5 forgot password requests/min |
| **CORS** | Configurable `FRONTEND_ORIGINS` whitelist |
| **Security Headers** | CSP, X-Frame-Options: DENY, HSTS-ready |
| **Upload Protection** | 15MB limit, EXIF stripping, skin pixel validation |
| **RBAC** | `require_roles("ADMIN")` decorator guard on admin endpoints |
| **Clock Drift Tolerance** | 300-second JWT leeway for distributed systems |
| **Session Sync** | Cross-tab `storage` event listener for consistent logout |

---

## 9. AI/LLM Integration

### 9.1 Multi-Provider LLM Service

The `llm_service.py` abstraction supports 4 providers with automatic fallback:

```
Priority: Gemini → Azure OpenAI → Ollama → OpenAI (fallback)
```

| Provider | Model | Use Case |
|----------|-------|----------|
| **Google Gemini** | gemini-1.5-flash | Primary — fast, free tier available |
| **Azure OpenAI** | Configurable deployment | Enterprise — HIPAA compliant |
| **Ollama** | Local models (Llama3, etc.) | Offline / air-gapped deployment |
| **OpenAI** | gpt-4o-mini | Fallback |

### 9.2 LLM Features

1. **Diagnostic Report Generation** — Given a lesion prediction + patient profile, generate a detailed medical summary
2. **AI Chat Bot** — Streaming responses for real-time conversational feel
3. **Clinical Notes Generation** — Summarize doctor-patient chat history into structured notes (Doctor Copilot)
4. **Temperature Control** — Configurable via `LLM_TEMPERATURE` (default 0.1 for medical accuracy)

### 9.3 Streaming Architecture

```
Client                      Backend                      LLM Provider
  │                            │                              │
  │── POST /llm/chat_stream ──>│                              │
  │                            │── generate_content(stream) ──>│
  │                            │                              │
  │<── SSE: chunk 1 ──────────│<── chunk 1 ──────────────────│
  │<── SSE: chunk 2 ──────────│<── chunk 2 ──────────────────│
  │<── SSE: chunk 3 ──────────│<── chunk 3 ──────────────────│
  │<── SSE: [DONE] ───────────│<── stream end ───────────────│
```

---

## 10. Training Results

### 10.1 Training Curve

Training was conducted on an NVIDIA RTX A6000 (48 GB VRAM) on Jarvis Labs cloud platform.

```
Val Accuracy (%) vs Epoch
100 ┤
 98 ┤                                          ●──●  98.22%
 96 ┤                              ●──●──●──●─╯
 94 ┤                    ●──●──●─╯
 92 ┤               ●──╯
 90 ┤          ●
 88 ┤        ●
 84 ┤      ●
 80 ┤
 68 ┤    ●
 36 ┤  ●
    ├──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──
    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25→30
```

### 10.2 Epoch-by-Epoch Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time | Notes |
|-------|-----------|-----------|----------|---------|------|-------|
| 1 | — | — | — | 35.63% | 1291s | Initial learning |
| 2 | — | — | — | 67.90% | 1267s | Rapid improvement |
| 3 | — | — | — | 83.49% | 1263s | Features learned |
| 4 | — | — | — | 88.68% | 1265s | Strong baseline |
| 5 | — | — | — | 90.84% | 1262s | No low-sensitivity warnings |
| 6 | — | — | — | 92.24% | — | |
| 7 | — | — | — | 93.12% | — | |
| 8 | — | — | — | 93.72% | — | *Interrupted (SIGINT, saved checkpoint)* |
| 9 | — | — | — | — | — | *Resumed from checkpoint* |
| 10 | — | — | — | 94.24% | — | New best after resume |
| 11 | — | — | — | 94.65% | — | |
| 12 | — | — | — | 95.00% | — | |
| 14 | 0.4156 | 96.4% | 0.1438 | 95.72% | 1318s | |
| 15 | 0.3974 | 96.9% | 0.1407 | 96.03% | 1319s | |
| 16 | 0.3850 | 97.2% | 0.1379 | 96.34% | 1319s | |
| 17 | 0.3829 | 97.7% | 0.1354 | 96.73% | 1319s | |
| 18 | 0.3738 | 97.9% | 0.1334 | 96.96% | 1317s | |
| 19 | 0.3617 | 98.1% | 0.1312 | 97.27% | 1316s | |
| 20 | 0.3474 | 98.4% | 0.1295 | 97.48% | 1315s | |
| 21 | 0.3369 | 98.7% | 0.1282 | 97.75% | 1318s | |
| 22 | 0.3340 | 98.8% | 0.1273 | 97.92% | 1318s | |
| 23 | 0.3419 | 99.0% | 0.1267 | 98.10% | 1317s | |
| **24** | **0.3232** | **99.2%** | **0.1264** | **98.22%** | **1317s** | **Best so far** |
| 25–30 | — | — | — | ~98.5%* | — | *In progress / projected* |

### 10.3 Key Performance Indicators

| Metric | Value | Target (SRS) | Status |
|--------|-------|-------------|--------|
| **Overall Val Accuracy** | 98.22% | ≥95% | Exceeded |
| **Train/Val Gap** | 0.98% | <5% | Excellent (no overfit) |
| **Val Loss** | 0.1264 (monotonically decreasing) | N/A | Healthy |
| **Epoch Time** | ~22 min | N/A | Consistent |
| **GPU Utilization** | 100% | N/A | Optimal |
| **GPU Temperature** | 78°C | <83°C | Safe |
| **VRAM Usage** | 29.9 / 49.1 GB | <48 GB | Within budget |

### 10.4 Hardware Utilization

| Resource | Value |
|----------|-------|
| **GPU** | NVIDIA RTX A6000 (48 GB VRAM) |
| **VRAM Used** | 29.9 GB / 49.1 GB (61%) |
| **GPU Utilization** | 100% |
| **Power** | 292W / 300W TDP |
| **Temperature** | 78°C (no throttling) |
| **Training Speed** | 1.71 it/s (2159 batches/epoch) |
| **Total Training Time** | ~11 hours (30 epochs) |

---

## 11. Deployment Architecture

### 11.1 Docker Container

```dockerfile
FROM python:3.11-slim
WORKDIR /app
# Security: non-root user
RUN adduser --disabled-password --gecos '' appuser
# Health check built-in
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 11.2 Environment Configuration

The application is fully configurable via environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATABASE_URL` | Primary DB connection | `sqlite:///./dev.db` |
| `MYSQL_HOST/DB/USER/PASSWORD` | MySQL auto-config | — |
| `JWT_SECRET` | Token signing key | dev-secret (warning) |
| `FRONTEND_ORIGINS` | CORS whitelist | localhost:3000,5173 |
| `GEMINI_API_KEY` | Google Gemini LLM | — |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI | — |
| `OLLAMA_BASE_URL` + `OLLAMA_MODEL` | Local LLM | — |
| `LESION_MODEL_WEIGHTS` | ML model path | — |
| `LESION_MODEL_BACKBONE` | Model architecture | resnet18 |
| `LESION_MALIGNANT_THRESHOLD` | Cancer alert threshold | 0.5 |
| `MAX_UPLOAD_BYTES` | Upload size limit | 15 MB |
| `ADMIN_EMAIL/USERNAME/PASSWORD` | Bootstrap admin | — |

### 11.3 Production Deployment Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ GitHub Repo   │────>│ Docker Build  │────>│  Production  │
│ git push      │     │ Dockerfile    │     │  Server      │
└──────────────┘     └──────────────┘     │              │
                                           │  uvicorn     │
┌──────────────┐                          │  :8000       │
│ Vite Build    │─── npm run build ───>   │              │
│ dist/         │─── Static Files ────>   │  nginx       │
└──────────────┘                          │  :80/:443    │
                                           └──────────────┘
```

---

## 12. Screenshots & UI Showcase

> **Note:** Screenshots should be captured from the running application and placed in `docs/screenshots/`. The following section describes each screen in detail for reference.

### 12.1 Landing Page

```
┌─────────────────────────────────────────────────────────────┐
│  🧪 AI Skin Doctor          Departments │ Why Us │ Sign in  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ● Next-Gen Dermatology                                      │
│                                                               │
│  World-class care with                    ┌────────────────┐ │
│  AI precision                             │ AI Diagnostic   │ │
│                                            │ Preview         │ │
│  Experience the future of skin health.     │                 │ │
│  Instant AI analysis, expert               │ Analysis        │ │
│  consultations, and personalized           │ ████████ 98.5% │ │
│  care journeys.                            │                 │ │
│                                            │ 🟢 Low Risk    │ │
│  [Start Diagnosis] [Explore Departments]   │ Routine         │ │
│                                            │ monitoring      │ │
│  50+ Experts │ 25k+ Patients │ 12+ Clinics│ advised         │ │
│                                            └────────────────┘ │
│                                            🛡️ HIPAA Compliant │
├─────────────────────────────────────────────────────────────┤
│            Specialized Departments                            │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐              │
│  │Dermato-│ │Cosmeto-│ │Onco-   │ │Pedia-  │              │
│  │logy    │ │logy    │ │logy    │ │trics   │              │
│  └────────┘ └────────┘ └────────┘ └────────┘              │
├─────────────────────────────────────────────────────────────┤
│  🤖 AI-Assisted    👨‍⚕️ Top           🔒 Secure &          │
│     Triage           Specialists       Private              │
├─────────────────────────────────────────────────────────────┤
│  Ready to transform your skin health?    [Book Appointment]  │
└─────────────────────────────────────────────────────────────┘
```

### 12.2 Patient Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│  [Sidebar]    Good morning, Arjun 👋                         │
│  🏠 Home      Saturday, March 1                              │
│  🔬 Scan      ─────────────────────────────────────────────  │
│  💬 AI Chat                                                   │
│  📧 Messages  ┌──────────────────┐ ┌──────────────────────┐ │
│  📅 Appts.    │ 🌤️ Weather       │ │ ☀️ UV Index          │ │
│  💊 Routine   │ 28°C Partly Cloudy│ │ 7 (High)            │ │
│  📸 Journey   │ Humidity: 65%     │ │ SPF 30+ recommended │ │
│  🩺 Doctors   └──────────────────┘ └──────────────────────┘ │
│  💰 Billing                                                   │
│  📞 Contact   ┌──────────────────────────────────────────┐  │
│               │ Quick Actions                              │  │
│  [Logout]     │ [🔬 Scan Skin] [💬 Ask AI] [📅 Book]    │  │
│               └──────────────────────────────────────────┘  │
│                                                               │
│               ┌──────────────────────────────────────────┐  │
│               │ 🩺 Upcoming Appointments                  │  │
│               │ Dr. Sharma — Mar 3, 10:00 AM              │  │
│               └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 12.3 AI Skin Scan

```
┌─────────────────────────────────────────────────────────────┐
│  AI Skin Analysis                                             │
│  Upload a photo for instant AI-powered skin assessment        │
│                                                               │
│  ┌────────────────────────────┐  ┌────────────────────────┐ │
│  │                              │  │ ⚙️ Settings           │ │
│  │    ┌──────────────────┐    │  │                        │ │
│  │    │  📤 Drop image   │    │  │ 🔴 High Sensitivity   │ │
│  │    │     here or      │    │  │    Mode: ON           │ │
│  │    │  click to browse │    │  │                        │ │
│  │    │                  │    │  │ Threshold: 0.35       │ │
│  │    │  JPG, PNG, WebP  │    │  │ (lower = more         │ │
│  │    │  Max 10MB        │    │  │  sensitive to cancer)  │ │
│  │    └──────────────────┘    │  │                        │ │
│  │                              │  └────────────────────────┘ │
│  │  [ Analyze Image ]          │                              │
│  └────────────────────────────┘                              │
│                                                               │
│  ═══════════ RESULT ═══════════                              │
│                                                               │
│  Prediction: Nevus (Benign Mole)                             │
│  Confidence: 94.7%                                            │
│  Risk: 🟢 LOW (p_malignant: 0.03)                           │
│                                                               │
│  ┌───────────────┐  ┌───────────────────────────────────┐   │
│  │ 🔥 Grad-CAM   │  │ 📋 Diagnosis Report               │   │
│  │  [heatmap]     │  │ This lesion appears to be a       │   │
│  │               │  │ melanocytic nevus (common mole).   │   │
│  │               │  │ The AI model identifies symmetric  │   │
│  │               │  │ borders and uniform pigmentation.  │   │
│  │               │  │                                    │   │
│  │               │  │ Recommendation: Routine            │   │
│  └───────────────┘  │ monitoring. Seek evaluation if     │   │
│                      │ changes occur.                     │   │
│                      └───────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 12.4 AI Chat Interface

```
┌─────────────────────────────────────────────────────────────┐
│  [💬 Sessions]        AI Dermatology Assistant                │
│  ──────────────                                               │
│  + New Chat           ┌──────────────────────────────────┐  │
│                        │ 🧑 You                           │  │
│  Acne concern          │ What are the best treatments for │  │
│  Eczema question       │ hormonal acne on my jawline?     │  │
│  Mole check            │                                   │  │
│                        │ 🤖 AI                             │  │
│                        │ For hormonal acne along the       │  │
│                        │ jawline, I'd recommend a multi-   │  │
│                        │ pronged approach:                  │  │
│                        │                                   │  │
│                        │ **1. Topical Retinoids**          │  │
│                        │ - Tretinoin (0.025-0.05%)         │  │
│                        │ - Apply at night, use sunscreen   │  │
│                        │                                   │  │
│                        │ **2. Benzoyl Peroxide (2.5-5%)**  │  │
│                        │ - Antibacterial, morning use      │  │
│                        │                                   │  │
│                        │ **3. Oral Options**               │  │
│                        │ - Spironolactone (consult doctor) │  │
│                        │ ▊ (streaming...)                  │  │
│                        └──────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────┐ [Send ➤]         │
│  │ Ask about your skin concern...         │                   │
│  └───────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### 12.5 Doctor Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│  [Sidebar]        Doctor Dashboard                            │
│  🏠 Dashboard                                                 │
│  📅 Appointments  Welcome back, Dr. Sharma                   │
│  👥 Patients      ─────────────────────────────────          │
│  🗓️ Availability                                             │
│  💊 Prescriptions ┌──────────────────┐ ┌──────────────────┐ │
│  👤 Profile       │ 📊 Today's Queue │ │ 💰 This Month    │ │
│  💰 Billing       │ 8 appointments   │ │ ₹45,200 revenue  │ │
│  ⚙️ Settings      │ 3 pending review │ │ 42 consultations │ │
│                    └──────────────────┘ └──────────────────┘ │
│  [Logout]                                                     │
│               ┌──────────────────────────────────────────┐  │
│               │ 👥 Patient Queue                          │  │
│               │ 1. Rahul M. — Mole check — 10:00 AM     │  │
│               │ 2. Priya K. — Eczema follow-up — 10:30   │  │
│               │ 3. Amit J. — New lesion — 11:00 AM       │  │
│               └──────────────────────────────────────────┘  │
│                                                               │
│               ┌──────────────────────────────────────────┐  │
│               │ 📝 Quick Prescription                     │  │
│               │ Patient: [Select] Diagnosis: [Enter]      │  │
│               │ [Create Treatment Plan]                    │  │
│               └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 12.6 Admin Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│  Admin Control Panel                                          │
│  ─────────────────────────────────────────────────────       │
│                                                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ 👥 Users │ │ 👨‍⚕️ Doctors│ │ 📋 Apps  │ │ 💰 Rev   │      │
│  │  1,247   │ │    32     │ │    5      │ │ ₹2.1L    │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│                                                               │
│  Doctor Applications                                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Dr. Priya Nair │ Dermatology │ PENDING │ [✅] [❌]   │  │
│  │ Dr. Raj Patel  │ Oncology    │ PENDING │ [✅] [❌]   │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  User Management                  Audit Log                   │
│  ┌────────────────────────┐     ┌─────────────────────────┐ │
│  │ [Search users...]       │     │ Admin approved doctor #5│ │
│  │ User1 │ PATIENT │ Active│     │ User login: user@email  │ │
│  │ User2 │ DOCTOR  │ Active│     │ Settings updated        │ │
│  └────────────────────────┘     └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 13. API Reference

### 13.1 Authentication

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/auth/login` | Login with email + password | No |
| GET | `/auth/me` | Get current user + auto-refresh token | Yes |
| POST | `/auth/forgot` | Request password reset | No |
| POST | `/auth/change_password` | Change password (old + new) | Yes |
| POST | `/auth/logout_all` | Invalidate all tokens | Yes |

### 13.2 Patients

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/patients/register` | Register new patient | No |
| GET | `/patients` | List patients (admin/doctor) | Yes |

### 13.3 Lesions (AI Scan)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/lesions/predict` | Upload image + get AI prediction | Yes |
| GET | `/lesions/status` | Model status and configuration | No |
| POST | `/lesions/{id}/report` | Generate LLM diagnosis report | Yes |
| GET | `/lesions/reports` | List diagnosis reports | Yes |
| POST | `/lesions/reports/{id}/send` | Send report to doctor | Yes |

### 13.4 Doctors

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/doctors` | List all visible doctors | Yes |
| POST | `/doctors/apply` | Submit doctor application | No |
| GET | `/doctors/{id}/availability` | Get weekly schedule | Yes |
| POST | `/doctors/{id}/availability` | Set weekly schedule | Yes (Doctor) |

### 13.5 Appointments

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/appointments/` | Book appointment | Yes |
| GET | `/appointments/` | List user's appointments | Yes |
| PATCH | `/appointments/{id}/status` | Update status | Yes |

### 13.6 Chat (Doctor-Patient Messaging)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/chat/rooms` | List rooms | Yes |
| POST | `/chat/rooms` | Create room | Yes |
| GET | `/chat/rooms/{id}/messages` | List messages (paginated) | Yes |
| POST | `/chat/rooms/{id}/messages` | Send message | Yes |
| PUT | `/chat/messages/{id}` | Edit/delete message | Yes |
| PUT | `/chat/messages/{id}/urgent` | Mark urgent | Yes |
| PUT | `/chat/rooms/{id}/video-link` | Set video call link | Yes |
| POST | `/chat/rooms/{id}/read` | Mark room as read | Yes |
| GET | `/chat/unread` | Get unread counts | Yes |
| POST | `/chat/online` | Heartbeat (online status) | Yes |
| POST | `/chat/rooms/{id}/typing` | Typing indicator | Yes |

### 13.7 AI Chat

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/llm/chat` | Synchronous LLM chat | Yes |
| POST | `/llm/chat_stream` | Streaming SSE chat | Yes |
| GET | `/llm/status` | Provider status | No |
| POST | `/llm/generate_notes` | Doctor copilot notes | Yes |
| GET | `/ai_chat/sessions` | List sessions | Yes |
| POST | `/ai_chat/sessions` | Create session | Yes |
| DELETE | `/ai_chat/sessions/{id}` | Delete session | Yes |
| GET | `/ai_chat/sessions/{id}/messages` | List session messages | Yes |
| POST | `/ai_chat/sessions/{id}/messages` | Add message | Yes |

### 13.8 Personalization

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/profile/me` | Get user profile | Yes |
| PUT | `/profile/me` | Update profile | Yes |
| GET | `/journey/` | Get skin journal | Yes |
| POST | `/journey/` | Add journal entry | Yes |
| DELETE | `/journey/{id}` | Delete entry | Yes |
| GET | `/routine/` | Get treatment plans | Yes |
| POST | `/routine/plans` | Create plan (Doctor) | Yes |
| POST | `/routine/plans/{id}/steps` | Add medication step | Yes |
| GET | `/routine/plans/{id}/adherence` | Get adherence data | Yes |
| POST | `/routine/plans/{id}/adherence` | Record adherence | Yes |
| POST | `/routine/plans/{id}/report-side-effect` | Report side effect | Yes |
| GET | `/recommendations/uv-risk` | Fitzpatrick-aware UV risk | Yes |
| POST | `/recommendations/suggest` | Doctor product suggestion | Yes |
| GET | `/recommendations/suggestions/{id}` | Patient's suggestions | Yes |

### 13.9 Admin

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/admin/overview` | System statistics | Admin |
| GET | `/admin/doctor_applications` | List applications | Admin |
| POST | `/admin/doctor_applications/{id}/approve` | Approve doctor | Admin |
| POST | `/admin/doctor_applications/{id}/reject` | Reject doctor | Admin |
| GET | `/admin/users` | List/search users | Admin |
| PATCH | `/admin/users/{id}/role` | Change user role | Admin |
| PATCH | `/admin/users/{id}/status` | Suspend/activate user | Admin |
| POST | `/admin/users/{id}/terminate` | Terminate account | Admin |
| GET | `/admin/audit_logs` | View audit trail | Admin |
| GET/POST | `/admin/settings` | System settings KV | Admin |

---

## 14. Testing & Validation

### 14.1 Model Validation Strategy

| Test Type | Method | Status |
|-----------|--------|--------|
| **Holdout Test Set** | 25,904 images (10% stratified split) | Pending (after epoch 30) |
| **Per-Class Recall** | Classification report from sklearn | Pending |
| **Cancer Sensitivity** | Melanoma/BCC/SCC/AK recall ≥ 94% target | Pending |
| **Confusion Matrix** | Heatmap visualization | Pending |
| **Test-Time Augmentation** | 5x augmented inference averaging | Supported (`--tta` flag) |
| **Balanced Accuracy** | Accounts for class imbalance | Pending |

### 14.2 Backend Validation

| Area | Validation | Implementation |
|------|-----------|---------------|
| **Input** | Password strength | Regex: 8+ chars, upper, lower, digit, special |
| **Input** | Email format | Pydantic `EmailStr` |
| **Input** | Image format | MIME type + extension + magic bytes |
| **Input** | Image content | Skin pixel ratio ≥ 12% (YCbCr + HSV) |
| **Auth** | Token expiry | JWT exp + 300s leeway |
| **Auth** | Role guards | `require_roles()` decorator |
| **Output** | EXIF stripping | PIL metadata removal |
| **Output** | Response schemas | Pydantic `response_model` validation |
| **Rate** | Brute force | SlowAPI: 10 login/min, 5 forgot/min |

### 14.3 Frontend Validation

| Area | Validation |
|------|-----------|
| **File upload** | Type check (JPEG/PNG/WebP/HEIC), 10MB limit |
| **Auth state** | Cross-tab sync via `storage` event |
| **Session** | Auto-redirect on 401, `auth:expired` event |
| **Error boundary** | React ErrorBoundary wraps entire app |
| **Offline** | `useOnlineStatus` hook + OfflineBanner |
| **3D render** | Safe3DBoundary catches Three.js crashes |

---

## 15. Future Enhancements

### Short-Term (Next Release)

1. **Model Integration** — Deploy the 98%+ ConvNeXt-Base model into the FastAPI backend, replacing the current ResNet-18 inference path
2. **Final Evaluation** — Run `evaluate.py` with `--tta` on the 25,904-image test set
3. **WebSocket Chat** — Replace polling with real-time WebSocket for doctor-patient messaging
4. **Push Notifications** — Firebase Cloud Messaging for mobile alerts
5. **ONNX Export** — Convert model to ONNX for CPU-optimized inference

### Medium-Term

6. **Fitzpatrick Fairness Audit** — Per-skin-tone accuracy breakdown
7. **DICOM Support** — Accept dermatoscope images directly
8. **Appointment Video Calls** — Built-in WebRTC (currently video link to external platform)
9. **Multi-Language** — i18n support (Hindi, Tamil, Spanish)
10. **Mobile App** — React Native wrapper with camera integration

### Long-Term

11. **FDA 510(k) Preparation** — Clinical trial documentation for regulatory clearance
12. **EHR Integration** — HL7 FHIR API for hospital system interoperability
13. **Federated Learning** — Train on hospital data without centralizing patient images
14. **Edge Deployment** — TensorRT optimized model on NVIDIA Jetson for clinic devices

---

## 16. Appendix

### A. Running the Application

#### Backend (Development)
```bash
cd backend
pip install -r ../requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend (Development)
```bash
cd frontend-react
npm install
npm run dev    # → http://localhost:5173
```

#### ML Training (Jarvis Labs)
```bash
cd jarvis-training
bash setup.sh
python download_data.py
python prepare_data.py
python train.py --backbone convnext_base.fb_in22k_ft_in1k_384 \
  --img-size 320 --batch-size 96 --lr 6e-4 --epochs 30
python evaluate.py --tta
```

### B. Environment Variables Reference

```env
# Database
DATABASE_URL=mysql+pymysql://user:pass@host:3306/skin_doc
MYSQL_HOST=localhost
MYSQL_DB=skin_doc
MYSQL_USER=root
MYSQL_PASSWORD=secret

# Security
JWT_SECRET=your-256-bit-secret
ACCESS_TOKEN_EXPIRE_MINUTES=120

# Frontend
FRONTEND_ORIGINS=http://localhost:5173,https://app.example.com

# ML Model
LESION_MODEL_WEIGHTS=backend/ml/weights/best_model.pth
LESION_MODEL_BACKBONE=convnext_base
LESION_MALIGNANT_THRESHOLD=0.5

# LLM (pick one)
GEMINI_API_KEY=your-key
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Admin Bootstrap
ADMIN_EMAIL=admin@example.com
ADMIN_USERNAME=admin
ADMIN_PASSWORD=Admin@123

# Azure Storage (optional)
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
AZURE_STORAGE_CONTAINER=lesion-images
```

### C. Project Statistics

| Metric | Count |
|--------|-------|
| **Total Files** | 150+ |
| **Python Backend LOC** | ~8,000 |
| **React Frontend LOC** | ~12,000 |
| **ML Pipeline LOC** | ~3,070 |
| **API Endpoints** | 70+ |
| **React Pages** | 28 |
| **Database Models** | 25+ |
| **Pydantic Schemas** | 30+ |
| **Custom React Hooks** | 3 |
| **Reusable Components** | 15+ |

### D. Team Contributions

| Member | Contributions |
|--------|--------------|
| **Arjungopal Anilkumar** (Team Lead) | Backend architecture, FastAPI routes, ML pipeline design, training infrastructure, database design, inference engine, security implementation, deployment |
| **Suryansh Ram Menon** | Frontend development, React UI, dashboard components, API integration, dataset download scripts, data preparation |
| **Divagar** | Inference & explainability (Grad-CAM), evaluation scripts, use case documentation, quality attributes |

### E. References

1. Liu, Z., Mao, H., Wu, C. Y., et al. (2022). "A ConvNet for the 2020s." CVPR 2022.
2. Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection." ICCV 2017.
3. Tschandl, P., et al. (2018). "The HAM10000 dataset." Scientific Data.
4. ISIC Archive: https://www.isic-archive.com/
5. Fitzpatrick, T. B. (1988). "Sun-reactive skin types I through VI."
6. Groh, M., et al. (2021). "Evaluating Deep Neural Networks Trained on Clinical Images." Nature Medicine.
7. FastAPI Documentation: https://fastapi.tiangolo.com/
8. PyTorch Documentation: https://pytorch.org/docs/
9. timm Library: https://huggingface.co/docs/timm/

---

**Document prepared by Team C1**
**DermaTriage v1.0 — March 2026**
**Amrita Vishwa Vidyapeetham**
