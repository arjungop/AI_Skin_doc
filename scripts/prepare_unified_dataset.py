#!/usr/bin/env python3
"""
Hierarchical Dataset Preparation (Maximized Integration)
Sources:
1. ISIC 2019 (~25k) - CSV Labeled
2. DermNet (~9k) - Folder Labeled
3. Skin Disease Dataset (~700) - Folder Labeled
4. Fitzpatrick17k (~16.5k) - CSV Labeled (Full Spectrum)
5. Diverse Derm (~2.3k) - Folder Labeled
6. ISIC 2018 (~10k) - Metadata Fallback
"""
import os
import csv
import shutil
import hashlib
from pathlib import Path
from tqdm import tqdm

# Configuration
DATASETS_DIR = Path("datasets")
UNIFIED_DIR = Path("data/unified_train")

# Paths
ISIC_2019_CSV = DATASETS_DIR / "isic_data/isic_2019/ISIC_2019_Training_GroundTruth.csv"
ISIC_2019_IMG = DATASETS_DIR / "isic_data/isic_2019/ISIC_2019_Training_Input"
ISIC_2018_CSV = DATASETS_DIR / "ISIC2018_Task3_Training_Input/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"
ISIC_2018_IMG = DATASETS_DIR / "ISIC2018_Task3_Training_Input"
DERMNET_DIR = DATASETS_DIR / "dermnet_main/train"
SKIN_DISEASE_DIR = DATASETS_DIR / "skin_disease/skin-disease-datasaet/train_set"
FITZPATRICK_CSV = DATASETS_DIR / "data/fitzpatrick17k.csv"
FITZPATRICK_IMG = DATASETS_DIR / "data/finalfitz17k"
DIVERSE_DERM_DIR = DATASETS_DIR / "diverse_derm/Skin cancer ISIC The International Skin Imaging Collaboration"

# --- Hierarchical Structure ---
HIERARCHY = {
    "cancer": ["melanoma", "bcc", "scc", "ak"],
    "benign": ["nevus", "seborrheic_keratosis", "angioma", "wart"],
    "inflammatory": ["eczema", "psoriasis", "lichen_planus", "urticaria"],
    "infectious": ["impetigo", "herpes", "candida", "scabies"],
    "pigmentary": ["vitiligo", "melasma", "hyperpigmentation"],
}

AUX_MAPPING = {
    "acne": "inflammatory", "rosacea": "inflammatory", 
    "lupus": "inflammatory", "bullous": "inflammatory", "vasculitis": "inflammatory",
    "tinea": "infectious", "cellulitis": "infectious", "exanthem": "infectious",
    # Generic Aux categories (for things that don't match specific diseases)
    "benign_other": "benign",
    "inflammatory_other": "inflammatory",
    "infectious_other": "infectious", 
    "cancer_other": "cancer",
    "pigmentary_other": "pigmentary"
}

MAIN_DISEASES = set([d for cat in HIERARCHY.values() for d in cat])

def clean_and_create_dirs():
    if UNIFIED_DIR.exists(): shutil.rmtree(UNIFIED_DIR)
    for d in MAIN_DISEASES: (UNIFIED_DIR/d).mkdir(parents=True, exist_ok=True)
    for aux in AUX_MAPPING.keys(): (UNIFIED_DIR/f"_aux_{aux}").mkdir(parents=True, exist_ok=True)

def link_file(src: Path, target: str, prefix: str):
    if not src.exists(): return
    
    dest_sub = None
    if target in MAIN_DISEASES: dest_sub = target
    elif target in AUX_MAPPING: dest_sub = f"_aux_{target}"
    
    if dest_sub:
        dest = UNIFIED_DIR / dest_sub / f"{prefix}_{src.name}"
        try: os.symlink(src.resolve(), dest)
        except: pass

# ... (ISIC / DermNet / SkinDisease / DiverseDerm functions same as before) ...
def process_isic_2019():
    print("Processing ISIC 2019...")
    cols = {'MEL':'melanoma','BCC':'bcc','SCC':'scc','AK':'ak','NV':'nevus','BKL':'seborrheic_keratosis','VASC':'angioma'}
    if ISIC_2019_CSV.exists():
        with open(ISIC_2019_CSV) as f:
            for row in tqdm(csv.DictReader(f)):
                for k,v in cols.items():
                    if float(row.get(k,0))==1.0:
                        link_file(ISIC_2019_IMG/(row['image']+".jpg"), v, "isic19")
                        break

def process_isic_2018():
    print("Processing ISIC 2018...")
    cols = {'MEL':'melanoma', 'BCC':'bcc', 'AKIEC':'ak','NV':'nevus', 'BKL':'seborrheic_keratosis', 'VASC':'angioma'}
    if ISIC_2018_CSV.exists():
        with open(ISIC_2018_CSV) as f:
            for row in tqdm(csv.DictReader(f)):
                target=None
                for k,v in cols.items():
                    if float(row.get(k,0))==1.0: target=v; break
                if target: link_file(ISIC_2018_IMG/(row['image']+".jpg"), target, "isic18")

def process_dermnet():
    print("Processing DermNet...")
    keywords = [
        ("acne","acne"),("rosacea","rosacea"),("lupus","lupus"),("bullous","bullous"),
        ("vasculitis","vasculitis"),("tinea","tinea"),("cellulitis","cellulitis"),
        ("vitiligo","vitiligo"),("melasma","melasma"),("hyperpigmentation","hyperpigmentation"),
        ("melanoma","melanoma"),("basal cell","bcc"),("squamous cell","scc"),("actinic","ak"),
        ("nevus","nevus"),("mole","nevus"),("seborrheic","seborrheic_keratosis"),
        ("angioma","angioma"),("wart","wart"),("lichen planus","lichen_planus"),
        ("psoriasis","psoriasis"),("eczema","eczema"),("dermatitis","eczema"),
        ("urticaria","urticaria"),("hives","urticaria"),("impetigo","impetigo"),
        ("herpes","herpes"),("candida","candida"),("scabies","scabies")
    ]
    if DERMNET_DIR.exists():
        for root,_,files in os.walk(DERMNET_DIR):
            for f in files:
                if f.startswith('.'): continue
                fl = f.lower()
                for k,v in keywords:
                    if k in fl: link_file(Path(root)/f, v, "dermnet"); break

def process_skin_disease():
    print("Processing Skin Disease Dataset...")
    mapping = {
        "BA- cellulitis":"cellulitis", "BA-impetigo":"impetigo",
        "FU-athlete-foot":"tinea", "FU-ringworm":"tinea",
        "VI-shingles":"herpes", "VI-chickenpox":"exanthem"
    }
    if SKIN_DISEASE_DIR.exists():
        for d in SKIN_DISEASE_DIR.iterdir():
            if d.name in mapping:
                for f in d.iterdir():
                    if not f.name.startswith('.'): link_file(f, mapping[d.name], "sd")

def process_diverse_derm():
    print("Processing Diverse Derm...")
    mapping = {
        "melanoma": "melanoma", "pigmented benign keratosis": "seborrheic_keratosis",
        "nevus": "nevus", "basal cell carcinoma": "bcc", "actinic keratosis": "ak",
        "squamous cell carcinoma": "scc", "vascular lesion": "angioma",
        "seborrheic keratosis": "seborrheic_keratosis", "acne": "acne", 
        "tinea": "tinea", "wart": "wart", "eczema": "eczema", "psoriasis": "psoriasis"
    }
    if DIVERSE_DERM_DIR.exists():
        for root, dirs, files in os.walk(DIVERSE_DERM_DIR):
            target = mapping.get(Path(root).name.lower())
            if target:
                for f in files:
                    if not f.startswith('.'): link_file(Path(root)/f, target, "diverse")

def process_fitzpatrick():
    print("Processing Fitzpatrick17k (Full Spectrum)...")
    if not FITZPATRICK_CSV.exists(): return
    
    # MASTER MAPPING (114 LABELS)
    # Maps label string -> Target Disease OR Aux Category
    # If mapped to Aux Category, it goes to e.g. _aux_benign_other
    label_map = {
        # CANCER
        "melanoma": "melanoma", "malignant melanoma": "melanoma",
        "superficial spreading melanoma ssm": "melanoma", "lentigo maligna": "melanoma",
        "basal cell carcinoma": "bcc", "basal cell carcinoma morpheiform": "bcc",
        "solid cystic basal cell carcinoma": "bcc",
        "squamous cell carcinoma": "scc", "keratoacanthoma": "scc", # often grouped
        "actinic keratosis": "ak", "porokeratosis actinic": "ak",
        "kaposi sarcoma": "cancer_other", "mycosis fungoides": "cancer_other",
        
        # BENIGN
        "nevus": "nevus", "becker nevus": "nevus", "halo nevus": "nevus",
        "congenital nevus": "nevus", "epidermal nevus": "nevus", "nevocytic nevus": "nevus",
        "nevus sebaceous of jadassohn": "nevus", "naevus comedonicus": "nevus",
        "seborrheic keratosis": "seborrheic_keratosis",
        "angioma": "angioma", "cherry angioma": "angioma", "port wine stain": "angioma",
        "pyogenic granuloma": "angioma", "lymphangioma": "angioma", "telangiectases": "angioma",
        "wart": "wart", "warts": "wart",
        # Benign Other (Aux)
        "dermatofibroma": "benign_other", "keloid": "benign_other",
        "mucous cyst": "benign_other", "pilar cyst": "benign_other", "epidermoid cyst": "benign_other",
        "lipoma": "benign_other", "syringoma": "benign_other", "neurofibromatosis": "benign_other",
        "seborrheic dermatitis": "benign_other", "milia": "benign_other",
        "pilomatricoma": "benign_other", "calcinosis cutis": "benign_other",
        "fordyce spots": "benign_other",
        
        # INFLAMMATORY
        "eczema": "eczema", "dyshidrotic eczema": "eczema", 
        "atopic dermatitis": "eczema", "allergic contact dermatitis": "eczema",
        "psoriasis": "psoriasis", "pustular psoriasis": "psoriasis",
        "lichen planus": "lichen_planus", "lichen simplex": "lichen_planus",
        "lichen amyloidosis": "lichen_planus",
        "urticaria": "urticaria", "urticaria pigmentosa": "urticaria",
        "acne": "acne", "acne vulgaris": "acne", "folliculitis": "acne", 
        "rhinophyma": "rosacea", "rosacea": "rosacea",
        "hidradenitis": "acne", # closest fit for visual inflammation
        "lupus erythematosus": "lupus", "lupus subacute": "lupus",
        # Inflammatory Other (Aux)
        "granuloma annulare": "inflammatory_other", "granuloma pyogenic": "inflammatory_other",
        "pityriasis rosea": "inflammatory_other", "pityriasis lichenoides chronica": "inflammatory_other",
        "pityriasis rubra pilaris": "inflammatory_other", 
        "erythema nodosum": "inflammatory_other", "erythema multiforme": "inflammatory_other",
        "erythema annulare centrifigum": "inflammatory_other", "erythema elevatum diutinum": "inflammatory_other",
        "prurigo nodularis": "inflammatory_other", "neurodermatitis": "inflammatory_other",
        "scleroderma": "inflammatory_other", "sarcoidosis": "inflammatory_other",
        "dermatomyositis": "inflammatory_other", "neutrophilic dermatoses": "inflammatory_other",
        # Bullous
        "bullous": "bullous", "epidermolysis bullosa": "bullous", 
        "hailey hailey disease": "bullous", "pemphigus": "bullous",
        
        # INFECTIOUS
        "impetigo": "impetigo",
        "herpes": "herpes",
        "scabies": "scabies", "pediculosis lids": "scabies", 
        "tinea corporis": "tinea", "tinea pedis": "tinea",
        "molluscum": "infectious_other", "viral": "infectious_other",
        "myiasis": "infectious_other", "tungiasis": "infectious_other",
        "paronychia": "infectious_other", # often bacterial/fungal
        
        # PIGMENTARY
        "vitiligo": "vitiligo", "acanthosis nigricans": "pigmentary_other",
        "melasma": "melasma", "lentigo maligna": "melanoma", # Oops, malignant! (Fixed above)
        "post-inflammatory": "hyperpigmentation", 
        "incontinentia pigmenti": "pigmentary_other",
        "xeroderma pigmentosum": "pigmentary_other"
    }

    with open(FITZPATRICK_CSV) as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            label = row['label']
            target = label_map.get(label)
            
            # Simple keyword fallback if no direct match
            if not target:
                if "melanoma" in label: target = "melanoma"
                elif "carcinoma" in label: target = "cancer_other"
                elif "dermatitis" in label: target = "eczema"
                elif "lichen" in label: target = "lichen_planus"
                elif "tinea" in label: target = "tinea"
            
            if target:
                md5 = row['md5hash']
                src = FITZPATRICK_IMG / f"{md5}.jpg"
                link_file(src, target, "fitz")

def save_hierarchy():
    import json
    all_f = [d.name for d in UNIFIED_DIR.iterdir() if d.is_dir()]
    f_cat = {}
    for f in all_f:
        if f.startswith("_aux_"): f_cat[f] = AUX_MAPPING[f.replace("_aux_","")]
        else:
            for c,ds in HIERARCHY.items():
                if f in ds: f_cat[f]=c; break
    
    meta = {
        "hierarchy": HIERARCHY,
        "categories": list(HIERARCHY.keys()),
        "folder_to_category": f_cat,
        "is_auxiliary": {f: f.startswith("_aux_") for f in all_f}
    }
    with open(UNIFIED_DIR/"hierarchy.json",'w') as f: json.dump(meta,f)

if __name__ == "__main__":
    clean_and_create_dirs()
    process_isic_2019()
    process_isic_2018()
    process_dermnet()
    process_fitzpatrick()
    process_skin_disease()
    process_diverse_derm()
    save_hierarchy()
    
    total = 0
    print("\n--- Final Counts ---")
    for d in sorted(os.listdir(UNIFIED_DIR)):
        if (UNIFIED_DIR/d).is_dir():
            n = len(list((UNIFIED_DIR/d).iterdir()))
            print(f"{d}: {n}")
            total += n
    print(f"Total: {total}")
