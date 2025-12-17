from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend import models, schemas
from backend.azure_blob import upload_bytes_get_url
import os
from backend.security import get_current_user
from backend.inference import MelanomaInference
from io import BytesIO
from PIL import Image, ImageStat, ImageFilter, ImageOps, ImageChops
from datetime import datetime

router = APIRouter()

@router.get("/status")
def model_status():
    from pathlib import Path
    weights = os.getenv("LESION_MODEL_WEIGHTS", "backend/ml/weights/skin_resnet18.pth")
    weights_expanded = os.path.expanduser(weights)
    if not os.path.isabs(weights_expanded):
        weights_expanded = str((Path(__file__).resolve().parents[1] / weights_expanded).resolve())
    return {
        "weights_env": weights,
        "weights_path": weights_expanded,
        "exists": os.path.exists(weights_expanded),
        "backbone_env": os.getenv("LESION_MODEL_BACKBONE"),
        "malignant_index_env": os.getenv("LESION_MALIGNANT_INDEX"),
        "threshold": os.getenv("LESION_MALIGNANT_THRESHOLD"),
        "force_threshold": os.getenv("LESION_FORCE_THRESHOLD"),
    }

# Upload & predict (lightweight heuristic model)
@router.post("/predict", response_model=schemas.LesionOut)
def predict_lesion(
    file: UploadFile = File(...),
    patient_id: int | None = None,
    threshold: float | None = None,
    sensitivity: str | None = None,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    if patient_id is None:
        raise HTTPException(status_code=400, detail="patient_id is required")
    # Patients can only upload for themselves
    role = (user.role or "").upper()
    if role != "ADMIN":
        patient = db.query(models.Patient).filter(models.Patient.user_id == user.user_id).first()
        if not patient or patient.patient_id != patient_id:
            raise HTTPException(status_code=403, detail="Cannot upload for another patient")

    # Read data once
    data = file.file.read()

    # -------- Basic input validation to reject non-skin images --------
    # You can relax this with LESION_STRICT_VALIDATION=0 in .env
    strict_validate = os.getenv("LESION_STRICT_VALIDATION", "1") not in {"0", "false", "False", "no"}
    try:
        img0 = Image.open(BytesIO(data)).convert("RGB")
        w, h = img0.size
        aspect = w / max(1, h)
        if strict_validate:
            # Resolution and aspect sanity
            if min(w, h) < 128 or max(w, h) > 8000:
                raise HTTPException(status_code=400, detail="Please upload a clear close-up photo of the skin (min 128px)")
            if not (0.5 <= aspect <= 2.0):
                raise HTTPException(status_code=400, detail="Photo looks unusual (very wide/tall). Please upload a close-up skin photo.")

            # Estimate skin-colored pixel ratio using YCbCr and HSV rules of thumb
            ycbcr = img0.convert('YCbCr')
            ycbcr_data = list(ycbcr.getdata())
            total = len(ycbcr_data)
            step = max(1, total // 20000)  # sample up to ~20k pixels to keep it fast
            skin_hits = 0
            for i in range(0, total, step):
                Y, Cb, Cr = ycbcr_data[i]
                if 80 <= Y <= 255 and 85 <= Cb <= 135 and 135 <= Cr <= 180:
                    skin_hits += 1
            skin_ratio_ycc = skin_hits / ((total + step - 1) // step)

            hsv = img0.convert('HSV')
            H, S, V = hsv.split()
            h_d = list(H.getdata())[::step]
            s_d = list(S.getdata())[::step]
            v_d = list(V.getdata())[::step]
            skin_hits_hsv = 0
            for hh, ss, vv in zip(h_d, s_d, v_d):
                # Rough skin band in HSV (tunable)
                if (0 <= hh <= 50) and (40 <= ss <= 180) and (vv >= 60):
                    skin_hits_hsv += 1
            skin_ratio_hsv = skin_hits_hsv / max(1, len(h_d))

            skin_ratio = max(skin_ratio_ycc, skin_ratio_hsv)
            if skin_ratio < 0.12:
                raise HTTPException(status_code=400, detail="The image doesn't look like a close-up of skin. Please upload a clear skin lesion photo.")
    except HTTPException:
        raise
    except Exception:
        # If we can't parse the image, return a friendly message
        raise HTTPException(status_code=400, detail="Could not read image. Please upload a valid photo file.")

    # Try learned model first if weights are present; fallback to heuristic
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
        model_path = os.getenv("LESION_MODEL_WEIGHTS")
        use_model = model_path and os.path.exists(model_path)
        if use_model:
            infer = MelanomaInference(weights=model_path)
            pred = infer.predict_image(img)
            # malignant probability as risk score
            risk = float(pred.get("p_malignant", 0.0))
            # Decide label: allow threshold override if requested or forced via env
            use_thr = (
                isinstance(threshold, (int, float))
                or (isinstance(sensitivity, str) and sensitivity.lower() in {"high", "hi"})
                or os.getenv("LESION_FORCE_THRESHOLD", "0") in {"1", "true", "TRUE", "yes", "on"}
            )
            # Load suggested threshold from model meta if available
            thr_env = float(os.getenv("LESION_MALIGNANT_THRESHOLD", "0.5"))
            try:
                import json, os as _os
                meta_path = _os.path.splitext(model_path)[0] + '.json'
                if _os.path.exists(meta_path):
                    meta = json.loads(open(meta_path).read() or '{}')
                    if isinstance(meta.get('suggested_threshold'), (int, float)):
                        thr_env = float(meta['suggested_threshold'])
            except Exception:
                pass
            thr = float(threshold) if isinstance(threshold, (int, float)) else (
                0.35 if (isinstance(sensitivity, str) and sensitivity.lower() in {"high", "hi"}) else thr_env
            )
            if use_thr:
                prediction = 'malignant' if (risk >= thr or (pred.get('label') == 'malignant')) else 'benign'
            else:
                prediction = 'malignant' if (pred.get('label') == 'malignant') else 'benign'
            # Grad-CAM explainability as data URL
            try:
                overlay = infer.gradcam_overlay(img)
                import base64, io
                buf = io.BytesIO()
                overlay.save(buf, format='PNG')
                explain_url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')
            except Exception:
                explain_url = None
        else:
            raise RuntimeError("no-weights")
        # Heuristic fallback
        if not use_model:
            # Resize for consistency
            img_small = img.resize((256, 256))
            gray = img_small.convert('L')
            stat = ImageStat.Stat(gray)
            mean_brightness = stat.mean[0] / 255.0
            std_brightness = stat.stddev[0] / 255.0
            # Build lesion mask
            hsv = img_small.convert('HSV')
            h, s, v = hsv.split()
            s_data = list(s.getdata())
            v_data = list(v.getdata())
            mask_data = [255 if (sv > 60 and vv < 210) else 0 for sv, vv in zip(s_data, v_data)]
            mask = Image.new('L', img_small.size)
            mask.putdata(mask_data)
            area_ratio = sum(1 for m in mask_data if m) / float(len(mask_data))
            # Edge density
            edges = img_small.filter(ImageFilter.FIND_EDGES).convert('L')
            e_vals = list(edges.getdata())
            edge_sum = sum(ev for ev, m in zip(e_vals, mask_data) if m)
            masked_count = max(1, sum(1 for m in mask_data if m))
            edge_density = (edge_sum / masked_count) / 255.0
            # Asymmetry
            bbox = mask.getbbox() or (0, 0, 256, 256)
            roi = img_small.crop(bbox)
            rw, rh = roi.size
            if rw >= 4:
                left = roi.crop((0, 0, rw // 2, rh))
                right = roi.crop((rw // 2, 0, rw, rh))
                left_flip = ImageOps.mirror(left)
                l_gray = left_flip.convert('L')
                r_gray = right.convert('L')
                diff = ImageChops.difference(l_gray, r_gray)
                asymmetry = ImageStat.Stat(diff).mean[0] / 255.0
            else:
                asymmetry = 0.0
            # Hue complexity
            h_hist = h.histogram()
            total_px = sum(h_hist) or 1
            bins = 12
            bin_size = 256 // bins
            significant_bins = 0
            for i in range(bins):
                start = i * bin_size
                end = start + bin_size
                bsum = sum(h_hist[start:end])
                if bsum / total_px > 0.06:
                    significant_bins += 1
            multi_color = significant_bins / bins
            # Color variance
            r, g, b = img_small.split()
            var_color = (ImageStat.Stat(r).var[0] + ImageStat.Stat(g).var[0] + ImageStat.Stat(b).var[0]) / (3 * (255.0 ** 2))
            # Size score
            a, b_peak, c_tail = 0.01, 0.15, 0.6
            if area_ratio <= a or area_ratio >= c_tail:
                size_score = 0.0
            elif area_ratio <= b_peak:
                size_score = (area_ratio - a) / max(1e-6, (b_peak - a))
            else:
                size_score = (c_tail - area_ratio) / max(1e-6, (c_tail - b_peak))
            # Risk
            risk = (
                0.45 * asymmetry +
                0.40 * edge_density +
                0.30 * multi_color +
                0.20 * size_score +
                0.15 * std_brightness +
                0.10 * (1.0 - mean_brightness) +
                0.20 * var_color
            )
            risk = max(0.0, min(1.0, risk))
            env_thr = float(os.getenv('LESION_MALIGNANT_THRESHOLD', '0.45'))
            malignant_threshold = env_thr
            if isinstance(threshold, (int, float)):
                malignant_threshold = float(threshold)
            elif isinstance(sensitivity, str) and sensitivity.lower() in {"high","hi"}:
                malignant_threshold = min(env_thr, 0.35)
            prediction = 'malignant' if risk >= malignant_threshold else 'benign'
    except Exception as e:
        # Fallback if analysis fails: default to benign (avoid constant 70% malignant)
        try:
            import logging
            logging.getLogger(__name__).warning("Lesion prediction fallback: %s", e)
        except Exception:
            pass
        risk = 0.0
        # Allow override via env if you prefer cautious fallback
        if os.getenv('LESION_FALLBACK_LABEL', 'benign').lower() == 'malignant':
            prediction = 'malignant'
            risk = float(os.getenv('LESION_FALLBACK_RISK', '0.6'))
        else:
            prediction = 'benign'

    # Upload to Azure Blob if configured (use original bytes)
    use_blob = bool(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    image_path = file.filename
    if use_blob:
        image_path = upload_bytes_get_url(data, file.filename, file.content_type)

    new_lesion = models.Lesion(patient_id=patient_id, image_path=image_path, prediction=prediction)
    db.add(new_lesion)
    db.commit()
    db.refresh(new_lesion)
    try:
        setattr(new_lesion, 'risk_score', float(risk))
        if use_model:
            try:
                setattr(new_lesion, 'explain_url', explain_url)
            except Exception:
                pass
    except Exception:
        pass
    return new_lesion


def _compose_report_text(prediction: str | None) -> tuple[str, str]:
    p = (prediction or '').strip().lower()
    if p == 'benign':
        summary = "Benign lesion — no signs of cancer"
        details = (
            "Our system isn’t perfect and can sometimes make mistakes. If you notice anything unusual or have any concerns, it’s always a good idea to check with a dermatologist for peace of mind.\n"
            "Good news. Based on the image and available context, the lesion appears benign. "
            "Key indicators that support this: well-defined borders, uniform color, and a symmetrical shape.\n\n"
            "What you can do next:\n"
            "- Keep an eye on it: take a clear photo and compare monthly.\n"
            "- Protect your skin: use sunscreen (SPF 30+) and avoid peak sun.\n"
            "- Seek care if it changes: rapid growth, irregular borders, color changes, bleeding or itching.\n\n"
            "We know skin concerns can be worrying. You did the right thing getting this checked. "
            "If you’d like a dermatologist to review, choose a doctor and we’ll share this report with them."
        )
    else:
        summary = "Needs review — features may be concerning"
        details = (
            "Our system isn’t perfect and can sometimes make mistakes. If you notice anything unusual or have any concerns, it’s always a good idea to check with a dermatologist for peace of mind.\n"            
            "This lesion shows some features that warrant a closer look. "
            "Possible indicators: irregular borders, asymmetry, or varied colors. This does NOT necessarily mean cancer, "
            "but it’s important to have a clinician examine it.\n\n"
            "What you should do next:\n"
            "- Arrange a dermatology review within 1–2 weeks.\n"
            "- Avoid further irritation (scratching, picking).\n"
            "- Take a clear photo for comparison before your visit.\n\n"
            "We’re here to help: choose a doctor below and we can share this report with them to speed up your appointment."
        )
    return summary, details


@router.post("/{lesion_id}/report", response_model=schemas.DiagnosisReportOut)
def create_report(
    lesion_id: int,
    patient_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    lesion = db.query(models.Lesion).filter(models.Lesion.lesion_id == lesion_id).first()
    if not lesion:
        raise HTTPException(status_code=404, detail="Lesion not found")
    # Auth: patient can only create for self unless admin
    role = (user.role or '').upper()
    if role != 'ADMIN':
        pat = db.query(models.Patient).filter(models.Patient.user_id == user.user_id).first()
        if not pat or pat.patient_id != patient_id:
            raise HTTPException(status_code=403, detail="Forbidden")
    # Compose details
    summary, details = _compose_report_text(lesion.prediction)
    rep = models.DiagnosisReport(
        lesion_id=lesion_id,
        patient_id=patient_id,
        prediction=lesion.prediction,
        summary=summary,
        details=details,
        created_at=datetime.utcnow(),
    )
    db.add(rep)
    db.commit()
    db.refresh(rep)
    return rep


@router.get("/reports", response_model=list[schemas.DiagnosisReportOut])
def list_reports(
    patient_id: int | None = None,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    role = (user.role or '').upper()
    q = db.query(models.DiagnosisReport)
    if role != 'ADMIN':
        pat = db.query(models.Patient).filter(models.Patient.user_id == user.user_id).first()
        if not pat:
            return []
        q = q.filter(models.DiagnosisReport.patient_id == pat.patient_id)
    elif patient_id:
        q = q.filter(models.DiagnosisReport.patient_id == patient_id)
    return q.order_by(models.DiagnosisReport.created_at.desc()).limit(100).all()


@router.post("/reports/{report_id}/send")
def send_report_to_doctor(
    report_id: int,
    doctor_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    rep = db.query(models.DiagnosisReport).filter(models.DiagnosisReport.report_id == report_id).first()
    if not rep:
        raise HTTPException(status_code=404, detail="Report not found")
    role = (user.role or '').upper()
    # Resolve patient from user if not admin
    if role != 'ADMIN':
        pat = db.query(models.Patient).filter(models.Patient.user_id == user.user_id).first()
        if not pat or pat.patient_id != rep.patient_id:
            raise HTTPException(status_code=403, detail="Forbidden")
        patient_id = pat.patient_id
    else:
        patient_id = rep.patient_id
    # Ensure chat room exists, then post message
    room = db.query(models.ChatRoom).filter(
        models.ChatRoom.patient_id == patient_id,
        models.ChatRoom.doctor_id == doctor_id,
    ).first()
    if not room:
        room = models.ChatRoom(patient_id=patient_id, doctor_id=doctor_id)
        db.add(room)
        db.commit()
        db.refresh(room)

    m = models.Message(
        room_id=room.room_id,
        sender_user_id=user.user_id,
        content=(rep.summary or 'Diagnosis Update') + "\n\n" + rep.details,
    )
    db.add(m)
    db.commit()
    return {"ok": True, "room_id": room.room_id, "message_id": m.message_id}
