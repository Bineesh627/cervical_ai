import joblib
import shap
import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('Agg')

import os, joblib, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from django.conf import settings

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH          = os.path.join(MODELS_DIR, "clinical_xgb.joblib")
SCALER_PATH         = os.path.join(MODELS_DIR, "clinical_scaler.joblib")
NUM_IMPUTER_PATH    = os.path.join(MODELS_DIR, "clinical_num_imputer.joblib")
OHE_PATH            = os.path.join(MODELS_DIR, "clinical_ohe.joblib")
FEAT_AFTER_OHE_PATH = os.path.join(MODELS_DIR, "clinical_feature_names.joblib")

# MEDIA_ROOT is already configured to point inside your static tree.
SHAP_DIR = os.path.join(settings.MEDIA_ROOT, "shap")

def _safe_to_float(x):
    import re, numpy as np
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    # strip surrounding brackets/quotes
    s = s.strip("[](){}\"'")
    # grab first number-like token (incl. exponent)
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            pass
    return 0.0

def _ensure_training_schema(df_in: pd.DataFrame):
    """
    Build two frames in the *training* raw schema:
      - df_num: numeric raw columns in the order num_imputer.feature_names_in_
      - df_cat: categorical raw columns in the order ohe.feature_names_in_
    Values come from df_in when present, otherwise sensible defaults:
      - numeric: 0
      - categorical: the first category seen during fit (avoids unknowns)
    """
    num_imp   = joblib.load(NUM_IMPUTER_PATH)
    ohe       = joblib.load(OHE_PATH)

    if not hasattr(num_imp, "feature_names_in_"):
        raise RuntimeError("Numeric imputer lacks feature_names_in_. Re-train saving this attribute.")

    num_cols    = list(num_imp.feature_names_in_)
    cat_cols    = list(getattr(ohe, "feature_names_in_", []))
    cat_choices = getattr(ohe, "categories_", None)

    # numeric defaults, override from df_in when present
    row_num = {c: np.nan for c in num_cols}
    for c in num_cols:
        if c in df_in.columns:
            row_num[c] = _safe_to_float(df_in.iloc[0][c])

    # categorical defaults = first seen category; override from df_in when present
    row_cat = {}
    for i, c in enumerate(cat_cols):
        default = (cat_choices[i][0] if cat_choices is not None and len(cat_choices[i]) else "Unknown")
        val = df_in.iloc[0][c] if c in df_in.columns else default
        row_cat[c] = str(val) if pd.notna(val) else str(default)

    df_num = pd.DataFrame([row_num], columns=num_cols)
    df_cat = pd.DataFrame([row_cat], columns=cat_cols).astype(str) if cat_cols else pd.DataFrame(index=[0])

    return df_num, df_cat, num_cols, cat_cols

def _transform_to_model_space(df_num: pd.DataFrame, df_cat: pd.DataFrame):
    """Apply imputer→OHE→concat→reindex→scaler to get X_full (unscaled) and X_scaled."""
    num_imp        = joblib.load(NUM_IMPUTER_PATH)
    ohe            = joblib.load(OHE_PATH)
    scaler         = joblib.load(SCALER_PATH)
    feat_after_ohe = joblib.load(FEAT_AFTER_OHE_PATH)

    # numeric branch
    num_imp_arr = num_imp.transform(df_num)
    df_num_imp  = pd.DataFrame(num_imp_arr, columns=df_num.columns, index=df_num.index)

    # categorical branch
    if len(df_cat.columns) > 0:
        ohe_out = ohe.transform(df_cat)
        try:
            ohe_arr = ohe_out.toarray()      # sparse -> dense
        except AttributeError:
            ohe_arr = np.asarray(ohe_out)    # already dense
        ohe_cols = list(
            ohe.get_feature_names_out(df_cat.columns)
            if hasattr(ohe, "get_feature_names_out")
            else ohe.get_feature_names(df_cat.columns)
        )
        df_cat_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df_cat.index)
    else:
        df_cat_ohe = pd.DataFrame(index=df_num.index)

    # combine to training layout and reindex to final order the model saw
    X_full = pd.concat([df_num_imp, df_cat_ohe], axis=1)
    if isinstance(feat_after_ohe, (list, tuple, np.ndarray)):
        X_full = X_full.reindex(columns=list(feat_after_ohe), fill_value=0.0)

    X_scaled = scaler.transform(X_full)
    return X_full, X_scaled

def generate_shap(df_raw_like: pd.DataFrame, record_id: int) -> str:
    """
    Best-effort SHAP: returns a static-relative path like
    'cervical/uploads/shap/record_<id>_shap.png' or '' on failure.
    Accepts a row containing *any* subset; missing columns are defaulted.
    """
    try:
        model = joblib.load(MODEL_PATH)

        # 1) Build training-schema frames from whatever we got
        if isinstance(df_raw_like, pd.Series):
            df_raw_like = df_raw_like.to_frame().T
        df_num, df_cat, _, _ = _ensure_training_schema(df_raw_like)

        # 2) Transform to model space exactly as during training
        X_full, X_scaled = _transform_to_model_space(df_num, df_cat)

        # 3) SHAP values
        explainer = shap.TreeExplainer(model)
        try:
            shap_values = explainer(X_scaled).values   # newer SHAP
        except Exception:
            sv = explainer.shap_values(X_scaled)       # older SHAP
            shap_values = sv[1] if isinstance(sv, list) else sv

        vals = shap_values[0]
        cols = X_full.columns

        # top features by absolute contribution
        idx = np.argsort(np.abs(vals))[::-1][:10]
        top_feats = [cols[j] for j in idx]
        top_vals  = vals[idx]

        os.makedirs(SHAP_DIR, exist_ok=True)
        save_path = os.path.join(SHAP_DIR, f"record_{record_id}_shap.png")

        plt.figure(figsize=(8, 4))
        y = np.arange(len(top_feats))
        plt.barh(y, top_vals[::-1], color=np.where(top_vals[::-1] >= 0, 'green', 'red'))
        plt.yticks(y, top_feats[::-1])
        plt.title("Top SHAP Feature Impacts")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        # return static-relative path that templates can render
        return os.path.join("cervical", "uploads", "shap", f"record_{record_id}_shap.png").replace("\\", "/")

    except Exception as e:
        print("[Warning] SHAP generation failed (non-fatal):", e)
        return ""
