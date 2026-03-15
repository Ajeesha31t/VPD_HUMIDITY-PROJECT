import os, json, datetime, warnings, urllib.request
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, jsonify, request, Response

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import csv, io

app = Flask(__name__)
PKL = "trained_models.pkl"
CSV = "coimbatore_with_vpd.csv"
FEATURE_NAMES = ["YEAR", "DOY", "T2M", "sin_doy", "cos_doy", "T2M_sq", "T2M_sin"]

def get_vpd_alert(vpd):
    if vpd is None: return None
    vpd = float(vpd)
    if vpd < 0.4:
        return {"level":"optimal","label":"Optimal Conditions",
                "message":"VPD is in the ideal range. Plants are transpiring efficiently.",
                "color":"#00d68f"}
    elif vpd < 0.8:
        return {"level":"low","label":"Low Stress",
                "message":"Slightly low VPD. Monitor for potential fungal issues.",
                "color":"#22d3a0"}
    elif vpd < 1.2:
        return {"level":"medium","label":"Moderate Stress",
                "message":"Moderate VPD. Plants may experience mild water stress.",
                "color":"#fbbf24"}
    elif vpd < 1.6:
        return {"level":"high","label":"High Stress",
                "message":"High VPD detected. Increase irrigation to prevent wilting.",
                "color":"#f97316"}
    else:
        return {"level":"severe","label":"Severe Stress",
                "message":"Severe VPD! Crops at risk. Immediate irrigation recommended.",
                "color":"#ff4444"}

def make_features(df):
    X = pd.DataFrame()
    X["YEAR"]    = df["YEAR"]
    X["DOY"]     = df["DOY"]
    X["T2M"]     = df["T2M"]
    X["sin_doy"] = np.sin(2 * np.pi * df["DOY"] / 365)
    X["cos_doy"] = np.cos(2 * np.pi * df["DOY"] / 365)
    X["T2M_sq"]  = df["T2M"] ** 2
    X["T2M_sin"] = df["T2M"] * np.sin(2 * np.pi * df["DOY"] / 365)
    return X

def train_all():
    print("Training models... please wait")
    df = pd.read_csv(CSV)
    X  = make_features(df)
    results = {}
    targets = {"VPD": df["VPD"], "Humidity": df["RH2M"]}
    needs_scale = {"SVR (RBF)", "Polynomial SVM"}
    model_defs = {
        "SVR (RBF)":         lambda: SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1),
        "Polynomial SVM":    lambda: SVR(kernel='poly', degree=2, C=10, epsilon=0.1),
        "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
        "AdaBoost":          lambda: AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }
    for tname, y in targets.items():
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        base_oof, base_te = {}, {}
        for mname, mfn in model_defs.items():
            scaler = StandardScaler()
            if mname in needs_scale:
                Xtr_s = scaler.fit_transform(X_tr)
                Xte_s = scaler.transform(X_te)
            else:
                Xtr_s = X_tr.values; Xte_s = X_te.values
            kf  = KFold(n_splits=5, shuffle=True, random_state=42)
            oof = np.zeros(len(X_tr))
            for fi, vi in kf.split(Xtr_s):
                m = mfn(); m.fit(Xtr_s[fi], y_tr.values[fi])
                oof[vi] = m.predict(Xtr_s[vi])
            m_final = mfn()
            m_final.fit(Xtr_s, y_tr)
            pred = m_final.predict(Xte_s)
            base_oof[mname] = oof
            base_te[mname]  = pred
            fi_dict = None
            if hasattr(m_final, "feature_importances_"):
                fi_dict = dict(zip(FEATURE_NAMES, [round(v,4) for v in m_final.feature_importances_]))
            results[f"{mname}|{tname}"] = {
                "actual": y_te.tolist(), "pred": pred.tolist(),
                "r2":   round(r2_score(y_te, pred), 4),
                "mae":  round(mean_absolute_error(y_te, pred), 4),
                "rmse": round(np.sqrt(mean_squared_error(y_te, pred)), 4),
                "feature_importance": fi_dict,
                "model_obj": m_final,
                "scaler": scaler if mname in needs_scale else None,
            }
        S_tr = np.column_stack([base_oof[m] for m in model_defs])
        S_te = np.column_stack([base_te[m]  for m in model_defs])
        meta = Ridge(alpha=1.0)
        meta.fit(S_tr, y_tr)
        h_pred = meta.predict(S_te)
        results[f"Hybrid|{tname}"] = {
            "actual": y_te.tolist(), "pred": h_pred.tolist(),
            "r2":   round(r2_score(y_te, h_pred), 4),
            "mae":  round(mean_absolute_error(y_te, h_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_te, h_pred)), 4),
            "feature_importance": results[f"Gradient Boosting|{tname}"]["feature_importance"],
            "model_obj": None, "scaler": None,
            "meta_coefs": dict(zip(list(model_defs.keys()), [round(c,4) for c in meta.coef_])),
        }
    joblib.dump(results, PKL)
    print("All models trained and saved!")
    return results

def load():
    if os.path.exists(PKL):
        print("Loading saved models...")
        return joblib.load(PKL)
    return train_all()

try:
    RESULTS = load()
except Exception as e:
    print(f"Error loading models: {e}")
    RESULTS = {}

def predict_row(year, doy, t2m):
    row = np.array([[year, doy, t2m,
                     np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365),
                     t2m**2, t2m*np.sin(2*np.pi*doy/365)]])
    base_names = ["SVR (RBF)", "Polynomial SVM", "Gradient Boosting", "AdaBoost"]
    preds = {}
    for mname in base_names:
        for tname in ["VPD", "Humidity"]:
            key = f"{mname}|{tname}"
            if key not in RESULTS: continue
            m_obj  = RESULTS[key]["model_obj"]
            scaler = RESULTS[key]["scaler"]
            if m_obj is None: continue
            x   = scaler.transform(row) if scaler else row
            val = round(float(m_obj.predict(x)[0]), 3)
            preds.setdefault(mname, {})[tname] = val
    for tname in ["VPD", "Humidity"]:
        key = f"Hybrid|{tname}"
        if key in RESULTS:
            coefs = RESULTS[key].get("meta_coefs", {})
            val   = sum(coefs.get(m, 0) * preds.get(m, {}).get(tname, 0) for m in base_names)
            preds.setdefault("Hybrid", {})[tname] = round(val, 3)
    return preds

def safe_json(d):
    return {k: v for k,v in d.items() if k not in ("model_obj","scaler")}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/data")
def api_data():
    models  = ["SVR (RBF)", "Polynomial SVM", "Gradient Boosting", "AdaBoost", "Hybrid"]
    targets = ["VPD", "Humidity"]
    out = {}
    for m in models:
        out[m] = {}
        for t in targets:
            key = f"{m}|{t}"
            if key in RESULTS:
                out[m][t] = safe_json(RESULTS[key])
    return jsonify(out)

@app.route("/api/live")
def api_live():
    today = datetime.date.today()
    doy   = today.timetuple().tm_yday
    year  = today.year
    try:
        url = ("https://api.open-meteo.com/v1/forecast"
               "?latitude=11.0168&longitude=76.9558"
               "&current_weather=true&hourly=relativehumidity_2m&forecast_days=1")
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        t2m = data["current_weather"]["temperature"]
        rh  = data["hourly"]["relativehumidity_2m"][datetime.datetime.now().hour]
    except Exception:
        t2m, rh = 28.0, 65.0
    preds      = predict_row(year, doy, t2m)
    hybrid_vpd = preds.get("Hybrid", {}).get("VPD", None)
    vpd_alert  = get_vpd_alert(hybrid_vpd)
    return jsonify({
        "date": str(today), "doy": doy, "year": year,
        "temperature": t2m, "humidity_live": rh,
        "predictions": preds, "vpd_alert": vpd_alert
    })

@app.route("/api/predict")
def api_predict():
    year      = int(request.args.get("year", 2026))
    doy       = int(request.args.get("doy",  180))
    t2m       = float(request.args.get("t2m", 28.0))
    sel_model = request.args.get("model", None)
    preds     = predict_row(year, doy, t2m)
    chosen_vpd = preds.get(sel_model, {}).get("VPD") if sel_model else preds.get("Hybrid", {}).get("VPD")
    vpd_alert  = get_vpd_alert(chosen_vpd)
    if sel_model and sel_model in preds:
        return jsonify({"predictions": {sel_model: preds[sel_model]}, "vpd_alert": vpd_alert})
    return jsonify({"predictions": preds, "vpd_alert": vpd_alert})

@app.route("/api/export_csv")
def api_export_csv():
    year_start = int(request.args.get("year_start", 2026))
    year_end   = int(request.args.get("year_end",   2026))
    doy_start  = int(request.args.get("doy_start",  1))
    doy_end    = int(request.args.get("doy_end",    365))
    model      = request.args.get("model", "Hybrid")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Year","DOY","Date","VPD (kPa)","Humidity (%)","Model","VPD Stress Level","Alert Message"])

    for year in range(year_start, year_end + 1):
        for doy in range(doy_start, doy_end + 1):
            preds = predict_row(year, doy, 28.0)
            vpd   = preds.get(model, {}).get("VPD", "")
            hum   = preds.get(model, {}).get("Humidity", "")
            try:
                date_str = (datetime.date(year,1,1) + datetime.timedelta(days=doy-1)).strftime("%d-%m-%Y")
            except Exception:
                date_str = ""
            alert        = get_vpd_alert(vpd)
            stress_level = alert["label"]   if alert else ""
            stress_msg   = alert["message"] if alert else ""
            writer.writerow([year, doy, date_str, vpd, hum, model, stress_level, stress_msg])

    output.seek(0)
    filename = f"predictions_{model}_{year_start}-{year_end}_doy{doy_start}-{doy_end}.csv"
    return Response(output.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)