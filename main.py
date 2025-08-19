#===================================================================#
#                     Importation des bibliothèques                 #
#===================================================================#

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import time
import io
import joblib

app = FastAPI(title="Billets API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#===================================================================#
#                     DECLARER DES VAR | DF                         #
#===================================================================#

colonnes = ['diagonal','height_left','height_right','margin_up','margin_low','length']
model_path = "detection_billet_lr_model.sav"
scaler_path = "standar_scaler.sav"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

#===================================================================#
#                     Endpoint                                      #
#===================================================================#

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)) -> Dict[str, Any]:

    # Chargement et lecture du csv
    try:
        df = pd.read_csv(file.file, sep=";")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire de CSV: {e}")
    
    # Définition des médiannes
    mediane_diagonal = df["diagonal"].median()
    mediane_height_left = df["height_left"].median()
    mediane_height_right = df["height_right"].median()
    mediane_margin_low = df["margin_low"].median()
    mediane_margin_up = df["margin_up"].median()
    mediane_length = df["length"].median()

    # Remplacement des valeurs manquantes par les médiannes
    df["diagonal"].fillna(mediane_diagonal, inplace=True)
    df["height_left"].fillna(mediane_height_left, inplace=True)
    df["height_right"].fillna(mediane_height_right, inplace=True)
    df["margin_low"].fillna(mediane_margin_low, inplace=True)
    df["margin_up"].fillna(mediane_margin_up, inplace=True)
    df["length"].fillna(mediane_length, inplace=True)

    # Vérification des colonnes du dataset

    col_manquante = [c for c in colonnes if c not in df.columns]
    col_en_trop = [c for c in df.columns if c not in colonnes]
    if col_manquante:
        raise HTTPException(
            status_code=400,
            detail=f"Colonnes manquantes: {col_manquante}. Colonnes attendues: {colonnes}"
        )
    
    if col_en_trop:
        raise HTTPException(
            status_code= 400,
            detail=f"Colonnes en trop: {col_en_trop}. Colonnes attendues: {colonnes}"
        )
    
    # Normalisations des données
    df_scaled = scaler.transform(df)

    # Prédiction
    try:
        pred = model.predict(df_scaled)
        proba = model.predict_proba(df_scaled)[:,1]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur prédiction: {e}"
        )
    
    # Réponse Json
    prediction: List[Dict[str, Any]] = []
    for idx, (p, c) in enumerate(zip(proba, pred)):
        prediction.append({
            "index" : int(idx),
            "proba_vrai": round(float(p), 4),
            "classe_predite": "vrai" if int(c) == 1 else "faux"

        })
    
    counts = {
        "Vrai": int(pred.sum()),
        "Faux": int(len(pred) - pred.sum())
    }

    return {
        "n": len(prediction),
        "counts": counts,
        "predictions": prediction
    }

@app.post("/predict-file-csv")
async def predict_file(file: UploadFile = File(...)) -> Dict[str, Any]:

    # Chargement et lecture du csv
    try:
        df = pd.read_csv(file.file, sep=";")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire de CSV: {e}")
    
    # Définition des médiannes
    mediane_diagonal = df["diagonal"].median()
    mediane_height_left = df["height_left"].median()
    mediane_height_right = df["height_right"].median()
    mediane_margin_low = df["margin_low"].median()
    mediane_margin_up = df["margin_up"].median()
    mediane_length = df["length"].median()

    # Remplacement des valeurs manquantes par les médiannes
    df["diagonal"].fillna(mediane_diagonal, inplace=True)
    df["height_left"].fillna(mediane_height_left, inplace=True)
    df["height_right"].fillna(mediane_height_right, inplace=True)
    df["margin_low"].fillna(mediane_margin_low, inplace=True)
    df["margin_up"].fillna(mediane_margin_up, inplace=True)
    df["length"].fillna(mediane_length, inplace=True)

    # Vérification des colonnes du dataset

    col_manquante = [c for c in colonnes if c not in df.columns]
    col_en_trop = [c for c in df.columns if c not in colonnes]
    if col_manquante:
        raise HTTPException(
            status_code=400,
            detail=f"Colonnes manquantes: {col_manquante}. Colonnes attendues: {colonnes}"
        )
    
    if col_en_trop:
        raise HTTPException(
            status_code= 400,
            detail=f"Colonnes en trop: {col_en_trop}. Colonnes attendues: {colonnes}"
        )
    
    # Normalisations des données
    df_scaled = scaler.transform(df)

    # Prédiction
    try:
        pred = model.predict(df_scaled)
        proba = model.predict_proba(df_scaled)[:,1]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur prédiction: {e}"
        )
    
    # Réponse Json
    prediction: List[Dict[str, Any]] = []
    for idx, (p, c) in enumerate(zip(proba, pred)):
        prediction.append({
            "index" : int(idx),
            "proba_vrai": round(float(p), 4),
            "classe_predite": "vrai" if int(c) == 1 else "faux"

        })
    
    out_df = df.copy()
    out_df["proba_vrai"] = [round(float(p), 4) for p in proba]
    out_df["classe_predite"] = ["vrai" if int(c) == 1 else "faux" for c in pred]

    buffer = io.StringIO()
    out_df.to_csv(buffer, sep=";", index=False)
    buffer.seek(0)

    filename = f"predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    return StreamingResponse (
        buffer,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachement; filename="{filename}"'
        }
    )