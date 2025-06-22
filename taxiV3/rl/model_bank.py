import numpy as np
import json
import datetime
import torch
from pathlib import Path

MODELS_DIR = Path("saved_models")
MODELS_DIR.mkdir(exist_ok=True)

def _meta_path(name):
    return MODELS_DIR / f"{name}.json"

def _data_path(name, algo):
    return MODELS_DIR / f"{name}.{ 'npy' if algo == 'q_learning' else 'pt' }"

def save(name: str, payload: dict, hparams: dict):
    name = name.strip().replace(" ", "_") or f"model_{int(datetime.datetime.now().timestamp())}"
    algo = payload["algo"]
    data_path = _data_path(name, algo)
    if data_path.exists():
        return False, "Nom déjà pris."

    if algo == "q_learning":
        np.save(data_path, payload["q"])
    else:
        torch.save(payload["policy_state"], data_path)

    meta = {
        "algo": algo,
        **hparams,
        **payload["metrics"],
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    _meta_path(name).write_text(json.dumps(meta, indent=2))
    return True, name

def delete_model(name: str) -> bool:
    """
    Supprime les fichiers .json + .npy / .pt du modèle.
    Renvoie True si au moins un fichier a été supprimé.
    """
    removed = False
    for ext in ("json", "npy", "pt"):
        f = MODELS_DIR / f"{name}.{ext}"
        if f.exists():
            f.unlink()
            removed = True
    return removed

def load(name):
    """Charge data + meta. Compatible avec les anciens modèles sans clé 'algo'."""
    meta_path = _meta_path(name)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    meta = json.loads(meta_path.read_text())
    algo = meta.get("algo")

    if algo is None:
        if (_data_path(name, "q_learning")).exists():
            algo = "q_learning"
        elif (_data_path(name, "dqn")).exists():
            algo = "dqn"
        else:
            raise FileNotFoundError("Aucun fichier .npy ou .pt pour ce modèle")
        meta["algo"] = algo

    data_path = _data_path(name, algo)
    data = np.load(data_path) if algo == "q_learning" else torch.load(data_path)
    return algo, data, meta

def _data_path(name, algo):
    # Q-learning → .npy  /  DQN → .pt
    return MODELS_DIR / f"{name}.{ 'npy' if algo == 'q_learning' else 'pt' }"