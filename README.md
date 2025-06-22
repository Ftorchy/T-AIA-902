# Taxi-v3 - STG-6

**Membres du groupe:**
- Hamdi NASSRI
- Jeffrey WINKLER
- Mathéo VITALI
- Florian TORCHY

**Interface web Streamlit** pour expérimenter l’apprentissage par renforcement sur l’environnement **Gym / Taxi-v3**.  

---

## Sommaire
- [Taxi-v3 - STG-6](#taxi-v3---stg-6)
  - [Sommaire](#sommaire)
  - [Aperçu](#aperçu)
  - [Installation](#installation)
  - [Lancement](#lancement)
  - [Fonctionnalités](#fonctionnalités)
  - [Rapport de datascience](#rapport-de-datascience)
  - [Crédits \& Licence](#crédits--licence)

---------------------------------------------------------------------------------

## Aperçu
![alt text](/taxiV3/Documentation/taxi.gif)


---

## Installation

git clone https://github.com/Ftorchy/T-AIA-902.git

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

---

## Lancement

streamlit run app.py
(L'application s'ouvre automatiquement dans le navigateur http://localhost:8501)

---

## Fonctionnalités

| Bloc                     | Description                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------ |
| **Sidebar — Q-Learning** | Réglage α, γ, ε, nombre d’épisodes. Bouton « Entraîner & sauvegarder ».              |
| **Liste de modèles**     | Liste déroulante des `.json` enregistrés ; bouton 🗑️ pour supprimer.                |
| **Fiche modèle**         | Hyper-paramètres, KPI (Reward/Steps/Succès) + graphiques Altair sur 100 runs.        |
| **Comparatif**           | Tableau trié *Reward ↓ , Steps ↑* avec hyper-paramètres et date.                     |

**La feature deep-q-learning sera ajoutée sous peu**

---

## Rapport de datascience
https://epitechfr-my.sharepoint.com/:w:/r/personal/florian_torchy_epitech_eu/Documents/Rapport%20Data%20Science.docx?d=w8a894ec3e03348208c1dbe0d808c9e35&csf=1&web=1&e=ToV1OZ

---

## Crédits & Licence

| Ressource               | Licence                                  |
| ----------------------- | ---------------------------------------- |
| **Gymnasium / Taxi-v3** | MIT                                      |
| **Streamlit**           | Apache 2                                 |
| **Altair**              | BSD-3                                    |
| **Code du projet**      | STG-6 © 2025 - Epitech Strasbourg        |
