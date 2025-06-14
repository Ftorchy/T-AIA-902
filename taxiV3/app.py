import numpy as np
import streamlit as st
import pandas as pd
import datetime
import altair as alt
import gymnasium as gym
from pathlib import Path
from rl.trainers import q_learning as ql
# Pour plus tard import : from rl.trainers import dqn
from rl import model_bank as bank

MODELS_DIR = bank.MODELS_DIR
# Mise en page globale
st.set_page_config(page_title="Taxi RL Playground", layout="wide")

# Deux onglets dans la barre latérale
q_tab, d_tab = st.sidebar.tabs(["Q-Learning", "Deep Q-Learning"])

# Contenu onglet Q-Learning (sidebar)
with q_tab:
    st.subheader("Paramètres Q‑Learning")
    hp = dict(
        episodes  = st.number_input("Episodes", 1000, 50000, 5000, 500, key="ql_ep"),
        alpha     = st.slider("α", 0.0, 1.0, 0.7, 0.05, key="ql_alpha"),
        gamma     = st.slider("γ", 0.0, 0.99, 0.95, 0.01, key="ql_gamma"),
        eps0      = st.slider("ε₀", 0.0, 1.0, 1.0, 0.05, key="ql_eps0"),
        eps_min   = st.slider("ε min", 0.0, 0.5, 0.05, 0.01, key="ql_epsmin"),
        eps_decay = st.number_input("ε decay", 0.0001, 0.01, 0.0005, 0.0001,
                                    format="%f", key="ql_epsdec"),
    )
    q_name = st.text_input("Nom modèle", placeholder="ql_best", key="ql_name")
    if st.button("Entraîner & sauvegarder", key="ql_train"):
        res = ql.train(ql.QLParams(**hp))
        ok, saved_name = bank.save(q_name, res, hp)
        st.success(f"Modèle enregistré sous « {saved_name} »") if ok else st.error(saved_name)
        st.rerun()

# Partie modèles sauvegardés
    st.divider(); st.subheader("Modèles sauvegardés")
    model_files = sorted(p.stem for p in MODELS_DIR.glob("*.json"))
    chosen = st.selectbox("Choisir un modèle", ["--"] + model_files, index=0, key="sel_model")
    if chosen != "--" and st.button("🗑️ Supprimer", key="del_model"):
        bank.delete_model(chosen); st.rerun()

# Contenu onglet Deep-Q-Learning
with d_tab:
    st.info("Deep Q‑Learning — à venir")
    
    
# Contenu page principal (si Deep-Q-Learning selectionné)
st.title("Taxi-v3 – Modèles enregistrés")

if 'sel_model' in st.session_state and st.session_state.sel_model != "--":
    model_name = st.session_state.sel_model
    algo, data, meta = bank.load(model_name)
    st.subheader(f"Détails du modèle : {model_name}")

    with st.expander("Hyper‑paramètres & méta"):
        st.json(meta)

    def evaluate_qtable(q, n_eval=100, max_steps=200):
        env = gym.make("Taxi-v3"); rows=[]
        for _ in range(n_eval):
            s,_ = env.reset(); tot_r=n=0; succ=0
            for _ in range(max_steps):
                a=int(np.argmax(q[s])); n+=1
                s,r,term,trunc,_ = env.step(a); tot_r+=r
                if term and r==20: succ=1
                if term or trunc: break
            rows.append((tot_r,n,succ))
        env.close()
        df=pd.DataFrame(rows, columns=["reward","steps","success"])
        return df

    if algo == "q_learning":
        eva = evaluate_qtable(data)
        k1,k2,k3 = st.columns(3)
        k1.metric("Reward moyen", f"{eva.reward.mean():.2f}")
        k2.metric("Succès %", f"{100*eva.success.mean():.1f}%")
        k3.metric("Steps moyen", f"{eva.steps.mean():.1f}")

        c1,c2 = st.columns(2)
        line_data = eva.reset_index().rename(columns={"index":"episode"})
        c1.altair_chart(
            alt.Chart(line_data).mark_line().encode(
                x='episode', y='reward', tooltip=['episode','reward']
            ).properties(height=250, title="Reward par épisode (éval×100)"),
            use_container_width=True)

        hist_data = eva.steps.value_counts().sort_index().reset_index()
        hist_data.columns=['steps','count']
        c2.altair_chart(
            alt.Chart(hist_data).mark_bar().encode(
                x=alt.X('steps:O', title='Steps'),
                y=alt.Y('count:Q', title='Occurrences'),
                tooltip=['steps','count']
            ).properties(height=250, title="Distribution des steps"),
            use_container_width=True)
    else:
        st.info("Évaluation rapide non implémentée pour ce type de modèle.")

st.divider(); st.subheader("Classement des modèles entrainés")
rows=[]
for meta_path in MODELS_DIR.glob("*.json"):
    name=meta_path.stem; _,_,meta=bank.load(name)
    rows.append({
        "Modèle":name,"Algo":meta["algo"],"Reward":meta.get("reward_mean",0),
        "Succès %":meta.get("success_pct",0),"Steps":meta.get("steps_mean",0),
        "Episodes":meta.get("episodes",meta.get("EPIS","")),
        "α":meta.get("alpha",meta.get("ALPHA","")),
        "γ":meta.get("gamma",meta.get("GAMMA","")),
        "ε₀":meta.get("eps0",meta.get("EPS0","")),
        "ε min":meta.get("eps_min",meta.get("EPS_MIN","")),
        "ε decay":meta.get("eps_decay",meta.get("EPS_DECAY","")),
        "Date": (
            lambda d: datetime.datetime.fromisoformat(d).strftime("%d/%m/%Y %H:%M")
            if d else ""
            )(meta.get("created", ""))
    })
if rows:
    comp=pd.DataFrame(rows).sort_values(by=["Reward","Steps"],ascending=[False,True])
    st.dataframe(comp,use_container_width=True,height=380)
else:
    st.info("Aucun modèle enregistré.")
