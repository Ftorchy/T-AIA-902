import streamlit as st
import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
import os, io, pickle
from pathlib import Path
from QLearning import train_tabular
from DeepQLearning import DeepQLearning

st.set_page_config(page_title="RL Project - TaxiV3", layout="wide")
st.title("Taxi-v3 – T-AIA Project - STG6")

# Choix algo
algo = st.sidebar.radio(
    "Choose algorithm :",
    ["Q-Learning", "Deep Q-Learning (DQN)"],
)
st.sidebar.markdown("---")


# Section Q-Learning
if algo == "Q-Learning":
    st.header("Tabular Q-Learning")

    # réglage hyperparamètres
    episodes  = st.sidebar.slider("Episodes", 100, 25_000, 5_000, 100)
    max_steps = st.sidebar.slider("Max steps / episode", 10, 500, 200, 1)
    alpha     = st.sidebar.number_input("Learning-rate (α)", 0.01, 1.0, 0.20, 0.01)
    gamma     = st.sidebar.number_input("Discount (γ)",     0.00, 0.999, 0.95, 0.01)
    eps0      = st.sidebar.number_input("Epsilon", 0.0, 1.0, 1.00, 0.05)
    eps_min   = st.sidebar.number_input("Epsilon min",     0.0, 1.0, 0.01, 0.01)
    decay     = st.sidebar.number_input("Epsilon decay",   0.0, 0.1, 0.0005, 0.0001,format="%.4f")
    colT, colL = st.columns(2)
    train_btn  = colT.button("Train")
    load_pkl   = colL.file_uploader("📂 Load model", type=["pkl"])

    # Train & SAVE
    if train_btn:
        prog = st.progress(0.0, text="Training …")
        stats = train_tabular("Taxi-v3", episodes, alpha, gamma,
                              eps0, eps_min, decay, max_steps, prog)
        prog.empty()
        st.session_state["tab_stats"] = stats
        st.success("Training complete")

        # Sauvegarde
        payload = {
            "algo":  "qlearning",
            "stats": stats,
            "hp":    {
                "episodes": episodes, "max_steps": max_steps,
                "alpha": alpha, "gamma": gamma,
                "eps0": eps0, "eps_min": eps_min, "decay": decay
            }
        }
        buf = io.BytesIO(); pickle.dump(payload, buf)
        st.download_button("Save model",
                           data=buf.getvalue(),
                           file_name="qlearning_taxi.pkl")

    # LOAD
    if load_pkl is not None:
        obj = pickle.load(load_pkl)
        if obj.get("algo") == "qlearning":
            st.session_state["tab_stats"] = obj["stats"]
            st.session_state["tab_hp"]    = obj.get("hp", {})
            st.success("Q-Learning model loaded")
        else:
            st.error("❌ This file is not a Q-Learning model")

    # DISPLAY
    if "tab_stats" in st.session_state:
        stats = st.session_state["tab_stats"]
        hp    = st.session_state.get("tab_hp", {})

        # Hyper-parameters utilisés pour l'entrainement
        st.subheader("Hyper-parameters")
        hp_table = pd.DataFrame({
            "Hyperparamètre": ["episodes","max_steps","alpha","gamma",
                               "eps0","eps_min","decay"],
            "Valeur": [hp.get("episodes", len(stats["rewards"])),
                       hp.get("max_steps", max(stats["steps"])),
                       f'{hp.get("alpha",alpha):.3f}',
                       f'{hp.get("gamma",gamma):.3f}',
                       f'{hp.get("eps0",eps0):.2f}',
                       f'{hp.get("eps_min",eps_min):.2f}',
                       f'{hp.get("decay",decay):.4f}'],
        }).set_index("Hyperparamètre")
        st.table(hp_table)

        # Curves
        st.subheader("Metrics")
        fig, ax = plt.subplots(2, 2, figsize=(11, 6))
        ax[0,0].plot(stats["rewards"], alpha=.3)
        ax[0,0].plot(stats["smooth_rewards"]); ax[0,0].set_title("Reward")
        ax[0,1].plot(stats["steps"], alpha=.3)
        ax[0,1].plot(stats["smooth_steps"]);  ax[0,1].set_title("Steps")
        ax[1,0].plot(stats["epsilons"]);      ax[1,0].set_title("ε")
        ax[1,1].plot(stats["cpu_times"]);     ax[1,1].set_title("CPU time/ep (s)")
        for a in ax.ravel(): a.grid(alpha=.3)
        plt.tight_layout(); st.pyplot(fig)

        # Metrics
        n_cpu       = os.cpu_count() or 1
        cpu_single  = stats["total_cpu"] / stats["total_wall"]
        cpu_global  = 100 * cpu_single / n_cpu
        total_steps = stats["steps"].sum()
        global_thr  = total_steps / stats["total_wall"]
        reward_mean = stats["rewards"].mean()
        steps_mean  = stats["steps"].mean()
        reward_last = stats["rewards"][-100:].mean()
        steps_last  = stats["steps"][-100:].mean()
        success_mean = stats.get("success", np.zeros_like(stats["rewards"])).mean()
        success_last = stats.get("success", np.zeros_like(stats["rewards"]))[-100:].mean()

        st.subheader("📊 Global metrics")
        st.markdown(f"""
        **Elapsed time:** {stats['total_wall']:.2f}s  
        **Throughput:** {global_thr:,.0f} steps/s  
        **Total steps:** {total_steps:,}  
        **CPU time:** {stats['total_cpu']:.2f}s
        **Avg. CPU load:** {cpu_global:.1f}%

        **Avg. reward (global):** {reward_mean:.2f}  
        **Avg. steps (global):** {steps_mean:.1f}  
        **Success rate (global):** {success_mean*100:.1f}%  

        **Avg. reward (last 100):** {reward_last:.2f}  
        **Avg. steps (last 100):** {steps_last:.1f}  
        **Success rate (last 100):** {success_last*100:.1f}%  
        """)


# Section Deep Q-Learning (DQN)
else:
    st.header("Deep Q-Learning (DQN)")

    # réglage des Hyper-parameters
    epochs = st.sidebar.slider("Epochs", 1_000, 50_000, 10_000, 1_000)
    batch_size = st.sidebar.number_input("Batch size", 16, 512, 64, 16)
    learning_rate = st.sidebar.number_input("Learning-rate", 1e-4, 1.0, 0.001, 0.0005,
                                            format="%.4f")
    gamma = st.sidebar.number_input("Discount γ", 0.0, 0.999, 0.95, 0.01)
    exploration = st.sidebar.number_input("Initial ε", 0.0, 1.0, 0.6, 0.05)
    target_freq = st.sidebar.slider("Target-net update freq (steps)", 10, 1_000, 100, 10)
    mem_size = st.sidebar.number_input("Replay memory size", 1_000, 50_000, 10_000, 1_000)

    colT, colS, colL = st.columns(3)
    train_clicked = colT.button("Train")
    save_clicked  = colS.button("Save model")
    load_pkl      = colL.file_uploader("Load model", type=["pkl"])

    # Train
    if train_clicked:
        with st.spinner("Training DQN …"):
            agent = DeepQLearning(
                learning_rate=learning_rate,
                gamma=gamma,
                exploration_prob=exploration,
                batch_size=batch_size,
                target_net_update_freq=target_freq,
                memory_size=mem_size,
            )
            agent.train(epochs=epochs)
            st.session_state["dqn_agent"] = agent
        st.success("Training complete")

    # Save
    if save_clicked:
        if "dqn_agent" in st.session_state:
            buf = io.BytesIO()
            pickle.dump({"algo": "dqn", "agent": st.session_state["dqn_agent"]}, buf)
            st.download_button("⬇️ Download DQN model",
                               data=buf.getvalue(),
                               file_name="dqn_taxi.pkl")
        else:
            st.warning("No DQN model to save")

    # Load
    if load_pkl is not None:
        obj = pickle.load(load_pkl)
        if obj.get("algo") == "dqn":
            st.session_state["dqn_agent"] = obj["agent"]
            st.success("DQN model loaded")
        else:
            st.error("❌ This file is not a DQN model")

    # Metrics & curves
    if "dqn_agent" in st.session_state:
        agent = st.session_state["dqn_agent"]
        if not agent.metrics["rewards"]:
            with st.spinner("Evaluating agent (1000 episodes)…"):
                agent.calculate_metrics()

        m = agent.get_metrics()
        st.subheader("Evaluation metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Successes", f"{m['success_rate']} / 1000",
                  f"{m['success_rate']/10:.1f}%")
        c2.metric("Avg. steps",   f"{np.mean(m['steps']):.1f}")
        c3.metric("Avg. reward",  f"{np.mean(m['rewards']):.2f}")

        fig, ax = plt.subplots()
        ax.plot(m["steps"])
        ax.set_title("Steps per episode (evaluation)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.grid(alpha=.3)
        st.pyplot(fig)