"""Build results/report.html, an interactive page with the results and the review log.

Usage:
    python scripts/report.py results
"""
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marl_offload.env import EnvConfig

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
HERE = os.path.dirname(__file__)

# The greedy+toll best-response run was stopped (low memory) after 5 of 6 devices.
PARTIAL_NOTE = ("Résultat partiel pour greedy+toll (5 appareils sur 6, calcul interrompu) : gains de 2 à 8 % "
                "par appareil, NashConv ≥ 0,41. Le péage est aussi efficace que l’apprentissage, mais pas stable : "
                "chaque appareil a intérêt à ne pas le respecter.")

# Numbers from the first version of the project (before the fixes), kept for the review log.
BEFORE = {"greedy": 1.573, "IQL": 1.451, "IPPO": 1.490, "MAPPO": 1.489, "IQL_seeds": 3}


def load(name, default=None):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return default
    with open(path) as fh:
        return json.load(fh)


def curve_summary(runs, n_agents):
    """Mean and std over seeds at each evaluation point, x in agent-steps."""
    xs = [h["env_steps"] * n_agents for h in runs[0]]
    y = np.array([[h["cost_per_task"] for h in r] for r in runs])
    return [[int(x), float(m), float(s)] for x, m, s in zip(xs, y.mean(0), y.std(0))]


def f3(v):
    return f"{v:.3f}".replace(".", ",")


def findings(final, stack, expl):
    F = final
    su = stack["summary"]
    sw = stack["sweep"]
    i_soc = sw["price"].index(su["social_optimal_price"])
    soc0, soc_opt = sw["learned"][0]["social_cost_per_task"], sw["learned"][i_soc]["social_cost_per_task"]
    ci = {a: F[a]["cost_per_task_ci95"] for a in ("IQL", "IPPO", "MAPPO")}
    n_seeds = len(F["IQL"]["seeds"])
    expl_txt = "Calcul en cours."
    if expl:
        expl_txt = ("NashConv : " + " · ".join(f"{k} {f3(v['nash_conv'])}" for k, v in expl.items())
                    + " · greedy+toll ≥ 0,41 (partiel, 5 appareils sur 6). Les politiques apprises sont proches d’un "
                    "équilibre ; le greedy et le péage ne le sont pas.")
    return [
        dict(n=1, cat="fond", sev="majeur", status="fixed",
             title="Le signal « retardé » donnait la charge exacte du moment",
             problem="Le signal observé était la file avant service ; la file actuelle s’en déduit exactement (max(0, signal − 3)). "
                     "Le greedy faisait d’ailleurs ce calcul. L’observabilité partielle annoncée n’existait pas, et la variable "
                     "ajoutée au critique de MAPPO était redondante.",
             before=f"greedy {f3(BEFORE['greedy'])} · IPPO ≈ MAPPO ({f3(BEFORE['IPPO'])} / {f3(BEFORE['MAPPO'])})",
             after=f"Nouveau paramètre <code>signal_delay</code> (défaut 2) : le signal est vraiment vieux de 2 pas. "
                   f"greedy {f3(F['greedy']['cost_per_task'])} · IPPO {f3(F['IPPO']['cost_per_task'])} · MAPPO {f3(F['MAPPO']['cost_per_task'])}",
             files="marl_offload/env.py · marl_offload/baselines.py · tests/test_env.py::test_load_signal_is_delayed"),
        dict(n=2, cat="fond", sev="majeur", status="fixed",
             title="Le tarif edge était compté comme un coût social",
             problem="Le README concluait que le prix optimal pour les appareils est 0, ce qui est vrai par construction : le tarif "
                     "est un transfert vers l’opérateur, pas une perte pour le système.",
             before="« Le prix optimal pour les appareils est 0. »",
             after=f"Nouvelle métrique <code>social_cost_per_task</code> (tarif edge exclu). Optimum social au prix "
                   f"{f3(su['social_optimal_price'])} : {f3(soc0)} → {f3(soc_opt)}. Le prix agit comme un péage de congestion.",
             files="marl_offload/baselines.py · scripts/stackelberg.py · tests/test_env.py::test_social_cost_excludes_edge_price"),
        dict(n=3, cat="fond", sev="majeur", status="fixed",
             title="IQL bootstrappait sur des Q-valeurs jamais entraînées",
             problem="Les transitions sans tâche étaient ignorées, mais environ 20 % des états suivants o₂ sont sans tâche : "
                     "la cible utilisait une extrapolation arbitraire du réseau.",
             before=f"IQL {f3(BEFORE['IQL'])} (3 graines, ancien environnement)",
             after=f"Toutes les transitions sont stockées. IQL {f3(F['IQL']['cost_per_task'])} ± {f3(ci['IQL'])} ({n_seeds} graines, "
                   f"signal retardé). Le chiffre n’est pas comparable directement : l’environnement a changé (remarque 1).",
             files="marl_offload/iql.py · tests/test_agents.py::test_iql_runs_and_reloads"),
        dict(n=4, cat="fond", sev="majeur", status="fixed" if expl else "todo",
             title="Aucune mesure de distance à l’équilibre",
             problem="Le jeu est à somme générale, mais rien ne vérifiait qu’aucun appareil ne gagnerait à dévier.",
             before="Pas de mesure.",
             after=f"Nouveau script d’exploitabilité (meilleure réponse PPO par appareil). {expl_txt}",
             files="scripts/exploitability.py · onglet Équilibre"),
        dict(n=5, cat="fond", sev="moyen", status="fixed",
             title="Trop peu de graines pour classer les méthodes",
             problem="L’écart IQL/PPO (0,04) reposait sur 3 graines, avec un écart-type d’environ 0,02.",
             before="3 graines, aucun intervalle",
             after=f"{n_seeds} graines, IC 95 % de Student : IQL ± {f3(ci['IQL'])} · IPPO ± {f3(ci['IPPO'])} · MAPPO ± {f3(ci['MAPPO'])}. "
                   "Les intervalles se chevauchent : aucun classement net entre les trois apprenants.",
             files="scripts/train.py"),
        dict(n=6, cat="modele", sev="moyen", status="fixed",
             title="La baseline greedy était trop faible",
             problem="Sans baseline réglée, un gain de quelques pour cent est difficile à interpréter.",
             before="greedy seul",
             after=f"Ajout de <code>greedy+toll</code> : péage virtuel {f3(F['greedy+toll']['toll'])} réglé sur des graines séparées "
                   f"→ {f3(F['greedy+toll']['cost_per_task'])}, mieux que tous les apprenants "
                   f"(meilleur : {f3(min(F[a]['cost_per_task'] for a in ('IQL', 'IPPO', 'MAPPO')))}).",
             files="marl_offload/baselines.py · scripts/train.py"),
        dict(n=7, cat="modele", sev="moyen", status="doc",
             title="L’option cloud est presque toujours dominée",
             problem="À prix nul, le cloud n’est presque jamais choisi : le problème se réduit en pratique à deux actions.",
             before="part cloud ≈ 0", after=f"Inchangé (part cloud MAPPO {F['MAPPO']['share_cloud']:.3f}). Piste : baisser "
                   "<code>cloud_delay</code> ou <code>cloud_fee</code> pour un régime où le cloud compte.",
             files="marl_offload/env.py · EnvConfig"),
        dict(n=8, cat="modele", sev="mineur", status="doc",
             title="Le leader est statique et le bandit redouble le balayage",
             problem="UCB choisit parmi les 16 prix déjà évalués exhaustivement ; il confirme le balayage sans rien apprendre de plus.",
             before=f"UCB → {su['bandit_price']}", after=f"Inchangé, documenté. UCB → {su['bandit_price']}, balayage → "
                   f"{su['sweep_optimal_price']}. Piste : prix dynamique appris conjointement.",
             files="scripts/stackelberg.py"),
        dict(n=9, cat="code", sev="moyen", status="fixed",
             title="PPO n’était pas reproductible",
             problem="Le mélange des minibatchs utilisait le générateur global de NumPy, sans graine.",
             before="<code>np.random.shuffle(idx)</code>", after="Générateur dédié et seedé ; test de reproductibilité bit à bit.",
             files="marl_offload/ppo.py · tests/test_agents.py::test_ppo_is_reproducible"),
        dict(n=10, cat="code", sev="mineur", status="fixed",
             title="Épisodes d’évaluation vus à l’entraînement",
             problem="PPO graine 0, mise à jour 10, s’entraînait sur les graines 10000 à 10007, qui sont aussi des graines d’évaluation.",
             before="graines entraînement et évaluation qui se chevauchent",
             after="<code>TRAIN_SEED = 10⁸</code>, <code>EVAL_SEED = 10⁴</code>, <code>TUNE_SEED = 5·10⁴</code> : trois espaces disjoints.",
             files="marl_offload/ppo.py · marl_offload/iql.py · marl_offload/baselines.py"),
        dict(n=11, cat="code", sev="mineur", status="fixed",
             title="Le prix « collait » d’un reset à l’autre",
             problem="reset(price=None) gardait le prix précédent au lieu de revenir à cfg.edge_price.",
             before="prix de l’épisode précédent", after="retour à <code>cfg.edge_price</code> ; test dédié.",
             files="marl_offload/env.py · tests/test_env.py::test_price_resets_to_config_default"),
        dict(n=12, cat="code", sev="mineur", status="fixed",
             title="Figures non reproductibles et valeurs codées en dur",
             problem="La figure par appareil ne reposait que sur MAPPO graine 0 ; « × 6 » au lieu du nombre d’agents ; "
                     "arguments par défaut différents du README.",
             before="1 graine, constantes en dur", after="Moyenne sur toutes les graines, IQL et greedy+toll ajoutés ; "
                   "défauts alignés (400 mises à jour PPO, 1000 épisodes IQL, 5 graines).",
             files="scripts/plots.py · scripts/train.py"),
        dict(n=13, cat="code", sev="mineur", status="fixed",
             title="Tests fragiles et aucun test des algorithmes",
             problem="Deux tests supposaient que tous les canaux étaient dans le même état ; IQL et PPO n’étaient pas testés.",
             before="5 tests", after="10 tests : latences exactes de la file, signal retardé, prix, coût social, IQL et PPO.",
             files="tests/test_env.py · tests/test_agents.py"),
        dict(n=14, cat="code", sev="mineur", status="todo",
             title="Pas de dépôt git ni de packaging",
             problem="Le dossier n’est pas versionné ; les scripts modifient sys.path ; Python ≥ 3.10 n’est déclaré nulle part.",
             before="—", after="À faire : <code>git init</code> et un <code>pyproject.toml</code>.",
             files="racine du projet"),
    ]


def main():
    final = load("final.json")
    curves = load("curves.json")
    stack = load("stackelberg.json")
    expl = load("exploitability.json", {})
    cfg = EnvConfig()
    data = {
        "generated": datetime.date.today().isoformat(),
        "signal_delay": cfg.signal_delay,
        "speeds": list(cfg.local_speeds[: cfg.n_agents]),
        "final": final,
        "curves": {k: curve_summary(v, cfg.n_agents) for k, v in curves.items()},
        "per_agent": load("per_agent.json", {}),
        "stackelberg": stack,
        "exploitability": expl,
        "exploitability_seed": 0,
        "exploitability_note": PARTIAL_NOTE,
        "findings": findings(final, stack, expl),
    }
    for k in list(final):  # drop bulky tuning details from the page
        final[k].pop("toll_tuning", None)
    tpl = open(os.path.join(HERE, "report_template.html"), encoding="utf-8").read()
    out = os.path.join(RES, "report.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(tpl.replace("__DATA__", json.dumps(data, default=float).replace("</", "<\\/")))
    print("wrote", out)


if __name__ == "__main__":
    main()
