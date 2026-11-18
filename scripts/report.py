"""Build the interactive report from the JSON results, in French and in English.

    results/report.html     (French)
    results/report_en.html  (English)

Usage:
    python scripts/report.py results
"""
import datetime
import json
import os
import sys

import numpy as np

from marl_offload.env import EnvConfig

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
HERE = os.path.dirname(__file__)
LEARNED = ("IQL", "IPPO", "MAPPO")
RULES = ("greedy", "greedy+toll")

# Numbers from the first version of the project (before the review fixes), kept for the review log.
BEFORE = {"greedy": 1.573, "IQL": 1.451, "IPPO": 1.490, "MAPPO": 1.489}


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


def num(v, lang, d=3):
    s = f"{v:.{d}f}"
    return s.replace(".", ",") if lang == "fr" else s


def expl_summary(expl, lang):
    """One data-driven sentence on the exploitability results."""
    if not expl:
        return "Calcul en cours." if lang == "fr" else "Not computed yet."
    names = [n for n in RULES + LEARNED if n in expl]
    parts = " · ".join(f"{n} {num(expl[n]['nash_conv'], lang)}" for n in names)
    learned = [expl[n]["nash_conv"] for n in LEARNED if n in expl]
    rules = [expl[n]["nash_conv"] for n in RULES if n in expl]
    seeds = max((len(expl[n]["seeds"]) for n in LEARNED if n in expl), default=1)
    if lang == "fr":
        s = f"NashConv : {parts} (apprenants : moyenne sur {seeds} graine{'s' if seeds > 1 else ''})."
        if learned and rules and max(learned) < min(rules):
            s += " Les politiques apprises sont plus proches d’un équilibre que les deux règles."
        return s
    s = f"NashConv: {parts} (learners: mean over {seeds} seed{'s' if seeds > 1 else ''})."
    if learned and rules and max(learned) < min(rules):
        s += " The learned policies are closer to an equilibrium than both rules."
    return s


def findings(final, stack, expl, lang):
    F, su, sw = final, stack["summary"], stack["sweep"]
    i_soc = sw["price"].index(su["social_optimal_price"])
    soc0, soc_opt = sw["learned"][0]["social_cost_per_task"], sw["learned"][i_soc]["social_cost_per_task"]
    ci = {a: F[a]["cost_per_task_ci95"] for a in LEARNED}
    n_seeds = len(F["IQL"]["seeds"])
    best = min(F[a]["cost_per_task"] for a in LEARNED)
    f = lambda v, d=3: num(v, lang, d)
    ex = expl_summary(expl, lang)
    if lang == "fr":
        L = [
            ("fond", "majeur", "fixed", "Le signal « retardé » donnait la charge exacte du moment",
             "Le signal observé était la file avant service ; la file actuelle s’en déduit exactement (max(0, signal − 3)). Le greedy faisait d’ailleurs ce calcul. L’observabilité partielle annoncée n’existait pas, et la variable ajoutée au critique de MAPPO était redondante.",
             f"greedy {f(BEFORE['greedy'])} · IPPO ≈ MAPPO ({f(BEFORE['IPPO'])} / {f(BEFORE['MAPPO'])})",
             f"Nouveau paramètre <code>signal_delay</code> (défaut 2) : le signal est vraiment vieux de 2 pas. greedy {f(F['greedy']['cost_per_task'])} · IPPO {f(F['IPPO']['cost_per_task'])} · MAPPO {f(F['MAPPO']['cost_per_task'])}",
             "marl_offload/env.py · marl_offload/baselines.py · tests/test_env.py::test_load_signal_is_delayed"),
            ("fond", "majeur", "fixed", "Le tarif edge était compté comme un coût social",
             "Le README concluait que le prix optimal pour les appareils est 0, ce qui est vrai par construction : le tarif est un transfert vers l’opérateur, pas une perte pour le système.",
             "« Le prix optimal pour les appareils est 0. »",
             f"Nouvelle métrique <code>social_cost_per_task</code> (tarif edge exclu). Optimum social au prix {f(su['social_optimal_price'], 1)} : {f(soc0)} → {f(soc_opt)}. Le prix agit comme un péage de congestion.",
             "marl_offload/baselines.py · scripts/stackelberg.py · tests/test_env.py::test_social_cost_excludes_edge_price"),
            ("fond", "majeur", "fixed", "IQL bootstrappait sur des Q-valeurs jamais entraînées",
             "Les transitions sans tâche étaient ignorées, mais environ 20 % des états suivants o′ sont sans tâche : la cible utilisait une extrapolation arbitraire du réseau.",
             f"IQL {f(BEFORE['IQL'])} (3 graines, ancien environnement)",
             f"Toutes les transitions sont stockées. IQL {f(F['IQL']['cost_per_task'])} ± {f(ci['IQL'])} ({n_seeds} graines, signal retardé). Le chiffre n’est pas comparable directement : l’environnement a changé (remarque 1).",
             "marl_offload/iql.py · tests/test_agents.py::test_iql_runs_and_reloads"),
            ("fond", "majeur", "fixed" if expl else "todo", "Aucune mesure de distance à l’équilibre",
             "Le jeu est à somme générale, mais rien ne vérifiait qu’aucun appareil ne gagnerait à dévier.",
             "Pas de mesure.", f"Analyse d’exploitabilité par meilleure réponse PPO, appareil par appareil. {ex}",
             "marl_offload/exploitability.py · scripts/exploitability.py · onglet Équilibre"),
            ("fond", "majeur", "fixed" if expl else "todo", "La première mesure d’exploitabilité testait surtout l’optimiseur",
             "Pour MAPPO et IPPO, la meilleure réponse (200 mises à jour depuis zéro) faisait pire que la politique de départ pour presque tous les appareils, et les gains négatifs étaient ramenés à 0 : NashConv = 0 mesurait un optimiseur trop faible, pas un équilibre. Le calcul pour greedy+toll s’était aussi arrêté à 5 appareils sur 6.",
             "Meilleure réponse depuis zéro, 100 épisodes non appariés, gains négatifs effacés, 1 graine, greedy+toll incomplet.",
             "Départ depuis la politique actuelle, choix du meilleur point de contrôle sur la validation, avant et après sur les mêmes épisodes (300) avec IC 95 %, gains bruts conservés, signalement des non-convergences, 2 graines pour les apprenants, greedy+toll complet. Test : une meilleure réponse initialisée ainsi ne fait pas pire que la politique de départ.",
             "marl_offload/exploitability.py · tests/test_exploitability.py"),
            ("fond", "moyen", "fixed", "Trop peu de graines pour classer les méthodes",
             "L’écart IQL/PPO (0,04) reposait sur 3 graines, avec un écart-type d’environ 0,02.",
             "3 graines, aucun intervalle",
             f"{n_seeds} graines, IC 95 % de Student : IQL ± {f(ci['IQL'])} · IPPO ± {f(ci['IPPO'])} · MAPPO ± {f(ci['MAPPO'])}. Les intervalles se chevauchent : aucun classement net entre les trois apprenants.",
             "scripts/train.py"),
            ("modele", "moyen", "fixed", "La baseline greedy était trop faible",
             "Sans baseline réglée, un gain de quelques pour cent est difficile à interpréter.",
             "greedy seul",
             f"Ajout de <code>greedy+toll</code> : péage virtuel {f(F['greedy+toll']['toll'], 1)} réglé sur des graines séparées → {f(F['greedy+toll']['cost_per_task'])} (meilleur apprenant : {f(best)}).",
             "marl_offload/baselines.py · scripts/train.py"),
            ("modele", "moyen", "doc", "L’option cloud est presque toujours dominée",
             "À prix nul, le cloud n’est presque jamais choisi : le problème se réduit en pratique à deux actions.",
             "part cloud ≈ 0",
             f"Inchangé (part cloud MAPPO {f(F['MAPPO']['share_cloud'])}). Piste : baisser <code>cloud_delay</code> ou <code>cloud_fee</code>.",
             "marl_offload/env.py · EnvConfig"),
            ("modele", "mineur", "doc", "Le leader est statique et le bandit ne fait que confirmer le balayage",
             "UCB choisit parmi les 16 prix déjà évalués exhaustivement ; il confirme le balayage sans rien apprendre de plus.",
             f"UCB → {f(su['bandit_price'], 1)}",
             f"Inchangé, documenté. UCB → {f(su['bandit_price'], 1)}, balayage → {f(su['sweep_optimal_price'], 1)}. Piste : prix dynamique appris conjointement.",
             "scripts/stackelberg.py"),
            ("code", "moyen", "fixed", "PPO n’était pas reproductible",
             "Le mélange des minibatchs utilisait le générateur global de NumPy, sans graine.",
             "<code>np.random.shuffle(idx)</code>", "Générateur dédié et seedé ; test de reproductibilité bit à bit.",
             "marl_offload/ppo.py · tests/test_agents.py::test_ppo_is_reproducible"),
            ("code", "mineur", "fixed", "Épisodes d’évaluation vus à l’entraînement",
             "PPO graine 0, mise à jour 10, s’entraînait sur les graines 10000 à 10007, qui sont aussi des graines d’évaluation.",
             "graines entraînement et évaluation qui se chevauchent",
             "<code>TRAIN_SEED = 10⁸</code>, <code>EVAL_SEED = 10⁴</code>, <code>TUNE_SEED = 5·10⁴</code>, <code>VAL_SEED = 3·10⁴</code> : espaces disjoints.",
             "marl_offload/ppo.py · marl_offload/iql.py · marl_offload/baselines.py"),
            ("code", "mineur", "fixed", "Le prix « collait » d’un reset à l’autre",
             "reset(price=None) gardait le prix précédent au lieu de revenir à cfg.edge_price.",
             "prix de l’épisode précédent", "retour à <code>cfg.edge_price</code> ; test dédié.",
             "marl_offload/env.py · tests/test_env.py::test_price_resets_to_config_default"),
            ("code", "mineur", "fixed", "Figures non reproductibles et valeurs codées en dur",
             "La figure par appareil ne reposait que sur MAPPO graine 0 ; « × 6 » au lieu du nombre d’agents ; arguments par défaut différents du README.",
             "1 graine, constantes en dur",
             "Moyenne sur toutes les graines, IQL et greedy+toll ajoutés ; défauts alignés (400 mises à jour PPO, 1000 épisodes IQL, 5 graines).",
             "scripts/plots.py · scripts/train.py"),
            ("code", "mineur", "fixed", "Tests fragiles et aucun test des algorithmes",
             "Deux tests supposaient que tous les canaux étaient dans le même état ; IQL, PPO et l’exploitabilité n’étaient pas testés.",
             "5 tests", "14 tests : latences exactes de la file, signal retardé, prix, coût social, IQL, PPO, meilleure réponse.",
             "tests/"),
            ("code", "mineur", "fixed", "Pas de dépôt git ni de packaging",
             "Le dossier n’était pas versionné ; les scripts modifiaient sys.path ; Python ≥ 3.10 n’était déclaré nulle part.",
             "sys.path modifié dans chaque script",
             "<code>pyproject.toml</code> (Python ≥ 3.10), <code>pip install -e .</code>, plus aucun sys.path ; dépôt git.",
             "pyproject.toml · .gitignore"),
        ]
    else:
        L = [
            ("core", "major", "fixed", "The \"delayed\" signal gave the exact current load",
             "The observed signal was the queue before service; the current queue follows from it exactly (max(0, signal − 3)), and greedy did that computation. The claimed partial observability did not exist, and the extra input of the MAPPO critic was redundant.",
             f"greedy {f(BEFORE['greedy'])} · IPPO ≈ MAPPO ({f(BEFORE['IPPO'])} / {f(BEFORE['MAPPO'])})",
             f"New parameter <code>signal_delay</code> (default 2): the signal really is 2 steps old. greedy {f(F['greedy']['cost_per_task'])} · IPPO {f(F['IPPO']['cost_per_task'])} · MAPPO {f(F['MAPPO']['cost_per_task'])}",
             "marl_offload/env.py · marl_offload/baselines.py · tests/test_env.py::test_load_signal_is_delayed"),
            ("core", "major", "fixed", "The edge price was counted as a social cost",
             "The README concluded that the best price for the devices is 0, which is true by construction: the price is a transfer to the operator, not a loss for the system.",
             "\"The best price for the devices is 0.\"",
             f"New metric <code>social_cost_per_task</code> (edge price excluded). Social optimum at price {f(su['social_optimal_price'], 1)}: {f(soc0)} → {f(soc_opt)}. The price acts as a congestion toll.",
             "marl_offload/baselines.py · scripts/stackelberg.py · tests/test_env.py::test_social_cost_excludes_edge_price"),
            ("core", "major", "fixed", "IQL bootstrapped on Q-values that were never trained",
             "Transitions without a task were skipped, but about 20 % of the next states o′ have no task: the target used an arbitrary extrapolation of the network.",
             f"IQL {f(BEFORE['IQL'])} (3 seeds, old environment)",
             f"All transitions are stored. IQL {f(F['IQL']['cost_per_task'])} ± {f(ci['IQL'])} ({n_seeds} seeds, delayed signal). Not directly comparable: the environment changed (finding 1).",
             "marl_offload/iql.py · tests/test_agents.py::test_iql_runs_and_reloads"),
            ("core", "major", "fixed" if expl else "todo", "No measure of the distance to equilibrium",
             "The game is general-sum, but nothing checked that no device would gain by deviating.",
             "No measure.", f"Exploitability analysis with a PPO best response, device by device. {ex}",
             "marl_offload/exploitability.py · scripts/exploitability.py · Equilibrium tab"),
            ("core", "major", "fixed" if expl else "todo", "The first exploitability measure mostly tested the optimizer",
             "For MAPPO and IPPO, the best response (200 updates from scratch) did worse than the base policy for almost every device, and negative gains were clipped to 0: NashConv = 0 measured a weak optimizer, not an equilibrium. The greedy+toll run had also stopped after 5 of 6 devices.",
             "Best response from scratch, 100 unpaired episodes, negative gains erased, 1 seed, greedy+toll incomplete.",
             "Warm start from the current policy, best checkpoint selected on validation episodes, before and after on the same 300 episodes with 95 % CI, raw gains kept, non-convergence flagged, 2 seeds for the learners, greedy+toll complete. Test: a best response initialised this way does not do worse than the base policy.",
             "marl_offload/exploitability.py · tests/test_exploitability.py"),
            ("core", "medium", "fixed", "Too few seeds to rank the methods",
             "The IQL/PPO gap (0.04) rested on 3 seeds, with a standard deviation of about 0.02.",
             "3 seeds, no interval",
             f"{n_seeds} seeds, Student 95 % CI: IQL ± {f(ci['IQL'])} · IPPO ± {f(ci['IPPO'])} · MAPPO ± {f(ci['MAPPO'])}. The intervals overlap: no clear ranking of the three learners.",
             "scripts/train.py"),
            ("model", "medium", "fixed", "The greedy baseline was too weak",
             "Without a tuned baseline, a gain of a few percent is hard to interpret.",
             "greedy only",
             f"Added <code>greedy+toll</code>: virtual toll {f(F['greedy+toll']['toll'], 1)} tuned on separate seeds → {f(F['greedy+toll']['cost_per_task'])} (best learner: {f(best)}).",
             "marl_offload/baselines.py · scripts/train.py"),
            ("model", "medium", "doc", "The cloud option is almost always dominated",
             "At zero price the cloud is almost never chosen: in practice the problem has two actions.",
             "cloud share ≈ 0",
             f"Unchanged (MAPPO cloud share {f(F['MAPPO']['share_cloud'])}). Next step: lower <code>cloud_delay</code> or <code>cloud_fee</code>.",
             "marl_offload/env.py · EnvConfig"),
            ("model", "minor", "doc", "The leader is static and the bandit only confirms the sweep",
             "UCB picks among the 16 prices already evaluated exhaustively; it confirms the sweep and learns nothing more.",
             f"UCB → {f(su['bandit_price'], 1)}",
             f"Unchanged, documented. UCB → {f(su['bandit_price'], 1)}, sweep → {f(su['sweep_optimal_price'], 1)}. Next step: a dynamic price learned jointly.",
             "scripts/stackelberg.py"),
            ("code", "medium", "fixed", "PPO was not reproducible",
             "Minibatch shuffling used NumPy's global generator, without a seed.",
             "<code>np.random.shuffle(idx)</code>", "Dedicated seeded generator; bit-exact reproducibility test.",
             "marl_offload/ppo.py · tests/test_agents.py::test_ppo_is_reproducible"),
            ("code", "minor", "fixed", "Evaluation episodes seen during training",
             "PPO seed 0, update 10, trained on seeds 10000 to 10007, which are also evaluation seeds.",
             "overlapping training and evaluation seeds",
             "<code>TRAIN_SEED = 10⁸</code>, <code>EVAL_SEED = 10⁴</code>, <code>TUNE_SEED = 5·10⁴</code>, <code>VAL_SEED = 3·10⁴</code>: disjoint ranges.",
             "marl_offload/ppo.py · marl_offload/iql.py · marl_offload/baselines.py"),
            ("code", "minor", "fixed", "The price stuck from one reset to the next",
             "reset(price=None) kept the previous price instead of going back to cfg.edge_price.",
             "price of the previous episode", "back to <code>cfg.edge_price</code>; dedicated test.",
             "marl_offload/env.py · tests/test_env.py::test_price_resets_to_config_default"),
            ("code", "minor", "fixed", "Figures not reproducible and hard-coded values",
             "The per-device figure used MAPPO seed 0 only; \"× 6\" instead of the number of agents; default arguments differed from the README.",
             "1 seed, hard-coded constants",
             "Mean over all seeds, IQL and greedy+toll added; defaults aligned (400 PPO updates, 1000 IQL episodes, 5 seeds).",
             "scripts/plots.py · scripts/train.py"),
            ("code", "minor", "fixed", "Fragile tests and no test of the algorithms",
             "Two tests assumed that all channels were in the same state; IQL, PPO and exploitability were not tested.",
             "5 tests", "14 tests: exact queue latencies, delayed signal, price, social cost, IQL, PPO, best response.",
             "tests/"),
            ("code", "minor", "fixed", "No git repository and no packaging",
             "The folder was not versioned; scripts edited sys.path; Python ≥ 3.10 was not declared anywhere.",
             "sys.path edited in every script",
             "<code>pyproject.toml</code> (Python ≥ 3.10), <code>pip install -e .</code>, no more sys.path; git repository.",
             "pyproject.toml · .gitignore"),
        ]
    return [dict(n=k + 1, cat=c, sev=s, status=st, title=t, problem=p, before=b, after=a, files=fl)
            for k, (c, s, st, t, p, b, a, fl) in enumerate(L)]


def main():
    final = load("final.json")
    curves = load("curves.json")
    stack = load("stackelberg.json")
    expl = load("exploitability.json", {})
    for k in list(final):  # drop bulky tuning details from the page
        final[k].pop("toll_tuning", None)
    for v in expl.values():  # validation curves are not shown on the page
        for r in v["per_agent"]:
            r.pop("validation_curve", None)
    cfg = EnvConfig()
    for lang, tpl_name, out_name in (("fr", "report_template.html", "report.html"),
                                     ("en", "report_template_en.html", "report_en.html")):
        data = {
            "generated": datetime.date.today().isoformat(),
            "signal_delay": cfg.signal_delay,
            "speeds": list(cfg.local_speeds[: cfg.n_agents]),
            "final": final,
            "curves": {k: curve_summary(v, cfg.n_agents) for k, v in curves.items()},
            "per_agent": load("per_agent.json", {}),
            "stackelberg": stack,
            "exploitability": expl,
            "findings": findings(final, stack, expl, lang),
        }
        tpl = open(os.path.join(HERE, tpl_name), encoding="utf-8").read()
        out = os.path.join(RES, out_name)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(tpl.replace("__DATA__", json.dumps(data, default=float).replace("</", "<\\/")))
        print("wrote", out)


if __name__ == "__main__":
    main()
