"""Rerun E3-E8 on Abilene (second verified real topology) with the same
settings as their NSFNET runs, to check whether the headline findings hold
on a second real network. Not part of experiments/run.py's registry since
it is a validation pass, not a new experiment id.

Usage: PYTHONPATH=. uv run python scripts/run_abilene_validation.py [E3 E4 ...]
(no arguments: all of E3-E8, sequentially).
"""

from __future__ import annotations

import sys

import yaml

from experiments import e3, e4, e5, e6, e7, e8
from experiments.meta import write_meta
from qkdmp.config import load_config

TOPOLOGY = "abilene"


def _load(exp_id: str) -> dict:
    with open(f"configs/experiments/{exp_id}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cfg(exp_cfg: dict) -> dict:
    base = load_config()
    cfg = dict(base)
    if "policy_overrides" in exp_cfg:
        cfg["policy"] = dict(base["policy"], **exp_cfg["policy_overrides"])
    if "pool_overrides" in exp_cfg:
        cfg["pool"] = dict(base["pool"], **exp_cfg["pool_overrides"])
    return cfg


def run_e3() -> None:
    c = _load("E3")
    df = e3.run_e3(load_config(), rho_a_values=c["rho_A_values"], placements=c["placements"], methods=c["methods"],
                   n_seeds=c["n_seeds"], n_requests=c["n_requests_per_seed"], topology=TOPOLOGY)
    e3.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(e3.RESULTS_DIR / f"e3_security_vs_attacker_{TOPOLOGY}.csv", index=False)
    gate = e3.decision_gate_table(df)
    gate.to_csv(e3.RESULTS_DIR / f"e3_decision_gate_{TOPOLOGY}.csv", index=False)
    e3.plot_e3_exposure(df, e3.FIGURES_DIR / TOPOLOGY)
    write_meta(e3.RESULTS_DIR / TOPOLOGY, f"E3_{TOPOLOGY}", c, list(range(c["n_seeds"])))
    print(gate.to_string(index=False))


def run_e4() -> None:
    c = _load("E4")
    df = e4.run_e4(_cfg(c), rho_values=c["rho_values"], methods=c["methods"], n_seeds=c["n_seeds"],
                   n_requests=c["n_requests_per_seed"], topology=TOPOLOGY)
    e4.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(e4.RESULTS_DIR / f"e4_cost_blocking_vs_load_{TOPOLOGY}.csv", index=False)
    e4.plot_e4(df[~df["method"].str.contains("+", regex=False)], e4.FIGURES_DIR / TOPOLOGY)
    e4.plot_admission_families(df, "rho", "secure_goodput", "Secure goodput",
                               e4.FIGURES_DIR / TOPOLOGY / "e4_secure_goodput_admission.png")
    write_meta(e4.RESULTS_DIR / TOPOLOGY, f"E4_{TOPOLOGY}", c, list(range(c["n_seeds"])))
    print(df.to_string(index=False))


def run_e5() -> None:
    c = _load("E5")
    df = e5.run_e5(_cfg(c), rho_values=c["rho_values"], methods=c["methods"], hotspot_fraction=c["hotspot_fraction"],
                   hotspot_pairs=c["hotspot_pairs"], n_seeds=c["n_seeds"], n_requests=c["n_requests_per_seed"],
                   topology=TOPOLOGY)
    e5.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(e5.RESULTS_DIR / f"e5_hotspot_{TOPOLOGY}.csv", index=False)
    e4.plot_admission_families(df, "rho", "secure_goodput", "Secure goodput",
                               e5.FIGURES_DIR / TOPOLOGY / "e5_secure_goodput_admission.png")
    write_meta(e5.RESULTS_DIR / TOPOLOGY, f"E5_{TOPOLOGY}", c, list(range(c["n_seeds"])))
    print(df.to_string(index=False))


def run_e6() -> None:
    c = _load("E6")
    df = e6.run_e6(_cfg(c), drop_tamper_values=[tuple(v) for v in c["drop_tamper_values"]], methods=c["methods"],
                   rho=c["rho"], n_seeds=c["n_seeds"], n_requests=c["n_requests_per_seed"], topology=TOPOLOGY,
                   rho_a=c.get("rho_A"))
    e6.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(e6.RESULTS_DIR / f"e6_drop_tamper_{TOPOLOGY}.csv", index=False)
    e6.plot_e6(df, e6.FIGURES_DIR / f"e6_drop_tamper_{TOPOLOGY}.png")
    write_meta(e6.RESULTS_DIR / TOPOLOGY, f"E6_{TOPOLOGY}", c, list(range(c["n_seeds"])))
    print(df.to_string(index=False))


def run_e7() -> None:
    c = _load("E7")
    base = load_config()
    df = e7.run_e7(base, eps_target_values=c["eps_target_values"], a_target=c["a_target"], rho_a=c["rho_A"],
                   n_seeds=c["n_seeds"], n_requests=c["n_requests_per_seed"], topology=TOPOLOGY)
    e7.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(e7.RESULTS_DIR / f"e7_pareto_{TOPOLOGY}.csv", index=False)
    lb = c["level_b"]
    cfg_b = dict(base)
    cfg_b["pool"] = dict(base["pool"], **lb.get("pool_overrides", {}))
    df_b = e7.run_e7_level_b(cfg_b, eps_target_values=c["eps_target_values"], a_target=c["a_target"],
                             rho_a=c["rho_A"], rho=lb["rho"], n_seeds=c["n_seeds"],
                             n_requests=c["n_requests_per_seed"], topology=TOPOLOGY)
    df_b.to_csv(e7.RESULTS_DIR / f"e7_pareto_level_b_{TOPOLOGY}.csv", index=False)
    write_meta(e7.RESULTS_DIR / TOPOLOGY, f"E7_{TOPOLOGY}", c, list(range(c["n_seeds"])))
    print(df.to_string(index=False))
    print(df_b.to_string(index=False))


def run_e8() -> None:
    c = _load("E8")
    df = e8.run_e8(_cfg(c), rho=c["rho"], n_seeds=c["n_seeds"], n_requests=c["n_requests_per_seed"],
                   topology=TOPOLOGY)
    e8.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(e8.RESULTS_DIR / f"e8_ablation_{TOPOLOGY}.csv", index=False)
    e8.plot_e8(df, e8.FIGURES_DIR / f"e8_ablation_{TOPOLOGY}.png")
    write_meta(e8.RESULTS_DIR / TOPOLOGY, f"E8_{TOPOLOGY}", c, list(range(c["n_seeds"])))
    print(df.to_string(index=False))


RUNNERS = {"E3": run_e3, "E4": run_e4, "E5": run_e5, "E6": run_e6, "E7": run_e7, "E8": run_e8}


def main() -> None:
    for exp_id in sys.argv[1:] or list(RUNNERS):
        print(f"=== {exp_id} on {TOPOLOGY} ===")
        RUNNERS[exp_id]()


if __name__ == "__main__":
    main()
