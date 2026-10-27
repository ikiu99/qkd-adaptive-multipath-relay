"""Experiment CLI (README Section 9/10): `python -m experiments.run --config configs/experiments/E4.yaml`."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments import e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12
from experiments.meta import write_meta
from qkdmp.config import load_config


def run_e2(exp_config: dict) -> None:
    base_config = load_config()
    results_dir = e2.RESULTS_DIR
    figures_dir = e2.FIGURES_DIR

    part_a = exp_config["part_a"]
    df_a = e2.run_e2a(
        base_config,
        rho_a_values=part_a["rho_A_values"],
        n_seeds=part_a["n_seeds"],
        n_requests=part_a["n_requests_per_seed"],
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    df_a.to_csv(results_dir / "e2a_b1_vs_analytic.csv", index=False)
    e2.plot_e2a(df_a, figures_dir / "e2a_b1_vs_analytic.png")

    part_b = exp_config["part_b"]
    df_b = e2.run_e2b(
        base_config,
        rho_a_values=part_b["rho_A_values"],
        n_values=part_b["n_values"],
        n_seeds=part_b["n_seeds"],
        n_requests=part_b["n_requests_per_seed"],
    )
    df_b.to_csv(results_dir / "e2b_b3_grid.csv", index=False)
    e2.plot_e2b(df_b, figures_dir / "e2b_b3_grid.png")
    monotonic = e2.check_e2b_monotonic(df_b)

    all_seeds = list(range(max(part_a["n_seeds"], part_b["n_seeds"])))
    write_meta(results_dir, "E2", exp_config, all_seeds)

    print("E2(a) B1 vs analytic bound:")
    print(df_a.to_string(index=False))
    print("\nE2(b) B3 on grid, monotonic-in-rho_A check per n:", monotonic)
    print(df_b.to_string(index=False))


def run_e3(exp_config: dict) -> None:
    base_config = load_config()
    results_dir = e3.RESULTS_DIR
    figures_dir = e3.FIGURES_DIR

    df = e3.run_e3(
        base_config,
        rho_a_values=exp_config["rho_A_values"],
        placements=exp_config["placements"],
        methods=exp_config["methods"],
        n_seeds=exp_config["n_seeds"],
        n_requests=exp_config["n_requests_per_seed"],
        topology=exp_config["topology"],
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "e3_security_vs_attacker.csv", index=False)
    e3.plot_e3_exposure(df, figures_dir)
    e3.plot_e3_cost(df, figures_dir)

    gate = e3.decision_gate_table(df)
    gate.to_csv(results_dir / "e3_decision_gate.csv", index=False)

    write_meta(results_dir, "E3", exp_config, list(range(exp_config["n_seeds"])))

    print("E3 decision gate (Adaptive vs baselines, averaged over rho_A > 0):")
    print(gate.to_string(index=False))
    print(f"\nFull results: {results_dir / 'e3_security_vs_attacker.csv'}")


def run_e10(exp_config: dict) -> None:
    base_config = load_config()
    results_dir = e10.RESULTS_DIR
    figures_dir = e10.FIGURES_DIR

    results_dir.mkdir(parents=True, exist_ok=True)
    for topology in exp_config["topologies"]:
        df = e10.run_e10(
            base_config,
            detector_quality_values=exp_config["detector_quality_values"],
            methods=exp_config["methods"],
            rho_a=exp_config["rho_A"],
            n_seeds=exp_config["n_seeds"],
            n_requests=exp_config["n_requests_per_seed"],
            topology=topology,
        )
        df.to_csv(results_dir / f"e10_detector_quality_{topology}.csv", index=False)
        e10.plot_e10(df, figures_dir / f"e10_detector_quality_{topology}.png", title=f"({topology})")
        print(f"E10 detector quality sweep ({topology}):")
        print(df.to_string(index=False))

    write_meta(results_dir, "E10", exp_config, list(range(exp_config["n_seeds"])))


def run_e9(exp_config: dict) -> None:
    base_config = load_config()
    cfg = dict(base_config)
    cfg["policy"] = dict(base_config["policy"], **exp_config["policy_overrides"])
    cfg["pool"] = dict(base_config["pool"], **exp_config["pool_overrides"])

    results_dir = e9.RESULTS_DIR
    figures_dir = e9.FIGURES_DIR
    df, training_curves = e9.run_e9(
        cfg,
        topologies=exp_config["topologies"],
        n_requests_per_batch=exp_config["n_requests_per_batch"],
        rl_timesteps=exp_config["rl_timesteps"],
        seed=exp_config["seed"],
        rl_ent_coef=exp_config.get("rl_ent_coef", 0.0),
        rl_design=e9.LEGACY_DESIGN if exp_config.get("rl_design") == "legacy" else e9.RLEnvDesign(),
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "e9_optimality_scalability.csv", index=False)
    e9.plot_training_curves(training_curves, figures_dir / "e9_rl_training_curve.png")

    write_meta(results_dir, "E9", exp_config, [exp_config["seed"]])

    print("E9 optimality & scalability:")
    print(df.to_string(index=False))


def run_e4(exp_config: dict) -> None:
    base_config = load_config()
    cfg = dict(base_config)
    cfg["policy"] = dict(base_config["policy"], **exp_config["policy_overrides"])
    if "pool_overrides" in exp_config:
        cfg["pool"] = dict(base_config["pool"], **exp_config["pool_overrides"])

    results_dir, figures_dir = e4.RESULTS_DIR, e4.FIGURES_DIR
    df = e4.run_e4(cfg, rho_values=exp_config["rho_values"], methods=exp_config["methods"],
                   n_seeds=exp_config["n_seeds"], n_requests=exp_config["n_requests_per_seed"],
                   topology=exp_config["topology"])
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "e4_cost_blocking_vs_load.csv", index=False)
    e4.plot_e4(df[~df["method"].str.contains("+", regex=False)], figures_dir)
    e4.plot_admission_families(df, "rho", "secure_goodput", "Secure goodput",
                               figures_dir / "e4_secure_goodput_admission.png")
    write_meta(results_dir, "E4", exp_config, list(range(exp_config["n_seeds"])))
    print("E4 cost & blocking vs load:")
    print(df.to_string(index=False))


def run_e5(exp_config: dict) -> None:
    base_config = load_config()
    cfg = dict(base_config)
    cfg["policy"] = dict(base_config["policy"], **exp_config["policy_overrides"])
    if "pool_overrides" in exp_config:
        cfg["pool"] = dict(base_config["pool"], **exp_config["pool_overrides"])

    results_dir, figures_dir = e5.RESULTS_DIR, e5.FIGURES_DIR
    df = e5.run_e5(cfg, rho_values=exp_config["rho_values"], methods=exp_config["methods"],
                   hotspot_fraction=exp_config["hotspot_fraction"], hotspot_pairs=exp_config["hotspot_pairs"],
                   n_seeds=exp_config["n_seeds"], n_requests=exp_config["n_requests_per_seed"],
                   topology=exp_config["topology"])
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "e5_hotspot.csv", index=False)
    e5.plot_e5(df[~df["method"].str.contains("+", regex=False)], figures_dir / "e5_hotspot_blocking.png")
    e4.plot_admission_families(df, "rho", "secure_goodput", "Secure goodput",
                               figures_dir / "e5_secure_goodput_admission.png")
    write_meta(results_dir, "E5", exp_config, list(range(exp_config["n_seeds"])))
    print("E5 hotspot traffic:")
    print(df.to_string(index=False))


def run_e6(exp_config: dict) -> None:
    base_config = load_config()
    cfg = dict(base_config)
    cfg["policy"] = dict(base_config["policy"], **exp_config["policy_overrides"])

    results_dir, figures_dir = e6.RESULTS_DIR, e6.FIGURES_DIR
    drop_tamper_values = [tuple(v) for v in exp_config["drop_tamper_values"]]
    df = e6.run_e6(cfg, drop_tamper_values=drop_tamper_values, methods=exp_config["methods"],
                   rho=exp_config["rho"], n_seeds=exp_config["n_seeds"],
                   n_requests=exp_config["n_requests_per_seed"], topology=exp_config["topology"],
                   rho_a=exp_config.get("rho_A"))
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "e6_drop_tamper.csv", index=False)
    e6.plot_e6(df, figures_dir / "e6_drop_tamper.png")
    write_meta(results_dir, "E6", exp_config, list(range(exp_config["n_seeds"])))
    print("E6 drop/tamper sweep:")
    print(df.to_string(index=False))


def run_e7(exp_config: dict) -> None:
    base_config = load_config()
    results_dir, figures_dir = e7.RESULTS_DIR, e7.FIGURES_DIR
    df = e7.run_e7(base_config, eps_target_values=exp_config["eps_target_values"],
                   a_target=exp_config["a_target"], rho_a=exp_config["rho_A"],
                   n_seeds=exp_config["n_seeds"], n_requests=exp_config["n_requests_per_seed"],
                   topology=exp_config["topology"])
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "e7_pareto.csv", index=False)
    e7.plot_e7(df, figures_dir / "e7_pareto_front.png", level_label="Level A")
    print("E7 Pareto front (Level A):")
    print(df.to_string(index=False))

    if "level_b" in exp_config:
        lb = exp_config["level_b"]
        cfg_b = dict(base_config)
        cfg_b["pool"] = dict(base_config["pool"], **lb.get("pool_overrides", {}))
        df_b = e7.run_e7_level_b(
            cfg_b, eps_target_values=exp_config["eps_target_values"], a_target=exp_config["a_target"],
            rho_a=exp_config["rho_A"], rho=lb["rho"], n_seeds=exp_config["n_seeds"],
            n_requests=exp_config["n_requests_per_seed"], topology=exp_config["topology"],
        )
        df_b.to_csv(results_dir / "e7_pareto_level_b.csv", index=False)
        e7.plot_e7(df_b, figures_dir / "e7_pareto_front_level_b.png", level_label="Level B")
        print("E7 Pareto front (Level B):")
        print(df_b.to_string(index=False))

    write_meta(results_dir, "E7", exp_config, list(range(exp_config["n_seeds"])))


def run_e8(exp_config: dict) -> None:
    base_config = load_config()
    cfg = dict(base_config)
    cfg["policy"] = dict(base_config["policy"], **exp_config["policy_overrides"])
    if "pool_overrides" in exp_config:
        cfg["pool"] = dict(base_config["pool"], **exp_config["pool_overrides"])

    results_dir, figures_dir = e8.RESULTS_DIR, e8.FIGURES_DIR
    df = e8.run_e8(cfg, rho=exp_config["rho"], n_seeds=exp_config["n_seeds"],
                   n_requests=exp_config["n_requests_per_seed"], topology=exp_config["topology"])
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "e8_ablation.csv", index=False)
    e8.plot_e8(df, figures_dir / "e8_ablation.png")
    write_meta(results_dir, "E8", exp_config, list(range(exp_config["n_seeds"])))
    print("E8 ablation:")
    print(df.to_string(index=False))


def run_e11(exp_config: dict) -> None:
    base_config = load_config()
    cfg = dict(base_config)
    cfg["policy"] = dict(base_config["policy"], **exp_config["policy_overrides"])
    cfg["pool"] = dict(base_config["pool"], **exp_config["pool_overrides"])

    results_dir, figures_dir = e11.RESULTS_DIR, e11.FIGURES_DIR
    overall, windows = e11.run_e11(
        cfg, topologies=exp_config["topologies"], placements=exp_config["placements"],
        activity_values=exp_config["activity_values"], rho=exp_config["rho"], rho_a=exp_config["rho_A"],
        n_seeds=exp_config["n_seeds"], n_requests=exp_config["n_requests_per_seed"],
        n_windows=exp_config["n_windows"], learning=exp_config["learning"],
        no_info_prior=exp_config["no_info_prior"], detector_qualities=exp_config["detector_qualities"],
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(results_dir / "e11_learned_suspicion.csv", index=False)
    windows.to_csv(results_dir / "e11_learning_curves.csv", index=False)
    e11.plot_e11_curves(windows, figures_dir)
    write_meta(results_dir, "E11", exp_config, list(range(exp_config["n_seeds"])))


def run_e12(exp_config: dict) -> None:
    """Tuning/test protocol: sweep Adaptive's eps_target on tuning seeds,
    pick one operating point by a rule fixed in advance (`e12.choose_eps`),
    then evaluate everything on held-out seeds."""
    import json

    base_config = load_config()
    cfg = dict(base_config)
    cfg["policy"] = dict(base_config["policy"], **exp_config["policy_overrides"])
    cfg["pool"] = dict(base_config["pool"], **exp_config["pool_overrides"])
    results_dir, figures_dir = e12.RESULTS_DIR, e12.FIGURES_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    methods = exp_config["baselines"] + [f"Adaptive@{e}" for e in exp_config["eps_values"]]
    common = dict(topologies=exp_config["topologies"], rho=exp_config["rho"],
                  rho_a_values=exp_config["rho_A_values"], placements=exp_config["placements"],
                  methods=methods, n_requests=exp_config["n_requests_per_seed"])

    tuning = e12.run_e12(cfg, n_seeds=exp_config["tuning"]["n_seeds"],
                         seed_offset=exp_config["tuning"]["seed_offset"], **common)
    tuning.to_csv(results_dir / "e12_tuning.csv", index=False)
    eps = e12.choose_eps(tuning)
    chosen = f"Adaptive@{eps}"
    (results_dir / "e12_chosen_eps.json").write_text(json.dumps({"eps_target": eps}), encoding="utf-8")
    print(f"chosen on tuning seeds: {chosen}", flush=True)

    test = e12.run_e12(cfg, n_seeds=exp_config["test"]["n_seeds"],
                       seed_offset=exp_config["test"]["seed_offset"], **common)
    test.to_csv(results_dir / "e12_test.csv", index=False)
    dom = e12.dominance_table(test, chosen)
    dom.to_csv(results_dir / "e12_dominance.csv", index=False)
    e12.plot_e12(test[test["method"].isin(exp_config["baselines"] + [chosen])], figures_dir)
    write_meta(results_dir, "E12", exp_config, list(range(exp_config["test"]["seed_offset"],
                                                          exp_config["test"]["seed_offset"] + exp_config["test"]["n_seeds"])))
    print(f"{chosen}: dominates {int(dom['dominates'].sum())}/{len(dom)}, "
          f"significant goodput lead {int(dom['goodput_lead_significant'].sum())}, "
          f"significantly worse {int(dom['goodput_significantly_worse'].sum())}, "
          f"some operating point dominates {int(dom['some_operating_point_dominates'].sum())}/{len(dom)}")


EXPERIMENTS = {
    "E2": run_e2, "E3": run_e3, "E4": run_e4, "E5": run_e5, "E6": run_e6,
    "E7": run_e7, "E8": run_e8, "E9": run_e9, "E10": run_e10, "E11": run_e11, "E12": run_e12,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    import yaml

    with args.config.open("r", encoding="utf-8") as f:
        exp_config = yaml.safe_load(f)

    exp_id = exp_config["exp_id"]
    if exp_id not in EXPERIMENTS:
        raise NotImplementedError(f"experiment {exp_id!r} is not implemented yet")
    EXPERIMENTS[exp_id](exp_config)


if __name__ == "__main__":
    main()
