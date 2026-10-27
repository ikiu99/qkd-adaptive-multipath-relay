from qkdmp.config import load_config


def test_default_config_loads():
    config = load_config()
    assert config["key_size_bits"] == 256
    assert config["rate_model"]["mode"] == "length_dependent"


def test_override_merges_onto_default(tmp_path):
    override_path = tmp_path / "override.yaml"
    override_path.write_text("seed: 123\nrate_model:\n  mode: uniform\n")

    config = load_config(override_path)

    assert config["seed"] == 123
    assert config["rate_model"]["mode"] == "uniform"
    # untouched nested keys survive the merge
    assert config["rate_model"]["R0_bps"] == 1.0e5
    assert config["key_size_bits"] == 256
