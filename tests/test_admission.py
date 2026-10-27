import networkx as nx
import pandas as pd
import pytest

from analysis.metrics import key_efficiency, secure_goodput
from qkdmp.config import load_config
from qkdmp.policies.admission import AdmissionControlled
from qkdmp.policies.base import BLOCK, Request
from qkdmp.policies.baselines import B0Policy, B1Policy
from qkdmp.sim.graph_mc import get_policy


@pytest.fixture
def config():
    return load_config()


def _square(level_01: float = 300.0) -> nx.Graph:
    """0-1-3 (short, preferred by hop count via tie-break on insertion) and
    0-2-3; link 0-1 holds `level_01` bits."""
    graph = nx.Graph()
    for u, v in [(0, 1), (1, 3), (0, 2), (2, 3)]:
        graph.add_edge(u, v, length_km_norm=10.0, rate_bps=1e5, capacity_bits=300.0, level_bits=300.0)
    graph.edges[0, 1]["level_bits"] = level_01
    return graph


def test_raw_baseline_ignores_depleted_link(config):
    decision = B0Policy().decide(Request(0, 3, 0.5, 0.5), _square(100.0), {}, config)
    assert decision.paths == [[0, 1, 3]]  # pool-unaware: happily picks the link that can't carry a share


def test_ac_refuses_instead_of_sending_on_depleted_link(config):
    policy = AdmissionControlled(B0Policy(), "AC")
    assert policy.decide(Request(0, 3, 0.5, 0.5), _square(100.0), {}, config) is BLOCK


def test_pa_routes_around_depleted_link(config):
    policy = AdmissionControlled(B0Policy(), "PA")
    decision = policy.decide(Request(0, 3, 0.5, 0.5), _square(100.0), {}, config)
    assert decision is not BLOCK
    assert decision.paths == [[0, 2, 3]]


def test_pa_blocks_when_depletion_disconnects(config):
    graph = _square(100.0)
    graph.edges[0, 2]["level_bits"] = 100.0
    assert AdmissionControlled(B1Policy(), "PA").decide(Request(0, 3, 0.5, 0.5), graph, {}, config) is BLOCK


def test_wrappers_are_transparent_when_pools_are_full(config):
    graph = _square(300.0)
    raw = B1Policy().decide(Request(0, 3, 0.5, 0.5), graph, {}, config)
    for tier in ("AC", "PA"):
        wrapped = AdmissionControlled(B1Policy(), tier).decide(Request(0, 3, 0.5, 0.5), graph, {}, config)
        assert wrapped.paths == raw.paths and wrapped.t == raw.t


def test_get_policy_parses_tier_suffix(config):
    policy = get_policy("B2+PA", config)
    assert isinstance(policy, AdmissionControlled) and policy.name == "B2+PA"


def test_secure_goodput_counts_only_unexposed_correct_deliveries():
    df = pd.DataFrame([
        {"blocked": False, "served": True, "exposed": False, "silent_failure": False, "bits_consumed": 512, "ref_bits": 512},
        {"blocked": False, "served": True, "exposed": True, "silent_failure": False, "bits_consumed": 512, "ref_bits": 512},
        {"blocked": False, "served": True, "exposed": False, "silent_failure": True, "bits_consumed": 512, "ref_bits": 512},
        {"blocked": True, "bits_consumed": 0.0, "ref_bits": 512},
    ])
    assert secure_goodput(df) == pytest.approx(0.25)
    assert key_efficiency(df) == pytest.approx(512 / 1536)
