import pytest
from simulator import NavShieldSimulator

def test_tunnel_200m():
    sim = NavShieldSimulator()
    passed = sim.run_scenario("TUNNEL_200M", 40.0, {"velocity": 11.1, "gnss_noise": 2.0})
    assert passed

def test_gnss_jump():
    sim = NavShieldSimulator()
    passed = sim.run_scenario("GNSS_JUMP", 60.0, {"velocity": 11.1, "gnss_noise": 2.0})
    assert passed

def test_long_stop():
    sim = NavShieldSimulator()
    passed = sim.run_scenario("LONG_STOP", 45.0, {"velocity": 0.0, "gnss_noise": 0.1})
    assert passed
