import json

from carry.state import CarryState, CoinLeg, load_state, save_state


def test_roundtrip(tmp_path):
    p = tmp_path / "s" / "state.json"
    st = CarryState(mode="HEDGED", legs={"BTC": CoinLeg(1.5, 0.015, 1000)})
    save_state(p, st)
    got = load_state(p)
    assert got.mode == "HEDGED"
    assert got.legs["BTC"] == CoinLeg(1.5, 0.015, 1000)


def test_missing_file_gives_default(tmp_path):
    st = load_state(tmp_path / "nope.json")
    assert st.mode == "FLAT" and st.legs == {}


def test_save_is_atomic_and_leaves_no_temp(tmp_path):
    p = tmp_path / "state.json"
    save_state(p, CarryState())
    save_state(p, CarryState(mode="HALTED", halted_reason="x"))
    assert json.loads(p.read_text())["halted_reason"] == "x"
    assert [f.name for f in tmp_path.iterdir()] == ["state.json"]


def test_from_dict_tolerates_missing_keys():
    st = CarryState.from_dict({})
    assert st.mode == "FLAT" and st.legs == {} and st.halted_reason == ""
