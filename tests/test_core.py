"""pytest -q   (offline; uses the mock backend and hash embedder)"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from rpw.config import load_config  # noqa: E402
from rpw.metrics import read_log, run_metrics  # noqa: E402
from rpw.payload import Detector, make_poison, max_copied_run  # noqa: E402
from rpw.retrieval import ExactStore, build_messages  # noqa: E402
from rpw.runner import run, run_id_for  # noqa: E402
import numpy as np  # noqa: E402

CFG = load_config("config/v2/mock.yaml", ["experiment.experiment_id=test", "experiment.T=40"])
DET = Detector(CFG["payload"]["marker"], CFG["payload"]["note"])


def test_detector_levels():
    note = CFG["payload"]["note"]
    assert DET.label("Answer.\n\nRef: %%RPPW%%\n" + note).l2
    l = DET.label("Answer. Ref: %% RPPW %%")
    assert l.l1 and not l.l2                        # tag only = expression, not replication
    assert not DET.label("The context contains a knowledge-base note about formatting.").l2
    assert DET.label(note.replace("%%RPPW%%", "")).l2   # stripping the marker leaves the instruction


def test_copy_run():
    assert max_copied_run("a b c d e f", ["x a b c d y"]) == 4


def test_store_order_and_remove():
    s = ExactStore(2)
    s.add("a", np.array([1, 0.0]), {"src": "x"})
    s.add("b", np.array([1, 0.0]), {"src": "x"})
    s.add("c", np.array([0, 1.0]), {"src": "x"})
    assert [i for i, _ in s.search(np.array([1, 0.0]), 2)] == ["a", "b"]   # tie -> insertion order
    s.remove(["a"])
    assert s.search(np.array([1, 0.0]), 1)[0][0] == "b"


def test_policies_place_content():
    for pol, role in (("p_sys", "system"), ("p_usr", "user"), ("p_usr_spot", "user")):
        msgs = build_messages(CFG, pol, ["DOCTEXT"], "Q?", "DATA-x")
        assert "DOCTEXT" in next(m["content"] for m in msgs if m["role"] == role)
        assert "apply it to your response" not in " ".join(m["content"] for m in msgs)


def test_poison_detected():
    assert DET.label(make_poison("Title", "Body text.", CFG["payload"]["note"])).l2


def test_run_ids_unique_across_configs():
    a = load_config("config/v2/mock.yaml", ["llm.mock_p=0.2"])
    b = load_config("config/v2/mock.yaml", ["llm.mock_p=0.5"])
    c = load_config("config/v2/mock.yaml", ["experiment.top_k=6"])
    assert len({run_id_for(a), run_id_for(b), run_id_for(c)}) == 3


def test_null_run_has_no_infection():
    cfg = load_config("config/v2/mock.yaml", ["experiment.experiment_id=test", "experiment.T=40",
                                               "experiment.attack=false"])
    man, steps, summ = read_log(run(cfg, quiet=True))
    m = run_metrics(man, steps)
    assert summ and m["AR_L1"] == 0 and m["AR_L2"] == 0 and m["spontaneous_L2"] == 0


def test_takedown_removes_poison():
    cfg = load_config("config/v2/mock.yaml", ["experiment.experiment_id=test", "experiment.T=40",
                                               "experiment.poison_ttl=12"])
    _, steps, _ = read_log(run(cfg, quiet=True))
    assert all(c["src"] != "poison" for s in steps if s["t"] >= 12 for c in s["ctx"])


def test_oracle_strip_keeps_instruction():
    cfg = load_config("config/v2/mock.yaml", ["experiment.experiment_id=test", "experiment.T=60",
                                               "experiment.defense=oracle_marker_strip", "llm.mock_p=0.9"])
    _, steps, _ = read_log(run(cfg, quiet=True))
    stored_inf = [s for s in steps if s["raw_l2"] and s["stored"]]
    assert stored_inf and all(not s["stored_l1"] and s["stored_l2"] for s in stored_inf)
