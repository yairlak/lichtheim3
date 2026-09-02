"""Regression tests for the joint driver's RNG checkpoint contract.

Operational bug reproduced by Jean Zay preflight job 1643901 (V100, torch
2.6.0): the 30->60 resume died in `load_state_dict` with

    TypeError: RNG state must be a torch.ByteTensor

`main()` loads with `torch.load(..., map_location=args.device)`, and
map_location relocates EVERY tensor in the file, RNG states included.  On a
GPU node the saved CPU ByteTensors therefore came back as CUDA uint8 tensors,
which `torch.cuda.set_rng_state_all` rejects (`torch.ByteTensor` is a CPU
type).  The CPU `torch` key already had an inline `.cpu()` repair; the `cuda`
key did not.  The checkpoint FILES were always correct -- the corruption was
purely at load time -- so the load-side contract repairs every checkpoint
already written, on any device.

These tests pin the contract:
  * capture/restore representation (CPU uint8 ByteTensors, list for CUDA);
  * a save -> load(map_location=<accelerator>) -> restore round trip, which
    reproduces the exact failure mode on any non-CPU device (MPS locally,
    CUDA on Jean Zay);
  * the real CUDA path when CUDA is present;
  * tolerated legacy/odd representations;
  * cross-device checkpoints (GPU run resumed on a CPU-only machine).
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.train_joint_scratch import (          # noqa: E402
    FINAL_FULL_MODE, JointScratchTrainer, _as_cpu_byte_tensor,
    capture_rng_states, restore_rng_states,
)

TINY = dict(device="cpu", max_words=400,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            dorsal_pool_size=32, batch_size=8, subset_mode=FINAL_FULL_MODE,
            subset_per_band=822, subset_size=32, lr_boundary_steps=6,
            allow_glove_fallback=True, require_subset_hash=False,
            glove_path="tests/_no_such_glove_file.txt")


def accelerator() -> str | None:
    """A non-CPU device that `map_location` can relocate tensors to.

    This is what makes the regression reproducible off-cluster: the failure
    depends on the state not being on CPU, not on the device being CUDA.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


# ==========================================================  representation ==

def test_capture_returns_cpu_byte_tensors():
    rs = capture_rng_states()
    assert isinstance(rs["torch"], torch.Tensor)
    assert rs["torch"].dtype == torch.uint8 and rs["torch"].device.type == "cpu"
    torch.set_rng_state(rs["torch"])            # accepted by the setter
    assert rs["numpy"] is not None and rs["python"] is not None
    if torch.cuda.is_available():
        assert isinstance(rs["cuda"], list)
        assert len(rs["cuda"]) == torch.cuda.device_count()
        for s in rs["cuda"]:
            assert s.dtype == torch.uint8 and s.device.type == "cpu"
    else:
        assert rs["cuda"] is None


def test_as_cpu_byte_tensor_normalises_dtype_device_and_sequences():
    st = torch.get_rng_state()
    assert torch.equal(_as_cpu_byte_tensor(st.to(torch.int64)), st)
    assert torch.equal(_as_cpu_byte_tensor(st.tolist()), st)
    out = _as_cpu_byte_tensor(st)
    assert out.dtype == torch.uint8 and out.device.type == "cpu"
    dev = accelerator()
    if dev:
        moved = _as_cpu_byte_tensor(st.to(dev))
        assert moved.device.type == "cpu" and torch.equal(moved, st)


# ============================================================  round trips  ==

def _draw3():
    return (torch.randint(0, 2**31, (3,)).tolist(),
            np.random.randint(0, 2**31, 3).tolist(),
            [random.randrange(2**31) for _ in range(3)])


def test_save_load_roundtrip_restores_every_generator(tmp_path):
    torch.manual_seed(1234); np.random.seed(99); random.seed(7)
    rs = capture_rng_states()
    expected = _draw3()

    p = tmp_path / "rng.pt"
    torch.save({"rng_states": rs}, str(p))
    loaded = torch.load(str(p), map_location="cpu", weights_only=False)

    torch.manual_seed(0); np.random.seed(0); random.seed(0)      # perturb
    names = restore_rng_states(loaded["rng_states"])
    assert {"torch", "numpy", "python"} <= set(names)
    assert _draw3() == expected


@pytest.mark.skipif(accelerator() is None, reason="needs a non-CPU device")
def test_relocated_checkpoint_is_repaired(tmp_path):
    """THE regression: map_location moves RNG states off CPU (the Jean Zay
    failure); the contract must repair them and restore the streams exactly."""
    dev = accelerator()
    torch.manual_seed(4321); np.random.seed(11); random.seed(3)
    rs = capture_rng_states()
    # A CUDA-style entry is present even on MPS-only machines, so the
    # cuda-key code path is exercised wherever this test can run.
    rs = dict(rs, cuda=[torch.get_rng_state().clone()])
    expected = _draw3()

    p = tmp_path / "ck.pt"
    torch.save({"rng_states": rs}, str(p))
    loaded = torch.load(str(p), map_location=dev, weights_only=False)

    # precondition: the bug's trigger really is present after this load
    assert loaded["rng_states"]["torch"].device.type == dev
    assert loaded["rng_states"]["cuda"][0].device.type == dev
    with pytest.raises(TypeError, match="ByteTensor"):
        torch.set_rng_state(loaded["rng_states"]["cuda"][0])

    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    restore_rng_states(loaded["rng_states"])                     # must not raise
    assert _draw3() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_rng_states_restored_bitwise():
    rs = capture_rng_states()
    before = [s.clone() for s in torch.cuda.get_rng_state_all()]
    torch.cuda.manual_seed_all(12345)                            # perturb
    restore_rng_states(rs)
    after = torch.cuda.get_rng_state_all()
    assert len(after) == len(before)
    assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(after, before))


# =====================================================  tolerated variants  ==

def test_legacy_and_partial_representations_are_accepted():
    st = torch.get_rng_state()
    restore_rng_states({})                                       # nothing saved
    restore_rng_states({"torch": st, "numpy": None, "python": None})
    restore_rng_states({"torch": st.to(torch.int64)})            # odd dtype
    restore_rng_states({"torch": st, "cuda": st.clone()})        # single tensor
    restore_rng_states({"torch": st, "cuda": None})


def test_cuda_checkpoint_resumes_on_cpu_only_machine():
    """A Jean Zay (GPU) checkpoint must load on the Mac: CUDA states are
    simply not restored when CUDA is absent, and nothing raises."""
    rs = dict(capture_rng_states(), cuda=[torch.get_rng_state().clone()])
    names = restore_rng_states(rs)
    if not torch.cuda.is_available():
        assert not any(n.startswith("cuda") for n in names)


# =========================================================  end-to-end path ==

@pytest.mark.skipif(accelerator() is None, reason="needs a non-CPU device")
def test_driver_resume_through_relocating_map_location(tmp_path):
    """Exercises the exact line that failed on Jean Zay: a driver checkpoint
    loaded with a relocating map_location and fed to load_state_dict."""
    dev = accelerator()
    a = JointScratchTrainer(regime="j0", seed=22, **TINY)
    for _ in range(4):
        a.train_step()
    p = tmp_path / "step4.pt"
    torch.save(a.state_dict(), str(p))
    for _ in range(3):
        a.train_step()

    b = JointScratchTrainer(regime="j0", seed=22, **TINY)
    ck = torch.load(str(p), map_location=dev, weights_only=False)   # relocates
    b.load_state_dict(ck, source="test")                           # must not raise
    assert b.global_step == 4 and b.cursors["naming"] == 4
    for _ in range(3):
        b.train_step()

    sa, sb = a.model.state_dict(), b.model.state_dict()
    bad = [k for k in sa if not torch.equal(sa[k], sb[k])]
    assert not bad, f"resume diverged on {len(bad)} tensors, e.g. {bad[:3]}"
