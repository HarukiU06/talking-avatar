import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Runs in a fresh interpreter: the fix patches torch globally.
PROBE = textwrap.dedent("""
    import sys
    sys.path.insert(0, {root!r})
    import sbv2_run
    sbv2_run.single_process_distributed()

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    # What SBV2's trainer does: a gloo process group, then DDP-wrapped models.
    dist.init_process_group(backend="gloo", init_method="env://")
    assert (dist.get_rank(), dist.get_world_size()) == (0, 1)
    model = DDP(torch.nn.Linear(4, 1), device_ids=[0])
    assert list(model.state_dict()) == ["module.weight", "module.bias"]
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    torch.manual_seed(0)
    x, y = torch.randn(32, 4), torch.randn(32, 1)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with model.no_sync():
        pass
    assert losses[-1] < losses[0], losses
    dist.destroy_process_group()
    print("ok")
""")


class SingleProcessDistributed(unittest.TestCase):
    def test_trainer_style_setup_trains_without_gloo(self):
        result = subprocess.run([sys.executable, "-c", PROBE.format(root=str(ROOT))],
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
