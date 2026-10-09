#!/usr/bin/env python3
"""Run one Style-Bert-VITS2 script with two runtime fixes. Runs in .venv-sbv2.

    python sbv2_run.py SCRIPT [ARGS...]          (cwd: the Style-Bert-VITS2 checkout)

The checkout stays untouched (it is gitignored and re-cloned by setup_sbv2.sh);
like infinitetalk_run.py, this patches at runtime and then hands over.

1. pyannote's speaker-embedding model, which style_gen.py loads, is a pickle
   that names four plain classes (a version string, a namedtuple, two enums).
   torch >= 2.6 loads checkpoints with weights_only=True and refuses them, so
   style vectors never get made. Exactly those four are allowlisted, which
   keeps the safe loader for everything else, unlike
   TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, which would let any pickle run code.

2. torch 2.8's Windows wheels ship gloo without a working transport:
   init_process_group(backend="gloo") fails with "makeDeviceForHostname():
   unsupported gloo device" whatever GLOO_DEVICE_TRANSPORT says, and SBV2's
   trainer always builds a process group and wraps its models in DDP. Training
   here is one process on one GPU, so a "fake" process group answers its
   rank/world-size questions, and DDP becomes a pass-through module: with a
   single process there is nothing to synchronise. The pass-through keeps
   DDP's "module." prefix, so checkpoints are identical to a real DDP run's.
   (torch's fake backend can't stand in for DDP itself: its collectives return
   no tensors and DDP's gradient reducer asserts on that.)
"""
import contextlib
import os
import platform
import runpy
import sys


def allow_pyannote_checkpoint() -> None:
    import torch
    from torch.torch_version import TorchVersion

    try:
        from pyannote.audio.core.task import Problem, Resolution, Specifications
    except ImportError:
        return  # not installed: nothing will load its checkpoint
    torch.serialization.add_safe_globals([TorchVersion, Specifications, Problem, Resolution])


def single_process_distributed() -> None:
    import torch
    import torch.distributed as dist
    import torch.nn.parallel
    import torch.nn.parallel.distributed
    from torch.testing._internal.distributed.fake_pg import FakeStore

    real_init = dist.init_process_group

    def init_process_group(*args, **kwargs):
        real_init(backend="fake", store=FakeStore(), rank=0, world_size=1)

    class PassThroughDDP(torch.nn.Module):
        def __init__(self, module, *args, **kwargs):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

        def no_sync(self):
            return contextlib.nullcontext()

    dist.init_process_group = init_process_group
    torch.nn.parallel.DistributedDataParallel = PassThroughDDP
    torch.nn.parallel.distributed.DistributedDataParallel = PassThroughDDP


def main() -> None:
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.getcwd())  # the checkout's own modules (config, style_bert_vits2, ...)
    allow_pyannote_checkpoint()
    if platform.system() == "Windows":
        single_process_distributed()
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
