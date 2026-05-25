from __future__ import annotations

import bdb
import pdb
import runpy
import sys
import traceback
import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from transformers_sae.tokenization import DataBatch


def _get_all_frames(traceback):
    """Walk the traceback and collect all frames."""
    frames = []
    t = traceback
    while t is not None:
        frames.append(t.tb_frame)
        t = t.tb_next
    return frames


class DebugUtils:
    @staticmethod
    def locals():
        frame = inspect.currentframe()
        return frame.f_back.f_back.f_locals

    @staticmethod
    def max_feature(f: torch.Tensor, batch: DataBatch | None = None):
        if batch is None:
            batch: DataBatch = DebugUtils.locals()["batch"]
        return f[batch.token_mask.bool()].max()

    @staticmethod
    def features_over_thresh(f: torch.Tensor, t: float):
        return (f > t).any(-1)

    @staticmethod
    def tokens_over_thresh(
        f: torch.Tensor, t: float, batch: DataBatch | None = None, tokenizer=None
    ):
        if batch is None:
            batch: DataBatch = DebugUtils.locals()["batch"]
        if tokenizer is None:
            tokenizer = DebugUtils.locals()["tokenizer"]
        over_thresh = DebugUtils.features_over_thresh(f, t)[batch.token_mask.bool()]
        return tokenizer.batch_decode(
            batch.input_ids[batch.token_mask.bool()][over_thresh].unsqueeze(-1)
        )


UTILS = {"du": DebugUtils}


class RichPdb(pdb.Pdb):
    def interaction(self, frame, traceback):
        # Inject into every frame's globals in the traceback
        for f in _get_all_frames(traceback):
            f.f_globals.update(UTILS)
        # Also inject into the current frame if provided
        if frame is not None:
            frame.f_globals.update(UTILS)
        super().interaction(frame, traceback)

    def set_trace(self, frame=None):
        if frame is None:
            frame = sys._getframe().f_back
        # Inject into the calling frame's globals
        frame.f_globals.update(UTILS)
        super().set_trace(frame)


def post_mortem(t=None):
    """Drop-in replacement for pdb.post_mortem() with utilities pre-loaded."""
    if t is None:
        t = sys.exc_info()[2]
    if t is None:
        raise ValueError("No traceback to debug — call inside an except block")
    p = RichPdb()
    p.reset()
    p.interaction(None, t)


def _breakpoint_hook(*args, **kwargs):
    p = RichPdb()
    p.set_trace(sys._getframe().f_back)


sys.breakpointhook = _breakpoint_hook

try:
    runpy.run_path(f"scripts/{sys.argv[1]}", run_name="__main__")
except (SystemExit, bdb.BdbQuit):
    raise
except Exception:
    traceback.print_exc()
    post_mortem()
    sys.exit(1)
