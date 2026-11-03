"""One real optimiser step at a time, off the FastAPI event loop.

The network, the loss function, the optimiser, the DataLoader and the optimiser
update all come from ``training/training.py`` (``train_step()``). This module
only decides *when* the next step runs, how the browser is paced, and what it is
told afterwards; it never re-implements the training maths.

Three things matter for correctness:

* every step runs in a caller-supplied single-worker executor, so no PyTorch
  computation (training or inference) can ever overlap another one, and the
  event loop stays free to answer WebSocket messages;
* pause and reset take effect *between* steps, never inside
  ``loss.backward()``/``optimiser.step()``;
* speed changes only the pause after a step and how many steps share one visual
  snapshot. It never changes the learning rate, the optimiser, the batch size or
  the number of samples that PyTorch processes.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from concurrent.futures import Executor
from typing import Any

from .visualiser import (
    build_training_step,
    linear_layers,
    restore_parameters,
    select_partial_indices,
    snapshot_parameters,
    training_totals,
    weights_checksum,
)

logger = logging.getLogger("visualiser.training")

# `delay`: seconds to wait after a completed optimiser step (visual pacing).
# `frame_every`: how many optimiser steps share one frame sent to the browser.
# Neither value touches the training mathematics.
SPEEDS: dict[str, dict[str, float]] = {
    "slow": {"delay": 0.35, "frame_every": 1},
    "normal": {"delay": 0.05, "frame_every": 1},
    "fast": {"delay": 0.0, "frame_every": 20},
}
DEFAULT_SPEED = "normal"

STATES = ("idle", "running", "paused", "complete")

MESSAGES = {
    "idle": "Training has not started. The diagram shows the untrained weights.",
    "running": "Training: every step is one real optimiser update on one MNIST image.",
    "paused": "Training is paused between optimiser steps; the weights are kept.",
    "complete": "The configured epoch is complete. Start again to retrain from scratch.",
}


class Outbox:
    """Bounded, latest-wins queue of payloads for one WebSocket client.

    Control payloads (`model_info`, `inference`, `weights`, `training_state`) are
    all delivered in order. Training frames are not: if the browser has not read
    the previous frame yet, that frame is replaced by the newer one, so a slow
    renderer can never build an unbounded backlog. Nothing is dropped from
    training itself — every frame describes a step PyTorch has already done.

    ``put`` is called from the event loop, ``get`` from the per-client sender
    task, so no locking is needed.
    """

    def __init__(self, capacity: int = 8) -> None:
        self.capacity = capacity
        self._frames: deque[dict] = deque()
        self._closed = False
        self._ready = asyncio.Event()

    def put(self, payload: dict) -> None:
        if self._closed:
            return
        if payload.get("type") == "training_step":
            for index in range(len(self._frames) - 1, -1, -1):
                if self._frames[index].get("type") == "training_step":
                    self._frames[index] = payload
                    break
            else:
                self._frames.append(payload)
        else:
            self._frames.append(payload)

        while len(self._frames) > self.capacity:
            self._frames.popleft()

        self._ready.set()

    async def get(self) -> dict | None:
        while not self._frames:
            if self._closed:
                return None
            self._ready.clear()
            await self._ready.wait()
        return self._frames.popleft()

    @property
    def pending(self) -> int:
        return len(self._frames)

    def close(self) -> None:
        self._closed = True
        self._frames.clear()
        self._ready.set()


class TrainingSession:
    """The one training run this server process is allowed to have.

    Only the event loop calls these methods; the steps themselves run in
    `executor`, which must have a single worker so two steps (or a step and an
    inference pass) can never touch the model at the same time.
    """

    def __init__(self, module: Any, executor: Executor) -> None:
        self.module = module
        self.executor = executor

        self.state = "idle"
        self.speed = DEFAULT_SPEED
        self.step = 0
        # Bumped by every reset, so frames that were already in flight when a
        # run restarted are recognisable and can be ignored by the browser.
        self.run = 0
        self.total_steps, self.epochs, self.train_size = training_totals(module)

        # The untrained tensors, so Stop / Reset can undo a run completely.
        self.initial_parameters = snapshot_parameters(module)

        # The Partial view's connections, chosen once from those initial weights
        # and never recomputed while training: every frame describes this same
        # set, so the drawn lines stay put while their values move.
        self.selection = select_partial_indices(module)

        self.last_frame: dict | None = None
        self._iterator = None
        self._task: asyncio.Task | None = None
        self._loop_active = False
        self._subscribers: list[asyncio.Queue] = []

    # -- status ------------------------------------------------------------

    def epoch(self) -> int:
        if self.step <= 0:
            return 0
        return min(self.epochs - 1, (self.step - 1) // self.train_size)

    @property
    def frame_every(self) -> int:
        return int(SPEEDS[self.speed]["frame_every"])

    @property
    def delay(self) -> float:
        return float(SPEEDS[self.speed]["delay"])

    def status(self) -> dict:
        """The `training_state` message (also embedded in `model_info`)."""
        return {
            "type": "training_state",
            "state": self.state,
            "run": self.run,
            "step": self.step,
            "total_steps": self.total_steps,
            "epoch": self.epoch(),
            "epochs": self.epochs,
            "train_size": self.train_size,
            "batch_size": getattr(self.module, "batch_size", 1),
            "speed": self.speed,
            "speeds": list(SPEEDS),
            "frame_every": self.frame_every,
            "delay": self.delay,
            "message": MESSAGES[self.state],
        }

    def training_info(self) -> dict:
        """The same status without the message type, for `model_info`."""
        payload = self.status()
        payload.pop("type")
        return payload

    # -- subscribers -------------------------------------------------------

    def subscribe(self, outbox: Outbox) -> None:
        self._subscribers.append(outbox)

    def unsubscribe(self, outbox: Outbox) -> None:
        outbox.close()
        if outbox in self._subscribers:
            self._subscribers.remove(outbox)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def deliver(self, outbox: Outbox, payload: dict) -> None:
        """Hand a payload to one client through its bounded outbox.

        If the browser has not read the previous training frame yet, that frame
        is replaced instead of queued, so a slow renderer can never make the
        server accumulate an unbounded backlog of snapshots.
        """
        outbox.put(payload)

    async def broadcast(self, payload: dict) -> None:
        for outbox in list(self._subscribers):
            self.deliver(outbox, payload)

    # -- controls ----------------------------------------------------------

    async def start(self) -> None:
        """Start training, or resume a paused run."""
        if self.state == "running":
            return
        if self.state == "complete":
            # One full pass is the configured run, so a finished run starts a new
            # one from the initial weights instead of doing nothing.
            await self.reset()

        self.state = "running"
        self._loop_active = True
        logger.info(
            "training started at step %d/%d, %s speed (pause %.2fs, frame every %d step(s))",
            self.step,
            self.total_steps,
            self.speed,
            self.delay,
            self.frame_every,
        )
        await self.broadcast(self.status())

        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="training-loop")

    async def pause(self) -> None:
        """Stop before the next optimiser step and keep the current weights."""
        if self.state != "running":
            return
        self._loop_active = False
        self.state = "paused"
        logger.info("training paused after %d/%d steps", self.step, self.total_steps)
        await self.broadcast(self.status())

    async def reset(self) -> None:
        """Stop the run and put the network back to its initial weights."""
        self._loop_active = False
        await self._join()

        restore_parameters(self.module, self.initial_parameters)
        # The initial weights are back, so the load-time rule derives the same
        # selected connections again; nothing has to be remembered separately.
        self.selection = select_partial_indices(self.module)
        self._iterator = None
        self.last_frame = None
        self.step = 0
        self.run += 1
        self.state = "idle"
        logger.info(
            "training reset: initial weights restored, progress back to 0, "
            "%d partial connections selected again",
            sum(int(array.size) for array in self.selection),
        )
        await self.broadcast(self.status())

    async def set_speed(self, speed: str) -> None:
        """Change only the visual pacing; the training maths stays identical."""
        if speed not in SPEEDS:
            raise ValueError(f"unknown speed {speed!r}, expected one of {list(SPEEDS)}")
        if speed == self.speed:
            return
        self.speed = speed
        logger.info(
            "visualisation speed %s (pause %.2fs, frame every %d step(s)); "
            "learning rate, optimiser and batch size are unchanged",
            speed,
            self.delay,
            self.frame_every,
        )
        await self.broadcast(self.status())

    async def pause_for_disconnect(self) -> None:
        """No browser is watching any more: stop at the next step boundary."""
        if self.state != "running":
            return
        logger.info("last client disconnected; pausing at the next step boundary")
        await self.pause()

    async def shutdown(self) -> None:
        self._loop_active = False
        await self._join()

    # -- the loop ----------------------------------------------------------

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while self._loop_active and self.step < self.total_steps:
                speed = self.speed
                frame_every = int(SPEEDS[speed]["frame_every"])
                with_frame = (self.step + 1) % frame_every == 0

                # Only fingerprinted when a frame will be sent: hashing 109,200
                # parameters is not free and fast mode skips most frames.
                checksum_before = None
                if with_frame:
                    checksum_before = await loop.run_in_executor(
                        self.executor, weights_checksum, linear_layers(self.module)
                    )

                images, labels = self._next_batch()
                result = await loop.run_in_executor(
                    self.executor, self.module.train_step, images, labels
                )
                self.step += 1

                if with_frame:
                    frame = build_training_step(
                        self.module,
                        result,
                        run=self.run,
                        step=self.step,
                        total_steps=self.total_steps,
                        epochs=self.epochs,
                        train_size=self.train_size,
                        speed=speed,
                        frame_every=frame_every,
                        checksum_before=checksum_before,
                        selection=self.selection,
                    )
                    self.last_frame = frame
                    await self.broadcast(frame)

                if self.step >= self.total_steps:
                    self.state = "complete"
                    self._loop_active = False
                    logger.info(
                        "training complete: %d/%d steps, loss %.4f",
                        self.step,
                        self.total_steps,
                        float(result["loss"]),
                    )
                    break

                if self.delay > 0:
                    await asyncio.sleep(self.delay)
                else:
                    # Yield so WebSocket traffic and control messages are served
                    # while steps run as fast as the CPU allows.
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - never let the loop die silently
            logger.exception("training step failed; pausing")
            self.state = "paused"
        finally:
            self._loop_active = False
            if self.state == "running":
                self.state = "paused"
            await self.broadcast(self.status())

    def _next_batch(self) -> tuple[Any, Any]:
        """The next batch of the existing DataLoader (shuffled, batch size 1)."""
        while True:
            if self._iterator is None:
                self._iterator = iter(self.module.train_loader)
            try:
                return next(self._iterator)
            except StopIteration:
                self._iterator = None

    async def _join(self) -> None:
        """Wait for the step currently in flight, then forget the task."""
        task = self._task
        self._task = None
        if task is None or task.done():
            return
        try:
            await task
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - already logged by the training loop
            pass
