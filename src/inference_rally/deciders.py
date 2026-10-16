"""Deciders: the "brains" behind racers.

A decider receives the rendered prompt for a junction and returns the raw model reply
plus timing. The race engine does not care whether the reply came from ExecuTorch,
llama.cpp, GenieX or the simulator — that is what makes the Mac → Snapdragon switch a
config change.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from inference_rally.errors import DeciderError, DecisionTimeoutError
from inference_rally.maze import Maze, Move
from inference_rally.metrics import TokenMeter
from inference_rally.models import DecisionMetrics, Observation, RacerSpec, SimProfile

HUMAN_TIMEOUT_S = 20.0
MODEL_TIMEOUT_S = 30.0
LOCAL_API_KEY = "local"  # local servers ignore the key, but the client requires one

TokenCallback = Callable[[TokenMeter], None]


@dataclass(frozen=True)
class DecisionResult:
    """Raw reply text and the client-side timing for one decision."""

    text: str
    metrics: DecisionMetrics


class Decider(Protocol):
    """Anything that can answer a junction prompt."""

    async def warmup(self) -> None:
        """Get ready before the countdown (load the model, open connections)."""
        ...

    async def decide(
        self, messages: list[dict[str, str]], obs: Observation, on_token: TokenCallback
    ) -> DecisionResult:
        """Return the model's reply, calling ``on_token`` as tokens stream in."""
        ...


class SimulatedDecider:
    """Fake model with realistic speed and imperfect judgment, for UI work and tests.

    With probability ``accuracy`` it picks a move on a shortest route; otherwise it
    behaves like a plausible small model: heads roughly toward the exit and avoids
    roads it has already driven, which still leads it into dead ends sometimes.
    """

    def __init__(self, profile: SimProfile, maze: Maze, rng: random.Random) -> None:
        """Create a simulated racer.

        Args:
            profile: Speed and accuracy settings.
            maze: Track, used to know which moves are actually optimal.
            rng: Dedicated RNG so each racer's behavior is reproducible.
        """
        self._profile = profile
        self._maze = maze
        self._rng = rng

    async def warmup(self) -> None:
        """Nothing to load for a simulated racer."""

    async def decide(
        self, messages: list[dict[str, str]], obs: Observation, on_token: TokenCallback
    ) -> DecisionResult:
        """Choose a move, then stream a fake reply at the profile's speed."""
        move, why = self._choose(obs)
        text = json.dumps({"move": move.value, "why": why})
        meter = TokenMeter()

        await asyncio.sleep(self._jittered(self._profile.ttft_ms / 1000))
        low, high = self._profile.reply_tokens
        for _ in range(self._rng.randint(low, high)):
            meter.add_tokens(1)
            on_token(meter)
            await asyncio.sleep(self._jittered(1 / self._profile.tps))

        return DecisionResult(text=text, metrics=meter.finish())

    def _choose(self, obs: Observation) -> tuple[Move, str]:
        options = self._maze.options(obs.node, obs.heading)
        optimal = [m for m, dst in options.items() if self._maze.is_optimal_step(obs.node, dst)]
        if optimal and self._rng.random() < self._profile.accuracy:
            move = self._rng.choice(optimal)
            return move, self._reason_toward(obs, move)
        return self._heuristic(obs)

    def _heuristic(self, obs: Observation) -> tuple[Move, str]:
        """Greedy "small model" logic: prefer unexplored roads pointing at the exit."""
        visits = {option.move: option.visits for option in obs.options}
        candidates = [m for m in visits if m is not Move.BACK] or list(visits)

        def score(move: Move) -> float:
            dx, dy = obs.heading.turn(move).vector
            toward = dx * _sign(obs.goal_dx) + dy * _sign(obs.goal_dy)
            return toward - 1.5 * visits[move] + self._rng.uniform(-0.8, 0.8)

        move = max(candidates, key=score)
        if visits[move]:
            return move, "trying another way"
        return move, self._reason_toward(obs, move)

    def _reason_toward(self, obs: Observation, move: Move) -> str:
        compass = obs.heading.turn(move).compass
        templates = (f"exit is {compass}", f"head {compass}", f"{compass} looks open")
        return self._rng.choice(templates)

    def _jittered(self, seconds: float) -> float:
        spread = self._profile.jitter
        return max(0.0, seconds * self._rng.uniform(1 - spread, 1 + spread))


class OpenAIDecider:
    """A real model behind any OpenAI-compatible server (llama.cpp, ExecuTorch, GenieX).

    Replies are streamed so TTFT and tokens/sec are measured on our side, the same way
    for every runtime.
    """

    def __init__(self, spec: RacerSpec, client: object | None = None) -> None:
        """Create a decider for one racer.

        Args:
            spec: Racer with ``base_url`` and ``model`` set.
            client: Injected ``AsyncOpenAI``-like client (tests); built from ``spec`` if omitted.

        Raises:
            DeciderError: If ``base_url``/``model`` are missing or ``openai`` is not installed.
        """
        if not spec.base_url or not spec.model:
            raise DeciderError(f"Racer {spec.id!r} needs base_url and model")
        self._spec = spec
        if client is None:
            try:
                # Imported lazily: only real-model profiles need the package.
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise DeciderError("Install the 'openai' package to race real models") from exc
            client = AsyncOpenAI(
                base_url=spec.base_url, api_key=LOCAL_API_KEY, timeout=MODEL_TIMEOUT_S, max_retries=0
            )
        self._client = client

    async def warmup(self) -> None:
        """One tiny request so model loading / GPU shader compilation happens before GO."""
        await self._complete([{"role": "user", "content": "Say ok."}], max_tokens=1, on_token=None)

    async def decide(
        self, messages: list[dict[str, str]], obs: Observation, on_token: TokenCallback
    ) -> DecisionResult:
        """Stream the model's answer for one junction."""
        return await self._complete(messages, self._spec.max_tokens, on_token)

    async def _complete(
        self, messages: list[dict[str, str]], max_tokens: int, on_token: TokenCallback | None
    ) -> DecisionResult:
        meter = TokenMeter()
        parts: list[str] = []
        usage_tokens: int | None = None
        try:
            stream = await self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=self._spec.model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)
                    # Local servers stream one token per chunk, so a chunk is a token.
                    meter.add_tokens(1)
                    if on_token:
                        on_token(meter)
                if getattr(chunk, "usage", None):
                    usage_tokens = chunk.usage.completion_tokens
        except Exception as exc:  # noqa: BLE001 - any client/network failure means "offline"
            raise DeciderError(f"{self._spec.id}: {exc}") from exc

        if usage_tokens:
            meter.tokens = usage_tokens  # exact count when the server reports it
        return DecisionResult(text="".join(parts), metrics=meter.finish())


class HumanDecider:
    """The human player: waits for a move submitted from the browser (arrow keys).

    Reaction time is reported as TTFT/latency so the human sits on the same
    leaderboard as the models.
    """

    def __init__(self, timeout_s: float = HUMAN_TIMEOUT_S) -> None:
        """Create the decider; ``timeout_s`` is how long the player may think."""
        self._timeout_s = timeout_s
        self._pending: asyncio.Future[Move] | None = None
        self._allowed: set[Move] = set()

    @property
    def awaiting(self) -> bool:
        """True while the car is stopped at a junction waiting for the player."""
        return self._pending is not None and not self._pending.done()

    def submit(self, move: Move) -> bool:
        """Deliver the player's move. Returns False if it is not their turn or not legal here."""
        if not self.awaiting or move not in self._allowed:
            return False
        assert self._pending is not None
        self._pending.set_result(move)
        return True

    async def warmup(self) -> None:
        """Humans are always warmed up."""

    async def decide(
        self, messages: list[dict[str, str]], obs: Observation, on_token: TokenCallback
    ) -> DecisionResult:
        """Wait for the player's arrow key; raise if they take too long."""
        self._allowed = {option.move for option in obs.options}
        self._pending = asyncio.get_running_loop().create_future()
        started = time.perf_counter()
        try:
            move = await asyncio.wait_for(self._pending, timeout=self._timeout_s)
        except asyncio.TimeoutError as exc:
            raise DecisionTimeoutError("player did not choose in time") from exc
        finally:
            self._pending = None
        reaction_ms = round((time.perf_counter() - started) * 1000, 1)
        return DecisionResult(
            text=json.dumps({"move": move.value, "why": ""}),
            metrics=DecisionMetrics(ttft_ms=reaction_ms, tps=0, tokens=0, latency_ms=reaction_ms),
        )


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)
