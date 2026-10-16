"""Tests for the real-model and human deciders (no network: fake client, fake player)."""

import asyncio
from types import SimpleNamespace

import pytest

from inference_rally.deciders import HumanDecider, OpenAIDecider
from inference_rally.errors import DeciderError, DecisionTimeoutError
from inference_rally.maze import Heading, Move
from inference_rally.models import Observation, OptionView, RacerSpec

OBS = Observation(
    node=(1, 1),
    heading=Heading.NORTH,
    goal_dx=3,
    goal_dy=2,
    options=[
        OptionView(move=Move.LEFT, compass="west", visits=0),
        OptionView(move=Move.RIGHT, compass="east", visits=0),
    ],
)
SPEC = RacerSpec(
    id="llama_gpu",
    name="Llama",
    device="GPU",
    runtime="openai",
    color="#7F77DD",
    base_url="http://127.0.0.1:1/v1",
    model="llama",
)


def _chunk(content: str | None = None, usage: int | None = None) -> SimpleNamespace:
    choices = [] if content is None else [SimpleNamespace(delta=SimpleNamespace(content=content))]
    return SimpleNamespace(
        choices=choices,
        usage=None if usage is None else SimpleNamespace(completion_tokens=usage),
    )


class FakeClient:
    """Mimics ``AsyncOpenAI().chat.completions.create(stream=True)``."""

    def __init__(self, pieces: list[str], usage: int | None = None, fail: bool = False) -> None:
        self.pieces, self.usage, self.fail = pieces, usage, fail
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.fail:
            raise ConnectionError("connection refused")

        async def stream():  # type: ignore[no-untyped-def]
            for piece in self.pieces:
                await asyncio.sleep(0.001)
                yield _chunk(piece)
            if self.usage is not None:
                yield _chunk(usage=self.usage)

        return stream()


def test_openai_decider_streams_and_times_reply() -> None:
    client = FakeClient(['{"move"', ': "right"', ', "why": "east"}'], usage=11)
    seen: list[int] = []
    decider = OpenAIDecider(SPEC, client=client)
    result = asyncio.run(decider.decide([{"role": "user", "content": "hi"}], OBS, lambda m: seen.append(m.tokens)))
    assert result.text == '{"move": "right", "why": "east"}'
    assert seen == [1, 2, 3]  # live updates while streaming
    assert result.metrics.tokens == 11  # server-reported usage wins over chunk count
    assert result.metrics.tps > 0
    assert client.calls[0]["stream"] is True and client.calls[0]["temperature"] == 0


def test_openai_decider_reports_offline_server() -> None:
    decider = OpenAIDecider(SPEC, client=FakeClient([], fail=True))
    with pytest.raises(DeciderError):
        asyncio.run(decider.warmup())


def test_openai_decider_requires_endpoint() -> None:
    with pytest.raises(DeciderError):
        OpenAIDecider(SPEC.model_copy(update={"base_url": None}), client=FakeClient([]))


def test_human_decider_accepts_legal_move_on_their_turn() -> None:
    async def scenario() -> tuple[bool, bool, object]:
        human = HumanDecider(timeout_s=2)
        task = asyncio.create_task(human.decide([], OBS, lambda _: None))
        await asyncio.sleep(0.01)
        illegal = human.submit(Move.STRAIGHT)  # not an option at this junction
        legal = human.submit(Move.LEFT)
        return illegal, legal, await task

    illegal, legal, result = asyncio.run(scenario())
    assert illegal is False and legal is True
    assert '"left"' in result.text  # type: ignore[attr-defined]
    assert result.metrics.tokens == 0  # type: ignore[attr-defined]


def test_human_decider_rejects_moves_when_not_waiting() -> None:
    assert HumanDecider().submit(Move.LEFT) is False


def test_human_decider_times_out() -> None:
    with pytest.raises(DecisionTimeoutError):
        asyncio.run(HumanDecider(timeout_s=0.05).decide([], OBS, lambda _: None))
