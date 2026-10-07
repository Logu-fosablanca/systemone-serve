from __future__ import annotations
from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: str
    criteria: dict[str, str]  # option_key -> description, up to 255


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: str
    criteria: list[str]  # ordered level labels, 2-10


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: str
    criteria: dict[str, str] | None = None  # optional {"true": "...", "false": "..."}


Question = Annotated[
    Union[ChoiceQuestion, ScoreQuestion, NoulQuestion],
    Field(discriminator="type"),
]


class SystemOneRequest(BaseModel):
    model: str | None = None
    state: Any  # text, or any JSON value (Clef renders JSON with sorted keys)
    questions: dict[str, Question]
    scoring: Literal["single", "multi"] = "single"
    # single: 1 forward pass, first token of each option (fast, works when option keys are short)
    # multi:  1 forward pass per option, mean log-prob over all tokens (slower, accurate for long keys)

    @field_validator("state")
    @classmethod
    def state_not_none(cls, v: Any) -> Any:
        if v is None:
            raise ValueError("state must not be null")
        return v


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float
    confidence: float


Answer = Annotated[Union[ChoiceAnswer, ScoreAnswer, NoulAnswer], Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int = 0  # always 0 — we read logits, never generate


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
