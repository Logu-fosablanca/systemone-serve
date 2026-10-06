from __future__ import annotations
import torch
import torch.nn.functional as F  # used for softmax in answer_* methods
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import config
from schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class Engine:
    def __init__(self) -> None:
        self.device = config.device
        self.model_name = config.model_path.rstrip("/").split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=_DTYPES[config.dtype],
            device_map=config.device,
        )
        model.eval()
        self.model = torch.compile(model)

    def _build_prefix(self, state: str, instructions: str, criteria_text: str) -> torch.Tensor:
        prompt = (
            f"State:\n{state}\n\n"
            f"Question: {instructions}\n"
            f"{criteria_text}\n"
            f"Answer:"
        )
        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

    def _score_options(
        self, prefix_ids: torch.Tensor, options: list[str], scoring: str = "single"
    ) -> list[float]:
        if scoring == "multi":
            return self._score_options_multi(prefix_ids, options)
        with torch.inference_mode():
            last_logits = self.model(prefix_ids).logits[0, -1]  # (vocab,) — 1 forward pass
        scores = []
        for opt in options:
            ids = self.tokenizer.encode(opt, add_special_tokens=False)
            scores.append(last_logits[ids[0]].item() if ids else float("-inf"))
        return scores

    def _score_options_multi(self, prefix_ids: torch.Tensor, options: list[str]) -> list[float]:
        """Mean log-prob over all option tokens — 1 forward pass per option."""
        prefix_len = prefix_ids.shape[1]
        log_probs: list[float] = []
        with torch.inference_mode():
            for opt in options:
                ids = self.tokenizer.encode(opt, add_special_tokens=False)
                if not ids:
                    log_probs.append(float("-inf"))
                    continue
                full_ids = torch.cat(
                    [prefix_ids, torch.tensor([ids], device=self.device)], dim=1
                )
                logits = self.model(full_ids).logits[0]
                lp = sum(
                    F.log_softmax(logits[prefix_len - 1 + i], dim=-1)[tok].item()
                    for i, tok in enumerate(ids)
                )
                log_probs.append(lp / len(ids))
        return log_probs

    def answer_choice(self, state: str, q: ChoiceQuestion, scoring: str) -> tuple[ChoiceAnswer, int]:
        criteria_text = "\n".join(f"- {k}: {v}" for k, v in q.criteria.items())
        prefix_ids = self._build_prefix(state, q.instructions, criteria_text)
        keys = list(q.criteria.keys())
        raw = self._score_options(prefix_ids, keys, scoring)
        probs = F.softmax(torch.tensor(raw), dim=-1).tolist()
        probs_map = dict(zip(keys, probs))
        best = max(probs_map, key=probs_map.__getitem__)
        return (
            ChoiceAnswer(choice=best, confidence=probs_map[best], probabilities=probs_map),
            prefix_ids.shape[1],
        )

    def answer_score(self, state: str, q: ScoreQuestion, scoring: str) -> tuple[ScoreAnswer, int]:
        levels = q.criteria
        criteria_text = "\n".join(f"{i}. {lv}" for i, lv in enumerate(levels))
        prefix_ids = self._build_prefix(state, q.instructions, criteria_text)
        raw = self._score_options(prefix_ids, [str(i) for i in range(len(levels))], scoring)
        probs = F.softmax(torch.tensor(raw), dim=-1).tolist()
        probs_map = {str(i): p for i, p in enumerate(probs)}
        score = sum(i * p for i, p in enumerate(probs))
        legend = {str(i): lv for i, lv in enumerate(levels)}
        return (
            ScoreAnswer(
                score=score,
                legend=legend,
                probabilities=probs_map,
                confidence=max(probs),
            ),
            prefix_ids.shape[1],
        )

    def answer_noul(self, state: str, q: NoulQuestion, scoring: str) -> tuple[NoulAnswer, int]:
        criteria_text = "\n".join(f"- {k}: {v}" for k, v in (q.criteria or {}).items())
        prefix_ids = self._build_prefix(state, q.instructions, criteria_text)
        raw = self._score_options(
            prefix_ids, [config.noul_yes_token, config.noul_no_token], scoring
        )
        probs = F.softmax(torch.tensor(raw), dim=-1).tolist()
        return NoulAnswer(noul=probs[0]), prefix_ids.shape[1]

    def run(
        self,
        state: str,
        questions: dict[str, ChoiceQuestion | ScoreQuestion | NoulQuestion],
        scoring: str,
    ) -> tuple[dict[str, Answer], int]:
        answers: dict[str, Answer] = {}
        total_tokens = 0
        for qid, q in questions.items():
            if isinstance(q, ChoiceQuestion):
                ans, tokens = self.answer_choice(state, q, scoring)
            elif isinstance(q, ScoreQuestion):
                ans, tokens = self.answer_score(state, q, scoring)
            else:
                ans, tokens = self.answer_noul(state, q, scoring)
            answers[qid] = ans
            total_tokens += tokens
        return answers, total_tokens


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


if __name__ == "__main__":
    import os, json

    os.environ.setdefault("MODEL_PATH", "gpt2")  # swap for your jev checkpoint
    os.environ.setdefault("DEVICE", "cpu")
    os.environ.setdefault("DTYPE", "float32")

    from config import Config
    import config as _cfg
    _cfg.config = Config()

    eng = Engine()
    answers, tokens = eng.run(
        state="My order #1234 hasn't arrived in two weeks and I'm very upset.",
        questions={
            "dept": ChoiceQuestion(
                type="choice",
                instructions="Which department should handle this?",
                criteria={"shipping": "delivery issues", "billing": "payment issues", "returns": "return requests"},
            ),
            "urgent": NoulQuestion(
                type="noul",
                instructions="Does this need urgent human attention?",
                criteria={"true": "customer is upset or waiting", "false": "routine enquiry"},
            ),
            "frustration": ScoreQuestion(
                type="score",
                instructions="How frustrated is the customer?",
                criteria=["Calm", "Mildly frustrated", "Very frustrated"],
            ),
        },
    )
    print(json.dumps({k: v.model_dump() for k, v in answers.items()}, indent=2))
    print(f"tokens: {tokens}")
