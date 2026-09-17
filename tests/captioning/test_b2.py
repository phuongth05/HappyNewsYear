from pathlib import Path
from types import SimpleNamespace

from kric.captioning.config import (
    B2ExperimentConfig,
    FullArticleContextConfig,
    GenerationConfig,
    ModelConfig,
    PromptConfig,
)
from kric.captioning.instructblip import InstructBlipArticleCaptioner
from kric.captioning.runner import _assert_b1_b2_comparable, _select_b2_context


class WordTokenizer:
    model_max_length = 512

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


class FakeProcessor:
    tokenizer = WordTokenizer()
    qformer_tokenizer = WordTokenizer()


def _captioner(max_context_tokens=6):
    captioner = InstructBlipArticleCaptioner(
        ModelConfig("instructblip", "model", "revision", dtype="float16"),
        GenerationConfig(),
        PromptConfig(
            instruction="Describe the image in one factual news-style sentence.",
            context_prefix="Article context:\n",
            context_suffix="\n\n",
        ),
        FullArticleContextConfig(max_context_tokens=max_context_tokens),
        2026,
    )
    captioner.processor = FakeProcessor()
    captioner.language_max_positions = 512
    captioner.qformer_max_positions = 512
    captioner.num_query_tokens = 32
    return captioner


def test_b2_selection_ranks_first_but_passes_context_in_article_order() -> None:
    sample = SimpleNamespace(
        article_text="one two three four five six seven eight nine ten",
        article_sentences=("named person", "irrelevant detail", "visual person"),
    )
    ranking = {
        "ranked_sentences": [
            {"sentence_id": 2, "text": "visual person", "score": 0.9, "rank": 1},
            {"sentence_id": 0, "text": "named person", "score": 0.8, "rank": 2},
            {"sentence_id": 1, "text": "irrelevant detail", "score": 0.1, "rank": 3},
        ]
    }
    context, selection = _select_b2_context(_captioner(), sample, ranking, 2)
    assert context == "named person\nvisual person"
    assert selection["selected_sentence_ids"] == [0, 2]
    assert selection["selected_ranks"] == [2, 1]
    assert selection["context_token_count"] == 4
    assert selection["fraction_of_article_represented"] == 0.4


def test_b2_skips_sentence_that_would_exceed_budget() -> None:
    sample = SimpleNamespace(
        article_text="one two three four five six seven",
        article_sentences=("one two three four five six", "short fact"),
    )
    ranking = {
        "ranked_sentences": [
            {
                "sentence_id": 0,
                "text": "one two three four five six",
                "score": 1.0,
                "rank": 1,
            },
            {"sentence_id": 1, "text": "short fact", "score": 0.5, "rank": 2},
        ]
    }
    context, selection = _select_b2_context(_captioner(max_context_tokens=3), sample, ranking, 1)
    assert context == "short fact"
    assert selection["selected_sentence_ids"] == [1]
    assert selection["skipped_sentence_ids_for_budget"] == [0]


def test_all_six_b2_configs_are_controlled_against_b1() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "configs" / "experiments").glob("b2_instructblip_*.yaml"))
    assert len(paths) == 6
    observed = set()
    for path in paths:
        config = B2ExperimentConfig.from_file(path)
        baseline = _assert_b1_b2_comparable(config)
        assert baseline.model == config.model
        assert baseline.generation == config.generation
        assert baseline.prompt == config.prompt
        assert baseline.context == config.context
        observed.add((config.retrieval.method, config.retrieval.k))
    assert observed == {
        (method, k) for method in ("bm25", "semantic") for k in (1, 3, 5)
    }
