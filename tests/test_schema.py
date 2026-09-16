from dataclasses import fields

from kric.data.schema import DatasetSample, InferenceSample


def test_canonical_schema_and_inference_boundary() -> None:
    sample = DatasetSample(
        sample_id="story_0",
        image_path="/images/story_0.jpg",
        article_text="An article.",
        article_sentences=("An article.",),
        reference_caption="A target caption.",
        entities=["Target-only entity"],
        metadata={"official_split": "train"},
    )

    assert set(sample.to_dict()) == {
        "sample_id",
        "image_path",
        "article_text",
        "article_sentences",
        "reference_caption",
        "entities",
        "metadata",
    }

    inference = sample.to_inference_sample()
    assert isinstance(inference, InferenceSample)
    assert "reference_caption" not in inference.to_dict()
    assert "entities" not in inference.to_dict()
    assert "reference_caption" not in {item.name for item in fields(InferenceSample)}


def test_schema_rejects_non_string_caption() -> None:
    try:
        DatasetSample(
            sample_id="story_0",
            image_path="image.jpg",
            article_text="Article",
            article_sentences=("Article",),
            reference_caption=None,  # type: ignore[arg-type]
        )
    except TypeError as error:
        assert "reference_caption must be str" in str(error)
    else:
        raise AssertionError("invalid schema was accepted")

