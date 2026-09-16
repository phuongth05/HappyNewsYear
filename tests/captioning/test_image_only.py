from dataclasses import fields

import pytest

from kric.captioning.blip import BlipImageOnlyCaptioner
from kric.captioning.types import ImageOnlyInput


def test_image_only_contract_has_no_article_or_reference_fields() -> None:
    assert [field.name for field in fields(ImageOnlyInput)] == ["sample_id", "image_path"]


def test_blip_batch_rejects_any_text_tokens() -> None:
    BlipImageOnlyCaptioner.assert_image_only_batch({"pixel_values": object()})
    with pytest.raises(RuntimeError, match="forbidden text inputs"):
        BlipImageOnlyCaptioner.assert_image_only_batch(
            {"pixel_values": object(), "input_ids": object()}
        )

