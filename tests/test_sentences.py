from kric.data.sentences import segment_sentences


def test_sentence_segmentation_handles_news_abbreviations_and_decimals() -> None:
    text = (
        "Dr. Rivera arrived in the U.S. at 9 a.m. She paid $3.50. "
        "Was the meeting useful? Yes!"
    )

    assert segment_sentences(text) == [
        "Dr. Rivera arrived in the U.S. at 9 a.m.",
        "She paid $3.50.",
        "Was the meeting useful?",
        "Yes!",
    ]


def test_sentence_segmentation_preserves_paragraph_boundaries() -> None:
    assert segment_sentences("First headline\nSecond line without punctuation") == [
        "First headline",
        "Second line without punctuation",
    ]
    assert segment_sentences("  ") == []
