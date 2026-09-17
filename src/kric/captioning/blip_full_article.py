"""Full-article BLIP baseline with explicit token accounting."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .blip import BlipImageOnlyCaptioner
from .config import FullArticleContextConfig, GenerationConfig, ModelConfig
from .types import FullArticleInput, GeneratedCaption


class BlipFullArticleCaptioner(BlipImageOnlyCaptioner):
    """Use article tokens as the sole decoder prefix added to the B0 model."""

    def __init__(
        self,
        model: ModelConfig,
        generation: GenerationConfig,
        context: FullArticleContextConfig,
        seed: int,
    ):
        super().__init__(model, generation, seed)
        self.context = context

    def load(self) -> None:
        super().load()
        tokenizer = self.processor.tokenizer
        text_config = getattr(self.model.config, "text_config", None)
        capacity = getattr(text_config, "max_position_embeddings", None)
        if capacity is not None:
            special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
            required = (
                self.context.max_article_tokens
                + special_tokens
                + self.generation.max_new_tokens
            )
            if required > capacity:
                raise ValueError(
                    "context and generation budgets exceed BLIP text capacity: "
                    f"{required} > {capacity}"
                )

    def _encode_article(self, article: str) -> tuple[dict[str, Any], dict[str, float | int]]:
        tokenizer = self.processor.tokenizer
        original_ids = tokenizer.encode(article, add_special_tokens=False)
        used_ids = original_ids[: self.context.max_article_tokens]
        encoded = tokenizer.prepare_for_model(
            used_ids,
            add_special_tokens=True,
            return_attention_mask=True,
            truncation=False,
        )
        original_count = len(original_ids)
        used_count = len(used_ids)
        truncation_ratio = (
            (original_count - used_count) / original_count if original_count else 0.0
        )
        return encoded, {
            "original_article_tokens": original_count,
            "used_article_tokens": used_count,
            "truncation_ratio": truncation_ratio,
            "max_context_tokens": self.context.max_context_tokens,
            "generation_tokens": self.generation.max_new_tokens,
        }

    def generate(self, inputs: Sequence[FullArticleInput]) -> list[GeneratedCaption]:
        if self.model is None or self.processor is None:
            self.load()
        import torch
        from PIL import Image

        tokenizer = self.processor.tokenizer
        results: list[GeneratedCaption] = []
        for offset in range(0, len(inputs), self.config.batch_size):
            batch_inputs = inputs[offset : offset + self.config.batch_size]
            images = []
            try:
                encoded_articles = []
                context_stats = []
                for item in batch_inputs:
                    article, stats = self._encode_article(item.article_text)
                    encoded_articles.append(article)
                    context_stats.append(stats)
                    with Image.open(Path(item.image_path)) as source:
                        images.append(source.convert("RGB"))

                image_batch = self.processor(images=images, return_tensors="pt")
                self.assert_image_only_batch(image_batch)
                text_batch = tokenizer.pad(
                    encoded_articles, padding=True, return_tensors="pt"
                )
                input_ids = text_batch["input_ids"].to(self.device)
                attention_mask = text_batch["attention_mask"].to(self.device)
                pixel_values = image_batch["pixel_values"].to(self.device)
                # BLIP's generate() passes input_ids[:, :-1] to its decoder,
                # removing the terminal separator inserted above.
                decoder_prefix_width = input_ids.shape[1] - 1

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **self.generation.to_generate_kwargs(),
                    )
                generated_ids = output_ids[:, decoder_prefix_width:]
                texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                results.extend(
                    GeneratedCaption(item.sample_id, text.strip(), stats)
                    for item, text, stats in zip(
                        batch_inputs, texts, context_stats, strict=True
                    )
                )
            finally:
                for image in images:
                    image.close()
        return results

    def model_info(self) -> dict[str, Any]:
        info = super().model_info()
        info["context_policy"] = {
            "source": "article_text",
            "max_context_tokens": self.context.max_context_tokens,
            "truncation": self.context.truncation,
            "framing": "raw article decoder prefix; no added instruction",
            "status": "exploratory_legacy",
            "truncation_ratio_definition": "(original-used)/original",
        }
        return info
