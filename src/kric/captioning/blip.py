"""Image-only BLIP caption generator."""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .config import GenerationConfig, ModelConfig
from .types import GeneratedCaption, ImageOnlyInput


class BlipImageOnlyCaptioner:
    """Unconditional BLIP generation with an enforced pixel-only model call."""

    _TEXT_KEYS = frozenset({"input_ids", "attention_mask", "token_type_ids"})

    def __init__(self, model: ModelConfig, generation: GenerationConfig, seed: int):
        self.config = model
        self.generation = generation
        self.seed = seed
        self.processor: Any | None = None
        self.model: Any | None = None
        self.device = "unloaded"
        self.resolved_revision: str | None = None
        self.load_seconds = 0.0

    @classmethod
    def assert_image_only_batch(cls, batch: Any) -> None:
        keys = set(batch.keys())
        forbidden = sorted(keys & cls._TEXT_KEYS)
        if forbidden:
            raise RuntimeError(f"B0 processor produced forbidden text inputs: {forbidden}")
        if "pixel_values" not in keys:
            raise RuntimeError("B0 processor did not produce pixel_values")

    def load(self) -> None:
        started = time.perf_counter()
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

        self.device = (
            "cuda" if self.config.device == "auto" and torch.cuda.is_available()
            else "cpu" if self.config.device == "auto"
            else self.config.device
        )
        dtype_by_name = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.config.dtype not in dtype_by_name:
            raise ValueError(f"unsupported dtype: {self.config.dtype}")
        if self.device == "cpu" and self.config.dtype == "float16":
            raise ValueError("float16 BLIP inference is not supported on CPU")

        self.processor = BlipProcessor.from_pretrained(
            self.config.name, revision=self.config.revision
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.config.name,
            revision=self.config.revision,
            torch_dtype=dtype_by_name[self.config.dtype],
        ).to(self.device)
        self.model.eval()
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)
        self.load_seconds = time.perf_counter() - started

    def generate(self, inputs: Sequence[ImageOnlyInput]) -> list[GeneratedCaption]:
        if self.model is None or self.processor is None:
            self.load()
        import torch
        from PIL import Image

        results: list[GeneratedCaption] = []
        for offset in range(0, len(inputs), self.config.batch_size):
            batch_inputs = inputs[offset : offset + self.config.batch_size]
            images = []
            try:
                for item in batch_inputs:
                    with Image.open(Path(item.image_path)) as source:
                        images.append(source.convert("RGB"))
                encoded = self.processor(images=images, return_tensors="pt")
                self.assert_image_only_batch(encoded)
                pixel_values = encoded["pixel_values"].to(self.device)
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        pixel_values=pixel_values,
                        **self.generation.to_generate_kwargs(),
                    )
                texts = self.processor.batch_decode(output_ids, skip_special_tokens=True)
                results.extend(
                    GeneratedCaption(item.sample_id, text.strip())
                    for item, text in zip(batch_inputs, texts, strict=True)
                )
            finally:
                for image in images:
                    image.close()
        return results

    def model_info(self) -> dict[str, Any]:
        parameter_count = (
            sum(parameter.numel() for parameter in self.model.parameters())
            if self.model is not None
            else None
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        return {
            "type": self.config.type,
            "name": self.config.name,
            "requested_revision": self.config.revision,
            "resolved_revision": self.resolved_revision,
            "class": type(self.model).__name__ if self.model is not None else None,
            "processor_class": type(self.processor).__name__ if self.processor is not None else None,
            "parameter_count": parameter_count,
            "device": self.device,
            "dtype": self.config.dtype,
            "transformers_version": version("transformers"),
            "torch_version": version("torch"),
            "tokenizer": {
                "bos_token_id": getattr(tokenizer, "bos_token_id", None),
                "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                "pad_token_id": getattr(tokenizer, "pad_token_id", None),
                "model_max_length": getattr(tokenizer, "model_max_length", None),
            },
        }
