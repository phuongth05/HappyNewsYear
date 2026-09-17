"""Controlled InstructBLIP image-only and article-conditioned generation."""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .config import FullArticleContextConfig, GenerationConfig, ModelConfig, PromptConfig
from .types import FullArticleInput, GeneratedCaption, ImageOnlyInput


class _InstructBlipCaptioner:
    """Shared model and prompt machinery for a controlled B0/B1 comparison."""

    def __init__(
        self,
        model: ModelConfig,
        generation: GenerationConfig,
        prompt: PromptConfig,
        context: FullArticleContextConfig,
        seed: int,
    ) -> None:
        self.config = model
        self.generation = generation
        self.prompt = prompt
        self.context = context
        self.seed = seed
        self.processor: Any | None = None
        self.model: Any | None = None
        self.device = "unloaded"
        self.torch_dtype: Any | None = None
        self.resolved_revision: str | None = None
        self.load_seconds = 0.0
        self.language_max_positions: int | None = None
        self.qformer_max_positions: int | None = None
        self.num_query_tokens = 0

    def load(self) -> None:
        started = time.perf_counter()
        import torch
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

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
            raise ValueError("float16 InstructBLIP inference is not supported on CPU")
        self.torch_dtype = dtype_by_name[self.config.dtype]

        self.processor = InstructBlipProcessor.from_pretrained(
            self.config.name, revision=self.config.revision
        )
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            self.config.name,
            revision=self.config.revision,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        if not getattr(self.model.config.text_config, "is_encoder_decoder", False):
            raise ValueError(
                "controlled baseline requires an encoder-decoder InstructBLIP checkpoint"
            )
        text_config = self.model.config.text_config
        self.language_max_positions = int(
            getattr(text_config, "n_positions", None)
            or getattr(text_config, "max_position_embeddings", 512)
        )
        self.qformer_max_positions = int(
            getattr(self.model.config.qformer_config, "max_position_embeddings", 512)
        )
        self.num_query_tokens = int(self.model.config.num_query_tokens)
        # Recent Transformers processors expand a dedicated image placeholder
        # into this many tokens. Set it explicitly so accounting and runtime
        # behavior cannot diverge on checkpoints with older processor metadata.
        self.processor.num_query_tokens = self.num_query_tokens
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)
        self.load_seconds = time.perf_counter() - started

    @staticmethod
    def _token_ids(tokenizer: Any, text: str) -> list[int]:
        # PreTrainedTokenizerFast.encode() warns when an intentionally
        # untruncated article is longer than model_max_length, even when the
        # IDs are used only for accounting and are truncated before model
        # input construction. The backend API performs the same unbounded
        # tokenization without implying that these IDs will be fed to a model.
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None:
            return list(backend.encode(text, add_special_tokens=False).ids)
        if hasattr(tokenizer, "tokenize") and hasattr(tokenizer, "convert_tokens_to_ids"):
            return list(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))
        return list(tokenizer.encode(text, add_special_tokens=False))

    def _fits(self, text: str) -> bool:
        tokenizer = self.processor.tokenizer
        qformer_tokenizer = self.processor.qformer_tokenizer
        language_limit = self.language_max_positions or int(tokenizer.model_max_length)
        qformer_limit = self.qformer_max_positions or int(qformer_tokenizer.model_max_length)
        language_tokens = len(self._token_ids(tokenizer, text)) + self.num_query_tokens
        qformer_tokens = len(self._token_ids(qformer_tokenizer, text))
        return language_tokens <= language_limit and qformer_tokens <= qformer_limit

    def _build_image_prompt(self) -> str:
        instruction = self.prompt.instruction.strip()
        if not instruction or not self._fits(instruction):
            raise ValueError("caption instruction is empty or exceeds the model context window")
        return instruction

    def _build_article_prompt(
        self, article: str
    ) -> tuple[str, dict[str, float | int]]:
        tokenizer = self.processor.tokenizer
        original_ids = self._token_ids(tokenizer, article)
        upper = min(len(original_ids), self.context.max_context_tokens)

        def render(count: int) -> str:
            used_article = tokenizer.decode(original_ids[:count], skip_special_tokens=True)
            return (
                f"{self.prompt.context_prefix}{used_article}"
                f"{self.prompt.context_suffix}{self.prompt.instruction.strip()}"
            )

        # Binary search the longest article prefix that fits both the language
        # encoder (including image query tokens) and the Q-Former tokenizer.
        low, high, best = 0, upper, -1
        while low <= high:
            middle = (low + high) // 2
            if self._fits(render(middle)):
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best < 0:
            raise ValueError("prompt framing and instruction exceed the model context window")

        original_count = len(original_ids)
        ratio = (original_count - best) / original_count if original_count else 0.0
        return render(best), {
            "original_article_tokens": original_count,
            "used_article_tokens": best,
            "truncation_ratio": ratio,
            "max_context_tokens": self.context.max_context_tokens,
            "generation_tokens": self.generation.max_new_tokens,
        }

    def _assert_encoded_input_lengths(self, encoded: dict[str, Any]) -> None:
        """Validate the processor's actual model-bound tensors before generation."""

        language_length = int(encoded["input_ids"].shape[-1])
        language_limit = self.language_max_positions or int(
            self.processor.tokenizer.model_max_length
        )
        if language_length > language_limit:
            raise RuntimeError(
                "final InstructBLIP language input exceeds its supported "
                f"sequence length: {language_length} > {language_limit}"
            )
        qformer_ids = encoded.get("qformer_input_ids")
        if qformer_ids is not None:
            qformer_length = int(qformer_ids.shape[-1])
            qformer_limit = self.qformer_max_positions or int(
                self.processor.qformer_tokenizer.model_max_length
            )
            if qformer_length > qformer_limit:
                raise RuntimeError(
                    "final InstructBLIP Q-Former text input exceeds its supported "
                    f"sequence length: {qformer_length} > {qformer_limit}"
                )

    def _generate_prompts(
        self,
        inputs: Sequence[ImageOnlyInput] | Sequence[FullArticleInput],
        prompts: Sequence[str],
        context_stats: Sequence[dict[str, float | int]],
    ) -> list[GeneratedCaption]:
        if self.model is None or self.processor is None:
            self.load()
        import torch
        from PIL import Image

        results: list[GeneratedCaption] = []
        for offset in range(0, len(inputs), self.config.batch_size):
            batch_inputs = inputs[offset : offset + self.config.batch_size]
            batch_prompts = prompts[offset : offset + self.config.batch_size]
            batch_stats = context_stats[offset : offset + self.config.batch_size]
            images = []
            try:
                for item in batch_inputs:
                    with Image.open(Path(item.image_path)) as source:
                        images.append(source.convert("RGB"))
                encoded = self.processor(
                    images=images,
                    text=list(batch_prompts),
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                self._assert_encoded_input_lengths(encoded)
                encoded = {
                    key: (
                        value.to(self.device, dtype=self.torch_dtype)
                        if key == "pixel_values"
                        else value.to(self.device)
                    )
                    for key, value in encoded.items()
                }
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **encoded, **self.generation.to_generate_kwargs()
                    )
                texts = self.processor.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                results.extend(
                    GeneratedCaption(item.sample_id, text.strip(), dict(stats))
                    for item, text, stats in zip(
                        batch_inputs, texts, batch_stats, strict=True
                    )
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
                "model_max_length": getattr(tokenizer, "model_max_length", None),
                "language_max_positions": self.language_max_positions,
                "qformer_max_positions": self.qformer_max_positions,
                "num_query_tokens": self.num_query_tokens,
            },
            "prompt": {
                "instruction": self.prompt.instruction,
                "context_prefix": self.prompt.context_prefix,
                "context_suffix": self.prompt.context_suffix,
            },
            "conditioning": "image query embeddings plus encoder text; generated text is not a continuation of the article",
        }


class InstructBlipImageOnlyCaptioner(_InstructBlipCaptioner):
    """B0: the narrow input type makes article/reference access impossible."""

    def generate(self, inputs: Sequence[ImageOnlyInput]) -> list[GeneratedCaption]:
        if self.model is None or self.processor is None:
            self.load()
        prompt = self._build_image_prompt()
        return self._generate_prompts(inputs, [prompt] * len(inputs), [{} for _ in inputs])


class InstructBlipArticleCaptioner(_InstructBlipCaptioner):
    """B1: image and article context condition the same model and instruction."""

    def generate(self, inputs: Sequence[FullArticleInput]) -> list[GeneratedCaption]:
        if self.model is None or self.processor is None:
            self.load()
        built = [self._build_article_prompt(item.article_text) for item in inputs]
        for item, (_, stats) in zip(inputs, built, strict=True):
            if int(stats["used_article_tokens"]) > self.context.max_context_tokens:
                raise RuntimeError(
                    f"context budget exceeded for {item.sample_id}: "
                    f"{stats['used_article_tokens']} > {self.context.max_context_tokens}"
                )
        prompts = [prompt for prompt, _ in built]
        stats = [item_stats for _, item_stats in built]
        return self._generate_prompts(inputs, prompts, stats)
