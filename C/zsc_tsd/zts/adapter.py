from __future__ import annotations

import gc
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .logits_memo import ExactInputLogitsMemo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = PROJECT_ROOT / "legacy" / "zero-shot-GLS"
LEGACY_SRC = LEGACY_ROOT / "src"
COMPAT_ROOT = PROJECT_ROOT / "zsc_tsd" / "compat"

DEFAULT_MODEL_DIR = (
    LEGACY_ROOT
    / "tmp_saves"
    / "local_models"
    / "llama-2-7b-chat-gptq-4bit-32g-actorder"
)


@dataclass(frozen=True, slots=True)
class ZTSConfig:
    """Parameters shared by the ZTS sender and receiver."""

    prompt_seed: int = 2023
    deterministic_seed: int = 42

    egs_mode: str = "huffman"
    threshold: float = 0.002
    temperature: float = 1.0
    temperature_alpha: float = 1.25
    max_bpw: int = 5
    max_new_tokens: int = 160

    def __post_init__(self) -> None:
        if self.egs_mode not in {"huffman", "block"}:
            raise ValueError(
                f"Unsupported EGS mode: {self.egs_mode!r}"
            )

        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        if self.temperature_alpha <= 0:
            raise ValueError(
                "temperature_alpha must be greater than zero"
            )

        if self.max_bpw <= 0:
            raise ValueError("max_bpw must be greater than zero")

        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero"
            )


@dataclass(frozen=True, slots=True)
class LegacyModules:
    """References to the original zero-shot-GLS implementation."""

    codec: Any
    hide_extract: Any
    prompt_gen: Any
    p_utils: Any


def _prepend_sys_path(path: Path) -> None:
    """Insert a local source directory without adding duplicates."""

    resolved = str(path.resolve())

    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def load_legacy_modules() -> LegacyModules:
    """
    Load the original zero-shot-GLS modules.

    The Windows-compatible zgls_utils implementation is inserted before
    importing the legacy source tree.
    """

    if not LEGACY_SRC.is_dir():
        raise FileNotFoundError(
            f"Legacy source directory does not exist: {LEGACY_SRC}"
        )

    if not COMPAT_ROOT.is_dir():
        raise FileNotFoundError(
            f"Compatibility directory does not exist: {COMPAT_ROOT}"
        )

    # insert(0) is used twice, so COMPAT_ROOT ends up before LEGACY_SRC.
    _prepend_sys_path(LEGACY_SRC)
    _prepend_sys_path(COMPAT_ROOT)

    return LegacyModules(
        codec=importlib.import_module("codec"),
        hide_extract=importlib.import_module("hide_extract"),
        prompt_gen=importlib.import_module("prompt_gen"),
        p_utils=importlib.import_module("p_utils"),
    )


class ZTSModelAdapter:
    """
    Lazy-loading adapter around the legacy Llama-2 GPTQ checkpoint.

    Heavy GPU dependencies are imported only when load() is called. This
    keeps the lightweight zsc-tsd-root environment able to import the
    package without installing PyTorch, Transformers, or GPTQModel.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        device: str = "cuda:0",
        config: ZTSConfig | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.device = device
        self.config = config or ZTSConfig()

        self._legacy: LegacyModules | None = None
        self._tokenizer: Any | None = None
        self._wrapper: Any | None = None
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def legacy(self) -> LegacyModules:
        if self._legacy is None:
            self._legacy = load_legacy_modules()

        return self._legacy

    @property
    def tokenizer(self) -> Any:
        self.load()
        return self._tokenizer

    @property
    def model(self) -> Any:
        self.load()
        return self._model

    @property
    def wrapper(self) -> Any:
        self.load()
        return self._wrapper

    def validate_files(self) -> None:
        required_files = (
            "config.json",
            "model.safetensors",
            "quantize_config.json",
            "tokenizer.model",
            "tokenizer_config.json",
        )

        missing = [
            name
            for name in required_files
            if not (self.model_dir / name).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                "The local ZTS model directory is incomplete. "
                f"Missing files: {missing}. "
                f"Directory: {self.model_dir}"
            )

    def load(self) -> "ZTSModelAdapter":
        if self.is_loaded:
            return self

        self.validate_files()

        # Deliberately lazy imports: the root integration environment does
        # not contain these GPU packages.
        import torch
        from gptqmodel import BACKEND, GPTQModel
        from transformers import AutoTokenizer

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available, but the ZTS adapter was configured "
                f"for device {self.device!r}."
            )

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
            use_fast=False,
            legacy=True,
        )

        wrapper = GPTQModel.load(
            str(self.model_dir),
            device=self.device,
            backend=BACKEND.TORCH,
            trust_remote_code=False,
            local_files_only=True,
        )

        model = getattr(wrapper, "model", wrapper).eval()

        # The original zero-shot-GLS functions read model.vocab_size
        # directly, while modern Transformers stores it in model.config.
        if not hasattr(model, "vocab_size"):
            model.vocab_size = model.config.vocab_size

        self._tokenizer = tokenizer
        self._wrapper = wrapper
        self._model = ExactInputLogitsMemo(model)

        return self

    def configure_determinism(self) -> None:
        import torch

        self.legacy.p_utils.seed_everything(
            self.config.deterministic_seed,
            deterministic=True,
            warn_only=True,
        )

        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )

    def normalize_prompt(self, prompt: str) -> str:
        """
        Return the tokenizer-normalized prompt text.

        Encoding the complete prompt-plus-continuation text from this
        normalized boundary preserves SentencePiece whitespace tokens.
        """

        tokenizer = self.tokenizer

        prompt_ids = tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids

        return tokenizer.decode(
            prompt_ids[0].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def release(self) -> None:
        """Release model references and cached CUDA memory."""

        self._model = None
        self._wrapper = None
        self._tokenizer = None

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass