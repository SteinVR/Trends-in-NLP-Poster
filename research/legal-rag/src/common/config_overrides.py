"""Environment override helpers for application config."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .schemas import Phase

if TYPE_CHECKING:
    from .config import AppConfig


def apply_env_overrides(config: "AppConfig", env: Mapping[str, str]) -> "AppConfig":
    """Apply supported environment overrides to the loaded application config."""

    phase_override = env.get("COMPETITION_PHASE")
    base_url_override = env.get("COMPETITION_BASE_URL") or env.get("EVAL_BASE_URL")
    data_root_override = env.get("COMPETITION_DATA_ROOT")
    timeout_override = env.get("COMPETITION_TIMEOUT_SECONDS")
    extract_override = env.get("COMPETITION_EXTRACT_DOCUMENTS")
    overwrite_override = env.get("COMPETITION_OVERWRITE_DOWNLOADS")
    unanswerable_policy_override = env.get("COMPETITION_UNANSWERABLE_RETRIEVAL_POLICY")
    enable_ocr_override = env.get("COMPETITION_ENABLE_OCR")
    ocr_unavailable_policy_override = env.get("COMPETITION_OCR_UNAVAILABLE_POLICY")
    ocr_device_override = env.get("COMPETITION_OCR_DEVICE")
    ocr_enable_hpi_override = env.get("COMPETITION_OCR_ENABLE_HPI")
    ocr_enable_mkldnn_override = env.get("COMPETITION_OCR_ENABLE_MKLDNN")
    ocr_cpu_threads_override = env.get("COMPETITION_OCR_CPU_THREADS")
    ocr_render_dpi_override = env.get("COMPETITION_OCR_RENDER_DPI")
    ocr_text_detection_limit_side_len_override = env.get("COMPETITION_OCR_TEXT_DETECTION_LIMIT_SIDE_LEN")
    ocr_text_recognition_batch_size_override = env.get("COMPETITION_OCR_TEXT_RECOGNITION_BATCH_SIZE")
    ocr_model_root_dir_override = env.get("COMPETITION_OCR_MODEL_ROOT_DIR")
    ocr_text_detection_model_dir_override = env.get("COMPETITION_OCR_TEXT_DETECTION_MODEL_DIR")
    ocr_text_recognition_model_dir_override = env.get("COMPETITION_OCR_TEXT_RECOGNITION_MODEL_DIR")
    enable_semantic_table_blocks_override = env.get("COMPETITION_ENABLE_SEMANTIC_TABLE_BLOCKS")
    pass_a_enabled_override = env.get("COMPETITION_PASS_A_ENABLED")
    page_validation_mode_override = env.get("COMPETITION_PAGE_VALIDATION_MODE")
    require_gpu_override = env.get("COMPETITION_REQUIRE_GPU")
    decomposition_enabled_override = env.get("COMPETITION_DECOMPOSITION_ENABLED")
    log_level_override = env.get("COMPETITION_LOG_LEVEL")
    log_directory_override = env.get("COMPETITION_LOG_DIRECTORY")
    debug_documents_override = env.get("COMPETITION_DEBUG_DOCUMENTS")
    debug_questions_override = env.get("COMPETITION_DEBUG_QUESTIONS")

    if phase_override:
        config = config.model_copy(update={"phase": Phase(phase_override.strip().lower())})

    if base_url_override:
        config = config.model_copy(
            update={"api": config.api.model_copy(update={"base_url": base_url_override.strip()})}
        )

    if data_root_override:
        config = config.model_copy(
            update={"storage": config.storage.model_copy(update={"root_dir": Path(data_root_override.strip())})}
        )

    if timeout_override:
        config = config.model_copy(
            update={"api": config.api.model_copy(update={"timeout_seconds": float(timeout_override.strip())})}
        )

    if extract_override:
        config = config.model_copy(
            update={"download": config.download.model_copy(update={"extract_documents": parse_bool(extract_override)})}
        )

    if overwrite_override:
        config = config.model_copy(
            update={"download": config.download.model_copy(update={"overwrite": parse_bool(overwrite_override)})}
        )

    if unanswerable_policy_override:
        config = config.model_copy(
            update={
                "evaluation": config.evaluation.model_copy(
                    update={"unanswerable_retrieval_policy": unanswerable_policy_override.strip()}
                )
            }
        )

    ingestion_updates: dict[str, Any] = {}
    if enable_ocr_override:
        ingestion_updates["enable_ocr"] = parse_bool(enable_ocr_override)
    if ocr_unavailable_policy_override:
        ingestion_updates["ocr_unavailable_policy"] = ocr_unavailable_policy_override.strip()
    if ocr_device_override:
        ingestion_updates["ocr_device"] = ocr_device_override.strip()
    if ocr_enable_hpi_override:
        ingestion_updates["ocr_enable_hpi"] = parse_bool(ocr_enable_hpi_override)
    if ocr_enable_mkldnn_override:
        ingestion_updates["ocr_enable_mkldnn"] = parse_bool(ocr_enable_mkldnn_override)
    if ocr_cpu_threads_override:
        ingestion_updates["ocr_cpu_threads"] = int(ocr_cpu_threads_override)
    if ocr_render_dpi_override:
        ingestion_updates["ocr_render_dpi"] = int(ocr_render_dpi_override)
    if ocr_text_detection_limit_side_len_override:
        ingestion_updates["ocr_text_detection_limit_side_len"] = int(ocr_text_detection_limit_side_len_override)
    if ocr_text_recognition_batch_size_override:
        ingestion_updates["ocr_text_recognition_batch_size"] = int(ocr_text_recognition_batch_size_override)
    if ocr_model_root_dir_override:
        ingestion_updates["ocr_model_root_dir"] = Path(ocr_model_root_dir_override.strip()).expanduser()
    if ocr_text_detection_model_dir_override:
        ingestion_updates["ocr_text_detection_model_dir"] = Path(
            ocr_text_detection_model_dir_override.strip()
        ).expanduser()
    if ocr_text_recognition_model_dir_override:
        ingestion_updates["ocr_text_recognition_model_dir"] = Path(
            ocr_text_recognition_model_dir_override.strip()
        ).expanduser()
    if enable_semantic_table_blocks_override:
        ingestion_updates["enable_semantic_table_blocks"] = parse_bool(enable_semantic_table_blocks_override)

    if ingestion_updates:
        config = config.model_copy(update={"ingestion": config.ingestion.model_copy(update=ingestion_updates)})

    page_attribution_updates: dict[str, Any] = {}
    if pass_a_enabled_override:
        page_attribution_updates["pass_a_enabled"] = parse_bool(pass_a_enabled_override)
    if page_validation_mode_override:
        page_attribution_updates["validation_mode"] = parse_page_validation_mode(page_validation_mode_override)
    if page_attribution_updates:
        config = config.model_copy(
            update={"page_attribution": config.page_attribution.model_copy(update=page_attribution_updates)}
        )

    if require_gpu_override:
        config = config.model_copy(
            update={"runtime": config.runtime.model_copy(update={"require_gpu": parse_bool(require_gpu_override)})}
        )

    if decomposition_enabled_override:
        config = config.model_copy(
            update={
                "decomposition": config.decomposition.model_copy(
                    update={"enabled": parse_bool(decomposition_enabled_override)}
                )
            }
        )

    if log_level_override:
        config = config.model_copy(
            update={"logging": config.logging.model_copy(update={"level": log_level_override.strip().upper()})}
        )

    if log_directory_override:
        config = config.model_copy(
            update={"logging": config.logging.model_copy(update={"directory": Path(log_directory_override.strip())})}
        )

    if debug_documents_override:
        config = config.model_copy(
            update={
                "logging": config.logging.model_copy(
                    update={"debug_documents": parse_bool(debug_documents_override)}
                )
            }
        )

    if debug_questions_override:
        config = config.model_copy(
            update={
                "logging": config.logging.model_copy(
                    update={"debug_questions": parse_bool(debug_questions_override)}
                )
            }
        )

    return config


def parse_bool(raw_value: str) -> bool:
    """Parse an environment boolean override."""

    normalized = raw_value.strip().lower()
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}

    if normalized in truthy:
        return True
    if normalized in falsy:
        return False

    from .config import ConfigError

    raise ConfigError(f"Expected a boolean-like environment override, got: {raw_value!r}")


def parse_page_validation_mode(raw_value: str) -> str:
    """Normalize and validate page validation mode overrides."""

    normalized = re.sub(r"[\s-]+", "_", raw_value.strip().lower())
    allowed_values = {"degrade_first", "strict_suppress"}
    if normalized in allowed_values:
        return normalized

    from .config import ConfigError

    raise ConfigError(
        "Invalid COMPETITION_PAGE_VALIDATION_MODE override. "
        f"Expected one of {sorted(allowed_values)}, got: {raw_value!r}"
    )
