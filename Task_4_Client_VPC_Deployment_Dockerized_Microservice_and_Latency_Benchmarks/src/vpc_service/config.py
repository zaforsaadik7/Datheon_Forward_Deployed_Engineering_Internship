"""VPC Microservice runtime settings loaded from environment variables."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal, Optional

from src.config import BASE_DIR


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class VPCSettings:
    """Immutable runtime configuration for the Client VPC FastAPI Microservice."""
    env: str = "development"
    airgapped_mode: bool = True
    planner_mode: Literal["rule", "openai"] = "rule"
    auto_seed_on_startup: bool = True
    allow_sample_seed_in_prod: bool = False
    max_body_bytes: int = 1_048_576
    rate_limit_rps: float = 500.0
    rate_limit_burst: int = 1000
    var_dir: Path = BASE_DIR / "var"
    sample_docs_dir: Path = BASE_DIR / "data" / "sample_documents"
    erp_seed_json: Path = BASE_DIR / "data" / "seed_enterprise_data.json"
    api_keys_file: Optional[Path] = None
    metrics_token_file: Optional[Path] = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"


def load_vpc_settings(
    *,
    env: Optional[str] = None,
    var_dir: Optional[Path] = None,
    airgapped_mode: Optional[bool] = None,
    planner_mode: Optional[str] = None,
    auto_seed_on_startup: Optional[bool] = None,
    allow_sample_seed_in_prod: Optional[bool] = None,
    max_body_bytes: Optional[int] = None,
    rate_limit_rps: Optional[float] = None,
    rate_limit_burst: Optional[int] = None,
    api_keys_file: Optional[Path] = None,
    metrics_token_file: Optional[Path] = None,
) -> VPCSettings:
    """Resolve VPCSettings from explicit overrides or environment variables."""
    resolved_env = (env or os.environ.get("FDE_ENV", "development")).strip().lower()
    resolved_airgap = (
        airgapped_mode
        if airgapped_mode is not None
        else _env_bool("FDE_AIRGAPPED_MODE", True)
    )
    raw_planner = (planner_mode or os.environ.get("FDE_PLANNER", "rule")).strip().lower()
    # In air-gapped VPC mode, force deterministic rule planner so zero outbound LLM calls occur
    effective_planner: Literal["rule", "openai"] = (
        "rule" if resolved_airgap or raw_planner != "openai" else "openai"
    )

    resolved_auto_seed = (
        auto_seed_on_startup
        if auto_seed_on_startup is not None
        else _env_bool("FDE_AUTO_SEED", True)
    )
    resolved_allow_prod_seed = (
        allow_sample_seed_in_prod
        if allow_sample_seed_in_prod is not None
        else _env_bool("FDE_ALLOW_SAMPLE_SEED", False)
    )
    resolved_max_body = (
        max_body_bytes
        if max_body_bytes is not None
        else int(os.environ.get("FDE_MAX_BODY_BYTES", "1048576"))
    )
    resolved_rps = (
        rate_limit_rps
        if rate_limit_rps is not None
        else float(os.environ.get("FDE_RATE_LIMIT_RPS", "500.0"))
    )
    resolved_burst = (
        rate_limit_burst
        if rate_limit_burst is not None
        else int(os.environ.get("FDE_RATE_LIMIT_BURST", "1000"))
    )

    env_keys_file = os.environ.get("FDE_API_KEYS_JSON_FILE", "").strip()
    env_metrics_file = os.environ.get("FDE_METRICS_TOKEN_FILE", "").strip()

    return VPCSettings(
        env=resolved_env,
        airgapped_mode=resolved_airgap,
        planner_mode=effective_planner,
        auto_seed_on_startup=resolved_auto_seed,
        allow_sample_seed_in_prod=resolved_allow_prod_seed,
        max_body_bytes=resolved_max_body,
        rate_limit_rps=resolved_rps,
        rate_limit_burst=resolved_burst,
        var_dir=Path(var_dir) if var_dir is not None else BASE_DIR / "var",
        api_keys_file=api_keys_file or (Path(env_keys_file) if env_keys_file else None),
        metrics_token_file=metrics_token_file or (Path(env_metrics_file) if env_metrics_file else None),
        llm_base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        llm_model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    )
