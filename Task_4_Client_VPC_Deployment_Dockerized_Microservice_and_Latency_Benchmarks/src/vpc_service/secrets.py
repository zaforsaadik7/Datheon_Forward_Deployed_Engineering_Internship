"""Hardened environment and Docker-Compose file secret loader with fail-closed production checks and redaction."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
from typing import Any, Dict, Optional, Tuple

from src.vpc_service.auth import (
    DEFAULT_DEV_VPC_KEYS,
    DEFAULT_DEV_VPC_PRINCIPALS,
    VPCPrincipal,
)

FORBIDDEN_PROD_PREFIXES = ("dev-key-", "key-", "REPLACE_WITH_", "test-", "changeme")


class SecretConfigurationError(RuntimeError):
    """Raised when production secrets are missing, malformed, weak, or left at dev defaults."""


@dataclass(frozen=True)
class SecretBundle:
    """Loaded secret material with redaction guarantees on repr(), str(), and serialization."""
    env: str
    api_key_map: Dict[str, VPCPrincipal]
    metrics_token: Optional[str] = None
    openai_api_key: Optional[str] = None
    source: str = "env"

    def __repr__(self) -> str:
        return (
            f"SecretBundle(env={self.env!r}, principals={len(self.api_key_map)}, "
            f"metrics_token={'***REDACTED***' if self.metrics_token else None}, "
            f"openai_api_key={'***REDACTED***' if self.openai_api_key else None}, "
            f"source={self.source!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def to_redacted_dict(self) -> Dict[str, Any]:
        return {
            "env": self.env,
            "source": self.source,
            "principal_count": len(self.api_key_map),
            "principals": sorted(p.principal_id for p in self.api_key_map.values()),
            "metrics_token": "***REDACTED***" if self.metrics_token else None,
            "openai_api_key": "***REDACTED***" if self.openai_api_key else None,
        }


def _read_secret_file(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise SecretConfigurationError(f"Configured secret file does not exist: {p.name}")
    content = p.read_text(encoding="utf-8").strip()
    if not content:
        raise SecretConfigurationError(f"Configured secret file is empty: {p.name}")
    return content


def _validate_production_key(token: str) -> None:
    if len(token) < 20 or any(token.startswith(pref) for pref in FORBIDDEN_PROD_PREFIXES):
        raise SecretConfigurationError(
            "Refusing to start in production mode: API key is too short (<20 chars), weak, "
            "or matches a known development/placeholder token."
        )


def _parse_api_keys_json(raw_json: str, is_prod: bool) -> Dict[str, VPCPrincipal]:
    try:
        parsed = json.loads(raw_json)
    except Exception as exc:
        raise SecretConfigurationError("Malformed JSON in FDE_API_KEYS_JSON.") from exc

    if not isinstance(parsed, dict) or not parsed:
        raise SecretConfigurationError("FDE_API_KEYS_JSON must be a non-empty JSON object.")

    mapping: Dict[str, VPCPrincipal] = {}
    for raw_token, spec in parsed.items():
        token = str(raw_token).strip()
        if not token:
            continue
        if is_prod:
            _validate_production_key(token)

        if isinstance(spec, str):
            # Shorthand mapping to a known persona name
            if spec not in DEFAULT_DEV_VPC_PRINCIPALS:
                raise SecretConfigurationError(f"Unknown principal reference '{spec}' in FDE_API_KEYS_JSON.")
            mapping[token] = DEFAULT_DEV_VPC_PRINCIPALS[spec]
        elif isinstance(spec, dict):
            mapping[token] = VPCPrincipal(
                principal_id=str(spec["principal_id"]),
                tenant_id=str(spec["tenant_id"]),
                clearance_tier=int(spec["clearance_tier"]),
                departments=tuple(str(d) for d in spec.get("departments", ())),
                agent_role=spec.get("agent_role", "operator"),
            )
        else:
            raise SecretConfigurationError("Invalid principal entry in FDE_API_KEYS_JSON.")

    if not mapping:
        raise SecretConfigurationError("No valid API keys parsed from FDE_API_KEYS_JSON.")
    return mapping


def load_secret_bundle(
    *,
    env: Optional[str] = None,
    api_keys_file: Optional[Path] = None,
    metrics_token_file: Optional[Path] = None,
) -> SecretBundle:
    """
    Load API keys and scrape tokens from Docker/K8s secret files (*_FILE) or environment variables.
    In FDE_ENV=production, fails closed if FDE_API_KEYS_JSON / FDE_API_KEYS_JSON_FILE is unset or uses
    weak/development keys.
    """
    resolved_env = (env or os.environ.get("FDE_ENV", "development")).strip().lower()
    is_prod = resolved_env == "production"

    eff_keys_file = api_keys_file
    if eff_keys_file is None:
        env_kf = os.environ.get("FDE_API_KEYS_JSON_FILE", "").strip()
        if env_kf:
            eff_keys_file = Path(env_kf)

    eff_metrics_file = metrics_token_file
    if eff_metrics_file is None:
        env_mf = os.environ.get("FDE_METRICS_TOKEN_FILE", "").strip()
        if env_mf:
            eff_metrics_file = Path(env_mf)

    source = "env"
    raw_keys_json = _read_secret_file(eff_keys_file)
    if raw_keys_json is not None:
        source = "file"
    else:
        raw_keys_json = os.environ.get("FDE_API_KEYS_JSON", "").strip() or None

    if raw_keys_json is not None:
        key_map = _parse_api_keys_json(raw_keys_json, is_prod=is_prod)
    else:
        if is_prod:
            raise SecretConfigurationError(
                "Refusing to start in production mode (FDE_ENV=production): "
                "neither FDE_API_KEYS_JSON_FILE nor FDE_API_KEYS_JSON was provided."
            )
        source = "dev_fallback"
        key_map = {
            token: DEFAULT_DEV_VPC_PRINCIPALS[p_id]
            for token, p_id in DEFAULT_DEV_VPC_KEYS.items()
        }

    metrics_token = _read_secret_file(eff_metrics_file)
    if metrics_token is None:
        metrics_token = os.environ.get("FDE_METRICS_TOKEN", "").strip() or None
    if is_prod and metrics_token is not None:
        _validate_production_key(metrics_token)

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip() or None

    return SecretBundle(
        env=resolved_env,
        api_key_map=key_map,
        metrics_token=metrics_token,
        openai_api_key=openai_key,
        source=source,
    )


def init_dev_secret_files(secrets_dir: Path, overwrite: bool = False) -> Tuple[Path, Path]:
    """
    Generate cryptographically random secret files (`fde_api_keys.json` and `fde_metrics_token.txt`)
    using `secrets.token_urlsafe(32)` with 0o644 container-readable permissions so Docker Compose
    and local tests pass strict production secret checks without committing secrets to git.
    """
    target_dir = Path(secrets_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    keys_path = target_dir / "fde_api_keys.json"
    metrics_path = target_dir / "fde_metrics_token.txt"

    if overwrite or not keys_path.exists():
        payload: Dict[str, Any] = {}
        for p_id, principal in DEFAULT_DEV_VPC_PRINCIPALS.items():
            random_key = f"vpc_{p_id}_{secrets.token_urlsafe(24)}"
            payload[random_key] = {
                "principal_id": principal.principal_id,
                "tenant_id": principal.tenant_id,
                "clearance_tier": principal.clearance_tier,
                "departments": list(principal.departments),
                "agent_role": principal.agent_role,
            }
        keys_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:
            os.chmod(keys_path, 0o644)
        except OSError:
            pass

    if overwrite or not metrics_path.exists():
        token = f"prom_scrape_{secrets.token_urlsafe(28)}"
        metrics_path.write_text(token + "\n", encoding="utf-8")
        try:
            os.chmod(metrics_path, 0o644)
        except OSError:
            pass

    return keys_path, metrics_path
