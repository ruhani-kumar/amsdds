from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# Repo root = one level above this file's package.
REPO_ROOT = Path(__file__).resolve().parent.parent

# The handover ML engine now owns preprocessing / Layer 1 / gate / Layer 2 /
# risk. Its config + code live under backend/ml_core/ (a lightly-adapted copy
# of handover/amsdds_backend/amsdds/backend - see backend/ml_core/README.md).
ML_CORE_DIR = REPO_ROOT / "backend" / "ml_core"
DEFAULT_THRESHOLDS_PATH = ML_CORE_DIR / "config" / "thresholds.yaml"

# Weights are local/external and never committed. Default location is the
# handover drop; override with AMSDDS_WEIGHTS.
DEFAULT_WEIGHTS_DIR = (
    REPO_ROOT / "handover" / "amsdds_backend" / "amsdds" / "weights"
)

LAYER1_MODEL_NAME = (
    "MobileNetV3-Large multimodal (image + age/sex/localization, HAM10000 7-class)"
)

_DEFAULT_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
_DEFAULT_MALIGNANT = ["mel", "bcc", "akiec"]


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no dependency). Existing env vars win; lines are
    `KEY=VALUE`, `#` comments and blanks ignored. Called once by load_config()."""
    path = path or (REPO_ROOT / ".env")
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DecisionEngineConfig:
    """Mirror of backend/ml_core/config/thresholds.yaml::decision_engine.

    Informational only. The handover Engine reads the YAML itself and is the
    single source of truth for the gate / risk / uncertainty maths. These
    fields exist so /model-info can report the thresholds without importing
    torch, and so the legacy backend.decision_engine unit tests still build.
    """

    conf_threshold: float
    entropy_threshold: float
    uncertain_floor: float
    risk_threshold: float = 0.26


@dataclass(frozen=True)
class AppConfig:
    checkpoint_path: Path          # <weights_dir>/<layer1 file> - existence = "weights present"
    thresholds_path: Path          # backend/ml_core/config/thresholds.yaml
    weights_dir: Path              # absolute; also exported to AMSDDS_WEIGHTS
    device: str
    mock_mode: bool
    mock_scenario: str | None      # "A", "B", or None (auto per-image)
    decision_engine: DecisionEngineConfig
    classes: list[str]
    malignant_classes: list[str]
    class_names: dict
    temperature: float             # display fallback only; real value is in the checkpoint

    @property
    def checkpoint_exists(self) -> bool:
        return self.checkpoint_path.is_file()


def load_thresholds_file(path: Path) -> dict:
    if not Path(path).is_file():
        raise FileNotFoundError(f"thresholds file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config() -> AppConfig:
    """Build the AppConfig from environment + backend/ml_core/config/thresholds.yaml."""
    load_dotenv()
    thresholds_path = Path(
        os.environ.get("THRESHOLDS_PATH", str(DEFAULT_THRESHOLDS_PATH))
    )
    raw = load_thresholds_file(thresholds_path)

    de = raw.get("decision_engine", {})
    decision_engine = DecisionEngineConfig(
        conf_threshold=float(de.get("conf_threshold", 0.5)),
        entropy_threshold=float(de.get("entropy_threshold", 0.3)),
        uncertain_floor=float(de.get("uncertain_floor", 0.5)),
        risk_threshold=float(de.get("risk_threshold", 0.26)),
    )

    weights_dir = Path(os.environ.get("AMSDDS_WEIGHTS", str(DEFAULT_WEIGHTS_DIR)))
    if not weights_dir.is_absolute():
        weights_dir = (REPO_ROOT / weights_dir).resolve()

    layer1_file = raw.get("layer1", {}).get(
        "file", "layer1_multimodal_mobilenetv3.pt"
    )
    checkpoint_path = weights_dir / layer1_file

    scenario = os.environ.get("MOCK_SCENARIO")
    if scenario:
        scenario = scenario.strip().upper()

    # Temperature is authoritative INSIDE the checkpoint (0.959). The new YAML
    # has no `calibration` block; this is a display fallback for /model-info.
    temperature = float(raw.get("calibration", {}).get("temperature", 0.959))

    return AppConfig(
        checkpoint_path=checkpoint_path,
        thresholds_path=thresholds_path,
        weights_dir=weights_dir,
        device=os.environ.get("DEVICE", "cpu"),
        mock_mode=_env_bool("DEVELOPMENT_MOCK_MODE", default=False),
        mock_scenario=scenario,
        decision_engine=decision_engine,
        classes=list(raw.get("classes", _DEFAULT_CLASSES)),
        malignant_classes=list(raw.get("malignant_classes", _DEFAULT_MALIGNANT)),
        class_names=dict(raw.get("class_names", {})),
        temperature=temperature,
    )
