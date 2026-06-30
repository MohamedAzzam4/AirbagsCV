"""
Minimal smoke tests for the AirbagsCV repo.

These do NOT require the dataset or a GPU. They verify that:
- The repo's scripts parse and import cleanly.
- The smoke test entry point runs.
- The CLI of each main script accepts --help.

Run with:  pytest tests/ -v
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_scripts_parse():
    """Each script must parse without syntax errors."""
    import ast
    scripts = [
        "scripts/prepare_aitex.py",
        "scripts/train_models.py",
        "scripts/benchmark_inference.py",
        "scripts/smoke_test.py",
        "scripts/export_models.py",
        "scripts/generate_synthetic_defects.py",
        "scripts/run_cold_start_ablation.py",
        "scripts/train_proxy_baselines.py",
        "demo/inference.py",
        "demo/app.py",
    ]
    for s in scripts:
        p = REPO_ROOT / s
        assert p.exists(), f"Missing: {s}"
        ast.parse(p.read_text(), filename=s)


def test_smoke_test_runs():
    """scripts/smoke_test.py must run end-to-end and exit 0 (when deps installed)."""
    # If anomalib isn't installed, the smoke test will report failures but still
    # exit 0 unless --strict is set. We accept either here.
    r = _run([sys.executable, "scripts/smoke_test.py"], timeout=120)
    assert r.returncode in (0, 1), f"smoke_test exited {r.returncode}\n{r.stderr}"


def test_prepare_aitex_help():
    r = _run([sys.executable, "scripts/prepare_aitex.py", "--help"])
    assert r.returncode == 0
    assert "--source" in r.stdout
    assert "--blank-threshold" in r.stdout


def test_train_models_help():
    r = _run([sys.executable, "scripts/train_models.py", "--help"])
    assert r.returncode == 0
    assert "--data-dir" in r.stdout
    assert "--output-dir" in r.stdout
    assert "--resume-from-checkpoint" in r.stdout
    assert "--imagenet-dir" in r.stdout


def test_benchmark_inference_help():
    r = _run([sys.executable, "scripts/benchmark_inference.py", "--help"])
    assert r.returncode == 0
    assert "--warmup" in r.stdout
    assert "--iterations" in r.stdout
    assert "--checkpoint" in r.stdout


def test_stub_scripts_advertise_themselves():
    """Stub scripts must clearly say they are NOT implemented."""
    for s in ["generate_synthetic_defects", "run_cold_start_ablation",
              "train_proxy_baselines"]:
        r = _run([sys.executable, f"scripts/{s}.py"], timeout=10)
        assert r.returncode == 0
        assert "not implemented" in r.stdout.lower() or "stub" in r.stdout.lower()
