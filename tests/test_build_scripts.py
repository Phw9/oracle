from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_build_script_defaults_to_cpu_llama_cpp() -> None:
    script_text = (ROOT_DIR / "build.sh").read_text(encoding="utf-8")

    assert 'FORCE_CUDA="0"' in script_text
    assert "--cpu                    Force build llama.cpp in CPU-only mode (default)" in script_text
    assert "-DGGML_CUDA=OFF" in script_text
    assert "llama_cuda_cache_enabled" in script_text
    assert "reconfiguring CPU-only build" in script_text


def test_build_script_supports_hwonlygpu_profile() -> None:
    script_text = (ROOT_DIR / "build.sh").read_text(encoding="utf-8")

    assert "hwonlygpu" in script_text
    assert 'BUILD_PROFILE="hwonlygpu"' in script_text
    assert "Using hwonlygpu laptop build profile" in script_text
    assert 'FORCE_CUDA="1"' in script_text
    assert "llama_cuda_cache_disabled" in script_text
    assert "reconfiguring CUDA build" in script_text
    assert "building local llama.cpp even though llama-server is on PATH" in script_text
    assert "CUDA Toolkit with nvcc is required" in script_text
    assert "CMAKE_CUDA_COMPILER" in script_text
    assert "configure_windows_cuda_cmake" in script_text
    assert "cuda=$CUDA_TOOLKIT_ROOT" in script_text
    assert "detect_cuda_architectures" in script_text
    assert "ORACLE_CUDA_ARCHITECTURES" in script_text
    assert "CMAKE_CUDA_ARCHITECTURES" in script_text
    assert "Visual Studio 17 2022" in script_text
    assert "CMAKE_SELECTED_GENERATOR" in script_text


def test_run_script_keeps_llama_cpu_by_default() -> None:
    script_text = (ROOT_DIR / "run.sh").read_text(encoding="utf-8")

    assert "running llama.cpp on CPU by default" in script_text
    assert "automatically setting --n-gpu-layers 99" not in script_text


def test_run_script_supports_hwonly_runtime_profile() -> None:
    script_text = (ROOT_DIR / "run.sh").read_text(encoding="utf-8")

    assert "apply_run_profile" in script_text
    assert "Using hwonly laptop run profile" in script_text
    assert 'LLAMA_NGL="99"' in script_text
    assert "ORACLE_CAMERA_BACKEND" in script_text
    assert "ORACLE_CAMERA_AUTO_DETECT" in script_text
    assert "ORACLE_PREFER_LOCAL_LLAMA_SERVER" in script_text
    assert "ORACLE_REQUIRE_MANAGED_LLAMA_SERVER" in script_text
    assert "hwonly requires its own managed llama.cpp server" in script_text
    assert "stop_recorded_llama_server" in script_text
    assert "process_command_line" in script_text
    assert "ps -W" in script_text
    assert 'RUN_LLAMA_PARALLEL="1"' in script_text
    assert 'LLAMA_BATCH_SIZE="2048"' in script_text
    assert 'LLAMA_UBATCH_SIZE="512"' in script_text
    assert 'LLAMA_FLASH_ATTN="on"' in script_text
    assert 'LLAMA_POLL="100"' in script_text


def test_scripts_parse_dotenv_without_sourcing_shell() -> None:
    build_script = (ROOT_DIR / "build.sh").read_text(encoding="utf-8")
    run_script = (ROOT_DIR / "run.sh").read_text(encoding="utf-8")

    for script_text in (build_script, run_script):
        assert "load_dotenv_file" in script_text
        assert 'source "$ROOT_DIR/.env"' not in script_text
        assert 'export "$key=$value"' in script_text


def test_run_script_accepts_windows_virtualenv() -> None:
    script_text = (ROOT_DIR / "run.sh").read_text(encoding="utf-8")

    assert "activate_repo_venv" in script_text
    assert "Scripts/activate" in script_text


def test_run_script_consumes_legacy_distributed_options() -> None:
    script_text = (ROOT_DIR / "run.sh").read_text(encoding="utf-8")

    assert "--distributed-role)" in script_text
    assert "--distributed-split)" in script_text
    assert "--slave-addrs)" in script_text


def test_scripts_find_windows_llama_server_binary() -> None:
    build_script = (ROOT_DIR / "build.sh").read_text(encoding="utf-8")
    run_script = (ROOT_DIR / "run.sh").read_text(encoding="utf-8")
    service_script = (ROOT_DIR / "scripts" / "run_llama_server.sh").read_text(
        encoding="utf-8",
    )

    assert "llama-server.exe" in build_script
    assert "llama-server.exe" in run_script
    assert "llama-server.exe" in service_script
