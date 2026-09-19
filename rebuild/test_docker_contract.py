from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docker_stack_is_deepstream_and_qdrant():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "nvcr.io/nvidia/deepstream:8.0-triton-multiarch" in docker
    assert "qdrant/qdrant:v1.19.1" in compose
    assert "gpus: all" in compose
    assert 'QDRANT_URL: "${QDRANT_URL:-http://127.0.0.1:6333}"' in compose
    assert '"${QDRANT_PORT:-6333}:6333"' in compose
    assert "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so" in compose
    assert "qdrant_data:/qdrant/storage" in compose
    assert "shm_size: \"8gb\"" in compose
    assert "QDRANT_PORT" in compose
    assert "pyds-1.2.2-cp312-cp312-linux_x86_64.whl" in docker


def test_container_runtime_is_strict():
    entry = (ROOT / "docker/entrypoint.sh").read_text(encoding="utf-8")
    assert "libnvds_meta.so" in entry
    assert "libnvds_nvmultiobjecttracker.so" in entry
    assert "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so" in entry
    assert "nvtracker" in entry
    assert "pyds" in entry
    assert "/readyz" in entry


def test_qdrant_client_supports_remote_service():
    value = (ROOT / "src/live/qdrant_gallery.py").read_text(encoding="utf-8")
    assert 'QDRANT_URL' in value
    assert 'QdrantClient(' in value
    assert 'url=self.url' in value

def test_one_command_runner_is_present():
    runner = (ROOT / "scripts/run_docker_reid.sh").read_text(encoding="utf-8")
    assert "docker compose up -d qdrant" in runner
    assert "docker compose build reid" in runner
    assert "verify_reid_runtime.py" in runner
    assert "batch_state_final" in runner
    assert "_live_src_cam_219.mp4" in runner
    assert "selecting Qdrant endpoint" in runner
