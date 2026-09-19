from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docker_stack_is_deepstream_and_qdrant():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "nvcr.io/nvidia/deepstream:8.0-triton-multiarch" in docker
    assert "qdrant/qdrant" in compose
    assert "gpus: all" in compose
    assert "QDRANT_URL: http://127.0.0.1:6333" in compose


def test_container_runtime_is_strict():
    entry = (ROOT / "docker/entrypoint.sh").read_text(encoding="utf-8")
    assert "libnvds_meta.so" in entry
    assert "libnvds_nvmultiobjecttracker.so" in entry
    assert "nvtracker" in entry
    assert "pyds" in entry


def test_qdrant_client_supports_remote_service():
    value = (ROOT / "src/live/qdrant_gallery.py").read_text(encoding="utf-8")
    assert 'QDRANT_URL' in value
    assert 'QdrantClient(' in value
    assert 'url=self.url' in value
