FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics

RUN apt-get update && apt-get install -y --no-install-recommends     python3-pip     python3-dev     python3-gi     python3-gst-1.0     gir1.2-gstreamer-1.0     ffmpeg     git     curl     pkg-config     libglib2.0-0     libsm6     libxext6     libxrender1     libgomp1     && rm -rf /var/lib/apt/lists/*

# DeepStream 8.0 on CPython 3.12 needs matching NVIDIA PyDS.
# Prefer an SDK-shipped wheel; otherwise install NVIDIA's official 1.2.2 wheel.
RUN set -eux; \
    found=0; \
    for wheel in /opt/nvidia/deepstream/deepstream/lib/pyds-*.whl \
                 /opt/nvidia/deepstream/deepstream/lib/python/pyds-*.whl; do \
        if [ -f "$wheel" ]; then \
            python3 -m pip install --no-deps --force-reinstall "$wheel"; \
            found=1; \
            break; \
        fi; \
    done; \
    if [ "$found" -eq 0 ]; then \
        curl -fL --retry 3 --retry-delay 2 \
          https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.2/pyds-1.2.2-cp312-cp312-linux_x86_64.whl \
          -o /tmp/pyds.whl; \
        python3 -m pip install --no-deps /tmp/pyds.whl; \
        rm -f /tmp/pyds.whl; \
    fi

WORKDIR /workspace/inference_person_reid

COPY requirements.txt /tmp/reid-requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel &&     python3 -m pip install --no-cache-dir -r /tmp/reid-requirements.txt &&     python3 -m pip uninstall -y onnxruntime >/dev/null 2>&1 || true &&     python3 -m pip install --no-cache-dir --no-deps insightface==2.0

COPY docker/requirements-extra.txt /tmp/reid-extra.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/reid-extra.txt

ENV PYTHONPATH=/workspace/inference_person_reid:/workspace/inference_person_reid/src
ENV NVDCF_DEEPSTREAM_ROOT=/opt/nvidia/deepstream/deepstream
ENV NVDCF_TRACKER_LIBRARY=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
ENV GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:/opt/nvidia/deepstream/deepstream/lib/gst-plugins:$LD_LIBRARY_PATH
ENV QDRANT_URL=http://127.0.0.1:6333

COPY docker/entrypoint.sh /usr/local/bin/reid-entrypoint
RUN chmod +x /usr/local/bin/reid-entrypoint

ENTRYPOINT ["/usr/local/bin/reid-entrypoint"]
CMD ["bash"]