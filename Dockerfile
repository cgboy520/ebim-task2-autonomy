# ROS 2 Jazzy base (Python 3.12) plus numpy, PyYAML, pydantic, loguru and
# ros-jazzy-vision-msgs (Detection2DArray for the live loose target bbox).
FROM ros:jazzy-ros-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/ebim-task2/src \
    DEBIAN_FRONTEND=noninteractive \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    FASTDDS_BUILTIN_TRANSPORTS=UDPv4

WORKDIR /opt/ebim-task2

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip ros-jazzy-vision-msgs \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages \
        "numpy>=1.26,<3.0" "PyYAML>=6.0" "pydantic>=2.6,<3.0" "loguru>=0.7"

COPY src/ /opt/ebim-task2/src/
COPY config/ /opt/ebim-task2/config/
COPY scripts/entrypoint.sh /opt/ebim-task2/scripts/entrypoint.sh
RUN chmod +x /opt/ebim-task2/scripts/entrypoint.sh

# Non-root runtime user.
RUN useradd --create-home --uid 1001 ebim \
    && chown -R ebim:ebim /opt/ebim-task2
USER ebim

ENTRYPOINT ["/opt/ebim-task2/scripts/entrypoint.sh"]
# Default CMD = the official run entry: loops the placement chain with an
# accept ladder, stops on an accepted lay, parks the arms on every exit path.
CMD ["python3", "-u", "-m", "ebim_task2.official_run"]
