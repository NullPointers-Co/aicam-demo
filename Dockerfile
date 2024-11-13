FROM debian:latest

ARG DEBIAN_FRONTEND=noninteractive

ENV HTTP_PROXY=http://host.docker.internal:7890
ENV HTTPS_PROXY=http://host.docker.internal:7890
ENV NO_PROXY="localhost,127.0.0.1"

RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \ 
    && apt-get install -y build-essential --no-install-recommends make \
        ca-certificates \
        git \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        libgl1 \
        libglib2.0-0

# Python and poetry installation
ARG PYTHON_VERSION=3.11

ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:/root/.local/bin:$PATH"

RUN echo "done 0" \
    && curl https://pyenv.run | bash \
    && echo "done 1" \
    && pyenv install ${PYTHON_VERSION} \
    && echo "done 2" \
    && pyenv global ${PYTHON_VERSION} \
    && echo "done 3" \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.in-project true

WORKDIR /workspace/ppp-demo-docker
COPY images images
COPY *.py .
COPY poetry.lock pyproject.toml .

RUN poetry install \
    && mkdir -p weights \
    && curl -L -o weights/yolov11n.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"

ENV TARGET="images/target.jpeg"
ENV REFERENCE="images/reference.jpeg"
ENV CONFIDENCE="0.35"

CMD ["/bin/sh", "-c", ".venv/bin/python cmdlt.py --target ${TARGET} --reference ${REFERENCE} --confidence ${CONFIDENCE}"]
