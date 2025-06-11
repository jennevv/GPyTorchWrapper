FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

RUN conda env create -f /app/environment.yml -n env

SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]
RUN pip install -e . --use-pep517
ENV PATH /opt/conda/envs/gpytorchenv/bin:$PATH

RUN echo "Make sure pytorch is installed:"
RUN  python -c "import torch; print(torch.__version__)"


