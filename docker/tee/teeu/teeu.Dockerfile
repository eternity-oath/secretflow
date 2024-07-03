FROM secretflow/occlum:0.29.5-ubuntu20.04 as base

RUN apt-get update -y && apt-get install wget && apt-get clean

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash ./Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN /root/miniconda3/bin/conda create --name secretflow python==3.8.10 Cython

SHELL ["/root/miniconda3/bin/conda", "run", "-n", "secretflow", "/bin/bash", "-c"]

RUN pip install "grpcio>=1.28.1,<=1.43.0"
RUN pip install ray[tune]==1.13.0 psutil && python -c "import ray; obj = ray.put(1); print(ray.get(obj));" && pip uninstall -y ray

RUN pip install secretflow_ray==1.13.0 



FROM secretflow/occlum:0.29.5-ubuntu20.04

COPY --from=base /root/miniconda3/envs/secretflow/lib /root/miniconda3/envs/secretflow/lib
COPY --from=base /root/miniconda3/envs/secretflow/bin/python \
    /root/miniconda3/envs/secretflow/bin/pip \
    /root/miniconda3/envs/secretflow/bin/ray \
    /root/miniconda3/envs/secretflow/bin/

RUN ln -s /root/miniconda3/envs/secretflow/bin/python /bin/python \
    && ln -s /root/miniconda3/envs/secretflow/bin/pip /bin/pip

# The `ray` script is generated by python setuptools, however, the interpreter's path is generated based on
# the one used for packaging. Need to modify it manully to the path in Occlum.
RUN sed -i 's/#!\/root\/miniconda3\/envs\/secretflow\/bin\/python/#!\/bin\/python/g' /root/miniconda3/envs/secretflow/bin/ray

RUN occlum new occlum_instance

WORKDIR /root/occlum_instance

COPY python.yaml /tmp/

RUN rm -rf image \
    && copy_bom -f /tmp/python.yaml --root image --include-dir /opt/occlum/etc/template


RUN PYTHONPATH=$PWD/image/opt/secretflow/lib/python3.8/site-packages/ image/opt/secretflow/bin/pip install \
    --prefix image/opt/secretflow \
    secretflow_ray==1.13.0 \
    "sf-heu>=0.4.3b2" \
    "spu>=0.3.3b0" \
    pandas \
    scikit-learn==1.1.3 \
    s3fs==2022.1.0 \
    cryptography==39.0.1 \
    termcolor \
    sdc-sdk \
    sdc-apis \
    xgboost==1.5.2 \
    && rm -rf /root/.cache

ARG sf_version

ENV version $sf_version

RUN PYTHONPATH=$PWD/image/opt/secretflow/lib/python3.8/site-packages/ pip install --prefix image/opt/secretflow --no-deps \
    secretflow-rayfed==0.1.1a3 \
    secretflow==${version}

COPY tls_cert.py image/opt/secretflow/lib/python3.8/site-packages/tls_cert.py

COPY Occlum.custom.json /tmp/Occlum.custom.json

RUN jq -s '.[0] * .[1]' Occlum.json /tmp/Occlum.custom.json > Occlum.new.json && mv Occlum.new.json Occlum.json
