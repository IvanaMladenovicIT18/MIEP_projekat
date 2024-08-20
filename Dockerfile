FROM ascendhub.huawei.com/public-ascendhub/ascend-pytorch:22.0.0-ubuntu18.04

RUN pip install --proxy=http://ftn.proxy:8080 --no-cache-dir torch torch_npu torchvision torchsummary tqdm matplotlib

WORKDIR /miep

COPY . /miep


#RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /miep
USER root

CMD ["python", "model1.py"]
