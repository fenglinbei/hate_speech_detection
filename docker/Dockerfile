FROM nvcr.io/nvidia/pytorch:23.08-py3

WORKDIR /workspace/
ENV PYTHONPATH /workspace/
COPY . /workspace

RUN groupadd -g 1000 fenglin && \
    useradd -u 1000 -g fenglin -m fenglin
USER fenglin

CMD ["tail", "-f", "/dev/null"]

# RUN cp /usr/local/cuda-11.8/NsightSystems-cli-2022.4.2/host-linux-x64/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/
# RUN cp -r ./cudnn/lib/* /usr/local/cuda/lib64 
# RUN cp -r ./cudnn/include/* /usr/local/cuda/include
# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/usr/local/cuda/lib64"

RUN pip install --no-cache-dir -r requirements.txt -i https://mirror.baidu.com/pypi/simple
# RUN pip uninstall opencv-python -y
# RUN pip uninstall opencv-contrib-python -y
# RUN pip uninstall opencv-python-headless -y
# RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2
# RUN pip install opencv-python-headless==4.8.0.76 -i https://mirror.baidu.com/pypi/simple

# RUN cp -r ./hanlp/ /root/.hanlp

# EXPOSE 5000