FROM nvcr.io/nvidia/pytorch:21.07-py3

RUN pip install timm==0.5.4
RUN pip install texttable==1.6.4
RUN pip install mmcv==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu114/torch1.10.0/index.html
RUN pip install tqdm
RUN pip install imageio==2.9.0
RUN pip install git+https://github.com/lucasb-eyer/pydensecrf.git
RUN pip install opencv-python-headless==4.4.0.46
RUN pip install omegaconf==2.0.0
RUN apt update && apt install -y swig
# RUN apt-get update
# RUN apt install -y libgl1-mesa-glx