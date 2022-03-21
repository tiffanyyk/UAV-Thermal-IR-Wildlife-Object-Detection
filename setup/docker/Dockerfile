ARG USERNAME
ARG PASSWORD
ARG USER_ID
ARG GROUP_ID
ARG BASE_IMG

FROM ${BASE_IMG}

# USER root
# # need this for pillow
# RUN apt-get update && apt-get install -y python3.8-dev

ENV TZ=America/Tijuana
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# # need this to avoid `NotFoundError: no lapack/blas resources found` when installing scipy
# RUN apt-get update && apt-get install -y gfortran libopenblas-dev liblapack-dev

# # install numpy first, otherwise no module named numpy error # 1.17.2
# RUN pip install numpy==1.17.3  

# install required packages for yolor
RUN pip install Cython
COPY yolor_requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt

# # need these dependencies for Pillow
# RUN apt-get update && apt-get install -y \
#     libtiff5-dev \
#     libjpeg8-dev \
#     zlib1g-dev \
#     libfreetype6-dev \
#     liblcms2-dev \
#     libwebp-dev \
#     libharfbuzz-dev \
#     libfribidi-dev \
#     tcl8.6-dev \
#     tk8.6-dev \
#     python-tk
# RUN pip install Pillow==6.2.0
# problem with installing: Pillow==6.2.0, cannot compile python.h perhaps you need to install python-dev|python-devel

# # required packages not listed in the requirements file
# RUN pip install imageio ujson

# # need this otherwise numba causes `TypeError: create_target_machine() got an unexpected keyword argument 'jitdebug'`
# RUN pip install -U llvmlite==0.32.1

# need this for cv2
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

## stuff for centernet2  # need to install detectron!
#RUN pip install omegaconf cloudpickle fvcore
#RUN git clone https://github.com/facebookresearch/detectron2.git && python -m pip install -e detectron2

RUN apt-get update && apt-get install -y ssh openssh-server net-tools

# add user for given user args
ARG USERNAME
ARG PASSWORD
ARG GROUP_ID
ARG USER_ID
RUN if [ ${USER_ID:-0} -ne 0 ] && [ ${GROUP_ID:-0} -ne 0 ]; then \
    groupadd -g ${GROUP_ID} ${USERNAME} && \
    useradd -m -l -u ${USER_ID} -g ${USERNAME} ${USERNAME} && \
    echo "${USERNAME}:${PASSWORD}" | chpasswd \
    ;fi
RUN echo "/usr/sbin/sshd" >> ~/.bashrc
RUN mkdir /var/run/sshd
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22

ENTRYPOINT ["/bin/bash"]
