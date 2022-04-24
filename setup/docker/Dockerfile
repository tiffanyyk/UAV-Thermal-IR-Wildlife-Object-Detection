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

## stuff for yolov5
RUN pip install seaborn
RUN pip install tqdm==4.63.1

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
