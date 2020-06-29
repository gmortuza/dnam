# Get python image from docker hub
FROM python:3.7
# set the maintanier
LABEL maintainer="golammdmortuza@u.boisestate.edu"
# Install required library we just used numpy
RUN pip install numpy
RUN pip install scipy
RUN pip install numba
RUN pip install matplotlib
RUN pip install lmfit
RUN pip install tqdm
RUN pip install yaml
RUN pip install h5py
# Copy current directory file to docker container
COPY ./ /

ENTRYPOINT ["python"]

COPY . /test
