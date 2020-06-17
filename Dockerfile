# Get python image from docker hub
FROM python:3.7
# set the maintanier
LABEL maintainer="golammdmortuza@u.boisestate.edu"
# Install required library we just used numpy
RUN pip install numpy
# Copy current directory file to docker container
COPY ./ /

ENTRYPOINT ["python"]

COPY . /test
