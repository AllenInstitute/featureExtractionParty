FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04 

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        module-init-tools \
        build-essential \
        ca-certificates \ 
	software-properties-common \
	apt-utils
RUN apt-get install -y python3-pip

RUN apt-get -y update && \
    apt-get -y install graphviz libxml2-dev python3-cairosvg parallel
#RUN pip install scikit-build 
RUN apt-get -y install cmake

# CGAL Dependencies ########################################################
RUN apt-get -y install libboost-all-dev libgmp-dev libmpfr-dev libcgal-dev libboost-wave-dev
RUN apt-get -y install vim
RUN apt-get -y install libassimp-dev
RUN apt-get -y install libspatialindex-dev


#CONDA
WORKDIR /conda
RUN apt-get -y install wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh --quiet -O miniconda.sh
#RUN echo "faa7cb0b0c8986ac3cacdbbd00fe4168  miniconda.sh" | md5sum --check
RUN bash miniconda.sh -b -up /conda
RUN rm miniconda.sh

# make sure conda bin is in front of PATH so we get correct pip
ENV PATH="/conda/bin:$PATH"

# create a conda env
RUN /conda/bin/conda config --set always_yes yes --set changeps1 no
RUN /conda/bin/conda create -q -n denv python=3.7

# make sure pip/conda is the latest
RUN pip install --upgrade pip
RUN /conda/bin/conda update -n base conda

# add conda-forge as remote channel
RUN /conda/bin/conda config --add channels conda-forge

# scikit-image is used for marching cubes
# pyembree is used for fast ray tests
RUN /conda/bin/conda install scikit-image pyembree

# actually install trimesh and pytest
RUN pip install trimesh[all] pytest pyassimp==4.1.3

# remove archives
RUN /conda/bin/conda clean --all -y


RUN pip install scikit-build
RUN pip install --upgrade setuptools

RUN pip install -U numpy
RUN pip install future
RUN pip install Pillow
#RUN pip install tifffile==2020.2.16
RUN pip install opencv_python
RUN pip install grpcio-tools==1.21.*
RUN pip install tensorflow-gpu


RUN pip install scipy
#RUN pip install pillow
RUN pip install matplotlib
RUN pip install jupyter
RUN pip install trimesh
RUN pip install umap-learn
#RUN pip install mplcursors

#RUN apt-get install libhdf5-dev -y
RUN pip install h5py
RUN pip install neuroglancer
RUN pip install meshparty
RUN pip install pandas
RUN pip install analysisdatalink
RUN pip install annotationframeworkclient



RUN mkdir -p /usr/local/featureExtractionParty
WORKDIR /usr/local/featureExtractionParty
COPY . /usr/local/featureExtractionParty
COPY jupyter_notebook_config.py /root/.jupyter/

#INSTALL CGAL PYTHON
ADD ./CGAL/cgal_functions /src/CGAL/cgal_functions
RUN apt-get -y install g++
RUN pip install -e /src/CGAL/cgal_functions

RUN apt install libgl1-mesa-glx -y
RUN pip install itkwidgets
RUN pip install tensorflow-gpu==1.14.0
#add the cgal scripts
#EXPOSE 9774
#RUN mkdir -p /scripts
#ADD ./jupyter/run_jupyter.sh /scripts/
#ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/
#ADD ./jupyter/custom.css /root/.jupyter/custom/
#RUN chmod -R a+x /scripts
#ENTRYPOINT ["/scripts/run_jupyter.sh"]

