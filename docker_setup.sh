docker build -t sharmi/fep . #--no-cache --pull
docker kill sharmi_fep
docker rm sharmi_fep

xhost +si:localuser:$(whoami) >/dev/null

#docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
#docker run --runtime=nvidia --shm-size=1g -e NVIDIA_VISIBLE_DEVICES=0 -d --name sharmi_pointnet_autoe \
nvidia-docker run -d --name sharmi_fep \
-v /allen/programs/celltypes/workgroups/em-connectomics/analysis_group/forSharmi/code/featureExtractionParty:/usr/local/featureExtractionParty \
-v /etc/hosts:/etc/hosts \
-v /allen:/allen \
-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
-p 9778:9778 \
-p 0.0.0.0:8050:8050 \
-e "PASSWORD=$JUPYTERPASSWORD" \
-e DISPLAY \
--privileged \
-i -t sharmi/fep  \
/bin/bash -c "sudo initialize-graphics >/dev/null 2>/dev/null; vglrun glxspheres64; jupyter notebook --allow-root ;"

