docker build -t sharmi/fep_test . #--no-cache --pull
docker kill sharmi_fep_test
docker rm sharmi_fep_test

#xhost +si:localuser:$(whoami) >/dev/null

#docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
#docker run --runtime=nvidia --shm-size=1g -e NVIDIA_VISIBLE_DEVICES=0 -d --name sharmi_pointnet_autoe \
nvidia-docker run -d --name sharmi_fep_test \
-v /allen:/usr/local/allen \
-v ~/.cloudvolume/secrets:/root/.cloudvolume/secrets \
-e AWS_ACCESS_KEY_ID=AKIAZE5KZHQ3L56GWEON \
-e AWS_SECRET_ACCESS_KEY=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  \
-e AWS_DEFAULT_REGION=us-west-2 \
-p 9779:9779 \
-e "PASSWORD=$JUPYTERPASSWORD" \
-e DISPLAY \
--privileged \
-i -t sharmi/fep_test  \
/bin/bash -c "sudo initialize-graphics >/dev/null 2>/dev/null; vglrun glxspheres64; jupyter notebook --allow-root ;"

#-v /allen/programs/celltypes/workgroups/em-connectomics/analysis_group/forSharmi/code/fep_docker_test:/usr/local/featureExtractionParty \
