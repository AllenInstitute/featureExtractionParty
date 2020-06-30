#/bin/bash

# TF1.4
/usr/local/cuda-9.2/bin/nvcc tf_approxmatch_g.cu -o tf_approxmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I /usr/local/lib/python3.6/dist-packages/tensorflow/include -I /usr/local/cuda-9.2/include -I /usr/local/lib/python3.6/dist-packages/tensorflow/include/external/nsync/public -lcudart -L/usr/local/cuda-9.2/lib64/ -L/usr/local/lib/python3.6/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0 
#-ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
