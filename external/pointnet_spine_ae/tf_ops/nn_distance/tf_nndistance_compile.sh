/usr/local/cuda-9.2/bin/nvcc tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I /conda/lib/python3.7/site-packages/tensorflow/include -I /usr/local/cuda-9.2/include -I /conda/lib/python3.7/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.2/lib64/ -L/conda/lib/python3.7/site-packages/tensorflow -O2  -ltensorflow_framework
#-ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

