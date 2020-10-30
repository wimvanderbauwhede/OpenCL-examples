nvcc -arch compute_30 -code sm_30 -c matacc_cuda_kernel.cu 
nvcc -x cu -arch compute_30 -code sm_30 -DVERBOSE -DREF=1 -DNRUNS=10 -DWIDTH=1024 -c matacc_cuda_driver.cc
nvcc -arch compute_30 -code sm_30 matacc_cuda_kernel.o matacc_cuda_driver.o -o matacc_cuda
