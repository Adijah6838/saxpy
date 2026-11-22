// saxpy.cu - SAXPY y = a*x + y: scalar vs float4 vectorized
// Compile: nvcc -O3 -arch=sm_80 saxpy.cu -o saxpy
// Run: ./saxpy [N]  e.g. ./saxpy 100000000
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#define BLOCK 256
#define CHECK(x) do{auto e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__); return 1;}}while(0)
// Scalar: 1 thread = 1 element, grid-stride
__global__ void saxpy_scalar(float a, const float* x, float* y, int N){
  for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<N; i+=gridDim.x*blockDim.x)
    y[i]=a*x[i]+y[i];
}
// Vectorized: 1 thread = float4 = 4 elements, grid-stride over N/4 vectors
__global__ void saxpy_vec4(float a, const float4* x4, float4* y4, int N4){
  for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<N4; i+=gridDim.x*blockDim.x){
    float4 xv=x4[i]; float4 yv=y4[i];
    yv.x=a*xv.x+yv.x; yv.y=a*xv.y+yv.y; yv.z=a*xv.z+yv.z; yv.w=a*xv.w+yv.w;
    y4[i]=yv;
  }
}
// Tail: scalar fallback for N%4 remaining elements
__global__ void saxpy_tail(float a, const float* x, float* y, int start, int N){
  for(int i=start+blockIdx.x*blockDim.x+threadIdx.x; i<N; i+=gridDim.x*blockDim.x)
    y[i]=a*x[i]+y[i];
}
void cpu_saxpy(float a, const float* x, float* y, int N){
  for(int i=0;i<N;i++) y[i]=a*x[i]+y[i];
}
int main(int argc,char** argv){
  int N=argc>1?atoi(argv[1]):100000000;
  float a=2.0f;
  std::vector<float> x(N), y_cpu(N), y_gpu(N);
  srand(42); for(int i=0;i<N;i++){ x[i]=(float)(rand()%100)/10.0f; y_cpu[i]=(float)(rand()%100)/10.0f; y_gpu[i]=y_cpu[i]; }
  cpu_saxpy(a,x.data(),y_cpu.data(),N);
  float *dx,*dy; CHECK(cudaMalloc(&dx,(size_t)N*4)); CHECK(cudaMalloc(&dy,(size_t)N*4));
  CHECK(cudaMemcpy(dx,x.data(),(size_t)N*4,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dy,y_gpu.data(),(size_t)N*4,cudaMemcpyHostToDevice));
  int blocks=1024;
  auto bench=[&](auto kern, const char* name, bool isVec){
    CHECK(cudaMemcpy(dy,y_gpu.data(),(size_t)N*4,cudaMemcpyHostToDevice));
    int N4=N/4;
    if(isVec) ((void(*)(float,const float4*,float4*,int))kern)<<<blocks,BLOCK>>>(a,(float4*)dx,(float4*)dy,N4);
    else ((void(*)(float,const float*,float*,int))kern)<<<blocks,BLOCK>>>(a,dx,dy,N);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for(int r=0;r<20;r++){
      if(isVec){ ((void(*)(float,const float4*,float4*,int))kern)<<<blocks,BLOCK>>>(a,(float4*)dx,(float4*)dy,N4); if(N%4!=0) saxpy_tail<<<1,BLOCK>>>(a,dx,dy,N-(N%4),N); }
      else ((void(*)(float,const float*,float*,int))kern)<<<blocks,BLOCK>>>(a,dx,dy,N);
    }
    cudaEventRecord(e); CHECK(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms,s,e); ms/=20;
    CHECK(cudaMemcpy(y_gpu.data(),dy,(size_t)N*4,cudaMemcpyDeviceToHost));
    float maxErr=0; for(int i=0;i<N;i++) maxErr=fmaxf(maxErr,fabsf(y_cpu[i]-y_gpu[i]));
    double gbs=(3.0*N*4)/(ms*1e6);
    printf("%-8s %7.3f ms  %7.2f GB/s  maxErr=%.2e %s\n",name,ms,gbs,maxErr,maxErr<1e-3?"PASS":"FAIL");
    cudaEventDestroy(s); cudaEventDestroy(e);
    return gbs;
  };
  double gS=bench(saxpy_scalar,"scalar",false);
  double gV=bench(saxpy_vec4,"float4",true);
  printf("Speedup: %.2fx (float4 %.1f GB/s vs scalar %.1f GB/s)\n",gV/gS,gV,gS);
  CHECK(cudaFree(dx)); CHECK(cudaFree(dy)); CHECK(cudaDeviceReset());
  return 0;
}
