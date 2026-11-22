# SAXPY - Vectorized float4

y = a*x + y - scalar vs float4, grid-stride, tail handling.

## Why float4
Scalar: 1 thread = 4B load + 4B load + 4B store = 3 transactions per 4 elems.
float4: 1 thread = 16B load + 16B store = 1 transaction per 4 elems. Fewer instructions, better coalescing, hits peak BW.
Alignment: cudaMalloc is 256B aligned, float4 cast is safe.

## Tail
N4 = N/4 vectors via float4. Remaining N%4 elems via saxpy_tail scalar kernel, start=N-N%4, masked i<N, never reads past end.

## Grid-Stride
for(i=tid; i<N; i+=grid*block) - works for N > max grid, any N.

## Benchmark (expected, T4 Colab, N=100M, 400MB)
| Kernel | Time | GB/s |
| Scalar | ~1.80ms | ~667 |
| float4 | ~1.35ms | ~889 |
Fill real numbers from Colab run.

## Build / Run
nvcc -O3 -arch=sm_80 saxpy.cu -o saxpy
./saxpy 100000000
