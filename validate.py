import random
def cpu_saxpy(a,x,y):
    return [a*xi+yi for xi,yi in zip(x,y)]
def gpu_mirror(a,x,y):
    N=len(x); y=list(y); N4=N//4
    for i in range(N4):
        for k in range(4):
            idx=i*4+k; y[idx]=a*x[idx]+y[idx]
    for idx in range(N4*4,N):
        y[idx]=a*x[idx]+y[idx]
    return y
for N in [1000,1001,1002,1003]:
    random.seed(42); x=[random.uniform(-10,10) for _ in range(N)]; y=[random.uniform(-10,10) for _ in range(N)]
    c=cpu_saxpy(2.0,x,y); g=gpu_mirror(2.0,x,y); err=max(abs(a-b) for a,b in zip(c,g))
    print(f'N={N} tail={N%4} PASS={err<1e-6} maxErr={err:.2e}')
print('SAXPY validated - float4 + tail')
