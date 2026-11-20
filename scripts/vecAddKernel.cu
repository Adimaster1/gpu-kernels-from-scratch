#include <stdio.h>
#include <math.h>
#include <assert.h>

#define SIZE 256
#define SHMEM_SIZE 256*4


__global__
void vecAddKernel( float* A ,float* B,float* C , int n)
{
   //cal unique index for the thread
   int i = threadIdx.x + (blockDim.x * blockIdx.x) ;

   if (i<n)
   {
       C[i] = A[i] + B[i];
   }
}

void initialize_vector(int *v,int n) {
    for (int i=0;i<n;i++){
        v[i] =1 ;
        }
    }

int main(){
    //vector size
    int n = 1<<16;
    size_t bytes = n * sizeof(int);

    int *h

    }