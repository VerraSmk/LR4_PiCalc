#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <ctime>
#include <vector>


using namespace std;

clock_t c_start, c_end;
int n = 1024 * 1024 * 32;

__global__ void count_pi(float* dev_randX, float* dev_randY, int* dev_threads_num, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int cont = 0;
	for (int i = tid * 128; i < 128 * (tid + 1); i++) {
		if (dev_randX[i] * dev_randX[i] + dev_randY[i] * dev_randY[i] < 1.0f) {
			cont++;
		}
	}
	dev_threads_num[tid] = cont;
}

int main() {

	vector<float> randX(n);
	vector<float> randY(n);

	srand((unsigned)time(NULL));
	for (int i = 0; i < n; i++) {
		randX[i] = float(rand()) / RAND_MAX;
		randY[i] = float(rand()) / RAND_MAX;
	}

	c_start = clock();
	int c_count = 0;

	for (int i = 0; i < n; i++) {
		if (randX[i] * randX[i] + randY[i] * randY[i] < 1.0f) {
			c_count++;
		}
	}
	c_end = clock();
	float t_cpu = (float)(c_end - c_start) / CLOCKS_PER_SEC;
	float c_num = float(c_count) * 4.0 / n;
	cout << "CPU Time" << endl;
	cout << c_num << endl;
	cout << "time= " << t_cpu * 1000 << " ms" << endl;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	size_t size = n * sizeof(float);
	float* dev_randX;
	float* dev_randY;
	cudaMalloc((void**)&dev_randX, size);
	cudaMalloc((void**)&dev_randY, size);

	cudaMemcpy(dev_randX, &randX.front(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_randY, &randY.front(), size, cudaMemcpyHostToDevice);

	int threadsPerBlock = 512;
	int block_num = n / (128 * threadsPerBlock);
	int* dev_threads_num;
	cudaMalloc((void**)&dev_threads_num, n / 128 * sizeof(int));

	count_pi << <block_num, threadsPerBlock >> > (dev_randX, dev_randY, dev_threads_num, n);

	int* threads_num = new int[n / 128];
	cudaMemcpy(threads_num, dev_threads_num, n / 128 * sizeof(int), cudaMemcpyDeviceToHost);

	int g_count = 0;
	for (int i = 0; i < n / 128; i++) {
		g_count += threads_num[i];
	};

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float t_gpu1;
	cudaEventElapsedTime(&t_gpu1, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	float g_num = float(g_count) * 4.0 / n;
	cout << "GPU_1 Time" << endl;
	cout << g_num << endl;
	cout << "time = " << t_gpu1 << " ms" << endl;
}