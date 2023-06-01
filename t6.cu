#include <cmath>
#include <string>
#include <cstdio>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cublas_v2.h>

using namespace std;

__global__ void sigmoid(float *arr, int size)
{
	int id = threadIdx.x;
	if(id < size && id >= 0) 
		arr[id] = 1 / (1 + exp(-arr[id]));
}

void check(float res){
	float real = round(0.0097 * 10000) / 10000;
	res = round(res * 10000) / 10000;
	if(real == res){
        cout << "correct"<< "\n";
    }else{
        cout << "result("<< res << ") != real(" << real << ")" << "\n";
    }
}

class Layer
{
private:
	cublasHandle_t handle;
	float alpha, beta;
	float *weights, *biases;
	int in_size, out_size;

	void read_weights(string pathToWeights){
		float *h_arr = new float [in_size*out_size];
		float *host_array = new float [in_size*out_size];
		try{
			ifstream s(pathToWeights);
			for (int i = 0; i < in_size*out_size; i++){
                s >> h_arr[i];
            }
			s.close();
		}
		catch(exception const& e){
			cout << "error: " << e.what() << "\n";
		}
		for(int i=0;i<in_size;i++){
			for(int j=0;j<out_size;j++){
				host_array[i*out_size+j] = h_arr[j*in_size+i];
			}
		}
		cudaMalloc(&weights, out_size * in_size * sizeof(float));
		cudaMemcpy(weights, host_array, out_size * in_size * sizeof(float), cudaMemcpyHostToDevice);
		delete[] h_arr;
		delete[] host_array;
	};

	void read_biases(string pathToWeights){
		float *h_arr = new float [out_size];
		try{
			ifstream s(pathToWeights);
			for (int i = 0; i < out_size; i++){
                s >> h_arr[i];
            }
			s.close();
		}
		catch(exception const& e){
			cout << "error: " << e.what() << "\n";
		}
		cudaMalloc(&biases, out_size * sizeof(float));
		cudaMemcpy(biases, h_arr, out_size*sizeof(float), cudaMemcpyHostToDevice);
		delete[] h_arr;
	};

public:
	Layer(){
		in_size = 32 * 32;
		out_size = 1;
		alpha = 1.0;
		beta = 1.0;
	};

	Layer(string pathToWeights, string pathToBiases, int inSize, int outSize){
		alpha = 1.0;
		beta = 1.0;
		in_size = inSize;
		out_size = outSize;
		read_weights(pathToWeights);
		read_biases(pathToBiases);
	};

	float* Linear(float* input){
		cublasCreate(&handle);
		cublasSgemv(handle, CUBLAS_OP_N, out_size, in_size, &alpha, weights,out_size, input, 1, &beta, biases, 1);
		cublasDestroy(handle);
		sigmoid<<<1, out_size>>> (biases, out_size);
		return biases;
	};

	~Layer(){
		cudaFree(weights);
		cudaFree(biases);
	};
};

class Model
{
private:
	float *array;
	int in_size, out_size;



	void read(string path){
		float *inp_arr = new float [in_size];
		try{
			ifstream s(path);
			for (int i = 0; i < in_size; i++){
                s >> inp_arr[i];
            }
			s.close();
		}
		catch(exception const& e){
			cout << "error: " << e.what() << "\n";
		}
		cudaMalloc(&array, in_size * sizeof(float));
		cudaMemcpy(array, inp_arr, in_size*sizeof(float), cudaMemcpyHostToDevice);
		delete[] inp_arr;
	};


	void print_res(float* arr){
		float* h_arr = new float[out_size];
		cudaMemcpy(h_arr, arr, out_size*sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Result: " << "\n";
		for (int i = 0; i < out_size; i++){
			cout << h_arr[i] << "\n";
		}
		check(h_arr[0]);
		delete[] h_arr;
	};

public:
	Model(){
		in_size = 32 * 32;
		out_size = 1;
	};

	void forward(string path){
		read(path);
		Layer layer1("w1.bin", "b1.bin", 32 * 32, 16 * 16);
		array = layer1.Linear(array);
		Layer layer2("w2.bin", "b2.bin", 16 * 16, 4 * 4);
		array = layer2.Linear(array);
		Layer layer3("w3.bin", "b3.bin", 4 * 4, 1);
		array = layer3.Linear(array);
		print_res(array);
	}

	~Model(){
		cudaFree(array);
	};
};

int main()
{
	Model model;
	model.forward("inp.bin");
	return 0;
}