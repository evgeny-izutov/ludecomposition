#include <iostream>
#include <fstream>
#include <windows.h>
#include <malloc.h>
#include <cilk\cilk.h>

using namespace std;

#define BlockSize 100
#define BlockGrainSize 150 //BlockGrainSize >= BlockSize; Good if BlockGrainSize < 2*BlockSize

double* ReadMatrixFromFile(char *fileName, int &N) {
	FILE *pFile = fopen(fileName, "r");
    if (!pFile) {
        cout << "Input file not found\n";
        return 0;
    }

	int h, w;
	fscanf(pFile, "%d %d", &h, &w);
	if (h != w) {
		fclose(pFile);
		printf("Error!\n");
        return 0;
	}
	else {
		N = h;
	}

	double *A = (double*)_mm_malloc(N*N*sizeof(double), 32);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			fscanf(pFile, "%lf", &A[i*N + j]);
		}
    }

	fclose(pFile);

	return A;
}

void WriteLMatrixToFile(char *fileName, const double *L, int N) {
	FILE *pFile = fopen(fileName, "w");
    if (!pFile) {
        printf("File %s not found!\n", fileName);
		return;
    }

	fprintf(pFile, "%d %d\n", N, N);
	for (int i = 0; i < N - 1; i++) {
		for (int j = 0; j <= i; j++) {
			fprintf(pFile, "%.16f ", L[i*(i + 1)/2 + j]);
		}
		for (int j = i + 1; j < N - 1; j++) {
			fprintf(pFile, "%.16f ", 0.0);
		}
		fprintf(pFile, "%.16f\n", 0.0);
    }
	for (int j = 0; j < N - 1; j++) {
		fprintf(pFile, "%.16f ", L[(N - 1)*N/2 + j]);
	}
	fprintf(pFile, "%.16f", 1.0);

	fclose(pFile);
}

void WriteUMatrixToFile(char *fileName, const double *U, int N) {
	FILE *pFile = fopen(fileName, "w");
    if (!pFile) {
        printf("File %s not found!\n", fileName);
		return;
    }

	fprintf(pFile, "%d %d\n", N, N);
	for (int i = 0; i < N - 1; i++) {
		for (int j = 0; j < i; j++) {
			fprintf(pFile, "%.16f ", 0.0);
		}
		for (int j = i; j < N - 1; j++) {
			fprintf(pFile, "%.16f ", U[j*(j + 1)/2 + i]);
		}
		fprintf(pFile, "%.16f\n", U[(N - 1)*N/2 + i]);
    }
	for (int j = 0; j < N - 1; j++) {
		fprintf(pFile, "%.16f ", 0.0);
	}
	fprintf(pFile, "%.16f", U[(N - 1)*N/2 + N - 1]);

	fclose(pFile);
}

void WriteTimeToFile(char *fileName, double time) {
	FILE *pFile = fopen(fileName, "w");
	if (!pFile) {
        printf("File %s not found!\n", fileName);
		return;
    }

	fprintf(pFile, "%.16f", time);

	fclose(pFile);
}

void DiagonalSubmatrixLUDecompose(int bias, int squareSize, int matrixSize, const double *A, double *L, double *U) {
	int N = bias + squareSize;

	int i, j, k;
	double sum;

	L[bias*(bias + 3)/2] = 1.0;
	for (j = bias; j < N; j++) {
		U[j*(j + 1)/2 + bias] = A[bias*matrixSize + j];
	}

	for (i = bias + 1; i < N; i++) {
		L[i*(i + 1)/2 + bias] = A[i*matrixSize + bias]/U[bias*(bias + 3)/2];
		for (j = bias; j < i; j++) {
			sum = 0.0;
			for (k = bias; k < j; k++) {
				sum += L[i*(i + 1)/2 + k]*U[j*(j + 1)/2 + k];
			}
			L[i*(i + 1)/2 + j] = (A[i*matrixSize + j] - sum)/U[j*(j + 3)/2];
		}
		L[i*(i + 3)/2] = 1.0;
		for (j = i; j < N; j++) {
			sum = 0.0;
			for (k = bias; k < i; k++) {
				sum += L[i*(i + 1)/2 + k]*U[j*(j + 1)/2 + k];
			}
			U[j*(j + 1)/2 + i] = A[i*matrixSize + j] - sum;
		}
	}
}

void SolveUpperEquation(int bias, int squareSize, int matrixSize, const double *A, const double *L, double *U) {

	cilk_for (int j = bias + squareSize; j < matrixSize; j++) 
	{
		U[j*(j + 1)/2 + bias] = A[bias*matrixSize + j];
		for (int i = bias + 1; i < bias + squareSize; i++) {
			double sum = 0.0;
			for (int k = bias; k < i; k++) {
				sum += U[j*(j + 1)/2 + k]*L[i*(i + 1)/2 + k];
			}
			U[j*(j + 1)/2 + i] = A[i*matrixSize + j] - sum;
		}
	}
}

void SolveLeftEquation(int bias, int squareSize, int matrixSize, const double *A, double *L, const double *U) {
	int i, j, k;
	double sum;

	for (i = bias + squareSize; i < matrixSize; i++) {
		L[i*(i + 1)/2 + bias] = A[i*matrixSize + bias]/U[bias*(bias + 3)/2];
		for (j = bias + 1; j < bias + squareSize; j++) {
			sum = 0;
			for (k = bias; k < j; k++) {
				sum += L[i*(i + 1)/2 + k]*U[j*(j + 1)/2 + k];
			}
			L[i*(i + 1)/2 + j] = (A[i*matrixSize + j] - sum)/U[j*(j + 3)/2];
		}
	}
}

void UpdateDiagonalSubmatrix(int bias, int squareSize, int matrixSize, double *A, double *L, double *U) {
	int i, j, k;
	double sum;

	for (i = bias + squareSize; i < matrixSize; i++) {
		for (j = bias + squareSize; j < matrixSize; j++) {
			sum = 0;
			for (k = bias; k < bias + squareSize; k++) {
				sum += L[i*(i + 1)/2 + k]*U[j*(j + 1)/2 + k];
			}
			A[i*matrixSize + j] -= sum;
		}
	}
}

void LUDecompose(int matrixSize, double *A, double *L, double *U) {
	for (int bias = 0; bias < matrixSize; bias += BlockSize) {
		if (matrixSize - bias <= BlockGrainSize) {
			DiagonalSubmatrixLUDecompose(bias, matrixSize - bias, matrixSize, A, L, U);
			break;
		}
		DiagonalSubmatrixLUDecompose(bias, BlockSize, matrixSize, A, L, U);
		SolveUpperEquation(bias, BlockSize, matrixSize, A, L, U);
		SolveLeftEquation(bias, BlockSize, matrixSize, A, L, U);
		UpdateDiagonalSubmatrix(bias, BlockSize, matrixSize, A, L, U);
	}
}

int main(int argc, char *argv[]) {
	if(argc < 5) {
        printf("Error!\n");
        return 0;
    }

	double *A, *L, *U;
	int N;

	A = ReadMatrixFromFile(argv[1], N);
	if (!A) {
		return 0;
	}

	int triangularMatrixSize = N*(N + 1)/2;
	L = (double*)_mm_malloc(triangularMatrixSize*sizeof(double), 32);
	U = (double*)_mm_malloc(triangularMatrixSize*sizeof(double), 32);
	
	LARGE_INTEGER LIFrequency;
	QueryPerformanceFrequency(&LIFrequency);
	double pcFreq = (double) LIFrequency.QuadPart;

	LARGE_INTEGER start, finish;
	QueryPerformanceCounter(&start);
	LUDecompose(N, A, L, U);
    QueryPerformanceCounter(&finish);
	
	WriteLMatrixToFile(argv[2], L, N);
	WriteUMatrixToFile(argv[3], U, N);
	WriteTimeToFile(argv[4], (finish.QuadPart - start.QuadPart)/pcFreq);
	
	_mm_free(A);
	_mm_free(L);
	_mm_free(U);
	
	return 0;
}