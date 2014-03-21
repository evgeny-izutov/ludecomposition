#include <iostream>
#include "mkl.h"

double* ReadMatrixFromFile(char *fileName, int &N) {
	FILE *pFile = fopen(fileName, "r");
    if (!pFile) {
        printf("File %s not found!\n", fileName);
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

	double *A = (double*) mkl_malloc(N*N*sizeof(double), 32);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			fscanf(pFile, "%Lf", &A[i*N + j]);
		}
    }

	fclose(pFile);

	return A;
}

int main(int argc, char *argv[]) {
	if (argc < 4) {
        printf("Error!\n");
        return 0;
    }

	int N;
	double *A = ReadMatrixFromFile(argv[1], N);
	double *L = ReadMatrixFromFile(argv[2], N);
	double *U = ReadMatrixFromFile(argv[3], N);
	double *gA = (double*) mkl_malloc(N*N*sizeof(double), 32);

	if (!A || !L || !U) {
		printf("Error!\n");
		return 0;
	}

	const int h = N, w = N, lda = N;

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, L, N, U, N, 0.0, gA, N);
	cblas_daxpy(N*N, -1.0, A, 1, gA, 1); 

	char normType = 'F';
	double normA = dlange(&normType, &h, &w, A, &lda, 0);
	double normDiff = dlange(&normType, &h, &w, gA, &lda, 0);

	printf("Norm A: %.16f\n", normA);
	printf("Norm difference: %.16f\n", normDiff);
	printf("(Norm difference)/(Norm A): %.16f\n", normDiff/normA);
		
	mkl_free(A);
	mkl_free(L);
	mkl_free(U);
	mkl_free(gA);

	return 0;
}