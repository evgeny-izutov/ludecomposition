# include "mpi.h"
#include <iostream>
#include <fstream>
#include <windows.h>
#include <malloc.h>

using namespace std;

#define BlockSize 100
#define BlockGrainSize 150 //BlockGrainSize >= BlockSize; Good if BlockGrainSize < 2*BlockSize
const int triangleBufferSize = BlockSize*(BlockSize + 1)/2;

int procRank, nextProcRank, prevProcRank, procCount;
double *A, *L, *U;
int N;
int *blocklengths, *displacements;
double *LTriangle, *UTriangle, *LSquare, *USquare, *leftPart, *upPart, *result;
int maxLocalBlockHeight;
int *localBlockHeights;
double startTime, finishTime;
MPI_Datatype *triangleBlockTypes, *leftSquareBlockTypes, *upSquareBlockTypes, *resultBlockTypes;
MPI_Request *requestsLOnRecv, *requestsUOnRecv, *requestsResultOnRecv, *requestsLeftOnSend, *requestsUpOnSend, *requestsResultOnSend;
MPI_Request requestResultOnRecv, requestLOnSend, requestUOnSend;
MPI_Status status, *recvStatuses;

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

void AllocateSourceData() {
	int triangularMatrixSize = N*(N + 1)/2;
	L = (double*)_mm_malloc(triangularMatrixSize*sizeof(double), 32);
	U = (double*)_mm_malloc(triangularMatrixSize*sizeof(double), 32);
}

void FreeSourceData() {
	_mm_free(A);
	_mm_free(L);
	_mm_free(U);
}

void AllocateBuffers() {
	LTriangle = (double*)_mm_malloc(triangleBufferSize*sizeof(double), 32);
	UTriangle = (double*)_mm_malloc(triangleBufferSize*sizeof(double), 32);

	maxLocalBlockHeight = (N - BlockSize)/procCount;
	if ((N - BlockSize)%procCount != 0) {
		maxLocalBlockHeight = N - BlockSize - maxLocalBlockHeight*(procCount - 1);
	}
	blocklengths = (int*)_mm_malloc(maxLocalBlockHeight*sizeof(int), 32);
	displacements = (int*)_mm_malloc(maxLocalBlockHeight*sizeof(int), 32);
	
	LSquare = (double*)_mm_malloc(BlockSize*maxLocalBlockHeight*sizeof(double), 32);
	USquare = (double*)_mm_malloc(BlockSize*maxLocalBlockHeight*sizeof(double), 32);
	leftPart = (double*)_mm_malloc(BlockSize*maxLocalBlockHeight*sizeof(double), 32);
	upPart = (double*)_mm_malloc(BlockSize*maxLocalBlockHeight*sizeof(double), 32);

	result = (double*)_mm_malloc((N - BlockSize)*maxLocalBlockHeight*sizeof(double), 32);

	localBlockHeights = (int*)_mm_malloc(procCount*sizeof(int), 32);
	triangleBlockTypes = (MPI_Datatype*)_mm_malloc(procCount*sizeof(MPI_Datatype), 32);
	leftSquareBlockTypes = (MPI_Datatype*)_mm_malloc(procCount*sizeof(MPI_Datatype), 32);
	upSquareBlockTypes = (MPI_Datatype*)_mm_malloc(procCount*sizeof(MPI_Datatype), 32);
	resultBlockTypes = (MPI_Datatype*)_mm_malloc(procCount*sizeof(MPI_Datatype), 32);
	
	if (procRank == 0) {
		requestsLOnRecv = (MPI_Request*)_mm_malloc(procCount*sizeof(MPI_Request), 32);
		requestsUOnRecv = (MPI_Request*)_mm_malloc(procCount*sizeof(MPI_Request), 32);
		requestsResultOnRecv = (MPI_Request*)_mm_malloc(procCount*sizeof(MPI_Request), 32);
		requestsLeftOnSend = (MPI_Request*)_mm_malloc(procCount*sizeof(MPI_Request), 32);
		requestsUpOnSend = (MPI_Request*)_mm_malloc(procCount*sizeof(MPI_Request), 32);
		requestsResultOnSend = (MPI_Request*)_mm_malloc(procCount*sizeof(MPI_Request), 32);
		recvStatuses = (MPI_Status*)_mm_malloc(procCount*sizeof(MPI_Status), 32);
	}
}

void FreeBuffers() {
	_mm_free(LTriangle);
	_mm_free(UTriangle);
	_mm_free(LSquare);
	_mm_free(USquare);
	_mm_free(leftPart);
	_mm_free(upPart);
	_mm_free(result);
	_mm_free(blocklengths);
	_mm_free(displacements);
	_mm_free(localBlockHeights);
	_mm_free(triangleBlockTypes);
	_mm_free(leftSquareBlockTypes);
	_mm_free(upSquareBlockTypes);
	_mm_free(resultBlockTypes);

	if (procRank == 0) {
		_mm_free(requestsLOnRecv);
		_mm_free(requestsUOnRecv);
		_mm_free(requestsResultOnRecv);
		_mm_free(requestsLeftOnSend);
		_mm_free(requestsUpOnSend);
		_mm_free(requestsResultOnSend);
		_mm_free(recvStatuses);
	}
}

void CalculateLocalBlockHeights(int bias) {
	int firstBlockHeight = (N - BlockSize - bias)/procCount;
	for (int i = 0; i < procCount - 1; i++) {
		localBlockHeights[i] = firstBlockHeight;
	}
	int lastBlockHeight = firstBlockHeight;
	if ((N - BlockSize - bias)%procCount != 0) {
		lastBlockHeight = N - BlockSize - bias - firstBlockHeight*(procCount - 1);
	}
	localBlockHeights[procCount - 1] = lastBlockHeight;
}

void CommitTriangleBlockTypes(int bias) {
	int iIndex = bias + BlockSize ;
	int jIndex = bias;
	for (int i = 0; i < procCount; i++) {
		for (int k = 0; k < localBlockHeights[i]; k++) {
			blocklengths[k] = BlockSize;
			displacements[k] = (iIndex + k)*(iIndex + k + 1)/2 + jIndex;
		}
		MPI_Type_indexed(localBlockHeights[i], blocklengths, displacements, MPI_DOUBLE, &triangleBlockTypes[i]);
		MPI_Type_commit(&triangleBlockTypes[i]);
		iIndex += localBlockHeights[i];
	}
}

void FreeTriangleBlockTypes() {
	for (int i = 1; i < procCount; i++) {
		MPI_Type_free(&triangleBlockTypes[i]);
	}
}

void CommitSquareBlockTypes(int bias) {
	int iIndex = bias + BlockSize ;
	int jIndex = bias;
	for (int i = 0; i < procCount; i++) {
		for (int k = 0; k < localBlockHeights[i]; k++) {
			blocklengths[k] = BlockSize;
			displacements[k] = (iIndex + k)*N + jIndex;
		}
		MPI_Type_indexed(localBlockHeights[i], blocklengths, displacements, MPI_DOUBLE, &leftSquareBlockTypes[i]);
		MPI_Type_commit(&leftSquareBlockTypes[i]);
		iIndex += localBlockHeights[i];
	}

	iIndex = bias;
	jIndex = bias + BlockSize;
	for (int i = 0; i < procCount; i++) {
		for (int k = 0; k < BlockSize; k++) {
			blocklengths[k] = localBlockHeights[i];
			displacements[k] = (iIndex + k)*N + jIndex;
		}
		MPI_Type_indexed(BlockSize, blocklengths, displacements, MPI_DOUBLE, &upSquareBlockTypes[i]);
		MPI_Type_commit(&upSquareBlockTypes[i]);
		jIndex += localBlockHeights[i];
	}

	iIndex = bias + BlockSize ;
	jIndex = bias + BlockSize;
	int blockWidth = N - bias - BlockSize;
	for (int i = 0; i < procCount; i++) {
		for (int k = 0; k < localBlockHeights[i]; k++) {
			blocklengths[k] = blockWidth;
			displacements[k] = (iIndex + k)*N + jIndex;
		}
		MPI_Type_indexed(localBlockHeights[i], blocklengths, displacements, MPI_DOUBLE, &resultBlockTypes[i]);
		MPI_Type_commit(&resultBlockTypes[i]);
		iIndex += localBlockHeights[i];
	}
}

void FreeSquareBlockTypes() {
	for (int i = 1; i < procCount; i++) {
		MPI_Type_free(&leftSquareBlockTypes[i]);
		MPI_Type_free(&upSquareBlockTypes[i]);
		MPI_Type_free(&resultBlockTypes[i]);
	}
}

void DataDistribution(int bias) {
	if (procRank == 0) {
		int border = bias + BlockSize;
		int index = 0;
		for (int i = bias; i < border; i++) {
			for (int j = bias; j <= i; j++) {
				LTriangle[index] = L[i*(i + 1)/2 + j];
				UTriangle[index] = U[i*(i + 1)/2 + j];
				index++;
			}
		}
	}
	
	MPI_Bcast(LTriangle, triangleBufferSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(UTriangle, triangleBufferSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (procRank == 0) {	
		for (int i = 0; i < procCount; i++) {
			MPI_Isend(A, 1, leftSquareBlockTypes[i], i, 0, MPI_COMM_WORLD, &requestsLeftOnSend[i]);
			MPI_Isend(A, 1, upSquareBlockTypes[i], i, 1, MPI_COMM_WORLD, &requestsUpOnSend[i]);
			MPI_Isend(A, 1, resultBlockTypes[i], i, 2, MPI_COMM_WORLD, &requestsResultOnSend[i]);
		}
	}

	MPI_Recv(leftPart, BlockSize*maxLocalBlockHeight, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
	MPI_Recv(upPart, BlockSize*maxLocalBlockHeight, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
	MPI_Irecv(result, (N - BlockSize)*maxLocalBlockHeight, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &requestResultOnRecv);
}

void DiagonalSubmatrixLUDecompose(int bias, int squareSize) {
	int border = bias + squareSize;

	L[bias*(bias + 3)/2] = 1.0;
	for (int j = bias; j < border; j++) {
		U[j*(j + 1)/2 + bias] = A[bias*N + j];
	}

	for (int i = bias + 1; i < border; i++) {
		L[i*(i + 1)/2 + bias] = A[i*N + bias]/U[bias*(bias + 3)/2];
		for (int j = bias; j < i; j++) {
			double sum = 0.0;
			for (int k = bias; k < j; k++) {
				sum += L[i*(i + 1)/2 + k]*U[j*(j + 1)/2 + k];
			}
			L[i*(i + 1)/2 + j] = (A[i*N + j] - sum)/U[j*(j + 3)/2];
		}
		L[i*(i + 3)/2] = 1.0;
		for (int j = i; j < border; j++) {
			double sum = 0.0;
			for (int k = bias; k < i; k++) {
				sum += L[i*(i + 1)/2 + k]*U[j*(j + 1)/2 + k];
			}
			U[j*(j + 1)/2 + i] = A[i*N + j] - sum;
		}
	}
}

void SolveUpperEquation(int bias) {
	int columnsCount = localBlockHeights[procRank];
	for (int i = 0; i < columnsCount; i++) {
		USquare[i*BlockSize] = upPart[i];
		for (int j = 1; j < BlockSize; j++) {
			double sum = 0.0;
			for (int k = 0; k < j; k++) {
				sum += USquare[i*BlockSize + k]*LTriangle[j*(j + 1)/2 + k];
			}
			USquare[i*BlockSize + j] = upPart[j*columnsCount + i] - sum;
		}
	}

	if (procRank == 0) {
		for (int i = 0; i < procCount; i++) {
			MPI_Irecv(U, 1, triangleBlockTypes[i], i, 1, MPI_COMM_WORLD, &requestsUOnRecv[i]);
		}
	}

	MPI_Isend(USquare, BlockSize*localBlockHeights[procRank], MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &requestUOnSend);
}

void SolveLeftEquation(int bias) {
	int rowsCount = localBlockHeights[procRank];
	for (int i = 0; i < rowsCount; i++) {
		LSquare[i*BlockSize] = leftPart[i*BlockSize]/UTriangle[0];
		for (int j = 1; j < BlockSize; j++) {
			double sum = 0.0;
			for (int k = 0; k < j; k++) {
				sum += LSquare[i*BlockSize + k]*UTriangle[j*(j + 1)/2 + k];
			}
			LSquare[i*BlockSize + j] = (leftPart[i*BlockSize + j] - sum)/UTriangle[j*(j + 3)/2];
		}
	}

	if (procRank == 0) {
		for (int i = 0; i < procCount; i++) {
			MPI_Irecv(L, 1, triangleBlockTypes[i], i, 0, MPI_COMM_WORLD, &requestsLOnRecv[i]);
		}
	}

	MPI_Isend(LSquare, BlockSize*localBlockHeights[procRank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requestLOnSend);
}

void UpdateDiagonalSubmatrix(int bias) {
	int localLBlockSize = localBlockHeights[procRank];
	int sourceProcRank = procRank;

	MPI_Wait(&requestUOnSend, &status);
	MPI_Wait(&requestResultOnRecv, &status);

	for (int iter = 0; iter < procCount; iter++) {
		int localUBlockSize = localBlockHeights[sourceProcRank];

		int jIndexBias = 0;
		for (int i = 0; i < sourceProcRank; i++) {
			jIndexBias += localBlockHeights[i];
		}
		
		for (int i = 0; i < localLBlockSize; i++) {
			for (int j = 0; j < localUBlockSize; j++) {
				double sum = 0.0;
				for (int k = 0; k < BlockSize; k++) {
					sum += LSquare[i*BlockSize + k]*USquare[j*BlockSize + k];
				}
				result[i*(N - bias - BlockSize) +  jIndexBias + j] -= sum;
			}
		}

		MPI_Sendrecv_replace(USquare, BlockSize*maxLocalBlockHeight, MPI_DOUBLE, nextProcRank, 0, prevProcRank, 0, MPI_COMM_WORLD, &status);

		sourceProcRank--;
		if (sourceProcRank < 0) {
			sourceProcRank = procCount - 1;
		}
	}

	if (procRank == 0) {
		for (int i = 0; i < procCount; i++) {
			MPI_Irecv(A, 1, resultBlockTypes[i], i, 0, MPI_COMM_WORLD, &requestsResultOnRecv[i]);
		}
	}
	MPI_Send(result, localBlockHeights[procRank]*(N - bias - BlockSize), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
}

void Synchronize() {
	if (procRank == 0) {
		MPI_Waitall(procCount, requestsLOnRecv, recvStatuses);
		MPI_Waitall(procCount, requestsUOnRecv, recvStatuses);
		MPI_Waitall(procCount, requestsResultOnRecv, recvStatuses);
	}

	MPI_Wait(&requestLOnSend, &status);
}

void LUDecompose() {
	for (int bias = 0; bias < N; bias += BlockSize) {
		if (N - bias <= BlockGrainSize) {
			if (procRank == 0) {
				DiagonalSubmatrixLUDecompose(bias, N - bias);
			}
			break;
		}
		if (procRank == 0) {
			DiagonalSubmatrixLUDecompose(bias, BlockSize);
		}

		CalculateLocalBlockHeights(bias);

		CommitTriangleBlockTypes(bias);
		CommitSquareBlockTypes(bias);

		DataDistribution(bias);
		SolveLeftEquation(bias);
		SolveUpperEquation(bias);
		UpdateDiagonalSubmatrix(bias);

		FreeTriangleBlockTypes();
		FreeSquareBlockTypes();

		Synchronize();
	}
}

int main(int argc, char *argv[]) {
	if(argc < 5) {
        return 0;
    }

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &procCount);
	MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
	nextProcRank = procRank + 1;
	if (nextProcRank >= procCount) {
		nextProcRank = 0;
	}
	prevProcRank = procRank - 1;
	if (prevProcRank < 0) {
		prevProcRank = procCount - 1;
	}

	if (procRank == 0) {
		A = ReadMatrixFromFile(argv[1], N);
		
		if (!A) {
			MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
			return 0;
		}

		AllocateSourceData();

		startTime = MPI_Wtime();
	}

	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

	AllocateBuffers();

	LUDecompose();

	FreeBuffers();

	MPI_Barrier(MPI_COMM_WORLD);
	if (procRank == 0) {
		finishTime = MPI_Wtime();
		WriteLMatrixToFile(argv[2], L, N);
		WriteUMatrixToFile(argv[3], U, N);
		WriteTimeToFile(argv[4], finishTime - startTime);

		FreeSourceData();
	}
    
	MPI_Finalize();

	return 0;
}