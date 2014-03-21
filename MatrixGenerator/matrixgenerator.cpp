#include <iostream>
#include <Windows.h>
#include <random>

int main(int argc, char *argv[]) {
	if (argc < 5) {
        printf("Error!\n");
        return 0;
    }

	char *matrixFilename = argv[1];
	int m = atoi(argv[2]);
    int n = atoi(argv[3]);
	double absMaxMatrixValue = atof(argv[4]);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-absMaxMatrixValue, absMaxMatrixValue);

	FILE *pFile = fopen(matrixFilename, "w");
	fprintf(pFile, "%d %d\n", m, n);
	for (int i = 0; i < m - 1; i++) {
		for (int j = 0; j < n - 1; j++) {
			double value = dis(gen);
			fprintf(pFile, "%.16f ", value);
		}
		fprintf(pFile, "%.16f\n", dis(gen));
	}
	for (int j = 0; j < n - 1; j++) {
		fprintf(pFile, "%.16f ", dis(gen));
	}
	fprintf(pFile, "%.16f", dis(gen));
	fclose(pFile);

	return 0;
}