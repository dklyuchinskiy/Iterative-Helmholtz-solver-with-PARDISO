#pragma once

/*****************************
Preprocessor definitions and
declaration of used structures
*****************************/

#include "../definitions.h"

struct BinaryMatrixTreeNode {

	int p = 0;
	double *U = NULL;
	double *VT = NULL;
	double *A = NULL;
	struct BinaryMatrixTreeNode *left;
	struct BinaryMatrixTreeNode *right;
};

typedef struct BinaryMatrixTreeNode mnode;

struct ComplexBinaryMatrixTreeNode 
{
	int n2 = 0;
	int p = 0;
	int n1 = 0;
	dtype *U = NULL;
	dtype *VT = NULL;
	dtype *A = NULL;
	struct ComplexBinaryMatrixTreeNode *left;
	struct ComplexBinaryMatrixTreeNode *right;
	bool solve = false;
};

typedef struct ComplexBinaryMatrixTreeNode cmnode;

struct ComplexBinaryUnsymmetricMatrixTreeNode 
{
	cmnode *A21;
	cmnode *A12;
	struct ComplexBinaryUnsymmetricMatrixTreeNode *left = NULL;
	struct ComplexBinaryUnsymmetricMatrixTreeNode *right = NULL;
};

typedef struct ComplexBinaryUnsymmetricMatrixTreeNode cumnode;

struct list {
	cmnode* node;
	struct list* next;
};

typedef struct list qlist;

struct list2 {
	cumnode* node;
	struct list2* next;
};

typedef struct list2 qlist2;

struct my_queue {
	struct list *first, *last;
};

struct my_queue2 {
	struct list2 *first, *last;
};

#define STRUCT_CSR

#ifdef STRUCT_CSR
#define ONLINE
#endif

#define CHOLESKY

//#define FULL_SVD

//#define DIM_3D

//#define COL_UPDATE
//#define COL_ADD


// Функция выделения памяти под массив

template <typename MatrixType>
double RelError(double(*LANGE)(const char *, const int*, const int*, const MatrixType*, const int*, double *),
	int m, int n, const MatrixType *Hrec, int ldh1, const MatrixType *Hinit, int ldh2, double eps)
{
	double norm = 0;
	MatrixType *Hdiff = alloc_arr<MatrixType>(m * n);
	int ldh = m;

	// Norm of residual
#pragma omp parallel for schedule(static)
	for (int j = 0; j < n; j++)
#pragma omp simd
		for (int i = 0; i < m; i++)
			Hdiff[i + ldh * j] = Hrec[i + ldh1 * j] - Hinit[i + ldh2 * j];

	norm = LANGE("Frob", &m, &n, Hdiff, &ldh, NULL);
	norm = norm / LANGE("Frob", &m, &n, Hinit, &ldh2, NULL);

	free_arr(Hdiff);

#if 0
	if (norm < eps) printf("Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
	else printf("Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);
#endif

	return norm;
}

template <typename MatrixType>
double RelErrorPart(double(*LANGE)(const char *, const int*, const int*, const MatrixType*, const int*, double *),
	char part, int m, int n, const MatrixType *Hrec, int ldh1, const MatrixType *Hinit, int ldh2, double eps)
{
	double norm = 0;
	MatrixType *Hdiff = alloc_arr<MatrixType>(m * n);
	int ldh = m;

	if (part == 'L')
	{
		// Norm of residual
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
#pragma omp simd
			for (int i = j; i < m; i++)
				Hdiff[i + ldh * j] = Hrec[i + ldh1 * j] - Hinit[i + ldh2 * j];
	}
	else if (part == 'U')
	{
#pragma omp parallel for schedule(static)
		for (int i = 0; i < m; i++)
#pragma omp simd
			for (int j = i; j < n; j++)
				Hdiff[i + ldh * j] = Hrec[i + ldh1 * j] - Hinit[i + ldh2 * j];
	}
	else
	{
		// Norm of residual
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
#pragma omp simd
			for (int i = 0; i < m; i++)
				Hdiff[i + ldh * j] = Hrec[i + ldh1 * j] - Hinit[i + ldh2 * j];
	}

//#define PRINT
#ifdef PRINT
	printf("diff\n");
	PrintMat(m, n, Hdiff, ldh);
#endif

	norm = LANGE("Frob", &m, &n, Hdiff, &ldh, NULL);
	norm = norm / LANGE("Frob", &m, &n, Hinit, &ldh2, NULL);

	free_arr(Hdiff);

#if 0
	if (norm < eps) printf("Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
	else printf("Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);
#endif

	return norm;
}

template <typename MatrixType>
double AbsError(double(*LANGE)(const char *, const int*, const int*, const MatrixType*, const int*, double *),
	int m, int n, const MatrixType *Hrec, int ldh1, const MatrixType *Hinit, int ldh2, double eps)
{
	double norm = 0;
	MatrixType *Hdiff = alloc_arr<MatrixType>(m * n);
	int ldh = m;

	// Norm of residual
#pragma omp parallel for schedule(static)
	for (int j = 0; j < n; j++)
#pragma omp simd
		for (int i = 0; i < m; i++)
			Hdiff[i + ldh * j] = Hrec[i + ldh1 * j] - Hinit[i + ldh2 * j];

	norm = LANGE("Frob", &m, &n, Hdiff, &ldh, NULL);

	free_arr(Hdiff);

#if 0
	if (norm < eps) printf("Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
	else printf("Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);
#endif

	return norm;
}





