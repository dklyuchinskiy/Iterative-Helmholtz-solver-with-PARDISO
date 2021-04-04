#include "../templates.h"
#include "definitionsHODLR.h"
#include "templatesHODLR.h"
#include "TestSuiteHODLR.h"

/********************************************************
Source file contains functionality to work
with compressed matrices with HSS structure, as presented
in functions.cpp (for example, Add, Mult, Inverse and etc.).

But now, all algorithms to work with compressed matrices
are implemented by using Binary Tree Structure, defined
in definitions.h.

In addition, the computation of Size, MaxDepth
and Ranks of the matrix tree is presented here.
********************************************************/

int TreeSize(cmnode* root)
{
	int size = 0;

	if (root == NULL)
	{
		return 0;
	}
	else
	{
		int ld = 0, rd = 0;
#pragma omp task shared(ld)
		{
			ld = TreeSize(root->left);
		}
#pragma omp task shared(rd)
		{
			rd = TreeSize(root->right);
		}
#pragma omp taskwait
		return ld + 1 + rd;
	}

}

int MaxDepth(cmnode* root)
{
	if (root == NULL)
	{
		return 0;
	}
	else
	{
		int ld = 0, rd = 0;
#pragma omp task shared(ld)
		{
			ld = MaxDepth(root->left);
		}
#pragma omp task shared(rd)
		{
			rd = MaxDepth(root->right);
		}
#pragma omp taskwait
		if (ld > rd) return (ld + 1); // + root node
		else return (rd + 1);
	}

}

int CountElementsInMatrixTree(int n, cmnode* root)
{
	if (root->left == NULL && root->right == NULL)
	{
		return n * n;
	}
	else
	{
		int n2 = ceil(n / 2.0);
		int n1 = n - n2;
		int ld = 0, rd = 0;

		if (root->left != NULL) ld = CountElementsInMatrixTree(n1, root->left);
		if (root->right != NULL) rd = CountElementsInMatrixTree(n2, root->right);

		return ld + rd + root->p * (n1 + n2);
	}
}

int GetNumberOfLeaves(cumnode *root)
{
	if (root->left == NULL && root->right == NULL)
	{
		return 1;
	}
	else
	{
		int ld = 0, rd = 0;
		if (root->left != NULL) ld = GetNumberOfLeaves(root->left);
		if (root->right != NULL) rd = GetNumberOfLeaves(root->right);

		return ld + rd;
	}
}

void GetDistances(cumnode *root, int *dist, int &count)
{
	if (root->left == NULL && root->right == NULL)
	{		
		dist[count++] = root->A21->n1;
	}
	else
	{
		if (root->left != NULL) GetDistances(root->left, dist, count);
		if (root->right!= NULL) GetDistances(root->right, dist, count);
	}
}

void PrintRanks(mnode* root)
{
	if (root == NULL)
	{
		return;
	}
	else
	{
		printf("%3d ", root->p);
		PrintRanks(root->left);
		PrintRanks(root->right);
	}
}

void PrintRanksInWidth(cmnode *root)
{
	if (root == NULL)
	{
		return;
	}
	queue<cmnode*> q; // Создаем очередь
	q.push(root); // Вставляем корень в очередь

	while (!q.empty()) // пока очередь не пуста
	{
		cmnode* temp = q.front(); // Берем первый элемент в очереди
		q.pop();  // Удаляем первый элемент в очереди
		printf("%3d ", temp->p); // Печатаем значение первого элемента в очереди

		if (temp->left != NULL)
			q.push(temp->left);  // Вставляем  в очередь левого потомка

		if (temp->right != NULL)
			q.push(temp->right);  // Вставляем  в очередь правого потомка
	}
}

void UnsymmPrintRanksInWidth(cumnode *root)
{
	if (root == NULL)
	{
		return;
	}
	queue<cumnode*> q; // Создаем очередь
	q.push(root); // Вставляем корень в очередь

	while (!q.empty()) // пока очередь не пуста
	{
		cumnode* temp = q.front(); // Берем первый элемент в очереди
		q.pop();  // Удаляем первый элемент в очереди
		printf("A21->p: %3d ", temp->A21->p); // Печатаем значение первого элемента в очереди
		printf("A12->p: %3d ", temp->A12->p);

		if (temp->left != NULL)
			q.push(temp->left);  // Вставляем  в очередь левого потомка

		if (temp->right != NULL)
			q.push(temp->right);  // Вставляем  в очередь правого потомка
	}
}


void PrintRanksInWidthList(cmnode *root)
{
	if (root == NULL)
	{
		return;
	}
	struct my_queue* q; // Создаем очередь
	init(q);
	push(q, root); // Вставляем корень в очередь

#ifdef DEBUG
	print_queue(q);
#endif
	while (!my_empty(q)) // пока очередь не пуста
	{
		cmnode* temp = front(q); // Берем первый элемент в очереди
		pop(q);  // Удаляем первый элемент в очереди
		printf("%3d ", temp->p); // Печатаем значение первого элемента в очереди

		if (temp->left != NULL) 
			push(q, temp->left);  // Вставляем  в очередь левого потомка

		if (temp->right != NULL) 
			push(q, temp->right);  // Вставляем  в очередь правого потомка
#ifdef DEBUG
		print_queue(q);
#endif
	}
}

void UnsymmPrintRanksInWidthList(cumnode *root)
{
	if (root == NULL)
	{
		return;
	}
	struct my_queue2* q; // Создаем очередь
	init(q);
	push(q, root); // Вставляем корень в очередь

#ifdef DEBUG
	print_queue(q);
#endif
	while (!my_empty(q)) // пока очередь не пуста
	{
		cumnode* temp = front(q); // Берем первый элемент в очереди
		pop(q);  // Удаляем первый элемент в очереди
		printf("A21->p: %3d ", temp->A21->p); // Печатаем значение первого элемента в очереди
		printf("A12->p: %3d ", temp->A12->p); 

		if (temp->left != NULL)
			push(q, temp->left);  // Вставляем  в очередь левого потомка

		if (temp->right != NULL)
			push(q, temp->right);  // Вставляем  в очередь правого потомка
#ifdef DEBUG
		print_queue(q);
#endif
	}
}

void PrintStruct(int n, cmnode *root)
{
	if (root == NULL)
	{
		return;
	}
	else
	{
		int n2 = ceil(n / 2.0);
		int n1 = n - n2;
		printf("%3d ", root->p);
		print(n2, root->p, root->U, n2, "U");
		printf("\n");
		print(root->p, n1, root->VT, root->p, "VT");

		PrintStruct(n1, root->left);
		PrintStruct(n2, root->right);
	}
}

// ---------------HSS technology----------


void LowRankApproxStruct(int n2, int n1 /* size of A21 = A */,
	dtype *A /* A is overwritten by U */, int lda, cmnode* &Astr, double eps, char *method)
{
	int mn = min(n1, n2);
	int info = 0;
	int lwork = -1;
	dtype wkopt;
	double ropt;
	double time;

	dtype *VT = (dtype*)malloc(n1 * n1 * sizeof(dtype)); int ldvt = n1;
	double *S = (double*)malloc(mn * sizeof(double));
#ifndef FULL_SVD
	zgesvd("Over", "Sing", &n2, &n1, A, &lda, S, VT, &ldvt, VT, &ldvt, &wkopt, &lwork, &ropt, &info);
	lwork = (int)wkopt.real();
	dtype *work = (dtype*)malloc(lwork * sizeof(dtype));
	double *rwork = (double*)malloc(5 * mn * sizeof(double));

	// A = U1 * S * V1
	time = omp_get_wtime();
	zgesvd("Over", "Sing", &n2, &n1, A, &lda, S, VT, &ldvt, VT, &ldvt, work, &lwork, rwork, &info);
	time = omp_get_wtime() - time;
	//printf("time SVD = %lf\n", time);
#else
	dtype *U = (dtype*)malloc(n2 * n2 * sizeof(dtype)); int ldu = n2;
	// Workspace Query
	zgesvd("All", "All", &n2, &n1, A, &lda, S, U, &ldu, VT, &ldvt, &wkopt, &lwork, &ropt, &info);
	lwork = (int)wkopt.real();
	dtype *work = (dtype*)malloc(lwork * sizeof(dtype));
	double *rwork = (double*)malloc(5 * mn * sizeof(double));

	// A = U1 * S * V1
	zgesvd("All", "All", &n2, &n1, A, &lda, S, U, &ldu, VT, &ldvt, work, &lwork, rwork, &info);
#endif

	for (int j = 0; j < mn; j++)
	{
		if (S[j] / S[0] < eps)
		{
			//	printf("S[%d] / S[0] = %20.18lf\n", j, S[j] / S[0]);
			//	printf("S[%d] = %20.18lf\n", j, S[j]);
			//	printf("S[0] = %20.18lf\n", S[0]);
			break;
		}
		Astr->p = j + 1;
#ifndef FULL_SVD
#pragma omp parallel for simd schedule(static)
		for (int i = 0; i < n2; i++)
			A[i + lda * j] *= S[j];
#else
#pragma omp parallel for simd schedule(static)
		for (int i = 0; i < n2; i++)
			U[i + ldu * j] *= S[j];

#endif
	}

	// Alloc new node
	Astr->U = (dtype*)malloc(n2 * Astr->p * sizeof(dtype));
	Astr->VT = (dtype*)malloc(Astr->p * n1 * sizeof(dtype));

	Astr->n2 = n2;
	Astr->n1 = n1;

#ifndef FULL_SVD
	zlacpy("All", &n2, &Astr->p, A, &lda, Astr->U, &n2);
	zlacpy("All", &Astr->p, &n1, VT, &ldvt, Astr->VT, &Astr->p);
#else
	zlacpy("All", &n2, &Astr->p, U, &ldu, Astr->U, &n2);
	zlacpy("All", &Astr->p, &n1, VT, &ldvt, Astr->VT, &Astr->p);
#endif

#ifdef DEBUG
	printf("LowRankStructure function after SVD: n2 = %d, n1 = %d, p = %d\n", n2, n1, Astr->p);
	//	print(n2, Astr->p, Astr->U, n2, "U");
	//	print(Astr->p, n1, Astr->VT, Astr->p, "VT");

#endif
	free_arr(VT);
	free_arr(work);
	free_arr(rwork);
	free_arr(S);

#ifdef FULL_SVD
	free_arr(U);
#endif

	return;
}

void SymRecCompressStruct(int n /* order of A */, dtype *A /* init matrix */, const int lda,
	/*output*/ cmnode* &ACstr,
	const int small_size, double eps,
	char *method /* SVD or other */)
{
	ACstr = (cmnode*)malloc(sizeof(cmnode)); // на каждом шаге мы должны создавать новую структуру, нельзя выносить за функцию этот вызов

	if (n <= small_size)
	{
		alloc_dense_node(n, ACstr);
		zlacpy("All", &n, &n, A, &lda, ACstr->A, &n);
	}
	else
	{
		int n1, n2; // error 3  - неправильное выделение подматриц - похоже на проблему 2
		n2 = (int)ceil(n / 2.0); // округление в большую сторону
		n1 = n - n2; // n2 > n1

		// LowRank A21
		LowRankApproxStruct(n2, n1, &A[n1 + lda * 0], lda, ACstr, eps, method);

#ifdef DEBUG
		printf("SymRecCompressStruct: n = %d n1 = %d n2 = %d p = %d\n", n, n1, n2, ACstr->p);
		print(n1, n1, &A[0 + lda * 0], lda, "Astr");
		print(n2, n2, &A[n1 + lda * n1], lda, "Astr");
#endif

		SymRecCompressStruct(n1, &A[0 + lda * 0], lda, ACstr->left, small_size, eps, method);
		SymRecCompressStruct(n2, &A[n1 + lda * n1], lda, ACstr->right, small_size, eps, method);
	}

}

void UnsymmRecCompressStruct(int n /* order of A */, dtype *A /* init matrix */, const int lda,
	/*output*/ cumnode* &ACstr,
	const int smallsize, double eps,
	char *method /* SVD or other */)
{
	ACstr = (cumnode*)malloc(sizeof(cumnode)); // на каждом шаге мы должны создавать новую структуру, нельзя выносить за функцию этот вызов

	if (n <= smallsize)
	{
		alloc_dense_unsymm_node(n, ACstr);
		zlacpy("All", &n, &n, A, &lda, ACstr->A21->A, &n);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // округление в большую сторону
		int n1 = n - n2; // n2 > n1	

		// LowRank A21 and A12
		ACstr->A21 = (cmnode*)malloc(sizeof(cmnode));
		LowRankApproxStruct(n2, n1, &A[n1 + lda * 0], lda, ACstr->A21, eps, method);
		ACstr->A12 = (cmnode*)malloc(sizeof(cmnode));
		LowRankApproxStruct(n1, n2, &A[0 + lda * n1], lda, ACstr->A12, eps, method);

#ifdef DEBUG
		printf("SymRecCompressStruct: n = %d n1 = %d n2 = %d p = %d\n", n, n1, n2, ACstr->p);
		print(n1, n1, &A[0 + lda * 0], lda, "Astr");
		print(n2, n2, &A[n1 + lda * n1], lda, "Astr");
#endif

		UnsymmRecCompressStruct(n1, &A[0 + lda * 0], lda, ACstr->left, smallsize, eps, method);
		UnsymmRecCompressStruct(n2, &A[n1 + lda * n1], lda, ACstr->right, smallsize, eps, method);
	}

}

/* Рекурсивная функция вычисления DAD, где D - диагональная матрица, а Astr - сжатая в структуре */
void DiagMultStruct(int n, cmnode* Astr, dtype *d, int smallsize)
{
	if (n <= smallsize)
	{
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
#pragma omp simd
			for (int i = 0; i < n; i++)
			{
				Astr->A[i + j * n] *= d[j]; // справа D - каждый j - ый столбец A умножается на d[j]
				Astr->A[i + j * n] *= d[i]; // слева D - каждая строка A умножается на d[j]
			}
	}
	else
	{
		int n2 = (int)ceil(n / 2.0);
		int n1 = n - n2;

		DiagMultStruct(n1, Astr->left, &d[0], smallsize);
		DiagMultStruct(n2, Astr->right, &d[n1], smallsize);

		// D * U - каждая i-ая строка U умножается на элемент вектора d[i]
#pragma omp parallel for schedule(static)
		for (int j = 0; j < Astr->p; j++)
#pragma omp simd
			for (int i = 0; i < n2; i++)
				Astr->U[i + n2 * j] *= d[n1 + i]; // вторая часть массива D

												  // VT * D - каждый j-ый столбец умножается на элемент вектора d[j]
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n1; j++)
#pragma omp simd
			for (int i = 0; i < Astr->p; i++)
				Astr->VT[i + Astr->p * j] *= d[j];
		// так так вектора матрицы V из разложения A = U * V лежат в транспонированном порядке,
		// то матрицу D стоит умножать на VT слева
	}
}

/* G = XT * A * X, where A - compressed n * n, X - dense n * m, Y - dense m * m */
void DenseMultStruct(int n, int m, cmnode* Astr, dtype *U, int ldu, dtype *G, int ldg, int smallsize)
{
	dtype alpha = 1.0;
	dtype beta = 0.0;
	dtype *Y = alloc_arr2<dtype>(n * m); int ldy = n;

	// Y = (n x n) * (n * m)
	RecMultLStruct(n, m, Astr, U, ldu, Y, ldy, smallsize);

	// G = (m x n) * Y (n x m)
	zgemm("Trans", "No", &m, &m, &n, &alpha, U, &ldu, Y, &ldy, &beta, G, &ldg);
}

void UnsymmDenseMultStruct(int p1, int n, int p2, cumnode* Astr, dtype *U, int ldu, dtype *VT, int ldvt, dtype *G, int ldg, int smallsize)
{
	dtype alpha = 1.0;
	dtype beta = 0.0;
	dtype *Y = alloc_arr2<dtype>(n * p2); int ldy = n;

	// Y = (n x n) * (n * m)
	UnsymmRecMultLStruct(n, p2, Astr, VT, ldvt, Y, ldy, smallsize);

	// G = (m x n) * Y (n x m)
	zgemm("No", "No", &p1, &p2, &n, &alpha, U, &ldu, Y, &ldy, &beta, G, &ldg);
}

#if 0
void UnsymmLUfact(int n, cumnode* Astr, int *ipiv, int smallsize)
{
	if (n <= smallsize)
	{
		int info = 0;
		zgetrf(&n, &n, Astr->A21->A, &n, ipiv, &info);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0);
		int n1 = n - n2;
		int ione = 1;
		dtype alpha = 1.0;
		dtype beta = 0.0;
		dtype *G1 = alloc_arr<dtype>(n2 * n1); int ldg1 = n2;
		dtype *G2 = alloc_arr<dtype>(n1 * n2); int ldg2 = n1;
		dtype *RES = alloc_arr<dtype>(n2 * n2); int ldr = n2;

#if 0
		cumnode* Da;
		printf("Copy\n");
		CopyUnsymmStruct(n1, Astr->left, Da, smallsize);
#endif

		printf("LU A[1][1]\n");
		UnsymmLUfact(n1, Astr->left, &ipiv[0], smallsize);

		//Apply to A21: = U1 * V1T * (UP ^ {-1}) or to solve triangular system X * UP = VT
		printf("Apply to A21\n");
		ztrsm("Right", "Up", "No", "NonUnit", &Astr->A21->p, &n1, &alpha, Astr->left->A21->A, &n1, Astr->A21->VT, &Astr->A21->p);

		//Apply to A12: =  (L ^ {-1}) (P ^ {-1}) * U2 * V2T or to solve triangular system  L * X  = (P ^ {-1}) * U
		printf("swap rows in U");
		zlaswp(&Astr->A12->p, Astr->A12->U, &n1, &ione, &n1, &ipiv[0], &ione);
		printf("Apply to A12\n");
		ztrsm("Left", "Low", "No", "Unit", &n1, &Astr->A12->p, &alpha, Astr->left->A21->A, &n1, Astr->A12->U, &n1);

		//Double update D:= D - U1 * V1T * (U^{-1}) * (L^{-1}) * U2 * V2T

		printf("Update 2.1\n");
		zgemm("no", "no", &n2, &n1, &Astr->A21->p, &alpha, Astr->A21->U, &n2, Astr->A21->VT, &Astr->A21->p, &beta, G1, &ldg1);
		printf("Update 2.2\n");
		zgemm("no", "no", &n1, &n2, &Astr->A12->p, &alpha, Astr->A12->U, &n1, Astr->A12->VT, &Astr->A12->p, &beta, G2, &ldg2);
		printf("Update 2.3\n");
		zgemm("no", "no", &n2, &n2, &n1, &alpha, G1, &ldg1, G2, &ldg2, &beta, RES, &ldr);

		// D: = D - RES
		printf("Update D:= D - RES\n");
		for (int j = 0; j < n2; j++)
			for (int i = 0; i < n2; i++)
				Astr->right->A21->A[i + n2 * j] -= RES[i + n2 * j];

		printf("LU A[2][2]\n");
		UnsymmLUfact(n2, Astr->right, &ipiv[n1], smallsize);

		free_arr(G1);
		free_arr(G2);
		free_arr(RES);
	}
}
#else

void CopyLfactor(int n, cumnode* Astr, cumnode* &Lstr, int smallsize)
{
	Lstr = (cumnode*)malloc(sizeof(cumnode));
	if (n <= smallsize)
	{
		alloc_dense_unsymm_node(n, Lstr);
		zlacpy("Low", &n, &n, Astr->A21->A, &n, Lstr->A21->A, &n);

		for (int i = 0; i < n; i++)
			Lstr->A21->A[i + n * i] = 1.0;
	}
	else
	{
		int n2 = (int)ceil(n / 2.0);
		int n1 = n - n2;

		Lstr->A21 = (cmnode*)malloc(sizeof(cmnode));
		Lstr->A12 = (cmnode*)malloc(sizeof(cmnode));

		Lstr->A21->p = Astr->A21->p;
		Lstr->A12->p = Astr->A12->p;

		Lstr->A21->U = alloc_arr<dtype>(n2 * Astr->A21->p);
		Lstr->A12->U = alloc_arr<dtype>(n1 * Astr->A12->p);
		Lstr->A21->VT = alloc_arr<dtype>(Lstr->A21->p * n1);
		Lstr->A12->VT = alloc_arr<dtype>(Lstr->A12->p * n2);

		zlacpy("All", &n2, &Lstr->A21->p, Astr->A21->U, &n2, Lstr->A21->U, &n2);
		zlacpy("All", &Lstr->A21->p, &n1, Astr->A21->VT, &Astr->A21->p, Lstr->A21->VT, &Lstr->A21->p);

		Clear(n1, Lstr->A12->p, Lstr->A12->U, n1);
		Clear(Lstr->A12->p, n2, Lstr->A12->VT, Lstr->A12->p);

		CopyLfactor(n1, Astr->left, Lstr->left, smallsize);
		CopyLfactor(n2, Astr->right, Lstr->right, smallsize);
	}
}

void CopyRfactor(int n, cumnode* Astr, cumnode* &Rstr, int smallsize)
{
	Rstr = (cumnode*)malloc(sizeof(cumnode));
	if (n <= smallsize)
	{
		alloc_dense_unsymm_node(n, Rstr);
		zlacpy("Up", &n, &n, Astr->A21->A, &n, Rstr->A21->A, &n);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0);
		int n1 = n - n2;

		Rstr->A21 = (cmnode*)malloc(sizeof(cmnode));
		Rstr->A12 = (cmnode*)malloc(sizeof(cmnode));

		Rstr->A21->p = Astr->A21->p;
		Rstr->A12->p = Astr->A12->p;

		Rstr->A21->U = alloc_arr2<dtype>(n2 * Astr->A21->p);
		Rstr->A12->U = alloc_arr2<dtype>(n1 * Astr->A12->p);
		Rstr->A21->VT = alloc_arr2<dtype>(Rstr->A21->p * n1);
		Rstr->A12->VT = alloc_arr2<dtype>(Rstr->A12->p * n2);

		zlacpy("All", &n1, &Rstr->A12->p, Astr->A12->U, &n1, Rstr->A12->U, &n1);
		zlacpy("All", &Rstr->A12->p, &n2, Astr->A12->VT, &Astr->A12->p, Rstr->A12->VT, &Rstr->A12->p);

		Clear(n2, Astr->A21->p, Rstr->A21->U, n2);
		Clear(Astr->A21->p, n1, Rstr->A21->VT, Astr->A21->p);

		CopyRfactor(n1, Astr->left, Rstr->left, smallsize);
		CopyRfactor(n2, Astr->right, Rstr->right, smallsize);
	}
}

void ApplyToA21(int n, cmnode* Astr, cumnode* R, int smallsize, double eps, char *method)
{
	int ione = 1;
	dtype alpha = 1.0;
	dtype beta = 0.0;

	if (n <= smallsize)
	{
		ztrsm("Right", "Up", "No", "NonUnit", &Astr->p, &n, &alpha, R->A21->A, &n, Astr->VT, &Astr->p);
	}
	else
	{
		cumnode* Rinv;
		dtype *Y = alloc_arr<dtype>(Astr->p * n); int ldy = Astr->p;

		// Rinv = R^{-1}
		UnsymmCompRecInvUpperTriangStruct(n, R, Rinv, smallsize, eps, method);

		// Apply Rinv to the right of A21 VT (save result in Y)
		UnsymmRecMultUpperRStruct(n, Astr->p, Rinv, Astr->VT, Astr->p, Y, ldy, smallsize);

		// Replace A21 VT with Y
		zlacpy("All", &Astr->p, &n, Y, &ldy, Astr->VT, &Astr->p);

		free_arr(Y);
	}
}

#if 1
// VT * R^(-1) = X  ---->  X * R = VT  <--->  (p * n) * (n * n) = (p * n) // p строк
void ApplyToA21Ver2(int p, int n, dtype* VT, int ldvt, cumnode* R, int smallsize, double eps, char *method)
{
	int ione = 1;
	dtype zero = 0.0;
	dtype one = 1.0;
	dtype mone = -1.0;

	if (n <= smallsize)
	{
		ztrsm("Right", "Up", "No", "NonUnit", &p, &n, &one, R->A21->A, &n, VT, &ldvt);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		int p2 = (int)ceil(p / 2.0); // p2 > p1
		int p1 = p - p2;

		dtype *inter1 = alloc_arr<dtype>(p1 * R->A12->p);
		dtype *inter2 = alloc_arr<dtype>(p2 * R->A12->p);

		if (p1 != 0) ApplyToA21Ver2(p1, n1, &VT[0 + ldvt * 0], ldvt, R->left, smallsize, eps, method);
		ApplyToA21Ver2(p2, n1, &VT[p1 + ldvt * 0], ldvt, R->left, smallsize, eps, method);

		if (p1 != 0)
		{
			zgemm("No", "No", &p1, &R->A12->p, &n1, &one, &VT[0 + ldvt * 0], &ldvt, R->A12->U, &n1, &zero, inter1, &p1);
			zgemm("No", "No", &p1, &n2, &R->A12->p, &mone, inter1, &p1, R->A12->VT, &R->A12->p, &one, &VT[0 + ldvt * n1], &ldvt);
		}

		zgemm("No", "No", &p2, &R->A12->p, &n1, &one, &VT[p1 + ldvt * 0], &ldvt, R->A12->U, &n1, &zero, inter2, &p2);
		zgemm("No", "No", &p2, &n2, &R->A12->p, &mone, inter2, &p2, R->A12->VT, &R->A12->p, &one, &VT[p1 + ldvt * n1], &ldvt);

		if (p1 != 0) ApplyToA21Ver2(p1, n2, &VT[0 + ldvt * n1], ldvt, R->right, smallsize, eps, method);
		ApplyToA21Ver2(p2, n2, &VT[p1 + ldvt * n1], ldvt, R->right, smallsize, eps, method);

		free_arr(inter1);
		free_arr(inter2);
	}
}
#endif

void ApplyToA12(int n, cmnode* A12, cumnode* L, int smallsize, double eps, char *method)
{
	int ione = 1;
	int mione = -1;
	dtype alpha = 1.0;
	dtype beta = 0.0;

	if (n <= smallsize)
	{
		ztrsm("Left", "Low", "No", "Unit", &n, &A12->p, &alpha, L->A21->A, &n, A12->U, &n);
	}
	else
	{
		cumnode* Linv;
		dtype *Y = alloc_arr<dtype>(n * A12->p); int ldy = n;

		// Linv = L^{-1}
		UnsymmCompRecInvLowerTriangStruct(n, L, Linv, smallsize, eps, method);

		// Apply L^{-1} to U12
		UnsymmRecMultLowerLStruct(n, A12->p, Linv, A12->U, n, Y, ldy, smallsize);

		// Replace A21 VT with Y
		zlacpy("All", &n, &A12->p, Y, &ldy, A12->U, &n);

		free_arr(Y);
	}
}

#if 1
void ApplyToA12Ver2(int n, int p, dtype* U, int ldu, cumnode* L, int smallsize, double eps, char *method)
{
	int ione = 1;
	dtype one = 1.0;
	dtype zero = 0.0;
	dtype mone = -1.0;

	if (n <= smallsize)
	{
		ztrsm("Left", "Low", "No", "Unit", &n, &p, &one, L->A21->A, &n, U, &ldu);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		int p2 = (int)ceil(p / 2.0); // p2 > p1
		int p1 = p - p2;

		dtype *inter1 = alloc_arr<dtype>(L->A21->p * p1);
		dtype *inter2 = alloc_arr<dtype>(L->A21->p * p2);

		if (p1 != 0) ApplyToA12Ver2(n1, p1, &U[0 + ldu * 0], ldu, L->left, smallsize, eps, method);
		ApplyToA12Ver2(n1, p2, &U[0 + ldu * p1], ldu, L->left, smallsize, eps, method);

		if (p1 != 0)
		{
			zgemm("No", "No", &L->A21->p, &p1, &n1, &one, L->A21->VT, &L->A21->p, &U[0 + ldu * 0], &ldu, &zero, inter1, &L->A21->p);
			zgemm("No", "No", &n2, &p1, &L->A21->p, &mone, L->A21->U, &n2, inter1, &L->A21->p, &one, &U[n1 + ldu * 0], &ldu);
		}

		zgemm("No", "No", &L->A21->p, &p2, &n1, &one, L->A21->VT, &L->A21->p, &U[0 + ldu * p1], &ldu, &zero, inter2, &L->A21->p);
		zgemm("No", "No", &n2, &p2, &L->A21->p, &mone, L->A21->U, &n2, inter2, &L->A21->p, &one, &U[n1 + ldu * p1], &ldu);

		if (p1 != 0) ApplyToA12Ver2(n2, p1, &U[n1 + ldu * 0], ldu, L->right, smallsize, eps, method);
		ApplyToA12Ver2(n2, p2, &U[n1 + ldu * p1], ldu, L->right, smallsize, eps, method);

		free_arr(inter1);
		free_arr(inter2);
	}
}
#endif

void UnsymmLUfact(int n, cumnode* Astr, int *ipiv, int smallsize, double eps, char* method)
{
	int ione = 1;
	int mione = -1;
	dtype alpha = 1.0;
	dtype alpha_mone = -1.0;
	dtype beta = 0.0;

	if (n <= smallsize)
	{
		int info = 0;
		zgetrf(&n, &n, Astr->A21->A, &n, ipiv, &info);
#ifdef PRINT
		for (int i = 0; i < n; i++)
			if (ipiv[i] != i + 1) printf("HSS LU for n = %d: ROW interchange: %d with %d\n", n, i + 1, ipiv[i]);
#endif
	}
	else
	{
		int n2 = (int)ceil(n / 2.0);
		int n1 = n - n2;

		// LU for A11
		UnsymmLUfact(n1, Astr->left, &ipiv[0], smallsize, eps, method);

		// Apply to A21: = U1 * V1T * (UP ^ {-1}) or to solve triangular system X * UP = VT
		ApplyToA21Ver2(Astr->A21->p, n1, Astr->A21->VT, Astr->A21->p, Astr->left, smallsize, eps, method);

		// Swap row in A12
		zlaswp(&Astr->A12->p, Astr->A12->U, &n1, &ione, &n1, ipiv, &ione);

		// Apply to A12: =  (L ^ {-1}) (P ^ {-1}) * U2 * V2T or to solve triangular system  L * X  = (P ^ {-1}) * U
		ApplyToA12Ver2(n1, Astr->A12->p, Astr->A12->U, n1, Astr->left, smallsize, eps, method);

		// Double update D:= D - U1 * V1T * (U^{-1}) * (L^{-1}) * U2 * V2T		
		// Update compressed block A[2][2]
		dtype *Y = alloc_arr<dtype>(Astr->A21->p * Astr->A12->p); int ldy = Astr->A21->p;
		zgemm("no", "no", &Astr->A21->p, &Astr->A12->p, &n1, &alpha, Astr->A21->VT, &Astr->A21->p, Astr->A12->U, &n1, &beta, Y, &ldy);
	
		cumnode* Bstr;
		// (n2 x n2) = (n2 x n2) - (n2 x p2) * (p2 x p1) * (p1 x n2)
		UnsymmCompUpdate3Struct(n2, Astr->A21->p, Astr->A12->p, Astr->right, alpha_mone, Y, ldy, Astr->A21->U, n2, Astr->A12->VT, Astr->A12->p, Bstr, smallsize, eps, method);

		FreeUnsymmNodes(n2, Astr->right, smallsize);
		CopyUnsymmStruct(n2, Bstr, Astr->right, smallsize);
		FreeUnsymmNodes(n2, Bstr, smallsize);

		free_arr(Y);

		// LU for A22
		UnsymmLUfact(n2, Astr->right, &ipiv[n1], smallsize, eps, method);

		// Swap rows in A21
		zlaswp(&Astr->A21->p, Astr->A21->U, &n2, &ione, &n2, &ipiv[n1], &ione);

		// Adjust pivot indexes to level up
#pragma omp parallel for simd schedule(static)
		for (int i = n1; i < n; i++)
			ipiv[i] += n1;
	}
}

void SymLUfactLowRankStruct(int n, cumnode* Astr, int *ipiv, int smallsize, double eps, char* method)
{
#if 1
	int ione = 1;
	int mione = -1;
	dtype alpha = 1.0;
	dtype alpha_mone = -1.0;
	dtype beta = 0.0;

	if (n <= smallsize)
	{
		int info = 0;
		zgetrf(&n, &n, Astr->A21->A, &n, ipiv, &info);
#ifdef PRINT
		for (int i = 0; i < n; i++)
			if (ipiv[i] != i + 1) printf("HSS LU for n = %d: ROW interchange: %d with %d\n", n, i + 1, ipiv[i]);
#endif
	}
	else
	{
		int n2 = (int)ceil(n / 2.0);
		int n1 = n - n2;

		// LU for A11
		SymLUfactLowRankStruct(n1, Astr->left, &ipiv[0], smallsize, eps, method);

		// Apply to A21: = U1 * V1T * (UP ^ {-1}) or to solve triangular system X * UP = VT
		ApplyToA21Ver2(Astr->A21->p, n1, Astr->A21->VT, Astr->A21->p, Astr->left, smallsize, eps, method);

		// Swap row in A12
		zlaswp(&Astr->A12->p, Astr->A12->U, &n1, &ione, &n1, ipiv, &ione);

		// Apply to A12: =  (L ^ {-1}) (P ^ {-1}) * U2 * V2T or to solve triangular system  L * X  = (P ^ {-1}) * U
		ApplyToA12Ver2(n1, Astr->A12->p, Astr->A12->U, n1, Astr->left, smallsize, eps, method);

		// Double update D:= D - U1 * V1T * (U^{-1}) * (L^{-1}) * U2 * V2T		
		// Update compressed block A[2][2]
		dtype *Y = alloc_arr<dtype>(Astr->A21->p * Astr->A12->p); int ldy = Astr->A21->p;
		zgemm("no", "no", &Astr->A21->p, &Astr->A12->p, &n1, &alpha, Astr->A21->VT, &Astr->A21->p, Astr->A12->U, &n1, &beta, Y, &ldy);

		cumnode* Bstr;
#if 0
		// (n2 x n2) = (n2 x n2) - (n2 x p2) * (p2 x p1) * (p1 x n2)
		UnsymmCompUpdate3Struct(n2, Astr->A21->p, Astr->A12->p, Astr->right, alpha_mone, Y, ldy, Astr->A21->U, n2, Astr->A12->VT, Astr->A12->p, Bstr, smallsize, eps, method);
#else
		SymCompUpdate4LowRankStruct(n2, Astr->A21->p, Astr->A12->p, Astr->right, alpha_mone, Y, ldy, Astr->A21->U, n2, Astr->A12->VT, Astr->A12->p, Bstr, smallsize, eps, method);
#endif
		FreeUnsymmNodes(n2, Astr->right, smallsize);
		CopyUnsymmStruct(n2, Bstr, Astr->right, smallsize);
		FreeUnsymmNodes(n2, Bstr, smallsize);

		free_arr(Y);

		// LU for A22
		SymLUfactLowRankStruct(n2, Astr->right, &ipiv[n1], smallsize, eps, method);

		// Swap rows in A21
		zlaswp(&Astr->A21->p, Astr->A21->U, &n2, &ione, &n2, &ipiv[n1], &ione);

		// Adjust pivot indexes to level up
#pragma omp parallel for simd schedule(static)
		for (int i = n1; i < n; i++)
			ipiv[i] += n1;
	}
#endif
}
#endif

/* Y = A * X, where A - compressed n * n, X - dense n * m, Y - dense n * m */
// work array = 2 * n * m + n * n / 2
//
void RecMultLStructWork(int n, int m, cmnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, dtype *work1, int lwork1, dtype *work2, int lwork2, int smallsize)
{
	dtype alpha = 1.0;
	dtype beta = 0.0;

	if (n <= smallsize)
	{
		zgemm("No", "No", &n, &m, &n, &alpha, Astr->A, &n, X, &ldx, &beta, Y, &ldy);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // rounding up
		int n1 = n - n2;
		int full_memory = 2 * m * n;
		dtype *Y12 = work2; 
		dtype *Y21 = &work2[n1 * m]; 
		dtype *Y11 = &work2[n2 * m + n1 * m]; 
		dtype *Y22 = &work2[n1 * m + n2 * m + n1 * m]; 

		dtype *inter1 = work1; // column major - lda = column
		dtype *inter2 = &work1[n1 * n2];

		int ldy12 = n1;
		int ldy21 = n2;
		int ldy11 = n1;
		int ldy22 = n2;

		int cur_lwork = lwork2 - full_memory;
		int cur_lwork2 = cur_lwork / 2;

		// A21 = A21 * A12 (the result of multiplication is A21 matrix with size n2 x n1)
		zgemm("No", "No", &n2, &n1, &Astr->p, &alpha, Astr->U, &n2, Astr->VT, &Astr->p, &beta, inter1, &n2);

		// Y21 = inter1 (n2 x n1) * X(1...n1, :) (n1 x n)
		zgemm("No", "No", &n2, &m, &n1, &alpha, inter1, &n2, &X[0 + 0 * ldx], &ldx, &beta, Y21, &ldy21);

		// A12 = A21*T = A12*T * A21*T (the result of multiplication is A21 matrix with size n1 x n2)
		zgemm("Trans", "Trans", &n1, &n2, &Astr->p, &alpha, Astr->VT, &Astr->p, Astr->U, &n2, &beta, inter2, &n1);

		// Y12 = inter2 (n1 x n2) * X(n1...m, :) (n2 x n)
		zgemm("No", "No", &n1, &m, &n2, &alpha, inter2, &n1, &X[n1 + 0 * ldx], &ldx, &beta, Y12, &ldy12); // we have already transposed this matrix in previous dgemm

		RecMultLStructWork(n1, m, Astr->left, &X[0 + ldx * 0], ldx, Y11, ldy11, work1, lwork1, &work2[full_memory], cur_lwork2, smallsize);
		RecMultLStructWork(n2, m, Astr->right, &X[n1 + ldx * 0], ldx, Y22, ldy22, work1, lwork1, &work2[full_memory + cur_lwork2], cur_lwork2, smallsize);

		// first part of Y = Y11 + Y12
		mkl_zomatadd('C', 'N', 'N', n1, m, 1.0, Y11, ldy11, 1.0, Y12, ldy12, &Y[0 + ldy * 0], ldy);
		// op_mat(n1, m, Y11, Y12, n1, '+');
		// dlacpy("All", &n1, &m, Y11, &n1, &Y[0 + ldy * 0], &ldy);

		// second part of Y = Y21 + Y22
		mkl_zomatadd('C', 'N', 'N', n2, m, 1.0, Y21, ldy21, 1.0, Y22, ldy22, &Y[n1 + ldy * 0], ldy);
		// op_mat(n2, m, Y21, Y22, n2, '+');
		// dlacpy("All", &n2, &m, Y21, &n2, &Y[n1 + ldy * 0], &ldy);
	}
}

void RecMultLStructWork2(int n, int m, cmnode* Astr, dtype* X, int ldx, dtype beta, dtype* Y, int ldy, dtype* work, int lwork, int smallsize)
{
	dtype zero = 0.0;
	dtype fone = 1.0;

	if (n <= smallsize)
	{
		zgemm("No", "No", &n, &m, &n, &fone, Astr->A, &n, X, &ldx, &beta, Y, &ldy);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // rounding up
		int n1 = n - n2;
		int full_memory = 2 * m * Astr->p;

		dtype* inter1 = work; // column major - lda = column
		dtype* inter2 = &work[Astr->p * m];

		RecMultLStructWork2(n1, m, Astr->left, &X[0 + ldx * 0], ldx, beta, &Y[0 + ldy * 0], ldy, work, lwork, smallsize);

		// Y21 = A21 * (A12 * X(n1...m, :))
		// first multiply low-rank VT * X(1...n1, :) to get (p x m) = (p x n2) * (n2 x m)
		// second multiply low-rank U * inter1 to get (n1 x m) = (n1 x p) * (p x m)
		zgemm("Trans", "No", &Astr->p, &m, &n2, &fone, Astr->U, &n2, &X[n1 + 0 * ldx], &ldx, &zero, inter1, &Astr->p);
		zgemm("Trans", "No", &n1, &m, &Astr->p, &fone, Astr->VT, &Astr->p, inter1, &Astr->p, &fone, &Y[0 + ldy * 0], &ldy);

		RecMultLStructWork2(n2, m, Astr->right, &X[n1 + ldx * 0], ldx, beta, &Y[n1 + ldy * 0], ldy, work, lwork, smallsize);

		// A12 = A21*T = A12*T * (A21*T * X(1...n1, :))
		// first multiply low-rank UT * X(1...n1, :) to get (p x m) = (p x n1) * (n1 x m)
		// second multiply low-rank VT^T * inter1 to get (n2 x m) = (n2 x p) * (p x m)
		zgemm("No", "No", &Astr->p, &m, &n1, &fone, Astr->VT, &Astr->p, &X[0 + 0 * ldx], &ldx, &zero, inter2, &Astr->p);
		zgemm("No", "No", &n2, &m, &Astr->p, &fone, Astr->U, &n2, inter2, &Astr->p, &fone, &Y[n1 + ldy * 0], &ldy);
	}
}

void RecMultLStruct(int n, int m, cmnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, int smallsize)
{
	dtype alpha = 1.0;
	dtype beta = 0.0;

	if (n <= smallsize)
	{
		zgemm("No", "No", &n, &m, &n, &alpha, Astr->A, &n, X, &ldx, &beta, Y, &ldy);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // rounding up
		int n1 = n - n2;

		dtype * Y12 = alloc_arr2<dtype>(n1 * m); int ldy12 = n1;
		dtype * Y21 = alloc_arr2<dtype>(n2 * m); int ldy21 = n2;
		dtype * Y11 = alloc_arr2<dtype>(n1 * m); int ldy11 = n1;
		dtype * Y22 = alloc_arr2<dtype>(n2 * m); int ldy22 = n2;

		dtype *inter1 = alloc_arr2<dtype>(n2 * n1); // column major - lda = column
		dtype *inter2 = alloc_arr2<dtype>(n1 * n2);

		// A21 = A21 * A12 (the result of multiplication is A21 matrix with size n2 x n1)
		zgemm("No", "No", &n2, &n1, &Astr->p, &alpha, Astr->U, &n2, Astr->VT, &Astr->p, &beta, inter1, &n2);

		// Y21 = inter1 (n2 x n1) * X(1...n1, :) (n1 x n)
		zgemm("No", "No", &n2, &m, &n1, &alpha, inter1, &n2, &X[0 + 0 * ldx], &ldx, &beta, Y21, &ldy21);

		// A12 = A21*T = A12*T * A21*T (the result of multiplication is A21 matrix with size n1 x n2)
		zgemm("Trans", "Trans", &n1, &n2, &Astr->p, &alpha, Astr->VT, &Astr->p, Astr->U, &n2, &beta, inter2, &n1);

		// Y12 = inter2 (n1 x n2) * X(n1...m, :) (n2 x n)
		zgemm("No", "No", &n1, &m, &n2, &alpha, inter2, &n1, &X[n1 + 0 * ldx], &ldx, &beta, Y12, &ldy12); // we have already transposed this matrix in previous dgemm

		RecMultLStruct(n1, m, Astr->left, &X[0 + ldx * 0], ldx, Y11, ldy11, smallsize);
		RecMultLStruct(n2, m, Astr->right, &X[n1 + ldx * 0], ldx, Y22, ldy22, smallsize);

		// first part of Y = Y11 + Y12
		mkl_zomatadd('C', 'N', 'N', n1, m, 1.0, Y11, ldy11, 1.0, Y12, ldy12, &Y[0 + ldy * 0], ldy);
		// op_mat(n1, m, Y11, Y12, n1, '+');
		// dlacpy("All", &n1, &m, Y11, &n1, &Y[0 + ldy * 0], &ldy);

		// second part of Y = Y21 + Y22
		mkl_zomatadd('C', 'N', 'N', n2, m, 1.0, Y21, ldy21, 1.0, Y22, ldy22, &Y[n1 + ldy * 0], ldy);
		// op_mat(n2, m, Y21, Y22, n2, '+');
		// dlacpy("All", &n2, &m, Y21, &n2, &Y[n1 + ldy * 0], &ldy);

		free_arr(Y11);
		free_arr(Y12);
		free_arr(Y21);
		free_arr(Y22);
		free_arr(inter1);
		free_arr(inter2);
	}
}

/* Y = A * X, where A - compressed n * n, X - dense n * m, Y - dense n * m */
void UnsymmRecMultLStruct(int n, int m, cumnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, int smallsize)
{
	dtype alpha = 1.0;
	dtype beta = 0.0;

	if (n <= smallsize)
	{
		zgemm("No", "No", &n, &m, &n, &alpha, Astr->A21->A, &n, X, &ldx, &beta, Y, &ldy);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // rounding up
		int n1 = n - n2;
		dtype *Y12 = alloc_arr2<dtype>(n1 * m); int ldy12 = n1;
		dtype *Y21 = alloc_arr2<dtype>(n2 * m); int ldy21 = n2;
		dtype *Y11 = alloc_arr2<dtype>(n1 * m); int ldy11 = n1;
		dtype *Y22 = alloc_arr2<dtype>(n2 * m); int ldy22 = n2;
		dtype *inter1 = alloc_arr2<dtype>(Astr->A21->p * m); // column major - lda = column
		dtype *inter2 = alloc_arr2<dtype>(Astr->A12->p * m);

		// Y21 = inter1 (n2 x n1) * X(1...n1, :) = (n2 x p) * (p x n1) * (n1 * m)
		zgemm("No", "No", &Astr->A21->p, &m, &n1, &alpha, Astr->A21->VT, &Astr->A21->p, &X[0 + 0 * ldx], &ldx, &beta, inter1, &Astr->A21->p);
		zgemm("No", "No", &n2, &m, &Astr->A21->p, &alpha, Astr->A21->U, &n2, inter1, &Astr->A21->p, &beta, Y21, &ldy21);

		// Y12 = inter2 (n1 x n2) * X(n1...m, :) = (n1 x p) * (p x n2) * (n2 x m)
		zgemm("No", "No", &Astr->A12->p, &m, &n2, &alpha, Astr->A12->VT, &Astr->A12->p, &X[n1 + 0 * ldx], &ldx, &beta, inter2, &Astr->A12->p);
		zgemm("No", "No", &n1, &m, &Astr->A12->p, &alpha, Astr->A12->U, &n1, inter2, &Astr->A12->p, &beta, Y12, &ldy12);

		UnsymmRecMultLStruct(n1, m, Astr->left, &X[0 + ldx * 0], ldx, Y11, ldy11, smallsize);
		UnsymmRecMultLStruct(n2, m, Astr->right, &X[n1 + ldx * 0], ldx, Y22, ldy22, smallsize);

		// first part of Y = Y11 + Y12
		mkl_zomatadd('C', 'N', 'N', n1, m, 1.0, Y11, ldy11, 1.0, Y12, ldy12, &Y[0 + ldy * 0], ldy);
		// op_mat(n1, m, Y11, Y12, n1, '+');
		// dlacpy("All", &n1, &m, Y11, &n1, &Y[0 + ldy * 0], &ldy);

		// second part of Y = Y21 + Y22
		mkl_zomatadd('C', 'N', 'N', n2, m, 1.0, Y21, ldy21, 1.0, Y22, ldy22, &Y[n1 + ldy * 0], ldy);
		// op_mat(n2, m, Y21, Y22, n2, '+');
		// dlacpy("All", &n2, &m, Y21, &n2, &Y[n1 + ldy * 0], &ldy);

		free_arr(Y11);
		free_arr(Y12);
		free_arr(Y21);
		free_arr(Y22);
		free_arr(inter1);
		free_arr(inter2);

	}
}

/* Y = X * A, where A - compressed n * n, X - dense m * n, Y - dense m * n */
void UnsymmRecMultRStruct(int n, int m, cumnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, int smallsize)
{
	dtype alpha = 1.0;
	dtype beta = 0.0;

	if (n <= smallsize)
	{
		zgemm("No", "No", &m, &n, &n, &alpha, X, &ldx, Astr->A21->A, &n, &beta, Y, &ldy);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // rounding up
		int n1 = n - n2;
		dtype *Y12 = alloc_arr2<dtype>(m * n2); int ldy12 = m;
		dtype *Y21 = alloc_arr2<dtype>(m * n1); int ldy21 = m;
		dtype *Y11 = alloc_arr2<dtype>(m * n1); int ldy11 = m;
		dtype *Y22 = alloc_arr2<dtype>(m * n2); int ldy22 = m;
		dtype *inter1 = alloc_arr2<dtype>(m * Astr->A21->p); // column major - lda = column
		dtype *inter2 = alloc_arr2<dtype>(m * Astr->A12->p);

		// Y21 = X(..., n1:n) * inter1 (n2 x n1) = (m x n2) * (n2 x p) * (p x n1)
		zgemm("No", "No", &m, &Astr->A21->p, &n2, &alpha, &X[0 + ldx * n1], &ldx, Astr->A21->U, &n2, &beta, inter1, &m);
		zgemm("No", "No", &m, &n1, &Astr->A21->p, &alpha, inter1, &m, Astr->A21->VT, &Astr->A21->p, &beta, Y21, &ldy21);

		// Y12 = X(..., 0:n1) * inter2 (n1 x n2)  = (m x n1) * (n1 x p) * (p x n2)
		zgemm("No", "No", &m, &Astr->A12->p, &n1, &alpha, &X[0 + ldx * 0], &ldx, Astr->A12->U, &n1, &beta, inter2, &m);
		zgemm("No", "No", &m, &n2, &Astr->A12->p, &alpha, inter2, &m, Astr->A12->VT, &Astr->A12->p, &beta, Y12, &ldy12); // we have already transposed this matrix in previous dgemm

		UnsymmRecMultRStruct(n1, m, Astr->left, &X[0 + ldx * 0], ldx, Y11, ldy11, smallsize);
		UnsymmRecMultRStruct(n2, m, Astr->right, &X[0 + ldx * n1], ldx, Y22, ldy22, smallsize);

		// first part of Y = Y11 + Y21
		mkl_zomatadd('C', 'N', 'N', m, n1, 1.0, Y11, ldy11, 1.0, Y21, ldy21, &Y[0 + ldy * 0], ldy);
		// op_mat(n1, m, Y11, Y12, n1, '+');
		// dlacpy("All", &n1, &m, Y11, &n1, &Y[0 + ldy * 0], &ldy);

		// second part of Y = Y12 + Y22
		mkl_zomatadd('C', 'N', 'N', m, n2, 1.0, Y12, ldy12, 1.0, Y22, ldy22, &Y[0 + ldy * n1], ldy);
		// op_mat(n2, m, Y21, Y22, n2, '+');
		// dlacpy("All", &n2, &m, Y21, &n2, &Y[n1 + ldy * 0], &ldy);

		free_arr(Y11);
		free_arr(Y12);
		free_arr(Y21);
		free_arr(Y22);
		free_arr(inter1);
		free_arr(inter2);

	}
}

void UnsymmRecMultUpperLStruct(int n, int m, cumnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, int smallsize)
{
	dtype alpha = 1.0;
	dtype beta = 0.0;

	if (n <= smallsize)
	{
		zgemm("No", "No", &n, &m, &n, &alpha, Astr->A21->A, &n, X, &ldx, &beta, Y, &ldy);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // rounding up
		int n1 = n - n2;
		dtype *Y12 = alloc_arr2<dtype>(n1 * m); int ldy12 = n1;
		dtype *Y11 = alloc_arr2<dtype>(n1 * m); int ldy11 = n1;
		dtype *Y22 = alloc_arr2<dtype>(n2 * m); int ldy22 = n2;
		dtype *inter = alloc_arr2<dtype>(Astr->A12->p * m);

		// Y12 = U12 * V12T *  * X(n1...m, :) = (n1 x p) * (p x n2) * (n2 x m)
		zgemm("No", "No", &Astr->A12->p, &m, &n2, &alpha, Astr->A12->VT, &Astr->A12->p, &X[n1 + 0 * ldx], &ldx, &beta, inter, &Astr->A12->p);
		zgemm("No", "No", &n1, &m, &Astr->A12->p, &alpha, Astr->A12->U, &n1, inter, &Astr->A12->p, &beta, Y12, &ldy12);
		
		UnsymmRecMultUpperLStruct(n1, m, Astr->left, &X[0 + ldx * 0], ldx, Y11, ldy11, smallsize);
		UnsymmRecMultUpperLStruct(n2, m, Astr->right, &X[n1 + ldx * 0], ldx, Y22, ldy22, smallsize);

		// first part of Y = Y11 + Y12
		mkl_zomatadd('C', 'N', 'N', n1, m, 1.0, Y11, ldy11, 1.0, Y12, ldy12, &Y[0 + ldy * 0], ldy);

		// second part of Y = Y22
		zlacpy("All", &n2, &m, Y22, &ldy22, &Y[n1 + ldy * 0], &ldy);

		free_arr(Y11);
		free_arr(Y12);
		free_arr(Y22);
		free_arr(inter);
	}
}

void UnsymmRecMultUpperRStruct(int n, int m, cumnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, int smallsize)
{
	dtype alpha = 1.0;
	dtype beta = 0.0;

	if (n <= smallsize)
	{
		zgemm("No", "No", &m, &n, &n, &alpha, X, &ldx, Astr->A21->A, &n, &beta, Y, &ldy);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // rounding up
		int n1 = n - n2;
		dtype *Y12 = alloc_arr2<dtype>(m * n2); int ldy12 = m;
		dtype *Y11 = alloc_arr2<dtype>(m * n1); int ldy11 = m;
		dtype *Y22 = alloc_arr2<dtype>(m * n2); int ldy22 = m;
		dtype *inter = alloc_arr2<dtype>(m * Astr->A12->p);
		double time;

		// Y12 =  X(..., 0:n1) * U12 * V12 = (m x n1) * (n1 x p) * (p x n2)
		zgemm("No", "No", &m, &Astr->A12->p, &n1, &alpha, &X[0 + ldx * 0], &ldx, Astr->A12->U, &n1, &beta, inter, &m);
		zgemm("No", "No", &m, &n2, &Astr->A12->p, &alpha, inter, &m, Astr->A12->VT, &Astr->A12->p, &beta, Y12, &ldy12);

		UnsymmRecMultUpperRStruct(n1, m, Astr->left, &X[0 + ldx * 0], ldx, Y11, ldy11, smallsize);
		UnsymmRecMultUpperRStruct(n2, m, Astr->right, &X[0 + ldx * n1], ldx, Y22, ldy22, smallsize);

		// first part of Y = Y11
		zlacpy("All", &m, &n1, Y11, &ldy11, &Y[0 + ldy * 0], &ldy);

		// second part of Y = Y12 + Y22
		mkl_zomatadd('C', 'N', 'N', m, n2, 1.0, Y12, ldy12, 1.0, Y22, ldy22, &Y[0 + ldy * n1], ldy);

		free_arr(Y11);
		free_arr(Y12);
		free_arr(Y22);
		free_arr(inter);

	}
}

void UnsymmRecMultLowerLStruct(int n, int m, cumnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, int smallsize)
{
	dtype alpha = 1.0;
	dtype beta = 0.0;

	if (n <= smallsize)
	{
		zgemm("No", "No", &n, &m, &n, &alpha, Astr->A21->A, &n, X, &ldx, &beta, Y, &ldy);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // rounding up
		int n1 = n - n2;
		dtype *Y21 = alloc_arr2<dtype>(n2 * m); int ldy21 = n2;
		dtype *Y11 = alloc_arr2<dtype>(n1 * m); int ldy11 = n1;
		dtype *Y22 = alloc_arr2<dtype>(n2 * m); int ldy22 = n2;
		dtype *inter = alloc_arr2<dtype>(Astr->A21->p * m); // column major - lda = column

		// Y21 = U21 * V21T * X = (n2 x p) * (p x n1) * (n1 x m)
		zgemm("No", "No", &Astr->A21->p, &m, &n1, &alpha, Astr->A21->VT, &Astr->A21->p, &X[0 + 0 * ldx], &ldx, &beta, inter, &Astr->A21->p);
		zgemm("No", "No", &n2, &m, &Astr->A21->p, &alpha, Astr->A21->U, &n2, inter, &Astr->A21->p, &beta, Y21, &ldy21);
	
		UnsymmRecMultLowerLStruct(n1, m, Astr->left, &X[0 + ldx * 0], ldx, Y11, ldy11, smallsize);
		UnsymmRecMultLowerLStruct(n2, m, Astr->right, &X[n1 + ldx * 0], ldx, Y22, ldy22, smallsize);

		// first part of Y = Y11
		zlacpy("All", &n1, &m, Y11, &ldy11, &Y[0 + ldy * 0], &ldy);

		// second part of Y = Y21 + Y22
		mkl_zomatadd('C', 'N', 'N', n2, m, 1.0, Y21, ldy21, 1.0, Y22, ldy22, &Y[n1 + ldy * 0], ldy);


		free_arr(Y11);
		free_arr(Y21);
		free_arr(Y22);
		free_arr(inter);
	}
}

void UnsymmRecMultLowerRStruct(int n, int m, cumnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, int smallsize)
{
	dtype alpha = 1.0;
	dtype beta = 0.0;

	if (n <= smallsize)
	{
		zgemm("No", "No", &m, &n, &n, &alpha, X, &ldx, Astr->A21->A, &n, &beta, Y, &ldy);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // rounding up
		int n1 = n - n2;
		dtype *Y21 = alloc_arr2<dtype>(m * n1); int ldy21 = m;
		dtype *Y11 = alloc_arr2<dtype>(m * n1); int ldy11 = m;
		dtype *Y22 = alloc_arr2<dtype>(m * n2); int ldy22 = m;
		dtype *inter = alloc_arr2<dtype>(m * Astr->A21->p);

		// Y21 = X * U21 * V21T = (m x n2) * (n2 x p) * (p x n1)
		zgemm("No", "No", &m, &Astr->A21->p, &n2, &alpha, &X[0 + ldx * n1], &ldx, Astr->A21->U, &n2, &beta, inter, &m);
		zgemm("No", "No", &m, &n1, &Astr->A21->p, &alpha, inter, &m, Astr->A21->VT, &Astr->A21->p, &beta, Y21, &ldy21);

		UnsymmRecMultLowerRStruct(n1, m, Astr->left, &X[0 + ldx * 0], ldx, Y11, ldy11, smallsize);
		UnsymmRecMultLowerRStruct(n2, m, Astr->right, &X[0 + ldx * n1], ldx, Y22, ldy22, smallsize);


		// first part of Y = Y11 + Y21
		mkl_zomatadd('C', 'N', 'N', m, n1, 1.0, Y11, ldy11, 1.0, Y21, ldy21, &Y[0 + ldy * 0], ldy);

		// second part of Y = Y22
		zlacpy("All", &m, &n2, Y22, &ldy22, &Y[0 + ldy * n1], &ldy);

		free_arr(Y11);
		free_arr(Y21);
		free_arr(Y22);
		free_arr(inter);
	}
}

void AddStruct(int n, dtype alpha, cmnode* Astr, dtype beta, cmnode* Bstr, cmnode* &Cstr, int smallsize, double eps, char *method)
{
	dtype alpha_loc = 1.0;
	dtype beta_loc = 0.0;

	Cstr = (cmnode*)malloc(sizeof(cmnode));

	// n - order of A, B and C
	if (n <= smallsize)
	{
		alloc_dense_node(n, Cstr);
		mkl_zomatadd('C', 'N', 'N', n, n, alpha, Astr->A, n, beta, Bstr->A, n, Cstr->A, n);
		//Add_dense(n, n, alpha, A, lda, beta, B, ldb, C, ldc);
	}
	else
	{
		int p1 = 0, p2 = 0;
		int n2 = (int)ceil(n / 2.0); // округление в большую сторону
		int n1 = n - n2;
		int n1_dbl = Astr->p + Bstr->p;

		dtype *Y21 = alloc_arr2<dtype>(n2 * n1_dbl); int ldy21 = n2;
		dtype *Y12 = alloc_arr2<dtype>(n1 * n1_dbl); int ldy12 = n1;

		dtype *V21 = alloc_arr2<dtype>(n2 * n1_dbl); int ldv21 = n2;
		dtype *V12 = alloc_arr2<dtype>(n1 * n1_dbl); int ldv12 = n1;

		dtype *AU = alloc_arr2<dtype>(n2 * Astr->p); int ldau = n2;
		dtype *BU = alloc_arr2<dtype>(n2 * Bstr->p); int ldbu = n2;

		dtype *AV = alloc_arr2<dtype>(n1 * Astr->p); int ldav = n1;
		dtype *BV = alloc_arr2<dtype>(n1 * Bstr->p); int ldbv = n1;

		// Filling AV and BV - workspaces
		Mat_Trans(Astr->p, n1, Astr->VT, Astr->p, AV, ldav);
		Mat_Trans(Bstr->p, n1, Bstr->VT, Bstr->p, BV, ldbv);

		// Multiplying AU = alpha * AU and BU = beta * BU
		mkl_zomatcopy('C', 'N', n2, Astr->p, alpha, Astr->U, n2, AU, n2);
		mkl_zomatcopy('C', 'N', n2, Bstr->p, beta, Bstr->U, n2, BU, n2);
		//Add_dense(n2, n1, alpha, &A[n1 + lda * 0], lda, 0.0, B, ldb, &A[n1 + lda * 0], lda);
		//Add_dense(n2, n1, beta, &B[n1 + ldb * 0], ldb, 0.0, B, ldb, &B[n1 + ldb * 0], ldb);

		// Y21 = [alpha*A{2,1} beta*B{2,1}];
		zlacpy("All", &n2, &Astr->p, AU, &n2, &Y21[0 + ldy21 * 0], &ldy21);
		zlacpy("All", &n2, &Bstr->p, BU, &n2, &Y21[0 + ldy21 * Astr->p], &ldy21);

		// Y12 = [A{1,2}; B{1,2}];
		zlacpy("All", &n1, &Astr->p, AV, &ldav, &Y12[0 + ldy12 * 0], &ldy12);
		zlacpy("All", &n1, &Bstr->p, BV, &ldbv, &Y12[0 + ldy12 * Astr->p], &ldy12);

		// произведение Y21 и Y12 - это матрица n2 x n1
		//LowRankApprox(n2, n1_dbl, Y21, ldy21, V21, ldv21, p1, eps, "SVD"); // перезапись Y21
		//LowRankApprox(n1_dbl, n1, Y12, ldy12, V12, ldv12, p2, eps, "SVD");  // перезапись Y12

		cmnode* Y21str = (cmnode*)malloc(sizeof(cmnode));
		cmnode* Y12str = (cmnode*)malloc(sizeof(cmnode));
		LowRankApproxStruct(n2, n1_dbl, Y21, ldy21, Y21str, eps, "SVD");
		LowRankApproxStruct(n1, n1_dbl, Y12, ldy12, Y12str, eps, "SVD");

		zlacpy("All", &n2, &Y21str->p, Y21str->U, &n2, Y21, &ldy21);
		zlacpy("All", &Y21str->p, &n1_dbl, Y21str->VT, &Y21str->p, V21, &ldv21);

		zlacpy("All", &n1, &Y12str->p, Y12str->U, &n1, Y12, &ldy12);
		zlacpy("All", &Y12str->p, &n1_dbl, Y12str->VT, &Y12str->p, V12, &ldv12);

		p1 = Y21str->p;
		p2 = Y12str->p;

		// Y = V21'*V12;
		dtype *Y = alloc_arr2<dtype>(p1 * p2);
		zgemm("No", "Trans", &p1, &p2, &n1_dbl, &alpha_loc, V21, &ldv21, V12, &ldv12, &beta_loc, Y, &p1);

		// C{2,1} = U21*Y;   
		Cstr->U = alloc_arr2<dtype>(n2 * p2);
		zgemm("No", "No", &n2, &p2, &p1, &alpha_loc, Y21, &ldy21, Y, &p1, &beta_loc, Cstr->U, &n2); // mn

		// C{1,2} = U12';
		dtype *Y12_tr = alloc_arr2<dtype>(p2 * n1);
		Mat_Trans(n1, p2, Y12, ldy12, Y12_tr, p2);

		Cstr->VT = alloc_arr2<dtype>(p2 * n1);  Cstr->p = p2;
		zlacpy("All", &p2, &n1, Y12_tr, &p2, Cstr->VT, &p2);

		AddStruct(n1, alpha, Astr->left, beta, Bstr->left, Cstr->left, smallsize, eps, method);
		AddStruct(n2, alpha, Astr->right, beta, Bstr->right, Cstr->right, smallsize, eps, method);


		free_arr(Y21str->U);
		free_arr(Y21str->VT);
		free_arr(Y12str->U);
		free_arr(Y12str->VT);
		free_arr(Y21);
		free_arr(Y12);
		free_arr(V21);
		free_arr(V12);
		free_arr(AU);
		free_arr(BU);
		free_arr(AV);
		free_arr(BV);
		free_arr(Y);
		free_arr(Y12_tr);
		free_arr(Y21str);
		free_arr(Y12str);
	}

}

void UnsymmAddSubroutine(int n2, int n1, dtype alpha, cmnode* Astr, dtype beta, cmnode* Bstr, cmnode* &Cstr, int smallsize, double eps, char *method)
{
	int p1 = 0, p2 = 0;
	dtype alpha_loc = 1.0;
	dtype beta_loc = 0.0;
	int n1_dbl = Astr->p + Bstr->p;

	dtype *Y21 = alloc_arr2<dtype>(n2 * n1_dbl); int ldy21 = n2;
	dtype *Y12 = alloc_arr2<dtype>(n1 * n1_dbl); int ldy12 = n1;

	dtype *V21 = alloc_arr2<dtype>(n2 * n1_dbl); int ldv21 = n2;
	dtype *V12 = alloc_arr2<dtype>(n1 * n1_dbl); int ldv12 = n1;

	dtype *AU = alloc_arr2<dtype>(n2 * Astr->p); int ldau = n2;
	dtype *BU = alloc_arr2<dtype>(n2 * Bstr->p); int ldbu = n2;

	dtype *AV = alloc_arr2<dtype>(n1 * Astr->p); int ldav = n1;
	dtype *BV = alloc_arr2<dtype>(n1 * Bstr->p); int ldbv = n1;

	// Filling AV and BV - workspaces
	Mat_Trans(Astr->p, n1, Astr->VT, Astr->p, AV, ldav);
	Mat_Trans(Bstr->p, n1, Bstr->VT, Bstr->p, BV, ldbv);

	// Multiplying AU = alpha * AU and BU = beta * BU
	mkl_zomatcopy('C', 'N', n2, Astr->p, alpha, Astr->U, n2, AU, n2);
	mkl_zomatcopy('C', 'N', n2, Bstr->p, beta, Bstr->U, n2, BU, n2);
	//Add_dense(n2, n1, alpha, &A[n1 + lda * 0], lda, 0.0, B, ldb, &A[n1 + lda * 0], lda);
	//Add_dense(n2, n1, beta, &B[n1 + ldb * 0], ldb, 0.0, B, ldb, &B[n1 + ldb * 0], ldb);

	// Y21 = [alpha*A{2,1} beta*B{2,1}];
	zlacpy("All", &n2, &Astr->p, AU, &n2, &Y21[0 + ldy21 * 0], &ldy21);
	zlacpy("All", &n2, &Bstr->p, BU, &n2, &Y21[0 + ldy21 * Astr->p], &ldy21);

	// Y12 = [A{1,2}; B{1,2}];
	zlacpy("All", &n1, &Astr->p, AV, &ldav, &Y12[0 + ldy12 * 0], &ldy12);
	zlacpy("All", &n1, &Bstr->p, BV, &ldbv, &Y12[0 + ldy12 * Astr->p], &ldy12);

	// произведение Y21 и Y12 - это матрица n2 x n1
	//LowRankApprox(n2, n1_dbl, Y21, ldy21, V21, ldv21, p1, eps, "SVD"); // перезапись Y21
	//LowRankApprox(n1_dbl, n1, Y12, ldy12, V12, ldv12, p2, eps, "SVD");  // перезапись Y12

	cmnode* Y21str = (cmnode*)malloc(sizeof(cmnode));
	cmnode* Y12str = (cmnode*)malloc(sizeof(cmnode));
	LowRankApproxStruct(n2, n1_dbl, Y21, ldy21, Y21str, eps, "SVD");
	LowRankApproxStruct(n1, n1_dbl, Y12, ldy12, Y12str, eps, "SVD");

	zlacpy("All", &n2, &Y21str->p, Y21str->U, &n2, Y21, &ldy21);
	zlacpy("All", &Y21str->p, &n1_dbl, Y21str->VT, &Y21str->p, V21, &ldv21);

	zlacpy("All", &n1, &Y12str->p, Y12str->U, &n1, Y12, &ldy12);
	zlacpy("All", &Y12str->p, &n1_dbl, Y12str->VT, &Y12str->p, V12, &ldv12);

	p1 = Y21str->p;
	p2 = Y12str->p;

	// Y = V21'*V12;
	dtype *Y = alloc_arr2<dtype>(p1 * p2);
	zgemm("No", "Trans", &p1, &p2, &n1_dbl, &alpha_loc, V21, &ldv21, V12, &ldv12, &beta_loc, Y, &p1);

	// C{2,1} = U21*Y;   
	Cstr->U = alloc_arr2<dtype>(n2 * p2);
	zgemm("No", "No", &n2, &p2, &p1, &alpha_loc, Y21, &ldy21, Y, &p1, &beta_loc, Cstr->U, &n2); // mn

																									 // C{1,2} = U12';
	dtype *Y12_tr = alloc_arr2<dtype>(p2 * n1);
	Mat_Trans(n1, p2, Y12, ldy12, Y12_tr, p2);

	Cstr->VT = alloc_arr2<dtype>(p2 * n1); Cstr->p = p2;
	zlacpy("All", &p2, &n1, Y12_tr, &p2, Cstr->VT, &p2);


	free_arr(Y21str->U);
	free_arr(Y21str->VT);
	free_arr(Y12str->U);
	free_arr(Y12str->VT);
	free_arr(Y21);
	free_arr(Y12);
	free_arr(V21);
	free_arr(V12);
	free_arr(AU);
	free_arr(BU);
	free_arr(AV);
	free_arr(BV);
	free_arr(Y);
	free_arr(Y12_tr);
	free_arr(Y21str);
	free_arr(Y12str);
}

void UnsymmAddStruct(int n, dtype alpha, cumnode* Astr, dtype beta, cumnode* Bstr, cumnode* &Cstr, int smallsize, double eps, char *method)
{
	Cstr = (cumnode*)malloc(sizeof(cumnode));

	// n - order of A, B and C
	if (n <= smallsize)
	{
		alloc_dense_unsymm_node(n, Cstr);
		mkl_zomatadd('C', 'N', 'N', n, n, alpha, Astr->A21->A, n, beta, Bstr->A21->A, n, Cstr->A21->A, n);
		//Add_dense(n, n, alpha, A, lda, beta, B, ldb, C, ldc);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // округление в большую сторону
		int n1 = n - n2;

		Cstr->A21 = (cmnode*)malloc(sizeof(cmnode));
		Cstr->A12 = (cmnode*)malloc(sizeof(cmnode));

		UnsymmAddSubroutine(n2, n1, alpha, Astr->A21, beta, Bstr->A21, Cstr->A21, smallsize, eps, method);
		UnsymmAddSubroutine(n1, n2, alpha, Astr->A12, beta, Bstr->A12, Cstr->A12, smallsize, eps, method);
		
		UnsymmAddStruct(n1, alpha, Astr->left, beta, Bstr->left, Cstr->left, smallsize, eps, method);
		UnsymmAddStruct(n2, alpha, Astr->right, beta, Bstr->right, Cstr->right, smallsize, eps, method);
	}

}


void SymCompUpdate2Struct(int n, int k, cmnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V, int ldv, cmnode* &Bstr, int smallsize, double eps, char* method)
{
	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;
	int p1 = 0, p2 = 0;

	if (abs(alpha) < eps)
	{
		CopyStruct(n, Astr, Bstr, smallsize);
		return;
	}

	Bstr = (cmnode*)malloc(sizeof(cmnode));

	if (n <= smallsize)
	{
		// X = X + alpha * V * Y * VT

		// C = V * Y
		dtype *C = alloc_arr2<dtype>(n * k); int ldc = n;
		zsymm("Right", "Up", &n, &k, &alpha_one, Y, &ldy, V, &ldv, &beta_zero, C, &ldc);

		// Copy Astr->A to A_init
		dtype *A_init = alloc_arr2<dtype>(n * n); int lda = n;
		zlacpy("All", &n, &n, Astr->A, &lda, A_init, &lda);

		// X = X + alpha * C * Vt
		zgemm("No", "Trans", &n, &n, &k, &alpha, C, &ldc, V, &ldv, &beta_one, A_init, &lda);

		// B = A
		alloc_dense_node(n, Bstr);
		zlacpy("All", &n, &n, A_init, &lda, Bstr->A, &lda);

		free_arr(C);
		free_arr(A_init);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		int nk = Astr->p + k;
		// for this division n2 > n1 we can store a low memory

		dtype *Y12 = alloc_arr2<dtype>(n1 * nk); int ldy12 = n1;
		dtype *Y21 = alloc_arr2<dtype>(n2 * nk); int ldy21 = n2;

		dtype *V_up = alloc_arr2<dtype>(n1 * k); int ldvup = n1;
		dtype *A12 = alloc_arr2<dtype>(n1 * Astr->p); int lda12 = n1;

		dtype *VY = alloc_arr2<dtype>(n2 * k); int ldvy = n2;

		dtype *V12 = alloc_arr2<dtype>(n1 * nk); int ldv12 = n1;
		dtype *V21 = alloc_arr2<dtype>(n2 * nk); int ldv21 = n2;

		zgemm("No", "No", &n2, &k, &k, &alpha, &V[n1 + ldv * 0], &ldv, Y, &ldy, &beta_zero, VY, &ldvy);

		// Y21 = [A{2,1} alpha*V(m:n,:)*Y];
		zlacpy("All", &n2, &Astr->p, Astr->U, &n2, &Y21[0 + ldy21 * 0], &ldy21);
		zlacpy("All", &n2, &k, VY, &ldvy, &Y21[0 + ldy21 * Astr->p], &ldy21);

		// Y12 = [A{1,2} V(1:n1,:)];
		zlacpy("All", &n1, &k, &V[0 + ldv * 0], &ldv, V_up, &ldvup);
		Mat_Trans(Astr->p, n1, Astr->VT, Astr->p, A12, lda12);
		zlacpy("All", &n1, &Astr->p, A12, &lda12, &Y12[0 + ldy12 * 0], &ldy12);
		zlacpy("All", &n1, &k, V_up, &ldvup, &Y12[0 + ldy12 * Astr->p], &ldy12);

		//	LowRankApprox(n2, nk, Y21, ldy21, V21, ldv21, p1, eps, "SVD");
		//	LowRankApprox(n1, nk, Y12, ldy12, V12, ldv12, p2, eps, "SVD");

		cmnode* Y21str = (cmnode*)malloc(sizeof(cmnode));
		cmnode* Y12str = (cmnode*)malloc(sizeof(cmnode));

		// [U21,V21] = LowRankApprox (Y21, eps, method);
		LowRankApproxStruct(n2, nk, Y21, ldy21, Y21str, eps, "SVD");

		// [U12, V12] = LowRankApprox(Y12, eps, method);
		LowRankApproxStruct(n1, nk, Y12, ldy12, Y12str, eps, "SVD");

		zlacpy("All", &n2, &Y21str->p, Y21str->U, &n2, Y21, &ldy21);
		zlacpy("All", &Y21str->p, &nk, Y21str->VT, &Y21str->p, V21, &ldv21);

		zlacpy("All", &n1, &Y12str->p, Y12str->U, &n1, Y12, &ldy12);
		zlacpy("All", &Y12str->p, &nk, Y12str->VT, &Y12str->p, V12, &ldv12);

		p1 = Y21str->p;
		p2 = Y12str->p;
		Bstr->p = p2;

		// V21 * Y12
		dtype *VV = alloc_arr2<dtype>(p1 * p2);
		dtype *V_tr = alloc_arr2<dtype>(nk * p2);
		Mat_Trans(p2, nk, V12, ldv12, V_tr, nk);
		zgemm("No", "No", &p1, &p2, &nk, &alpha_one, V21, &ldv21, V_tr, &nk, &beta_zero, VV, &p1);

		// B{2,1} = U21*(V21'*V12);
		Bstr->U = alloc_arr2<dtype>(n2 * p2);
		zgemm("No", "No", &n2, &p2, &p1, &alpha_one, Y21, &ldy21, VV, &p1, &beta_zero, Bstr->U, &n2);

		// B{1,2} = U12;
		Bstr->VT = alloc_arr2<dtype>(p2 * n1);
		Mat_Trans(n1, p2, Y12, ldy12, Bstr->VT, p2);

		// B{1,1} = SymCompUpdate2 (A{1,1}, Y, V(1:n1,:), alpha, eps, method);
		SymCompUpdate2Struct(n1, k, Astr->left, alpha, Y, ldy, &V[0 + ldv * 0], ldv, Bstr->left, smallsize, eps, method);

		// B{2,2} = SymCompUpdate2 (A{2,2}, Y, V(m:n ,:), alpha, eps, method);
		SymCompUpdate2Struct(n2, k, Astr->right, alpha, Y, ldy, &V[n1 + ldv * 0], ldv, Bstr->right, smallsize, eps, method);


		free_arr(Y12str->U);
		free_arr(Y12str->VT);
		free_arr(Y21str->U);
		free_arr(Y21str->VT);
		free_arr(Y21);
		free_arr(Y12);
		free_arr(V21);
		free_arr(V12);
		free_arr(VY);
		free_arr(VV);
		free_arr(V_tr);
		free_arr(Y21str);
		free_arr(Y12str);
	}
}

void UnsymmUpdate2Subroutine(int n2, int n1, int k, cmnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V1, int ldv1, dtype* V2, int ldv2, cmnode* &Bstr, int smallsize, double eps, char* method)
{
	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;
	int p1 = 0, p2 = 0;

	int nk = Astr->p + k;
	// for this division n2 > n1 we can store a low memory

	dtype *Y12 = alloc_arr2<dtype>(n1 * nk); int ldy12 = n1;
	dtype *Y21 = alloc_arr2<dtype>(n2 * nk); int ldy21 = n2;

	dtype *V_up = alloc_arr2<dtype>(n1 * k); int ldvup = n1;
	dtype *A12 = alloc_arr2<dtype>(n1 * Astr->p); int lda12 = n1;

	dtype *VY = alloc_arr2<dtype>(n2 * k); int ldvy = n2;

	dtype *V12 = alloc_arr2<dtype>(n1 * nk); int ldv12 = n1;
	dtype *V21 = alloc_arr2<dtype>(n2 * nk); int ldv21 = n2;

	zgemm("No", "No", &n2, &k, &k, &alpha, V1, &ldv1, Y, &ldy, &beta_zero, VY, &ldvy);

	// Y21 = [A{2,1} alpha*V(m:n,:)*Y];
	zlacpy("All", &n2, &Astr->p, Astr->U, &n2, &Y21[0 + ldy21 * 0], &ldy21);
	zlacpy("All", &n2, &k, VY, &ldvy, &Y21[0 + ldy21 * Astr->p], &ldy21);

	// Y12 = [A{1,2} V(1:n1,:)];
	zlacpy("All", &n1, &k, V2, &ldv2, V_up, &ldvup);
	Mat_Trans(Astr->p, n1, Astr->VT, Astr->p, A12, lda12);
	zlacpy("All", &n1, &Astr->p, A12, &lda12, &Y12[0 + ldy12 * 0], &ldy12);
	zlacpy("All", &n1, &k, V_up, &ldvup, &Y12[0 + ldy12 * Astr->p], &ldy12);

	//	LowRankApprox(n2, nk, Y21, ldy21, V21, ldv21, p1, eps, "SVD");
	//	LowRankApprox(n1, nk, Y12, ldy12, V12, ldv12, p2, eps, "SVD");

	cmnode* Y21str = (cmnode*)malloc(sizeof(cmnode));
	cmnode* Y12str = (cmnode*)malloc(sizeof(cmnode));

	// [U21,V21] = LowRankApprox (Y21, eps, method);
	LowRankApproxStruct(n2, nk, Y21, ldy21, Y21str, eps, "SVD");

	// [U12, V12] = LowRankApprox(Y12, eps, method);
	LowRankApproxStruct(n1, nk, Y12, ldy12, Y12str, eps, "SVD");

	zlacpy("All", &n2, &Y21str->p, Y21str->U, &n2, Y21, &ldy21);
	zlacpy("All", &Y21str->p, &nk, Y21str->VT, &Y21str->p, V21, &ldv21);

	zlacpy("All", &n1, &Y12str->p, Y12str->U, &n1, Y12, &ldy12);
	zlacpy("All", &Y12str->p, &nk, Y12str->VT, &Y12str->p, V12, &ldv12);

	p1 = Y21str->p;
	p2 = Y12str->p;
	Bstr->p = p2;

	// V21 * Y12
	dtype *VV = alloc_arr2<dtype>(p1 * p2);
	dtype *V_tr = alloc_arr2<dtype>(nk * p2);
	Mat_Trans(p2, nk, V12, ldv12, V_tr, nk);
	zgemm("No", "No", &p1, &p2, &nk, &alpha_one, V21, &ldv21, V_tr, &nk, &beta_zero, VV, &p1);

	// B{2,1} = U21*(V21'*V12);
	Bstr->U = alloc_arr2<dtype>(n2 * p2);
	zgemm("No", "No", &n2, &p2, &p1, &alpha_one, Y21, &ldy21, VV, &p1, &beta_zero, Bstr->U, &n2);

	// B{1,2} = U12;
	Bstr->VT = alloc_arr2<dtype>(p2 * n1);
	Mat_Trans(n1, p2, Y12, ldy12, Bstr->VT, p2);

	free_arr(Y12str->U);
	free_arr(Y12str->VT);
	free_arr(Y21str->U);
	free_arr(Y21str->VT);
	free_arr(Y21);
	free_arr(Y12);
	free_arr(V21);
	free_arr(V12);
	free_arr(VY);
	free_arr(VV);
	free_arr(V_tr);
	free_arr(Y21str);
	free_arr(Y12str);
}

void UnsymmCompUpdate2Struct(int n, int k, cumnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V, int ldv, cumnode* &Bstr, int smallsize, double eps, char* method)
{
	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;

	if (abs(alpha) < 10e-8)
	{
		CopyUnsymmStruct(n, Astr, Bstr, smallsize);
		return;
	}

	Bstr = (cumnode*)malloc(sizeof(cumnode));

	if (n <= smallsize)
	{
		// X = X + alpha * V * Y * VT

		// C = V * Y
		dtype *C = alloc_arr2<dtype>(n * k); int ldc = n;
		zgemm("No", "No", &n, &k, &k, &alpha_one, V, &ldv, Y, &ldy, &beta_zero, C, &ldc);

		// Copy Astr->A to A_init
		dtype *A_init = alloc_arr2<dtype>(n * n); int lda = n;
		zlacpy("All", &n, &n, Astr->A21->A, &lda, A_init, &lda);

		// X = X + alpha * C * Vt
		zgemm("No", "Trans", &n, &n, &k, &alpha, C, &ldc, V, &ldv, &beta_one, A_init, &lda);

		// B = A
		alloc_dense_unsymm_node(n, Bstr);
		zlacpy("All", &n, &n, A_init, &lda, Bstr->A21->A, &lda);

		free_arr(C);
		free_arr(A_init);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		Bstr->A21 = (cmnode*)malloc(sizeof(cmnode));
		Bstr->A12 = (cmnode*)malloc(sizeof(cmnode));

		UnsymmUpdate2Subroutine(n2, n1, k, Astr->A21, alpha, Y, ldy, &V[n1 + ldv * 0], ldv, &V[0 + ldv * 0], ldv, Bstr->A21, smallsize, eps, method);
		UnsymmUpdate2Subroutine(n1, n2, k, Astr->A12, alpha, Y, ldy, &V[0 + ldv * 0], ldv, &V[n1 + ldv * 0], ldv, Bstr->A12, smallsize, eps, method);

		// B{1,1} = SymCompUpdate2 (A{1,1}, Y, V(1:n1,:), alpha, eps, method);
		UnsymmCompUpdate2Struct(n1, k, Astr->left, alpha, Y, ldy, &V[0 + ldv * 0], ldv, Bstr->left, smallsize, eps, method);

		// B{2,2} = SymCompUpdate2 (A{2,2}, Y, V(m:n ,:), alpha, eps, method);
		UnsymmCompUpdate2Struct(n2, k, Astr->right, alpha, Y, ldy, &V[n1 + ldv * 0], ldv, Bstr->right, smallsize, eps, method);
	}
}
/* (n2 x k1) * (k1 x k2) * (k2 x n1) */
#if 0
void UnsymmUpdate3Subroutine(int n2, int n1, int k, cmnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V1, int ldv1, dtype* V2, int ldv2, cmnode* &Bstr, int smallsize, double eps, char* method)
{
	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;
	int p1 = 0, p2 = 0;

	int nk = Astr->p + k;
	// for this division n2 > n1 we can store a low memory

	dtype *Y12 = alloc_arr2<dtype>(n1 * nk); int ldy12 = n1;
	dtype *Y21 = alloc_arr2<dtype>(n2 * nk); int ldy21 = n2;

	dtype *V2_up = alloc_arr2<dtype>(k * n1); int ldv2up = k;
	dtype *V2_tr = alloc_arr2<dtype>(n1 * k); int ldv2tr = n1;
	dtype *A12 = alloc_arr2<dtype>(n1 * Astr->p); int lda12 = n1;

	dtype *VY = alloc_arr2<dtype>(n2 * k); int ldvy = n2;

	dtype *V12 = alloc_arr2<dtype>(n1 * nk); int ldv12 = n1;
	dtype *V21 = alloc_arr2<dtype>(n2 * nk); int ldv21 = n2;

	zgemm("No", "No", &n2, &k, &k, &alpha, V1, &ldv1, Y, &ldy, &beta_zero, VY, &ldvy);

	// Y21 = [A{2,1} alpha*V1(m:n,:)*Y];
	zlacpy("All", &n2, &Astr->p, Astr->U, &n2, &Y21[0 + ldy21 * 0], &ldy21);
	zlacpy("All", &n2, &k, VY, &ldvy, &Y21[0 + ldy21 * Astr->p], &ldy21);

	// Y12 = [A{1,2} V2(1:n1,:)];
	zlacpy("All", &k, &n1, V2, &ldv2, V2_up, &ldv2up);
	Mat_Trans(Astr->p, n1, Astr->VT, Astr->p, A12, lda12);
	zlacpy("All", &n1, &Astr->p, A12, &lda12, &Y12[0 + ldy12 * 0], &ldy12);

	Mat_Trans(k, n1, V2_up, ldv2up, V2_tr, ldv2tr);
	zlacpy("All", &n1, &k, V2_tr, &ldv2tr, &Y12[0 + ldy12 * Astr->p], &ldy12);

	//	LowRankApprox(n2, nk, Y21, ldy21, V21, ldv21, p1, eps, "SVD");
	//	LowRankApprox(n1, nk, Y12, ldy12, V12, ldv12, p2, eps, "SVD");

	cmnode* Y21str = (cmnode*)malloc(sizeof(cmnode));
	cmnode* Y12str = (cmnode*)malloc(sizeof(cmnode));

	// [U21,V21] = LowRankApprox (Y21, eps, method);
	LowRankApproxStruct(n2, nk, Y21, ldy21, Y21str, eps, "SVD");

	// [U12, V12] = LowRankApprox(Y12, eps, method);
	LowRankApproxStruct(n1, nk, Y12, ldy12, Y12str, eps, "SVD");

	zlacpy("All", &n2, &Y21str->p, Y21str->U, &n2, Y21, &ldy21);
	zlacpy("All", &Y21str->p, &nk, Y21str->VT, &Y21str->p, V21, &ldv21);

	zlacpy("All", &n1, &Y12str->p, Y12str->U, &n1, Y12, &ldy12);
	zlacpy("All", &Y12str->p, &nk, Y12str->VT, &Y12str->p, V12, &ldv12);

	p1 = Y21str->p;
	p2 = Y12str->p;
	Bstr->p = p2;

	// V21 * Y12
	dtype *VV = alloc_arr2<dtype>(p1 * p2);
	dtype *V_tr = alloc_arr2<dtype>(nk * p2);
	Mat_Trans(p2, nk, V12, ldv12, V_tr, nk);
	zgemm("No", "No", &p1, &p2, &nk, &alpha_one, V21, &ldv21, V_tr, &nk, &beta_zero, VV, &p1);

	// B{2,1} = U21*(V21'*V12);
	Bstr->U = alloc_arr2<dtype>(n2 * p2);
	zgemm("No", "No", &n2, &p2, &p1, &alpha_one, Y21, &ldy21, VV, &p1, &beta_zero, Bstr->U, &n2);

	// B{1,2} = U12;
	Bstr->VT = alloc_arr2<dtype>(p2 * n1);
	Mat_Trans(n1, p2, Y12, ldy12, Bstr->VT, p2);

	free_arr(Y12str->U);
	free_arr(Y12str->VT);
	free_arr(Y21str->U);
	free_arr(Y21str->VT);
	free_arr(Y21);
	free_arr(Y12);
	free_arr(V21);
	free_arr(V12);
	free_arr(VY);
	free_arr(VV);
	free_arr(V_tr);
	free_arr(Y21str);
	free_arr(Y12str);
}
#else
/* (n2 x k1) * (k1 x k2) * (k2 x n1) */
void UnsymmUpdate3Subroutine(int n2, int n1, int k1, int k2, cmnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V1, int ldv1, dtype* V2, int ldv2, cmnode* &Bstr, int smallsize, double eps, char* method)
{
	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;
	int p1 = 0, p2 = 0;

	int nk1 = Astr->p + k1;
	int nk2 = Astr->p + k2;
	// for this division n2 > n1 we can store a low memory

	dtype *Y12 = alloc_arr2<dtype>(n1 * nk2); int ldy12 = n1;
	dtype *Y21 = alloc_arr2<dtype>(n2 * nk2); int ldy21 = n2;

	dtype *V2_up = alloc_arr2<dtype>(k2 * n1); int ldv2up = k2;
	dtype *V2_tr = alloc_arr2<dtype>(n1 * k2); int ldv2tr = n1;
	dtype *A12 = alloc_arr2<dtype>(n1 * Astr->p); int lda12 = n1;

	dtype *VY = alloc_arr2<dtype>(n2 * k2); int ldvy = n2;

	dtype *V12 = alloc_arr2<dtype>(n1 * nk2); int ldv12 = n1;
	dtype *V21 = alloc_arr2<dtype>(n2 * nk2); int ldv21 = n2;

	zgemm("No", "No", &n2, &k2, &k1, &alpha, V1, &ldv1, Y, &ldy, &beta_zero, VY, &ldvy);

	// Y21 = [A{2,1} alpha*V1(m:n,:)*Y];
	zlacpy("All", &n2, &Astr->p, Astr->U, &n2, &Y21[0 + ldy21 * 0], &ldy21);
	zlacpy("All", &n2, &k2, VY, &ldvy, &Y21[0 + ldy21 * Astr->p], &ldy21);

	// Y12 = [A{1,2} V2(1:n1,:)];
	zlacpy("All", &k2, &n1, V2, &ldv2, V2_up, &ldv2up);
	Mat_Trans(Astr->p, n1, Astr->VT, Astr->p, A12, lda12);
	zlacpy("All", &n1, &Astr->p, A12, &lda12, &Y12[0 + ldy12 * 0], &ldy12);

	Mat_Trans(k2, n1, V2_up, ldv2up, V2_tr, ldv2tr);
	zlacpy("All", &n1, &k2, V2_tr, &ldv2tr, &Y12[0 + ldy12 * Astr->p], &ldy12);

	//	LowRankApprox(n2, nk, Y21, ldy21, V21, ldv21, p1, eps, "SVD");
	//	LowRankApprox(n1, nk, Y12, ldy12, V12, ldv12, p2, eps, "SVD");

	cmnode* Y21str = (cmnode*)malloc(sizeof(cmnode));
	cmnode* Y12str = (cmnode*)malloc(sizeof(cmnode));

	// [U21, V21] = LowRankApprox (Y21, eps, method);
	LowRankApproxStruct(n2, nk2, Y21, ldy21, Y21str, eps, "SVD");

	// [U12, V12] = LowRankApprox (Y12, eps, method);
	LowRankApproxStruct(n1, nk2, Y12, ldy12, Y12str, eps, "SVD");

	zlacpy("All", &n2, &Y21str->p, Y21str->U, &n2, Y21, &ldy21);
	zlacpy("All", &Y21str->p, &nk2, Y21str->VT, &Y21str->p, V21, &ldv21);

	zlacpy("All", &n1, &Y12str->p, Y12str->U, &n1, Y12, &ldy12);
	zlacpy("All", &Y12str->p, &nk2, Y12str->VT, &Y12str->p, V12, &ldv12);

	p1 = Y21str->p;
	p2 = Y12str->p;
	Bstr->p = p2;

	// V21 * Y12
	dtype *VV = alloc_arr2<dtype>(p1 * p2);
	dtype *V_tr = alloc_arr2<dtype>(nk2 * p2);
	Mat_Trans(p2, nk2, V12, ldv12, V_tr, nk2);
	zgemm("No", "No", &p1, &p2, &nk2, &alpha_one, V21, &ldv21, V_tr, &nk2, &beta_zero, VV, &p1);

	// B{2,1} = U21*(V21'*V12);
	Bstr->U = alloc_arr2<dtype>(n2 * p2);
	zgemm("No", "No", &n2, &p2, &p1, &alpha_one, Y21, &ldy21, VV, &p1, &beta_zero, Bstr->U, &n2);

	// B{1,2} = U12;
	Bstr->VT = alloc_arr2<dtype>(p2 * n1);
	Mat_Trans(n1, p2, Y12, ldy12, Bstr->VT, p2);

	free_arr(Y12str->U);
	free_arr(Y12str->VT);
	free_arr(Y21str->U);
	free_arr(Y21str->VT);
	free_arr(Y21);
	free_arr(Y12);
	free_arr(V21);
	free_arr(V12);
	free_arr(VY);
	free_arr(VV);
	free_arr(V_tr);
	free_arr(Y21str);
	free_arr(Y12str);
}
#endif

// U_21 * VT_21 + V1 * Y * V2 = U_21 * (E + Y) * V2
void SymUpdate4Subroutine(int n2, int n1, dtype alpha, cmnode* Astr, const dtype *Y, int ldy, cmnode* &Bstr, int smallsize, double eps, char* method)
{
	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;

	// Ranks
	int p = Astr->p;
	Bstr->p = p;

	dtype *EY = alloc_arr<dtype>(p * p);
	zlacpy("All", &p, &p, Y, &ldy, EY, &ldy);

	MultVectorConst(p * p, EY, alpha, EY);

	// Y = E + Y
	for (int i = 0; i < p; i++)
		EY[i + ldy * i] += 1.0;

	// B{2,1} = U21*(V21'*V12);
	Bstr->U = alloc_arr2<dtype>(n2 * p);
	zgemm("No", "No", &n2, &p, &p, &alpha_one, Astr->U, &n2, EY, &ldy, &beta_zero, Bstr->U, &n2);

	Bstr->VT = alloc_arr2<dtype>(p * n1);
	zlacpy("All", &p, &n1, Astr->VT, &p, Bstr->VT, &p);

	free_arr(EY);
}

#if 0
/* B: = A - V1 * Xunsymm * V2 
(n x k1) * (k1 x k2) * (k2 x n) */

void UnsymmCompUpdate3Struct(int n, int k, cumnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V1, int ldv1, dtype *V2, int ldv2, cumnode* &Bstr, int smallsize, double eps, char* method)
{
	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;

	if (abs(alpha) < 10e-8)
	{
		CopyUnsymmStruct(n, Astr, Bstr, smallsize);
		return;
	}

	Bstr = (cumnode*)malloc(sizeof(cumnode));

	if (n <= smallsize)
	{
		// X = X + alpha * V * Y * VT

		// C = V * Y
		dtype *C = alloc_arr2<dtype>(n * k); int ldc = n;
		zgemm("No", "No", &n, &k, &k, &alpha_one, V1, &ldv1, Y, &ldy, &beta_zero, C, &ldc);

		// Copy Astr->A to A_init (we do not change A)
		dtype *A_init = alloc_arr2<dtype>(n * n); int lda = n;
		zlacpy("All", &n, &n, Astr->A21->A, &lda, A_init, &lda);

		// X = X + alpha * C * V
		zgemm("No", "No", &n, &n, &k, &alpha, C, &ldc, V2, &ldv2, &beta_one, A_init, &lda);

		// B = A
		alloc_dense_unsymm_node(n, Bstr);
		zlacpy("All", &n, &n, A_init, &lda, Bstr->A21->A, &lda);

		free_arr(C);
		free_arr(A_init);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		Bstr->A21 = (cmnode*)malloc(sizeof(cmnode));
		Bstr->A12 = (cmnode*)malloc(sizeof(cmnode));

		UnsymmUpdate3Subroutine(n2, n1, k, Astr->A21, alpha, Y, ldy, &V1[n1 + ldv1 * 0], ldv1, &V2[0 + ldv2 * 0], ldv2, Bstr->A21, smallsize, eps, method);
		UnsymmUpdate3Subroutine(n1, n2, k, Astr->A12, alpha, Y, ldy, &V1[0 + ldv1 * 0], ldv1, &V2[0 + ldv2 * n1], ldv2, Bstr->A12, smallsize, eps, method);

		// B{1,1} = SymCompUpdate2 (A{1,1}, Y, V(1:n1,:), alpha, eps, method);
		UnsymmCompUpdate3Struct(n1, k, Astr->left, alpha, Y, ldy, &V1[0 + ldv1 * 0], ldv1, &V2[0 + ldv2 * 0], ldv2, Bstr->left, smallsize, eps, method);

		// B{2,2} = SymCompUpdate2 (A{2,2}, Y, V(m:n ,:), alpha, eps, method);
		UnsymmCompUpdate3Struct(n2, k, Astr->right, alpha, Y, ldy, &V1[n1 + ldv1 * 0], ldv1, &V2[0 + ldv2 * n1], ldv2, Bstr->right, smallsize, eps, method);
	}
}
#else
/* B: = A - V1 * Xunsymm * V2 = (n x k1) * (k1 x k2) * (k2 x n) */
void UnsymmCompUpdate3Struct(int n, int k1, int k2, cumnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V1, int ldv1, dtype *V2, int ldv2, cumnode* &Bstr, int smallsize, double eps, char* method)
{
	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;

	if (abs(alpha) < 10e-8)
	{
		CopyUnsymmStruct(n, Astr, Bstr, smallsize);
		return;
	}

	Bstr = (cumnode*)malloc(sizeof(cumnode));

	if (n <= smallsize)
	{
		// X = X + alpha * V * Y * VT

		// C = V * Y
		dtype *C = alloc_arr2<dtype>(n * k2); int ldc = n;
		zgemm("No", "No", &n, &k2, &k1, &alpha_one, V1, &ldv1, Y, &ldy, &beta_zero, C, &ldc);

		// Copy Astr->A to A_init (we do not change A)
		dtype *A_init = alloc_arr2<dtype>(n * n); int lda = n;
		zlacpy("All", &n, &n, Astr->A21->A, &lda, A_init, &lda);

		// X = X + alpha * C * V
		zgemm("No", "No", &n, &n, &k2, &alpha, C, &ldc, V2, &ldv2, &beta_one, A_init, &lda);

		// B = A
		alloc_dense_unsymm_node(n, Bstr);
		zlacpy("All", &n, &n, A_init, &lda, Bstr->A21->A, &lda);

		free_arr(C);
		free_arr(A_init);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		Bstr->A21 = (cmnode*)malloc(sizeof(cmnode));
		Bstr->A12 = (cmnode*)malloc(sizeof(cmnode));

		UnsymmUpdate3Subroutine(n2, n1, k1, k2, Astr->A21, alpha, Y, ldy, &V1[n1 + ldv1 * 0], ldv1, &V2[0 + ldv2 * 0], ldv2, Bstr->A21, smallsize, eps, method);
		UnsymmUpdate3Subroutine(n1, n2, k1, k2, Astr->A12, alpha, Y, ldy, &V1[0 + ldv1 * 0], ldv1, &V2[0 + ldv2 * n1], ldv2, Bstr->A12, smallsize, eps, method);

		// B{1,1} = SymCompUpdate2 (A{1,1}, Y, V(1:n1,:), alpha, eps, method);
		UnsymmCompUpdate3Struct(n1, k1, k2, Astr->left, alpha, Y, ldy, &V1[0 + ldv1 * 0], ldv1, &V2[0 + ldv2 * 0], ldv2, Bstr->left, smallsize, eps, method);

		// B{2,2} = SymCompUpdate2 (A{2,2}, Y, V(m:n ,:), alpha, eps, method);
		UnsymmCompUpdate3Struct(n2, k1, k2, Astr->right, alpha, Y, ldy, &V1[n1 + ldv1 * 0], ldv1, &V2[0 + ldv2 * n1], ldv2, Bstr->right, smallsize, eps, method);
	}
}
#endif

void SymCompUpdate4LowRankStruct(int n, int k1, int k2, cumnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V1, int ldv1, dtype *V2, int ldv2, cumnode* &Bstr, int smallsize, double eps, char* method)
{
	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;

	if (abs(alpha) < 10e-8)
	{
		CopyUnsymmStruct(n, Astr, Bstr, smallsize);
		return;
	}

	Bstr = (cumnode*)malloc(sizeof(cumnode));

	if (n <= smallsize)
	{
		// X = X + alpha * V * Y * VT

		// C = V * Y
		dtype *C = alloc_arr2<dtype>(n * k2); int ldc = n;
		zgemm("No", "No", &n, &k2, &k1, &alpha_one, V1, &ldv1, Y, &ldy, &beta_zero, C, &ldc);

		// Copy Astr->A to A_init (we do not change A)
		dtype *A_init = alloc_arr2<dtype>(n * n); int lda = n;
		zlacpy("All", &n, &n, Astr->A21->A, &lda, A_init, &lda);

		// X = X + alpha * C * V
		zgemm("No", "No", &n, &n, &k2, &alpha, C, &ldc, V2, &ldv2, &beta_one, A_init, &lda);

		// B = A
		alloc_dense_unsymm_node(n, Bstr);
		zlacpy("All", &n, &n, A_init, &lda, Bstr->A21->A, &lda);

		free_arr(C);
		free_arr(A_init);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		Bstr->A21 = (cmnode*)malloc(sizeof(cmnode));
		Bstr->A12 = (cmnode*)malloc(sizeof(cmnode));

//#define TEST

#ifdef TEST
		//	UnsymmUpdate3Subroutine(n2, n1, k1, k2, Astr->A21, alpha, Y, ldy, &V1[n1 + ldv1 * 0], ldv1, &V2[0 + ldv2 * 0], ldv2, Bstr->A21, smallsize, eps, method);
		//  UnsymmUpdate3Subroutine(n1, n2, k1, k2, Astr->A12, alpha, Y, ldy, &V1[0 + ldv1 * 0], ldv1, &V2[0 + ldv2 * n1], ldv2, Bstr->A12, smallsize, eps, method);

	//	V1[n1 + ldv1 * 0],  V2[0 + ldv2 * 0]
	//	V1[0 + ldv1 * 0],   V2[0 + ldv2 * n1]

		RelError(zlange, n2, k1, Astr->A21->U, n2, &V1[n1 + ldv1 * 0], ldv1, eps);
		RelError(zlange, k1, n1, Astr->A21->VT, k1, &V2[0 + ldv2 * 0], ldv2, eps);

		RelError(zlange, n1, k1, Astr->A12->U, n1, &V1[0 + ldv1 * 0], ldv1, eps);
		RelError(zlange, k1, n2, Astr->A12->VT, k1, &V2[0 + ldv2 * n1], ldv2, eps);
		system("pause");
#endif

		SymUpdate4Subroutine(n2, n1, alpha, Astr->A21, Y, ldy, Bstr->A21, smallsize, eps, method);
		SymUpdate4Subroutine(n1, n2, alpha, Astr->A12, Y, ldy, Bstr->A12, smallsize, eps, method);

		// B{1,1} = SymCompUpdate2 (A{1,1}, Y, V(1:n1,:), alpha, eps, method);
		SymCompUpdate4LowRankStruct(n1, k1, k2, Astr->left, alpha, Y, ldy, &V1[0 + ldv1 * 0], ldv1, &V2[0 + ldv2 * 0], ldv2, Bstr->left, smallsize, eps, method);

		// B{2,2} = SymCompUpdate2 (A{2,2}, Y, V(m:n ,:), alpha, eps, method);
		SymCompUpdate4LowRankStruct(n2, k1, k2, Astr->right, alpha, Y, ldy, &V1[n1 + ldv1 * 0], ldv1, &V2[0 + ldv2 * n1], ldv2, Bstr->right, smallsize, eps, method);
	}
}

void SymCompRecInvStruct(int n, cmnode* Astr, cmnode* &Bstr, int smallsize, double eps, char *method)
{
	dtype alpha_one = 1.0;
	dtype alpha_mone = -1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;
	int info = 0;
	dtype wquery = 0;
	int lwork = -1;

	Bstr = (cmnode*)malloc(sizeof(cmnode));

	if (n <= smallsize)
	{
		int *ipiv = alloc_arr2<int>(n);
		dtype *A_init = alloc_arr2<dtype>(n * n);

		zlacpy("All", &n, &n, Astr->A, &n, A_init, &n);

		// LU factorization of A
		zgetrf(&n, &n, A_init, &n, ipiv, &info);

		// space query
		zgetri(&n, A_init, &n, ipiv, &wquery, &lwork, &info);

		lwork = (int)wquery.real();
		dtype *work = alloc_arr2<dtype>(lwork);

		// inversion of A
		zgetri(&n, A_init, &n, ipiv, work, &lwork, &info);

		// dlacpy
		alloc_dense_node(n, Bstr);
		zlacpy("All", &n, &n, A_init, &n, Bstr->A, &n);

		free_arr(work);
		free_arr(A_init);
		free_arr(ipiv);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		Bstr->p = Astr->p;
		dtype *X11 = alloc_arr2<dtype>(n1 * n1); int ldx11 = n1;
		dtype *X22 = alloc_arr2<dtype>(n2 * n2); int ldx22 = n2;
		dtype *V = alloc_arr2<dtype>(n1 * Astr->p); int ldv = n1;
		dtype *B12 = alloc_arr2<dtype>(n1 * Bstr->p); int ldb12 = n1;
		dtype *Y = alloc_arr2<dtype>(Astr->p * Bstr->p); int ldy = Astr->p;
		cmnode *X11str, *X22str;

		// Inversion of A22 to X22
		SymCompRecInvStruct(n2, Astr->right, X22str, smallsize, eps, method);

		// Save X22 * U to B{2,1}
		Bstr->U = alloc_arr2<dtype>(n2 * Bstr->p);
		RecMultLStruct(n2, Bstr->p, X22str, Astr->U, n2, Bstr->U, n2, smallsize);

		// Compute Y = UT * X22 * U = | A[2,1]T * B{2,1} | = | (p x n2) x (n2 x p)  = (p x p) |
		zgemm("Trans", "No", &Astr->p, &Bstr->p, &n2, &alpha_one, Astr->U, &n2, Bstr->U, &n2, &beta_zero, Y, &ldy);

		// Update X11 = A11 - V * UT * X22 * U * VT = | A11 - V * Y * VT | = | (n1 x n1) - (n1 x p) * (p x p) * (p x n1) |
		Mat_Trans(Astr->p, n1, Astr->VT, Astr->p, V, ldv);
		SymCompUpdate2Struct(n1, Astr->p, Astr->left, alpha_mone, Y, ldy, V, ldv, X11str, smallsize, eps, method);

		// Inversion of X11 to B11
		SymCompRecInvStruct(n1, X11str, Bstr->left, smallsize, eps, method);

		// Fill B{1,2} as B12 = -B{1,1} * A{1,2} = -X11 * V = (n1 x n1) * (n1 x p) = (n1 x p)
		RecMultLStruct(n1, Bstr->p, Bstr->left, V, ldv, B12, ldb12, smallsize);
		mkl_zimatcopy('C', 'N', n1, Bstr->p, -1.0, B12, ldb12, ldb12);

		// B{1,2} = transpose(B12)
		Bstr->VT = alloc_arr2<dtype>(Bstr->p * n1);
		Mat_Trans(n1, Bstr->p, B12, ldb12, Bstr->VT, Bstr->p);

		// Y = -(A{1,2})' * B{1,2} = -VT * (-X11 * V) = - VT * B12 = (p x n1) * (n1 x p)
		zgemm("No", "No", &Astr->p, &Bstr->p, &n1, &alpha_mone, Astr->VT, &Astr->p, B12, &ldb12, &beta_zero, Y, &ldy);

		// Update X22 + (X22*U) * VT * X11 * V (UT * X22) = X22 + B21 * Y * B21T = (n2 x n2) + (n2 x p) * (p x p) * (p x n2)
		SymCompUpdate2Struct(n2, Bstr->p, X22str, alpha_one, Y, ldy, Bstr->U, n2, Bstr->right, smallsize, eps, method);

		FreeNodes(n1, X11str, smallsize);
		FreeNodes(n2, X22str, smallsize);
		free_arr(X11);
		free_arr(X22);
		free_arr(Y);
		free_arr(V);
		free_arr(B12);
	}
}
#if 1
void UnsymmCompRecInvStruct(int n, cumnode* Astr, cumnode* &Bstr, int smallsize, double eps, char *method)
{
	dtype alpha_one = 1.0;
	dtype alpha_mone = -1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;
	int info = 0;
	dtype wquery = 0;
	int lwork = -1;

	Bstr = (cumnode*)malloc(sizeof(cumnode));

	if (n <= smallsize)
	{
		int *ipiv = alloc_arr2<int>(n);
		dtype *A_init = alloc_arr2<dtype>(n * n);

		zlacpy("All", &n, &n, Astr->A21->A, &n, A_init, &n);

		// LU factorization of A
		zgetrf(&n, &n, A_init, &n, ipiv, &info);

		// space query
		zgetri(&n, A_init, &n, ipiv, &wquery, &lwork, &info);

		lwork = (int)wquery.real();
		dtype *work = alloc_arr2<dtype>(lwork);

		// inversion of A
		zgetri(&n, A_init, &n, ipiv, work, &lwork, &info);

		// dlacpy
		alloc_dense_unsymm_node(n, Bstr);
		zlacpy("All", &n, &n, A_init, &n, Bstr->A21->A, &n);

		free_arr(work);
		free_arr(A_init);
		free_arr(ipiv);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;


		Bstr->A12 = (cmnode*)malloc(sizeof(cmnode));
		Bstr->A21 = (cmnode*)malloc(sizeof(cmnode));

		int p2 = Bstr->A21->p = Astr->A21->p;
		int p1 = Bstr->A12->p = Astr->A12->p;

		dtype *X11 = alloc_arr2<dtype>(n1 * n1); int ldx11 = n1;
		dtype *X22 = alloc_arr2<dtype>(n2 * n2); int ldx22 = n2;
		dtype *Y = alloc_arr2<dtype>(p2 * p1);
		cumnode *X11str, *X22str;


		Bstr->A21->U = alloc_arr2<dtype>(n2 * p2);
		Bstr->A21->VT = alloc_arr2<dtype>(p2 * n1);

		Bstr->A12->U = alloc_arr2<dtype>(n1 * p1);
		Bstr->A12->VT = alloc_arr2<dtype>(p1 * n2);

		// Inversion of A22 to X22
		UnsymmCompRecInvStruct(n2, Astr->right, X22str, smallsize, eps, method);

		// Save X22 * U to B{2,1} U	
		UnsymmRecMultLStruct(n2, p2, X22str, Astr->A21->U, n2, Bstr->A21->U, n2, smallsize);

		// Save VT * X22 to -B{1,2} VT
		UnsymmRecMultRStruct(n2, p1, X22str, Astr->A12->VT, p1, Bstr->A12->VT, p1, smallsize);

		// Compute Y = VT * (X22 * U) = | (p1 x n2) x (n2 x p2)  = (p1 x p2) |
		zgemm("No", "No", &p1, &p2, &n2, &alpha_one, Astr->A12->VT, &p1, Bstr->A21->U, &n2, &beta_zero, Y, &p1);

		// Update X11 = A11 - U12 * V21T * X22 * U21 * V21T = | A11 - V * Y * VT | = | (n1 x n1) - (n1 x p1) * (p1 x p2) * (p2 x n1) |
		
		//здесь будут разные матрицы: A:= A - V1 * Xunsymm * V2		
		UnsymmCompUpdate3Struct(n1, p1, p2, Astr->left, alpha_mone, Y, p1, Astr->A12->U, n1, Astr->A21->VT, p2, X11str, smallsize, eps, method);
	
		// Inversion of X11 to B11
		UnsymmCompRecInvStruct(n1, X11str, Bstr->left, smallsize, eps, method);

		// Fill B{1,2} U as B{1,1} * A{1,2} U = X11 * U = (n1 x n1) * (n1 x p1) = (n1 x p1)
		UnsymmRecMultLStruct(n1, p1, Bstr->left, Astr->A12->U, n1, Bstr->A12->U, n1, smallsize);

		// Fill B{2,1} VT as A{2,1} VT * B11
		UnsymmRecMultRStruct(n1, p2, Bstr->left, Astr->A21->VT, p2, Bstr->A21->VT, p2, smallsize);

		// Y = (A21 VT * X11) * U12 = B21 VT * B12 = (p2 x n1) * (n1 x p1)
		zgemm("No", "No", &p2, &p1, &n1, &alpha_one, Bstr->A21->VT, &p2, Astr->A12->U, &n1, &beta_zero, Y, &p2);

		// B21 U = -B21 U
		mkl_zimatcopy('C', 'N', n2, p2, -1.0, Bstr->A21->U, n2, n2);

		// B21 VT = -B21 VT
		mkl_zimatcopy('C', 'N', p1, n2, -1.0, Bstr->A12->VT, p1, p1);
	
		// Update X22 + (-X22*U) * VT * X11 * U (-VT * X22) = X22 + B21 U * Y * B21 VT = (n2 x n2) + (n2 x p2) * (p2 x p1) * (p1 x n2)
		UnsymmCompUpdate3Struct(n2, p2, p1, X22str, alpha_one, Y, p2, Bstr->A21->U, n2, Bstr->A12->VT, p1, Bstr->right, smallsize, eps, method);

		FreeUnsymmNodes(n1, X11str, smallsize);
		FreeUnsymmNodes(n2, X22str, smallsize);
		free_arr(X11);
		free_arr(X22);
		free_arr(Y);
	}
}
#endif

void UnsymmCompRecInvLowerTriangStruct(int n, cumnode* Lstr, cumnode* &Bstr, int smallsize, double eps, char *method)
{
	dtype alpha_one = 1.0;
	dtype alpha_mone = -1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;
	int info = 0;
	dtype wquery = 0;
	int lwork = -1;

	Bstr = (cumnode*)malloc(sizeof(cumnode));

	if (n <= smallsize)
	{
		// dlacpy
		alloc_dense_unsymm_node(n, Bstr);
		Eye(n, Bstr->A21->A, n);

		ztrsm("Left", "Low", "No", "Unit", &n, &n, &alpha_one, Lstr->A21->A, &n, Bstr->A21->A, &n);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		Bstr->A21 = (cmnode*)malloc(sizeof(cmnode));

		int p2 = Bstr->A21->p = Lstr->A21->p;

		Bstr->A21->U = alloc_arr2<dtype>(n2 * p2);
		Bstr->A21->VT = alloc_arr2<dtype>(p2 * n1);

		// Inversion of L11 to B11
		UnsymmCompRecInvLowerTriangStruct(n1, Lstr->left, Bstr->left, smallsize, eps, method);

		// Inversion of L22 to B22
		UnsymmCompRecInvLowerTriangStruct(n2, Lstr->right, Bstr->right, smallsize, eps, method);

		// Fill B{2,1} U as L22 * A{2,1} U
		UnsymmRecMultLowerLStruct(n2, p2, Bstr->right, Lstr->A21->U, n2, Bstr->A21->U, n2, smallsize);

		// Fill B{2,1} VT as A{2,1} VT * L11
		UnsymmRecMultLowerRStruct(n1, p2, Bstr->left, Lstr->A21->VT, p2, Bstr->A21->VT, p2, smallsize);

		// B21 U = -B21 U
		mkl_zimatcopy('C', 'N', n2, p2, -1.0, Bstr->A21->U, n2, n2);
	}
}

void UnsymmCompRecInvUpperTriangStruct(int n, cumnode* Rstr, cumnode* &Bstr, int smallsize, double eps, char *method)
{
	dtype alpha_one = 1.0;
	dtype alpha_mone = -1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;
	int info = 0;
	dtype wquery = 0;
	int lwork = -1;

	Bstr = (cumnode*)malloc(sizeof(cumnode));

	if (n <= smallsize)
	{
		// dlacpy
		alloc_dense_unsymm_node(n, Bstr);
		Eye(n, Bstr->A21->A, n);

		ztrsm("Right", "Up", "No", "NonUnit", &n, &n, &alpha_one, Rstr->A21->A, &n, Bstr->A21->A, &n);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		Bstr->A12 = (cmnode*)malloc(sizeof(cmnode));

		int p1 = Bstr->A12->p = Rstr->A12->p;

		Bstr->A12->U = alloc_arr2<dtype>(n1 * p1);
		Bstr->A12->VT = alloc_arr2<dtype>(p1 * n2);

		// Inversion of R11 to B11
		UnsymmCompRecInvUpperTriangStruct(n1, Rstr->left, Bstr->left, smallsize, eps, method);

		// Inversion of R22 to B22
		UnsymmCompRecInvUpperTriangStruct(n2, Rstr->right, Bstr->right, smallsize, eps, method);

		// Fill B{1,2} U as U11 * A{1,2} U
		UnsymmRecMultUpperLStruct(n1, p1, Bstr->left, Rstr->A12->U, n1, Bstr->A12->U, n1, smallsize);

		// Fill B{1,2} VT as A{1,2} VT * U22
		UnsymmRecMultUpperRStruct(n2, p1, Bstr->right, Rstr->A12->VT, p1, Bstr->A12->VT, p1, smallsize);

		// B12 U = -B12 U
		mkl_zimatcopy('C', 'N', n1, p1, -1.0, Bstr->A12->U, n1, n1);
	}
}


void SymResRestoreStruct(int n, cmnode* H1str, dtype *H2 /*recovered*/, int ldh, int smallsize)
{
	dtype alpha = 1.0;
	dtype beta = 0.0;

	if (n <= smallsize)
	{
		zlacpy("All", &n, &n, H1str->A, &n, H2, &ldh);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0);
		int n1 = n - n2;

		// A21 = A21 * A12
		zgemm("Notrans", "Notrans", &n2, &n1, &H1str->p, &alpha, H1str->U, &n2, H1str->VT, &H1str->p, &beta, &H2[n1 + ldh * 0], &ldh);

		// A12 = A21*T = A12*T * A21*T
		zgemm("Trans", "Trans", &n1, &n2, &H1str->p, &alpha, H1str->VT, &H1str->p, H1str->U, &n2, &beta, &H2[0 + ldh * n1], &ldh);


		SymResRestoreStruct(n1, H1str->left, &H2[0 + ldh * 0], ldh, smallsize);
		SymResRestoreStruct(n2, H1str->right, &H2[n1 + ldh * n1], ldh, smallsize);
	}
}

void UnsymmResRestoreStruct(int n, cumnode* H1str, dtype *H2 /*recovered*/, int ldh, int smallsize)
{
	dtype alpha = 1.0;
	dtype beta = 0.0;

	if (n <= smallsize)
	{
		zlacpy("All", &n, &n, H1str->A21->A, &n, H2, &ldh);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0);
		int n1 = n - n2;

		// A21 = A21 U * A12 VT
		zgemm("No", "No", &n2, &n1, &H1str->A21->p, &alpha, H1str->A21->U, &n2, H1str->A21->VT, &H1str->A21->p, &beta, &H2[n1 + ldh * 0], &ldh);

		// A12 = A12 U * A 21 VT
		zgemm("No", "No", &n1, &n2, &H1str->A12->p, &alpha, H1str->A12->U, &n1, H1str->A12->VT, &H1str->A12->p, &beta, &H2[0 + ldh * n1], &ldh);


		UnsymmResRestoreStruct(n1, H1str->left, &H2[0 + ldh * 0], ldh, smallsize);
		UnsymmResRestoreStruct(n2, H1str->right, &H2[n1 + ldh * n1], ldh, smallsize);
	}
}

void LowRankToUnsymmHSS(int n, int r, dtype *U, int ldu, dtype *VT, int ldvt, cumnode *&Aout, int smallsize)
{
	Aout = (cumnode*)malloc(sizeof(cumnode));
	dtype done = 1.0;
	dtype zero = 0.0;

	if (n <= smallsize)
	{
		alloc_dense_unsymm_node(n, Aout);
		Aout->A12->A = (dtype*)malloc(n * n * sizeof(dtype));
		zgemm("no", "no", &n, &n, &r, &done, U, &ldu, VT, &ldvt, &zero, Aout->A21->A, &n);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0);
		int n1 = n - n2;

		Aout->A12 = (cmnode*)malloc(sizeof(cmnode));
		Aout->A21 = (cmnode*)malloc(sizeof(cmnode));

		Aout->A21->U = (dtype*)malloc(n2 * r * sizeof(dtype));
		Aout->A21->VT = (dtype*)malloc(r * n1 * sizeof(dtype));

		Aout->A12->U = (dtype*)malloc(n1 * r * sizeof(dtype));
		Aout->A12->VT = (dtype*)malloc(r * n2 * sizeof(dtype));

		zlacpy("all", &n2, &r, &U[n1 + ldu * 0], &ldu, Aout->A21->U, &n2);
		zlacpy("all", &n1, &r, &U[0 + ldu * 0], &ldu, Aout->A12->U, &n1);
	
		zlacpy("all", &r, &n1, &VT[0 + ldvt * 0], &ldvt, Aout->A21->VT, &r);
		zlacpy("all", &r, &n2, &VT[0 + ldvt * n1], &ldvt, Aout->A12->VT, &r);

		Aout->A21->p = r;
		Aout->A12->p = r;

		LowRankToUnsymmHSS(n1, r, U, ldu, VT, ldvt, Aout->left, smallsize);
		LowRankToUnsymmHSS(n2, r, &U[n1 + ldu * 0], ldu, &VT[0 + ldvt * n1], ldvt, Aout->right, smallsize);
	}
}


// Solver
#if 1
void Block3DSPDSolveFastStruct(size_m x, size_m y, dtype *D, int ldd, dtype *B, dtype *f, zcsr* Dcsr, double thresh, int smallsize, int ItRef, char *bench,
	/* output */ 	cmnode** &Gstr, dtype *x_sol, int &success, double &RelRes, int &itcount, double beta_eq)
{
#ifndef ONLINE
	if (D == NULL)
	{
		printf("D is Null - error\n");
		return;
	}
#endif

	int size = x.n * y.n;
	int n = x.n;
	double tt;
	double tt1;

	// set frequency
	dtype kwave_beta2;
	size_m z;	
	SetFrequency("NO_FFT", x, y, z, y.n / 2, kwave_beta2, beta_eq);
	
	int lwork = size + n;
	int levels = ceil(log2(ceil((double)n / smallsize))) + 1;
	int lwork2 = n * n / 2;
	int lwork3 = 2 * n * 1 * levels;
	dtype *work = alloc_arr<dtype>(lwork + lwork2 + lwork3);
	dtype *sound2D;
	double kww = 0;

	printf("Factorization of matrix...\n");
	tt = omp_get_wtime();

#ifndef ONLINE
	DirFactFastDiagStruct(x.n, y.n, z.n, D, ldd, B, Gstr, thresh, smallsize, bench);
#else
	DirFactFastDiagStructOnline(x, y, Gstr, B, sound2D, kww, beta_eq, work, lwork, thresh, smallsize);
#endif
	tt = omp_get_wtime() - tt;
	if (compare_str(7, bench, "print_time"))
	{
		printf("Total factorization time: %lf\n", tt);
	}

	printf("Solving of the system...\n");

	tt = omp_get_wtime();
	DirSolveFastDiagStruct(x.n, y.n, Gstr, B, f, x_sol, work, lwork, thresh, smallsize);
	tt = omp_get_wtime() - tt;

	if (compare_str(7, bench, "print_time"))
	{
		printf("Solving time: %lf\n", tt);
	}

	dtype *g = alloc_arr2<dtype>(size);
	dtype *x1 = alloc_arr2<dtype>(size);
	RelRes = 1;

	ResidCSR(x.n, y.n, Dcsr, x_sol, f, g, RelRes);

	printf("RelRes = %10.8lf\n", RelRes);
	//if (RelRes < thresh)
	if (1)
	{
		success = 1;
		itcount = 0;
	}
	else {
		int success = 0;
		if (ItRef > 0) {
			if (compare_str(7, bench, "print_time")) printf("Iterative refinement started\n");
			tt1 = omp_get_wtime();
			itcount = 0;
			while ((RelRes > thresh) && (itcount < ItRef))
			{
				system("pause");
				tt = omp_get_wtime();

				DirSolveFastDiagStruct(x.n, y.n, Gstr, B, g, x1, work, lwork, thresh, smallsize);

#pragma omp parallel for simd schedule(static)
				for (int i = 0; i < size; i++)
					x_sol[i] = x_sol[i] + x1[i];

				// начальное решение f сравниваем с решением A_x0 + A_x1 + A_x2
				ResidCSR(x.n, y.n, Dcsr, x_sol, f, g, RelRes);

				itcount = itcount + 1;
				tt = omp_get_wtime() - tt;
				if (compare_str(7, bench, "print_time")) printf("itcount = %d, RelRes = %lf, Time = %lf\n", itcount, RelRes, tt);
			}
			if ((RelRes < thresh) && (itcount < ItRef)) success = 1; // b

			tt1 = omp_get_wtime() - tt1;
			if (compare_str(7, bench, "print_time")) printf("Iterative refinement total time: %lf\n", tt1);
		}
	}

	free_arr(g);
	free_arr(x1);
	free_arr(work);
}
#endif

/* Функция вычисления разложения симметричной блочно-диагональной матрицы с использование сжатого формата.
Внедиагональные блоки предполагаются диагональными матрицами */
void DirFactFastDiagStructOnline(size_m x, size_m y, cmnode** &Gstr, dtype *B, dtype *sound2D, double kww, double beta_eq, dtype *work, int lwork,
	double eps, int smallsize)
{
	int n = x.n;
	int nbr = y.n; // size of D is equal to nbr blocks by n elements
	int size = n * nbr;
	double tt;

#ifdef DISPLAY
		printf("**********************************\n");
		printf("Timing DirFactFastDiagStructOnline\n");
		printf("**********************************\n");
#endif

	cmnode *DCstr;
	dtype *DD = work; int lddd = n;

	// Gen diagonal B
	GenerateSubdiagonalB(x, y, B);

	Clear(n, n, DD, lddd);
	tt = omp_get_wtime();
	GenerateDiagonal1DBlockHODLR(0, x, y, DD, lddd, sound2D, kww, beta_eq);
	SymRecCompressStruct(n, DD, lddd, DCstr, smallsize, eps, "SVD");
	tt = omp_get_wtime() - tt;
#ifdef DISPLAY
	printf("Compressing D(0) time: %lf\n", tt);
#endif
	Gstr = (cmnode**)malloc(nbr * sizeof(cmnode*));
	tt = omp_get_wtime();
	SymCompRecInvStruct(n, DCstr, Gstr[0], smallsize, eps, "SVD");
	tt = omp_get_wtime() - tt;
#ifdef DISPLAY
	printf("Computing G(1) time: %lf\n", tt);
#endif

	//printf("Block %d. ", 0);
	//	Test_RankEqual(DCstr, Gstr[0]);

	FreeNodes(n, DCstr, smallsize);

	for (int k = 1; k < nbr; k++)
	{
		cmnode *DCstr, *TDstr, *TD1str;

		Clear(n, n, DD, lddd);

		tt = omp_get_wtime();
		GenerateDiagonal1DBlockHODLR(k, x, y, DD, lddd, sound2D, kww, beta_eq);
		SymRecCompressStruct(n, DD, lddd, DCstr, smallsize, eps, "SVD");
		tt = omp_get_wtime() - tt;
#ifdef DISPLAY
		printf("Compressing D(%d) time: %lf\n", k, tt);
#endif

		tt = omp_get_wtime();
		CopyStruct(n, Gstr[k - 1], TD1str, smallsize);

		DiagMultStruct(n, TD1str, &B[ind(k - 1, n)], smallsize);
		tt = omp_get_wtime() - tt;
#ifdef DISPLAY
		printf("Mult D(%d) time: %lf\n", k, tt);
#endif

		tt = omp_get_wtime();
		AddStruct(n, 1.0, DCstr, -1.0, TD1str, TDstr, smallsize, eps, "SVD");
		tt = omp_get_wtime() - tt;
#ifdef DISPLAY
		printf("Add %d time: %lf\n", k, tt);
#endif

		tt = omp_get_wtime();
		SymCompRecInvStruct(n, TDstr, Gstr[k], smallsize, eps, "SVD");
		tt = omp_get_wtime() - tt;
#ifdef DISPLAY
		printf("Computing G(%d) time: %lf\n", k, tt);
#endif

		tt = omp_get_wtime();
		FreeNodes(n, DCstr, smallsize);
		FreeNodes(n, TDstr, smallsize);
		FreeNodes(n, TD1str, smallsize);
		tt = omp_get_wtime() - tt;
#ifdef DISPLAY
		printf("Memory deallocation G(%d) time: %lf\n\n", k, tt);
#endif
	}

#ifdef DISPLAY
		printf("****************************\n");
		printf("End of DirFactFastDiag\n");
		printf("****************************\n");
#endif
}

void DirSolveFastDiagStruct(int n1, int n2, cmnode** Gstr, dtype* B, const dtype* f, dtype* x, dtype* work, int lwork, double eps, int smallsize)
{
#if 1
	int n = n1;
	int nbr = n2;
	int size2D = n * nbr;
	int ione = 1;
	dtype fone = { 1.0, 0.0 };
	dtype mone = { -1.0, 0.0 };
	dtype zero = { 0.0, 0.0 };

	// lwork = size + n
	dtype* tb = work;
	dtype* y = &work[size2D];

	zcopy(&n, f, &ione, tb, &ione);

	int lwork1 = n * n / 2;
	//int levels = ceil(log2(ceil((double)n / smallsize))) + 1;
	//int lwork2 = 2 * n * 1 * levels;

	for (int k = 1; k < nbr; k++)
	{
		RecMultLStructWork2(n, 1, Gstr[k - 1], &tb[ind(k - 1, n)], size2D, zero, y, n, &work[size2D + n], lwork1, smallsize);
		//RecMultLStructWork(n, 1, Gstr[k - 1], &tb[ind(k - 1, n)], size2D, y, n, &work[size2D + n], lwork1, &work[size2D + n + lwork1], lwork2, smallsize);
		//RecMultLStruct(n, 1, Gstr[k - 1], &tb[ind(k - 1, n)], size2D, y, n, smallsize);
		DenseDiagMult(n, &B[ind(k - 1, n)], y, y);

		OpTwoMatrices(n, 1, &f[ind(k, n)], y, &tb[ind(k, n)], n, '-');
	}

	RecMultLStructWork2(n, 1, Gstr[nbr - 1], &tb[ind(nbr - 1, n)], size2D, zero, &x[ind(nbr - 1, n)], size2D, &work[size2D + n], lwork1, smallsize);
	//RecMultLStructWork(n, 1, Gstr[nbr - 1], &tb[ind(nbr - 1, n)], size2D, &x[ind(nbr - 1, n)], size2D, &work[size2D + n], lwork1, &work[size2D + n + lwork1], lwork2, smallsize);
	//RecMultLStruct(n, 1, Gstr[nbr - 1], &tb[ind(nbr - 1, n)], size2D, &x[ind(nbr - 1, n)], size2D, smallsize);

	for (int k = nbr - 2; k >= 0; k--)
	{
		DenseDiagMult(n, &B[ind(k, n)], &x[ind(k + 1, n)], y);

		zaxpby(&n, &fone, &tb[ind(k, n)], &ione, &mone, y, &ione);

		RecMultLStructWork2(n, 1, Gstr[k], y, n, zero, &x[ind(k, n)], size2D, &work[size2D + n], lwork1, smallsize);
		//RecMultLStructWork(n, 1, Gstr[k], y, n, &x[ind(k, n)], size2D, &work[size2D + n], lwork1, &work[size2D + n + lwork1], lwork2, smallsize);
		//RecMultLStruct(n, 1, Gstr[k], y, n, &x[ind(k, n)], size2D, smallsize);
	}
#endif
}

void alloc_dense_node(int n, cmnode* &Cstr)
{
	Cstr->A = alloc_arr2<dtype>(n * n);
	Cstr->n1 = n;
	Cstr->p = -1;
	Cstr->n2 = n;
	Cstr->U = NULL;
	Cstr->VT = NULL;
	Cstr->left = NULL;
	Cstr->right = NULL;
}

void alloc_dense_simple_node(int n, cmnode* &Cstr)
{
	Cstr->A = NULL;
	Cstr->n1 = 0;
	Cstr->p = -2;
	Cstr->n2 = 0;
	Cstr->U = NULL;
	Cstr->VT = NULL;
	Cstr->left = NULL;
	Cstr->right = NULL;
}

void alloc_dense_unsymm_node(int n, cumnode* &Cstr)
{
	Cstr->A21 = (cmnode*)malloc(sizeof(cmnode));
	Cstr->A12 = (cmnode*)malloc(sizeof(cmnode));
	Cstr->left = NULL;
	Cstr->right = NULL;
	alloc_dense_node(n, Cstr->A21);
	alloc_dense_simple_node(n, Cstr->A12);
}

void FreeNodes(int n, cmnode* &Astr, int smallsize)
{
	if (n <= smallsize)
	{
		free_arr(Astr->A);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		free_arr(Astr->U);
		free_arr(Astr->VT);

		FreeNodes(n1, Astr->left, smallsize);
		FreeNodes(n2, Astr->right, smallsize);
	}

	free_arr(Astr);
}

void FreeUnsymmNodes(int n, cumnode* &Astr, int smallsize)
{
	if (n <= smallsize)
	{
		free_arr(Astr->A21->A);
		free_arr(Astr->A21);
		free_arr(Astr->A12);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		free_arr(Astr->A21->U);
		free_arr(Astr->A21->VT);

		free_arr(Astr->A12->U);
		free_arr(Astr->A12->VT);

		FreeUnsymmNodes(n1, Astr->left, smallsize);
		FreeUnsymmNodes(n2, Astr->right, smallsize);
	}

	free_arr(Astr);
}

void CopyStruct(int n, cmnode* Gstr, cmnode* &TD1str, int smallsize)
{
	TD1str = (cmnode*)malloc(sizeof(cmnode));
	if (n <= smallsize)
	{
		alloc_dense_node(n, TD1str);
		zlacpy("All", &n, &n, Gstr->A, &n, TD1str->A, &n);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		TD1str->p = Gstr->p;
		TD1str->U = alloc_arr2<dtype>(n2 * TD1str->p);
		TD1str->VT = alloc_arr2<dtype>(TD1str->p * n1);
		zlacpy("All", &n2, &TD1str->p, Gstr->U, &n2, TD1str->U, &n2);
		zlacpy("All", &TD1str->p, &n1, Gstr->VT, &TD1str->p, TD1str->VT, &TD1str->p);

		CopyStruct(n1, Gstr->left, TD1str->left, smallsize);
		CopyStruct(n2, Gstr->right, TD1str->right, smallsize);
	}
}

void CopyUnsymmStruct(int n, cumnode* Astr, cumnode* &Bstr, int smallsize)
{
	Bstr = (cumnode*)malloc(sizeof(cumnode));
	if (n <= smallsize)
	{
		alloc_dense_unsymm_node(n, Bstr);
		zlacpy("All", &n, &n, Astr->A21->A, &n, Bstr->A21->A, &n);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		Bstr->A21 = (cmnode*)malloc(sizeof(cmnode));
		Bstr->A12 = (cmnode*)malloc(sizeof(cmnode));

		Bstr->A21->p = Astr->A21->p;
		Bstr->A12->p = Astr->A12->p;

		Bstr->A21->U = alloc_arr2<dtype>(n2 * Astr->A21->p);
		Bstr->A12->U = alloc_arr2<dtype>(n1 * Astr->A12->p);
		Bstr->A21->VT = alloc_arr2<dtype>(Bstr->A21->p * n1);
		Bstr->A12->VT = alloc_arr2<dtype>(Bstr->A12->p * n2);

		zlacpy("All", &n2, &Bstr->A21->p, Astr->A21->U, &n2, Bstr->A21->U, &n2);
		zlacpy("All", &n1, &Bstr->A12->p, Astr->A12->U, &n1, Bstr->A12->U, &n1);

		zlacpy("All", &Bstr->A21->p, &n1, Astr->A21->VT, &Astr->A21->p, Bstr->A21->VT, &Bstr->A21->p);
		zlacpy("All", &Bstr->A12->p, &n2, Astr->A12->VT, &Astr->A12->p, Bstr->A12->VT, &Bstr->A12->p);

		CopyUnsymmStruct(n1, Astr->left, Bstr->left, smallsize);
		CopyUnsymmStruct(n2, Astr->right, Bstr->right, smallsize);
	}
}

void UnsymmCopyStruct(int n, cumnode* Astr, cumnode* Bstr, int smallsize)
{
	// B is already allocated
	// ranks of A are equal to ranks of B

	if (n <= smallsize)
	{
		zlacpy("All", &n, &n, Astr->A21->A, &n, Bstr->A21->A, &n);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		zlacpy("All", &n2, &Bstr->A21->p, Astr->A21->U, &n2, Bstr->A21->U, &n2);
		zlacpy("All", &n1, &Bstr->A12->p, Astr->A12->U, &n1, Bstr->A12->U, &n1);

		zlacpy("All", &Bstr->A21->p, &n1, Astr->A21->VT, &Astr->A21->p, Bstr->A21->VT, &Bstr->A21->p);
		zlacpy("All", &Bstr->A12->p, &n2, Astr->A12->VT, &Astr->A12->p, Bstr->A12->VT, &Bstr->A12->p);

		UnsymmCopyStruct(n1, Astr->left, Bstr->left, smallsize);
		UnsymmCopyStruct(n2, Astr->right, Bstr->right, smallsize);
	}
}

void UnsymmClearStruct(int n, cumnode* Astr, int smallsize)
{
	// B is already allocated
	// ranks of A are equal to ranks of B

	if (n <= smallsize)
	{
		Clear(n, n, Astr->A21->A, n);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		Clear(n2, Astr->A21->p, Astr->A21->U, n2);
		Clear(n1, Astr->A12->p, Astr->A12->U, n1);

		Clear(Astr->A21->p, n1, Astr->A21->VT, Astr->A21->p);
		Clear(Astr->A12->p, n2, Astr->A12->VT, Astr->A12->p);

		UnsymmClearStruct(n1, Astr->left, smallsize);
		UnsymmClearStruct(n2, Astr->right, smallsize);
	}
}


#if 0

//#define COL_ADD
// Функция вычисления линейной комбинации двух сжатых матриц
#ifdef COL_ADD
void AddStruct(int n, double alpha, mnode* Astr, double beta, mnode* Bstr, mnode* &Cstr, int smallsize, double eps, char *method)
{
	double alpha_loc = 1.0;
	double beta_loc = 0.0;
	
	Cstr = (mnode*)malloc(sizeof(mnode));

	// n - order of A, B and C
	if (n <= smallsize)
	{
		alloc_dense_node(n, Cstr);
		mkl_domatadd('C', 'N', 'N', n, n, alpha, Astr->A, n, beta, Bstr->A, n, Cstr->A, n);
		//Add_dense(n, n, alpha, A, lda, beta, B, ldb, C, ldc);
	}
	else
	{
		int p1 = 0, p2 = 0;
		int n2 = (int)ceil(n / 2.0); // округление в большую сторону
		int n1 = n - n2;

		int n1_dbl = Astr->p + Bstr->p;

		double *Y21 = alloc_arr(n2 * n1_dbl); int ldy21 = n2;
		double *Y12 = alloc_arr(n1_dbl * n1); int ldy12 = n1_dbl;

		double *V21 = alloc_arr(n2 * n1_dbl);
		int ldv21 = n2;

		double *V12 = alloc_arr(n1_dbl * n1);
		int ldv12 = n1_dbl;


		double *AU = alloc_arr(n2 * Astr->p); int ldau = n2;
		double *BU = alloc_arr(n2 * Bstr->p); int ldbu = n2;

		dlacpy("All", &n2, &Astr->p, Astr->U, &n2, AU, &ldau);
		dlacpy("All", &n2, &Bstr->p, Bstr->U, &n2, BU, &ldbu);

		mkl_dimatcopy('C', 'N', n2, Astr->p, alpha, AU, n2, n2);
		mkl_dimatcopy('C', 'N', n2, Bstr->p, beta, BU, n2, n2);
		//Add_dense(n2, n1, alpha, &A[n1 + lda * 0], lda, 0.0, B, ldb, &A[n1 + lda * 0], lda);
		//Add_dense(n2, n1, beta, &B[n1 + ldb * 0], ldb, 0.0, B, ldb, &B[n1 + ldb * 0], ldb);

		// Y21 = [alpha*A{2,1} beta*B{2,1}];
		dlacpy("All", &n2, &Astr->p, AU, &n2, &Y21[0 + ldy21 * 0], &ldy21);
		dlacpy("All", &n2, &Bstr->p, BU, &n2, &Y21[0 + ldy21 * Astr->p], &ldy21);

		// Y12 = [A{1,2}; B{1,2}];
		dlacpy("All", &Astr->p, &n1, Astr->VT, &Astr->p, &Y12[0 + ldy12 * 0], &ldy12);
		dlacpy("All", &Bstr->p, &n1, Bstr->VT, &Bstr->p, &Y12[Astr->p + ldy12 * 0], &ldy12);

		// произведение Y21 и Y12 - это матрица n2 x n1
		//LowRankApprox(n2, n1_dbl, Y21, ldy21, V21, ldv21, p1, eps, "SVD"); // перезапись Y21
		//LowRankApprox(n1_dbl, n1, Y12, ldy12, V12, ldv12, p2, eps, "SVD");  // перезапись Y12

		mnode* Y21str = (mnode*)malloc(sizeof(mnode));
		LowRankApproxStruct(n2, n1_dbl, Y21, ldy21, Y21str, eps, "SVD");
		p1 = Y21str->p;
		dlacpy("All", &n2, &Y21str->p, Y21str->U, &n2, Y21, &ldy21);
		dlacpy("All", &Y21str->p, &n1_dbl, Y21str->VT, &Y21str->p, V21, &ldv21);

		mnode* Y12str = (mnode*)malloc(sizeof(mnode));
		LowRankApproxStruct(n1_dbl, n1, Y12, ldy12, Y12str, eps, "SVD");
		p2 = Y12str->p;
		dlacpy("All", &n1_dbl, &Y12str->p, Y12str->U, &n1_dbl, Y12, &ldy12);
		dlacpy("All", &Y12str->p, &n1, Y12str->VT, &Y12str->p, V12, &ldv12);

		// Y = V21'*V12;
		double *Y = alloc_arr(p1 * p2);
		dgemm("No", "No", &p1, &p2, &n1_dbl, &alpha_loc, V21, &ldv21, Y12, &ldy12, &beta_loc, Y, &p1); // mn, mn

		// C{2,1} = U21*Y;   
		Cstr->U = alloc_arr(n2 * p2);
		dgemm("No", "No", &n2, &p2, &p1, &alpha_loc, Y21, &ldy21, Y, &p1, &beta_loc, Cstr->U, &n2); // mn

		// C{1,2} = U12';
		Cstr->VT = alloc_arr(p2 * n1);
		dlacpy("All", &p2, &n1, V12, &ldv12, Cstr->VT, &p2); // n1, n2
		Cstr->p = p2;

		AddStruct(n1, alpha, Astr->left, beta, Bstr->left, Cstr->left, smallsize, eps, method);
		AddStruct(n2, alpha, Astr->right, beta, Bstr->right, Cstr->right, smallsize, eps, method);

		free_arr(&Y21);
		free_arr(&Y12);
		free_arr(&V21);
		free_arr(&V12);
		free_arr(&Y);
	}

}
#else
void AddStruct(int n, double alpha, mnode* Astr, double beta, mnode* Bstr, mnode* &Cstr, int smallsize, double eps, char *method)
{
	double alpha_loc = 1.0;
	double beta_loc = 0.0;

	Cstr = (mnode*)malloc(sizeof(mnode));

	// n - order of A, B and C
	if (n <= smallsize)
	{
		alloc_dense_node(n, Cstr);
		mkl_domatadd('C', 'N', 'N', n, n, alpha, Astr->A, n, beta, Bstr->A, n, Cstr->A, n);
		//Add_dense(n, n, alpha, A, lda, beta, B, ldb, C, ldc);
	}
	else
	{
		int p1 = 0, p2 = 0;
		int n2 = (int)ceil(n / 2.0); // округление в большую сторону
		int n1 = n - n2;
		int n1_dbl = Astr->p + Bstr->p;

		double *Y21 = alloc_arr(n2 * n1_dbl); int ldy21 = n2;
		double *Y12 = alloc_arr(n1 * n1_dbl); int ldy12 = n1;

		double *V21 = alloc_arr(n2 * n1_dbl); int ldv21 = n2;
		double *V12 = alloc_arr(n1 * n1_dbl); int ldv12 = n1;

		double *AU = alloc_arr(n2 * Astr->p); int ldau = n2;
		double *BU = alloc_arr(n2 * Bstr->p); int ldbu = n2;

		double *AV = alloc_arr(n1 * Astr->p); int ldav = n1;
		double *BV = alloc_arr(n1 * Bstr->p); int ldbv = n1;

		// Filling AU and BU - workspaces
		dlacpy("All", &n2, &Astr->p, Astr->U, &n2, AU, &ldau);
		dlacpy("All", &n2, &Bstr->p, Bstr->U, &n2, BU, &ldbu);

		// Filling AV and BV - workspaces
		Mat_Trans(Astr->p, n1, Astr->VT, Astr->p, AV, ldav);
		Mat_Trans(Bstr->p, n1, Bstr->VT, Bstr->p, BV, ldbv);

		// Multiplying AU = alpha * AU and BU = beta * BU
		mkl_dimatcopy('C', 'N', n2, Astr->p, alpha, AU, n2, n2);
		mkl_dimatcopy('C', 'N', n2, Bstr->p, beta, BU, n2, n2);
		//Add_dense(n2, n1, alpha, &A[n1 + lda * 0], lda, 0.0, B, ldb, &A[n1 + lda * 0], lda);
		//Add_dense(n2, n1, beta, &B[n1 + ldb * 0], ldb, 0.0, B, ldb, &B[n1 + ldb * 0], ldb);

		// Y21 = [alpha*A{2,1} beta*B{2,1}];
		dlacpy("All", &n2, &Astr->p, AU, &n2, &Y21[0 + ldy21 * 0], &ldy21);
		dlacpy("All", &n2, &Bstr->p, BU, &n2, &Y21[0 + ldy21 * Astr->p], &ldy21);

		// Y12 = [A{1,2}; B{1,2}];
		dlacpy("All", &n1, &Astr->p, AV, &ldav, &Y12[0 + ldy12 * 0], &ldy12);
		dlacpy("All", &n1, &Bstr->p, BV, &ldbv, &Y12[0 + ldy12 * Astr->p], &ldy12);

		// произведение Y21 и Y12 - это матрица n2 x n1
		//LowRankApprox(n2, n1_dbl, Y21, ldy21, V21, ldv21, p1, eps, "SVD"); // перезапись Y21
		//LowRankApprox(n1_dbl, n1, Y12, ldy12, V12, ldv12, p2, eps, "SVD");  // перезапись Y12

		mnode* Y21str = (mnode*)malloc(sizeof(mnode));
		mnode* Y12str = (mnode*)malloc(sizeof(mnode));
		LowRankApproxStruct(n2, n1_dbl, Y21, ldy21, Y21str, eps, "SVD");
		LowRankApproxStruct(n1, n1_dbl, Y12, ldy12, Y12str, eps, "SVD");
	
		dlacpy("All", &n2, &Y21str->p, Y21str->U, &n2, Y21, &ldy21);
		dlacpy("All", &Y21str->p, &n1_dbl, Y21str->VT, &Y21str->p, V21, &ldv21);

		dlacpy("All", &n1, &Y12str->p, Y12str->U, &n1, Y12, &ldy12);
		dlacpy("All", &Y12str->p, &n1_dbl, Y12str->VT, &Y12str->p, V12, &ldv12);

		p1 = Y21str->p;
		p2 = Y12str->p;

		// Y = V21'*V12;
		double *Y = alloc_arr(p1 * p2);
		dgemm("No", "Trans", &p1, &p2, &n1_dbl, &alpha_loc, V21, &ldv21, V12, &ldv12, &beta_loc, Y, &p1);

		// C{2,1} = U21*Y;   
		Cstr->U = alloc_arr(n2 * p2);
		dgemm("No", "No", &n2, &p2, &p1, &alpha_loc, Y21, &ldy21, Y, &p1, &beta_loc, Cstr->U, &n2); // mn

		// C{1,2} = U12';
		double *Y12_tr = alloc_arr(p2 * n1);
		Mat_Trans(n1, p2, Y12, ldy12, Y12_tr, p2);

		Cstr->VT = alloc_arr(p2 * n1);  Cstr->p = p2;
		dlacpy("All", &p2, &n1, Y12_tr, &p2, Cstr->VT, &p2);

		AddStruct(n1, alpha, Astr->left, beta, Bstr->left, Cstr->left, smallsize, eps, method);
		AddStruct(n2, alpha, Astr->right, beta, Bstr->right, Cstr->right, smallsize, eps, method);


		free_arr(&Y21str->U);
		free_arr(&Y21str->VT);
		free_arr(&Y12str->U);
		free_arr(&Y12str->VT);
		free_arr(&Y21);
		free_arr(&Y12);
		free_arr(&V21);
		free_arr(&V12);
		free_arr(&AU);
		free_arr(&BU);
		free_arr(&AV);
		free_arr(&BV);
		free_arr(&Y);
		free_arr(&Y12_tr);
		free(Y21str);
		free(Y12str);
	}

}
#endif

#ifdef COL_UPDATE
/* Функция вычисления симметричного малорангового дополнения A:= A + alpha * V * Y * V'
A - симметрическая сжатая (n x n)
Y - плотная симметричная размера k x k, k << n , V - плотная прямоугольная n x k
(n x n) = (n x n) + (n x k) * (k x k) * (k * n) */
void SymCompUpdate2Struct(int n, int k, mnode* Astr, double alpha, double *Y, int ldy, double *V, int ldv, mnode* &Bstr, int smallsize, double eps, char* method)
{
	double alpha_one = 1.0;
	double beta_zero = 0.0;
	double beta_one = 1.0;
	int p1 = 0, p2 = 0;

	if (fabs(alpha) < eps)
	{
		CopyStruct(n, Astr, Bstr, smallsize);
		return;
	}

	Bstr = (mnode*)malloc(sizeof(mnode));

	if (n <= smallsize)
	{
		// X = X + alpha * V * Y * VT

		// C = V * Y
		double *C = alloc_arr(n * k); int ldc = n;
		dsymm("Right", "Up", &n, &k, &alpha_one, Y, &ldy, V, &ldv, &beta_zero, C, &ldc);

		// Copy Astr->A to A_init
		double *A_init = alloc_arr(n * n); int lda = n;
		dlacpy("All", &n, &n, Astr->A, &lda, A_init, &lda);

		// X = X + alpha * C * Vt
		dgemm("No", "Trans", &n, &n, &k, &alpha, C, &ldc, V, &ldv, &beta_one, A_init, &lda);

		// B = A
		alloc_dense_node(n, Bstr);
		dlacpy("All", &n, &n, A_init, &lda, Bstr->A, &lda);

		free_arr(&C);
		free_arr(&A_init);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		int nk = Astr->p + k;
		// for this division n2 > n1 we can store a low memory

		double *Y12 = alloc_arr(nk * n1); int ldy12 = nk;
		double *Y21 = alloc_arr(n2 * nk); int ldy21 = n2;

		double *V_uptr = alloc_arr(k * n1); int ldvuptr = k;
		double *VY = alloc_arr(n2 * k); int ldvy = n2;

		double *V12 = alloc_arr(nk * n1); int ldv12 = nk;
		double *V21 = alloc_arr(n2 * nk); int ldv21 = n2;

		dgemm("No", "No", &n2, &k, &k, &alpha, &V[n1 + ldv * 0], &ldv, Y, &ldy, &beta_zero, VY, &ldvy);

		// Y21 = [A{2,1} alpha*V(m:n,:)*Y];
		dlacpy("All", &n2, &Astr->p, Astr->U, &n2, &Y21[0 + ldy21 * 0], &ldy21);
		dlacpy("All", &n2, &k, VY, &ldvy, &Y21[0 + ldy21 * Astr->p], &ldy21);

		//mkl_domatcopy('C', 'T', 1.0, n1, k, &V[0 + ldv * 0], ldv, V_uptr, ldvuptr);

		// Y12 = [A{1,2} V(1:n1,:)];
		Mat_Trans(n1, k, &V[0 + ldv * 0], ldv, V_uptr, ldvuptr);
		dlacpy("All", &Astr->p, &n1, Astr->VT, &Astr->p, &Y12[0 + ldy12 * 0], &ldy12);
		dlacpy("All", &k, &n1, V_uptr, &ldvuptr, &Y12[Astr->p + ldy12 * 0], &ldy12);

		mnode* Y21str = (mnode*)malloc(sizeof(mnode));
		mnode* Y12str = (mnode*)malloc(sizeof(mnode));
		// LowRankApprox(n2, nk, Y21, ldy21, V21, ldv21, p1, eps, "SVD");
		// LowRankApprox(n1, nk, Y12, ldy12, V12, ldv12, p2, eps, "SVD");

		// [U21,V21] = LowRankApprox (Y21, eps, method);
		LowRankApproxStruct(n2, nk, Y21, ldy21, Y21str, eps, "SVD");

		// [U12, V12] = LowRankApprox(Y12, eps, method);
		LowRankApproxStruct(nk, n1, Y12, ldy12, Y12str, eps, "SVD");
	
		dlacpy("All", &n2, &Y21str->p, Y21str->U, &n2, Y21, &ldy21);
		dlacpy("All", &Y21str->p, &nk, Y21str->VT, &Y21str->p, V21, &ldv21);

		p1 = Y21str->p;
		p2 = Y12str->p;
		dlacpy("All", &nk, &Y12str->p, Y12str->U, &nk, Y12, &ldy12);
		dlacpy("All", &Y12str->p, &n1, Y12str->VT, &Y12str->p, V12, &ldv12);
		Bstr->p = p2;

		// V21 * Y12
		double *VV = alloc_arr(p1 * p2); int ldvv = p1;
		dgemm("No", "No", &p1, &p2, &nk, &alpha_one, V21, &ldv21, Y12, &ldy12, &beta_zero, VV, &ldvv);
	
		// B{2,1} = U21*(V21'*V12);
		Bstr->U = alloc_arr(n2 * p2);
		dgemm("No", "No", &n2, &p2, &p1, &alpha_one, Y21, &ldy21, VV, &ldvv, &beta_zero, Bstr->U, &n2);
	
		// B{1,2} = U12;
		Bstr->VT = alloc_arr(p2 * n1);
		dlacpy("All", &p2, &n1, V12, &ldv12, Bstr->VT, &p2);
	
		// B{1,1} = SymCompUpdate2 (A{1,1}, Y, V(1:n1,:), alpha, eps, method);
		SymCompUpdate2Struct(n1, k, Astr->left, alpha, Y, ldy, &V[0 + ldv * 0], ldv, Bstr->left, smallsize, eps, method);

		// B{2,2} = SymCompUpdate2 (A{2,2}, Y, V(m:n ,:), alpha, eps, method);
		SymCompUpdate2Struct(n2, k, Astr->right, alpha, Y, ldy, &V[n1 + ldv * 0], ldv, Bstr->right, smallsize, eps, method);

		free_arr(&Y21);
		free_arr(&Y12);
		free_arr(&V21);
		free_arr(&V12);
		free_arr(&VY);
		//free_arr(&V_uptr);
		free_arr(&VV);
	}
}

#else
void SymCompUpdate2Struct(int n, int k, mnode* Astr, double alpha, double *Y, int ldy, double *V, int ldv, mnode* &Bstr, int smallsize, double eps, char* method)
{
	double alpha_one = 1.0;
	double beta_zero = 0.0;
	double beta_one = 1.0;
	int p1 = 0, p2 = 0;

	if (fabs(alpha) < eps)
	{
		CopyStruct(n, Astr, Bstr, smallsize);
		return;
	}

	Bstr = (mnode*)malloc(sizeof(mnode));

	if (n <= smallsize)
	{
		// X = X + alpha * V * Y * VT

		// C = V * Y
		double *C = alloc_arr(n * k); int ldc = n;
		dsymm("Right", "Up", &n, &k, &alpha_one, Y, &ldy, V, &ldv, &beta_zero, C, &ldc);

		// Copy Astr->A to A_init
		double *A_init = alloc_arr(n * n); int lda = n;
		dlacpy("All", &n, &n, Astr->A, &lda, A_init, &lda);

		// X = X + alpha * C * Vt
		dgemm("No", "Trans", &n, &n, &k, &alpha, C, &ldc, V, &ldv, &beta_one, A_init, &lda);

		// B = A
		alloc_dense_node(n, Bstr);
		dlacpy("All", &n, &n, A_init, &lda, Bstr->A, &lda);

		free_arr(&C);
		free_arr(&A_init);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		int nk = Astr->p + k;
		// for this division n2 > n1 we can store a low memory

		double *Y12 = alloc_arr(n1 * nk); int ldy12 = n1;
		double *Y21 = alloc_arr(n2 * nk); int ldy21 = n2;

		double *V_up = alloc_arr(n1 * k); int ldvup = n1;
		double *A12 = alloc_arr(n1 * Astr->p); int lda12 = n1;

		double *VY = alloc_arr(n2 * k); int ldvy = n2;

		double *V12 = alloc_arr(n1 * nk); int ldv12 = n1;
		double *V21 = alloc_arr(n2 * nk); int ldv21 = n2;

		dgemm("No", "No", &n2, &k, &k, &alpha, &V[n1 + ldv * 0], &ldv, Y, &ldy, &beta_zero, VY, &ldvy);

		// Y21 = [A{2,1} alpha*V(m:n,:)*Y];
		dlacpy("All", &n2, &Astr->p, Astr->U, &n2, &Y21[0 + ldy21 * 0], &ldy21);
		dlacpy("All", &n2, &k, VY, &ldvy, &Y21[0 + ldy21 * Astr->p], &ldy21);

		//mkl_domatcopy('C', 'T', 1.0, n1, k, &V[0 + ldv * 0], ldv, V_uptr, ldvuptr);

		// Y12 = [A{1,2} V(1:n1,:)];
		dlacpy("All", &n1, &k, &V[0 + ldv * 0], &ldv, V_up, &ldvup);
		Mat_Trans(Astr->p, n1, Astr->VT, Astr->p, A12, lda12);
		dlacpy("All", &n1, &Astr->p, A12, &lda12, &Y12[0 + ldy12 * 0], &ldy12);
		dlacpy("All", &n1, &k, V_up, &ldvup, &Y12[0 + ldy12 * Astr->p], &ldy12);

		//	LowRankApprox(n2, nk, Y21, ldy21, V21, ldv21, p1, eps, "SVD");
		//	LowRankApprox(n1, nk, Y12, ldy12, V12, ldv12, p2, eps, "SVD");

		mnode* Y21str = (mnode*)malloc(sizeof(mnode));
		mnode* Y12str = (mnode*)malloc(sizeof(mnode));

		// [U21,V21] = LowRankApprox (Y21, eps, method);
		LowRankApproxStruct(n2, nk, Y21, ldy21, Y21str, eps, "SVD");

		// [U12, V12] = LowRankApprox(Y12, eps, method);
		LowRankApproxStruct(n1, nk, Y12, ldy12, Y12str, eps, "SVD");
	
		dlacpy("All", &n2, &Y21str->p, Y21str->U, &n2, Y21, &ldy21);
		dlacpy("All", &Y21str->p, &nk, Y21str->VT, &Y21str->p, V21, &ldv21);

		dlacpy("All", &n1, &Y12str->p, Y12str->U, &n1, Y12, &ldy12);
		dlacpy("All", &Y12str->p, &nk, Y12str->VT, &Y12str->p, V12, &ldv12);

		p1 = Y21str->p;
		p2 = Y12str->p;
		Bstr->p = p2;

		// V21 * Y12
		double *VV = alloc_arr(p1 * p2);
		double *V_tr = alloc_arr(nk * p2);
		Mat_Trans(p2, nk, V12, ldv12, V_tr, nk);
		dgemm("No", "No", &p1, &p2, &nk, &alpha_one, V21, &ldv21, V_tr, &nk, &beta_zero, VV, &p1);

		// B{2,1} = U21*(V21'*V12);
		Bstr->U = alloc_arr(n2 * p2);
		dgemm("No", "No", &n2, &p2, &p1, &alpha_one, Y21, &ldy21, VV, &p1, &beta_zero, Bstr->U, &n2);

		// B{1,2} = U12;
		Bstr->VT = alloc_arr(p2 * n1);
		Mat_Trans(n1, p2, Y12, ldy12, Bstr->VT, p2);

		// B{1,1} = SymCompUpdate2 (A{1,1}, Y, V(1:n1,:), alpha, eps, method);
		SymCompUpdate2Struct(n1, k, Astr->left, alpha, Y, ldy, &V[0 + ldv * 0], ldv, Bstr->left, smallsize, eps, method);

		// B{2,2} = SymCompUpdate2 (A{2,2}, Y, V(m:n ,:), alpha, eps, method);
		SymCompUpdate2Struct(n2, k, Astr->right, alpha, Y, ldy, &V[n1 + ldv * 0], ldv, Bstr->right, smallsize, eps, method);


		free_arr(&Y12str->U);
		free_arr(&Y12str->VT);
		free_arr(&Y21str->U);
		free_arr(&Y21str->VT);
		free_arr(&Y21);
		free_arr(&Y12);
		free_arr(&V21);
		free_arr(&V12);
		free_arr(&VY);
		free_arr(&VV);
		free_arr(&V_tr);
		free(Y21str);
		free(Y12str);
	}
}
#endif

/* Функция вычисления разложения симметричной блочно-диагональной матрицы с использование сжатого формата.
Внедиагональные блоки предполагаются диагональными матрицами */
void DirFactFastDiagStruct(int n1, int n2, int n3, double *D, int ldd, double *B, mnode** &Gstr,
	double eps, int smallsize, char *bench)
{
	int n = n1 * n2;
	int nbr = n3; // size of D is equal to nbr blocks by n elements
	int size = n * nbr;

	if (compare_str(7, bench, "display"))
	{
		printf("****************************\n");
		printf("Timing DirFactFastDiag\n");
		printf("****************************\n");
	}

	double tt = omp_get_wtime();
	mnode* *DCstr = (mnode**)malloc(n3 * sizeof(mnode*));
	SymRecCompressStruct(n, &D[ind(0, n)], ldd, DCstr[0], smallsize, eps, "SVD");
	tt = omp_get_wtime() - tt;

	if (compare_str(7, bench, "display")) printf("Compressing D(0) time: %lf\n", tt);

	tt = omp_get_wtime();

	Gstr = (mnode**)malloc(n3 * sizeof(mnode*));
	SymCompRecInvStruct(n, DCstr[0], Gstr[0], smallsize, eps, "SVD");
	tt = omp_get_wtime() - tt;
	if (compare_str(7, bench, "display")) printf("Computing G(1) time: %lf\n", tt);


	for (int k = 1; k < nbr; k++)
	{
		mnode *TDstr, *TD1str;
		tt = omp_get_wtime();
		SymRecCompressStruct(n, &D[ind(k, n)], ldd, DCstr[k], smallsize, eps, "SVD");
		tt = omp_get_wtime() - tt;
		if (compare_str(7, bench, "display")) printf("Compressing D(%d) time: %lf\n", k, tt);
	
		tt = omp_get_wtime();
		CopyStruct(n, Gstr[k - 1], TD1str, smallsize);
	
		DiagMultStruct(n, TD1str, &B[ind(k - 1, n)], smallsize);
		tt = omp_get_wtime() - tt;
		if (compare_str(7, bench, "display")) printf("Mult D(%d) time: %lf\n", k, tt);

		tt = omp_get_wtime();
		AddStruct(n, 1.0, DCstr[k], -1.0, TD1str, TDstr, smallsize, eps, "SVD");
		tt = omp_get_wtime() - tt;
		if (compare_str(7, bench, "display")) printf("Add %d time: %lf\n", k, tt);

		tt = omp_get_wtime();
		SymCompRecInvStruct(n, TDstr, Gstr[k], smallsize, eps, "SVD");
		tt = omp_get_wtime() - tt;
		if (compare_str(7, bench, "display")) printf("Computing G(%d) time: %lf\n", k, tt);
		if (compare_str(7, bench, "display")) printf("\n");

		FreeNodes(n, TDstr, smallsize);
		FreeNodes(n, TD1str, smallsize);
	}

	if (compare_str(7, bench, "display"))
	{
		printf("****************************\n");
		printf("End of DirFactFastDiag\n");
		printf("****************************\n");
	}

	for (int i = n3 - 1; i >= 0; i--)
	{
		FreeNodes(n, DCstr[i], smallsize);
	}

	free(DCstr);
}
#if 1

#endif








#endif
