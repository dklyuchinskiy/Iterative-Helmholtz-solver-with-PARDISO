#include "definitionsHODLR.h"
#include "templatesHODLR.h"
#include "TestSuiteHODLR.h"

void SolveTriangSystemA21(int p, int n, dtype* VT, int ldvt, cmnode* R, int smallsize, double eps, char *method)
{
	int ione = 1;
	dtype zero = 0.0;
	dtype one = 1.0;
	dtype mone = -1.0;

	if (n <= smallsize)
	{
		ztrsm("Right", "Low", "Cong", "NonUnit", &p, &n, &one, R->A, &n, VT, &ldvt);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

		int p2 = (int)ceil(p / 2.0); // p2 > p1
		int p1 = p - p2;

		//printf("n2 = %d, n1 = %d, p2 = %d, p1 = %d\n", n2, n1, p2, p1);

		dtype *inter = alloc_arr<dtype>(p * R->p);

		if (p1 != 0) SolveTriangSystemA21(p1, n1, &VT[0 + ldvt * 0], ldvt, R->left, smallsize, eps, method);
		SolveTriangSystemA21(p2, n1, &VT[p1 + ldvt * 0], ldvt, R->left, smallsize, eps, method);

		if (p1 != 0)
		{
			zgemm("No", "Trans", &p1, &R->p, &n1, &one, &VT[0 + ldvt * 0], &ldvt, R->VT, &R->p, &zero, inter, &p);
			zgemm("No", "Trans", &p1, &n2, &R->p, &mone, inter, &p, R->U, &n2, &one, &VT[0 + ldvt * n1], &ldvt);
		}

		zgemm("No", "Trans", &p2, &R->p, &n1, &one, &VT[p1 + ldvt * 0], &ldvt, R->VT, &R->p, &zero, &inter[p1], &p);
		zgemm("No", "Trans", &p2, &n2, &R->p, &mone, &inter[p1], &p, R->U, &n2, &one, &VT[p1 + ldvt * n1], &ldvt);

		if (p1 != 0) SolveTriangSystemA21(p1, n2, &VT[0 + ldvt * n1], ldvt, R->right, smallsize, eps, method);
		SolveTriangSystemA21(p2, n2, &VT[p1 + ldvt * n1], ldvt, R->right, smallsize, eps, method);

		free_arr(inter);
	}
}

// W * WT - V * Y * VT, Y - symmetric
void SymUpdate5Subroutine(int n2, int n1, dtype alpha, cmnode* Astr, const dtype *Y, int ldy, dtype *V, int ldv, int smallsize, double eps, char* method)
{
	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;

	// Ranks
	int p = Astr->p;

	// Bstr->VT = Astr->VT + alpha * Y * VT 
	dtype *VT = alloc_arr2<dtype>(p * n1); int ldvt = p;
	Mat_Trans(n1, p, V, ldv, VT, ldvt);

	zsymm("Left", "Low", &p, &n1, &alpha, Y, &p, VT, &ldvt, &beta_one, Astr->VT, &p);

	free_arr(VT);
}

// (n x k) (k x k) (k x n)
void SymCompUpdate5LowRankStruct(int n, int k, cmnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V, int ldv, int smallsize, double eps, char* method)
{
	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;

	if (n <= smallsize)
	{
		// X = X + alpha * V * Y * VT

		// C = V * Y
		dtype *C = alloc_arr2<dtype>(n * k); int ldc = n;
		zsymm("Right", "Low", &n, &k, &alpha_one, Y, &ldy, V, &ldv, &beta_zero, C, &ldc);

		// X = X + alpha * C * Vt
		zgemm("No", "Trans", &n, &n, &k, &alpha, C, &ldc, V, &ldv, &beta_one, Astr->A, &n);

		free_arr(C);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // n2 > n1
		int n1 = n - n2;

#if 0
		printf("-----------------------------\n");
		double norm = RelError(zlange, n2, k, Astr->U, n2, &V[n1 + ldv * 0], ldv, eps);
		printf("norm1 diff size n1 = %d, n2 x p = %d x %d = %e\n", n1, n2, k, norm);

		dtype *Tr = alloc_arr<dtype>(k * n2); int ldtr = k;
		Mat_Trans(n1, k, &V[0 + ldv * 0], ldv, Tr, ldtr);
		printf("norm1 diff = %e\n", norm);
		norm = RelError(zlange, k, n1, Astr->VT, k, Tr, ldtr, eps);
	
//		PrintMat(k, n2, Astr->VT, k);
		printf("norm2 diff = %e\n", norm);
//		PrintMat(k, n2, Tr, ldtr);

		printf("-----------------------------\n");
#endif
		// n2 * n1 = (n2 x p) (p x n1) = (n2 x p) (p x n1 - p x p  *  p x n1)
		SymUpdate5Subroutine(n2, n1, alpha, Astr, Y, ldy, &V[0 + ldv * 0], ldv, smallsize, eps, method);

		if (Astr->p != k) printf("!!! error: different sizes of ranks!!!\n");

		// B{1,1} = SymCompUpdate2 (A{1,1}, Y, V(1:n1,:), alpha, eps, method);
		SymCompUpdate5LowRankStruct(n1, k, Astr->left, alpha, Y, ldy, &V[0 + ldv * 0], ldv, smallsize, eps, method);

		// B{2,2} = SymCompUpdate2 (A{2,2}, Y, V(m:n ,:), alpha, eps, method);
		SymCompUpdate5LowRankStruct(n2, k, Astr->right, alpha, Y, ldy, &V[n1 + ldv * 0], ldv, smallsize, eps, method);
	}
}

void LowRankCholeskyFact(int n, cmnode* Astr, dtype *work /*size of (p x p)*/, int smallsize, double eps, char* method)
{
	int ione = 1;
	int mione = -1;
	dtype alpha_one = 1.0;
	dtype alpha_mone = -1.0;
	dtype beta_zero = 0.0;
	int info = 0;

	if (n <= smallsize)
	{
		zpotrf("L", &n, Astr->A, &n, &info);
		if (info != 0) printf("!!! low rank zpotrf error n = %d info = %d\n", n, info);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0);
		int n1 = n - n2;

		// LU for A11
		LowRankCholeskyFact(n1, Astr->left, work, smallsize, eps, method);

#ifdef PRINT
		dtype * A = alloc_arr<dtype>(n * n);
		SymResRestoreStruct(n, Astr, A, n, smallsize);
		PrintMat(n, n, A, n);
#endif

		// Apply to A21: = U1 * V1T * (UP ^ {-1}) or to solve triangular system X * UP = VT
		SolveTriangSystemA21(Astr->p, n1, Astr->VT, Astr->p, Astr->left, smallsize, eps, method);

#ifdef PRINT
		printf("after trsm low rank\n");
		SymResRestoreStruct(n, Astr, A, n, smallsize);
		PrintMat(n, n, A, n);
#endif

		// Double update D:= D - U1 * V1T * (U^{-1}) * (L^{-1}) * U2 * V2T		
		// Update compressed block A[2][2]

		zsyrk("Low", "No", &Astr->p, &n1, &alpha_one, Astr->VT, &Astr->p, &beta_zero, work, &Astr->p);

#ifdef PRINT
		SymResRestoreStruct(n, Astr, A, n, smallsize);
		PrintMat(n, n, A, n);
#endif
		SymCompUpdate5LowRankStruct(n2, Astr->p, Astr->right, alpha_mone, work, Astr->p, Astr->U, n2, smallsize, eps, method);

#ifdef PRINT
		printf("after A22 low rank update\n");
		SymResRestoreStruct(n, Astr, A, n, smallsize);
		PrintMat(n, n, A, n);
#endif
		// LU for A22
		LowRankCholeskyFact(n2, Astr->right, work, smallsize, eps, method);
	}
}
#if 1
void CholeskyFact(int n, dtype* A, int lda, int smallsize, double eps, char* method)
{
	int ione = 1;
	int mione = -1;
	dtype alpha_one = 1.0;
	dtype alpha_mone = -1.0;
	dtype beta_zero = 0.0;
	int info = 0;

	if (n <= smallsize)
	{
		zpotrf("L", &n, A, &lda, &info);
		if (info != 0) printf("!!! local zpotrf error n = %d info = %d\n", n, info);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0);
		int n1 = n - n2;

		// LU for A11
		CholeskyFact(n1, &A[0 + lda * 0], lda, smallsize, eps, method);

	//	printf("Blocked Cholesky A[n1][n1]:\n");
	//	PrintMat(n, n, A, lda);

		ztrsm("Right", "Low", "Trans", "NonUnit", &n2, &n1, &alpha_one, &A[0 + lda * 0], &lda, &A[n1 + lda * 0], &lda);

		printf("after trsm\n");
		PrintMat(n, n, A, lda);

		// Double update D:= D - U1 * V1T * (U^{-1}) * (L^{-1}) * U2 * V2T		
		// Update compressed block A[2][2]
		dtype *Y = alloc_arr<dtype>(n2 * n2); int ldy = n2;

		zsyrk("Low", "No", &n2, &n1, &alpha_one, &A[n1 + lda * 0], &lda, &beta_zero, Y, &ldy);

		// X = X - alpha * V * VT
		for (int i = 0; i < n2; i++)
			for (int j = 0; j < n2; j++)
				A[n1 + i + lda * (n1 + j)] -= Y[i + ldy * j];


		printf("after A22 update\n");
		PrintMat(n, n, A, lda);

		// LU for A22
		CholeskyFact(n2, &A[n1 + lda * n1], lda, smallsize, eps, method);
	}
}
#endif

#if 0
void LowRankToSymmHSS(int n, int r, dtype *U, int ldu, dtype *VT, int ldvt, cmnode *&Aout, int smallsize)
{
	Aout = (cmnode*)malloc(sizeof(cmnode));
	dtype one = 1.0;
	dtype zero = 0.0;

	if (n <= smallsize)
	{
		alloc_dense_node(n, Aout);
		zgemm("no", "no", &n, &n, &r, &one, U, &ldu, VT, &ldvt, &zero, Aout->A, &n);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0);
		int n1 = n - n2;

		Aout->U = alloc_arr<dtype>(n2 * r);
		Aout->VT = alloc_arr<dtype>(r * n1);

		zlacpy("All", &n2, &r, &U[n1 + ldu * 0], &ldu, Aout->U, &n2);
		zlacpy("All", &r, &n1, &VT[0 + ldvt * 0], &ldvt, Aout->VT, &r);

		Aout->p = r;

		LowRankToSymmHSS(n1, r, &U[0  + ldu * 0], ldu, &VT[0 + ldvt *  0], ldvt, Aout->left, smallsize);
		LowRankToSymmHSS(n2, r, &U[n1 + ldu * 0], ldu, &VT[0 + ldvt * n1], ldvt, Aout->right, smallsize);
	}
}
#else
void LowRankToSymmHSS(int n, int r, dtype *U, int ldu, cmnode *&Aout, int smallsize)
{
	Aout = (cmnode*)malloc(sizeof(cmnode));
	dtype one = 1.0;
	dtype zero = 0.0;

	if (n <= smallsize)
	{
		alloc_dense_node(n, Aout);
		zsyrk("left", "no", &n, &r, &one, U, &ldu, &zero, Aout->A, &n);
	}
	else
	{
		int n2 = (int)ceil(n / 2.0);
		int n1 = n - n2;

		Aout->U = alloc_arr<dtype>(n2 * r);
		Aout->VT = alloc_arr<dtype>(r * n1);

		zlacpy("All", &n2, &r, &U[n1 + ldu * 0], &ldu, Aout->U, &n2);
		Mat_Trans(n1, r, &U[0 + ldu * 0], ldu, Aout->VT, r);

		Aout->p = r;

		LowRankToSymmHSS(n1, r, &U[0 + ldu * 0], ldu, Aout->left, smallsize);
		LowRankToSymmHSS(n2, r, &U[n1 + ldu * 0], ldu, Aout->right, smallsize);
	}
}
#endif

void AddSymmHSSDiag(int n, cmnode *Aout, dtype *Diag, int smallsize)
{
	if (n <= smallsize)
	{
		for (int i = 0; i < n; i++)
			Aout->A[i + n * i] += Diag[i];
	}
	else
	{
		int n2 = (int)ceil(n / 2.0);
		int n1 = n - n2;

		AddSymmHSSDiag(n1, Aout->left, &Diag[0], smallsize);
		AddSymmHSSDiag(n2, Aout->right, &Diag[n1], smallsize);
	}
}

