#include "definitions.h"
#include "templates.h"
#include "TemplatesForMatrixConstruction.h"

/*****************************************************
Source file contains functionality to work
with compressed matrices with HSS structure
(for example, Add, Mult, Inverse and etc.).

Also, source file has definitions of support functions,
declared in templates.h
******************************************************/

using namespace std;

// Test for the whole solver

int ind(int j, int n)
{
	return n * j;
}

int compare_str(int n, char *s1, char *s2)
{
	for (int i = 0; i < n; i++)
	{
		if (s1[i] != s2[i]) return 0;
	}
	return 1;
}

void print(int m, int n, dtype *u, int ldu, char *mess)
{
	printf("%s\n", mess);
	for (int i = 0; i < m; i++)
	{
		printf("%d ", i);
		for (int j = 0; j < n; j++)
		{
			printf("%5.3lf ", u[i + ldu*j].real());
		}
		printf("\n");
	}

	printf("\n");
}

void Eye(int n, dtype *H, int ldh)
{
	for (int j = 0; j < n; j++)
#pragma omp simd
		for (int i = 0; i < n; i++)
			if (j == i) H[i + ldh * j] = 1.0;
			else H[i + ldh * j] = 0.0;
}


void Hilbert(int n, dtype *H, int ldh)
{
#pragma omp parallel for schedule(static)
	for (int j = 0; j < n; j++)
#pragma omp simd
		for (int i = 0; i < n; i++)
			H[i + ldh * j] = 1.0 / (i + j + 1);
}

void Mat_Trans(int m, int n, dtype *H, int ldh, dtype *Hcomp_tr, int ldhtr)
{
#pragma omp parallel for schedule(static)
	for (int i = 0; i < m; i++)
#pragma omp simd
		for (int j = 0; j < n; j++)
			Hcomp_tr[j + ldhtr * i] = H[i + ldh * j];
}

void Add_dense(int m, int n, dtype alpha, dtype *A, int lda, dtype beta, dtype *B, int ldb, dtype *C, int ldc)
{
	double dzero = 0.0;

	if (beta == dzero)
	{
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
#pragma omp simd
			for (int i = 0; i < m; i++)
				C[i + ldc * j] = alpha * A[i + lda * j];
	}
	else if (alpha == dzero)
	{
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
#pragma omp simd
			for (int i = 0; i < m; i++)
				C[i + ldc * j] = beta * B[i + ldb * j];
	}
	else
	{
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
#pragma omp simd
			for (int i = 0; i < m; i++)
				C[i + ldc * j] = alpha * A[i + lda * j] + beta * B[i + ldb * j];
	}
}

void Clear(int m, int n, dtype* A, int lda)
{
#pragma omp parallel for schedule(runtime)
	for (int j = 0; j < n; j++)
#pragma omp simd
		for (int i = 0; i < m; i++)
			A[i + lda * j] = 0.0;
}

void GenerateDiagonal1DBlock(double w, int part_of_field, size_m y, size_m z, dtype *DD, int lddd)
{
	int n = y.n;

	Clear(n, n, DD, lddd);
	
	// diagonal blocks in dense format
#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < n; i++)
	{
		DD[i + lddd * i] = -2.0 * (1.0 / (y.h * y.h) + 1.0 / (z.h * z.h)) - dtype{ w , 0 };
		if (i > 0) DD[i + lddd * (i - 1)] = 1.0 / (y.h * y.h);
		if (i < n - 1) DD[i + lddd * (i + 1)] = 1.0 / (y.h * y.h);
	}

}

void GenRhs2D(int w, size_m x, size_m y, size_m z, dtype* f, dtype* f2D)
{
	int l = 0;
	
	// 0 < w < x.n
#if 0
	for (int k = 0; k < z.n; k++)
		for(int j = 0; j < y.n; j++)
			for (int i = 0; i < x.n; i++)
			{	
				if (i == w)
				{
					f2D[l++] = f[i + j * x.n + k * x.n * y.n];
				}
			}
#else
		for (int k = 0; k < y.n * z.n; k++)
			f2D[k] = f[x.n * k + w];
#endif
}

void GenSol1DBackward(int w, size_m x, size_m y, size_m z, dtype* x_sol_prd, dtype *u1D)
{
	int l = 0;

	// 0 < j < y.n * z.n

#if 0
	for (int k = 0; k < x.n; k++)
		for (int j = 0; j < y.n * z.n; j++)
			{
				if (j == w)
				{
					u1D[l++] = x_sol_prd[j + k * y.n * z.n];
				}
		}
#else
	for (int k = 0; k < x.n; k++)
			u1D[k] = x_sol_prd[w + k * y.n * z.n];
#endif
}

void GenerateDiagonal2DBlock(int part_of_field, size_m x, size_m y, size_m z, dtype *DD, int lddd)
{
	int n = x.n * y.n;
	int size = n * z.n;

	// diagonal blocks in dense format
//#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < n; i++)
	{
		DD[i + lddd * i] = -2.0 * (1.0 / (x.h * x.h) + 1.0 / (y.h * y.h) + 1.0 / (z.h * z.h));

		if (i > 0) DD[i + lddd * (i - 1)] = 1.0 / (x.h * x.h);
		if (i < n - 1) DD[i + lddd * (i + 1)] = 1.0 / (x.h * x.h);
		if (i >= x.n) DD[i + lddd * (i - x.n)] = 1.0 / (y.h * y.h);
		if (i <= n - x.n - 1)  DD[i + lddd * (i + x.n)] = 1.0 / (y.h * y.h);

	}
	for (int i = 0; i < n; i++)
	{
		if (i % x.n == 0 && i > 0)
		{
			DD[i - 1 + lddd * i] = 0;
			DD[i + lddd * (i - 1)] = 0;
		}
	}

}

void GenRHS2DandSolutionSyntetic(int i, size_m y, size_m z, dcsr* D2csr, dtype* u2Dsynt, dtype *f2D)
{

}

// v[i] = D[i] * v[i]
void DenseDiagMult(int n, dtype *diag, dtype *v, dtype *f)
{
#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < n; i++)
		f[i] = diag[i] * v[i];
}

void GenRHSandSolution(size_m x, size_m y, size_m z, /* output */ dtype* B, dtype *u, dtype *f)
{
	int n = x.n * y.n;
	int size = n * z.n;

	// Set vector B
#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < z.n - 1; j++)
#pragma omp simd
		for (int i = 0; i < n; i++)
			B[ind(j, n) + i] = 1.0 / (z.h * z.h);

	// approximation of exact right hand side (inner grid points)
#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < z.n; k++)
		for (int j = 0; j < y.n; j++)
#pragma omp simd
			for (int i = 0; i < x.n; i++)
				f[k * n + j * x.n + i] = F_ex(x, y, z, (i + 1) * x.h, (j + 1) * y.h, (k + 1) * z.h);

	// for boundaries z = 0 and z = Lz we distract blocks B0 and Bm from the RHS
#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < y.n; j++)
#pragma omp simd
		for (int i = 0; i < x.n; i++)
		{
			f[ind(0, n) + ind(j, x.n) + i] -= u_ex(x, y, z, (i + 1) * x.h, (j + 1) * y.h, 0) / (z.h * z.h); // u|z = 0
			f[ind(z.n - 1, n) + ind(j, x.n) + i] -= u_ex(z, y, z, (i + 1)  * x.h, (j + 1) * y.h, z.l) / (z.h * z.h); // u|z = h
		}


	// for each boundary 0 <= z <= Lz
	// we distract 4 known boundaries f0, fl, g0, gL from right hand side
#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < z.n; k++)
	{
#pragma omp simd
		for (int i = 0; i < x.n; i++)
		{
			f[k * n + 0 * x.n + i] -= u_ex(x, y, z, (i + 1) * x.h, 0, (k + 1) * z.h) / (y.h * y.h);
			f[k * n + (y.n - 1) * x.n + i] -= u_ex(x, y, z, (i + 1) * x.h, y.l, (k + 1) * z.h) / (y.h * y.h);
		}
		for (int j = 0; j < y.n; j++)
		{
			f[k * n + j * x.n + 0] -= u_ex(x, y, z, 0, (j + 1) * y.h, (k + 1) * z.h) / (x.h * x.h);
			f[k * n + j * x.n + x.n - 1] -= u_ex(x, y, z, x.l, (j + 1) * y.h, (k + 1) * z.h) / (x.h * x.h);
		}
	}

	// approximation of inner points values
#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < z.n; k++)
		for (int j = 0; j < y.n; j++)
#pragma omp simd
			for (int i = 0; i < x.n; i++)
				u[k * n + j * x.n + i] = u_ex(x, y, z, (i + 1) * x.h, (j + 1) * y.h, (k + 1) * z.h);


	printf("RHS and solution are constructed\n");
}


void GenSparseMatrixOnline3D(size_m x, size_m y, size_m z, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, ccsr* Acsr)
{
	int n = x.n * y.n;
	int size = n * z.n;
	int non_zeros_on_prev_level = 0;
	map<vector<int>, dtype> CSR;

	Diag(n, BL, ldbl, dtype{ 1.0 / (z.h * z.h), 0 });
	Diag(n, BR, ldbr, dtype{ 1.0 / (z.h * z.h), 0 });


	for (int blk = 0; blk < z.n; blk++)
	{
		GenerateDiagonal2DBlock(blk, x, y, z, A, lda);
		CSR = BlockRowMat_to_CSR(blk, x.n, y.n, z.n, BL, ldbl, A, lda, BR, ldbr, Acsr, non_zeros_on_prev_level); // ВL, ВR and A - is 2D dimensional matrices (n x n)
																												 //	print_map(CSR);
																												 //	system("pause");
	}
	printf("Non_zeros inside generating function: %d\n", non_zeros_on_prev_level);

}

map<vector<int>, dtype> BlockRowMat_to_CSR(int blk, int n1, int n2, int n3, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, ccsr* Acsr, int& non_zeros_on_prev_level)
{
	map<vector<int>, dtype> CSR_A;
	vector<int> v(2, 0);
	int n = n1 * n2;
	int k = 0;
	dtype *Arow = alloc_arr<dtype>(n * 3 * n); int ldar = n;

	if (blk == 0)
	{
		construct_block_row(zlacpy, n, n, (dtype*)NULL, ldbl, A, lda, BR, ldbr, Arow, ldar);
		CSR_A = dense_to_CSR(n, 2 * n, Arow, ldar, &Acsr->ia[0], &Acsr->ja[0], &Acsr->values[0]);
		non_zeros_on_prev_level = CSR_A.size();
	}
	else if (blk == n3 - 1)
	{
		construct_block_row(zlacpy, n, n, BL, ldbl, A, lda, (dtype*)NULL, ldbr, Arow, ldar);
		CSR_A = dense_to_CSR(n, 2 * n, Arow, ldar, &Acsr->ia[ind(blk, n)], &Acsr->ja[non_zeros_on_prev_level], &Acsr->values[non_zeros_on_prev_level]);
		shift_values(n, &Acsr->ia[ind(blk, n)], non_zeros_on_prev_level, CSR_A.size(), &Acsr->ja[non_zeros_on_prev_level], n * (blk - 1));
		non_zeros_on_prev_level += CSR_A.size();
	}
	else
	{
		construct_block_row(zlacpy, n, n, BL, ldbl, A, lda, BR, ldbr, Arow, ldar);
		CSR_A = dense_to_CSR(n, 3 * n, Arow, ldar, &Acsr->ia[ind(blk, n)], &Acsr->ja[non_zeros_on_prev_level], &Acsr->values[non_zeros_on_prev_level]);

		// shift values of arrays according to previous level
		shift_values(n, &Acsr->ia[ind(blk, n)], non_zeros_on_prev_level, CSR_A.size(), &Acsr->ja[non_zeros_on_prev_level], n * (blk - 1));
		non_zeros_on_prev_level += CSR_A.size();
	}

	free(Arow);
	return CSR_A;
}

map<vector<int>, dtype> Block1DRowMat_to_CSR(int blk, int n1, int n2, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, ccsr* Acsr, int& non_zeros_on_prev_level)
{
	map<vector<int>, dtype> CSR_A;
	map<vector<int>, dtype> CSR;
	vector<int> v(2, 0);

	int n = n1;
	dtype *Arow = alloc_arr<dtype>(n * 3 * n); int ldar = n;

	if (blk == 0)
	{
		construct_block_row(zlacpy, n, n, (dtype*)NULL, ldbl, A, lda, BR, ldbr, Arow, ldar);
		CSR_A = dense_to_CSR(n, 2 * n, Arow, ldar, &Acsr->ia[0], &Acsr->ja[0], &Acsr->values[0]);
		non_zeros_on_prev_level = CSR_A.size();
	}
	else if (blk == n2 - 1)
	{
		construct_block_row(zlacpy, n, n, BL, ldbl, A, lda, (dtype*)NULL, ldbr, Arow, ldar);
		CSR_A = dense_to_CSR(n, 2 * n, Arow, ldar, &Acsr->ia[ind(blk, n)], &Acsr->ja[non_zeros_on_prev_level], &Acsr->values[non_zeros_on_prev_level]);
		shift_values(n, &Acsr->ia[ind(blk, n)], non_zeros_on_prev_level, CSR_A.size(), &Acsr->ja[non_zeros_on_prev_level], n * (blk - 1));
		non_zeros_on_prev_level += CSR_A.size();
	}
	else
	{
		construct_block_row(zlacpy, n, n, BL, ldbl, A, lda, BR, ldbr, Arow, ldar);
		CSR_A = dense_to_CSR(n, 3 * n, Arow, ldar, &Acsr->ia[ind(blk, n)], &Acsr->ja[non_zeros_on_prev_level], &Acsr->values[non_zeros_on_prev_level]);

		// shift values of arrays according to previous level
		shift_values(n, &Acsr->ia[ind(blk, n)], non_zeros_on_prev_level, CSR_A.size(), &Acsr->ja[non_zeros_on_prev_level], n * (blk - 1));
		non_zeros_on_prev_level += CSR_A.size();
	}

	free_arr(Arow);

	return CSR_A;
}

void GenSparseMatrixOnline2D(int w, size_m y, size_m z, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, ccsr* Acsr)
{
	int n = y.n;
	int n2 = n / 2;
	int size = n * z.n;
	int non_zeros_on_prev_level = 0;
	map<vector<int>, dtype> CSR;

	Diag(n, BL, ldbl, dtype{ 1.0 / (z.h * z.h), 0 });
	Diag(n, BR, ldbr, dtype{ 1.0 / (z.h * z.h), 0 });

	double kww = 4 * PI * PI * (w - n2) * (w - n2) / (y.l * y.l);
	printf("Gen 2D matrix for frequency w = %d\n", w - n2);
	for (int blk = 0; blk < z.n; blk++)
	{
		//printf("Blk: %d\n", blk);
		GenerateDiagonal1DBlock(kww, blk, y, z, A, lda);
		CSR = Block1DRowMat_to_CSR(blk, y.n, z.n, BL, ldbl, A, lda, BR, ldbr, Acsr, non_zeros_on_prev_level); // ВL, ВR and A - is 2D dimensional matrices (n x n)
	//	print_map(CSR);
	}

	printf("Non_zeros in 2D block: %d\n", non_zeros_on_prev_level);
}

void SolvePardiso3D(size_m x, size_m y, size_m z, ccsr* Dcsr, dtype* x_pard, dtype* f, double thresh)
{
	int size = x.n * y.n * z.n;
	int mtype = 13;
	int *iparm = alloc_arr<int>(64);
	int *perm = alloc_arr<int>(size);
	size_t *pt = alloc_arr<size_t>(64);

	int maxfct = 1;
	int mnum = 1;
	int phase = 0;
	int rhs = 1;
	int msglvl = 0;
	int error = 0;
	phase = 13;

	pardiso(pt, &maxfct, &mnum, &mtype, &phase, &size, Dcsr->values, Dcsr->ia, Dcsr->ja, perm, &rhs, iparm, &msglvl, f, x_pard, &error);
}

void print_csr(int n, dcsr* A)
{
	for (int i = 0; i < n; i++)
		printf("i = %d j = %d value = %lf\n", A->ia[i], A->ja[i], A->values[i]);
}

void shift_values(int rows, int *ia, int shift_non_zeros, int non_zeros, int *ja, int shift_columns)
{
#pragma omp parallel for schedule(simd:static)
	for (int i = 0; i < rows; i++)
		ia[i] += shift_non_zeros;

#pragma omp parallel for schedule(simd:static)
	for (int i = 0; i < non_zeros; i++)
		ja[i] += shift_columns;
}

double F_ex(size_m xx, size_m yy, size_m zz, double x, double y, double z)
{
//	return -4.0 * PI * PI * (1.0 / (xx.n * xx.n) + 1.0 / (yy.n * yy.n) + 1.0 / (zz.n * zz.n)) * sin(2 * PI * x / xx.n) * sin(2 * PI * y / yy.n) * sin(2 * PI * z / zz.n);
//	return 0;

//	return 2.0 * (x * (x - xx.l) * z * (z - zz.l) + y * (y - yy.l) * z * (z - zz.l) + x * (x - xx.l) * y * (y - yy.l));

	return sin(2 * PI * x) * sin(2 * PI * y) * sin(2 * PI * z);
}

double u_ex(size_m xx, size_m yy, size_m zz, double x, double y, double z)
{
//	return 2.0 + sin(2.0 * PI * x / xx.n) * sin(2.0 * PI * y / yy.n) * sin(2.0 * PI * z / zz.n);
//	return x * x + y * y - 2.0 * z * z;
//	return x * x - y * y;

//	return x * y * z * (x - xx.l) * (y - yy.l) * (z - zz.l);

	return -sin(2 * PI * x) * sin(2 * PI * y) * sin(2 * PI * z) / (12.0 * PI * PI + 1.0);
}

void print_map(const map<vector<int>, dtype>& SD)
{
	cout << "SD size = " << SD.size() << endl;
	for (const auto& item : SD)
		cout << "m = " << item.first[0] << " n = " << item.first[1] << " value = " << item.second << endl;

}

double rel_error(int n, int k, double *Hrec, double *Hinit, int ldh, double eps)
{
	double norm = 0;

	// Norm of residual
#pragma omp parallel for schedule(static)
	for (int j = 0; j < k; j++)
#pragma omp simd
		for (int i = 0; i < n; i++)
			Hrec[i + ldh * j] = Hrec[i + ldh * j] - Hinit[i + ldh * j];

	norm = dlange("Frob", &n, &k, Hrec, &ldh, NULL);
	norm = norm / dlange("Frob", &n, &k, Hinit, &ldh, NULL);

	return norm;

	//if (norm < eps) printf("Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
	//else printf("Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);
}

void ResidCSR(int n1, int n2, int n3, ccsr* Dcsr, dtype* x_sol, dtype *f, dtype* g, double &RelRes)
{
	int n = n1 * n2;
	int size = n * n3;
	dtype *f1 = alloc_arr<dtype>(size);
	int ione = 1;

	// Multiply matrix A in CSR format by vector x_sol to obtain f1
	mkl_zcsrgemv("No", &size, Dcsr->values, Dcsr->ia, Dcsr->ja, x_sol, f1);

#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < size; i++)
		g[i] = f[i] - f1[i];

	//	for (int i = 0; i < size; i++)
	//		printf("%lf %lf\n", f[i], f1[i]);

#ifdef DEBUG
	print_vec(size, f, g, "f and g");
#endif

	RelRes = zlange("Frob", &size, &ione, g, &size, NULL);
	RelRes = RelRes / zlange("Frob", &size, &ione, f, &size, NULL);

	free_arr(f1);
}



void GenRHSandSolution2D_Syntetic(size_m x, size_m y, ccsr *Dcsr, dtype *u, dtype *f)
{
	printf("GenRHSandSolution2D_Syntetic...\n");
	int n = x.n;
	int size = n * y.n;

	// approximation of inner points values
	GenSolVector(size, u);

	printf("Multiply f := Acsr * u\n");
	mkl_zcsrgemv("No", &size, Dcsr->values, Dcsr->ia, Dcsr->ja, u, f);

	printf("RHS and solution are constructed\n");
}

void GenSolVector(int size, dtype *vector)
{
	srand((unsigned int)time(0));
	for (int i = 0; i < size; i++)
	{
		vector[i] = random(0.0, 1.0);
		//	printf("%lf\n", vector[i].real());
	}
}

double random(double min, double max)
{
	return (double)(rand()) / RAND_MAX * (max - min) + min;
}





#if 0
// Low Rank approximation
void LowRankApprox(int n2, int n1 /* size of A21 = A */, double *A /* A is overwritten by U */, int lda,
				   double *V /* V is stored in A12 */, int ldv, int &p, double eps, char *method)
{
	int mn = min(n1, n2);
	int info = 0;
	int lwork = -1;
	p = 0;

	double wkopt;
	double *work;
	double *S;

	if (compare_str(3, method, "SVD"))
	{
		S = alloc_arr<double>(mn);

		// query 
		dgesvd("Over", "Sing", &n2, &n1, A, &lda, S, V, &ldv, V, &ldv, &wkopt, &lwork, &info); // first V - not referenced
		lwork = (int)wkopt;
		work = alloc_arr<dtype>(lwork);

		// A = U1 * S * V1
		dgesvd("Over", "Sing", &n2, &n1, A, &lda, S, V, &ldv, V, &ldv, work, &lwork, &info); // first V - not reference
		// error 2 (как mkl складывает вектора columnwise)

		for (int j = 0; j < mn; j++)
		{
			double s1 = S[j] / S[0];
			if (s1 < eps)
			{
				break;
			}
			p = j + 1;
			for (int i = 0; i < n2; i++)
				A[i + lda * j] *= S[j];
		}

#ifdef DEBUG
		printf("LowRank after SVD: n2 = %d, n1 = %d, p = %d\n", n2, n1, p);
#endif

		// n1
		for (int j = p; j < mn; j++)   // original part: [n2 x n1], but overwritten part [n2 x min(n2,n1)]
			for (int i = 0; i < n2; i++)
				A[i + lda * j] = 0;

		for (int j = 0; j < n1; j++)   // transposed part: [min(n2,n1) x n1] 
			for (int i = p; i < mn; i++)
				V[i + ldv * j] = 0;
		
		free_arr(&S);
		free_arr(&work);
	}
	else
	{
		return;
	}
}

void GenMatrixandRHSandSolution(const int n1, const int n2, const int n3, double *D, int ldd, double *B, double *x1, double *f)
{
	// 1. Аппроксимация двумерной плоскости

	int n = n1 * n2; // size of blocks
	int nbr = n3; // number of blocks in one row
	int NBF = nbr * nbr; // full number of blocks in matrix
	dtype *DD = alloc_arr<dtype>(n * n); // память под двумерный диагональный блок
	int lddd = n;
	int size = n * nbr;

	dtype *f_help = alloc_arr<dtype>(n);

	double done = 1.0;
	double dzero = 0.0;
	int ione = 1;

	double time1, time2;

	// переделать все это размеры

	// size DD
	int m0 = n;
	int n0 = n;

	// diagonal blocks in dense format
	for (int i = 0; i < n; i++)
		DD[i + lddd * i] = 6.0;

	for (int j = 1; j < n; j++)  // count - until the end
	{
		DD[j + lddd * (j - 1)] = -1.0;
		DD[j - 1 + lddd * j] = -1.0;
	}

	for (int j = n1; j < n; j++) // count - until the end
	{
		DD[j + lddd * (j - n1)] = -1.0;
		DD[j - n1 + lddd * j] = -1.0;
	}

	//print(n, n, DD, lddd);

	// packing into sparse format
	// 5 diagonal matrix with size n2 * nbr + 2 * (n2 * nbr - 1) + 2 * (n2 * nbr - n1)

	int sparse_size = n + 2 * (n - 1) + 2 * (n - n1);
	double *d = (double*)malloc(sparse_size * sizeof(double));
	int *i_ind = (int*)malloc(sparse_size * sizeof(int));
	int *j_ind = (int*)malloc(sparse_size * sizeof(int));

	printf("sparse_size = %d\n", sparse_size);
	map<vector<int>, double> SD;
	SD = dense_to_sparse(n, n, DD, lddd, i_ind, j_ind, d);
	//print_map(SD);


	// Using only sparse matrix D - for 3D. Transfer each 2D block SD to 3D matrix D

	GenSolVector(size, x1);

#pragma omp parallel for
	for (int j = 0; j < nbr; j++)
	{
		dlacpy("All", &n, &n, DD, &lddd, &D[ind(j, n) + ldd * 0], &ldd);
		for (int i = 0; i < n; i++)
		{
			if (j < nbr - 1) B[ind(j, n) + i] = -1.0;
		}
	}

#if 0
	time1 = omp_get_wtime();
	// f[i + n * 0]
	// попробуем использовать один и тот же 2D вектор SD для всех блоков NB
#pragma omp parallel for schedule(guided)
	for (int i = 0; i < n; i++)
	{
		f[i + n * 0] = B[i + n * 0] * x1[i + n * 1]; // f[1]
		f[i + n * (nbr - 1)] = B[i + n * (nbr - 2)] * x1[i + n * (nbr - 2)]; // f[NB]
		for (int j = 0; j < n; j++)
		{
			vector<int> vect = { i,  j };
			if (SD.count(vect))
			{
				f[i + n * 0] += SD[vect] * x1[j + n * 0];
				f[i + n * (nbr - 1)] += SD[vect] * x1[j + n * (nbr - 1)];
			}
		}
	}

#pragma omp parallel for schedule(guided)
	for (int blk = 1; blk < nbr - 1; blk++)
		for (int i = 0; i < n; i++)
		{
			f[i + n * blk] = B[i + n * (blk - 1)] * x1[i + n * (blk - 1)] + B[i + n * blk] * x1[i + n * (blk + 1)];
			for (int j = 0; j < n; j++)
			{
				vector<int> vect = { i,  j };
				if (SD.count(vect))
				{
					f[i + n * blk] += SD[vect] * x1[j + n * blk];
				}
			}
	}
	time1 = omp_get_wtime() - time1;
#endif

	// через mkl

	time2 = omp_get_wtime();
	// f[1] = D[1] * x[1] + diag{B[1]} * x[2]
	DenseDiagMult(n, &B[ind(0, n)], &x1[ind(1, n)], &f[ind(0, n)]);
	dsymv("Up", &n, &done, &D[ind(0, n)], &ldd, &x1[ind(0, n)], &ione, &done, &f[ind(0, n)], &ione);

	// f[N] = diag(B{N-1}) * x{N-1} + D{N} * x{N};
	DenseDiagMult(n, &B[ind(nbr - 2, n)], &x1[ind(nbr - 2, n)], &f[ind(nbr - 1, n)]);
	dsymv("Up", &n, &done, &D[ind(nbr - 1, n)], &ldd, &x1[ind(nbr - 1, n)], &ione, &done, &f[ind(nbr - 1, n)], &ione);


	// f{ i } = diag(B{ i - 1 }) * x{ i - 1 } + D{ i } * x{ i } + diag(B{ i }) * x{ i + 1 };
	for (int blk = 1; blk < nbr - 1; blk++)
	{
		// f{ i } = diag(B{ i - 1 }) * x { i - 1 } + diag(B{ i }) * x { i + 1 };
		DenseDiagMult(n, &B[ind(blk - 1, n)], &x1[ind(blk - 1, n)], &f[ind(blk, n)]);
		DenseDiagMult(n, &B[ind(blk, n)], &x1[ind(blk + 1, n)], f_help);
		daxpby(&n, &done, f_help, &ione, &done, &f[ind(blk, n)], &ione);
		//Add_dense_vect(n, done, &f[ind(blk, n)], done, f_help, &f[ind(blk, n)]);

		// f{i} = f{i} + D{ i } * x{ i } 
		dsymv("Up", &n, &done, &D[ind(blk, n)], &ldd, &x1[ind(blk, n)], &ione, &done, &f[ind(blk, n)], &ione);
	}

	time2 = omp_get_wtime() - time2;

	printf("time_mkl = %lf\n", time2);

	free_arr(&DD);
	free_arr(&d);
	free(i_ind);
	free(j_ind);
	free_arr(&f_help);
}

/*! \brief \b F_ex
*
* === Documentation ===
*
* \par Purpose:
* ============
* 
 \verbatim

  F_ex performs exact right hand side function

 \endverbatim

 Arguments 
 ==========
 \param[in] x
 \verbatim
			x is double
			On entry, value x
 \endverbatim

  \param[in] y
	\verbatim
			y is double
			On entry, value z
 \endverbatim

 \param[in] z
 \verbatim
			z is double
			On entry, value z
 \endverbatim

 Authors:
========
 
\author Novosibirsk State University


\date January 2018

 \par Further Details:
  =====================
 
\verbatim

  Level 3 Blas routine.

  -- Written on 30-January-2018.
	Dmitriy Klyuchinskiy, Novosibirsk State University
\endverbatim
*/

void GenMatrixandRHSandSolution2(size_m x, size_m y, size_m z,
	/* output */ double *D, int ldd, double *B, double *u, double *f, double thresh)
{
	int n = x.n * y.n; // size of blocks
	int nbr = z.n; // number of blocks in one row
	int lddd = n;
	int size = n * nbr;
	double done = 1.0;
	double dzero = 0.0;
	int ione = 1;

	double *DD = alloc_arr(n * n); // 2D diagonal template block
	double *Au = alloc_arr(size); // mult of generated A and exact solution u

	// n - number of unknowns
	// n * n * n - all unknowns

	// f_rhs = f_inner + f_bound

	// approximation of exact right hand side (inner grid points)
#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < nbr; k++)
		for (int j = 0; j < y.n; j++)
#pragma omp simd
			for (int i = 0; i < x.n; i++)
				f[k * n + j * x.n + i] = F_ex((i + 1) * x.h, (j + 1) * y.h, (k + 1) * z.h);

	// for boundaries z = 0 and z = Lz we distract blocks B0 and Bm from the RHS
#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < y.n; j++)
		for (int i = 0; i < x.n; i++)
		{
			f[ind(0, n) + ind(j, x.n) + i] -= u_ex((i + 1) * x.h, (j + 1) * y.h, 0) / (z.h * z.h); // u|z = 0
			f[ind(nbr - 1, n) + ind(j, x.n) + i] -= u_ex((i + 1)  * x.h, (j + 1) * y.h, z.l) / (z.h * z.h); // u|z = h
		}


	// for each boundary 0 <= z <= Lz
	// we distract 4 known boundaries f0, fl, g0, gL from right hand side
#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < nbr; k++)
	{
#pragma omp simd
			for (int i = 0; i < x.n; i++)
			{
				f[k * n + 0 * x.n + i] -= u_ex((i + 1) * x.h, 0, (k + 1) * z.h) / (y.h * y.h);
				f[k * n + (y.n - 1) * x.n + i] -= u_ex((i + 1) * x.h, y.l, (k + 1) * z.h) / (y.h * y.h);
			}
			for (int j = 0; j < y.n; j++)
			{
				f[k * n + j * x.n + 0] -= u_ex(0, (j + 1) * y.h, (k  + 1) * z.h) / (x.h * x.h);
				f[k * n + j * x.n + x.n - 1] -= u_ex(x.l, (j + 1) * y.h, (k + 1) * z.h) / (x.h * x.h);
			}
	}
//	if (i % x.n == 0 || (i + 1) % x.n == 0) DD[i + lddd * i] = 1.0;

	// Set vector B
#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < z.n - 1; j++)
#pragma omp simd
		for (int i = 0; i < n; i++)
			B[ind(j, n) + i] = 1.0 / (z.h * z.h);

	for (int j = 0; j < nbr; j++)
	{
		GenerateDiagonal2DBlock(j, x, y, z, DD, lddd);
		dlacpy("All", &n, &n, DD, &lddd, &D[ind(j, n) + ldd * 0], &ldd);
	}
	
	// approximation of inner points values
#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < nbr; k++)
		for (int j = 0; j < y.n; j++)
#pragma omp simd
			for (int i = 0; i < x.n; i++)
				u[ind(k, n) + ind(j, x.n) + i] = u_ex((i + 1) * x.h, (j + 1) * y.h, (k + 1) * z.h);
	
	Mult_Au(x.n, y.n, z.n, D, ldd, B, u, Au);

#ifdef DEBUG
	print_vec(size - n, B, B, "B_vector");
	print_vec_mat(size, n, D, ldd, u, "D and u");
	print_vec(size, Au, f, "Au_ex vs F_ex");
	system("pause");
#endif

	// check error between Au and F
	rel_error(size, 1.0, Au, f, size, thresh);

	free(Au);
	free(DD);

}


void Mult_Au(int n1, int n2, int n3, double *D, int ldd, double *B, double *u, double *Au /*output*/)
{
	int n = n1 * n2;
	int nbr = n3;
	int size = n * nbr;
	double done = 1.0;
	double dzero = 0.0;
	int ione = 1;
	double *f_help = alloc_arr(n);

	// f[1] = D{1} * x{1} + diag(B{1}) * x{2};
	DenseDiagMult(n, &B[ind(0, n)], &u[ind(1, n)], &Au[ind(0, n)]);
	dgemv("No", &n, &n, &done, &D[ind(0, n)], &ldd, &u[ind(0, n)], &ione, &done, &Au[ind(0, n)], &ione);

	// f[N] = diag(B{N-1}) * x{N-1} + D{N} * x{N};
	DenseDiagMult(n, &B[ind(nbr - 2, n)], &u[ind(nbr - 2, n)], &Au[ind(nbr - 1, n)]);
	dgemv("No", &n, &n, &done, &D[ind(nbr - 1, n)], &ldd, &u[ind(nbr - 1, n)], &ione, &done, &Au[ind(nbr - 1, n)], &ione);

	// f{ i } = diag(B{ i - 1 }) * x{ i - 1 } + D{ i } * x{ i } + diag(B{ i }) * x{ i + 1 };
	for (int blk = 1; blk < nbr - 1; blk++)
	{
		// f{ i } = diag(B{ i - 1 }) * x { i - 1 } + diag(B{ i }) * x { i + 1 };
		DenseDiagMult(n, &B[ind(blk - 1, n)], &u[ind(blk - 1, n)], &Au[ind(blk, n)]);
		DenseDiagMult(n, &B[ind(blk, n)], &u[ind(blk + 1, n)], f_help);
		daxpby(&n, &done, f_help, &ione, &done, &Au[ind(blk, n)], &ione);

		// f{i} = f{i} + D{ i } * x{ i }  matrix D - non symmetric
		dgemv("No", &n, &n, &done, &D[ind(blk, n)], &ldd, &u[ind(blk, n)], &ione, &done, &Au[ind(blk, n)], &ione);
	}

	free_arr(&f_help);
}

inline void Add_dense_vect(int n, double alpha, double *a, double beta, double *b, double *c)
{
#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < n; i++)
		c[i] = alpha * a[i] + beta * b[i];
}

void GenSolVector(int size, double *vector)
{
	srand((unsigned int)time(0));
	for (int i = 0; i < size; i++)
		vector[i] = random(0.0, 1.0);
}

void Block3DSPDSolveFast(int n1, int n2, int n3, double *D, int ldd, double *B, double *f, double thresh, int smallsize, int ItRef, char *bench,
			/* output */ double *G, int ldg, double *x_sol, int &success, double &RelRes, int &itcount)
{
	int size = n1 * n2 * n3;
	int n = n1 * n2;
	double tt;
	double tt1;
	double *DI = alloc_arr(size * n); int lddi = size;
	dlacpy("All", &size, &n, D, &ldd, DI, &lddi);

	tt = omp_get_wtime();
	DirFactFastDiag(n1, n2, n3, D, ldd, B, G, ldg, thresh, smallsize, bench);
	tt = omp_get_wtime() - tt;
	if (compare_str(7, bench, "display"))
	{
		printf("Total factorization time: %lf\n", tt);
	}

	tt = omp_get_wtime();
	DirSolveFastDiag(n1, n2, n3, G, ldg, B, f, x_sol, thresh, smallsize);
	tt = omp_get_wtime() - tt;
	if (compare_str(7, bench, "display"))
	{
		printf("Solving time: %lf\n", tt);
	}

	double *g = alloc_arr(size);
	double *x1 = alloc_arr(size);
	RelRes = 1;
	Resid(n1, n2, n3, DI, lddi, B, x_sol, f, g, RelRes);

	printf("RelRes = %lf\n", RelRes);
	if (RelRes < thresh)
	{
		success = 1;
		itcount = 0;
	}
	else {
		int success = 0;
		if (ItRef > 0) {
			if (compare_str(7, bench, "display")) printf("Iterative refinement started\n");
			tt1 = omp_get_wtime();
			itcount = 0;
			while ((RelRes > thresh) && (itcount < ItRef))
			{
				tt = omp_get_wtime();
				DirSolveFastDiag(n1, n2, n3, G, ldg, B, g, x1, thresh, smallsize);

#pragma omp parallel for simd schedule(simd:static)
				for (int i = 0; i < size; i++)
					x_sol[i] = x_sol[i] + x1[i];

				Resid(n1, n2, n3, DI, lddi, B, x_sol, f, g, RelRes); // начальное решение f сравниваем с решением A_x0 + A_x1 + A_x2, где
				itcount = itcount + 1;
				tt = omp_get_wtime() - tt;
				if (compare_str(7, bench, "display")) printf("itcount=%d, RelRes=%lf, Time=%lf\n", itcount, RelRes, tt);
			}
			if ((RelRes < thresh) && (itcount < ItRef)) success = 1; // b

			tt1 = omp_get_wtime() - tt1;
			if (compare_str(7, bench, "display")) printf("Iterative refinement total time: %lf\n", tt1);
		}
	}

	free_arr(&DI);
	free_arr(&g);
	free_arr(&x1);
}

// невязка g = Ax - f

// RelRes - относительная невязка = ||g|| / ||f||
void Resid(int n1, int n2, int n3, double *D, int ldd, double *B, double *x_sol, double *f, double *g, double &RelRes)
{
	int n = n1 * n2;
	int size = n * n3;
	double *f1 = alloc_arr(size);
	double done = 1.0;
	int ione = 1;

	Mult_Au(n1, n2, n3, D, ldd, B, x_sol, f1);

#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < size; i++)
		g[i] = f[i] - f1[i];

#ifdef DEBUG
	print_vec(size, f, g, "f and g");
#endif

	RelRes = dlange("Frob", &size, &ione, g, &size, NULL);
	RelRes = RelRes / dlange("Frob", &size, &ione, f, &size, NULL);

	free_arr(&f1);

}

/* Функция вычисления разложения симметричной блочно-диагональной матрицы с использование сжатого формата. 
   Внедиагональные блоки предполагаются диагональными матрицами */
void DirFactFastDiag(int n1, int n2, int n3, double *D, int ldd, double *B, double *G /*factorized matrix*/, int ldg, 
									 double eps, int smallsize, char *bench)
{
	int n = n1 * n2;
	int nbr = n3; // size of D is equal to nbr blocks by n elements
	int size = n * nbr;
	double *TD1 = alloc_arr(n * n); int ldtd = n;
	double *TD = alloc_arr(n * n);

	if (compare_str(7, bench, "display"))
	{
		printf("****************************\n");
		printf("Timing DirFactFastDiag\n");
		printf("****************************\n");
	}

	double tt = omp_get_wtime();
	SymRecCompress(n, &D[ind(0, n)], ldd, smallsize, eps, "SVD");
	tt = omp_get_wtime() - tt;

	if (compare_str(7, bench, "display")) printf("Compressing D(0) time: %lf\n", tt);

	tt = omp_get_wtime();
	SymCompRecInv(n, &D[ind(0, n)], ldd, &G[ind(0, n)], ldg, smallsize, eps, "SVD");
	tt = omp_get_wtime() - tt;
	if (compare_str(7, bench, "display")) printf("Computing G(1) time: %lf\n", tt);


	for (int k = 1; k < nbr; k++)
	{
		tt = omp_get_wtime();
		SymRecCompress(n, &D[ind(k, n)], ldd, smallsize, eps, "SVD");
		tt = omp_get_wtime() - tt;
		if (compare_str(7, bench, "display")) printf("Compressing D(%d) time: %lf\n", k, tt);

		tt = omp_get_wtime();
		dlacpy("All", &n, &n, &G[ind(k - 1, n)], &ldg, TD1, &ldtd);
		DiagMult(n, TD1, ldtd, &B[ind(k - 1, n)], smallsize);
		tt = omp_get_wtime() - tt;
		if (compare_str(7, bench, "display")) printf("Mult D(%d) time: %lf\n", k, tt);

		tt = omp_get_wtime();
		Add(n, 1.0, &D[ind(k, n)], ldd, -1.0, TD1, ldtd, TD, ldtd, smallsize, eps, "SVD");
		tt = omp_get_wtime() - tt;
		if (compare_str(7, bench, "display")) printf("Add %d time: %lf\n", k, tt);

		tt = omp_get_wtime();
		SymCompRecInv(n, TD, ldtd, &G[ind(k,n) + ldg * 0], ldg, smallsize, eps, "SVD");
		tt = omp_get_wtime() - tt;
		if (compare_str(7, bench, "display")) printf("Computing G(%d) time: %lf\n", k, tt);
		if (compare_str(7, bench, "display")) printf("\n");
	}

	if (compare_str(7, bench, "display"))
	{
		printf("****************************\n");
		printf("End of DirFactFastDiag\n");
		printf("****************************\n");
	}

	free_arr(&TD);
	free_arr(&TD1);
}

void DirSolveFastDiag(int n1, int n2, int n3, double *G, int ldg, double *B, double *f, double *x, double eps, int smallsize)
{
	int n = n1 * n2;
	int nbr = n3;
	int size = n * nbr;
	double *tb = alloc_arr(size);
	double *y = alloc_arr(n);

#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < n; i++)
		tb[i] = f[i];

	for (int k = 1; k < nbr; k++)
	{
		RecMultL(n, 1, &G[ind(k - 1, n) + ldg * 0], ldg, &tb[ind(k - 1, n)], size, y, n, smallsize);	
		DenseDiagMult(n, &B[ind(k - 1, n)], y, y);

#pragma omp parallel for simd schedule(simd:static)
		for (int i = 0; i < n; i++)
			tb[ind(k, n) + i] = f[ind(k, n) + i] - y[i];

	}

	RecMultL(n, 1, &G[ind(nbr - 1, n) + ldg * 0], ldg, &tb[ind(nbr - 1, n)], size, &x[ind(nbr - 1, n)], size, smallsize);

	for (int k = nbr - 2; k >= 0; k--)
	{
		DenseDiagMult(n, &B[ind(k, n)], &x[ind(k + 1, n)], y);

#pragma omp parallel for simd schedule(simd:static)
		for (int i = 0; i < n; i++)
			y[i] = tb[ind(k, n) + i] - y[i];

		RecMultL(n, 1, &G[ind(k, n) + ldg * 0], ldg, y, n, &x[ind(k, n)], size, smallsize);
	}

	free_arr(&tb);
	free_arr(&y);
}


// Рекурсивная функция вычисления DAD, где D - диагональная матрица, а A - сжатая
void DiagMult(int n, double *A, int lda, double *d, int small_size)
{

	if (n <= small_size)     // error 4 - не копировалась матрица в этом случае
	{
#pragma omp parallel for simd schedule(simd:static)
		for (int j = 0; j < n; j++)
			for (int i = 0; i < n; i++)
			{
				A[i + j * lda] *= d[j]; // справа D - каждый j - ый столбец A умножается на d[j]
				A[i + j * lda] *= d[i]; // слева D - каждая строка A умножается на d[j]
			}
		return;
	}
	else
	{
		int n2 = (int)ceil(n / 2.0); // округление в большую сторону
		int n1 = n - n2;

		DiagMult(n1, &A[0 + lda * 0], lda, &d[0], small_size);
		DiagMult(n2, &A[n1 + lda * n1], lda, &d[n1], small_size);

		// D * U - каждая i-ая строка U умножается на элемент вектора d[i]
#pragma omp parallel for simd schedule(simd:static)
		for (int j = 0; j < n1; j++)
			for (int i = 0; i < n2; i++)
				A[i + n1 + lda * (0 + j)] *= d[n1 + i]; // вторая часть массива D

		// VT * D - каждый j-ый столбец умножается на элемент вектора d[j]
#pragma omp parallel for simd schedule(simd:static)
		for (int j = 0; j < n2; j++)
			for (int i = 0; i < n1; i++)
				A[i + 0 + lda * (n1 + j)] *= d[j]; 
		// так так вектора матрицы V из разложения A = U * V лежат в транспонированном порядке,
		// то матрицу D стоит умножать на VT слева
	}
}

void op_mat(int m, int n, double *Y11, double *Y12, int ldy, char sign)
{
	if (sign == '+')
	{
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
#pragma omp simd
			for (int i = 0; i < m; i++)
				Y11[i + ldy * j] += Y12[i + ldy *j];
	}
	else if (sign == '-')
	{
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
#pragma omp simd
			for (int i = 0; i < m; i++)
				Y11[i + ldy * j] -= Y12[i + ldy *j];
	}
	else
	{
		printf("Incorrect sign\n");
	}
}


map<vector<int>,double> dense_to_sparse(int m, int n, double *DD, int ldd, int *i_ind, int *j_ind, double *d)
{
	map<vector<int>, double> SD;
	vector<int> v(2);
	double thresh = 1e-8;
	int k = 0;
	for (int j = 0; j < n; j++)
		for (int i = 0; i < m; i++)
			if (fabs(DD[i + ldd * j]) != 0)
			{
				d[k] = DD[i + ldd * j];
				i_ind[k] = i;
				j_ind[k] = j;

				v[0] = i;
				v[1] = j;
				SD[v] = DD[i + ldd * j];

				k++;
			}

	return SD;
}

void count_dense_elements(int m, int n, double *A, int lda, int& non_zeros)
{
	int k = 0;
#pragma omp parallel for schedule(guided) reduction(+:k)
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (fabs(A[i + lda * j]) != 0)
			{
				k++;
			}
		}
	}
	non_zeros = k;
}

map<vector<int>, double> concat_maps(const map<vector<int>, double>& map1, const map<vector<int>, double>& map2)
{
	map<vector<int>, double> map_res;
	for (const auto& item : map1)
	{
		map_res.insert(item);
	}
	for (const auto& item : map2)
	{
		map_res.insert(item);
	}
	return map_res;
}



void GenSparseMatrix(size_m x, size_m y, size_m z, double *BL, int ldbl, double *A, int lda, double *BR, int ldbr, dcsr* Acsr)
{
	int n = x.n * y.n;
	int size = n * z.n;

	for (int blk = 0; blk < z.n; blk++)
	{
		GenerateDiagonal2DBlock(blk, x, y, z, &A[ind(blk, n)], lda);
		if (blk < z.n - 1)
		{
			Diag(n, &BL[ind(blk, n)], ldbl, 1.0 / (z.h * z.h));
			Diag(n, &BR[ind(blk, n)], ldbr, 1.0 / (z.h * z.h));
		}
	}

	map<vector<int>, double> CSR;
	CSR = block3diag_to_CSR(x.n, y.n, z.n, BL, ldbl, A, lda, BR, ldbr, Acsr);
}

map<vector<int>, double> block3diag_to_CSR(int n1, int n2, int blocks, double *BL, int ldbl, double *A, int lda, double *BR, int ldbr, dcsr* Acsr)
{
	map<vector<int>, double> CSR_A;
	map<vector<int>, double> CSR;
	vector<int> v(2, 0);
	int n = n1 * n2;
	int k = 0;
	double *AR = alloc_arr(n * 3 * n); int ldar = n;
	int non_zeros_on_prev_level = 0;

	for (int blk = 0; blk < blocks; blk++)
	{
		if (blk == 0)
		{
			construct_block_row(n, n, NULL, ldbl, &A[0], lda, &BR[0], ldbr, AR, ldar);
		//	print(n, n, &AR[0 + ldar * n], ldar, "AR");
			CSR_A = dense_to_CSR(n, 2 * n, AR, ldar, &Acsr->ia[0], &Acsr->ja[0], &Acsr->values[0]);
			non_zeros_on_prev_level = CSR_A.size();
		}
		else if (blk == blocks - 1)
		{
			construct_block_row(n, n, &BL[ind(blk - 1, n)], ldbl, &A[ind(blk, n)], lda, NULL, ldbr, AR, ldar);
			//print(n, 2 * n, AR, ldar, "ldar");
			CSR_A = dense_to_CSR(n, 2 * n, AR, ldar, &Acsr->ia[ind(blk, n)], &Acsr->ja[non_zeros_on_prev_level], &Acsr->values[non_zeros_on_prev_level]);
			shift_values(CSR_A, n, &Acsr->ia[ind(blk, n)], non_zeros_on_prev_level, CSR_A.size(), &Acsr->ja[non_zeros_on_prev_level], n * (blk - 1));
		}
		else
		{
			construct_block_row(n, n, &BL[ind(blk - 1, n)], ldbl, &A[ind(blk, n)], lda, &BR[ind(blk, n)], ldbr, AR, ldar);
			CSR_A = dense_to_CSR(n, 3 * n, AR, ldar, &Acsr->ia[ind(blk, n)], &Acsr->ja[non_zeros_on_prev_level], &Acsr->values[non_zeros_on_prev_level]);

			// shift values of arrays according to previous level
			shift_values(CSR_A, n, &Acsr->ia[ind(blk, n)], non_zeros_on_prev_level, CSR_A.size(), &Acsr->ja[non_zeros_on_prev_level], n * (blk - 1));
			non_zeros_on_prev_level += CSR_A.size();
		}
	}

	free(AR);
	return CSR;
}


void print_vec(int size, double *vec1, double *vec2, char *name)
{
	printf("%s\n", name);
	for (int i = 0; i < size; i++)
		printf("%d   %lf   %lf\n", i, vec1[i], vec2[i]);
}

void print_vec(int size, int *vec1, double *vec2, char *name)
{
	printf("%s\n", name);
	for (int i = 0; i < size; i++)
		printf("%d   %d   %lf\n", i, vec1[i], vec2[i]);
}

void print_vec_mat(int m, int n, double *u, int ldu, double *vec, char *mess)
{
	printf("%s\n", mess);
	for (int i = 0; i < m; i++)
	{
		printf("%d ", i);
		for (int j = 0; j < n; j++)
		{
			printf("%5.2lf ", u[i + ldu*j]);
		}
		printf("  %lf\n", vec[i]);
	}

	printf("\n");

}
#endif