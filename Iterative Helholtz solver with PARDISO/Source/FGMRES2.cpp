#include "templates.h"
#include "TemplatesForMatrixConstruction.h"
#include "TestSuite.h"

void FGMRES(size_m x, size_m y, size_m z, int m, int rest, const point source, dtype *x_sol, dtype* x_orig, const dtype *f, double thresh, double &diff_sol, double beta_eq)
{
	printf("-------------FGMRES-----------\n");

	// We need to solve iteratively: (I - \delta L * L_0^{-1})w = g

	int size = x.n * y.n * z.n;
	int size2D = x.n * y.n;
	int size_nopml = x.n_nopml * y.n_nopml * z.n_nopml;
	int size2D_nopml = x.n_nopml * y.n_nopml;
	int iterCount = m;
	int iter = 0;
	int ione = 1;
	double norm = 0;
	double norm_r0 = 0;
	double beta = 0;
	double RelRes;
	int i1, j1, k1;

	int nrhs = 1;
	int info = 0;
	int col_min;
	int row_min;
	dtype work_size;
	dtype done = { 1.0, 0.0 };
	dtype mone = { -1.0, 0.0 };

	int* freqs = alloc_arr<int>(size);
	dtype *deltaL = alloc_arr<dtype>(size);
	dtype *sound3D = alloc_arr<dtype>(size);
	dtype *sound2D = alloc_arr<dtype>(size2D);
	dtype zdum;
	double time;
	double k2 = double(kk) * double(kk);
	double kww;
	double norm_f;
	double Res;
	int count = 0;
	int ratio = 0;

	//float *sound3D_OT = ReadData();

	FILE *output;
	char str0[255];
	sprintf(str0, "convergence_N%d_PML%d_Lx%d_FREQ%d_SPG%6.lf_BETA%5.3lf_FGMRES.dat", x.n_nopml, x.pml_pts, (int)LENGTH_X, (int)nu, z.h * 2 * z.spg_pts, beta_eq);
	output = fopen(str0, "w");


	//-------- PARDISO ----------
	// Calling the solver
	int mtype = 13; // complex and unsymmetric
	//int mtype = 6; // complex and symmetric
	int *iparm = alloc_arr<int>(64 * z.n);
	int *perm = alloc_arr<int>(size2D * z.n);
	size_t *pt = alloc_arr<size_t>(64 * z.n);

	for (int i = 0; i < z.n; i++)
		pardisoinit(&pt[i * 64], &mtype, &iparm[i * 64]);

	int maxfct = 1;
	int mnum = 1;
	int rhs = 1;
	int msglvl = 0;
	int error = 0;
	int phase; // analisys + factorization
	//--------------------------

	printf("-----Step 0. Set sound speed and deltaL-----\n");
	//	SetSoundSpeed3D(x, y, z, sound3D, source);
	//	SetSoundSpeed2D(x, y, z, sound3D, sound2D, source);
#if 1
	// Gen velocity of sound in 3D domain
	//SetSoundSpeed3D(x, y, z, sound3D, source);
	SetSoundSpeed3DFromFile(x, y, z, sound3D, source);

	// Extension of 3D sound speed to PML zone
	HeteroSoundSpeed3DExtensionToPML(x, y, z, sound3D);

	// Gen velocity of sound in 3D domain
	SetSoundSpeed2D(x, y, z, sound3D, sound2D, source);

	// Extension of sound speed to PML zone
	HeteroSoundSpeed2DExtensionToPML(x, y, sound2D);

	// Gen DeltaL function
	GenerateDeltaL(x, y, z, sound3D, sound2D, deltaL, beta_eq);

	char str1[255] = "sound_speed2D";
	//output(str1, false, x, y, z, sound3D, deltaL);
	//	output2D(str1, false, x, y, sound2D, sound2D);

	char str2[255] = "sound_speed_deltaL";
	//	output(str2, false, x, y, z, sound3D, deltaL)

	free(sound3D);

	printf("-----Step 1. Memory allocation for 2D problems\n");

	zcsr *D2csr_zero;
	size_t non_zeros = 0;
#if DIAG_PATTERN == 3
	size_t non_zeros_in_2Dblock3diag = ((size_t)x.n + (x.n - 1) * 2) * y.n + 2 * (size2D - x.n);
	non_zeros = non_zeros_in_2Dblock3diag;
#elif DIAG_PATTERN == 9
	size_t non_zeros_in_2Dblock9diag = ((size_t)x.n + (x.n - 1) * 2) * y.n + 2 * (size2D - x.n) + 4 * (x.n - 1) * (y.n - 1);
	non_zeros = non_zeros_in_2Dblock9diag;
#else
	size_t non_zeros_in_2Dblock13diag = ((size_t)x.n + (x.n - 1) * 2 + (x.n - 2) * 2 + (x.n - 3) * 2) * y.n + 2 * (size2D - x.n) + 2 * (size2D - 2 * x.n) + 2 * (size2D - 3 * x.n);
	non_zeros = non_zeros_in_2Dblock13diag;
#endif

#if 1
	D2csr_zero = (zcsr*)malloc(sizeof(zcsr));
	D2csr_zero->values = alloc_arr<dtype>(non_zeros);
	D2csr_zero->ia = alloc_arr<int>((size_t)size2D + 1);
	D2csr_zero->ja = alloc_arr<int>(non_zeros);
	D2csr_zero->ia[size2D] = non_zeros + 1;
	D2csr_zero->non_zeros = non_zeros;
#else
	D2csr_zero = (zcsr*)malloc(sizeof(zcsr));
	D2csr_zero->values = alloc_arr<dtype>(non_zeros_in_2Dblock13diag);
	D2csr_zero->ia = alloc_arr<int>(size2D + 1);
	D2csr_zero->ja = alloc_arr<int>(non_zeros_in_2Dblock13diag);
	D2csr_zero->ia[size2D] = non_zeros_in_2Dblock13diag + 1;
	D2csr_zero->non_zeros = non_zeros_in_2Dblock13diag;

	printf("Non_zeros: %d\n", D2csr_zero->non_zeros);

	non_zeros = non_zeros_in_2Dblock13diag;
#endif
	point sourcePML = { x.l / 2.0, y.l / 2 };
	printf("SOURCE in 2D WITH PML AT: (%lf, %lf)\n", sourcePML.x, sourcePML.y);

	double sigma = 0.25;
	double mem_pard = 0;

	//rubish
	//GenSparseMatrixOnline2DwithPMLand9Points(-1, x, y, z, D2csr_zero, 0, freqs, sigma);
   
	printf("Generating sparse matrix...\n");
#ifndef HODLR
	time = omp_get_wtime();
#if DIAG_PATTERN == 3
	//GenSparseMatrixOnline2DwithPML(-1, x, y, D2csr_zero, 0, freqs);
	GenSparseMatrixOnline2DwithPMLFast(-1, x, y, D2csr_zero, 0, freqs);
	if (x.n_nopml <= 150)
		TestSymmSparseMatrixOnline2DwithPML(x, y, z, D2csr_zero);
#elif DIAG_PATTERN == 9
	GenSparseMatrixOnline2DwithPMLand9Pts(-1, x, y, D2csr_zero, 0, freqs);
	if (x.n_nopml <= 150)
		TestSymmSparseMatrixOnline2DwithPML9Pts(x, y, z, D2csr_zero);
#else
	GenSparseMatrixOnline2DwithPMLand13Pts(-1, x, y, D2csr_zero, 0, freqs);  // does not work now
#endif
	time = omp_get_wtime() - time;
	printf("time for constructing and test = %lf sec\n", time);
	system("pause");

	zcsr **D2csr;
	D2csr = (zcsr**)malloc(z.n * sizeof(zcsr*));
#else
	//try to make unsymmetric solver cumnode
	ntype* **Gstr;
	Gstr = (ntype***)malloc(z.n * sizeof(ntype**));
#endif

	printf("Generating and factorizing matrices for 2D problems...\n");

#ifndef HODLR
	for (int k = 0; k < z.n; k++)
	{
#define MKL_FFT

#ifdef MKL_FFT
		if (k < z.n / 2 - 1)
		{
			kww = 4.0 * double(PI) * double(PI) * k * k / (z.l * z.l);
		}
		else
		{
			kww = 4.0 * double(PI) * double(PI) * (z.n - k) * (z.n - k) / (z.l * z.l);
		}
#else
		kww = 4.0 * double(PI)* double(PI) * (i - nhalf) * (i - nhalf) / (z.l * z.l);
#endif

		D2csr[k] = (zcsr*)malloc(sizeof(zcsr));

		if (nu == 2) ratio = 15;
		else ratio = 3;

#ifdef SAVE_MEM
		if (kww < ratio * k2)
#else
		if (1)
	    //if (k < 5)
		//if (k == z.n / 2 || k == z.n / 2 - 1 || kk == z.n / 2 + 1)
#endif
		{
			dtype kwave_beta2 = k2 * dtype{ 1, beta_eq } -kww;
#ifdef PRINT
			printf("Solved k = %d beta2 = (%lf, %lf). ", k, kwave_beta2.real(), kwave_beta2.imag());
#endif
			D2csr[k]->values = alloc_arr<dtype>(non_zeros);
			D2csr[k]->ia = alloc_arr<int>(size2D + 1);
			D2csr[k]->ja = alloc_arr<int>(non_zeros);

#if 0
			D2csr[k]->ia[size2D] = non_zeros_in_2Dblock3diag + 1;
			D2csr[k]->non_zeros = non_zeros_in_2Dblock3diag;

			GenSparseMatrixOnline2DwithPML(-1, x, y, z, D2csr[k], kwave_beta2, freqs);
			D2csr[k]->solve = 1;

#else
			Copy2DCSRMatrix(size2D, non_zeros, D2csr_zero, D2csr[k]);
			D2csr[k]->solve = 1;

			dtype kxyz;

#ifdef HOMO
			//GenSparseMatrixOnline2DwithPML(k, x, y, D2csr[k], kwave_beta2, freqs);
			GenSparseMatrixOnline2DwithPMLand9Pts(k, x, y, D2csr[k], kwave_beta2, freqs);
#else
			for (int j = 0; j < y.n; j++)
				for (int i = 0; i < x.n; i++)
				{
					kxyz = double(omega) / sound2D[i + j * x.n];
					kwave_beta2 = kxyz * kxyz * dtype{ 1, beta_eq } -kww;
					D2csr[k]->values[freqs[i + x.n * j]] += kwave_beta2;
				}
#endif
#endif

#if 1
			// Factorization of matrices

			phase = 11;
			pardiso(&pt[k * 64], &maxfct, &mnum, &mtype, &phase, &size2D, D2csr[k]->values, D2csr[k]->ia, D2csr[k]->ja, &perm[k * size2D], &rhs, &iparm[k * 64], &msglvl, &zdum, &zdum, &error);
			if (error != 0) printf("!!! ANALYSIS ERROR: %d !!!\n", error);

			double mem_cur = max(iparm[14], iparm[15] + iparm[16]);
			mem_pard += mem_cur;
			printf("Mem = %lf Gb. Full_memory = %lf Gb\n", mem_cur / ((size_t)1024 * 1024), mem_pard / ((size_t)1024 * 1024));

			phase = 22;
			pardiso(&pt[k * 64], &maxfct, &mnum, &mtype, &phase, &size2D, D2csr[k]->values, D2csr[k]->ia, D2csr[k]->ja, &perm[k * size2D], &rhs, &iparm[k * 64], &msglvl, &zdum, &zdum, &error);
			if (error != 0) printf("!!! FACTORIZATION ERROR: %d !!!\n", error);
#endif

			count++;
			continue;

			//system("pause");
			// �������� � ������ ������ � �������� 
			//GenSparseMatrixOnline2D("FT", i, x, y, z, Bc_mat, n1, Dc, n1, Bc_mat, n1, D2csr);
		}
		else
		{
			D2csr[k]->solve = 0;
		}
	}
#else
	//int smallsize = x.n / 2 - 10;
	int smallsize = 50;
	dtype *B = alloc_arr<dtype>(((size_t)size2D - x.n) * z.n); // for right diagonal
	bool *solves = alloc_arr<bool>(z.n);
	int lwork_HODLR = x.n * x.n;
	dtype *work_HODLR = alloc_arr<dtype>(lwork_HODLR);

	printf("Size of 1D block: %d x %d\nSmallsize = %d\n", x.n, x.n, smallsize);
	system("pause");

	time = omp_get_wtime();
	for (int k = 0; k < z.n; k++)
	{
#define MKL_FFT

#ifdef MKL_FFT
		if (k < z.n / 2 - 1)
		{
			kww = 4.0 * double(PI) * double(PI) * k * k / (z.l * z.l);
		}
		else
		{
			kww = 4.0 * double(PI) * double(PI) * (z.n - k) * (z.n - k) / (z.l * z.l);
		}
#else
		kww = 4.0 * double(PI)* double(PI) * (i - nhalf) * (i - nhalf) / (z.l * z.l);
#endif

		if (nu == 2) ratio = 15;
		else ratio = 3;

		if (1)
		//if (k < 5)
		//if (kww < ratio * k2)
		{
			// only for homogeneous domain
#ifdef HOMO
			dtype kwave_beta2 = k2 * dtype{ 1, beta_eq } -kww;
			printf("Solved k = %d beta2 = (%lf, %lf)\n", k, kwave_beta2.real(), kwave_beta2.imag());
#else
			dtype kwave_beta2 = 0;
#endif
			// Factorization of matrices
			DirFactFastDiagStructOnline(x, y, Gstr[k], &B[k * (size2D - x.n)], sound2D, kww, beta_eq, work_HODLR, lwork_HODLR, thresh, smallsize);
			double mem_cur = Test_NonZeroElementsInFactors(x, y, Gstr[k], &B[k * (size2D - x.n)], thresh, smallsize);

			solves[k] = true;
			count++;

			mem_cur += (size2D - x.n); // 5th diagonal B
			mem_cur *= 8; // bytes
			mem_cur *= 2;
			mem_pard += mem_cur;

			printf("Solved k = %d. ", k);
			printf("Mem = %lf Gb. Full_memory = %lf Gb\n", mem_cur / ((size_t)1024 * 1024 * 1024), mem_pard / ((size_t)1024 * 1024 * 1024));

			// �������� � ������ ������ � �������� 
			//GenSparseMatrixOnline2D("FT", i, x, y, z, Bc_mat, n1, Dc, n1, Bc_mat, n1, D2csr);
		}
		else
		{
			solves[k] = false;
		}
	}
	free(work_HODLR);
	time = omp_get_wtime() - time;
	printf("Time of factorization: %lf\n", time);
	system("pause");
#endif

	double mem = 2.0 * non_zeros / ((size_t)1024 * 1024 * 1024);
	mem += ((double)size2D + 1) / ((size_t)1024 * 1024 * 1024);
	mem *= count;
	mem += ((double)z.n * size2D) / ((size_t)1024 * 1024 * 1024);

	mem *= 8; // bytes
	mem *= 2;

	printf("Memory for %d 2D matrices: %lf Gb\n", count, mem);

	mem_pard /= ((size_t)1024 * 1024);
	printf("Memory for %d PARDISO factor + analysis 2D matrices: %lf Gb\n", count, mem_pard);

#ifndef PERF
	system("pause");
#endif

	printf("-----Step 2. Memory allocation for FGMRES-----\n");
	// init cond
	dtype *x0 = alloc_arr<dtype>(size);

	// matrix of Krylov basis
	dtype* V = alloc_arr<dtype>((size_t)size * (m + 1)); int ldv = size;
	dtype* w = alloc_arr<dtype>(size);

	// residual vector
	dtype *work_vec = alloc_arr<dtype>(size);

	// Hessenberg matrix
	dtype *H = alloc_arr<dtype>((m + 1) * m); int ldh = m + 1;
	dtype *Hgels = alloc_arr<dtype>((m + 1) * m);

	// the vector of right-hand side for the system with Hessenberg matrix
	dtype *eBeta = alloc_arr<dtype>(m + 1); int ldb = m + 1;

	dtype *Ax0_nopml = alloc_arr<dtype>(size);

	// resid
#ifdef COMP_RESID
	dtype *f_rsd = alloc_arr<dtype>(size);
	dtype *f_rsd_nopml = alloc_arr<dtype>(size_nopml);
#endif

	dtype *x_sol_nopml = alloc_arr<dtype>(size_nopml);

#ifndef HOMO
	dtype *x_sol_prev_nopml = alloc_arr<dtype>(size_nopml);
#else
	dtype *x_orig_nopml = alloc_arr<dtype>(size_nopml);
#endif

//#define TEST_L0

	// test
#ifdef TEST_L0
	dtype *f_sol = alloc_arr<dtype>(size);
	dtype *x_gmres_nopml = alloc_arr<dtype>(size_nopml);
	dtype *f_sol_nopml = alloc_arr<dtype>(size_nopml);
#endif

#ifdef HOMO
	// test2
	double *x_orig_re = alloc_arr<double>(size_nopml);
	double *x_sol_re = alloc_arr<double>(size_nopml);
	double *x_orig_im = alloc_arr<double>(size_nopml);
	double *x_sol_im = alloc_arr<double>(size_nopml);
#endif

	// vars
	dtype calpha;
	double dalpha;
	double norm_re;
	double norm_im;

	int lwork = -1;
	int row_max = m + 1;
	int col_max = m;

	zgels("no", &row_max, &col_max, &nrhs, Hgels, &ldh, eBeta, &ldb, &work_size, &lwork, &info);

	lwork = (int)work_size.real();
	dtype *work = alloc_arr<dtype>(lwork);

#pragma omp parallel for simd schedule(static)
	for (int i = 0; i < size; i++)
		x0[i] = 0;

	for (int restart = 0; restart < rest; restart++)
	{
		printf("------RESTART = %d------\n", restart);

		// 1. First step. Compute r_0 and its norm
		// Multiply matrix A in CSR format by vector x_0 to obtain f1
#ifndef HODLR
		ApplyCoeffMatrixA_CSR(x, y, z, iparm, perm, pt, D2csr, x0, deltaL, w, beta_eq, thresh);
#else
		ApplyCoeffMatrixA_HODLR(x, y, z, Gstr, B, solves, x0, deltaL, w, thresh, smallsize);
#endif
		norm = dznrm2(&size, w, &ione);
		printf("norm ||Ax0|| = %e\n", norm);

		norm_f = dznrm2(&size, f, &ione);
		printf("norm ||f|| = %e\n", norm_f);

		zcopy(&size, f, &ione, work_vec, &ione);
		zaxpy(&size, &mone, w, &ione, work_vec, &ione); // r0: = f - Ax0

		norm = dznrm2(&size, work_vec, &ione);
		printf("norm ||r0|| = %e\n", norm);

		NormalizeVector(size, work_vec, &V[ldv * 0], beta); // v + size * j = &v[ldv * j]

		TestNormalizedVector(size, &V[0], thresh);

		// 2. The main iterations of algorithm
		printf("-----Step 3. Iterations-----\n");
		for (size_t j = 0; j < (size_t)m; j++)
		{
			printf("---------------------\nIteration = %d\n---------------------\n", restart * m + j);
			time = omp_get_wtime();
			// Compute w[j] := A * v[j]
#ifndef HODLR
			ApplyCoeffMatrixA_CSR(x, y, z, iparm, perm, pt, D2csr, &V[ldv * j], deltaL, w, beta_eq, thresh);
#else
			ApplyCoeffMatrixA_HODLR(x, y, z, Gstr, B, solves, &V[ldv * j], deltaL, w, thresh, smallsize);
#endif

			for (size_t i = 0; i <= j; i++)
			{
				// H[i + ldh * j] = (w_j * v_i) 
				zdotc(&H[i + ldh * j], &size, w, &ione, &V[ldv * i], &ione);
				//H[i + ldh * j] = zdot(size, w, &V[ldv * i]);

				//w[j] = w[j] - H[i][j]*v[i]
				calpha = -H[i + ldh * j];
				zaxpy(&size, &calpha, &V[ldv * i], &ione, w, &ione);
			}

			H[j + 1 + ldh * j] = dznrm2(&size, w, &ione);

			// Check the convergence to the exact solution
			if (abs(H[j + 1 + ldh * j]) < thresh)
			{
				iterCount = j + 1;
				printf("Break! value: %lf < thresh: %lf\n", H[j + 1 + ldh * j].real(), thresh);
				break;
			}

			// If not, construct the new vector of basis
			zcopy(&size, w, &ione, &V[ldv * (j + 1)], &ione);
			dalpha = 1.0 / H[j + 1 + ldh * j].real();
			zdscal(&size, &dalpha, &V[ldv * (j + 1)], &ione);

			TestNormalizedVector(size, &V[ldv * (j + 1)], thresh);
#if 0
			for (int i = 0; i <= j; i++)
			{
				TestOrtogonalizedVectors(size, &V[ldv * (j + 1)], &V[ldv * i], thresh);
			}
#endif

			// 3. Solving least squares problem to compute y_k
			// for x_k = x_0 + V_k * y_k
			printf("-----Step 3. LS problem-----\n");

			printf("size of basis: %d\n", iterCount);

			// Set eBeta
			for (int i = 0; i < m + 1; i++)
				eBeta[i] = 0;

			eBeta[0] = beta;

			// Set working H because it is destroyed after GELS
#pragma omp for simd
			for (int i = 0; i < m * (m + 1); i++)
				Hgels[i] = H[i];

			// Query
			row_min = j + 2;
			col_min = j + 1;

			// Run
			zgels("no", &row_min, &col_min, &nrhs, Hgels, &ldh, eBeta, &ldb, work, &lwork, &info);

			RelRes = dznrm2(&col_min, eBeta, &ione);
			printf("norm y_k[%d] = %e\n", j, RelRes);

			// 4. Multiplication x_k = x_0 + V_k * y_k
			printf("-----Step 4. Computing x_k-----\n");

			zgemv("no", &size, &col_min, &done, V, &ldv, eBeta, &ione, &done, x0, &ione);

			// 5. Check |(I - deltaL * L^{-1}) * x_k - f|
#ifndef HODLR
			ApplyCoeffMatrixA_CSR(x, y, z, iparm, perm, pt, D2csr, x0, deltaL, w, beta_eq, thresh);
#else
			ApplyCoeffMatrixA_HODLR(x, y, z, Gstr, B, solves, x0, deltaL, w, thresh, smallsize);
#endif
			zaxpy(&size, &mone, f, &ione, w, &ione); // Ax0: = Ax0 - f

		//	RelRes = dznrm2(&size, w, &ione);
		//	printf("-----------\n");
		//	printf("Residual in 3D with PML |(I - deltaL * L^{-1}) * x_sol - f| = %lf\n", RelRes);
		//	printf("-----------\n");

			//reducePML3D(x, y, z, size, w, size_nopml, work_vec);
			//Res = dznrm2(&size_nopml, work_vec, &ione);
			reducePML3D(x, y, z, size, w, size_nopml, Ax0_nopml);
			Res = dznrm2(&size_nopml, Ax0_nopml, &ione);
			RelRes = Res / norm_f;

			printf("-----------\n");
			printf("Residual in 3D phys domain |(I - deltaL * L^{-1}) * x_sol - f| = %e\n", Res);
			printf("Relative residual in 3D phys domain |(I - deltaL * L^{-1}) * x_sol - f| = %e\n", RelRes);

			// 6. Solve L_0 ^(-1) * x_gmres = x_sol
			printf("-----Step 5. Solve the last system-----\n");
#ifndef HODLR
			Solve3DSparseUsingFT_CSR(x, y, z, iparm, perm, pt, D2csr, x0, x_sol, beta_eq, thresh);
#else
			Solve3DSparseUsingFT_HODLR(x, y, z, Gstr, B, solves, x0, x_sol, thresh, smallsize);
#endif

#if defined(TEST_L0) && !defined(HODLR)
			// 7. Test ||L0 * u_sol - w|| / ||w||
			Multiply3DSparseUsingFT(x, y, z, iparm, perm, pt, D2csr, x_sol, f_sol, thresh);

		//	zero_out<dtype>(size_nopml, f_sol_nopml);
		//	zero_out<dtype>(size_nopml, x_gmres_nopml);
			reducePML3D(x, y, z, size, f_sol, size_nopml, f_sol_nopml);
			reducePML3D(x, y, z, size, x0, size_nopml, x_gmres_nopml);

			norm = RelError(zlange, size_nopml, 1, f_sol_nopml, x_gmres_nopml, size_nopml, thresh);

			printf("Residual in 3D phys domain |L0 * u_sol - x_gmres| / |x_gmres| = %e", norm);
			if (norm < thresh) printf(" - TEST PASSED!\n");
			else printf(" - TEST FAILED!!!\n");
			printf("-----------\n");
#endif

			// 8. Reduce pml
#ifdef HOMO
			printf("Nullify source...\n"); // comment for printing with gnuplot
			NullifySource2D(x, y, &x_sol[z.n / 2 * size2D], size2D / 2, 1);
			NullifySource2D(x, y, &x_orig[z.n / 2 * size2D], size2D / 2, 1);
#endif

			reducePML3D(x, y, z, size, x_sol, size_nopml, x_sol_nopml);
#ifdef HOMO
			reducePML3D(x, y, z, size, x_orig, size_nopml, x_orig_nopml);
			norm = RelError(zlange, size_nopml, 1, x_sol_nopml, x_orig_nopml, size_nopml, thresh);
			printf("Residual in 3D phys domain |x_sol - x_orig| / |x_orig| = %lf\n", norm);
			printf("-----------\n");

			fprintf(output, "%d %e %e %lf\n", j, Res, RelRes, norm);

			check_norm_result2(x.n_nopml, y.n_nopml, z.n_nopml, j, 0, 2 * z.spg_pts * z.h, x_orig_nopml, x_sol_nopml, x_orig_re, x_orig_im, x_sol_re, x_sol_im);

			norm_re = RelError(dlange, size_nopml, 1, x_sol_re, x_orig_re, size_nopml, thresh);
			norm_im = RelError(dlange, size_nopml, 1, x_sol_im, x_orig_im, size_nopml, thresh);
			norm = RelError(zlange, size_nopml, 1, x_sol_nopml, x_orig_nopml, size_nopml, thresh);

			printf("norm_re = %lf\n", norm_re);
			printf("norm_im = %lf\n", norm_im);
			printf("norm = %lf\n", norm);
#else
			diff_sol = RelError(zlange, size_nopml, 1, x_sol_nopml, x_sol_prev_nopml, size_nopml, thresh);
			MultVectorConst<dtype>(size_nopml, x_sol_nopml, 1.0, x_sol_prev_nopml);
			printf("norm |u_k+1 - u_k|= %e\n", diff_sol);
			fprintf(output, "%d %e %e %lf\n", restart * m + j, Res, RelRes, diff_sol);
			//if (diff_sol < RES_EXIT) break;
#endif
		    if (RelRes < REL_RES_EXIT) break;

			printf("--------------------------------------------------------------------------------\n");
			time = omp_get_wtime() - time;
			printf("Time of iteration = %lf\n", time);
		}
#endif

		FILE* conv = fopen("conv.plt", "w");
		fprintf(conv, "set term png font \"Times - Roman, 16\" \n \
		set output '%s.png' \n \
		plot '%s' u 1:3 w linespoints pt 7 pointsize 1 notitle", str0, str0);

		fclose(conv);

		system("conv.plt");

#ifdef COMP_RESID
		//	if (j == m - 1)
		{
			FILE* out = fopen("ResidualVector.txt", "w");
			for (int i = 0; i < size_nopml; i++)
			{
				take_coord3D(x.n_nopml, y.n_nopml, z.n_nopml, i, i1, j1, k1);
				fprintf(out, "%d %d %d %lf %lf\n", i1, j1, k1, f_rsd_nopml[i].real(), f_rsd_nopml[i].imag());
			}
			fclose(out);
		}
#endif
	} // End of iterations
	free_arr(work);
	fclose(output);
#ifdef COMP_RESID
	ComputeResidual(x, y, z, (double)kk, x_sol, f, f_rsd, RelRes);

	printf("-----------\n");
	printf("Residual in 3D with PML |A * x_sol - f| = %e\n", RelRes);
	printf("-----------\n");

	reducePML3D(x, y, z, size, f_rsd, size_nopml, f_rsd_nopml);

	RelRes = dznrm2(&size_nopml, f_rsd_nopml, &ione);

	printf("-----------\n");
	printf("Residual in 3D phys domain |A * x_sol - f| = %e\n", RelRes);
	printf("-----------\n");

	free_arr(f_rsd);
	free_arr(f_rsd_nopml);
#endif

#ifdef TEST_L0
	free_arr(f_sol_nopml);
	free_arr(f_sol);
	free_arr(x_gmres_nopml);
#endif

	free_arr(H);
	free_arr(Hgels);
	free_arr(w);
	free_arr(V);
	free_arr(x0);
	free_arr(work_vec);
#ifdef HOMO
	free_arr(x_orig_nopml);
	free_arr(x_orig_re);
	free_arr(x_orig_im);
	free_arr(x_sol_re);
	free_arr(x_sol_im);
#endif
}
