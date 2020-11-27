#include "templates.h"
#include "TemplatesForMatrixConstruction.h"
#include "TestSuite.h"


void BCGSTAB(size_m x, size_m y, size_m z, int m, const point source, dtype *x_sol, const dtype* x_orig, const dtype *f, double thresh, double &diff_sol, double beta_eq)
{
#ifdef PRINT
	printf("-------------BCGSTAB-----------\n");
#endif
	// We need to solve iteratively: (I - \delta L * L_0^{-1})w = g

	int size = x.n * y.n * z.n;
	int size2D = x.n * y.n;
	int size_nopml = x.n_nopml * y.n_nopml * z.n_nopml;
	int size2D_nopml = x.n_nopml * y.n_nopml;
	int iterCount = m;
	int iter = 0;
	int ione = 1;
	double norm = 0;
	double norm_f = 0;
	double norm_r0 = 0;
	double beta = 0;
	double Res;
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
	int count = 0;
	int ratio = 0;

#if 1
	FILE *output;
	char str0[255];
	char conv_str[255];
	sprintf(conv_str, "convergence_N%d_PML%d_Lx%d_FREQ%d_SPG%d_BETA%lf_BCG_het", x.n_nopml, x.pml_pts, LENGTH_X, nu, z.spg_pts, beta_eq);
	sprintf(str0, "%s.dat", conv_str);
	//sprintf(str0, "convergence_N%d_Lx%d_FREQ%d_SPG%5.lf_BETA%5.3lf.dat", x.n_nopml, (int)LENGTH_X, (int)nu, z.h * 2 * z.spg_pts, beta_eq);
	output = fopen(str0, "w");
#endif

	//-------- PARDISO ----------
	// Calling the solver
	int mtype = 13;
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

#ifdef PRINT
	printf("-----Step 0. Set sound speed and deltaL-----\n");
#endif
	//	SetSoundSpeed3D(x, y, z, sound3D, source);
	//	SetSoundSpeed2D(x, y, z, sound3D, sound2D, source);
#if 1
	// Gen velocity of sound in 3D domain
	SetSoundSpeed3D(x, y, z, sound3D, source);

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

#ifdef PRINT
	printf("-----Step 1. Memory allocation for 2D problems\n");
#endif

	zcsr *D2csr_zero;
	int non_zeros_in_2Dblock3diag = (x.n + (x.n - 1) * 2) * y.n + 2 * (size2D - x.n);
	int non_zeros_in_2Dblock9diag = (x.n + (x.n - 1) * 2) * y.n + 2 * (size2D - x.n) + 4 * (x.n - 1) * (y.n - 1);
	int non_zeros_in_2Dblock13diag = (x.n + (x.n - 1) * 2 + (x.n - 2) * 2 + (x.n - 3) * 2) * y.n + 2 * (size2D - x.n) + 2 * (size2D - 2 * x.n) + 2 * (size2D - 3 * x.n);


	int non_zeros;

#if 1
	non_zeros = non_zeros_in_2Dblock9diag;

	D2csr_zero = (zcsr*)malloc(sizeof(zcsr));
	D2csr_zero->values = alloc_arr<dtype>(non_zeros);
	D2csr_zero->ia = alloc_arr<int>(size2D + 1);
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
#ifdef PRINT
	printf("SOURCE in 2D WITH PML AT: (%lf, %lf)\n", sourcePML.x, sourcePML.y);
#endif
	double sigma = 0.25;
	double mem_pard = 0;

#ifndef HODLR
	time = omp_get_wtime();
	//GenSparseMatrixOnline2DwithPMLand9Points(-1, x, y, z, D2csr_zero, 0, freqs, sigma);
	//GenSparseMatrixOnline2DwithPML(-1, x, y, D2csr_zero, 0, freqs);
	GenSparseMatrixOnline2DwithPMLand9Pts(-1, x, y, D2csr_zero, 0, freqs);
	//GenSparseMatrixOnline2DwithPMLFast(-1, x, y, D2csr_zero, 0, freqs);
	//GenSparseMatrixOnline2DwithPMLand13Pts(-1, x, y, D2csr_zero, 0, freqs);  // does not work now
	time = omp_get_wtime() - time;
#ifdef PRINT
	printf("time for constructing = %lf sec\n", time);
#endif
	time = omp_get_wtime();
	//TestSymmSparseMatrixOnline2DwithPML(x, y, z, D2csr_zero);
	//TestSymmSparseMatrixOnline2DwithPML9Pts(x, y, z, D2csr_zero);
	time = omp_get_wtime() - time;
#endif

	// Memory for 2D CSR matrix
#ifndef HODLR
	zcsr **D2csr;
	D2csr = (zcsr**)malloc(z.n * sizeof(zcsr*));
#else
	cmnode* **Gstr;
	Gstr = (cmnode***)malloc(z.n * sizeof(cmnode**));
#endif

#ifdef PRINT
	printf("Generating and factorizing matrices for 2D problems...\n");
#endif
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

		//if (kww < ratio * k2)
		if (1)
		{
			dtype kwave_beta2 = k2 * dtype{ 1, beta_eq } -kww;
#ifdef PRINT
			printf("Solved k = %d beta2 = (%lf, %lf)\n", k, kwave_beta2.real(), kwave_beta2.imag());
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

			mem_pard = max(iparm[14], iparm[15] + iparm[16]);

			phase = 22;
			pardiso(&pt[k * 64], &maxfct, &mnum, &mtype, &phase, &size2D, D2csr[k]->values, D2csr[k]->ia, D2csr[k]->ja, &perm[k * size2D], &rhs, &iparm[k * 64], &msglvl, &zdum, &zdum, &error);
			if (error != 0) printf("!!! FACTORIZATION ERROR: %d !!!\n", error);
#endif

			count++;
			continue;

			// источник в каждой задаче в середине 
			//GenSparseMatrixOnline2D("FT", i, x, y, z, Bc_mat, n1, Dc, n1, Bc_mat, n1, D2csr);
		}
		else
		{
			D2csr[k]->solve = 0;
		}
	}
#else
	int smallsize = 20;
	dtype *B = alloc_arr<dtype>((size2D - x.n) * z.n); // for right diagonal
	bool *solves = alloc_arr<bool>(z.n);
	int lwork = x.n * x.n;
	dtype *work = alloc_arr2<dtype>(lwork);

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

		//if (1)
		if (kww < ratio * k2)
		{
			dtype kwave_beta2 = k2 * dtype{ 1, beta_eq } -kww;
			printf("Solved k = %d beta2 = (%lf, %lf)\n", k, kwave_beta2.real(), kwave_beta2.imag());

			// Factorization of matrices
			DirFactFastDiagStructOnline(x, y, Gstr[k], &B[k * (size2D - x.n)], kwave_beta2, work, lwork, thresh, smallsize);
			solves[k] = true;
			count++;

			// источник в каждой задаче в середине 
			//GenSparseMatrixOnline2D("FT", i, x, y, z, Bc_mat, n1, Dc, n1, Bc_mat, n1, D2csr);
		}
		else
		{
			solves[k] = false;
		}
		mem_pard = 0;
	}
#endif

	double mem = 2.0 * non_zeros_in_2Dblock9diag / (1024 * 1024 * 1024);
	mem += double(size2D + 1) / (1024 * 1024 * 1024);
	mem *= count;
	mem += double(z.n * size2D) / (1024 * 1024 * 1024);

	mem *= 8; // bytes
	mem *= 2;
#ifdef PRINT
	printf("Memory for %d 2D matrices: %lf Gb\n", count, mem);
#endif
	mem_pard /= (1024 * 1024);
	mem_pard *= count;
#ifdef PRINT
	printf("Memory for %d PARDISO factor + analysis 2D matrices: %lf Gb\n", count, mem_pard);
#endif
#ifndef PERF
	system("pause");
#endif

	dtype *x_sol_nopml = alloc_arr<dtype>(size_nopml);
	dtype *x_orig_nopml = alloc_arr<dtype>(size_nopml);

	//#define MANUFACTORED_SOLUTION

#ifdef MANUFACTORED_SOLUTION
	dtype *x_wave = alloc_arr<dtype>(size);
	dtype *x_orig_man = alloc_arr<dtype>(size);
	dtype *f_man = alloc_arr<dtype>(size);

	reducePML3D(x, y, z, size, x_orig, size_nopml, x_orig_nopml);

	// original solution with zeros in PML zone
	extendPML3D(x, y, z, size_nopml, x_orig_nopml, size, x_orig_man);

	Multiply3DSparseUsingFT(x, y, z, iparm, perm, pt, D2csr, x_orig_man, x_wave, thresh);
	ApplyCoeffMatrixA_CSR(x, y, z, iparm, perm, pt, D2csr, x_wave, deltaL, f_man, beta_eq, thresh);

	double diff_f = RelError(zlange, size, 1, f_man, f, size, thresh);
	printf("RelError between f original and f manufactored: %e\n", thresh);
	zcopy(&size, f_man, &ione, f, &ione);
#endif

#ifdef PRINT
	printf("-----Step 1. Memory allocation-----\n");
#endif
	// init cond
	dtype *x0 = alloc_arr<dtype>(size);
	dtype *p0 = alloc_arr<dtype>(size);
	dtype *pn = alloc_arr<dtype>(size);
	dtype *s0 = alloc_arr<dtype>(size);
	dtype *yn = alloc_arr<dtype>(size);
	dtype *zn = alloc_arr<dtype>(size);
	dtype *Ayn = alloc_arr<dtype>(size);
	dtype *Azn = alloc_arr<dtype>(size);
	dtype *x_init = alloc_arr<dtype>(size);
	dtype* w = alloc_arr<dtype>(size);
	dtype* g = alloc_arr<dtype>(size);

	// residual vector
	dtype *r0 = alloc_arr<dtype>(size);
	dtype *rn = alloc_arr<dtype>(size);
	dtype *r0st = alloc_arr<dtype>(size);

	// additional vector
	dtype *Ax0_nopml = alloc_arr<dtype>(size_nopml);

	// resid
#ifdef COMP_RESID
	dtype *f_rsd = alloc_arr<dtype>(size);
	dtype *f_rsd_nopml = alloc_arr<dtype>(size_nopml);
#endif

#ifndef HOMO
	dtype *x_sol_prev_nopml = alloc_arr<dtype>(size_nopml);
#endif

	// test2
	double *x_orig_re = alloc_arr<double>(size_nopml);
	double *x_sol_re = alloc_arr<double>(size_nopml);
	double *x_orig_im = alloc_arr<double>(size_nopml);
	double *x_sol_im = alloc_arr<double>(size_nopml);

	// vars
	dtype calpha;
	double dalpha;
	double norm_re;
	double norm_im;

#pragma omp parallel for simd schedule(static)
	for (int i = 0; i < size; i++)
		x_init[i] = 0;

	for (int restart = 0; restart < 1; restart++)
	{
#ifdef PRINT
		printf("------RESTART = %d------\n", restart);
#endif
		// 1. First step. Compute r_0 and its norm

		zcopy(&size, x_init, &ione, x0, &ione);
		nullifyPML3D(x, y, z, size, x0);

		// Multiply matrix A in CSR format by vector x_0 to obtain f1
#ifndef HODLR
		ApplyCoeffMatrixA_CSR(x, y, z, iparm, perm, pt, D2csr, x0, deltaL, w, beta_eq, thresh);
#else
		ApplyCoeffMatrixA_HODLR(x, y, z, Gstr, B, solves, x0, deltaL, w, thresh, smallsize);
#endif

		norm = dznrm2(&size, w, &ione);
		printf("norm ||Ax0|| = %lf\n", norm);

		norm_f = dznrm2(&size, f, &ione);
		printf("norm ||f|| = %lf\n", norm_f);

		zcopy(&size, f, &ione, r0, &ione);
		zaxpy(&size, &mone, w, &ione, r0, &ione); // r0: = f - Ax0

		norm = dznrm2(&size, r0, &ione);

		reducePML3D(x, y, z, size, r0, size_nopml, Ax0_nopml);
		norm = dznrm2(&size_nopml, Ax0_nopml, &ione);
		printf("norm ||Ax0 - f|| = ||r0|| = %e\n", norm);

		zcopy(&size, r0, &ione, p0, &ione);     // p0 = r0
		zcopy(&size, r0, &ione, r0st, &ione);   // r0st = r0
		double dlin = 2.0;
		zdscal(&size, &dlin, r0st, &ione);   // r0st = 2 * r0

		system("pause");
		//norm = RelError(zlange, size, 1, r0, f, size, thresh);
		//printf("r0 = f - Ax0, norm ||r0 - f|| = %lf\n", norm);

		// 2. The main iterations of algorithm
#ifdef PRINT
		printf("-----Step 2. Iterations-----\n");
#endif
		for (int j = 0; j < m; j++)
		{
#ifdef PRINT
			printf("\nIteration = %d\n", j);
#endif
#ifndef HODLR
			ApplyCoeffMatrixA_CSR(x, y, z, iparm, perm, pt, D2csr, p0, deltaL, Ayn, beta_eq, thresh);
			Solve3DSparseUsingFT_CSR(x, y, z, iparm, perm, pt, D2csr, p0, yn, beta_eq, thresh);
#else
			ApplyCoeffMatrixA_HODLR(x, y, z, Gstr, B, solves, &V[ldv * j], deltaL, p0, thresh, smallsize);
#endif
			dtype alpha_n = -ratio_dot(size, 1.0, 1.0, r0st, r0, Ayn);

			// s_n = r_n - alpha_n * Ay_n
			zcopy(&size, r0, &ione, s0, &ione);
			zaxpy(&size, &alpha_n, Ayn, &ione, s0, &ione);

#ifndef HODLR
			ApplyCoeffMatrixA_CSR(x, y, z, iparm, perm, pt, D2csr, s0, deltaL, Azn, beta_eq, thresh);
			Solve3DSparseUsingFT_CSR(x, y, z, iparm, perm, pt, D2csr, s0, zn, beta_eq, thresh);
#else
			ApplyCoeffMatrixA_HODLR(x, y, z, Gstr, B, solves, &V[ldv * j], deltaL, r0, thresh, smallsize);
#endif

			dtype omega_n = -ratio_dot(size, 1.0, 1.0, Azn, s0, Azn);

			// r_{n+1} = s_n - omega_n * Az_n
			zcopy(&size, s0, &ione, rn, &ione);
			zaxpy(&size, &omega_n, Azn, &ione, rn, &ione);

			alpha_n *= -1.0;
			omega_n *= -1.0;

			// x_{n+1} = x_{n} + alpha_n * y_n + omega_n * z_n
			zaxpy(&size, &alpha_n, yn, &ione, x0, &ione);
			zaxpy(&size, &omega_n, zn, &ione, x0, &ione);

			dtype beta_n = ratio_dot(size, alpha_n, omega_n, r0st, rn, r0);

			omega_n *= -1.0;
			zaxpy(&size, &omega_n, Ayn, &ione, p0, &ione);
			zcopy(&size, rn, &ione, pn, &ione);
			zaxpy(&size, &beta_n, p0, &ione, pn, &ione);

			zcopy(&size, pn, &ione, p0, &ione);
			zcopy(&size, rn, &ione, r0, &ione);


			// 5. Check |(I - deltaL * L^{-1}) * x_k - f|
#ifndef HODLR
			//ApplyCoeffMatrixA_CSR(x, y, z, iparm, perm, pt, D2csr, x0, deltaL, g, beta_eq, thresh);
			//Solve3DSparseUsingFT_CSR(x, y, z, iparm, perm, pt, D2csr, g, w, beta_eq, thresh);

			Multiply3DSparseUsingFT(x, y, z, iparm, perm, pt, D2csr, x0, g, thresh);
			ApplyCoeffMatrixA_CSR(x, y, z, iparm, perm, pt, D2csr, g, deltaL, w, beta_eq, thresh);
#else
			ApplyCoeffMatrixA_HODLR(x, y, z, Gstr, B, solves, x0, deltaL, w, thresh, smallsize);
#endif
			zaxpy(&size, &mone, f, &ione, w, &ione); // Ax0: = Ax0 - f

		//	RelRes = dznrm2(&size, w, &ione);
		//	printf("-----------\n");
		//	printf("Residual in 3D with PML |(I - deltaL * L^{-1}) * x_sol - f| = %lf\n", RelRes);
		//	printf("-----------\n");
			reducePML3D(x, y, z, size, w, size_nopml, Ax0_nopml);
			Res = dznrm2(&size_nopml, Ax0_nopml, &ione);
			RelRes = Res / norm_f;

#ifdef PRINT
			printf("-----------\n");
			printf("Residual in 3D phys domain |(I - deltaL * L^{-1}) * x_sol - f| = %e\n", Res);
			printf("Relative residual in 3D phys domain |(I - deltaL * L^{-1}) * x_sol - f| = %e\n", RelRes);
#endif

			// 6. Solve L_0 ^(-1) * x_gmres = x_sol
#ifdef PRINT
			printf("-----Step 5. Solve the last system-----\n");
#endif
#ifndef HODLR
			zcopy(&size, x0, &ione, x_sol, &ione);
			//Solve3DSparseUsingFT_CSR(x, y, z, iparm, perm, pt, D2csr, x0, x_sol, beta_eq, thresh);
#else
			Solve3DSparseUsingFT_HODLR(x, y, z, Gstr, B, solves, x0, x_sol, thresh, smallsize);
#endif

			// 8. Reduce pml
#ifdef HOMO
#ifdef PRINT
			printf("Nullify source...\n"); // comment for printing with gnuplot
#endif
			NullifySource2D(x, y, &x_sol[z.n / 2 * size2D], size2D / 2, 1);
			NullifySource2D(x, y, &x_orig[z.n / 2 * size2D], size2D / 2, 1);
#endif

			reducePML3D(x, y, z, size, x_sol, size_nopml, x_sol_nopml);
			reducePML3D(x, y, z, size, x_orig, size_nopml, x_orig_nopml);

#ifdef HOMO
			norm = RelError(zlange, size_nopml, 1, x_sol_nopml, x_orig_nopml, size_nopml, thresh);
#ifdef PRINT
			printf("Residual in 3D phys domain |x_sol - x_orig| / |x_orig| = %e\n", norm);
			printf("-----------\n");
#endif
			fprintf(output, "%d %e %lf\n", j, RelRes, norm);
			//fprintf(output, "%d %17.15lf\n", j, norm);

			check_norm_result2(x.n_nopml, y.n_nopml, z.n_nopml, j, 0, 2 * z.spg_pts * z.h, x_orig_nopml, x_sol_nopml, x_orig_re, x_orig_im, x_sol_re, x_sol_im);

			norm_re = RelError(dlange, size_nopml, 1, x_sol_re, x_orig_re, size_nopml, thresh);
			norm_im = RelError(dlange, size_nopml, 1, x_sol_im, x_orig_im, size_nopml, thresh);
			norm = RelError(zlange, size_nopml, 1, x_sol_nopml, x_orig_nopml, size_nopml, thresh);
#ifdef PRINT
			printf("norm_re = %lf\n", norm_re);
			printf("norm_im = %lf\n", norm_im);
			printf("norm = %lf\n", norm);
#endif
#else
			diff_sol = RelError(zlange, size_nopml, 1, x_sol_nopml, x_sol_prev_nopml, size_nopml, thresh);
			MultVectorConst<dtype>(size_nopml, x_sol_nopml, 1.0, x_sol_prev_nopml);
			printf("norm |u_k+1 - u_k|= %e\n", diff_sol);
			fprintf(output, "%d %e %e %lf\n", j, Res, RelRes, diff_sol);
#endif
			if (Res < RES_EXIT) break;
#ifdef PRINT
			printf("--------------------------------------------------------------------------------\n");
#endif
		}
		// For the next step
		zcopy(&size, g, &ione, x_init, &ione);
#endif

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

	FILE *conv = fopen("conv.plt", "w");

	fprintf(conv, "set term png font \"Times - Roman, 16\" \n \
		set output '%s.png' \n \
		plot '%s.dat' u 1:2 w linespoints pt 7 pointsize 1 notitle", conv_str, conv_str);

	fclose(conv);

	system("conv.plt");

	free(D2csr_zero->values);
	free(D2csr_zero->ia);
	free(D2csr_zero->ja);
	free(D2csr_zero);

	for (int k = 0; k < z.n; k++)
	{
		phase = -1;
		pardiso(&pt[k * 64], &maxfct, &mnum, &mtype, &phase, &size2D, D2csr[k]->values, D2csr[k]->ia, D2csr[k]->ja, &perm[k * size2D], &rhs, &iparm[k * 64], &msglvl, &zdum, &zdum, &error);
		free(D2csr[k]->values);
		free(D2csr[k]->ia);
		free(D2csr[k]->ja);
		free(D2csr[k]);
	}
	free(D2csr);
	free(freqs);
	free(x_sol_nopml);
	free(x_orig_nopml);

	free(p0);
	free(pn);
	free(s0);
	free(yn);
	free(zn);
	free(Ayn);
	free(Azn);
	free(g);

	// residual vector
	free(rn);
	free(r0st);

	free_arr(w);
	free_arr(r0);
	free_arr(x0);
	free_arr(x_init);
	free_arr(Ax0_nopml);
	free_arr(x_orig_re);
	free_arr(x_orig_im);
	free_arr(x_sol_re);
	free_arr(x_sol_im);

	free(deltaL);
	free(sound3D);
	free(sound2D);

	free(iparm);
	free(perm);
	free(pt);
}