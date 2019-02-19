#include "templates.h"
#include "TestSuite.h"
#include "TestFramework.h"
#include "TemplatesForMatrixConstruction.h"

/***************************************
Source file contains tests for testing
functionalites, implemented in
functions.cpp and BinaryTrees.cpp.

The interface is declared in TestSuite.h
****************************************/

void Test_TransferBlock3Diag_to_CSR(size_m x, size_m y, size_m z, ccsr* Dcsr, dtype* x_orig, dtype *f, double eps)
{
	int n = x.n * y.n;
	int size = n * z.n;
	double RelRes = 0;
	dtype *g = alloc_arr<dtype>(size);
	ResidCSR(x, y, z, Dcsr, x_orig, f, g, RelRes);

#ifdef OUTPUT
	FILE* fout;
	fout = fopen("resid.dat", "w");

	for (int i = 0; i < size; i++)
	{
		if (abs(g[i]) > 0.1) fprintf(fout, "%d %12.10lf %12.10lf  false\n", i, g[i].real(), g[i].imag());
		else fprintf(fout, "%d %12.10lf %12.10lf\n", i, g[i].real(), g[i].imag());
	}

	fclose(fout);
#endif


	if (RelRes < eps) printf("A * u_ex = f. Norm %10.8e < eps %10.8lf: PASSED\n", RelRes, eps);
	else printf("A * u_ex = f. Norm %10.8lf > eps %10.8e : FAILED\n", RelRes, eps);

	free_arr(g);
}

void TestInverseTraversal(size_m x, size_m y, size_m z, const point source, const dtype *x_sol, const dtype *f_orig, double thresh)
{
	// Test for postorder traversal of the algorithm

	int size = x.n * y.n * z.n;
	int size2D = x.n * y.n;
	int size_nopml = x.n_nopml * y.n_nopml * z.n_nopml;
	int size2D_nopml = x.n_nopml * y.n_nopml;
	int iter = 0;
	int ione = 1;
	double norm = 0;
	double norm_r0 = 0;
	double beta = 0;
	double RelRes;
	int i1, j1, k1;

	int nrhs = 1;
	int lwork = -1;
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
	dtype* work;
	dtype zdum;
	double time;
	double k2 = double(kk) * (kk);
	double kww;
	int count = 0;
	int ratio = 0;

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

	printf("-----Step 0. Set sound speed and deltaL-----\n");

	// Gen velocity of sound in 3D domain
	SetSoundSpeed3D(x, y, z, sound3D, source);

	// Gen velocity of sound in 3D domain
	SetSoundSpeed2D(x, y, z, sound3D, sound2D, source);

	char str1[255] = "sound_speed2D";
	//	output(str1, false, x, y, z, sound3D, deltaL);
	//	output2D(str1, false, x, y, sound2D, sound2D);

		// Gen DeltaL function
	GenerateDeltaL(x, y, z, sound3D, sound2D, deltaL);

	char str2[255] = "sound_speed_deltaL";
	//	output(str2, false, x, y, z, sound3D, deltaL);

	free_arr(sound2D);
	free_arr(sound3D);

	printf("-----Step 1. Memory allocation for 2D problems\n");

	ccsr *D2csr_zero;
	int non_zeros_in_2Dblock3diag = (x.n + (x.n - 1) * 2) * y.n + 2 * (size2D - x.n);
	int non_zeros_in_2Dblock9diag = (x.n + (x.n - 1) * 2) * y.n + 2 * (size2D - x.n) + 4 * (x.n - 1) * (y.n - 1);
#ifdef TEST_HELM_1D
	non_zeros_in_2Dblock3diag += y.n * 2;
	non_zeros_in_2Dblock3diag += x.n * 2;
#endif

	int non_zeros;

#if 1
	D2csr_zero = (ccsr*)malloc(sizeof(ccsr));
	D2csr_zero->values = alloc_arr<dtype>(non_zeros_in_2Dblock3diag);
	D2csr_zero->ia = alloc_arr<int>(size2D + 1);
	D2csr_zero->ja = alloc_arr<int>(non_zeros_in_2Dblock3diag);
	D2csr_zero->ia[size2D] = non_zeros_in_2Dblock3diag + 1;
	D2csr_zero->non_zeros = non_zeros_in_2Dblock3diag;

	non_zeros = non_zeros_in_2Dblock3diag;
#else
	D2csr_zero = (ccsr*)malloc(sizeof(ccsr));
	D2csr_zero->values = alloc_arr<dtype>(non_zeros_in_2Dblock9diag);
	D2csr_zero->ia = alloc_arr<int>(size2D + 1);
	D2csr_zero->ja = alloc_arr<int>(non_zeros_in_2Dblock9diag);
	D2csr_zero->ia[size2D] = non_zeros_in_2Dblock9diag + 1;
	D2csr_zero->non_zeros = non_zeros_in_2Dblock9diag;

	non_zeros = non_zeros_in_2Dblock9diag;
#endif

	point sourcePML = { x.l / 2.0, y.l / 2 };
	printf("SOURCE in 2D WITH PML AT: (%lf, %lf)\n", sourcePML.x, sourcePML.y);

	//double sigma = 0.25;
	//printf("Sigma coefficient = %lf\n", sigma);

	time = omp_get_wtime();
	//GenSparseMatrixOnline2DwithPMLand9Points(-1, x, y, z, D2csr_zero, 0, freqs, sigma);
	GenSparseMatrixOnline2DwithPML(-1, x, y, D2csr_zero, 0, freqs);
	time = omp_get_wtime() - time;
	printf("time for constructing = %lf sec\n", time);


	// Memory for 2D CSR matrix
	ccsr **D2csr;
	D2csr = (ccsr**)malloc(z.n * sizeof(ccsr*));

	printf("Generating and factorizing matrices for 2D problems...\n");

	for (int k = 0; k < z.n; k++)
	{
#define MKL_FFT

#ifdef MKL_FFT
		if (k < z.n / 2 - 1)
		{
			kww = 4.0 * PI * PI * k * k / (z.l * z.l);
		}
		else
		{
			kww = 4.0 * PI * PI * (z.n - k) * (z.n - k) / (z.l * z.l);
		}
#else
		kww = 4.0 * PI * PI * (i - nhalf) * (i - nhalf) / (z.l * z.l);
#endif

		D2csr[k] = (ccsr*)malloc(sizeof(ccsr));
		dtype kwave_beta2 = k2 * dtype{ 1, beta_eq } - kww;


		if (nu == 2) ratio = 15;
		else ratio = 3;

		//	if (kww < ratio * k2)
		if (1)
		{

#ifdef TEST_HELM_1D
			non_zeros_in_2Dblock3diag += y.n * 2;
			non_zeros_in_2Dblock3diag += x.n * 2;
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

#pragma omp parallel for simd schedule(static)
			for (int i = 0; i < size; i++)
				D2csr[k]->values[freqs[i]] += kwave_beta2;
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

	dtype *g = alloc_arr<dtype>(size);
	dtype *f_sol = alloc_arr<dtype>(size);

	// L_0 * u = g
	Multiply3DSparseUsingFT(x, y, z, iparm, perm, pt, D2csr, x_sol, g, thresh);

	for (int k = 0; k < z.n; k++)
	{
#if 1
		// Factorization of matrices

		phase = 11;
		pardiso(&pt[k * 64], &maxfct, &mnum, &mtype, &phase, &size2D, D2csr[k]->values, D2csr[k]->ia, D2csr[k]->ja, &perm[k * size2D], &rhs, &iparm[k * 64], &msglvl, &zdum, &zdum, &error);
		if (error != 0) printf("!!! ANALYSIS ERROR: %d !!!\n", error);

		phase = 22;
		pardiso(&pt[k * 64], &maxfct, &mnum, &mtype, &phase, &size2D, D2csr[k]->values, D2csr[k]->ia, D2csr[k]->ja, &perm[k * size2D], &rhs, &iparm[k * 64], &msglvl, &zdum, &zdum, &error);
		if (error != 0) printf("!!! FACTORIZATION ERROR: %d !!!\n", error);
#endif
	}

	// (I - deltaL * L^{-1}) * g = f_sol
	ApplyCoeffMatrixA(x, y, z, iparm, perm, pt, D2csr, g, deltaL, f_sol, thresh);

	dtype *f_orig_nopml = alloc_arr<dtype>(size_nopml);
	dtype *f_sol_nopml = alloc_arr<dtype>(size_nopml);

	reducePML3D(x, y, z, size, f_orig, size_nopml, f_orig_nopml);
	reducePML3D(x, y, z, size, f_sol, size_nopml, f_sol_nopml);

	// f_sol vs f_orig
	norm = AbsError(zlange, size_nopml, 1, f_sol_nopml, f_orig_nopml, size_nopml, thresh);

	if (norm < thresh) printf("Abs norm %12.10e < eps %12.10e: PASSED\n", norm, thresh);
	else printf("Abs norm %12.10lf > eps %12.10lf : FAILED\n", norm, thresh);

	FILE *check_rhs;
	check_rhs = fopen("RHS.dat", "w");

	for (int i = 0; i < size_nopml; i++)
		if (abs(f_sol_nopml[i]) > 0.000001 || abs(f_orig_nopml[i]) > 0.000001) fprintf(check_rhs, "%d %lf %lf %lf %lf\n", i, f_orig_nopml[i].real(), f_orig_nopml[i].imag(), f_sol_nopml[i].real(), f_sol_nopml[i].imag());

	fclose(check_rhs);

	free_arr(f_sol);
	free_arr(g);
	free_arr(f_orig_nopml);
	free_arr(f_sol_nopml);
}

void TestSymmSparseMatrixOnline2DwithPML(size_m x, size_m y, size_m z, ccsr *Acsr)
{
	double size = x.n * y.n;
	int nelems, nelems2 = 0;
	int ione = 1;
	int j1, k1;
	int j2, k2;

	//-----------------------//
	int error;
	sparse_struct* handle = (sparse_struct*)malloc(sizeof(sparse_struct));

	sparse_matrix_checker_init(handle);

	handle->n = size;
	handle->csr_ia = Acsr->ia;
	handle->csr_ja = Acsr->ja;
	handle->indexing = MKL_ONE_BASED;
	handle->matrix_structure = MKL_GENERAL_STRUCTURE;
	handle->matrix_format = MKL_CSR;
	handle->message_level = MKL_PRINT;
	handle->print_style = MKL_C_STYLE;

	error = sparse_matrix_checker(handle);

	printf("Error of constructing: %d\n", error);

	//------------------------------//

	matrix *mat2D = (matrix*)malloc(Acsr->non_zeros * sizeof(matrix));
	dtype *diff = (dtype*)malloc(Acsr->non_zeros * sizeof(dtype));

#if 1
	for (int l1 = 0; l1 < size; l1++)
	{
		for (int l2 = 0; l2 < size; l2++)
		{
			take_coord2D(x.n, y.n, l1, j1, k1);
			take_coord2D(x.n, y.n, l2, j2, k2);

			dtype alp = alpha(x, j1) * alpha(y, k1);

			if (l1 == l2)
			{
				mat2D[nelems].val = Acsr->values[nelems] / alp;
				mat2D[nelems].i = l1;
				mat2D[nelems].j = l2;
				nelems++;
			}
			else if (l1 == l2 - 1 && (l1 + 1) % x.n != 0)
			{
				mat2D[nelems].val = Acsr->values[nelems] / alp;
				mat2D[nelems].i = l1;
				mat2D[nelems].j = l2;
				nelems++;
			}
			else if (l1 == l2 + 1 && l1 % x.n != 0)
			{
				mat2D[nelems].val = Acsr->values[nelems] / alp;
				mat2D[nelems].i = l1;
				mat2D[nelems].j = l2;
				nelems++;
			}
			else if (l1 == l2 - x.n)
			{
				mat2D[nelems].val = Acsr->values[nelems] / alp;
				mat2D[nelems].i = l1;
				mat2D[nelems].j = l2;
				nelems++;
			}
			else if (l1 == l2 + x.n)
			{
				mat2D[nelems].val = Acsr->values[nelems] / alp;
				mat2D[nelems].i = l1;
				mat2D[nelems].j = l2;
				nelems++;
			}

		}
	}
#endif

	for (int k = 0; k < nelems; k++)
		for(int l = 0; l < nelems; l++)
			if (mat2D[l].i == mat2D[k].j && mat2D[l].j == mat2D[k].i)
			{
				diff[k] = mat2D[l].val - mat2D[k].val;
				nelems2++;
				//printf("val[%d][%d] = %lf vs val[%d][%d] = %lf\n", mat2D[l].i, mat2D[l].j, mat2D[l].val.real(), mat2D[k].i, mat2D[k].j, mat2D[k].val.real());
				break;
			}

	printf("Norm of SYMMETRY = %e, counts: %d vs %d\n", dznrm2(&Acsr->non_zeros, diff, &ione), nelems, nelems2);

	free(mat2D);
	free(diff);
}


void Test2DLaplaceLevander4th()
{
	printf("------------------\n--------Test Laplace--------\n---------------\n");
	size_m x, y;

	x.pml_pts = y.pml_pts = 4;
	x.spg_pts = y.spg_pts = 0;

	double norm, prev = 1;
	int dist = 1;

	for (int n = 7; n <= 7; n *= 2)
	{
		x.n = n - 1 + 2 * x.pml_pts;
		y.n = n - 1 + 2 * y.pml_pts;

		x.n_nopml = x.n - 2 * x.pml_pts;
		y.n_nopml = y.n - 2 * y.pml_pts;

		x.l = dist + (double)(2 * x.pml_pts * dist) / (x.n_nopml + 1);
		y.l = dist + (double)(2 * y.pml_pts * dist) / (y.n_nopml + 1);

		x.h = x.l / (x.n + 1);
		y.h = y.l / (y.n + 1);

		norm = Test2DLaplaceLevander4thKernel(x, y);
		printf("Size = %d, norm = %e, scale = %lf\n", n, norm, prev / norm);
		prev = norm;

		x.pml_pts *= 2;
		y.pml_pts *= 2;
	}
}

dtype Test2DLaplaceExact(size_m xx, size_m yy, double x, double y, point src)
{
	//return x * x - y * y;
	//return sin(2 * PI * x) * sin(2 * PI * y);
#if 0
	double sigma = 10;
	return exp(-((x - src.x) * (x - src.x) + (y - src.y) * (y - src.y)) / (sigma * sigma));
#else
	// Helmholtz + correct PML
	//return dtype{ x,  100.0 * x * x * x / (3.0 * double(omega)) } * dtype{ y, 100.0 * y * y * y / (3.0 * double(omega)) } ;
#endif
	// zero boundary
	return x * (x - xx.l) * y * (y - yy.l) ;
}

#if 0
dtype Test2DLaplaceRHS(size_m xx, size_m yy, double x, double y, point src, dtype k2)
{
	//return 0;
	//return -8.0 * PI * PI * sin(2 * PI * x) * sin(2 * PI * y);
	double sigma = 10;

	// Laplace
	//return Test2DLaplaceExact(x, y, src) * 4.0 * 
		//(((x - src.x) * (x - src.x) + (y - src.y) * (y - src.y)) / pow(sigma, 4) - 1.0 / pow(sigma, 2));

#if 0
	// Helmholtz
//	return Test2DLaplaceExact(x, y, src) * 4.0 *
//		(((x - src.x) * (x - src.x) + (y - src.y) * (y - src.y)) / pow(sigma, 4) - 1.0 / pow(sigma, 2))
//		+ k2 * Test2DLaplaceExact(x, y, src);
#else
	// Helmholtz + correct PML
	if ((x < xx.h * xx.pml_pts && y < yy.h * yy.pml_pts) || (x < xx.h * xx.pml_pts && y >= yy.l - yy.h * yy.pml_pts) ||
		(x >= xx.l - xx.h * xx.pml_pts && y < yy.h * yy.pml_pts) || (x >= xx.l - xx.h * xx.pml_pts && y >= yy.l - yy.h * yy.pml_pts))
	{
		// 4 corners
		return k2 * Test2DLaplaceExact(xx, yy, x, y, src);
	}
	else if (x < xx.h * xx.pml_pts || x >= xx.l - xx.h * xx.pml_pts)
	{
		return dtype{ 0, 200.0 * y / double(omega) } * dtype{ x, 100.0 * x * x * x / (3.0 * double(omega)) } + k2 * Test2DLaplaceExact(xx, yy, x, y, src);
	}
	else if (y < yy.h * yy.pml_pts || y >= yy.l - yy.h * yy.pml_pts)
	{
	    return dtype{ 0, 200.0 * x / double(omega) } * dtype{ y, 100.0 * y * y * y / (3.0 * double(omega)) } + k2 * Test2DLaplaceExact(xx, yy, x, y, src);
	}
	else
	{
		return dtype{ 0,  200.0 * x / double(omega) } * (dtype{ y, 100.0 * y * y * y / (3.0 * double(omega)) } ) +
			   dtype{ 0,  200.0 * y / double(omega) } * (dtype{ x, 100.0 * x * x * x / (3.0 * double(omega)) } )
			  + k2 * Test2DLaplaceExact(xx, yy, x, y, src);

	//	return dtype{ 0, 400.0 * x * y  } * dtype{ 3.0 * omega, 50.0 * (x * x + y * y) } / (3.0 * omega * omega) +
	//		k2 * Test2DLaplaceExact(xx, yy, x, y, src);

	//	return dtype{ 0, 3 * omega * 400.0 * x * y / (3.0 * omega * omega) } - (double)20000 * x * y * y * y / (3.0 * omega * omega) - (double)20000 * y * x * x * x / (3.0 * omega * omega);

	}
#endif
}
#else
dtype Test2DLaplaceRHS(size_m xx, size_m yy, double x, double y, point src, dtype k2)
{
	//return 0;
	//return -8.0 * PI * PI * sin(2 * PI * x) * sin(2 * PI * y);
	double sigma = 10;

	// Laplace
	//return Test2DLaplaceExact(x, y, src) * 4.0 * 
		//(((x - src.x) * (x - src.x) + (y - src.y) * (y - src.y)) / pow(sigma, 4) - 1.0 / pow(sigma, 2));

	// Helmholtz
//	return Test2DLaplaceExact(x, y, src) * 4.0 *
//		(((x - src.x) * (x - src.x) + (y - src.y) * (y - src.y)) / pow(sigma, 4) - 1.0 / pow(sigma, 2))
//		+ k2 * Test2DLaplaceExact(x, y, src);

	// Helmholtz + correct PML
	if ((x < xx.h * xx.pml_pts && y < yy.h * yy.pml_pts) || (x < xx.h * xx.pml_pts && y >= yy.l - yy.h * yy.pml_pts) ||
		(x >= xx.l - xx.h * xx.pml_pts && y < yy.h * yy.pml_pts) || (x >= xx.l - xx.h * xx.pml_pts && y >= yy.l - yy.h * yy.pml_pts))
	{
		// 4 corners
		return y * (y - yy.l) * alpha(xx, x / xx.h) * 2.0 * double(omega) * dtype { double(omega), 100.0 * x * (xx.l - x) } / (dtype{ double(omega), 100.0 * x * x } *dtype{ double(omega), 100.0 * x * x }) +
			   x * (x - xx.l) * alpha(yy, y / yy.h) * 2.0 * double(omega) * dtype { double(omega), 100.0 * y * (yy.l - y) } / (dtype{ double(omega), 100.0 * y * y } *dtype{ double(omega), 100.0 * y * y }) +
				 k2 * Test2DLaplaceExact(xx, yy, x, y, src);
	}
	else if (x < xx.h * xx.pml_pts || x >= xx.l - xx.h * xx.pml_pts)
	{
		return  y * (y - yy.l) * alpha(xx, x / xx.h) * 2.0 * double(omega) * dtype { double(omega), 100.0 * x * (xx.l - x) } / (dtype{ double(omega), 100.0 * x * x } *dtype{ double(omega), 100.0 * x * x })
					+ 2.0 * y * (y - yy.l) + k2 * Test2DLaplaceExact(xx, yy, x, y, src);
	}
	else if (y < yy.h * yy.pml_pts || y >= yy.l - yy.h * yy.pml_pts)
	{
		return  x * (x - xx.l) * alpha(yy, y / yy.h) * 2.0 * double(omega) * dtype { double(omega), 100.0 * y * (yy.l - y) } / (dtype{ double(omega), 100.0 * y * y } *dtype{ double(omega), 100.0 * y * y })
					+ 2.0 * x * (x - xx.l) + k2 * Test2DLaplaceExact(xx, yy, x, y, src);
	}
	else
	{
		return 2.0 * x * (x - xx.l) + 2.0 * y * (y - yy.l) + k2 * Test2DLaplaceExact(xx, yy, x, y, src);
	}
}
#endif


double Test2DLaplaceLevander4thKernel(size_m x, size_m y)
{
	// 2D test
	printf("-----------------TEST FOR h = %lf----------------\n", x.h);
	int size2D = x.n * y.n;
	int size2D_nopml = x.n_nopml * y.n_nopml;
	int non_zeros_in_2Dblock3diag = (x.n + (x.n - 1) * 2) * y.n + 2 * (size2D - x.n);
	int non_zeros_in_2Dblock13diag = (x.n + (x.n - 1) * 2 + (x.n - 2) * 2 + (x.n - 3) * 2) * y.n + 2 * (size2D - x.n) + 2 * (size2D - 2 * x.n) + 2 * (size2D - 3 * x.n);
	//int non_zeros_in_2Dblock13diag = (x.n + (x.n - 1) * 2) * y.n + 2 * (size2D - x.n);
	int ione = 1;
	int zero = 0;
	int itwo = 2;

	dtype *x_sol = alloc_arr<dtype>(size2D);
	dtype *x_orig = alloc_arr<dtype>(size2D);
	dtype *f = alloc_arr<dtype>(size2D);
	dtype *g = alloc_arr<dtype>(size2D);

	dtype *x_sol_ex_nopml = alloc_arr<dtype>(size2D_nopml);
	dtype *x_sol_prd_nopml = alloc_arr<dtype>(size2D_nopml);


	ccsr *Acsr = (ccsr*)malloc(sizeof(ccsr));
	Acsr->values = alloc_arr<dtype>(non_zeros_in_2Dblock3diag);
	Acsr->ia = alloc_arr<int>(size2D + 1);
	Acsr->ja = alloc_arr<int>(non_zeros_in_2Dblock3diag);
	Acsr->ia[size2D] = non_zeros_in_2Dblock3diag + 1;
	Acsr->non_zeros = non_zeros_in_2Dblock3diag;

	printf("Non_zeros = %d\n", Acsr->non_zeros);

	int* freqs = alloc_arr<int>(size2D);

	double norm;
	double thresh = 10e-6;

	//-------- PARDISO ----------
	int mtype = 13;
	int *iparm = alloc_arr<int>(64);
	int *perm = alloc_arr<int>(size2D);
	size_t *pt = alloc_arr<size_t>(64);

	pardisoinit(pt, &mtype, iparm);

	int maxfct = 1;
	int mnum = 1;
	int rhs = 1;
	int msglvl = 0;
	int error = 0;
	int phase; // analisys + factorization + solution
	int info;

	point src = { x.l / 2, y.l / 2 };
	//double k = 0.02;
	double k = 0;
	dtype k_beta2 = k * k * dtype{ 1.0, beta_eq };

	printf("Lx = %lf, Ly = %lf\n", x.l, y.l);
	printf("Source point Laplace: x = %lf, y = %lf\n", src.x, src.y);

	// Set original solution
	for (int j = 0; j < y.n; j++)
		for (int i = 0; i < x.n; i++)
			x_orig[i + x.n * j] = Test2DLaplaceExact(x, y, x.h * (i + 1), y.h * (j + 1), src);

	// Set right-hand-side
	for (int j = 0; j < y.n; j++)
		for (int i = 0; i < x.n; i++)
			f[i + x.n * j] = Test2DLaplaceRHS(x, y, x.h * (i + 1), y.h * (j + 1), src, k_beta2);

	FILE* out = fopen("RHStest.dat", "w+");
	for (int i = 0; i < size2D; i++)
		fprintf(out, "%d %lf %lf\n", i, f[i].real(), f[i].imag());

	fclose(out);

#if 1
    // для 5-ти точечного шаблона с ненулевыми граничными условиями
	for (int j = 0; j < y.n; j++)
	{
		f[j * x.n + 0] -= Test2DLaplaceExact(x, y, 0, y.h * (j + 1), src) / (x.h * x.h);
		f[j * x.n + x.n - 1] -= Test2DLaplaceExact(x, y, x.l, y.h * (j + 1), src) / (x.h * x.h);
	}

	for (int i = 0; i < x.n; i++)
	{
		f[0 * x.n + i] -= Test2DLaplaceExact(x, y, x.h * (i + 1), 0, src) / (y.h * y.h);
		f[(y.n - 1) * x.n + i] -= Test2DLaplaceExact(x, y, x.h * (i + 1), y.l, src) / (y.h * y.h);
	}
#endif

	FILE* out2 = fopen("RHStest_bound.dat", "w+");
	for (int i = 0; i < size2D; i++)
		fprintf(out2, "%d %lf %lf\n", i, f[i].real(), f[i].imag());

	fclose(out2);

	// Fill matrix
	//GenSparseMatrixOnline2DwithPMLand13Pts(-1, x, y, Acsr, k_beta2, freqs);
	GenSparseMatrixOnline2DwithPML(-1, x, y, Acsr, k_beta2, freqs);

	// Check residual norm
	ResidCSR2D(x, y, Acsr, x_orig, f, g, norm);
	printf("Resid norm |Au_ex - f| = %e\n", norm);

	dtype* A = alloc_arr<dtype>(size2D * size2D); int lda = size2D;

	system("pause");
	// CSR to dense
	printf("matrix gen\n");
	int count1 = 0, count2 = 0;
	int vals_count = 0;

#if 1

	for (int i = 0; i < size2D; i++)
	{
		int vals_in_row = Acsr->ia[i + 1] - Acsr->ia[i];
		for (int j = 0; j < vals_in_row; j++)
		{
			A[i + lda * (Acsr->ja[vals_count + j] - 1)] = Acsr->values[vals_count + j];
		}
		vals_count += vals_in_row;
	}

	if (vals_count != Acsr->non_zeros) printf("!!!Error!!! CSR to DENSE!\n");

#else
	int inf[6] = { 1, 0, 1, 2, Acsr->non_zeros, 7 };
	mkl_zdnscsr(inf, &size2D, &size2D, A, &lda, Acsr->values, Acsr->ja, Acsr->ia, &info);
#endif

	fprint(size2D, size2D, A, lda, "LaplacePML.dat");

	// Solution by PARDISO
	printf("pardiso solve\n");
	phase = 13;
	pardiso(pt, &maxfct, &mnum, &mtype, &phase, &size2D, Acsr->values, Acsr->ia, Acsr->ja, perm, &rhs, iparm, &msglvl, f, x_sol, &error);
	
	if (error != 0) printf("pardiso error code = %d\n", error);

	//norm = dznrm2(&size2D, x_sol, &ione);
	norm = RelError(zlange, size2D, 1, x_sol, x_orig, size2D, thresh);
	printf("Pardiso norm = %e\n", norm);

	int* ipiv = alloc_arr<int>(size2D);
	for (int i = 0; i < size2D; i++)
		x_sol[i] = f[i];

	zgetrf(&size2D, &size2D, A, &lda, ipiv, &info);
	printf("getrf error code = %d\n", info);

	dtype *work = alloc_arr<dtype>(2 * size2D);
	double *rwork = alloc_arr<double>(2 * size2D);
	double anorm = zlange("1", &size2D, &size2D, A, &lda, NULL);
	double rcond = 0;
	zgecon("1", &size2D, A, &lda, &anorm, &rcond, work, rwork, &info);
	printf("cond number = %e\n", rcond);

	zgetrs("no", &size2D, &ione, A, &lda, ipiv, x_sol, &lda, &info);
	printf("getrs error code = %d\n", info);

	norm = RelError(zlange, size2D, 1, x_sol, x_orig, size2D, thresh);
	printf("LAPACK norm = %e\n", norm);

	// Reduce PML
//	reducePML2D(x, y, size2D, x_sol, size2D_nopml, x_sol_ex_nopml);
//	reducePML2D(x, y, size2D, x_orig, size2D_nopml, x_sol_prd_nopml);

	// Check accuracy
	reducePML2D(x, y, size2D, x_orig, size2D_nopml, x_sol_ex_nopml);
	reducePML2D(x, y, size2D, x_sol, size2D_nopml, x_sol_prd_nopml);

	norm = RelError(zlange, size2D_nopml, 1, x_sol_prd_nopml, x_sol_ex_nopml, size2D_nopml, thresh);
	//norm = RelError(zlange, size2D, 1, x_sol, x_orig, size2D, thresh);

#if 1
	// Output
	char *str1, *str2, *str3;
	str1 = alloc_arr<char>(255);
	str2 = alloc_arr<char>(255);
	str3 = alloc_arr<char>(255);
	sprintf(str1, "Charts2D_v5/model_pml_%lf_%lf__%lf", k_beta2.real(), k_beta2.imag(), x.h);
	sprintf(str2, "Charts2D_v5/model_pml_ex_%lf_%lf__%lf", k_beta2.real(), k_beta2.imag(), x.h);
	sprintf(str3, "Charts2D_v5/model_pml_pard_%lf_%lf__%lf", k_beta2.real(), k_beta2.imag(), x.h);

	bool pml_flag = 0;
	output2D(str1, pml_flag, x, y, x_orig, x_sol);

	gnuplot2D(str1, str2, pml_flag, 3, x, y);
	gnuplot2D(str1, str3, pml_flag, 5, x, y);
#endif

	free_arr(x_sol);
	free_arr(x_orig);
	free_arr(f);
	free_arr(g);

	free_arr(x_sol_ex_nopml);
	free_arr(x_sol_prd_nopml);
	free_arr(Acsr->values);
	free_arr(Acsr->ia);
	free_arr(Acsr->ja);
	free_arr(Acsr);
	free_arr(A);

	return norm;
}


void Test2DHelmholtzLevander4th()
{
	printf("------------------\n--------Test Hekmholtz-------\n---------------\n");
	size_m x, y;

	x.pml_pts = y.pml_pts = 20;
	x.spg_pts = y.spg_pts = 0;

	double norm, prev = 1;

	for (int n = 50; n <= 400; n *= 2)
	{
		x.n = n - 1 + 2 * x.pml_pts;
		y.n = n - 1 + 2 * y.pml_pts;

		x.n_nopml = x.n - 2 * x.pml_pts;
		y.n_nopml = y.n - 2 * y.pml_pts;

		x.l = LENGTH_X + (double)(2 * x.pml_pts * LENGTH_X) / (x.n_nopml + 1);
		y.l = LENGTH_Y + (double)(2 * y.pml_pts * LENGTH_Y) / (y.n_nopml + 1);

		x.h = x.l / (x.n + 1);  // x.n + 1 grid points of the whole domain
		y.h = y.l / (y.n + 1);  // x.n - 1 - inner points

		norm = Test2DHelmholtzLevander4thKernel(x, y);
		printf("Size = %d, norm = %e, scale = %lf\n", n, norm, prev / norm);
		prev = norm;
		x.pml_pts *= 2;
		y.pml_pts *= 2;
	}
}


double Test2DHelmholtzLevander4thKernel(size_m x, size_m y)
{
	printf("-----------Test 2D Helmholtz--------\n");
	// Calling the solver
	int size2D = x.n * y.n;
	int size2D_nopml = x.n_nopml * y.n_nopml;
	int mtype = 13;
	int *iparm = alloc_arr<int>(64);
	int *perm = alloc_arr<int>(size2D);
	size_t *pt = alloc_arr<size_t>(64);

	printf("pardisoinit...\n");
	pardisoinit(pt, &mtype, iparm);

	int maxfct = 1;
	int mnum = 1;
	int phase = 13;
	int rhs = 1;
	int msglvl = 0;
	int error = 0;

	// Memory for 2D CSR matrix
	ccsr *D2csr;
	int non_zeros_in_2Dblock3diag = (x.n + (x.n - 1) * 2) * y.n + 2 * (size2D - x.n);
	int non_zeros_in_2Dblock9diag = (x.n + (x.n - 1) * 2) * y.n + 2 * (size2D - x.n) + 4 * (x.n - 1) * (y.n - 1);
	int non_zeros_in_2Dblock13diag = (x.n + (x.n - 1) * 2 + (x.n - 2) * 2 + (x.n - 3) * 2) * y.n + 2 * (size2D - x.n) + 2 * (size2D - 2 * x.n) + 2 * (size2D - 3 * x.n);

	int non_zeros = non_zeros_in_2Dblock13diag;

	D2csr = (ccsr*)malloc(sizeof(ccsr));
	D2csr->values = alloc_arr<dtype>(non_zeros);
	D2csr->ia = alloc_arr<int>(size2D + 1);
	D2csr->ja = alloc_arr<int>(non_zeros);
	D2csr->ia[size2D] = non_zeros + 1;
	D2csr->non_zeros = non_zeros;

	point sourcePML = { x.l / 2.0, y.l / 2.0 };

	double lambda = (double)(c_z) / nu;
	double ppw = lambda / x.h;

	printf("ppw = %lf\n", ppw);
	printf("Lx = %lf, Ly = %lf\n", x.l, y.l);
	printf("PML_x = %lf, PML_y = %lf\n", x.pml_pts * x.h, y.pml_pts * y.h);
	printf("Hx = %lf, Hy = %lf\n", x.h, y.h);
	printf("SOURCE in 2D WITH PML AT: (%lf, %lf)\n", sourcePML.x, sourcePML.y);
	double k = (double)kk;
	int nhalf = y.n / 2;
	int src = 0;

	char *str1, *str2, *str3;
	str1 = alloc_arr<char>(255);
	str2 = alloc_arr<char>(255);
	str3 = alloc_arr<char>(255);
	bool pml_flag = false;


	int count = 0;
	int i = nhalf;

	dtype alpha_k;
	dtype *x_sol_ex = alloc_arr<dtype>(size2D);
	dtype *x_sol_prd = alloc_arr<dtype>(size2D);
	dtype *f2D = alloc_arr<dtype>(size2D);
	int *freqs = alloc_arr<int>(size2D);

	dtype *x_sol_ex_nopml = alloc_arr<dtype>(size2D_nopml);
	dtype *x_sol_prd_nopml = alloc_arr<dtype>(size2D_nopml);


	double norm = 0;
	double thresh = 10e-6;

	//double ppw = c / nu / x.h;

	double kww = 4.0 * double(PI) * double(PI) * (i - nhalf) * (i - nhalf) / (y.l * y.l);
	double kwave2 = k * k - kww;
	dtype kwave_beta2 = dtype{ (k * k - kww), k * k * beta_eq };

	// источник в каждой задаче в середине 
	printf("Gen Matrix for kwave2 = (%lf, %lf)\n", kwave_beta2.real(), kwave_beta2.imag());
	GenSparseMatrixOnline2DwithPMLand13Pts(-1, x, y, D2csr, kwave_beta2, freqs);

	// Gen RHS and exact solution; check residual |A * u - f|
	printf("Gen RHS and check Residual\n");
	GenRHSandSolution2DComplexWaveNumber(x, y, D2csr, x_sol_ex, f2D, kwave_beta2, sourcePML, src);

	// normalization of rhs
	printf("Pardiso solve\n");
	pardiso(pt, &maxfct, &mnum, &mtype, &phase, &size2D, D2csr->values, D2csr->ia, D2csr->ja, perm, &rhs, iparm, &msglvl, f2D, x_sol_prd, &error);

	double eps = 0.01; // 1 percent

	sprintf(str1, "Charts2D_v3/model_pml_%lf_%lf__%lf", kwave_beta2.real(), kwave_beta2.imag(), x.h);
	sprintf(str2, "Charts2D_v3/model_pml_ex_%lf_%lf__%lf", kwave_beta2.real(), kwave_beta2.imag(), x.h);
	sprintf(str3, "Charts2D_v3/model_pml_pard_%lf_%lf__%lf", kwave_beta2.real(), kwave_beta2.imag(), x.h);

	pml_flag = false;

#if 0
	for (int j = 0; j < y.n; j++)
		for (int i = 0; i < x.n; i++)
			if (x_sol_ex[i + x.n * j].real() < -1.0) printf("i = %d j = %d, l = %d, lx = %lf, ly = %lf\n",
				i, j, i + x.n * j, i * x.h, j * y.h);
#endif

	NullifySource2D(x, y, x_sol_ex, src, 5);
	NullifySource2D(x, y, x_sol_prd, src, 5);
	x_sol_ex[src] = x_sol_prd[src] = 0;

	output2D(str1, pml_flag, x, y, x_sol_ex, x_sol_prd);

	gnuplot2D(str1, str2, pml_flag, 3, x, y);
	gnuplot2D(str1, str3, pml_flag, 5, x, y);

	reducePML2D(x, y, size2D, x_sol_ex, size2D_nopml, x_sol_ex_nopml);
	reducePML2D(x, y, size2D, x_sol_prd, size2D_nopml, x_sol_prd_nopml);

	norm = RelError(zlange, size2D_nopml, 1, x_sol_prd_nopml, x_sol_ex_nopml, size2D_nopml, thresh);
	printf("Norm 2D solution ||x_sol - x_ex|| / ||x_ex|| = %lf\n", norm);

	//	check_exact_sol_Hankel(alpha_k, kwave2, x, y, x_sol_prd_nopml, thresh);

	return norm;
}

#if 0
void Test_PMLBlock3Diag_in_CSR(size_m x, size_m y, size_m z, /* in */ ccsr* Dcsr, ccsr* Dcsr_nopml, /*out */ ccsr* Dcsr_reduced, double eps)
{
	int n = x.n * y.n;
	int size = x.n * y.n * z.n;
	int size2 = (n - x.n) * z.n;
	int size3 = (n - x.n) * z.n;
	int size4 = size - n;
	int non_zeros_in_3Dblock3diag = size + size2 * 2 + size3 * 2 + size4 * 2;
	int l2 = 0;
	int count = 0;
	int i1, j1, k1;
	int i2, j2, k2;
	int cur_elem = 0;
	int numb = 0;
	double RelRes = 0;

	int ERROR_RESULT = 0;

	sparse_struct *my_check;

	my_check = (sparse_struct*)malloc(sizeof(sparse_struct));

	my_check->n = size;
	my_check->csr_ia = Dcsr->ia;
	my_check->csr_ja = Dcsr->ja;
	my_check->indexing = MKL_ONE_BASED;
	my_check->matrix_structure = MKL_GENERAL_STRUCTURE;
	my_check->matrix_format = MKL_CSR;
	my_check->message_level = MKL_PRINT;
	my_check->print_style = MKL_C_STYLE;

	sparse_matrix_checker_init(my_check);

	ERROR_RESULT = sparse_matrix_checker(my_check);

	printf("ERROR_RESULT1: %d\n", ERROR_RESULT);

	my_check->n = x.n_nopml * y.n_nopml * z.n_nopml;
	my_check->csr_ia = Dcsr_nopml->ia;
	my_check->csr_ja = Dcsr_nopml->ja;
	my_check->indexing = MKL_ONE_BASED;
	my_check->matrix_structure = MKL_GENERAL_STRUCTURE;
	my_check->matrix_format = MKL_CSR;
	my_check->message_level = MKL_PRINT;
	my_check->print_style = MKL_C_STYLE;

	sparse_matrix_checker_init(my_check);

	ERROR_RESULT = sparse_matrix_checker(my_check);

	printf("ERROR_RESULT2: %d\n", ERROR_RESULT);

#ifdef OUTPUT
//	FILE* fout3;
//	fout3 = fopen("full_matrices.dat", "w");

//	for (int i = 0; i < non_zeros_in_3Dblock3diag; i++)
//	{
//		fprintf(fout3, "i = %d ; %d %12.10lf %12.10lf\n", i, Dcsr->ja[i] - 1, Dcsr->values[i].real(), Dcsr->values[i].imag());
//	}

//	fclose(fout3);
#endif

	for (int l1 = 0; l1 < size; l1++)
	{
		int elems_in_row = Dcsr->ia[l1 + 1] - Dcsr->ia[l1];

		for (int j_loc = 0; j_loc < elems_in_row; j_loc++)
		{
			l2 = Dcsr->ja[cur_elem] - 1;
			take_coord3D(x.n, y.n, z.n, l1, i1, j1, k1);
			take_coord3D(x.n, y.n, z.n, l2, i2, j2, k2);
			if ((i1 >= x.pml_pts && j1 >= y.pml_pts && k1 >= z.pml_pts && i1 < (x.n - x.pml_pts) && j1 < (y.n - y.pml_pts) && k1 < (z.n - z.pml_pts))
			  &&(i2 >= x.pml_pts && j2 >= y.pml_pts && k2 >= z.pml_pts && i2 < (x.n - x.pml_pts) && j2 < (y.n - y.pml_pts) && k2 < (z.n - z.pml_pts))) Dcsr_reduced->values[numb++] = Dcsr->values[cur_elem];

			cur_elem++;
		}
	}


	if (numb != Dcsr_reduced->non_zeros || cur_elem != Dcsr->non_zeros) printf("matrix PML reduction failed: %d != %d OR %d != %d\n", numb, Dcsr_reduced->non_zeros, cur_elem, Dcsr->non_zeros);
	else printf("matrix PML reduction succeed: %d elements!\n", numb);

	//system("pause");


#ifdef OUTPUT
	FILE* fout1;
	fout1 = fopen("matrices.dat", "w");

	for (int i = 0; i < numb; i++)
	{
		fprintf(fout1, "i = %d %d %12.10lf %12.10lf %12.10lf %12.10lf\n", i, Dcsr_nopml->ja[i] - 1, Dcsr_reduced->values[i].real(), Dcsr_reduced->values[i].imag(), Dcsr_nopml->values[i].real(), Dcsr_nopml->values[i].imag());
	}

	fclose(fout1);
#endif


	RelRes = rel_error(zlange, numb, 1, Dcsr_reduced->values, Dcsr_nopml->values, numb, eps);

#ifdef OUTPUT
	FILE* fout2;
	fout2 = fopen("resid_values.dat", "w");

	for (int i = 0; i < numb; i++)
	{
		if (abs(Dcsr_reduced->values[i]) > 0.1) fprintf(fout2, "%d %12.10lf %12.10lf  false\n", i, Dcsr_reduced->values[i].real(), Dcsr_reduced->values[i].imag());
		else fprintf(fout2, "%d %12.10lf %12.10lf\n", i, Dcsr_reduced->values[i].real(), Dcsr_reduced->values[i].imag());
	}

	fclose(fout2);
#endif


	if (RelRes < eps) printf("values[:] - values_no_pml[:]. Norm %10.8e < eps %10.8lf: PASSED\n", RelRes, eps);
	else printf("values[:] - values_no_pml[:]. Norm %10.8lf > eps %10.8e : FAILED\n", RelRes, eps);

	
}
#endif

void TestNormalizedVector(int n, dtype* vect, double eps)
{
	double norm;
	int ione = 1;

	norm = dznrm2(&n, vect, &ione);
	if (fabs(norm - 1.0) < eps) printf("Normalized vector: %12.10e = 1.0: PASSED\n", norm);
	else printf("Normalized vector: %12.10lf != 1.0 : FAILED\n", norm);

}

void TestOrtogonalizedVectors(int n, dtype* vect1, dtype* vect2, double eps)
{
	dtype value;
	int ione = 1;

	zdotc(&value, &n, vect1, &ione, vect2, &ione);

	if (abs(value) < eps) printf("zdot: (%12.10e, %12.10e) = 0.0: PASSED\n", value.real(), value.imag());
	else printf("zdot: (%12.10lf, %12.10lf) != 0.0 : FAILED\n", value.real(), value.imag());

}

void GivensRotations(dtype* H, int ldh, dtype* eBeta, int &rotationsCount, double eps)
{
	dtype c, s;
	dtype oldValue;
	printf("Givens rotations: %d\n", rotationsCount);
	for (int i = 0; i < rotationsCount; i++)
	{
		c = H[i + ldh * i] / sqrt(H[i + ldh * i] * H[i + ldh * i] + H[i + 1 + ldh * i] * H[i + 1 + ldh * i]);
		s = H[i + 1 + ldh * i] / sqrt(H[i + ldh * i] * H[i + ldh * i] + H[i + 1 + ldh * i] * H[i + 1 + ldh * i]);     //модификация матрицы H 
		for (int j = i; j < rotationsCount; j++)
		{
			oldValue = H[i + ldh * j];
			H[i + ldh * j] = oldValue * c + H[i + 1 + ldh * j] * s;
			H[i + 1 + ldh * j] = H[i + 1 + ldh * j] * c - oldValue * s;
		}

		// right-hand side modification
		oldValue = eBeta[i];
		eBeta[i] = oldValue * c;
		eBeta[i + 1] = -oldValue * s;

		if (abs(eBeta[i + 1]) < eps)
		{
			rotationsCount = i + 1;
			printf("Break: eBeta[%d] < %lf\n", i + 1, eps);
			break;
		}
	}



}

void TestFGMRES()
{
	printf("----Test FGMRES----\n");
//	int n = 541; 
	int n = 841;
	int i, j;
	double val_re, val_im;
	int count = 0;
//	int non_zeros = 4285;
	int non_zeros = 4089;
	int ione = 1;

	int size = n;
	int m = 1000;
	int iterCount = m;
	int iter = 0;
	double norm = 0;
	double norm_r0 = 0;
	double beta = 0;
	double RelRes;
	int i1, j1, k1;
	double thresh = 10e-8;

	int nrhs = 1;
	int lwork = -1;
	int info = 0;
	int col_min, row_min;
	dtype work_size;
	dtype done = { 1.0, 0.0 };
	dtype mone = { -1.0, 0.0 };
	dtype zero = { 0.0, 0.0 };

	dtype *A = alloc_arr<dtype>(n * n); int lda = n;
	dtype *x_orig = alloc_arr<dtype>(n);
	dtype *x_sol = alloc_arr<dtype>(n);
	dtype *f = alloc_arr<dtype>(n);

	dtype *g = alloc_arr<dtype>(size);
	dtype *x0 = alloc_arr<dtype>(size);
	dtype *deltaL = alloc_arr<dtype>(size);
	dtype* work;

	FILE *in = fopen("Matrix4.txt", "r");

	while (!feof(in))
	{
		//fscanf(in, "%d %d %lf\n", &i, &j, &val_re);
		fscanf(in, "%d %d %lf %lf\n", &i, &j, &val_re, &val_im);
		A[i - 1 + lda * (j - 1)] = dtype{ val_re, val_im };
		count++;
	}

	fclose(in);

	if (count != non_zeros) printf("ERROR! %d != %d\n", count, non_zeros);
	system("pause");
	//GenSolVector(n, x_orig);
	for (int i = 0; i < size; i++)
		x_orig[i] = { 1.0, 0.5 };

	printf("Multiply f := A * u\n");
	zgemv("no", &n, &n, &done, A, &lda, x_orig, &ione, &zero, f, &ione);


	dtype *f_rsd = alloc_arr<dtype>(size);
	for (int i = 0; i < size; i++)
		f_rsd[i] = f[i];

	zgemv("no", &n, &n, &done, A, &lda, x_orig, &ione, &mone, f_rsd, &ione);

	RelRes = dznrm2(&size, f_rsd, &ione);

	printf("-----------\n");
	printf("Residual in 3D with PML ||A * x_sol - f|| = %e\n", RelRes);
	printf("-----------\n");

	system("pause");
	printf("-------------FGMRES-----------\n");

	// We need to solve iteratively: (I - \delta L * L_0^{-1})w = g


	printf("-----Step 0. Memory allocation-----\n");
	// matrix of Krylov basis
	dtype* V = alloc_arr<dtype>(size * (m + 1)); int ldv = size;
	dtype* w = alloc_arr<dtype>(size);

	// residual vector
	dtype *r0 = alloc_arr<dtype>(size);

	// additional vector
	dtype *Ax0 = alloc_arr<dtype>(size);

	// Hessenberg matrix
	dtype *H = alloc_arr<dtype>((m + 1) * m); int ldh = m + 1;
	dtype *Hgels = alloc_arr<dtype>((m + 1) * m);

	// the vector of right-hand side for the system with Hessenberg matrix
	dtype *eBeta = alloc_arr<dtype>(m + 1); int ldb = m + 1;
	dtype calpha;
	double dalpha;


	// 1. First step. Compute r_0 and its norm
	printf("-----Step 1-----\n");
#pragma omp parallel for simd schedule(static)
	for (int i = 0; i < size; i++)
		x0[i] = 0.0;

	// Multiply matrix A in CSR format by vector x_0 to obtain f1
	zgemv("no", &n, &n, &done, A, &lda, x0, &ione, &zero, Ax0, &ione);

	norm = dznrm2(&size, f, &ione);
	printf("norm ||f|| = %lf\n", norm);

	//Add_dense(size, ione, 1.0, f, size, -1.0, Ax0, size, r0, size); // r0 = f - Ax0

	zcopy(&size, f, &ione, r0, &ione);
	zaxpy(&size, &mone, Ax0, &ione, r0, &ione); // r0: = f - Ax0

	norm = dznrm2(&size, r0, &ione);
	printf("norm ||r0|| = %lf\n", norm);

	//norm = RelError(zlange, size, 1, r0, f, size, thresh);
	//printf("r0 = f - Ax0, norm ||r0 - f|| = %lf\n", norm);

	NormalizeVector(size, r0, &V[ldv * 0], beta); // v + size * j = &v[ldv * j]

	TestNormalizedVector(size, &V[0], thresh);

	// 2. The main iterations of algorithm
	printf("-----Step 2. Iterations-----\n");
	for (int j = 0; j < m; j++)
	{
		printf("iter = %d\n", j);
		// Compute w[j] := A * v[j]
		zgemv("no", &n, &n, &done, A, &lda, &V[ldv * j], &ione, &zero, w, &ione);

		for (int i = 0; i <= j; i++)
		{
			// H[i + ldh * j] = (w_j * v_i) 
			//H[i + ldh * j] = zdot(size, w, &V[ldv * i]);
			
			zdotc(&H[i + ldh * j], &size, w, &ione, &V[ldv * i], &ione);

			//w[j] = w[j] - H[i][j]*v[i]
			//AddDenseVectorsComplex(size, 1.0, w, -H[i + ldh * j], &V[ldv * i], w);
			calpha = -H[i + ldh * j];
			zaxpy(&size, &calpha, &V[ldv * i], &ione, w, &ione);
		}

		H[j + 1 + ldh * j] = dznrm2(&size, w, &ione);
		printf("norm H[%d][%d] = %lf %lf\n", j, j, H[j + ldh * j].real(), H[j + ldh * j].imag());
		printf("norm H[%d][%d] = %lf %lf\n", j + 1, j, H[j + 1 + ldh * j].real(), H[j + 1 + ldh * j].imag());

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

		//MultVectorConst(size, w, 1.0 / H[j + 1 + ldh * j], &V[ldv * (j + 1)]);
		TestNormalizedVector(size, &V[ldv * (j + 1)], thresh);
		for (int i = 0; i <= j; i++)
		{
			//TestOrtogonalizedVectors(size, &V[ldv * (j + 1)], &V[ldv * i], thresh);
		}


		// 3. Solving least squares problem to compute y_k
		// for x_k = x_0 + V_k * y_k
		printf("-----Step 3. LS problem-----\n");

		printf("size of basis: %d\n", iterCount);

		// Set eBeta
#pragma omp parallel for simd schedule(static)
		for (int i = 0; i < m + 1; i++)
			eBeta[i] = 0;

		eBeta[0] = beta;

		lwork = -1;
		col_min = j + 1;
		row_min = j + 2;
		//col_min = iterCount;
		//row_min = iterCount + 1;

//#define GIVENS
				// Run
		for (int i = 0; i < m * (m + 1); i++)
			Hgels[i] = H[i];

#ifndef GIVENS
		// Query
		zgels("no", &row_min, &col_min, &nrhs, Hgels, &ldh, eBeta, &ldb, &work_size, &lwork, &info);

		lwork = (int)work_size.real();
		work = alloc_arr<dtype>(lwork);


		zgels("no", &row_min, &col_min, &nrhs, Hgels, &ldh, eBeta, &ldb, work, &lwork, &info);
		free_arr(work);

#else
		iterCount = col_min;
		//iterCount = m;
		GivensRotations(Hgels, ldh, eBeta, iterCount, thresh);

		printf("ztrsm...\n");
		ztrsm("left", "up", "no", "no", &iterCount, &iterCount, &done, Hgels, &ldh, eBeta, &ldb);
#endif
		// 4. Multiplication x_k = x_0 + V_k * y_k
		printf("-----Step 4. Computing x_k-----\n");

		iterCount = col_min;

		zgemv("no", &size, &iterCount, &done, V, &ldv, eBeta, &ione, &done, x0, &ione);

		//system("pause");

		for (int i = 0; i < size; i++)
			f_rsd[i] = f[i];

		zgemv("no", &n, &n, &done, A, &lda, x0, &ione, &mone, f_rsd, &ione);

		RelRes = dznrm2(&size, f_rsd, &ione);

		printf("-----------\n");
		printf("Residual in 3D with PML ||A * x_sol - f|| = %e\n", RelRes);
		printf("-----------\n");

		// Set init cond for next step
#pragma omp parallel for simd schedule(static)
		for (int i = 0; i < size; i++)
			x0[i] = 0.0;
	}
	
}

void Test_Poisson_FFT1D_Real(int n /* grid points in 1 dim */, double eps)
{
	DFTI_DESCRIPTOR_HANDLE my_desc1_handle;
	DFTI_DESCRIPTOR_HANDLE my_desc2_handle;
	DFTI_DESCRIPTOR_HANDLE hand = 0;
	MKL_LONG status;

	double L = 1.0;
	double h = L / (n + 1);
	double norm;
	double sum = 0;

	double *f = alloc_arr<double>(n);
	double *u = alloc_arr <double>(n);
	dtype *u_obt = alloc_arr<dtype>(n);
	dtype *f_FFT = alloc_arr<dtype>(n);
	dtype *f_MYFFT = alloc_arr<dtype>(n);

	double *u_FFT = alloc_arr<double>(n);
	double *u_MYFFT = alloc_arr<double>(n);

	double* x = alloc_arr<double>(n);
	double *lambda = alloc_arr<double>(n);

	for (int i = 0; i < n; i++)
	{
		x[i] = (i + 1) * h; // inner pointas only
		printf("%lf ", x[i]);
	}


	// u_xx = f
	// u = sin(2 * PI * x) , f = - 4 * PI * PI * sin(2 * PI * x)

	for (int i = 0; i < n; i++)
	{
		u[i] = -sin(2.0 * PI * x[i]) / (4.0 * PI * PI + 1.0);
		f[i] = sin(2.0 * PI * x[i]);

		//u[i] = x[i] * (x[i] - L) * (x[i] - L);
		//f[i] = 6 * x[i] - 4 * L - x[i] * (x[i] - L) * (x[i] - L);
	}


	status = DftiCreateDescriptor(&hand, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)(n));
	if (status != DFTI_NO_ERROR) goto failed;

	printf("Set configuration: out-of-place\n");
	status = DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	if (status != DFTI_NO_ERROR) goto failed;

	printf("Set configuration: CCE storage\n");
	status = DftiSetValue(hand, DFTI_CONJUGATE_EVEN_STORAGE,
		DFTI_COMPLEX_COMPLEX);

	//	status = DftiSetValue(hand, DFTI_FORWARD_SCALE, 1.0 / n);

	status = DftiCommitDescriptor(hand);
	if (status != DFTI_NO_ERROR) goto failed;

	status = DftiComputeForward(hand, f, f_FFT);
	if (status != DFTI_NO_ERROR) goto failed;

	MyFFT1D_ForwardReal(n, f, f_MYFFT);
	for (int i = 0; i < n; i++)
		printf("f_FFT[%d]: %14.12lf + I * %14.12lf  f_MYFFT[%d]: %14.12lf + I * %14.12lf\n", i, f_FFT[i].real(), f_FFT[i].imag(),
			i, f_MYFFT[i].real(), f_MYFFT[i].imag());

	//	norm = rel_error_complex(n / 2, 1, f_MYFFT, f_FFT, n, eps);
	//	if (norm < eps) printf("Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
	//	else printf("Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);
	//	system("pause");

	/*	for (int i = 0; i < n; i++)
	if (i < n / 2 - 1) lambda[i] = -(2.0 * PI * i / L) * (2.0 * PI * i / L) - 1.0;
	else lambda[i] = -(2.0 * PI * (n - i) / L) * (2.0 * PI * (n - i) / L) - 1.0;*/

	for (int i = 0; i < n; i++)
		lambda[i] = -((2 * PI * (i + 1) / L) * (2 * PI * (i + 1) / L) + 1);

	for (int i = 1; i < n; i++)
	{
		f_FFT[i] /= lambda[i];
		f_MYFFT[i] /= lambda[i];
		//	printf("lambda[%d] = %lf\n", i, lambda[i]);
	}

	MyFFT1D_BackwardReal(n, f_MYFFT, u_MYFFT);

	status = DftiComputeBackward(hand, f_FFT, u_FFT);
	if (status != DFTI_NO_ERROR) goto failed;

	for (int i = 0; i < n; i++)
	{
		u_FFT[i] /= n;
	}

	for (int i = 0; i < n; i++)
		printf("u_backward[%d]: %14.12lf  u_MKL[%d]: %14.12lf, u_ex[%d]:  %14.12lf\n", i, u_MYFFT[i],
			i, u_FFT[i], i, u[i]);

	norm = rel_error(dlange, n, 1, u_FFT, u, n, eps);
	if (norm < eps) printf("MKL: Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
	else printf("MKL: Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);

	norm = rel_error(dlange, n, 1, u_MYFFT, u, n, eps);
	if (norm < eps) printf("MyFFT: Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
	else printf("MyFFT: Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);
	system("pause");

	printf("Free DFTI descriptor\n");
	DftiFreeDescriptor(&hand);

	return;
failed:
	printf("ERROR\n");
	return;
}

void Test_ExactSolution_1D(int n, double h, double* u, double *f, double eps)
{
	double *u_left = alloc_arr<double>(n - 2);

	for (int i = 0; i < n - 2; i++)
	{
		u_left[i] = (u[i + 2] - 2 * u[i + 1] + u[i]) / (h * h) - u[i + 1];
		printf("%d: u_xx - u: %lf  f: %lf\n", i + 1, u_left[i], f[i + 1]);
	}

	double norm = rel_error(dlange, n - 2, 1, u_left, &f[1], n - 2, eps);
	if (norm < eps) printf("u_xx - u = f: Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
	else printf("u_xx - u = f: Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);
	//system("pause");
}

void Test_Poisson_FT1D_Real(int n /* grid points in 1 dim */, double eps)
{
	DFTI_DESCRIPTOR_HANDLE my_desc1_handle;
	DFTI_DESCRIPTOR_HANDLE my_desc2_handle;
	DFTI_DESCRIPTOR_HANDLE hand = 0;
	MKL_LONG status;

	n = 79;

	double L = 1.0;
	double h = L / (n - 1);
	double norm;
	double sum = 0;

	double *f = alloc_arr<double>(n);
	double *u = alloc_arr <double>(n);
	dtype *u_obt = alloc_arr<dtype>(n);
	dtype *f_FFT = alloc_arr<dtype>(n);
	dtype *f_MYFFT = alloc_arr<dtype>(n);

	double *u_FFT = alloc_arr<double>(n);
	double *u_MYFFT = alloc_arr<double>(n);

	double* x = alloc_arr<double>(n);
	double *lambda_my = alloc_arr<double>(n);
	double *lambda_mkl = alloc_arr<double>(n);

	for (int i = 0; i < n; i++)
	{
		x[i] = i * h; // all points from 0 to 1
		printf("%lf ", x[i]);
	}


	// u_xx = f
	// u = sin(2 * PI * x) , f = - 4 * PI * PI * sin(2 * PI * x)

	for (int i = 0; i < n; i++)
	{
		u[i] = -sin(2.0 * PI * x[i]) / (4.0 * PI * PI + 1.0);
		f[i] = sin(2.0 * PI * x[i]);

	//	u[i] = x[i] * (x[i] - L) * (x[i] - L);
	//	f[i] = 6 * x[i] - 4 * L - x[i] * (x[i] - L) * (x[i] - L);
	}

	Test_ExactSolution_1D(n, h, u, f, eps);


	status = DftiCreateDescriptor(&hand, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)(n));
	if (status != DFTI_NO_ERROR) goto failed;

	printf("Set configuration: out-of-place\n");
	status = DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	if (status != DFTI_NO_ERROR) goto failed;

	printf("Set configuration: CCE storage\n");
	status = DftiSetValue(hand, DFTI_CONJUGATE_EVEN_STORAGE,
		DFTI_COMPLEX_COMPLEX);

	//	status = DftiSetValue(hand, DFTI_FORWARD_SCALE, 1.0 / n);

	status = DftiCommitDescriptor(hand);
	if (status != DFTI_NO_ERROR) goto failed;

	status = DftiComputeForward(hand, f, f_FFT);
	if (status != DFTI_NO_ERROR) goto failed;

	MyFT1D_ForwardReal(n, h, f, f_MYFFT);
	printf("h = %lf\n", h);
	for (int i = 0; i < n; i++)
		printf("f_MYFFT[%d]: %14.12lf + I * %14.12lf, exact: %14.12lf\n",
			i, f_MYFFT[i].real(), f_MYFFT[i].imag(), -2.0 / (PI * (4 * (i - n/2) * (i - n / 2) - 1)));

	//	norm = rel_error_complex(n / 2, 1, f_MYFFT, f_FFT, n, eps);
	//	if (norm < eps) printf("Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
	//	else printf("Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);
	//	system("pause");

	/*	for (int i = 0; i < n; i++)
	if (i < n / 2 - 1) lambda[i] = -(2.0 * PI * i / L) * (2.0 * PI * i / L) - 1.0;
	else lambda[i] = -(2.0 * PI * (n - i) / L) * (2.0 * PI * (n - i) / L) - 1.0;*/

	for (int i = 0; i < n; i++)
	{
		lambda_my[i] = -((2 * PI * (i - n / 2) / L) * (2 * PI * (i - n / 2) / L) + 1);
		lambda_mkl[i] = -((2 * PI * i / L) * (2 * PI * i / L) + 1);

	}

	for (int i = 0; i < n; i++)
	{
		f_FFT[i] /= lambda_mkl[i];
		f_MYFFT[i] /= lambda_my[i];
		//	printf("lambda[%d] = %lf\n", i, lambda[i]);
	}

	MyFT1D_BackwardReal(n, h, f_MYFFT, u_MYFFT);

	status = DftiComputeBackward(hand, f_FFT, u_FFT);
	if (status != DFTI_NO_ERROR) goto failed;

	for (int i = 0; i < n; i++)
	{
		u_FFT[i] /= n;
	}

	for (int i = 0; i < n; i++)
		printf("u_backward[%d]: %14.12lf  u_MKL[%d]: %14.12lf, u_ex[%d]:  %14.12lf\n", i, u_MYFFT[i],
			i, u_FFT[i], i, u[i]);
	

	norm = rel_error(dlange, n, 1, u_FFT, u, n, eps);
	if (norm < eps) printf("MKL: Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
	else printf("MKL: Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);

	norm = rel_error(dlange, n, 1, u_MYFFT, u, n, eps);
	if (norm < eps) printf("MyFFT: Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
	else printf("MyFFT: Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);
	system("pause");

	printf("Free DFTI descriptor\n");
	DftiFreeDescriptor(&hand);

	return;
failed:
	printf("ERROR\n");
	return;
}

void Test_Poisson_FT1D_Complex(int n /* grid points in 1 dim */, double eps)
{
	DFTI_DESCRIPTOR_HANDLE my_desc1_handle;
	DFTI_DESCRIPTOR_HANDLE my_desc2_handle;
	DFTI_DESCRIPTOR_HANDLE hand = 0;
	MKL_LONG status;

	n = 80;

	size_m xx;

	double L = 10.0;
	double h = L / (n);
	double norm;
	double sum = 0;

	xx.h = h;
	xx.l = L;

	dtype *f = alloc_arr<dtype>(n);
	dtype *u = alloc_arr <dtype>(n);
	dtype *u_obt = alloc_arr<dtype>(n);
	dtype *f_FFT = alloc_arr<dtype>(n);
	dtype *f_MYFFT = alloc_arr<dtype>(n);

	dtype *u_FFT = alloc_arr<dtype>(n);
	dtype *u_MYFFT = alloc_arr<dtype>(n);

	double* x = alloc_arr<double>(n);
	double *lambda_my = alloc_arr<double>(n);
	double *lambda_mkl = alloc_arr<double>(n);

	for (int i = 0; i < n; i++)
	{
		x[i] = i * h; // all points from 0 to 1
		printf("%lf ", x[i]);
	}


	// u_xx = f
	// u = sin(2 * PI * x) , f = - 4 * PI * PI * sin(2 * PI * x)

	double pi2l = 2.0 * PI / xx.l;

	for (int i = 0; i < n; i++)
	{
		u[i] = -sin(pi2l * x[i]) / (pi2l * pi2l + 1.0);
		f[i] = sin(pi2l * x[i]);

		//	u[i] = x[i] * (x[i] - L) * (x[i] - L);
		//	f[i] = 6 * x[i] - 4 * L - x[i] * (x[i] - L) * (x[i] - L);
	}

	//Test_ExactSolution_1D(n, h, u, f, eps);


	status = DftiCreateDescriptor(&hand, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)(n));
	if (status != DFTI_NO_ERROR) goto failed;

	printf("Set configuration: out-of-place\n");
	status = DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	if (status != DFTI_NO_ERROR) goto failed;

	//printf("Set configuration: CCE storage\n");
	//status = DftiSetValue(hand, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	//status = DftiSetValue(hand, DFTI_COMPLEX_STORAGE, DFTI_CONJUGATE_EVEN_STORAGE);

	status = DftiSetValue(hand, DFTI_BACKWARD_SCALE, 1.0 / (n));
	if (status != DFTI_NO_ERROR) goto failed;

	//	status = DftiSetValue(hand, DFTI_FORWARD_SCALE, 1.0 / n);

	status = DftiCommitDescriptor(hand);
	if (status != DFTI_NO_ERROR) goto failed;

	status = DftiComputeForward(hand, f, f_FFT);
	if (status != DFTI_NO_ERROR) goto failed;

	MyFT1D_ForwardComplex(n, xx, f, f_MYFFT);
	printf("h = %lf\n", h);

	for (int i = 0; i < n; i++)
		printf("f_MYFFT[%d]: %14.12lf + I * %14.12lf, f_FFT[%d]: %14.12lf + I * %14.12lf\n",
			i, f_MYFFT[i].real(), f_MYFFT[i].imag(), i, f_FFT[i].real(), f_FFT[i].imag());


//	for (int i = 0; i < n; i++)
	//	printf("f_MYFFT[%d]: %14.12lf + I * %14.12lf, exact: %14.12lf\n",
		//	i, f_MYFFT[i].real(), f_MYFFT[i].imag(), -2.0 / (PI * (4 * (i - n / 2) * (i - n / 2) - 1)));

	//	norm = rel_error_complex(n / 2, 1, f_MYFFT, f_FFT, n, eps);
	//	if (norm < eps) printf("Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
	//	else printf("Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);
	//	system("pause");

	for (int i = 0; i < n; i++)
	if (i < n / 2 - 1) lambda_mkl[i] = -(pi2l * i) * (pi2l * i) - 1.0;  // 0 ... n / 2 ... 0
	else lambda_mkl[i] = -(pi2l * (n - i)) * (pi2l * (n - i)) - 1.0;

	for (int i = 0; i < n; i++)
	{
		lambda_my[i] = -(pi2l * (i - n / 2) * pi2l * (i - n / 2) + 1);   //  -n/2 ... 0 ... n / 2
	//	lambda_mkl[i] = -(pi2l * i * pi2l * i  + 1);

	}

	for (int i = 0; i < n; i++)
	{
		f_FFT[i] /= lambda_mkl[i];
		f_MYFFT[i] /= lambda_my[i];
		//	printf("lambda[%d] = %lf\n", i, lambda[i]);
	}

	MyFT1D_BackwardComplex(n, xx, f_MYFFT, u_MYFFT);

	status = DftiComputeBackward(hand, f_FFT, u_FFT);
	if (status != DFTI_NO_ERROR) goto failed;


	for (int i = 0; i < n; i++)
		printf("u_backward[%d]: %14.12lf  u_MKL[%d]: %14.12lf, u_ex[%d]:  %14.12lf\n", i, u_MYFFT[i].real(),
			i, u_FFT[i].real(), i, u[i].real());


	norm = rel_error(zlange, n, 1, u_FFT, u, n, eps);
	if (norm < eps) printf("MKL: Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
	else printf("MKL: Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);

	norm = rel_error(zlange, n, 1, u_MYFFT, u, n, eps);
	if (norm < eps) printf("MyFFT: Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
	else printf("MyFFT: Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);
	system("pause");

	printf("Free DFTI descriptor\n");
	DftiFreeDescriptor(&hand);

	return;
failed:
	printf("ERROR\n");
	return;
}

void TestHankel()
{
	printf("-----TEST HANKEL-------\n");
	double x1 = PI;
	double x2 = PI / 4;
	dtype z1 = { x1, 0 };
	dtype z2 = { x2, 0 };

	if (fabs(Hankel(x1).real() - Hankel(z1).real()) < EPS && (Hankel(x1).imag() - Hankel(z1).imag()) < EPS)
	printf("PASSED Hankel %lf vs Hankel (%lf, %lf): (%lf %lf) - (%lf, %lf)\n", x1, z1.real(), z1.imag(),
		Hankel(x1).real(), Hankel(x1).imag(), Hankel(z1).real(), Hankel(z1).imag());

	if ((Hankel(x2).real() - Hankel(z2).real()) < EPS && (Hankel(x2).imag() - Hankel(z2).imag()) < EPS)
	printf("PASSED Hankel %lf vs Hankel (%lf, %lf): (%lf %lf) - (%lf, %lf)\n", x2, z2.real(), z2.imag(),
		Hankel(x2).real(), Hankel(x2).imag(), Hankel(z2).real(), Hankel(z2).imag());

	int N = 1; // number of functions
	int nz;
	int ierr = 0;

	void *vmblock;

	//memory allocation for cyr, cyi, cwr, cwi
	vmblock = vminit();
	REAL *res_real = (REAL *)vmalloc(vmblock, VEKTOR, N + 1, 0); //index 0 not used
	REAL *res_imag = (REAL *)vmalloc(vmblock, VEKTOR, N + 1, 0);


	ZBESH(z1.real(), z1.imag(), 0, 1, 1, 1, res_real, res_imag, &nz, &ierr);
	//printf("ZBESH IERROR: %d\n", ierr);

	if ((Hankel(z1).real() - res_real[1]) < EPS && (Hankel(z1).imag() - res_imag[1]) < EPS)
	printf("PASSED Hankel (%lf, %lf): (%lf, %lf)\n", z1.real(), z1.imag(), res_real[1], res_imag[1]);

	ZBESH(z2.real(), z2.imag(), 0, 1, 1, 1, res_real, res_imag, &nz, &ierr);

//	printf("ZBESH IERROR: %d\n", ierr);
	if ((Hankel(z2).real() - res_real[1]) < EPS && (Hankel(z2).imag() - res_imag[1]) < EPS)
	printf("PASSED Hankel (%lf, %lf): (%lf, %lf)\n", z2.real(), z2.imag(), res_real[1], res_imag[1]);

}

int TestDomainCorrelation(int n1, int n2, dtype* u_sol, double thresh)
{
	int size2D = n1 * n2;
	int passed = 0;
	int ione = 1;
	double norm = 0;

	dtype numb = u_sol[3 * size2D / 4];

	for(int k = 0; k < size2D; k++)
		u_sol[k] -= numb;

	norm = dznrm2(&size2D, u_sol, &ione);

	if (norm < thresh) passed = 1;
	else printf("norm = %lf\n", norm);

	for(int k = 0; k < size2D; k++)
		u_sol[k] += numb;

	return passed;
}

void Test1DHelmholtz(int n1, int n2, int n3, dtype* u_sol, double thresh, char *str)
{
	int n = 0;
	int size2D = n1 * n2;
	for (int k = 0; k < n3; k++)
	{
		n += TestDomainCorrelation(n1, n2, &u_sol[k * size2D], thresh);
	}

	printf("PASSED PLAINS for %s SOLUTION: %d of %d\n", str, n, n3);
}

void Shell_FFT1D_Complex(ptr_test_fft func, const string& test_name, int& numb, int& fail_count)
{
	double eps = 10e-6;
	//	for (double eps = 1e-2; eps > 1e-8; eps /= 10)
	for (int n = 1001; n <= 1001; n += 1001)
	{
		try
		{
			numb++;
			func(n, eps);
		}
		catch (runtime_error& e)
		{
			++fail_count;
			cerr << test_name << " fail: " << e.what() << endl;
		}
		catch (...) {
			++fail_count;
			cerr << "Unknown exception caught" << endl;
		}
	}
}


void TestAll()
{
	TestRunner runner;
	//void(*pt_func)(int&) = NULL;
	//pt_func = &Shell_SymRecCompress;

	printf("***** TEST LIBRARY FUNCTIONS *******\n");
	printf("****Complex precision****\n");
//	runner.RunTest(Shell_LowRankApprox, Test_LowRankApproxStruct, "Test_LowRankApprox");
//	runner.RunTest(Shell_SymRecCompress, Test_SymRecCompressStruct, "Test_SymRecCompress");
//	runner.RunTest(Shell_DiagMult, Test_DiagMultStruct, "Test_DiagMult");
//	runner.RunTest(Shell_RecMultL, Test_RecMultLStruct, "Test_RecMultL");
//	runner.RunTest(Shell_Add, Test_AddStruct, "Test_Add");
//	runner.RunTest(Shell_SymCompUpdate2, Test_SymCompUpdate2Struct, "Test_SymCompUpdate2");
//	runner.RunTest(Shell_SymCompRecInv, Test_SymCompRecInvStruct, "Test_SymCompRecInv");
//	runner.RunTest(Shell_CopyStruct, Test_CopyStruct,  "Test_CopyStruct");
	printf("*******FFT*******\n");
//	runner.RunTest(Shell_FFT1D_Real, Test_FFT1D_Real, "Test_FFT1D");
//	runner.RunTest(Shell_FFT1D_Complex, Test_FFT1D_Complex, "Test_FFT1D_Complex");
//	runner.RunTest(Shell_FFT1D_Complex, Test_Poisson_FT1D_Real, "Test_Poisson_FT1D_Real");
	runner.RunTest(Shell_FFT1D_Complex, Test_Poisson_FT1D_Complex, "Test_Poisson_FT1D_Complex");


	printf("********************\n");
	printf("ALL TESTS: %d\nPASSED: %d \nFAILED: %d\n", runner.GetAll(), runner.GetPassed(), runner.GetFailed());

	printf("***** THE END OF TESTING*******\n\n");

}
#if 0
void Shell_LowRankApprox(ptr_test_low_rank func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";

	for (double eps = 1e-2; eps > 1e-8; eps /= 10)
		for (int m = 3; m <= 10; m++)
			for (int n = 1; n <= 10; n++)
			{
				try
				{
					numb++;
					func(m, n, eps, method);
				}
				catch (runtime_error& e)
				{
					++fail_count;
					cerr << test_name << " fail: " << e.what() << endl;
				}
				catch (...) {
					++fail_count;
					cerr << "Unknown exception caught" << endl;
				}
			}
}

void Shell_SymRecCompress(ptr_test_sym_rec_compress func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";
	int smallsize = 3;

	for (int n = 3; n <= 10; n++)
		for (double eps = 1e-2; eps > 1e-8; eps /= 10)
		{
			try
			{
				numb++;
				func(n, eps, method, smallsize);
			}
			catch (runtime_error& e)
			{
				++fail_count;
				cerr << test_name << " fail: " << e.what() << endl;
			}
			catch (...) {
				++fail_count;
				cerr << "Unknown exception caught" << endl;
			}
		}
}

void Shell_DiagMult(ptr_test_sym_rec_compress func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";
	int smallsize = 3;

	for (int n = 3; n <= 10; n++)
		for (double eps = 1e-2; eps > 1e-8; eps /= 10)
		{
			try
			{
				numb++;
				func(n, eps, method, smallsize);
			}
			catch (runtime_error& e)
			{
				++fail_count;
				cerr << test_name << " fail: " << e.what() << endl;
			}
			catch (...) {
				++fail_count;
				cerr << "Unknown exception caught" << endl;
			}
		}

}

void Shell_RecMultL(ptr_test_mult_diag func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";
	int smallsize = 3;

	for (double eps = 1e-2; eps > 1e-8; eps /= 10)
		for (int n = 3; n <= 10; n++)
			for (int k = 1; k <= 10; k++)
			{
				try
				{
					numb++;
					func(n, k, eps, method, smallsize);
				}
				catch (runtime_error& e)
				{
					++fail_count;
					cerr << test_name << " fail: " << e.what() << endl;
				}
				catch (...) {
					++fail_count;
					cerr << "Unknown exception caught" << endl;
				}
			}

}

void Shell_Add(ptr_test_add func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";
	int smallsize = 3;

	for (double eps = 1e-4; eps > 1e-9; eps /= 10)
		for (int n = 3; n <= 10; n++)
			for (int alpha = -10; alpha < 10; alpha += 2)
				for (int beta = -10; beta < 10; beta += 2)
				{
					if (alpha != 0 && beta != 0)
					{
						try
						{
							numb++;
							func(n, alpha, beta, eps, method, smallsize);
						}
						catch (runtime_error& e)
						{
							++fail_count;
							cerr << test_name << " fail: " << e.what() << endl;
						}
						catch (...) {
							++fail_count;
							cerr << "Unknown exception caught" << endl;
						}
					}
				}
}

void Shell_SymCompUpdate2(ptr_test_update func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";
	int smallsize = 3;

	for (double eps = 1e-3; eps > 1e-9; eps /= 10)
		for (double alpha = -10; alpha < 10; alpha += 2)
			for (int n = 3; n <= 10; n++)
				for (int k = 1; k <= 10; k++)
				{
					try
					{
						numb++;
						func(n, k, alpha, eps, method, smallsize);
					}
					catch (runtime_error& e)
					{
						++fail_count;
						cerr << test_name << " fail: " << e.what() << endl;
					}
					catch (...) {
						++fail_count;
						cerr << "Unknown exception caught" << endl;
					}
				}
}


void Shell_SymCompRecInv(ptr_test_sym_rec_compress func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";
	int smallsize = 3;

	for (double eps = 1e-2; eps > 1e-8; eps /= 10)
		for (int n = 3; n <= 10; n++)
		{
			try
			{
				numb++;
				func(n, eps, method, smallsize);
			}
			catch (runtime_error& e)
			{
				++fail_count;
				cerr << test_name << " fail: " << e.what() << endl;
			}
			catch (...) {
				++fail_count;
				cerr << "Unknown exception caught" << endl;
			}
		}

}

void Shell_CopyStruct(ptr_test_sym_rec_compress func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";
	int smallsize = 3;

	for (double eps = 1e-2; eps > 1e-8; eps /= 10)
		for (int n = 3; n <= 10; n++)
		{
			try
			{
				numb++;
				func(n, eps, method, smallsize);
			}
			catch (runtime_error& e)
			{
				++fail_count;
				cerr << test_name << " fail: " << e.what() << endl;
			}
			catch (...) {
				++fail_count;
				cerr << "Unknown exception caught" << endl;
			}
		}
}

void Shell_FFT1D_Real(ptr_test_fft func, const string& test_name, int& numb, int& fail_count)
{
	double eps = 10e-6;
//	for (double eps = 1e-2; eps > 1e-8; eps /= 10)
		for (int n = 1000; n <= 1000; n += 1000)
		{
			try
			{
				numb++;
				func(n, eps);
			}
			catch (runtime_error& e)
			{
				++fail_count;
				cerr << test_name << " fail: " << e.what() << endl;
			}
			catch (...) {
				++fail_count;
				cerr << "Unknown exception caught" << endl;
			}
		}
}

#if 0
void Test_FFT1D(int n /* grid points in 1 dim */, double eps)
{
	DFTI_DESCRIPTOR_HANDLE my_desc1_handle;
	DFTI_DESCRIPTOR_HANDLE my_desc2_handle;
	MKL_LONG status;

	double size = 1.0;
	double h = size / n;
	double norm;

	dtype *f = alloc_arr<dtype>(n);
	dtype *u = alloc_arr<dtype>(n);
	dtype *f_FFT = alloc_arr<dtype>(n);
	dtype *u_FFT = alloc_arr<dtype>(n);

	// u_xx = f
	// u = sin(2 * PI * x) , f = - 4 * PI * PI * sin(2 * PI * x)

	for (int i = 0; i < n; i++)
	{
		u[i] = sin(2 * PI * i * h);
		f[i] = -4 * PI * PI * sin(2 * PI * i * h);
	}

	printf("Real\n");
	for (int i = 0; i < n; i++)
		printf("%lf %lf\n", u[i].real(), f[i].real());

	printf("Complex\n");
	for (int i = 0; i < n; i++)
		printf("%lf %lf\n", u[i].imag(), f[i].imag());

	// Create 1D FFT of COMPLEX DOUBLE case
	status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)n);
	if (status != DFTI_NO_ERROR) goto failed;

	status = DftiSetValue(my_desc1_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	if (status != DFTI_NO_ERROR) goto failed;

	status = DftiCommitDescriptor(my_desc1_handle);
	if (status != DFTI_NO_ERROR) goto failed;

	status = DftiComputeForward(my_desc1_handle, f, f_FFT);
	if (status != DFTI_NO_ERROR) goto failed;

	for (int i = 1; i < n; i++)
		f_FFT[i] /= -i * i; // - w * w

	// Backward FFT of solution u
	status = DftiComputeBackward(my_desc1_handle, f_FFT, u_FFT);
	if (status != DFTI_NO_ERROR) goto failed;

	//status = DftiComputeForward(my_desc1_handle, u, u_FFT);

	printf("Real\n");
	for (int i = 0; i < n; i++)
		printf("%lf %lf\n", u_FFT[i], f_FFT[i]);

	
	char str[255];
	norm = rel_error_complex(n, 1, &u_FFT[1], &f_FFT[1], n, eps);
	sprintf(str, "Size: n = %d ", n);
	AssertLess(norm, eps, str);

failed:
	printf("ERROR\n");

}
#else
void Test_FFT1D_Real(int n /* grid points in 1 dim */, double eps)
{
	DFTI_DESCRIPTOR_HANDLE my_desc1_handle;
	DFTI_DESCRIPTOR_HANDLE my_desc2_handle;
	DFTI_DESCRIPTOR_HANDLE hand = 0;
	MKL_LONG status;

	double size = 1.0;
	double h = size / (n);
	double norm;
	double sum = 0;

	double *f = alloc_arr<double>(n + 1);
	double *u = alloc_arr<double>(n + 1);
	dtype *u_obt = alloc_arr<dtype>(n + 1);
	dtype *f_FFT = alloc_arr<dtype>(n + 1);
	dtype *f_MYFFT = alloc_arr<dtype>(n + 1);

	double *u_FFT = alloc_arr<double>(n + 1);
	//dtype *u_FFT = alloc_arr<dtype>(n + 1);
	dtype *u_MYFFT = alloc_arr<dtype>(n + 1);


	// u_xx = f
	// u = sin(2 * PI * x) , f = - 4 * PI * PI * sin(2 * PI * x)

	for (int i = 0; i < n + 1; i++)
	{
		u[i] = sin(PI * i * h);
		f[i] = - PI * PI * sin(PI * i * h);
	//	u[i] = (i * h) * (i * h) * (i * h);
	//	f[i] = 6 * (i * h);
	}

	printf("Real\n");
	for (int i = 0; i < n + 1; i++)
		printf("%14.12lf %14.12lf\n", u[i], f[i]);

	status = DftiCreateDescriptor(&hand, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)(n + 1));
	if (status != DFTI_NO_ERROR) goto failed;

	printf("Set configuration: out-of-place\n");
	status = DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	if (status != DFTI_NO_ERROR) goto failed;

	printf("Set configuration: CCE storage\n");
	status = DftiSetValue(hand, DFTI_CONJUGATE_EVEN_STORAGE,
		DFTI_COMPLEX_COMPLEX);
//	if (status != DFTI_NO_ERROR) goto failed;

	/* This is not needed for DFTI_COMPLEX_COMPLEX storage */
	//status = DftiSetValue(hand, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT); 
	/* if (status != DFTI_NO_ERROR) goto failed; */

	status = DftiCommitDescriptor(hand);
	if (status != DFTI_NO_ERROR) goto failed;

	status = DftiComputeForward(hand, f, f_FFT);
	if (status != DFTI_NO_ERROR) goto failed;

	MyFFT1D_ForwardReal(n + 1, f, f_MYFFT);

	for (int i = 0; i < n + 1; i++)
		printf("f_FFT[%d]: %14.12lf + I * %14.12lf  f_MYFFT[%d]: %14.12lf + I * %14.12lf\n", i, f_FFT[i].real(), f_FFT[i].imag(),
			i, f_MYFFT[i].real(), f_MYFFT[i].imag());

	MyFFT1D_BackwardComplex(n + 1, f_MYFFT, u_MYFFT);

	status = DftiComputeBackward(hand, f_FFT, u_FFT);
	if (status != DFTI_NO_ERROR) goto failed;

	for (int i = 0; i < n + 1; i++)
		printf("u_backward[%d]: %14.12lf + I * %14.12lf  u_MKL[%d]: %14.12lf  u_ex[%d]: %14.12lf\n", i, u_MYFFT[i].real(), u_MYFFT[i].imag(),
			i, u_FFT[i] / (n + 1), i, f[i]);
	system("pause");


//	for (int i = 0; i < n + 1; i++) sum += f[i];
//	printf("f_FFT[0] analyt: %14.12lf\n", sum);


//	for (int i = 1; i < n + 1; i++)
//		f_FFT[i] /= -i * i; // - w * w

//	for (int nn = 0; nn < n + 1; nn++)
//		for (int k = 1; k < n + 1; k++)
	//		u_obt[nn] += sqrt(2.0 / (n + 1)) * sin(PI * nn * k / (n + 1)) * f_FFT[k];

	// Backward FFT of solution u
	//status = DftiComputeBackward(hand, f_FFT, u_FFT);
//	if (status != DFTI_NO_ERROR) goto failed;

	// Scale backward computation
	//for (int i = 0; i < n + 1; i++)
	//	u_FFT[i] /= n + 1; // - w * w

	dtype *u_pois_FFT = alloc_arr<dtype>(n + 1);
	Clear(n + 1, 1, u_pois_FFT, n + 1);
	status = DftiComputeForward(hand, u, u_pois_FFT);

	for (int i = 0; i < n + 1; i++)
		u_pois_FFT[i] *= -i * i; // - w * w

	for (int i = 0; i < n + 1; i++)
		printf("u_pois_FFT[%d]: %14.12lf + I * %14.12lf  f_FFT[%d]: %14.12lf + I * %14.12lf\n", i, u_pois_FFT[i].real(), u_pois_FFT[i].imag(),
																							 i,	f_FFT[i].real(), f_FFT[i].imag());


	//for (int i = 0; i < n + 1; i++)
//		printf("u_FFT[%d]: %14.12lf + I * %14.12lf  u[%d]: %14.12lf + I * %14.12lf\n", i, u_obt[i].real(), u_obt[i].imag(),
//			i, u[i], 0);


//	printf("Real\n");
//	for (int i = 0; i < n; i++)
	//	printf("%14.12lf %14.12lf\n", u_FFT[i], u[i]);

	//char str[255];
//	norm = rel_error(n, 1, &u_FFT[1], &u[1], n, eps);
//	sprintf(str, "Size: n = %d ", n);
//	AssertLess(norm, eps, str);

	printf("Free DFTI descriptor\n");
	DftiFreeDescriptor(&hand);
	return;

failed:
	printf("ERROR\n");
	return;
}

void Test_FFT1D_Complex(int n /* grid points in 1 dim */, double eps)
{
	DFTI_DESCRIPTOR_HANDLE my_desc1_handle;
	DFTI_DESCRIPTOR_HANDLE my_desc2_handle;
	DFTI_DESCRIPTOR_HANDLE hand = 0;
	MKL_LONG status;

	double size = 1.0;
	double h = size / (n);
	double norm;
	double sum = 0;

	dtype *f = alloc_arr<dtype>(n + 1);
	dtype *u = alloc_arr <dtype> (n + 1);
	dtype *u_obt = alloc_arr<dtype>(n + 1);
	dtype *f_FFT = alloc_arr<dtype>(n + 1);
	dtype *f_MYFFT = alloc_arr<dtype>(n + 1);

	dtype *u_FFT = alloc_arr<dtype>(n + 1);
	dtype *u_MYFFT = alloc_arr<dtype>(n + 1);


	// u_xx = f
	// u = sin(2 * PI * x) , f = - 4 * PI * PI * sin(2 * PI * x)

	for (int i = 0; i < n + 1; i++)
	{
		u[i] = { sin(PI * i * h), cos(i * h) };
		f[i] = { -PI * PI * sin(PI * i * h), -cos(i * h) };
		//	u[i] = (i * h) * (i * h) * (i * h);
		//	f[i] = 6 * (i * h);
	}


	status = DftiCreateDescriptor(&hand, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)(n + 1));
	if (status != DFTI_NO_ERROR) goto failed;

	printf("Set configuration: out-of-place\n");
	status = DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	if (status != DFTI_NO_ERROR) goto failed;

	status = DftiCommitDescriptor(hand);
	if (status != DFTI_NO_ERROR) goto failed;

	status = DftiComputeForward(hand, f, f_FFT);
	if (status != DFTI_NO_ERROR) goto failed;

	MyFFT1D_ForwardComplex(n + 1, f, f_MYFFT);

	for (int i = 0; i < n + 1; i++)
		printf("f_FFT[%d]: %14.12lf + I * %14.12lf  f_MYFFT[%d]: %14.12lf + I * %14.12lf\n", i, f_FFT[i].real(), f_FFT[i].imag(),
			i, f_MYFFT[i].real(), f_MYFFT[i].imag());

	MyFFT1D_BackwardComplex(n + 1, f_MYFFT, u_MYFFT);

	status = DftiComputeBackward(hand, f_FFT, u_FFT);
	if (status != DFTI_NO_ERROR) goto failed;

	for (int i = 0; i < n + 1; i++)
		printf("u_backward[%d]: %14.12lf + I * %14.12lf  u_MKL[%d]: %14.12lf + I * %14.12lf, u_ex[%d]:  %14.12lf + I * %14.12lf\n", i, u_MYFFT[i].real(), u_MYFFT[i].imag(),
			i, u_FFT[i].real() / (n + 1), u_FFT[i].imag() / (n + 1), i, f[i].real(), f[i].imag());

	system("pause");

	printf("Free DFTI descriptor\n");
	DftiFreeDescriptor(&hand);

	return;
failed:
	printf("ERROR\n");
	return;
}

#endif


void Test_LowRankApproxStruct(int m, int n, double eps, char *method)
{
	// A - matrix in dense order
	dtype *A = alloc_arr<dtype>(m * n);
	dtype *A_init = alloc_arr<dtype>(m * n);
	dtype *A_rec = alloc_arr<dtype>(m * n);
	char str[255];

	int lda = m;

	double norm = 0;
	dtype alpha = 1.0;
	dtype beta = 0.0;

	for (int j = 0; j < n; j++)
		for (int i = 0; i < m; i++)
		{
			A[i + lda * j] = dtype(1.0 / (n + i + j + 1), 1.0 / (i + j + 1));
			A_init[i + lda * j] = dtype(1.0 / (n + i + j + 1), 1.0 / (i + j + 1));
		}

#if 1
	cmnode *Astr = (cmnode*)malloc(sizeof(cmnode));
	//printf("Test for LowRankApproximationStruct m = %d n = %d ", m, n);
	LowRankApproxStruct(m, n, A, lda, Astr, eps, "SVD"); // memory allocation for Astr inside function
#else
	mnode *Astr;
	//printf("Test for LowRankApproximationStruct2 using return m = %d n = %d ", m, n);
	Astr = LowRankApproxStruct2(m, n, A, lda, eps, "SVD"); // memory allocation for Astr inside function
#endif

	zgemm("no", "no", &m, &n, &Astr->p, &alpha, Astr->U, &m, Astr->VT, &Astr->p, &beta, A_rec, &lda);

	norm = rel_error(zlange, m, n, A_rec, A_init, lda, eps);
	sprintf(str, "Struct: n = %d m = %d ", n, m);
	AssertLess(norm, eps, str);


	free_arr<dtype>(A);
	free_arr<dtype>(A_init);
	free_arr<dtype>(A_rec);
	free_arr<dtype>(Astr->U);
	free_arr<dtype>(Astr->VT);
	free_arr<cmnode>(Astr);
}

void Test_SymRecCompressStruct(int n, double eps, char *method, int smallsize)
{
	//printf("*****Test for SymRecCompressStruct  n = %d eps = %e ******* ", n, eps);
	char frob = 'F';
	double norm = 0;

	dtype *H = alloc_arr<dtype>(n * n); // init
	dtype *H1 = alloc_arr<dtype>(n * n); // compressed
	dtype *H2 = alloc_arr<dtype>(n * n); // recovered init

	int ldh = n;
	for (int j = 0; j < n; j++)
		for (int i = 0; i < n; i++)
		{
			H[i + ldh * j] = 1.0 / (i + j + 1);
			H1[i + ldh * j] = 1.0 / (i + j + 1);
		}

#ifdef DEBUG
	print(n, n, H1, ldh, "H1");
#endif

	cmnode *H1str; // pointer to the tree head
	SymRecCompressStruct(n, H1, ldh, H1str, smallsize, eps, "SVD"); // recursive function means recursive allocation of memory for structure fields
	SymResRestoreStruct(n, H1str, H2, ldh, smallsize);

#ifdef DEBUG
	//print(n, n, H1, ldh, "H1 compressed");
	print(n, n, H2, ldh, "H recovered");
#endif

	// Norm of residual || A - L * U ||
	norm = rel_error(zlange, n, n, H2, H, ldh, eps);

#ifdef DEBUG
	print(n, n, H, ldh, "H init");
	print(n, n, H2, ldh, "diff");
#endif

	char str[255];
	sprintf(str, "Struct: n = %d ", n);
	AssertLess(norm, eps, str);

	FreeNodes(n, H1str, smallsize);
	free_arr<dtype>(H);
	free_arr<dtype>(H2);
	free_arr<dtype>(H1);
}

void Test_DiagMultStruct(int n, double eps, char *method, int smallsize)
{
	//printf("*****Test for DiagMultStruct  n = %d ******* ", n);
	dtype *Hd = alloc_arr<dtype>(n * n); // diagonal Hd = D * H * D
	dtype *H1 = alloc_arr<dtype>(n * n); // compressed H
	dtype *H2 = alloc_arr<dtype>(n * n); // recovered H after D * H1 * D
	dtype *d = alloc_arr<dtype>(n);
	char str[255];

	double norm = 0;
	int ldh = n;

	for (int j = 0; j < n; j++)
	{
		d[j] = j + 1;
	}

	for (int j = 0; j < n; j++)
		for (int i = 0; i < n; i++)
		{
			Hd[i + ldh * j] = 1.0 / (i + j + 1);
			Hd[i + ldh * j] *= d[j];
			Hd[i + ldh * j] *= d[i];
			H1[i + ldh * j] = 1.0 / (i + j + 1);
		}
#ifdef DEBUG
	print(n, n, H1, ldh, "Initial H");
#endif

	cmnode *HCstr;
	// Compress H1 to structured form
	SymRecCompressStruct(n, H1, ldh, HCstr, smallsize, eps, method);

	// Compressed H1 = D * H * D
	DiagMultStruct(n, HCstr, d, smallsize);

	// Recove H1 to uncompressed form
	SymResRestoreStruct(n, HCstr, H2, ldh, smallsize);

#ifdef DEBUG
	print(n, n, Hd, ldh, "Initial Hd = D * H * D");
	print(n, n, H2, ldh, "Recovered H2 = (D * H * D)comp");
#endif

	// Compare Hd and H2
	norm = rel_error(zlange, n, n, H2, Hd, ldh, eps);

	sprintf(str, "Struct: n = %d ", n);
	AssertLess(norm, eps, str);

	FreeNodes(n, HCstr, smallsize);
	free_arr<dtype>(Hd); // diagonal Hd = D * H * D
	free_arr<dtype>(H1); // compressed H
	free_arr<dtype>(H2); // recovered H after D * H1 * D
	free_arr<dtype>(d);
}

/* Тест на сравнение результатов умножения Y = H * X сжимаемой матрицы H на произвольную X.
Сравниваются результаты со сжатием и без */
void Test_RecMultLStruct(int n, int k, double eps, char *method, int smallsize)
{
	//printf("*****Test for RecMultLStruct  n = %d k = %d ******* ", n, k);
	dtype *H = alloc_arr<dtype>(n * n); // init and compressed
	dtype *X = alloc_arr<dtype>(n * k);
	dtype *Y = alloc_arr<dtype>(n * k); // init Y
	dtype *Y1 = alloc_arr<dtype>(n * k); // after multiplication woth compressed
	char str[255];

	double norm = 0;
	dtype alpha = 1.0;
	dtype beta = 0.0;

	int ldh = n;
	int ldy = n;
	int ldx = n;

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			H[i + ldh * j] = 1.0 / (i + j + 1);

		for (int j = 0; j < k; j++)
			X[i + ldx * j] = 1.0 / (i + j + 1);
	
	}

	zgemm("No", "No", &n, &k, &n, &alpha, H, &ldh, X, &ldx, &beta, Y, &ldy);

#ifdef DEBUG
	print(n, n, H, ldy, "H init");
	print(n, k, X, ldy, "X init");
	print(n, k, Y, ldy, "Y init");
#endif

	cmnode *Hstr;
	// Compress H
	SymRecCompressStruct(n, H, ldh, Hstr, smallsize, eps, method);

	// RecMult Y1 = comp(H) * X
	RecMultLStruct(n, k, Hstr, X, ldx, Y1, ldy, smallsize);

	norm = rel_error(zlange, n, k, Y1, Y, ldy, eps);
	sprintf(str, "Struct: n = %d k = %d ", n, k);
	AssertLess(norm, eps, str);

#ifdef DEBUG
	print(n, n, H, ldy, "H comp");
	print(n, k, Y1, ldy, "Y1 rec");
#endif

	FreeNodes(n, Hstr, smallsize);
	free_arr<dtype>(H);
	free_arr<dtype>(X);
	free_arr<dtype>(Y);
	free_arr<dtype>(Y1);
}

void Test_AddStruct(int n, dtype alpha, dtype beta, double eps, char *method, int smallsize)
{
	//printf("*****Test for Add n = %d ******* ", n);
	dtype *H1 = alloc_arr<dtype>(n * n);
	dtype *H2 = alloc_arr<dtype>(n * n);
	dtype *G = alloc_arr<dtype>(n * n);
	dtype *H1c = alloc_arr<dtype>(n * n);
	dtype *H2c = alloc_arr<dtype>(n * n);
	dtype *Gc = alloc_arr<dtype>(n * n);
	dtype *GcR = alloc_arr<dtype>(n * n);
	char str[255];

	int ldh = n;
	int ldg = n;
	double norm = 0;

#pragma omp parallel for simd schedule(static)
	for (int j = 0; j < n; j++)
		for (int i = 0; i < n; i++)
		{
			H1[i + ldh * j] = 1.0 / (i + j + 1);
			H2[i + ldh * j] = 1.0 / (i*i + j*j + 1);
			H1c[i + ldh * j] = 1.0 / (i + j + 1);
			H2c[i + ldh * j] = 1.0 / (i*i + j*j + 1);
		}

#ifdef DEBUG
	print(n, n, H1, ldh, "H1");
	print(n, n, H2, ldh, "H2");
#endif

	cmnode *H1str, *H2str;
	SymRecCompressStruct(n, H1c, ldh, H1str, smallsize, eps, method);
	SymRecCompressStruct(n, H2c, ldh, H2str, smallsize, eps, method);

#ifdef DEBUG
	print(n, n, H1c, ldh, "H1c");
	print(n, n, H2c, ldh, "H2c");
#endif

	cmnode *Gstr;
	Add_dense(n, n, alpha, H1, ldh, beta, H2, ldh, G, ldg);
	AddStruct(n, alpha, H1str, beta, H2str, Gstr, smallsize, eps, method);

#ifdef DEBUG
	print(n, n, G, ldg, "res_dense");
	print(n, n, Gc, ldg, "res_comp");
#endif

	SymResRestoreStruct(n, Gstr, GcR, ldg, smallsize);

#ifdef DEBUG
	print(n, n, GcR, ldg, "res_comp_restore");
#endif
	// |GcR - G| / |G|
	norm = rel_error(zlange, n, n, GcR, G, ldg, eps);
	sprintf(str, "Struct: n = %d n = %d alpha = %lf", n, n, alpha, beta);
	AssertLess(norm, eps, str);

	FreeNodes(n, H1str, smallsize);
	FreeNodes(n, H2str, smallsize);
	FreeNodes(n, Gstr, smallsize);
	free_arr(H1);
	free_arr(H2);
	free_arr(G);
	free_arr(H1c);
	free_arr(H2c);
	free_arr(Gc);
	free_arr(GcR);
}

// B = H - V * Y * VT
void Test_SymCompUpdate2Struct(int n, int k, dtype alpha, double eps, char* method, int smallsize)
{
	//printf("*****Test for SymCompUpdate2Struct  n = %d k = %d ***** ", n, k);
	dtype *B = alloc_arr<dtype>(n * n); int ldb = n;
	dtype *B_rec = alloc_arr<dtype>(n * n);
	dtype *Y = alloc_arr<dtype>(k * k); int ldy = k;
	dtype *V = alloc_arr<dtype>(n * k); int ldv = n; int ldvtr = k;
	dtype *HC = alloc_arr<dtype>(n * n); int ldh = n;
	dtype *H = alloc_arr<dtype>(n * n);
	dtype *C = alloc_arr<dtype>(n * k); int ldc = n;
	char str[255];

	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;
	double norm = 0;


	Hilbert(n, HC, ldh);
	Hilbert(n, H, ldh);

#pragma omp parallel for simd schedule(static)
	for (int i = 0; i < k; i++)
		Y[i + ldy * i] = i + 1;

#pragma omp parallel for simd schedule(static)
	for (int j = 0; j < k; j++)
		for (int i = 0; i < n; i++)
			V[i + ldv * j] = (i + j + 1);

	// C = V * Y
	zsymm("Right", "Up", &n, &k, &alpha_one, Y, &ldy, V, &ldv, &beta_zero, C, &ldc);

	// H = H + alpha * C * VT
	zgemm("No", "Trans", &n, &n, &k, &alpha, C, &ldc, V, &ldv, &beta_one, H, &ldh);

	cmnode *HCstr;
	// Compressed update
	SymRecCompressStruct(n, HC, ldh, HCstr, smallsize, eps, method);

	cmnode *Bstr;
	SymCompUpdate2Struct(n, k, HCstr, alpha, Y, ldy, V, ldv, Bstr, smallsize, eps, method);
	SymResRestoreStruct(n, Bstr, B_rec, ldh, smallsize);

#ifdef DEBUG
	print(n, n, B_rec, ldb, "B_rec");
	print(n, n, H, ldh, "H");
#endif

	// || B_rec - H || / || H ||
	norm = rel_error(zlange, n, n, B_rec, H, ldh, eps);
	sprintf(str, "Struct: n = %d k = %d alpha = %lf", n, k, alpha);
	AssertLess(norm, eps, str);

	FreeNodes(n, Bstr, smallsize);
	FreeNodes(n, HCstr, smallsize);
	free_arr(B);
	free_arr(B_rec);
	free_arr(H);
	free_arr(HC);
	free_arr(Y);
	free_arr(C);
	free_arr(V);
}

void Test_SymCompRecInvStruct(int n, double eps, char *method, int smallsize)
{
	//printf("***** Test_SymCompRecInvStruct n = %d eps = %lf ****", n, eps);
	dtype *H = alloc_arr<dtype>(n * n);
	dtype *Hc = alloc_arr<dtype>(n * n);
	dtype *Bc = alloc_arr<dtype>(n * n);
	dtype *Brec = alloc_arr<dtype>(n * n);
	dtype *Y = alloc_arr<dtype>(n * n);
	char str[255];

	int ldh = n;
	int ldb = n;
	int ldy = n;

	dtype alpha_mone = -1.0;
	dtype beta_one = 1.0;
	double norm = 0;

	Hilbert(n, H, ldh);
	Hilbert(n, Hc, ldh);

	// for stability
	for (int i = 0; i < n; i++)
	{
		H[i + ldh * i] += 1.0;
		Hc[i + ldh * i] += 1.0;
	}

	cmnode *HCstr, *BCstr;
	SymRecCompressStruct(n, Hc, ldh, HCstr, smallsize, eps, method);
	SymCompRecInvStruct(n, HCstr, BCstr, smallsize, eps, method);
	SymResRestoreStruct(n, BCstr, Brec, ldb, smallsize);

	Eye(n, Y, ldy);

	// Y = Y - H * Brec
	zgemm("No", "No", &n, &n, &n, &alpha_mone, H, &ldh, Brec, &ldb, &beta_one, Y, &ldy);

	norm = zlange("Frob", &n, &n, Y, &ldy, NULL);
	sprintf(str, "Struct: n = %d", n);
	AssertLess(norm, eps, str);

	//if (norm < eps) printf("Norm %10.8e < eps %10.8lf: PASSED\n", norm, eps);
	//else printf("Norm %10.8lf > eps %10.8e : FAILED\n", norm, eps);

	FreeNodes(n, HCstr, smallsize);
	FreeNodes(n, BCstr, smallsize);
	free_arr(H);
	free_arr(Hc);
	free_arr(Bc);
	free_arr(Brec);
	free_arr(Y);
}

void Test_CopyStruct(int n, double eps, char *method, int smallsize)
{
	dtype *H = alloc_arr<dtype>(n * n);
	dtype *H1 = alloc_arr<dtype>(n * n);
	dtype *H2 = alloc_arr<dtype>(n * n);
	char str[255];

	double norm = 0;
	int ldh = n;

	//printf("***Test CopyStruct n = %d ", n);

	Hilbert(n, H, ldh);
	Hilbert(n, H1, ldh);

	cmnode* Hstr, *Hcopy_str;
	SymRecCompressStruct(n, H, ldh, Hstr, smallsize, eps, method);
	CopyStruct(n, Hstr, Hcopy_str, smallsize);
	SymResRestoreStruct(n, Hcopy_str, H2, ldh, smallsize);

	norm = rel_error(zlange, n, n, H2, H1, ldh, eps);
	sprintf(str, "Struct: n = %d", n);
	AssertLess(norm, eps, str);

	FreeNodes(n, Hstr, smallsize);
	FreeNodes(n, Hcopy_str, smallsize);
	free_arr(H2);
	free_arr(H1);
	free_arr(H);
}
#if 0
void Test_DirFactFastDiagStructOnline(size_m x, size_m y, size_m z, cmnode** Gstr, dtype *B, double eps, int smallsize)
{
	printf("Testing factorization...\n");
	int n = x.n * y.n;
	int size = n * z.n;
	char bench[255] = "No";
	dtype *DD = alloc_arr<double>(n * n); int lddd = n;
	dtype *DR = alloc_arr<dtype>(n * n); int lddr = n;
	double norm = 0;

	double timer = 0;
	timer = omp_get_wtime();

	GenerateDiagonal2DBlock(0, x, y, z, DD, lddd);

	cmnode *DCstr;
	SymCompRecInvStruct(n, Gstr[0], DCstr, smallsize, eps, "SVD");
	SymResRestoreStruct(n, DCstr, DR, lddr, smallsize);

	printf("Block %d. ", 0);
	norm = rel_error(n, n, DR, DD, lddd, eps);

	if (norm < eps) printf("Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
	else printf("Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);

	free_arr(DR);
	FreeNodes(n, DCstr, smallsize);

	for (int k = 1; k < z.n; k++)
	{
		dtype *DR = alloc_arr<dtype>(n * n); int lddr = n;
		dtype *HR = alloc_arr<dtype>(n * n); int ldhr = n;
		cmnode *DCstr, *Hstr;

		printf("Block %d. ", k);

		SymCompRecInvStruct(n, Gstr[k], DCstr, smallsize, eps, "SVD");
		SymResRestoreStruct(n, DCstr, DR, lddr, smallsize);

		CopyStruct(n, Gstr[k - 1], Hstr, smallsize);
		DiagMultStruct(n, Hstr, &B[ind(k - 1, n)], smallsize);
		SymResRestoreStruct(n, Hstr, HR, ldhr, smallsize);

		// Norm of residual
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
#pragma omp simd
			for (int i = 0; i < n; i++)
				HR[i + ldhr * j] = HR[i + ldhr * j] + DR[i + lddr * j];

		norm = rel_error(n, n, HR, DD, lddd, eps);

		if (norm < eps) printf("Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
		else printf("Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);

		FreeNodes(n, DCstr, smallsize);
		FreeNodes(n, Hstr, smallsize);
		free_arr(DR);
		free_arr(HR);
	}
	timer = omp_get_wtime() - timer;
	printf("Time: %lf\n", timer);

	free_arr(DD);

}
#endif



#if 0




void Test_DirSolveFactDiagStructConvergence(size_m x, size_m y, size_m z, mnode** Gstr, double thresh, int smallsize)
{
	printf("------------Test convergence-----------\n");
	int n = x.n * y.n;
	double norm = 0;
	for (int i = 1; i < z.n; i++)
	{
		double *GR1 = alloc_arr(n * n); int ldg1 = n;
		double *GR2 = alloc_arr(n * n); int ldg2 = n;
		double *GRL = alloc_arr(n * n); int ldgl = n;
		printf("For block %2d. \n", i);

		SymResRestoreStruct(n, Gstr[i], GR2, ldg2, smallsize);
		SymResRestoreStruct(n, Gstr[i - 1], GR1, ldg1, smallsize);
		norm = rel_error(n, n, GR2, GR1, ldg1, thresh);
		printf("norm |G[%2d] - G[%2d]|/|G[%2d]| = %12.10lf\n", i - 1, i, i - 1, norm);

		SymResRestoreStruct(n, Gstr[z.n - 1], GRL, ldgl, smallsize);
		norm = rel_error(n, n, GR1, GRL, ldgl, thresh);
		printf("norm |G[%2d] - G[%2d]|/|G[%2d]| = %12.10lf\n\n", i - 1, z.n - 1, z.n - 1, norm);

		free_arr(&GR1);
		free_arr(&GR2);
		free_arr(&GRL);
	}
}

void Test_DirSolveFactDiagStructBlockRanks(size_m x, size_m y, size_m z, mnode** Gstr)
{
	printf("----------Trees information-----------\n");
	int *size = (int*)malloc(x.n * sizeof(int));
	int *depth = (int*)malloc(y.n * sizeof(int));

	double time = omp_get_wtime();
#pragma omp parallel
	{
#pragma omp single
		for (int i = 0; i < z.n; i++)
		{
			size[i] = TreeSize(Gstr[i]);
			depth[i] = MaxDepth(Gstr[i]);
		}
	}
	double result = omp_get_wtime() - time;
	printf("Computational time of TreeSize and MaxDepth for all %d trees: %lf\n", x.n, result);

	for (int i = 0; i < z.n; i++)
	{
		printf("For block %2d. Size: %d, MaxDepth: %d, Ranks: ", i, size[i], depth[i]);
		PrintRanksInWidthList(Gstr[i]);
		printf("\n");
	}

	free(size);
	free(depth);

}

void Test_RankEqual(mnode *Astr, mnode *Bstr)
{
	try
	{
		char str[255] = "Rank(A01) = Rank(B01) ";
		AssertEqual(Astr->p, Bstr->p, str);
	}
	catch (exception &ex)
	{
		cout << ex.what();
	}

	if (Astr->left != NULL || Bstr->left != NULL)
	{
		Test_RankEqual(Astr->left, Bstr->left);
	}

	if (Astr->right != NULL || Bstr->right != NULL)
	{
		Test_RankEqual(Astr->right, Bstr->right);
	}
}

void Test_RankAdd(mnode *Astr, mnode *Bstr, mnode *Cstr)
{
	try
	{
		char str[255] = "Rank(C01) <= Rank(A01) + Rank(B01) ";
		AssertLess(Astr->p + Bstr->p, Cstr->p, str);
	}
	catch (exception &ex)
	{
		cout << ex.what();
	}

}

void Test_Dense_to_CSR(size_m x, size_m y, size_m z, int non_zeros_in_3diag, double *D, int ldd)
{
	int n = x.n * y.n;
	printf("non_zeros: %d\n", non_zeros_in_3diag);
	double *values = alloc_arr(non_zeros_in_3diag);
	int *ia = (int*)malloc(non_zeros_in_3diag * sizeof(int));
	int *ja = (int*)malloc(non_zeros_in_3diag * sizeof(int));
	map<vector<int>, double> CSR;
	CSR = dense_to_CSR(n, n, &D[0], ldd, ia, ja, values);
	print(n, n, &D[0], ldd, "D[0]");
	print_map(CSR);
	free(ia);
	free(ja);
	free(values);

	/*
	print(n, n, &B_mat[0], ldb, "B[0]");
	print(size, n, D, ldd, "D");

	printf("all non_zero elements: %d\n", non_zeros_in_block3diag);
	for (int i = 0; i < size + 1; i ++)
	printf("%d: ia = %d value(ia) = %lf diff = %d\n", i, Dcsr->ia[i], Dcsr->values[Dcsr->ia[i] - 1], Dcsr->ia[i+1]- Dcsr->ia[i]);

	print_vec(non_zeros_in_block3diag, Dcsr->ja, Dcsr->values, "ja and values");
	print_map(CSR);*/

}

void Test_QueueList(int n, double eps, char* method, int smallsize)
{
	printf("----Test for queue implementation with LIST---\n n = %d \n", n);
	// A - matrix in dense order
	double *A = alloc_arr(n * n);
	double *A_init = alloc_arr(n * n);
	int lda = n;

	for (int j = 0; j < n; j++)
		for (int i = 0; i < n; i++)
		{
			A[i + lda * j] = 1.0 / (i + j + 1);
			A_init[i + lda * j] = 1.0 / (i + j + 1);
		}

	mnode *ACstr;
	SymRecCompressStruct(n, A, lda, ACstr, smallsize, eps, method);

	//printf("first level. p = %d\n", ACstr->p);
	//printf("second level. left: %d right: %d\n", Astr->left->p, Astr->right->p);

	printf("Size: %d, MaxDepth: %d, Ranks: ", TreeSize(ACstr), MaxDepth(ACstr));
	//	PrintRanksInWidth(ACstr);
	printf("List:\n");
	PrintRanksInWidthList(ACstr);

	FreeNodes(n, ACstr, smallsize);
	free_arr(&A);
	free_arr(&A_init);
}

void Test_CompareColumnsOfMatrix(int n1, int n2, int n3, double* D, int ldd, double* B, dcsr* Dcsr, double thresh)
{
	int n = n1 * n2;
	int size = n * n3;
	double RelRes = 0;
	double *g = alloc_arr(size);
	double *f1 = alloc_arr(size);
	double *f2 = alloc_arr(size);
	double Res1 = 0;
	double Res2 = 0;

	for (int i = 0; i < size; i++)
	{
		double *x_test = alloc_arr(size);
		x_test[i] = 1;
		Mult_Au(n1, n2, n3, D, ldd, B, x_test, f1);
		mkl_dcsrgemv("No", &size, Dcsr->values, Dcsr->ia, Dcsr->ja, x_test, f2);
		//	print_vec(size, f1, f2, "f1 and f2");
		rel_error(size, 1, f1, f2, size, thresh);
		free(x_test);
	}

	free_arr(&f1);
	free_arr(&f2);
	free_arr(&g);
}
#endif


#endif