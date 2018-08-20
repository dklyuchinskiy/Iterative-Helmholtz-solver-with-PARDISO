#include "definitions.h"
#include "templates.h"
#include "TestSuite.h"
#include "TemplatesForMatrixConstruction.h"

/***************************************************
Test for solving Laplace equation with Dirichlet
boundary conditions on the grid n1 x n2 x n3 points 
for the domain x.l x y.l x z.l with HSS technique.

The known solution is generated by function u_ex().
Then we check how correct we has constructed
coefficient matrix as : ||A * u_ex - F_ex|| < eps.

If test is passed, we run the solver for matrix A
and right hand side F.

In the output we compare the relative norm of
the solutuon as:
||u_ex - u_sol|| / ||u_ex|| < eps.

The two version of solver is enabled:
a) with storing the result of factorization in
the array G of doulbe
b) with storing the result of factorization
in the set of structures Gstr, which is defined 
in the definitions.h

The second variant, also, is supported by
storing the initial coefficient matrix A
in sparse CSR format to save memory.

*************************************************/

#if 0
int main()
{
	TestAll();
	system("pause");
#if 1
	int n1 = 40;		    // number of point across the directions
	int n2 = 40;
	int n3 = 40;
	int n = n1 * n2;		// size of blocks
	int NB = n3;			// number of blocks
	int size = n * NB;		// size of vector x and f: n1 * n2 * n3
	int smallsize = 400;
	double thresh = 1e-6;	// stop level of algorithm by relative error
	int ItRef = 200;		// Maximal number of iterations in refirement
	char bench[255] = "display"; // parameter into solver to show internal results
	int sparse_size = n + 2 * (n - 1) + 2 * (n - n1);
	int non_zeros_in_3diag = n + (n - 1) * 2 + (n - n1) * 2 - (n1 - 1) * 2;

	size_m x, y, z;

	x.n = n1;
	y.n = n2;
	z.n = n3;

	x.l = y.l = z.l = n1 + 1;
	x.h = x.l / (double)(x.n + 1);
	y.h = y.l / (double)(y.n + 1);
	z.h = z.l / (double)(z.n + 1);

	dtype *D;
	dtype *B_mat;

	// Memory allocation for coefficient matrix A
	// the size of matrix A: n^3 * n^3 = n^6
#ifndef ONLINE
	D = alloc_arr(size * n); // it's a matrix with size n^3 * n^2 = size * n
	B_mat = alloc_arr((size - n) * n); 
	int ldd = size;
	int ldb = size - n;
#else
	D = alloc_arr<dtype>(n * n); // it's a matrix with size n^3 * n^2 = size * n
	B_mat = alloc_arr<dtype>(n * n);
	int ldd = n;
	int ldb = n;
#endif

	// Factorization matrix
#ifndef STRUCT_CSR
	double *G = alloc_arr(size * n);
	int ldg = size;
#else
	cmnode **Gstr;
#endif

	// Solution, right hand side and block B
	dtype *B = alloc_arr<dtype>(size - n); // vector of diagonal elementes
	dtype *x_orig = alloc_arr<dtype>(size);
	dtype *x_sol = alloc_arr<dtype>(size);
	dtype *f = alloc_arr<dtype>(size);

#ifdef STRUCT_CSR
	// Memory for CSR matrix
	dcsr *Dcsr;
	int non_zeros_in_block3diag = (n + (n - 1) * 2 + (n - x.n) * 2 - (x.n - 1) * 2) * z.n + 2 * (size - n);
	Dcsr = (dcsr*)malloc(sizeof(dcsr));
	Dcsr->values = alloc_arr<dtype>(non_zeros_in_block3diag);
	Dcsr->ia = alloc_arr<int>(size + 1);
	Dcsr->ja = alloc_arr<int>(non_zeros_in_block3diag);
	Dcsr->ia[size] = non_zeros_in_block3diag + 1;
#endif

	int success = 0;
	int itcount = 0;
	double RelRes = 0;
	double norm = 0;
	int nthr = omp_get_max_threads();
	
	printf("Run in parallel on %d threads\n", nthr);
		
	printf("Grid steps: hx = %lf hy = %lf hz = %lf\n", x.h, y.h, z.h);

#ifndef STRUCT_CSR
	// Generation matrix of coefficients, vector of solution (to compare with obtained) and vector of RHS
	GenMatrixandRHSandSolution(n1, n2, n3, D, ldd, B, x_orig, f);
#else

	// Generation of vector of solution (to compare with obtained), vector of RHS and block B
	GenRHSandSolution(x, y, z, B, x_orig, f);

	// Generation of sparse coefficient matrix
#ifndef ONLINE
	GenSparseMatrix(x, y, z, B_mat, ldb, D, ldd, B_mat, ldb, Dcsr);
#else
	GenSparseMatrixOnline(x, y, z, B_mat, n, D, n, B_mat, n, Dcsr);
	free_arr(D);
#endif
	free_arr(B_mat);

	printf("Non_zeros in block-tridiagonal matrix: %d\n", non_zeros_in_block3diag);

	//	Test_CompareColumnsOfMatrix(n1, n2, n3, D, ldd, B, Dcsr, thresh);
	Test_TransferBlock3Diag_to_CSR(n1, n2, n3, Dcsr, x_orig, f, thresh);
#endif

	printf("Solving %d x %d x %d Laplace equation\n", n1, n2, n3);
	printf("The system has %d diagonal blocks of size %d x %d\n", n3, n1*n2, n1*n2);
	printf("Compressed blocks method\n");
	printf("Parameters: thresh = %g, smallsize = %d \n", thresh, smallsize);

	// Calling the solver
	
#ifndef STRUCT_CSR
	Block3DSPDSolveFast(n1, n2, n3, D, ldd, B, f, thresh, smallsize, ItRef, bench, G, ldg, x_sol, success, RelRes, itcount);
#else

#ifndef ONLINE
	Block3DSPDSolveFastStruct(x, y, z, D, ldd, B, f, Dcsr, thresh, smallsize, ItRef, bench, Gstr, x_sol, success, RelRes, itcount);
#else
	Block3DSPDSolveFastStruct(x, y, z, NULL, ldd, B, f, Dcsr, thresh, smallsize, ItRef, bench, Gstr, x_sol, success, RelRes, itcount);
#endif

#endif
	printf("success = %d, itcount = %d\n", success, itcount);
	printf("-----------------------------------\n");

	printf("Computing error ||x_{exact}-x_{comp}||/||x_{exact}||\n");
	norm = rel_error_complex(n, 1, x_sol, x_orig, size, thresh);

	if (norm < thresh) printf("Norm %12.10e < eps %12.10lf: PASSED\n", norm, thresh);
	else printf("Norm %12.10lf > eps %12.10lf : FAILED\n", norm, thresh);


#ifdef STRUCT_CSR
	Test_DirFactFastDiagStructOnline(x, y, z, Gstr, B, thresh, smallsize);
	//Test_DirSolveFactDiagStructConvergence(x, y, z, Gstr, thresh, smallsize);
	//Test_DirSolveFactDiagStructBlockRanks(x, y, z, Gstr);

	for (int i = z.n - 1; i >= 0; i--)
		FreeNodes(n, Gstr[i], smallsize);

	free(Gstr);
#endif


#ifndef ONLINE
	free_arr(D);
	free_arr(B);
#endif
	free_arr(x_orig);
	free_arr(x_sol);
	free_arr(f);

	system("pause");

	return 0;
#endif
}

#else

#if 1
int main()
{
	TestAll();
//	system("pause");
//	return 0;
#if 1

#ifdef PML
	int pml_pts = 15;
#else
	int pml_pts = 0;
#endif
	int pml_size = 2 * pml_pts;

	size_m x, y, z;
	size_m x_nopml, y_nopml, z_nopml;

	x.pml_pts = y.pml_pts = pml_pts;
	z.pml_pts = 0;

	x_nopml.pml_pts = y_nopml.pml_pts = z_nopml.pml_pts = 0;

	int n1 = 29 + 2 * x.pml_pts;		    // number of point across the directions
	int n2 = 29 + 2 * y.pml_pts;
	int n3 = 29 + 2 * z.pml_pts;
	int n = n1 * n2;		// size of blocks
	int NB = n3;			// number of blocks

	x.n = n1;
	y.n = n2;
	z.n = n3;

	int size = n * NB;		// size of vector x and f: n1 * n2 * n3
	int smallsize = 1600;
	double thresh = 1e-6;	// stop level of algorithm by relative error
	int ItRef = 200;		// Maximal number of iterations in refirement
	char bench[255] = "display"; // parameter into solver to show internal results
	int sparse_size = n + 2 * (n - 1) + 2 * (n - n1);
	int non_zeros_in_3diag = n + (n - 1) * 2 + (n - n1) * 2 - (n1 - 1) * 2;
	int ione = 1;
	bool pml_flag;

	double timer1, timer2, all_time;

	x.n_nopml = n1 - 2 * x.pml_pts;
	y.n_nopml = n2 - 2 * y.pml_pts;
	z.n_nopml = n3 - 2 * z.pml_pts;

	x_nopml.n = y_nopml.n = z_nopml.n = x.n_nopml;

	x_nopml.n_nopml = y_nopml.n_nopml = z_nopml.n_nopml = x_nopml.n;

	x.l = LENGTH + (double)(2 * x.pml_pts * LENGTH) / (x.n_nopml + 1);
	y.l = LENGTH + (double)(2 * y.pml_pts * LENGTH) / (y.n_nopml + 1);
	z.l = LENGTH + (double)(2 * z.pml_pts * LENGTH) / (z.n_nopml + 1);

	x.h = x.l / (x.n + 1);  // x.n + 1 grid points of the whole domain
	y.h = y.l / (y.n + 1);  // x.n - 1 - inner points
	z.h = z.l / (z.n + 1);  // 2 points - for the boundaries

	printf("Size of domain: Lx = %lf, Ly = %lf, Lz = %lf\n", x.l, y.l, z.l);
	printf("with points: Nx = %d, Ny = %d, Nz = %d\n", x.n, y.n, z.n);
	printf("Size of physical domain: Nx = %d, Ny = %d, Nz = %d\n", x.n_nopml, y.n_nopml, z.n_nopml);
	printf("Size of PML domain: Nx = %d, Ny = %d, Nz = %d\n", 2 * x.pml_pts, 2 * y.pml_pts, 2 * z.pml_pts);
	printf("Steps for physical domain: Hx = %lf, hy = %lf, hz = %lf\n", x.h, y.h, z.h);


	// ��� ���������� ���� � 2 ������ ������ ����������� � 4 ����!
	// 3D ������
	// ���������� - � ����������x
	// h = 10, 1280 x 1280, N = 120 - 2 �����
	// 40 ����� h = 30, L = 600, omega = 4, 6, 10

	TestHankel();
	system("pause");

	double ppw = (double)(c_z) / omega / z.h;

	printf("ppw: %lf\n", ppw);

	// Solution, right hand side and block B
	dtype *x_orig = alloc_arr<dtype>(size);
	dtype *f = alloc_arr<dtype>(size);
	dtype *f_FFT = alloc_arr<dtype>(size);
	dtype *x_sol_prd = alloc_arr<dtype>(size);
	//dtype *u2Dsynt = alloc_arr<dtype>(size);


	int n_nopml = x.n_nopml * y.n_nopml;
	int size_nopml = n_nopml * z.n_nopml;
	int size2D_nopml = n_nopml;

	int success = 0;
	int itcount = 0;
	double RelRes = 0;
	double norm = 0;

	dtype *x_orig_nopml = alloc_arr<dtype>(size_nopml);
	dtype *x_pard_nopml = alloc_arr<dtype>(size_nopml);
	dtype *x_pard_nopml_cpy = alloc_arr<dtype>(size_nopml);

	dtype *x_sol = alloc_arr<dtype>(size_nopml);
	dtype *x_sol_fft_nopml = alloc_arr<dtype>(size2D_nopml * n3);

	
	ccsr *Dcsr_nopml;
	int non_zeros_nopml = (n_nopml + (n_nopml - 1) * 2 + (n_nopml - x.n_nopml) * 2 - (y.n_nopml - 1) * 2) * z.n_nopml + 2 * (size_nopml - n_nopml);
	Dcsr_nopml = (ccsr*)malloc(sizeof(ccsr));
	Dcsr_nopml->values = alloc_arr<dtype>(non_zeros_nopml);
	Dcsr_nopml->ia = alloc_arr<int>(size_nopml + 1);
	Dcsr_nopml->ja = alloc_arr<int>(non_zeros_nopml);
	Dcsr_nopml->ia[size_nopml] = non_zeros_nopml + 1;
	Dcsr_nopml->non_zeros = non_zeros_nopml;

	ccsr *Dcsr_reduced;
	Dcsr_reduced = (ccsr*)malloc(sizeof(ccsr));
	Dcsr_reduced->values = alloc_arr<dtype>(non_zeros_nopml);
	Dcsr_reduced->ia = alloc_arr<int>(size_nopml + 1);
	Dcsr_reduced->ja = alloc_arr<int>(non_zeros_nopml);
	Dcsr_reduced->ia[size_nopml] = non_zeros_nopml + 1;
	Dcsr_reduced->non_zeros = non_zeros_nopml;


	x_nopml.l = y_nopml.l = z_nopml.l = (double)(LENGTH);
	x_nopml.h = x_nopml.l / (double)(x_nopml.n + 1);  // x.n + 1 grid points of the whole domain
	y_nopml.h = y_nopml.l / (double)(y_nopml.n + 1);  // x.n - 1 - inner points
	z_nopml.h = z_nopml.l / (double)(z_nopml.n + 1);  // 2 points - for the boundaries


	system("pause");


#define GEN_3D_MATRIX

#ifdef GEN_3D_MATRIX

	int non_zeros_in_3Dblock3diag = (n + (n - 1) * 2 + (n - x.n) * 2 - (y.n - 1) * 2) * z.n + 2 * (size - n);

	dtype *D, *D_nopml;
	dtype *B_mat, *B_mat_nopml;

	dtype *B = alloc_arr<dtype>(size - n); // vector of diagonal elementes
	dtype *B_nopml = alloc_arr<dtype>(size_nopml - n_nopml); // vector of diagonal elementes

	dtype *x_pard = alloc_arr<dtype>(size);

	// Memory for 3D CSR matrix
	ccsr *Dcsr;

	Dcsr = (ccsr*)malloc(sizeof(ccsr));
	Dcsr->values = alloc_arr<dtype>(non_zeros_in_3Dblock3diag);
	Dcsr->ia = alloc_arr<int>(size + 1);
	Dcsr->ja = alloc_arr<int>(non_zeros_in_3Dblock3diag);
	Dcsr->ia[size] = non_zeros_in_3Dblock3diag + 1;
	Dcsr->non_zeros = non_zeros_in_3Dblock3diag;

	// Memory allocation for coefficient matrix A
	// the size of matrix A: n^3 * n^3 = n^6

	D = alloc_arr<dtype>(n * n); // it's a matrix with size n^3 * n^2 = size * n
	B_mat = alloc_arr<dtype>(n * n);

	D_nopml = alloc_arr<dtype>(n_nopml * n_nopml);
	B_mat_nopml = alloc_arr<dtype>(n_nopml * n_nopml);

	int ldd = n;
	int ldb = n;
#endif

#ifdef _OPENMP
	int nthr = omp_get_max_threads();
	printf("Max_threads: %d threads\n", nthr);
	omp_set_dynamic(0);
	nthr = 2;
	omp_set_num_threads(nthr);
	printf("Run in parallel on %d threads\n", nthr);
#else
	printf("Run sequential version on 1 thread\n");
#endif

	printf("Grid steps: hx = %lf hy = %lf hz = %lf\n", x.h, y.h, z.h);

	all_time = omp_get_wtime();

#ifndef STRUCT_CSR
	// Generation matrix of coefficients, vector of solution (to compare with obtained) and vector of RHS
	GenMatrixandRHSandSolution(n1, n2, n3, D, ldd, B, x_orig, f);
#else

	// Generation of vector of solution (to compare with obtained) and vector of RHS
	printf("Gen right-hand side and solution...\n");
	GenRHSandSolution(x, y, z, x_orig, f);


	// Generation of sparse coefficient matrix
#ifdef GEN_3D_MATRIX
	//printf("--------------- Gen sparse 3D matrix in CSR format... ---------------\n");
	//GenSparseMatrixOnline3D(x, y, z, B, B_mat, n, D, n, B_mat, n, Dcsr);


#if 1
	printf("-------------- Gen sparse 3D matrix in CSR format with PML... ------------\n");
	timer1 = omp_get_wtime();
	GenSparseMatrixOnline3DwithPML(x, y, z, B, B_mat, n, D, n, B_mat, n, Dcsr, thresh);
	timer2 = omp_get_wtime() - timer1;

	printf("Time of GenSparseMatrixOnline3DwithPML: %lf\n", timer2);

	printf("-------------- Gen sparse 3D matrix in CSR format with no PML... ------------\n");
	timer1 = omp_get_wtime();
	GenSparseMatrixOnline3DwithPML(x_nopml, y_nopml, z_nopml, B_nopml, B_mat_nopml, n_nopml, D_nopml, n_nopml, B_mat_nopml, n_nopml, Dcsr_nopml, thresh);
	timer2 = omp_get_wtime() - timer1;

	printf("Time of GenSparseMatrixOnline3DnoPML: %lf\n", timer2);
#endif

	free_arr(D);
	free_arr(B_mat);
#endif

	printf("Analytic non_zeros in first row and last two 2D blocks: %d\n", non_zeros_in_3diag + n);
	printf("Analytic non_zeros in three 2D block-row: %d\n", non_zeros_in_3diag + 2 * n);
	printf("Analytic non_zeros in 3D block-tridiagonal matrix: %d\n", non_zeros_in_3Dblock3diag);

	// Test PML
	printf("----------------- Running test PML... -----------\n");
	timer1 = omp_get_wtime();
	Test_PMLBlock3Diag_in_CSR(x, y, z, Dcsr, Dcsr_nopml, Dcsr_reduced, thresh);
	timer2 = omp_get_wtime() - timer1;
	printf("Time of Test_PMLBlock3Diag_in_CSR: %lf\n", timer2);

	system("pause");

#ifdef SOLVE_3D_PROBLEM



	//	Test_CompareColumnsOfMatrix(n1, n2, n3, D, ldd, B, Dcsr, thresh);
//	printf("--------------- Running test CSR... ----------------\n");
	timer1 = omp_get_wtime();
	//Test_TransferBlock3Diag_to_CSR(x, y, z, Dcsr, x_orig, f, thresh);
	timer2 = omp_get_wtime() - timer1;
//	printf("Time of Test_TransferBlock3Diag_to_CSR: %lf\n", timer2);

	// Solve Pardiso
	printf("-------------- Solving 3D system with Pardiso... -------------\n");
	timer1 = omp_get_wtime();
	SolvePardiso3D(x, y, z, Dcsr, x_pard, f, thresh);
	timer2 = omp_get_wtime() - timer1;
	printf("Time of  SolvePardiso3D: %lf\n", timer2);
	printf("Computing error for 3D PARDISO ||x_{exact}-x_{comp}||/||x_{exact}||\n");

	reducePML3D(x, y, z, size, x_pard, size_nopml, x_pard_nopml);

	all_time = omp_get_wtime() - all_time;
	printf("Elapsed time: %lf\n", all_time);

	//norm = rel_error(zlange, size, 1, x_pard_cpy, x_orig, size, thresh);

	pml_flag = true;

#ifdef OUTPUT
	//output("ChartsN100PML/model", pml_flag, x, y, z, x_orig_nopml, x_pard_nopml);
#endif


#ifdef GNUPLOT
//	gnuplot("ChartsN100PML/model", "ChartsN100PML/helm_ex", pml_flag, 4, x, y, z);
//	gnuplot("ChartsN100PML/model", "ChartsN100PML/helm_pard", pml_flag, 6, x, y, z);
#endif
	
	//return 0;
	//system("pause");

	
#ifdef OUTPUT
	FILE* fout1 = fopen("solutions.dat", "w");

	for (int i = 0; i < size_nopml; i++)
		fprintf(fout1, "%d %12.10lf %12.10lf %12.10lf %12.10lf\n", i, x_orig_nopml[i].real(), x_orig_nopml[i].imag(), x_pard_nopml[i].real(), x_pard_nopml[i].imag());

	fclose(fout1);
#endif

	//for (int i = 0; i < size; i++)
	//fprintf(fout1, "%d %12.10lf %12.10lf %12.10lf %12.10lf\n", i, x_orig[i].real(), x_orig[i].imag(), x_pard[i].real(), x_pard[i].imag());

	zlacpy("All", &size_nopml, &ione, x_pard_nopml, &size_nopml, x_pard_nopml_cpy, &size_nopml);

	norm = rel_error(zlange, size_nopml, 1, x_pard_nopml_cpy, x_orig_nopml, size_nopml, thresh);

	if (norm < thresh) printf("Norm %12.10e < eps %12.10lf: PASSED\n", norm, thresh);
	else printf("Norm %12.10lf > eps %12.10lf : FAILED\n", norm, thresh);


	//free_arr(x_pard);
	//free_arr(x_pard_nopml);
	//free_arr(B);
	//free_arr(f);
	//free_arr(x_orig);
	//free_arr(x_orig_nopml);
	


	system("pause");
#endif

#endif

	Solve3DSparseUsingFT(x, y, z, f, f_FFT, x_sol_prd, x_pard_nopml, x_pard_nopml_cpy, x_sol_fft_nopml, x_sol, x_orig, x_orig_nopml, thresh);

	dtype *g_nopml = alloc_arr<dtype>(size_nopml);
	dtype *g = alloc_arr<dtype>(size);
	dtype *x1_nopml = alloc_arr<dtype>(size_nopml);
	dtype *f_nopml = alloc_arr<dtype>(size_nopml);
	RelRes = 0;

	printf("size = %d size_no_pml = %d\n", size, size_nopml);

	reducePML3D(x, y, z, size, f, size_nopml, f_nopml);

	ResidCSR(x_nopml, y_nopml, z_nopml, Dcsr_nopml, x_sol, f_nopml, g_nopml, RelRes);

	for (int i = 0; i < size_nopml; i++)
		printf("g_nopml[%d] = %lf %lf\n", i, g_nopml[i].real(), g_nopml[i].imag());

	printf("RelRes = %lf\n", RelRes);

	system("pause");
	ItRef = 100;

	if (RelRes < thresh)
	{
		success = 1;
		itcount = 0;
	}
	else {
		
		while (RelRes > thresh && itcount < ItRef)
		{
			extendPML3D(x, y, z, size_nopml, g_nopml, size, g);
			Solve3DSparseUsingFT(x, y, z, g, f_FFT, x_sol_prd, x_pard_nopml, x_pard_nopml_cpy, x_sol_fft_nopml, x1_nopml, x_orig, x_orig_nopml, thresh);

#pragma omp parallel for simd schedule(simd:static)
			for (int i = 0; i < size_nopml; i++)
				x_sol[i] += x1_nopml[i];

			ResidCSR(x_nopml, y_nopml, z_nopml, Dcsr_nopml, x_sol, f_nopml, g_nopml, RelRes);

			printf("---------------iteration: %d, RelRes = %lf---------------\n", itcount, RelRes);
			itcount++;
			system("pause");
		}

		if (RelRes < thresh && ItRef < itcount) success = 1;
	}


	if (success == 0) printf("No convergence\nResidual norm: %lf\n", RelRes);
	
	system("pause");
	// Output
	output("Charts100/model_ft", pml_flag, x, y, z, x_orig_nopml, x_sol);

	check_norm_result(x.n_nopml, y.n_nopml, z.n_nopml, x_orig_nopml, x_sol);

	printf("Computing error ||x_{exact}-x_{comp_fft}||/||x_{exact}||\n");
	norm = rel_error(zlange, size_nopml, 1, x_sol, x_orig_nopml, size_nopml, thresh);


	if (norm < thresh) printf("Norm %12.10e < eps %12.10lf: PASSED\n", norm, thresh);
	else printf("Norm %12.10lf > eps %12.10lf : FAILED\n", norm, thresh);

	printf("Computing error ||x_{comp_prd}-x_{comp_fft}||/||x_{comp_prd}||\n");
	norm = rel_error(zlange, size_nopml, 1, x_pard_nopml_cpy, x_pard_nopml, size_nopml, thresh);

	if (norm < thresh) printf("Norm %12.10e < eps %12.10lf: PASSED\n", norm, thresh);
	else printf("Norm %12.10lf > eps %12.10lf : FAILED\n", norm, thresh);

	printf("----------------------------------------------\n");

#ifdef GNUPLOT
	printf("Printing results...\n");
	gnuplot("Charts100/model_ft", "Charts100/real/ex_pard", pml_flag, 4, x, y, z);
	gnuplot("Charts100/model_ft", "Charts100/imag/ex_pard", pml_flag, 5, x, y, z);
	gnuplot("Charts100/model_ft", "Charts100/real/helm_ft", pml_flag, 6, x, y, z);
	gnuplot("Charts100/model_ft", "Charts100/imag/helm_ft", pml_flag, 7, x, y, z);
#else
	printf("No printing results...\n");
#endif

#ifndef ONLINE
	free_arr(D);
	free_arr(B);
#endif
	free_arr(x_orig);
	free_arr(x_sol);
	free_arr(f);

	system("pause");

	return 0;
#endif
}
#endif
#endif
