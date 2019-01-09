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
#define PERF

#ifndef PERF
	TestAll();
#endif
//	system("pause");
//	return 0;
#if 1
						
#ifdef PML			   // 50 pts   - 7 % and 8 % if beta = 0.3 (ppw = 26)
					  //			- 5 % and 6 % if beta = 0.25
	int pml_pts = 20; // 100 pts  - 10 % and 9 % if beta = 0.1
					   //		      6 % and 7 % if beta = 0.2
					   // 150 pts  - 20 % and 10 % if beta = 0.05;
					   //          - 6 % and 3 % if beta = 0.1
					   // 200 pts  - 4 % and 4 % if beta = 0.1, 6 % and ? if beta = 0.2
	int spg_pts = 50; // 250 pts  - 3 % and 3 % if beta = 0.1

	// 3D
	// 100 pt - 19 % if beta = 0.05
	//			15 % if beta = 0.005
	// 200 pt - 33 % if beta = 0.1
	//		  - 20 % if beta = 0.05
	//		  - 12 % if beta = 0.01
	//		  - 11 % if beta = 0.005
	// 250 pt - 20 % if beta = 0.05
	//        - 10 % if beta = 0.005 - the best (range - 0.08 - 0.01)
	//		  - 11 % if beta = 0.001
	// 1000 pt - 9% if beta = beta = 0.005

	// betas.doc - 28-30 % for ppw = 10 with spg = 250 pts
	//			 - 11% for ppw = 20 with spg = 500 pts
	//		     - 10% with stable z ppw = 20 with spg > 1000
#else
	int pml_pts = 0;
#endif
	int pml_size = 2 * pml_pts;

	size_m x, y, z;
	size_m x_nopml, y_nopml, z_nopml;

	x.pml_pts = y.pml_pts = pml_pts;
	z.pml_pts = 0;

	z.spg_pts = spg_pts;

	x_nopml.pml_pts = y_nopml.pml_pts = z_nopml.pml_pts = 0;

	int n1 = 99 + 2 * x.pml_pts;		    // number of point across the directions
	int n2 = 99 + 2 * y.pml_pts;
	int n3 = 99 + 2 * z.spg_pts;
	int n = n1 * n2;		// size of blocks
	int NB = n3;			// number of blocks

	x.n = n1;
	y.n = n2;
	z.n = n3;

	int size = n * NB;		// size of vector x and f: n1 * n2 * n3
	int size2D = n;
	int smallsize = 1600;
	double thresh = 1e-6;	// stop level of algorithm by relative error
	int ItRef = 200;		// Maximal number of iterations in refirement
	char bench[255] = "display"; // parameter into solver to show internal results
	int sparse_size = n + 2 * (n - 1) + 2 * (n - n1);
	int non_zeros_in_3diag = n + (n - 1) * 2 + (n - n1) * 2 - (n1 - 1) * 2;
	int ione = 1;
	int success = 0;
	int itcount = 0;
	double RelRes = 0;
	double norm = 0;
	bool pml_flag = 1;
	int i1, j1, k1;
	double norm_re, norm_im;
	

	double timer1, timer2, all_time;

	x.n_nopml = n1 - 2 * x.pml_pts;
	y.n_nopml = n2 - 2 * y.pml_pts;
	z.n_nopml = n3 - 2 * z.spg_pts;

	x_nopml.n = y_nopml.n = x.n_nopml;
	z_nopml.n = z.n_nopml;

	x_nopml.n_nopml = y_nopml.n_nopml = x_nopml.n;
	z_nopml.n_nopml = z_nopml.n;

	x.l = LENGTH_X + (double)(2 * x.pml_pts * LENGTH_X) / (x.n_nopml + 1);
	y.l = LENGTH_Y + (double)(2 * y.pml_pts * LENGTH_Y) / (y.n_nopml + 1);
	z.l = LENGTH_Z + (double)(2 * z.spg_pts * LENGTH_Z) / (z.n_nopml + 1);

	x.h = x.l / (x.n + 1);  // x.n + 1 grid points of the whole domain
	y.h = y.l / (y.n + 1);  // x.n - 1 - inner points
	z.h = z.l / (z.n + 1);  // 2 points - for the boundaries

	printf("Size of domain: Lx = %lf, Ly = %lf, Lz = %lf\n", x.l, y.l, z.l);
	printf("with points: Nx = %d, Ny = %d, Nz = %d\n", x.n, y.n, z.n);
	printf("------------------------------\n");
	printf("Size of physical domain: Lx = %lf, Ly = %lf, Lz = %lf\n", x.n_nopml * x.h, y.n_nopml * y.h, z.n_nopml * z.h);
	printf("with points: Nx = %d, Ny = %d, Nz = %d\n", x.n_nopml, y.n_nopml, z.n_nopml);
	printf("Size of PML domain: Lx = %lf, Ly = %lf, Lz = %lf\n", 2 * x.pml_pts * x.h, 2 * y.pml_pts * y.h, 2 * z.pml_pts * z.h);
	printf("with points: Nx = %d, Ny = %d, Nz = %d\n", 2 * x.pml_pts, 2 * y.pml_pts, 2 * z.pml_pts);
	printf("Size of SPNONGE domain: Lz = %lf\n", 2 * z.spg_pts * z.h);
	printf("with points: Nz = %d\n", 2 * z.spg_pts);
	printf("Steps for physical domain: hx = %lf, hy = %lf, hz = %lf\n", x.h, y.h, z.h);

	printf("Size of system Au = f : %d x %d \n", size, size);

	// ��� ���������� ���� � 2 ������ ������ ����������� � 4 ����!
	// 3D ������
	// ���������� - � ����������x
	// h = 10, 1280 x 1280, N = 120 - 2 �����
	// 40 ����� h = 30, L = 600, omega = 4, 6, 10

#ifndef PERF
	TestHankel();
	system("pause");
#endif

	double lambda = (double)(c_z) / nu;
	double ppw = lambda / x.h;
	int niter = 33;

	printf("The length of the wave: %lf\n", lambda);
	printf("ppw: %lf\n", ppw);
	printf("FGMRES number of iterations: %d\n", niter + 1);

	int n_nopml = x.n_nopml * y.n_nopml;
	int size_nopml = n_nopml * z.n_nopml;
	int size2D_nopml = n_nopml;

	printf("-----Memory required:-----\n");
	double total = 0;
	total = double(size) / (1024 * 1024 * 1024);
	total *= 4 + 2; // 2 for 2D problems - FFT + PARDISO
	total += double(size2D) / (1024 * 1024 * 1024);
	total += double(4 * size_nopml) / (1024 * 1024 * 1024);
	total *= 8;
	total *= 2;

	printf("Initial = %lf GB\n", total);

	total = double(size) / (1024 * 1024 * 1024);
	total *= (niter + 1) + 4;
	total *= 8;
	total *= 2;

	printf("FGMRES = %lf GB\n", total);


//#define TEST1D

#ifdef TEST1D
	dtype *f1D = alloc_arr<dtype>(x.n);
	dtype *x_sol1D = alloc_arr<dtype>(x.n);
	Solve1DSparseHelmholtz(x, y, z, f1D, x_sol1D, thresh);
#endif

//#define TEST2D

#ifdef TEST2D
	dtype *f2D = alloc_arr<dtype>(x.n * y.n);
	dtype *x_sol2D = alloc_arr<dtype>(x.n * y.n);
	Solve2DSparseHelmholtz(x, y, z, f2D, x_sol2D, thresh);
#endif

	system("pause");

	// Solution and right hand side
	dtype *x_orig = alloc_arr<dtype>(size);
	dtype *x_sol = alloc_arr<dtype>(size);
	dtype *f = alloc_arr<dtype>(size);
	dtype *g = alloc_arr<dtype>(size);

	dtype *x_orig_nopml = alloc_arr<dtype>(size_nopml);
	dtype *x_sol_nopml = alloc_arr<dtype>(size_nopml);
	dtype *f_nopml = alloc_arr<dtype>(size_nopml);
	dtype *g_nopml = alloc_arr<dtype>(size_nopml);
	

	x_nopml.l = y_nopml.l = z_nopml.l = (double)(LENGTH);
	x_nopml.h = x_nopml.l / (double)(x_nopml.n + 1);  // x.n + 1 grid points of the whole domain
	y_nopml.h = y_nopml.l / (double)(y_nopml.n + 1);  // x.n - 1 - inner points
	z_nopml.h = z_nopml.l / (double)(z_nopml.n + 1);  // 2 points - for the boundaries

#ifndef PERF
	system("pause");
#endif

#ifdef GEN_3D_MATRIX

	int non_zeros_in_3Dblock3diag = (n + (n - 1) * 2 + (n - x.n) * 2 - (y.n - 1) * 2) * z.n + 2 * (size - n);

	dtype *D, *D_nopml;
	dtype *B_mat, *B_mat_nopml;
	dtype *B, *B_nopml;

//	B = alloc_arr<dtype>(size - n); // vector of diagonal elementes
//	B_nopml = alloc_arr<dtype>(size_nopml - n_nopml); // vector of diagonal elementes

//	dtype *x_pard = alloc_arr<dtype>(size);

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

//	D = alloc_arr<dtype>(n * n); // it's a matrix with size n^3 * n^2 = size * n
//	B_mat = alloc_arr<dtype>(n * n);

//	D_nopml = alloc_arr<dtype>(n_nopml * n_nopml);
//	B_mat_nopml = alloc_arr<dtype>(n_nopml * n_nopml);

#endif

#ifdef _OPENMP
	int nthr = omp_get_max_threads();
	printf("Max_threads: %d threads\n", nthr);
	omp_set_dynamic(0);
	nthr = 2;
	omp_set_num_threads(nthr);
	mkl_set_num_threads(2);
	printf("Run in parallel on %d threads\n", nthr);
#else
	printf("Run sequential version on 1 thread\n");
#endif

	printf("Grid steps: hx = %lf hy = %lf hz = %lf\n", x.h, y.h, z.h);

#ifndef STRUCT_CSR
	// Generation matrix of coefficients, vector of solution (to compare with obtained) and vector of RHS
	GenMatrixandRHSandSolution(n1, n2, n3, D, ldd, B, x_orig, f);
#else

	point source = { x.l / 2.0, y.l / 2.0, z.l / 2.0 };

	// Generation of vector of solution (to compare with obtained) and vector of RHS
	printf("Gen right-hand side and solution...\n");
	//GenRHSandSolution(x, y, z, x_orig, f, source);

	GenRHSandSolutionViaSound3D(x, y, z, x_orig, f, source);

	// Generation of sparse coefficient matrix
#ifdef GEN_3D_MATRIX
	//printf("--------------- Gen sparse 3D matrix in CSR format... ---------------\n");
	//GenSparseMatrixOnline3D(x, y, z, B, B_mat, n, D, n, B_mat, n, Dcsr);


#if 1
	ccsr *Dcsr_nopml;
	int non_zeros_nopml = (n_nopml + (n_nopml - 1) * 2 + (n_nopml - x.n_nopml) * 2 - (y.n_nopml - 1) * 2) * z.n_nopml + 2 * (size_nopml - n_nopml);
	Dcsr_nopml = (ccsr*)malloc(sizeof(ccsr));
	Dcsr_nopml->values = alloc_arr<dtype>(non_zeros_nopml);
	Dcsr_nopml->ia = alloc_arr<int>(size_nopml + 1);
	Dcsr_nopml->ja = alloc_arr<int>(non_zeros_nopml);
	Dcsr_nopml->ia[size_nopml] = non_zeros_nopml + 1;
	Dcsr_nopml->non_zeros = non_zeros_nopml;

	printf("-------------- Gen sparse 3D matrix in CSR format with PML... ------------\n");
	timer1 = omp_get_wtime();
//	GenSparseMatrixOnline3DwithPML(x, y, z, B, B_mat, n, D, n, B_mat, n, Dcsr, thresh);
	timer2 = omp_get_wtime() - timer1;

	printf("Time of GenSparseMatrixOnline3DwithPML: %lf\n", timer2);

	printf("-------------- Gen sparse 3D matrix in CSR format with no PML... ------------\n");
	timer1 = omp_get_wtime();
//	GenSparseMatrixOnline3DwithPML(x_nopml, y_nopml, z_nopml, B_nopml, B_mat_nopml, n_nopml, D_nopml, n_nopml, B_mat_nopml, n_nopml, Dcsr_nopml, thresh);
	timer2 = omp_get_wtime() - timer1;

	printf("Time of GenSparseMatrixOnline3DnoPML: %lf\n", timer2);
#endif

	free_arr(D);
	free_arr(B_mat);


	printf("Analytic non_zeros in first row and last two 2D blocks: %d\n", non_zeros_in_3diag + n);
	printf("Analytic non_zeros in three 2D block-row: %d\n", non_zeros_in_3diag + 2 * n);
	printf("Analytic non_zeros in 3D block-tridiagonal matrix: %d\n", non_zeros_in_3Dblock3diag);

#if 0
	// Test PML
	ccsr *Dcsr_reduced;
	Dcsr_reduced = (ccsr*)malloc(sizeof(ccsr));
	Dcsr_reduced->values = alloc_arr<dtype>(non_zeros_nopml);
	Dcsr_reduced->ia = alloc_arr<int>(size_nopml + 1);
	Dcsr_reduced->ja = alloc_arr<int>(non_zeros_nopml);
	Dcsr_reduced->ia[size_nopml] = non_zeros_nopml + 1;
	Dcsr_reduced->non_zeros = non_zeros_nopml;

	printf("----------------- Running test PML... -----------\n");
	timer1 = omp_get_wtime();
	Test_PMLBlock3Diag_in_CSR(x, y, z, Dcsr, Dcsr_nopml, Dcsr_reduced, thresh);
	timer2 = omp_get_wtime() - timer1;
	printf("Time of Test_PMLBlock3Diag_in_CSR: %lf\n", timer2);

	system("pause");
#endif
#endif

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

//#define TEST3D

#ifdef TEST3D
	TestFGMRES();
#endif

	printf("---Residual exact solution---\n");
//	ComputeResidual(x, y, z, (double)kk, x_orig, f, g, RelRes);
	printf("-----------\n");
	printf("Residual in 3D with PML |A * x_sol - f| = %e\n", RelRes);
	printf("-----------\n");

//	reducePML3D(x, y, z, size, g, size_nopml, g_nopml);

//	RelRes = dznrm2(&size_nopml, g_nopml, &ione);
	printf("-----------\n");
	printf("Residual in 3D psys dom  |A * x_sol - f| = %e\n", RelRes);
	printf("-----------\n");

#ifndef PERF
	system("pause");
#endif

#ifdef OUTPUT
	FILE* out = fopen("ResidualVectorOrig.dat", "w");
	for (int i = 0; i < size; i++)
	{
		take_coord3D(x.n, y.n, z.n, i, i1, j1, k1);
	//	fprintf(out, "%d %d %d %lf %lf\n", i1, j1, k1, f_rsd[i].real(), f_rsd[i].imag());
	}
	fclose(out);
#endif

	printf("check right-hand-side f\n");
	for (int i = 0; i < size; i++)
		if (abs(f[i]) != 0) printf("f_FFT[%d] = %lf %lf\n", i, f[i].real(), f[i].imag());

	// ------------ FGMRES-------------
	all_time = omp_get_wtime();

	FGMRES(x, y, z, niter, source, x_sol, x_orig, f, thresh);

	all_time = omp_get_wtime() - all_time;
	printf("Time: %lf\n", all_time);

	RelRes = 0;

	printf("size = %d size_no_pml = %d\n", size, size_nopml);

#ifndef TEST_HELM_1D
	for (int k = 0; k < z.n; k++)
	{
		int src = size2D / 2;
		NullifySource2D(x, y, &x_sol[k * size2D], src, 5);
		NullifySource2D(x, y, &x_orig[k * size2D], src, 5);
	}
#endif

	reducePML3D(x, y, z, size, x_orig, size_nopml, x_orig_nopml);
	reducePML3D(x, y, z, size, x_sol, size_nopml, x_sol_nopml);
	reducePML3D(x, y, z, size, f, size_nopml, f_nopml);

	RelRes = dznrm2(&size_nopml, x_sol_nopml, &ione);
	printf("norm x_sol = %lf\n", RelRes);

//	for (int k = 0; k < z.n_nopml; k++)
	//	x_orig_nopml[k * size2D_nopml + size2D_nopml / 2] = x_sol_nopml[k * size2D_nopml + size2D_nopml / 2] = 0;

//	ResidCSR(x_nopml, y_nopml, z_nopml, Dcsr_nopml, x_sol_nopml, f_nopml, g_nopml, RelRes);
//	printf("-----------\n");
//	printf("Residual in 3D  ||A * x_sol - f|| = %lf\n", RelRes);
//	printf("-----------\n");

#ifndef PERF
	system("pause");
#endif
	// Output

#ifndef PERF
#define OUTPUT
#define GNUPLOT
#endif

#define OUTPUT
#ifdef OUTPUT
	output("ChartsFreq4/model_ft", pml_flag, x, y, z, x_orig_nopml, x_sol_nopml);
#endif

	printf("----------------------------------------------\n");

	free_arr(f);
	free_arr(g);
	free_arr(f_nopml);
	free_arr(g_nopml);

	double *x_orig_re = alloc_arr<double>(size_nopml);
	double *x_sol_re = alloc_arr<double>(size_nopml);
	double *x_orig_im = alloc_arr<double>(size_nopml);
	double *x_sol_im = alloc_arr<double>(size_nopml);

	printf("Computing error ||x_{exact}-x_{comp_fft}||/||x_{exact}||\n");
#ifndef TEST_HELM_1D
	check_norm_result2(x.n_nopml, y.n_nopml, z.n_nopml, niter, ppw, 2 * z.spg_pts * z.h, x_orig_nopml, x_sol_nopml, x_orig_re, x_orig_im, x_sol_re, x_sol_im);
	check_norm_circle(x_nopml, y_nopml, z_nopml, x_orig_nopml, x_sol_nopml, source, thresh);

	norm_re = rel_error(dlange, size_nopml, 1, x_sol_re, x_orig_re, size_nopml, thresh);
	norm_im = rel_error(dlange, size_nopml, 1, x_sol_im, x_orig_im, size_nopml, thresh);
	norm = rel_error(zlange, size_nopml, 1, x_sol_nopml, x_orig_nopml, size_nopml, thresh);

	printf("norm_re = %lf\n", norm_re, thresh);
	printf("norm_im = %lf\n", norm_im, thresh);
	printf("norm_full = %lf\n", norm, thresh);
#else
	check_test_3Dsolution_in1D(x.n_nopml, y.n_nopml, z.n_nopml, x_sol_nopml, x_orig_nopml, thresh);
#endif


//	printf("Computing error ||x_{comp_prd}-x_{comp_fft}||/||x_{comp_prd}||\n");
//	norm = rel_error(zlange, size_nopml, 1, x_pard_nopml_cpy, x_pard_nopml, size_nopml, thresh);

//	if (norm < thresh) printf("Norm %12.10e < eps %12.10lf: PASSED\n", norm, thresh);
//	else printf("Norm %12.10lf > eps %12.10lf : FAILED\n", norm, thresh);

	printf("----------------------------------------------\n");

#define GNUPLOT

#ifdef GNUPLOT
	pml_flag = true;
	gnuplot("ChartsFreq4/model_ft", "ChartsFreq4/real/ex_pard", pml_flag, 4, x, y, z);
	gnuplot("ChartsFreq4/model_ft", "ChartsFreq4/imag/ex_pard", pml_flag, 5, x, y, z);
	gnuplot("ChartsFreq4/model_ft", "ChartsFreq4/real/helm_ft", pml_flag, 6, x, y, z);
	gnuplot("ChartsFreq4/model_ft", "ChartsFreq4/imag/helm_ft", pml_flag, 7, x, y, z);
#else
	printf("No printing results...\n");
#endif

#ifndef ONLINE
	free_arr(D);
	free_arr(B);
#endif
	free_arr(x_orig);
	free_arr(x_sol);

	system("pause");

	return 0;
#endif
}
#endif
#endif
