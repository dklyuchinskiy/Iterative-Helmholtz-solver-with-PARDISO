#include "definitionsHODLR.h"
#include "templatesHODLR.h"
#include "TestSuiteHODLR.h"
#include "../TestSuite.h"
#include "TestFramework.h"

/***************************************
Source file contains tests for testing
functionalites, implemented in
functions.cpp and BinaryTrees.cpp.

The interface is declared in TestSuite.h
****************************************/

void TestAll()
{
	TestRunner runner;
	//void(*pt_func)(int&) = NULL;
	//pt_func = &Shell_SymRecCompress;

	printf("***** TEST LIBRARY FUNCTIONS *******\n");
	printf("****Complex precision****\n");
#if 1
	runner.RunTest(Shell_LowRankApprox, Test_LowRankApproxStruct, "Test_LowRankApprox");
	runner.RunTest(Shell_SymRecCompress, Test_SymRecCompressStruct, "Test_SymRecCompress");
	runner.RunTest(Shell_SymRecCompress, Test_UnsymmRecCompressStruct, "Test_UnsymmRecCompress");
	runner.RunTest(Shell_DiagMult, Test_DiagMultStruct, "Test_DiagMult");
	runner.RunTest(Shell_RecMultL, Test_RecMultLStruct, "Test_RecMultL");
	runner.RunTest(Shell_RecMultL, Test_UnsymmRecMultLStruct, "Test_UnsymmRecMultL");
	runner.RunTest(Shell_RecMultL, Test_UnsymmRecMultRStruct, "Test_UnsymmRecMultR");
	runner.RunTest(Shell_Add, Test_AddStruct, "Test_Add");
	runner.RunTest(Shell_Add, Test_UnsymmAddStruct, "Test_UnsymmAdd");
	runner.RunTest(Shell_SymCompUpdate2, Test_SymCompUpdate2Struct, "Test_SymCompUpdate2");
	runner.RunTest(Shell_SymCompUpdate2, Test_UnsymmCompUpdate2Struct, "Test_UnsymmCompUpdate2");
	runner.RunTest(Shell_UnsymmCompUpdate3, Test_UnsymmCompUpdate3Struct, "Test_UnsymmCompUpdate3");
	runner.RunTest(Shell_SymCompRecInv, Test_SymCompRecInvStruct, "Test_SymCompRecInv");
	runner.RunTest(Shell_SymCompRecInv, Test_UnsymmCompRecInvStruct, "Test_UnsymmCompRecInv");
	runner.RunTest(Shell_CopyStruct, Test_CopyStruct,  "Test_CopyStruct");
	runner.RunTest(Shell_CopyStruct, Test_CopyUnsymmStruct, "Test_CopyUnsymmStruct");
	runner.RunTest(Shell_CopyStruct, Test_CopyLfactorStruct, "Test_CopyLfactor");
	runner.RunTest(Shell_CopyStruct, Test_CopyRfactorStruct, "Test_CopyRfactor");
	runner.RunTest(Shell_CopyStruct, Test_UnsymmCopyStruct, "Test_UnsymmCopyStruct");
	runner.RunTest(Shell_LowRankApprox, Test_PermutLowRankApprox, "Test_PermutLowRankApprox");
	runner.RunTest(Shell_UnsymmLUfact, Test_MyLURecFact, "Test_MyLURecFact");
	runner.RunTest(Shell_ApplyToA21, Test_ApplyToA12, "Test_ApplyToA12");
	runner.RunTest(Shell_ApplyToA21, Test_ApplyToA21, "Test_ApplyToA21");
	runner.RunTest(Shell_ApplyToA21, Test_ApplyToA21Ver2, "Test_ApplyToA21Ver2");
	runner.RunTest(Shell_ApplyToA21, Test_ApplyToA12Ver2, "Test_ApplyToA12Ver2");
	runner.RunTest(Shell_LowRankToUnsymmHSS, Test_LowRankToUnsymmHSS, "Test_LowRankToUnsymmHSS");
	runner.RunTest(Shell_SymCompUpdate4LowRankStruct, Test_SymCompUpdate4LowRankStruct, "Test_SymCompUpdate4LowRankStruct");
	runner.RunTest(Shell_UnsymmLUfact, Test_UnsymmLUfact, "Test_LUfact");
	runner.RunTest(Shell_UnsymmLUfact, Test_SymLUfactLowRankStruct, "Test_SymLUfactLowRankStruct");
	runner.RunTest(Shell_ApplyToA21, Test_SolveTriangSystemA21, "Test_SolveTriangSystemA21");
	runner.RunTest(Shell_UnsymmLUfact, Test_SymLUfactLowRankStruct, "Test_SymLUfactLowRankStruct");
	runner.RunTest(Shell_SymCompUpdate5LowRankStruct, Test_SymCompUpdate5LowRankStruct, "Test_SymCompUpdate5LowRankStruct");
	runner.RunTest(Shell_LowRankCholeskyStruct, Test_LowRankCholeskyStruct, "Test_LowRankCholeskyStruct");
	printf("*******FFT*******\n");
	//	runner.RunTest(Shell_FFT1D_Real, Test_FFT1D_Real, "Test_FFT1D");
	//	runner.RunTest(Shell_FFT1D_Complex, Test_FFT1D_Complex, "Test_FFT1D_Complex");
	//	runner.RunTest(Shell_FFT1D_Complex, Test_Poisson_FT1D_Real, "Test_Poisson_FT1D_Real");
	//  runner.RunTest(Shell_FFT1D_Complex, Test_Poisson_FT1D_Complex, "Test_Poisson_FT1D_Complex");
#endif

	printf("********************\n");
	printf("ALL TESTS: %d\nPASSED: %d \nFAILED: %d\n", runner.GetAll(), runner.GetPassed(), runner.GetFailed());

	printf("***** THE END OF TESTING*******\n\n");

}

void Shell_LowRankApprox(ptr_test_low_rank func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";

	for (double eps = 1e-2; eps > 1e-9; eps /= 10)
		for (int m = 3; m <= 40; m++)
			for (int n = 1; n <= 40; n++)
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

	for (int n = 3; n <= 15; n++)
		for (double eps = 1e-2; eps > 1e-9; eps /= 10)
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

	for (double eps = 1e-2; eps > 1e-9; eps /= 10)
		for (int n = 3; n <= 15; n++)	
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

	for (double eps = 1e-2; eps > 1e-9; eps /= 10)
		for (int n = 3; n <= 15; n++)
			for (int k = 1; k <= 12; k++)
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

void Shell_ApplyToA21(ptr_test_mult_diag func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";

#if 0
	for (int smallsize = 3; smallsize < 6; smallsize++)
		for (double eps = 1e-4; eps > 1e-8; eps /= 10)
			for (int m = smallsize; m <= 40; m++)
				for (int n = smallsize; n <= 40; n++)
#endif

#define neps 4
	double eps[neps] = { 1e-4, 1e-5, 1e-7, 1e-9 };

	for (int smallsize = 3; smallsize < 6; smallsize++)
		for (int i = 0; i < neps; i++)
			for (int m = smallsize; m <= 40; m++)
				for (int n = smallsize; n <= 30; n++)
				{
					try
					{
						numb++;
						func(m, n, eps[i], method, smallsize);
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
		for (int n = 3; n <= 12; n++)
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
			for (int n = 3; n <= 15; n++)
				for (int k = 2; k <= 12; k++)
				{
					try
					{
						numb++;
						func(n, k, { alpha, 0 }, eps, method, smallsize);
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

void Shell_UnsymmCompUpdate3(ptr_test_update2 func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";
	int smallsize = 3;

	for (double eps = 1e-3; eps > 1e-9; eps /= 10)
		for (double alpha = -10; alpha < 10; alpha += 2)
			for (int n = 3; n <= 30; n += 2)
				for (int k1 = 1; k1 <= 12; k1 += 2)
					for (int k2 = 1; k2 <= 12; k2 += 2)
				{
					try
					{
						numb++;
						func(n, k1, k2, { alpha, 0 }, eps, method, smallsize);
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

void Shell_SymCompUpdate4LowRankStruct(ptr_test_sym_rec_compress_low_rank func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";
	int smallsize = 3;

	double alpha = 1.0;

	for (double eps = 1e-3; eps > 1e-9; eps /= 10)
		for(double alpha = -10; alpha <= 10; alpha += 2)
			for (int n = 3; n <= 50; n += 2)
					{
						try
						{
							numb++;
							func(n, dtype{ alpha, 0 }, eps, method, smallsize);
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

void Shell_SymCompUpdate5LowRankStruct(ptr_test_update func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";
	double alpha = 1.0;

	for (double eps = 1e-4; eps > 1e-8; eps /= 10)
		for (double alpha = -6; alpha <= 6; alpha += 2)
			for (int n = 3; n <= 40; n++)
				for (int p = 1; p <= n; p += 2)
					for (int smallsize = 3; smallsize <= 6; smallsize++)
					{
						try
						{
							numb++;
							func(n, p, dtype{ alpha, 0 }, eps, method, smallsize);
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

	for (double eps = 1e-2; eps > 1e-9; eps /= 10)
		for (int n = 3; n <= 20; n += 2)
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

void Shell_UnsymmLUfact(ptr_test_sym_rec_compress func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";

	for(int smallsize = 3; smallsize < 6; smallsize++)
		for (double eps = 1e-3; eps > 1e-9; eps /= 10)
			for (int n = smallsize; n <= 40; n++)
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

void Shell_LowRankToUnsymmHSS(ptr_test_sym_rec_compress func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";

	for (int smallsize = 3; smallsize < 6; smallsize++)
		for (double eps = 1e-4; eps > 1e-9; eps /= 10)
			for (int n = smallsize; n <= 50; n++)
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

void Shell_LowRankCholeskyStruct(ptr_test_mult_diag func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";

	for (double eps = 1e-3; eps > 1e-9; eps /= 10)
			for (int n = 3; n <= 50; n++)
				for (int k = 1; k <= n; k++)
					for (int smallsize = 3; smallsize <= 6; smallsize++)
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

void Shell_CopyStruct(ptr_test_sym_rec_compress func, const string& test_name, int &numb, int &fail_count)
{
	char method[255] = "SVD";
	int smallsize = 3;
//	double eps = 1e-2;

	for (double eps = 1e-2; eps > 1e-9; eps /= 10)
		for (int n = 3; n <= 15; n++)
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

void Test_LowRankApproxStruct(int m, int n, double eps, char *method)
{
	// A - matrix in dense order
	dtype *A = alloc_arr2<dtype>(m * n);
	dtype *A_init = alloc_arr2<dtype>(m * n);
	dtype *A_rec = alloc_arr2<dtype>(m * n);
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

	norm = rel_error_complex(m, n, A_rec, A_init, lda, eps);
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

	dtype *H = alloc_arr2<dtype>(n * n); // init
	dtype *H1 = alloc_arr2<dtype>(n * n); // compressed
	dtype *H2 = alloc_arr<dtype>(n * n); // recovered init

	int ldh = n;

	Hilbert(n, n, H, ldh);
	Hilbert(n, n, H1, ldh);

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
	norm = rel_error_complex(n, n, H2, H, ldh, eps);

#ifdef DEBUG
	print(n, n, H, ldh, "H init");
	print(n, n, H2, ldh, "diff");
#endif

	char str[255];
	sprintf(str, "Struct: n = %d ", n);
	AssertLess(norm, eps, str);

	FreeNodes(n, H1str, smallsize);
	free_arr(H);
	free_arr(H2);
	free_arr(H1);
}

void Test_UnsymmRecCompressStruct(int n, double eps, char *method, int smallsize)
{
	//printf("*****Test for SymRecCompressStruct  n = %d eps = %e ******* ", n, eps);
	char frob = 'F';
	double norm = 0;

	dtype *H = alloc_arr2<dtype>(n * n); // init
	dtype *H1 = alloc_arr2<dtype>(n * n); // compressed
	dtype *H2 = alloc_arr<dtype>(n * n); // recovered init

	int ldh = n;

	Hilbert4(n, n, H, ldh);
	Hilbert4(n, n, H1, ldh);

#ifdef DEBUG
	print(n, n, H1, ldh, "H1");
#endif

	cumnode *H1str; // pointer to the tree head
	UnsymmRecCompressStruct(n, H1, ldh, H1str, smallsize, eps, "SVD"); // recursive function means recursive allocation of memory for structure fields
	UnsymmResRestoreStruct(n, H1str, H2, ldh, smallsize);

#ifdef DEBUG
	//print(n, n, H1, ldh, "H1 compressed");
	print(n, n, H2, ldh, "H recovered");
#endif

	// Norm of residual || A - L * U ||
	norm = rel_error_complex(n, n, H2, H, ldh, eps);

#ifdef DEBUG
	print(n, n, H, ldh, "H init");
	print(n, n, H2, ldh, "diff");
#endif

	char str[255];
	sprintf(str, "Struct: n = %d ", n);
	AssertLess(norm, eps, str);

	FreeUnsymmNodes(n, H1str, smallsize);
	free_arr(H);
	free_arr(H2);
	free_arr(H1);
}

void Test_DiagMultStruct(int n, double eps, char *method, int smallsize)
{
	//printf("*****Test for DiagMultStruct  n = %d ******* ", n);
	dtype *Hd = alloc_arr2<dtype>(n * n); // diagonal Hd = D * H * D
	dtype *H1 = alloc_arr2<dtype>(n * n); // compressed H
	dtype *H2 = alloc_arr2<dtype>(n * n); // recovered H after D * H1 * D
	dtype *d = alloc_arr2<dtype>(n);
	char str[255];

	double norm = 0;
	int ldh = n;

	for (int j = 0; j < n; j++)
	{
		d[j] = j + 1;
	}

	Hilbert(n, n, H1, ldh);
	Hilbert(n, n, Hd, ldh);

	for (int j = 0; j < n; j++)
		for (int i = 0; i < n; i++)
		{
			Hd[i + ldh * j] *= d[j];
			Hd[i + ldh * j] *= d[i];
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
	norm = rel_error_complex(n, n, H2, Hd, ldh, eps);

	sprintf(str, "Struct: n = %d ", n);
	AssertLess(norm, eps, str);

	FreeNodes(n, HCstr, smallsize);
	free_arr(Hd); // diagonal Hd = D * H * D
	free_arr(H1); // compressed H
	free_arr(H2); // recovered H after D * H1 * D
	free_arr(d);
}

/* Тест на сравнение результатов умножения Y = H * X сжимаемой матрицы H на произвольную X.
Сравниваются результаты со сжатием и без */
void Test_RecMultLStruct(int n, int k, double eps, char *method, int smallsize)
{
	//printf("*****Test for RecMultLStruct  n = %d k = %d ******* ", n, k);
	dtype *H = alloc_arr2<dtype>(n * n); // init and compressed
	dtype *X = alloc_arr2<dtype>(n * k);
	dtype *Y = alloc_arr2<dtype>(n * k); // init Y
	dtype *Y1 = alloc_arr2<dtype>(n * k); // after multiplication woth compressed
	char str[255];

	double norm = 0;
	dtype alpha = 1.0;
	dtype beta = 0.0;

	int ldh = n;
	int ldy = n;
	int ldx = n;

	Hilbert(n, n, H, ldh);
	Hilbert(n, k, X, ldx);

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

	norm = rel_error_complex(n, k, Y1, Y, ldy, eps);
	sprintf(str, "Struct: n = %d k = %d ", n, k);
	AssertLess(norm, eps, str);

#ifdef DEBUG
	print(n, n, H, ldy, "H comp");
	print(n, k, Y1, ldy, "Y1 rec");
#endif

	FreeNodes(n, Hstr, smallsize);
	free_arr(H);
	free_arr(X);
	free_arr(Y);
	free_arr(Y1);
}

void Test_UnsymmRecMultLStruct(int n, int k, double eps, char *method, int smallsize)
{
	//printf("*****Test for RecMultLStruct  n = %d k = %d ******* ", n, k);
	dtype *H = alloc_arr2<dtype>(n * n); // init and compressed
	dtype *X = alloc_arr2<dtype>(n * k);
	dtype *Y = alloc_arr2<dtype>(n * k); // init Y
	dtype *Y1 = alloc_arr2<dtype>(n * k); // after multiplication woth compressed
	char str[255];

	double norm = 0;
	dtype alpha = 1.0;
	dtype beta = 0.0;

	int ldh = n;
	int ldy = n;
	int ldx = n;

	Hilbert3(n, n, H, ldh);
	Hilbert4(n, k, X, ldx);

	zgemm("No", "No", &n, &k, &n, &alpha, H, &ldh, X, &ldx, &beta, Y, &ldy);

#ifdef DEBUG
	print(n, n, H, ldy, "H init");
	print(n, k, X, ldy, "X init");
	print(n, k, Y, ldy, "Y init");
#endif

	cumnode *Hstr;
	// Compress H
	UnsymmRecCompressStruct(n, H, ldh, Hstr, smallsize, eps, method);

	// RecMult Y1 = comp(H) * X
	UnsymmRecMultLStruct(n, k, Hstr, X, ldx, Y1, ldy, smallsize);

	norm = rel_error_complex(n, k, Y1, Y, ldy, eps);
	sprintf(str, "Struct: n = %d k = %d ", n, k);
	AssertLess(norm, eps, str);

#ifdef DEBUG
	print(n, n, H, ldy, "H comp");
	print(n, k, Y1, ldy, "Y1 rec");
#endif

	FreeUnsymmNodes(n, Hstr, smallsize);
	free_arr(H);
	free_arr(X);
	free_arr(Y);
	free_arr(Y1);
}

/*Тест на сравнение результатов умножения Y = X * H сжимаемой матрицы H на произвольную X.
Сравниваются результаты со сжатием и без
(k x n) * (n x n) */
void Test_UnsymmRecMultRStruct(int n, int k, double eps, char *method, int smallsize)
{
	//printf("*****Test for RecMultLStruct  n = %d k = %d ******* ", n, k);
	dtype *H = alloc_arr2<dtype>(n * n); // init and compressed
	dtype *X = alloc_arr2<dtype>(k * n);
	dtype *Y = alloc_arr2<dtype>(k * n); // init Y
	dtype *Y1 = alloc_arr2<dtype>(k * n); // after multiplication woth compressed
	char str[255];

	double norm = 0;
	dtype alpha = 1.0;
	dtype beta = 0.0;

	int ldh = n;
	int ldy = k;
	int ldx = k;

	Hilbert4(n, n, H, ldh);
	Hilbert3(k, n, X, ldx);

	zgemm("No", "No", &k, &n, &n, &alpha, X, &ldx, H, &ldh, &beta, Y, &ldy);

#ifdef DEBUG
	print(n, n, H, ldy, "H init");
	print(n, k, X, ldy, "X init");
	print(n, k, Y, ldy, "Y init");
#endif

	cumnode *Hstr;
	// Compress H
	UnsymmRecCompressStruct(n, H, ldh, Hstr, smallsize, eps, method);

	// RecMult Y1 = X * comp(H)
	UnsymmRecMultRStruct(n, k, Hstr, X, ldx, Y1, ldy, smallsize);

	norm = rel_error_complex(k, n, Y1, Y, ldy, eps);
	sprintf(str, "Struct: n = %d k = %d ", n, k);
	AssertLess(norm, eps, str);

#ifdef DEBUG
	print(n, n, H, ldy, "H comp");
	print(n, k, Y1, ldy, "Y1 rec");
#endif

	FreeUnsymmNodes(n, Hstr, smallsize);
	free_arr(H);
	free_arr(X);
	free_arr(Y);
	free_arr(Y1);
}


void Test_AddStruct(int n, dtype alpha, dtype beta, double eps, char *method, int smallsize)
{
	//printf("*****Test for Add n = %d ******* ", n);
	dtype *H1 = alloc_arr2<dtype>(n * n);
	dtype *H2 = alloc_arr2<dtype>(n * n);
	dtype *G = alloc_arr2<dtype>(n * n);
	dtype *H1c = alloc_arr2<dtype>(n * n);
	dtype *H2c = alloc_arr2<dtype>(n * n);
	dtype *GcR = alloc_arr2<dtype>(n * n);
	char str[255];

	int ldh = n;
	int ldg = n;
	double norm = 0;

	Hilbert(n, n, H1, ldh);
	Hilbert(n, n, H1c, ldh);

#pragma omp parallel for simd schedule(simd:static)
	for (int j = 0; j < n; j++)
		for (int i = 0; i < n; i++)
		{
			H2[i + ldh * j] = 1.0 / (i*i + j*j + 1);
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
	norm = rel_error_complex(n, n, GcR, G, ldg, eps);
	sprintf(str, "Struct: n = %d n = %d alpha = %lf", n, n, alpha.real(), beta.real());
	AssertLess(norm, eps, str);

	FreeNodes(n, H1str, smallsize);
	FreeNodes(n, H2str, smallsize);
	FreeNodes(n, Gstr, smallsize);
	free_arr(H1);
	free_arr(H2);
	free_arr(G);
	free_arr(H1c);
	free_arr(H2c);
	free_arr(GcR);
}

void Test_UnsymmAddStruct(int n, dtype alpha, dtype beta, double eps, char *method, int smallsize)
{
	//printf("*****Test for Add n = %d ******* ", n);
	dtype *H1 = alloc_arr2<dtype>(n * n);
	dtype *H2 = alloc_arr2<dtype>(n * n);
	dtype *G = alloc_arr2<dtype>(n * n);
	dtype *H1c = alloc_arr2<dtype>(n * n);
	dtype *H2c = alloc_arr2<dtype>(n * n);
	dtype *GcR = alloc_arr2<dtype>(n * n);
	char str[255];

	int ldh = n;
	int ldg = n;
	double norm = 0;

	Hilbert3(n, n, H1, ldh);
	Hilbert3(n, n, H1c, ldh);

#pragma omp parallel for simd schedule(simd:static)
	for (int j = 0; j < n; j++)
		for (int i = 0; i < n; i++)
		{
			H2[i + ldh * j] = 1.0 / (i*i + j * j + 1);
			H2c[i + ldh * j] = 1.0 / (i*i + j * j + 1);
		}

#ifdef DEBUG
	print(n, n, H1, ldh, "H1");
	print(n, n, H2, ldh, "H2");
#endif

	cumnode *H1str, *H2str;
	UnsymmRecCompressStruct(n, H1c, ldh, H1str, smallsize, eps, method);
	UnsymmRecCompressStruct(n, H2c, ldh, H2str, smallsize, eps, method);

#ifdef DEBUG
	print(n, n, H1c, ldh, "H1c");
	print(n, n, H2c, ldh, "H2c");
#endif

	cumnode *Gstr;
	Add_dense(n, n, alpha, H1, ldh, beta, H2, ldh, G, ldg);
	UnsymmAddStruct(n, alpha, H1str, beta, H2str, Gstr, smallsize, eps, method);

#ifdef DEBUG
	print(n, n, G, ldg, "res_dense");
	print(n, n, Gc, ldg, "res_comp");
#endif

	UnsymmResRestoreStruct(n, Gstr, GcR, ldg, smallsize);

#ifdef DEBUG
	print(n, n, GcR, ldg, "res_comp_restore");
#endif
	// |GcR - G| / |G|
	norm = rel_error_complex(n, n, GcR, G, ldg, eps);
	sprintf(str, "Struct: n = %d n = %d alpha = %lf beta = %lf", n, n, alpha.real(), beta.real());
	AssertLess(norm, eps, str);

	FreeUnsymmNodes(n, H1str, smallsize);
	FreeUnsymmNodes(n, H2str, smallsize);
	FreeUnsymmNodes(n, Gstr, smallsize);
	free_arr(H1);
	free_arr(H2);
	free_arr(G);
	free_arr(H1c);
	free_arr(H2c);
	free_arr(GcR);
}

// B = H - V * Y * VT
void Test_SymCompUpdate2Struct(int n, int k, dtype alpha, double eps, char* method, int smallsize)
{
	//printf("*****Test for SymCompUpdate2Struct  n = %d k = %d ***** ", n, k);
	dtype *B_rec = alloc_arr<dtype>(n * n);
	dtype *Y = alloc_arr2<dtype>(k * k); int ldy = k;
	dtype *V = alloc_arr2<dtype>(n * k); int ldv = n; int ldvtr = k;
	dtype *HC = alloc_arr2<dtype>(n * n); int ldh = n;
	dtype *H = alloc_arr2<dtype>(n * n);
	dtype *C = alloc_arr2<dtype>(n * k); int ldc = n;
	char str[255];

	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;
	double norm = 0;


	Hilbert(n, n, HC, ldh);
	Hilbert(n, n, H, ldh);

	Clear(k, k, Y, ldy);
	Clear(n, k, V, ldv);

#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < k; i++)
		Y[i + ldy * i] = i + 1;

#pragma omp parallel for simd schedule(simd:static)
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
	norm = rel_error_complex(n, n, B_rec, H, ldh, eps);
	sprintf(str, "Struct: n = %d k = %d alpha = %lf", n, k, alpha.real());
	AssertLess(norm, eps, str);

	FreeNodes(n, Bstr, smallsize);
	FreeNodes(n, HCstr, smallsize);
	free_arr(B_rec);
	free_arr(H);
	free_arr(HC);
	free_arr(Y);
	free_arr(C);
	free_arr(V);
}

void Test_UnsymmCompUpdate2Struct(int n, int k, dtype alpha, double eps, char* method, int smallsize)
{
	//printf("*****Test for SymCompUpdate2Struct  n = %d k = %d ***** ", n, k);
	dtype *B_rec = alloc_arr<dtype>(n * n);
	dtype *Y = alloc_arr2<dtype>(k * k); int ldy = k;
	dtype *V = alloc_arr2<dtype>(n * k); int ldv = n; int ldvtr = k;
	dtype *HC = alloc_arr2<dtype>(n * n); int ldh = n;
	dtype *H = alloc_arr2<dtype>(n * n);
	dtype *C = alloc_arr2<dtype>(n * k); int ldc = n;
	char str[255];

	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;
	double norm = 0;


	Hilbert3(n, n, HC, ldh);
	Hilbert3(n, n, H, ldh);

	Clear(k, k, Y, ldy);
	Clear(n, k, V, ldv);


	Hilbert2(k, k, Y, ldy);
	Hilbert4(n, k, V, ldv);

	// C = V * Y
	zgemm("No", "No", &n, &k, &k, &alpha_one, V, &ldv, Y, &ldy, &beta_zero, C, &ldc);

	// H = H + alpha * C * VT
	zgemm("No", "Trans", &n, &n, &k, &alpha, C, &ldc, V, &ldv, &beta_one, H, &ldh);

	cumnode *HCstr;
	// Compressed update
	UnsymmRecCompressStruct(n, HC, ldh, HCstr, smallsize, eps, method);

	cumnode *Bstr;
	UnsymmCompUpdate2Struct(n, k, HCstr, alpha, Y, ldy, V, ldv, Bstr, smallsize, eps, method);
	UnsymmResRestoreStruct(n, Bstr, B_rec, ldh, smallsize);

#ifdef DEBUG
	print(n, n, B_rec, ldb, "B_rec");
	print(n, n, H, ldh, "H");
#endif

	// || B_rec - H || / || H ||
	norm = rel_error_complex(n, n, B_rec, H, ldh, eps);
	sprintf(str, "Struct: n = %d k = %d alpha = %lf", n, k, alpha.real());
	AssertLess(norm, eps, str);

	FreeUnsymmNodes(n, Bstr, smallsize);
	FreeUnsymmNodes(n, HCstr, smallsize);
	free_arr(B_rec);
	free_arr(H);
	free_arr(HC);
	free_arr(Y);
	free_arr(C);
	free_arr(V);
}

/* (n x k) * (k x k) * (k x n) 
k << n */

void Test_UnsymmCompUpdate3Struct(int n, int k1, int k2, dtype alpha, double eps, char* method, int smallsize)
{
	//printf("*****Test for SymCompUpdate2Struct  n = %d k = %d ***** ", n, k);
	dtype *B_rec = alloc_arr<dtype>(n * n);
	dtype *Y = alloc_arr2<dtype>(k1 * k2); int ldy = k1;
	dtype *V1 = alloc_arr2<dtype>(n * k1); int ldv1 = n;
	dtype *V2 = alloc_arr2<dtype>(k2 * n); int ldv2 = k2;
	dtype *HC = alloc_arr2<dtype>(n * n); int ldh = n;
	dtype *H = alloc_arr2<dtype>(n * n);
	dtype *C = alloc_arr2<dtype>(n * k2); int ldc = n;
	char str[255];

	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;
	double norm = 0;

	Clear(n, n, HC, ldh);
	Clear(n, n, H, ldh);

	Hilbert5(n, n, HC, ldh);
	Hilbert5(n, n, H, ldh);

	Clear(k1, k2, Y, ldy);
	Clear(n, k1, V1, ldv1);
	Clear(k2, n, V2, ldv2);

	int kmin = min(k1, k2);

	Hilbert2(k1, k2, Y, ldy);
	Hilbert3(n, k1, V1, ldv1);
	Hilbert4(k2, n, V2, ldv2);

	// C = V1 * Y
	zgemm("No", "No", &n, &k2, &k1, &alpha_one, V1, &ldv1, Y, &ldy, &beta_zero, C, &ldc);

	// H = H + alpha * C * V2
	zgemm("No", "No", &n, &n, &k2, &alpha, C, &ldc, V2, &ldv2, &beta_one, H, &ldh);

	cumnode *HCstr;
	// Compressed update
	UnsymmRecCompressStruct(n, HC, ldh, HCstr, smallsize, eps, method);

	cumnode *Bstr;
	UnsymmCompUpdate3Struct(n, k1, k2, HCstr, alpha, Y, ldy, V1, ldv1, V2, ldv2, Bstr, smallsize, eps, method);
#if 0
	printf("----------\n");
	printf("A: ");
	UnsymmPrintRanksInWidthList(HCstr);
	printf("\nB: ");
	UnsymmPrintRanksInWidthList(Bstr);
	printf("---------\n");
	UnsymmClearStruct(n, HCstr, smallsize)
	UnsymmCopyStruct(n, Bstr, HCstr, smallsize);
#endif
	UnsymmResRestoreStruct(n, Bstr, B_rec, ldh, smallsize);

#ifdef DEBUG
	print(n, n, B_rec, ldb, "B_rec");
	print(n, n, H, ldh, "H");
#endif

	// || B_rec - H || / || H ||
	norm = rel_error_complex(n, n, B_rec, H, ldh, eps);
	sprintf(str, "Struct: n = %d k1 = %d k2 = %d  alpha = %lf", n, k1, k2, alpha.real());
	AssertLess(norm, eps, str);

	FreeUnsymmNodes(n, Bstr, smallsize);
	FreeUnsymmNodes(n, HCstr, smallsize);
	free_arr(B_rec);
	free_arr(H);
	free_arr(HC);
	free_arr(Y);
	free_arr(C);
	free_arr(V1);
	free_arr(V2);
}
#ifdef CHOLESKY
void Test_SymCompUpdate4LowRankStruct(int n, dtype alpha, double eps, char* method, int smallsize)
{
	//printf("*****Test for SymCompUpdate2Struct  n = %d k = %d ***** ", n, k);
	dtype *B_rec = alloc_arr<dtype>(n * n);
	dtype *HC = alloc_arr2<dtype>(n * n); int ldh = n;
	dtype *H = alloc_arr2<dtype>(n * n);
	char str[255];

	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;
	double norm = 0;

	Clear(n, n, HC, ldh);
	Clear(n, n, H, ldh);

	Hilbert7LowRank(n, n, HC, ldh);
	Hilbert7LowRank(n, n, H, ldh);

	cmnode *HCstr = (cmnode*)malloc(sizeof(cmnode));
	// Compressed update
	LowRankApproxStruct(n, n, HC, ldh, HCstr, eps, method);
	int p = HCstr->p;

	// Gen intermediate matrix Y
	dtype *Y = alloc_arr2<dtype>(p * p); int ldy = p;
	dtype *C = alloc_arr2<dtype>(n * p); int ldc = n;
	Clear(p, p, Y, ldy);
	Hilbert4(p, p, Y, ldy);

	cumnode *HChss;
	LowRankToUnsymmHSS(n, p, HCstr->U, n, HCstr->VT, HCstr->p, HChss, smallsize);

	cumnode *Bstr;
	SymCompUpdate4LowRankStruct(n, p, p, HChss, alpha, Y, ldy, HCstr->U, n, HCstr->VT, p, Bstr, smallsize, eps, method);

	// C = V1 * Y
	zgemm("No", "No", &n, &p, &p, &alpha_one, HCstr->U, &n, Y, &ldy, &beta_zero, C, &ldc);

	// H = H + alpha * C * V2
	zgemm("No", "No", &n, &n, &p, &alpha, C, &ldc, HCstr->VT, &p, &beta_one, H, &ldh);
#if 0
	printf("----------\n");
	printf("A: ");
	UnsymmPrintRanksInWidthList(HCstr);
	printf("\nB: ");
	UnsymmPrintRanksInWidthList(Bstr);
	printf("---------\n");
	UnsymmClearStruct(n, HCstr, smallsize)
		UnsymmCopyStruct(n, Bstr, HCstr, smallsize);
#endif
	UnsymmResRestoreStruct(n, Bstr, B_rec, ldh, smallsize);

#ifdef DEBUG
	print(n, n, B_rec, ldb, "B_rec");
	print(n, n, H, ldh, "H");
#endif

	// || B_rec - H || / || H ||
	norm = rel_error_complex(n, n, B_rec, H, ldh, eps);
	sprintf(str, "Struct: n = %d k1 = %d k2 = %d  alpha = %lf", n, p, p, alpha_one.real());
	AssertLess(norm, eps, str);

	FreeUnsymmNodes(n, Bstr, smallsize);
	FreeUnsymmNodes(n, HChss, smallsize);
	free_arr(HCstr->U);
	free_arr(HCstr->VT);
	free_arr(HCstr);
	free_arr(B_rec);
	free_arr(H);
	free_arr(HC);
	free_arr(Y);
	free_arr(C);
}

void Test_SymCompUpdate5LowRankStruct(int n, int p, dtype alpha, double eps, char* method, int smallsize)
{
	//printf("*****Test for SymCompUpdate2Struct  n = %d k = %d ***** ", n, k);
	dtype *B_rec = alloc_arr<dtype>(n * n); int ldh = n;
	dtype *H = alloc_arr2<dtype>(n * n);
	dtype *W = alloc_arr<dtype>(n * p); int ldw = n;
	dtype *Y = alloc_arr2<dtype>(p * p); int ldy = p;
	dtype *C = alloc_arr2<dtype>(n * p); int ldc = n;
	char str[255];

	dtype alpha_one = 1.0;
	dtype beta_zero = 0.0;
	dtype beta_one = 1.0;
	double norm = 0;

	Clear(n, n, H, ldh);

	Hilbert5(n, p, W, ldw);
	zsyrk("Low", "No", &n, &p, &alpha_one, W, &ldw, &beta_zero, H, &ldh);

	// Gen intermediate symmetric matrix Y
	Clear(p, p, Y, ldy);
	Hilbert(p, p, Y, ldy);

	cmnode *HChss;
	LowRankToSymmHSS(n, p, W, ldw, HChss, smallsize);

	SymCompUpdate5LowRankStruct(n, p, HChss, alpha, Y, ldy, W, ldw, smallsize, eps, method);

	// C = V1 * Y
	zsymm("Right", "Low", &n, &p, &alpha_one, Y, &ldy, W, &ldw, &beta_zero, C, &ldc);

	// H = H + alpha * C * V2
	zgemm("No", "Trans", &n, &n, &p, &alpha, C, &ldc, W, &ldw, &beta_one, H, &ldh);

	SymResRestoreStruct(n, HChss, B_rec, ldh, smallsize);

#ifdef DEBUG
	print(n, n, B_rec, ldb, "B_rec");
	print(n, n, H, ldh, "H");
#endif

	// || B_rec - H || / || H ||
	
	norm = RelErrorPart(zlange, 'L', n, n, B_rec, ldh, H, ldh, eps);
//	printf("norm update = %e\n", norm);
	sprintf(str, "Struct: n = %d p = %d alpha = %lf", n, p, alpha.real());
	AssertLess(norm, eps, str);

	FreeNodes(n, HChss, smallsize);
	free_arr(B_rec);
	free_arr(H);
	free_arr(Y);
	free_arr(C);
	free_arr(W);
}

void Test_SymCompRecInvStruct(int n, double eps, char *method, int smallsize)
{
	//printf("***** Test_SymCompRecInvStruct n = %d eps = %lf ****", n, eps);
	dtype *H = alloc_arr2<dtype>(n * n);
	dtype *Hc = alloc_arr2<dtype>(n * n);
	dtype *Brec = alloc_arr2<dtype>(n * n);
	dtype *Y = alloc_arr2<dtype>(n * n);
	char str[255];

	int ldh = n;
	int ldb = n;
	int ldy = n;

	dtype alpha_mone = -1.0;
	dtype beta_one = 1.0;
	double norm = 0;

	Hilbert(n, n, H, ldh);
	Hilbert(n, n, Hc, ldh);

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
	free_arr(Brec);
	free_arr(Y);
}
#endif
void Test_CopyStruct(int n, double eps, char *method, int smallsize)
{
	dtype *H = alloc_arr2<dtype>(n * n);
	dtype *H1 = alloc_arr2<dtype>(n * n);
	dtype *H2 = alloc_arr2<dtype>(n * n);
	char str[255];

	double norm = 0;
	int ldh = n;

	//printf("***Test CopyStruct n = %d ", n);

	Hilbert(n, n, H, ldh);
	Hilbert(n, n, H1, ldh);

	cmnode* Hstr, *Hcopy_str;
	SymRecCompressStruct(n, H, ldh, Hstr, smallsize, eps, method);
	CopyStruct(n, Hstr, Hcopy_str, smallsize);
	SymResRestoreStruct(n, Hcopy_str, H2, ldh, smallsize);

	norm = rel_error_complex(n, n, H2, H1, ldh, eps);
	sprintf(str, "Struct: n = %d", n);
	AssertLess(norm, eps, str);

	FreeNodes(n, Hstr, smallsize);
	FreeNodes(n, Hcopy_str, smallsize);
	free_arr(H2);
	free_arr(H1);
	free_arr(H);
}

void Test_UnsymmCompRecInvStruct(int n, double eps, char *method, int smallsize)
{
	//printf("***** Test_SymCompRecInvStruct n = %d eps = %lf ****", n, eps);
	dtype *H = alloc_arr2<dtype>(n * n);
	dtype *Hc = alloc_arr2<dtype>(n * n);
	dtype *Brec = alloc_arr2<dtype>(n * n);
	dtype *Y = alloc_arr2<dtype>(n * n);
	char str[255];

	int ldh = n;
	int ldb = n;
	int ldy = n;

	dtype alpha_mone = -1.0;
	dtype beta_one = 1.0;
	double norm = 0;

	Hilbert2(n, n, H, ldh);
	Hilbert2(n, n, Hc, ldh);

	// for stability
	for (int i = 0; i < n; i++)
	{
		H[i + ldh * i] += 5.0;
		Hc[i + ldh * i] += 5.0;
	}

	cumnode *HCstr, *BCstr;
	UnsymmRecCompressStruct(n, Hc, ldh, HCstr, smallsize, eps, method);
	UnsymmCompRecInvStruct(n, HCstr, BCstr, smallsize, eps, method);
	UnsymmResRestoreStruct(n, BCstr, Brec, ldb, smallsize);

	Eye(n, Y, ldy);

	// Y = Y - H * Brec
	zgemm("No", "No", &n, &n, &n, &alpha_mone, H, &ldh, Brec, &ldb, &beta_one, Y, &ldy);

	norm = zlange("Frob", &n, &n, Y, &ldy, NULL);
	sprintf(str, "Struct: n = %d", n);
	AssertLess(norm, eps, str);

	//if (norm < eps) printf("Norm %10.8e < eps %10.8lf: PASSED\n", norm, eps);
	//else printf("Norm %10.8lf > eps %10.8e : FAILED\n", norm, eps);

	FreeUnsymmNodes(n, HCstr, smallsize);
	FreeUnsymmNodes(n, BCstr, smallsize);
	free_arr(H);
	free_arr(Hc);
	free_arr(Brec);
	free_arr(Y);
}

void Test_CopyUnsymmStruct(int n, double eps, char *method, int smallsize)
{
	dtype *H = alloc_arr2<dtype>(n * n);
	dtype *H1 = alloc_arr2<dtype>(n * n);
	dtype *H2 = alloc_arr2<dtype>(n * n);
	char str[255];

	double norm = 0;
	int ldh = n;

	//printf("***Test CopyStruct n = %d ", n);

	Hilbert(n, n, H, ldh);
	Hilbert(n, n, H1, ldh);

	cumnode* Hstr, *Hcopy_str;
	UnsymmRecCompressStruct(n, H, ldh, Hstr, smallsize, eps, method);
	CopyUnsymmStruct(n, Hstr, Hcopy_str, smallsize);
	UnsymmResRestoreStruct(n, Hcopy_str, H2, ldh, smallsize);

	norm = rel_error_complex(n, n, H2, H1, ldh, eps);
	sprintf(str, "Struct: n = %d", n);
	AssertLess(norm, eps, str);

	FreeUnsymmNodes(n, Hstr, smallsize);
	FreeUnsymmNodes(n, Hcopy_str, smallsize);
	free_arr(H2);
	free_arr(H1);
	free_arr(H);
}

void Test_ApplyToA21(int m, int n, double eps, char* method, int smallsize)
{
	dtype *LU = alloc_arr2<dtype>(n * n);
	dtype *H1 = alloc_arr2<dtype>(m * n);
	dtype *H2 = alloc_arr2<dtype>(m * n);
	dtype *H3 = alloc_arr2<dtype>(m * n);
	int *ipiv = alloc_arr2<int>(n);
	char str[255];

	double norm = 0;
	int ldh = m;
	int info = 0;
	int ione = 1.0;
	dtype alpha = 1.0;
	dtype beta = 0.0;

	Hilbert3(n, n, LU, n);
	Hilbert2(m, n, H1, ldh);
	Hilbert2(m, n, H2, ldh);

	for (int i = 0; i < n; i++)
	{
		LU[i + n * i] -= 5.0;
	}

	cumnode* LUstr, *Rstr;
	cmnode* H2str = (cmnode*)malloc(sizeof(cmnode));
	zgetrf(&n, &n, LU, &n, ipiv, &info);

//	for (int i = 0; i < n; i++)
	//	if (ipiv[i] != i + 1) printf("ROW CHANGE!\n");

	ztrsm("Right", "Up", "No", "NonUnit", &m, &n, &alpha, LU, &n, H1, &ldh);

	UnsymmRecCompressStruct(n, LU, n, LUstr, smallsize, eps, method);
	CopyRfactor(n, LUstr, Rstr, smallsize);

	LowRankApproxStruct(m, n, H2, ldh, H2str, eps, method);
	ApplyToA21(n, H2str, Rstr, smallsize, eps, method);
	zgemm("no", "no", &m, &n, &H2str->p, &alpha, H2str->U, &m, H2str->VT, &H2str->p, &beta, H3, &ldh);

	norm = rel_error_complex(m, n, H3, H1, ldh, eps);
	sprintf(str, "Struct: sz = %d m = %d n = %d", smallsize, m, n);
	AssertLess(norm, eps, str);

	FreeUnsymmNodes(n, LUstr, smallsize);
	free_arr(H1);
	free_arr(H2);
	free_arr(H3);
	free_arr(LU);
	free_arr(ipiv);
}

void Test_ApplyToA21Ver2(int m, int n, double eps, char* method, int smallsize)
{
	dtype *LU = alloc_arr2<dtype>(n * n);
	dtype *H1 = alloc_arr2<dtype>(m * n);
	dtype *H2 = alloc_arr2<dtype>(m * n);
	dtype *H3 = alloc_arr2<dtype>(m * n);
	int *ipiv = alloc_arr2<int>(n);
	char str[255];

	double norm = 0;
	int ldh = m;
	int info = 0;
	int ione = 1.0;
	dtype alpha = 1.0;
	dtype beta = 0.0;

	Hilbert3(n, n, LU, n);
	Hilbert2(m, n, H1, ldh);
	Hilbert2(m, n, H2, ldh);

	for (int i = 0; i < n; i++)
	{
		LU[i + n * i] -= 5.0;
	}

	cumnode* LUstr, *Rstr;
	cmnode* H2str = (cmnode*)malloc(sizeof(cmnode));
	zgetrf(&n, &n, LU, &n, ipiv, &info);

	//	for (int i = 0; i < n; i++)
		//	if (ipiv[i] != i + 1) printf("ROW CHANGE!\n");

	ztrsm("Right", "Up", "No", "NonUnit", &m, &n, &alpha, LU, &n, H1, &ldh);

	UnsymmRecCompressStruct(n, LU, n, LUstr, smallsize, eps, method);
	CopyRfactor(n, LUstr, Rstr, smallsize);

	LowRankApproxStruct(m, n, H2, ldh, H2str, eps, method);
	ApplyToA21Ver2(H2str->p, n, H2str->VT, H2str->p, Rstr, smallsize, eps, method);
	zgemm("no", "no", &m, &n, &H2str->p, &alpha, H2str->U, &m, H2str->VT, &H2str->p, &beta, H3, &ldh);

	norm = rel_error_complex(m, n, H3, H1, ldh, eps);
	sprintf(str, "Struct: sz = %d m = %d n = %d", smallsize, m, n);
	AssertLess(norm, eps, str);

	FreeUnsymmNodes(n, LUstr, smallsize);
	free_arr(H1);
	free_arr(H2);
	free_arr(H3);
	free_arr(LU);
	free_arr(ipiv);
}

void Test_SolveTriangSystemA21(int m, int n, double eps, char* method, int smallsize)
{
	dtype *LLT = alloc_arr2<dtype>(n * n);
	dtype *H1 = alloc_arr2<dtype>(m * n);
	dtype *H2 = alloc_arr2<dtype>(m * n);
	dtype *H3 = alloc_arr2<dtype>(m * n);
	int *ipiv = alloc_arr2<int>(n);
	char str[255];

	double norm = 0;
	int ldh = m;
	int info = 0;
	int ione = 1.0;
	dtype alpha = 1.0;
	dtype beta = 0.0;

	Hilbert(n, n, LLT, n);
	Hilbert2(m, n, H1, ldh);
	Hilbert2(m, n, H2, ldh);

	for (int i = 0; i < n; i++)
	{
		LLT[i + n * i] += 10.0;
	}

	cmnode* LLTstr;
	cmnode* H2str = (cmnode*)malloc(sizeof(cmnode));

	zpotrf("Low", &n, LLT, &n, &info);
	ztrsm("Right", "Low", "Trans", "NonUnit", &m, &n, &alpha, LLT, &n, H1, &ldh);

	SymRecCompressStruct(n, LLT, n, LLTstr, smallsize, eps, method);

	LowRankApproxStruct(m, n, H2, ldh, H2str, eps, method);
	SolveTriangSystemA21(H2str->p, n, H2str->VT, H2str->p, LLTstr, smallsize, eps, method);
	zgemm("no", "no", &m, &n, &H2str->p, &alpha, H2str->U, &m, H2str->VT, &H2str->p, &beta, H3, &ldh);

	norm = rel_error_complex(m, n, H3, H1, ldh, eps);
	sprintf(str, "Struct: sz = %d m = %d n = %d", smallsize, m, n);
	AssertLess(norm, eps, str);

	FreeNodes(n, LLTstr, smallsize);
	free_arr(H1);
	free_arr(H2);
	free_arr(H3);
	free_arr(LLT);
	free_arr(ipiv);
}


void Test_ApplyToA12(int m, int n, double eps, char* method, int smallsize)
{
#undef PRINT
	dtype *LU = alloc_arr2<dtype>(n * n);
	dtype *H1 = alloc_arr2<dtype>(m * n);
	dtype *H2 = alloc_arr2<dtype>(m * n);
	dtype *H3 = alloc_arr2<dtype>(m * n);
	int *ipiv = alloc_arr2<int>(n);
	char str[255];

	double norm = 0;
	int ldh = n;
	int info = 0;
	int ione = 1;
	dtype alpha = 1.0;
	dtype beta = 0.0;

	Hilbert3(n, n, LU, n);
	Hilbert2(n, m, H1, ldh);
	Hilbert2(n, m, H2, ldh);

	for (int i = 0; i < n; i++)
	{
		LU[i + n * i] -= 5.0;
	}

	cumnode* LUstr, *Lstr;
	cmnode* H2str = (cmnode*)malloc(sizeof(cmnode));
	zgetrf(&n, &n, LU, &n, ipiv, &info);

//	for (int i = 0; i < n; i++)
	//	if (ipiv[i] != i + 1) printf("ROW CHANGE!\n");

	zlaswp(&m, H1, &ldh, &ione, &n, ipiv, &ione);

	ztrsm("Left", "Low", "No", "Unit", &n, &m, &alpha, LU, &n, H1, &ldh);

	UnsymmRecCompressStruct(n, LU, n, LUstr, smallsize, eps, method);
	CopyLfactor(n, LUstr, Lstr, smallsize);

	//(правильный тест с пивотингом)
	// ipiv -> ipiv2
#ifdef PRINT
	printf("\nGet Nleaves\n");
#endif
	int nleaves = 0;
	nleaves = GetNumberOfLeaves(LUstr);
	int *dist = alloc_arr<int>(nleaves);
	int count = 0;
	int nn = 0;

#ifdef  PRINT
	printf("\nget distances\n");
#endif //  PRINT

#if 0
	GetDistances(LUstr, dist, count);
	for (int j = 0; j < nleaves; j++)
	{
		for (int i = 0; i < dist[j]; i++)
		{
			ipiv2[nn + i] = ipiv[nn + i] - nn;
		}
		nn += dist[j];
	}
#endif

	//--------------------

	LowRankApproxStruct(n, m, H2, ldh, H2str, eps, method);
	zlaswp(&H2str->p, H2str->U, &n, &ione, &n, ipiv, &ione);

	ApplyToA12(n, H2str, Lstr, smallsize, eps, method);
	zgemm("no", "no", &n, &m, &H2str->p, &alpha, H2str->U, &n, H2str->VT, &H2str->p, &beta, H3, &ldh);

	norm = rel_error_complex(n, m, H3, H1, ldh, eps);
	sprintf(str, "Struct: sz = %d n = %d m = %d", smallsize, n, m);
	AssertLess(norm, eps, str);

	FreeUnsymmNodes(n, LUstr, smallsize);
	free_arr(H1);
	free_arr(H2);
	free_arr(H3);
	free_arr(LU);
	free_arr(ipiv);
}

void Test_ApplyToA12Ver2(int m, int n, double eps, char* method, int smallsize)
{
	dtype *LU = alloc_arr2<dtype>(n * n);
	dtype *H1 = alloc_arr2<dtype>(m * n);
	dtype *H2 = alloc_arr2<dtype>(m * n);
	dtype *H3 = alloc_arr2<dtype>(m * n);
	int *ipiv = alloc_arr2<int>(n);
	char str[255];

	double norm = 0;
	int ldh = n;
	int info = 0;
	int ione = 1;
	dtype alpha = 1.0;
	dtype beta = 0.0;

	Hilbert3(n, n, LU, n);
	Hilbert2(n, m, H1, ldh);
	Hilbert2(n, m, H2, ldh);

	for (int i = 0; i < n; i++)
	{
		LU[i + n * i] -= 5.0;
	}

	cumnode* LUstr, *Lstr;
	cmnode* H2str = (cmnode*)malloc(sizeof(cmnode));
	zgetrf(&n, &n, LU, &n, ipiv, &info);

	//	for (int i = 0; i < n; i++)
		//	if (ipiv[i] != i + 1) printf("ROW CHANGE!\n");

	zlaswp(&m, H1, &ldh, &ione, &n, ipiv, &ione);

	ztrsm("Left", "Low", "No", "Unit", &n, &m, &alpha, LU, &n, H1, &ldh);

	UnsymmRecCompressStruct(n, LU, n, LUstr, smallsize, eps, method);
	CopyLfactor(n, LUstr, Lstr, smallsize);

	//(правильный тест с пивотингом)
	// ipiv -> ipiv2
#ifdef PRINT
	printf("\nGet Nleaves\n");
#endif
	int nleaves = 0;
	nleaves = GetNumberOfLeaves(LUstr);
	int *dist = alloc_arr<int>(nleaves);
	int count = 0;
	int nn = 0;

#ifdef  PRINT
	printf("\nget distances\n");
#endif //  PRINT

#if 0
	GetDistances(LUstr, dist, count);
	for (int j = 0; j < nleaves; j++)
	{
		for (int i = 0; i < dist[j]; i++)
		{
			ipiv2[nn + i] = ipiv[nn + i] - nn;
		}
		nn += dist[j];
	}
#endif

	//--------------------

	LowRankApproxStruct(n, m, H2, ldh, H2str, eps, method);
	zlaswp(&H2str->p, H2str->U, &n, &ione, &n, ipiv, &ione);

	ApplyToA12Ver2(n, H2str->p, H2str->U, n, Lstr, smallsize, eps, method);
	zgemm("no", "no", &n, &m, &H2str->p, &alpha, H2str->U, &n, H2str->VT, &H2str->p, &beta, H3, &ldh);

	norm = rel_error_complex(n, m, H3, H1, ldh, eps);
	sprintf(str, "Struct: sz = %d n = %d m = %d", smallsize, n, m);
	AssertLess(norm, eps, str);

	FreeUnsymmNodes(n, LUstr, smallsize);
	free_arr(H1);
	free_arr(H2);
	free_arr(H3);
	free_arr(LU);
	free_arr(ipiv);
}

void Test_UnsymmCopyStruct(int n, double eps, char *method, int smallsize)
{
	dtype *H = alloc_arr2<dtype>(n * n);
	dtype *H1 = alloc_arr2<dtype>(n * n);
	dtype *H2 = alloc_arr2<dtype>(n * n);
	dtype *Hres = alloc_arr2<dtype>(n * n);
	char str[255];

	double norm = 0;
	int ldh = n;

	//printf("***Test CopyStruct n = %d ", n);

	Hilbert2(n, n, H, ldh);
	Hilbert2(n, n, H1, ldh);
	Hilbert2(n, n, H2, ldh);

	cumnode* Hstr, *Hcp_str;
	UnsymmRecCompressStruct(n, H, ldh, Hstr, smallsize, eps, method);
	UnsymmRecCompressStruct(n, H1, ldh, Hcp_str, smallsize, eps, method);

	UnsymmClearStruct(n, Hcp_str, smallsize);

	UnsymmCopyStruct(n, Hstr, Hcp_str, smallsize);
	UnsymmResRestoreStruct(n, Hcp_str, Hres, ldh, smallsize);

	norm = rel_error_complex(n, n, H2, Hres, ldh, eps);
	sprintf(str, "Struct: n = %d", n);
	AssertLess(norm, eps, str);

	FreeUnsymmNodes(n, Hstr, smallsize);
	FreeUnsymmNodes(n, Hcp_str, smallsize);
	free_arr(H2);
	free_arr(H1);
	free_arr(H);
}

void Test_CopyLfactorStruct(int n, double eps, char *method, int smallsize)
{
	dtype *H = alloc_arr2<dtype>(n * n);
	dtype *H1 = alloc_arr2<dtype>(n * n);
	dtype *H2 = alloc_arr2<dtype>(n * n);
	char str[255];

	double norm = 0;
	int ldh = n;

	//printf("***Test CopyStruct n = %d ", n);

	Hilbert(n, n, H, ldh);
	Hilbert(n, n, H1, ldh);

	for (int i = 0; i < n; i++)
		H[i + ldh * i] = H1[i + ldh * i] = 1.0;

	cumnode* Hstr, *Lstr;
	UnsymmRecCompressStruct(n, H, ldh, Hstr, smallsize, eps, method);
	CopyLfactor(n, Hstr, Lstr, smallsize);
	UnsymmResRestoreStruct(n, Lstr, H2, ldh, smallsize);

	norm = RelErrorPart(zlange, 'L', n, n, H2, ldh, H1, ldh, eps);
	sprintf(str, "Struct: n = %d", n);
	AssertLess(norm, eps, str);

	FreeUnsymmNodes(n, Hstr, smallsize);
	FreeUnsymmNodes(n, Lstr, smallsize);
	free_arr(H2);
	free_arr(H1);
	free_arr(H);
}

void Test_CopyRfactorStruct(int n, double eps, char *method, int smallsize)
{
	dtype *H = alloc_arr2<dtype>(n * n);
	dtype *H1 = alloc_arr2<dtype>(n * n);
	dtype *H2 = alloc_arr2<dtype>(n * n);
	char str[255];

	double norm = 0;
	int ldh = n;

	//printf("***Test CopyStruct n = %d ", n);

	Hilbert(n, n, H, ldh);
	Hilbert(n, n, H1, ldh);

	for (int i = 0; i < n; i++)
		H[i + ldh * i] = H1[i + ldh * i] = 1.0;

	cumnode* Hstr, *Rstr;
	UnsymmRecCompressStruct(n, H, ldh, Hstr, smallsize, eps, method);
	CopyRfactor(n, Hstr, Rstr, smallsize);
	UnsymmResRestoreStruct(n, Rstr, H2, ldh, smallsize);

	norm = RelErrorPart(zlange, 'U', n, n, H2, ldh, H1, ldh, eps);
	sprintf(str, "Struct: n = %d", n);
	AssertLess(norm, eps, str);

	FreeUnsymmNodes(n, Hstr, smallsize);
	FreeUnsymmNodes(n, Rstr, smallsize);
	free_arr(H2);
	free_arr(H1);
	free_arr(H);
}

void Test_PermutLowRankApprox(int m, int n, double eps, char *method)
{
	dtype *H = alloc_arr<dtype>(m * n);
	dtype *Hinit = alloc_arr<dtype>(m * n);
	dtype *Hres = alloc_arr<dtype>(m * n);
	int *ipiv = alloc_arr<int>(m);
	int *ipiv2 = alloc_arr<int>(m);
	char str[255];

	int ldh = m;

	dtype alpha_one = 1.0;
	dtype alpha_mone = -1.0;
	dtype beta_one = 1.0;
	dtype beta_zero = 0.0;
	int ione = 1;
	int mione = -1;
	double norm = 0;
	int info;

	Hilbert(m, n, Hinit, ldh);
	Hilbert(m, n, H, ldh);

	srand((unsigned int)time(0));
	for (int i = 0; i < m; i++)
	{
		ipiv[i] = random(1, m);
		//printf("ipiv[%d] = %d\n", i, ipiv[i]);
	}

	zlaswp(&n, Hinit, &ldh, &ione, &m, ipiv, &ione);

	cmnode *Hstr = (cmnode*)malloc(sizeof(cmnode));
	LowRankApproxStruct(m, n, H, ldh, Hstr, eps, method);
	//printf("p = %d\n", Hstr->p);
	zlaswp(&Hstr->p, Hstr->U, &m, &ione, &m, ipiv, &ione);
	zgemm("no", "no", &m, &n, &Hstr->p, &alpha_one, Hstr->U, &m, Hstr->VT, &Hstr->p, &beta_zero, Hres, &ldh);

	norm = rel_error_complex(m, n, Hres, Hinit, ldh, eps);
	sprintf(str, "Struct: m = %d n = %d ", m, n);
	AssertLess(norm, eps, str);

	//if (norm < eps) printf("Norm %10.8e < eps %10.8e: PASSED for size m = %d n = %d\n", norm, eps, m, n);
	//else printf("Norm %10.8lf > eps %10.8e : FAILED for size m = %d n = %d\n", norm, eps, m, n);
	
	free(Hstr->U);
	free(Hstr->VT);
	free(Hstr);
	free(H);
	free(Hinit);
	free(Hres);
}

void Test_UnsymmLUfact(int n, double eps, char* method, int smallsize)
{
#undef PRINT
//#define PRINT
#ifdef PRINT
	printf("----Test LU fact n = %d-----\n", n);
#endif
	dtype *H = alloc_arr<dtype>(n * n);
	dtype *Hinit = alloc_arr<dtype>(n * n);
	dtype *Hc = alloc_arr<dtype>(n * n);
	dtype *LUrec = alloc_arr<dtype>(n * n);
	dtype *U = alloc_arr<dtype>(n * n);
	dtype *L = alloc_arr<dtype>(n * n);
	int *ipiv = alloc_arr<int>(n);
	int *ipiv2 = alloc_arr<int>(n);
	char str[255];

	int ldh = n;
	int ldlu = n;
	int ldu = n;
	int ldl = n;

	dtype alpha_one = 1.0;
	dtype alpha_mone = -1.0;
	dtype beta_one = 1.0;
	dtype beta_zero = 0.0;
	int ione = 1;
	int mione = -1;
	double norm = 0;
	int info;
	int pivot = 0;

#ifdef PRINT
	printf("filling the matrix...\n");
#endif


//#define SYMM

#ifdef SYMM
	Hilbert(n, n, Hinit, ldh);
	Hilbert(n, n, Hc, ldh);

	for (int i = 0; i < n; i++)
	{
		Hinit[i + ldh * i] += 3.0;
		Hc[i + ldh * i] += 3.0;
	}
#else
	Hilbert3(n, n, Hinit, ldh);
	Hilbert3(n, n, Hc, ldh);

	// for stability
#if 1
	for (int i = 0; i < n; i++)
	{
		Hinit[i + ldh * i] -= 3.0;
		Hc[i + ldh * i] -= 3.0;
	}
#endif
#endif

	cumnode *HCstr;
#ifdef PRINT
	printf("Compress");
#endif
	double time = omp_get_wtime();
	UnsymmRecCompressStruct(n, Hc, ldh, HCstr, smallsize, eps, method);
//	UnsymmPrintRanksInWidthList(HCstr);
	time = omp_get_wtime() - time;
#ifdef PRINT
	printf(" time = %lf\nLU", time);
#endif
	time = omp_get_wtime();
	UnsymmLUfact(n, HCstr, ipiv, smallsize, eps, method);
//	UnsymmPrintRanksInWidthList(HCstr);
	time = omp_get_wtime() - time;
#ifdef PRINT
	printf(" time = %lf\nRestore", time);
#endif
	time = omp_get_wtime();
	UnsymmResRestoreStruct(n, HCstr, LUrec, ldlu, smallsize);
	time = omp_get_wtime() - time;

#ifdef PRINT
	printf(" time = %lf\nCopy factors", time);
#endif
	time = omp_get_wtime();
	zlacpy("L", &n, &n, LUrec, &ldlu, L, &ldl);
	for (int i = 0; i < n; i++)
		L[i + ldl * i] = 1.0;

	for (int i = 0; i < n; i++)
	{
		if (ipiv[i] != i + 1) pivot++;
	}

#ifdef PRINT
//	for (int i = 0; i < n; i++)
	//	if (ipiv[i] != i + 1) printf("GLOBAL IPIV: row %d with %d\n", i + 1, ipiv[i]);
#endif

	zlaswp(&n, L, &ldl, &ione, &n, ipiv, &mione);

	zlacpy("U", &n, &n, LUrec, &ldlu, U, &ldu);
	time = omp_get_wtime() - time;

#ifdef PRINT
	printf(" time = %lf\n", time);
	printf("LU MKL");
#endif
	time = omp_get_wtime();
	zgetrf(&n, &n, H, &ldh, ipiv2, &info);
	time = omp_get_wtime() - time;

#ifdef PRINT
	for (int i = 0; i < n; i++)
		if (ipiv2[i] != i + 1) printf("ROW interchange\n");
#endif

#ifdef PRINT
	printf(" time = %lf\nZGEMM", time);
#endif
	time = omp_get_wtime();
	zgemm("No", "No", &n, &n, &n, &alpha_one, L, &ldl, U, &ldu, &beta_zero, Hc, &ldh);
	time = omp_get_wtime() - time;

#ifdef PRINT
	printf(" time = %lf\n", time);
#endif

	norm = rel_error_complex(n, n, Hc, Hinit, ldh, eps);
	sprintf(str, "Struct: sz = %d, n = %d ", smallsize, n);
	AssertLess(norm, eps, str);

//#define PRINT
#ifdef PRINT
	if (norm < eps) printf("Norm %10.8e < eps %10.8e: PASSED for size n = %d and sz = %d pivot = %d\n", norm, eps, n, smallsize, pivot);
	else printf("Norm %10.8lf > eps %10.8e : FAILED for size n = %d and sz = %d pivot = %d\n", norm, eps, n, smallsize, pivot);
#endif

	FreeUnsymmNodes(n, HCstr, smallsize);
	free_arr(H);
	free_arr(Hc);
	free_arr(L);
	free_arr(U);
	free_arr(LUrec);

}
#ifdef CHOLESKY
void Test_SymLUfactLowRankStruct(int n, double eps, char* method, int smallsize)
{
#define PRINT
#undef PRINT
#ifdef PRINT
	printf("----Test LU fact n = %d-----\n", n);
#endif
	dtype *H = alloc_arr<dtype>(n * n);
	dtype *Hinit = alloc_arr<dtype>(n * n);
	dtype *Hc = alloc_arr<dtype>(n * n);
	dtype *LUrec = alloc_arr<dtype>(n * n);
	dtype *U = alloc_arr<dtype>(n * n);
	dtype *L = alloc_arr<dtype>(n * n);
	int *ipiv = alloc_arr<int>(n);
	int *ipiv2 = alloc_arr<int>(n);
	char str[255];

	int ldh = n;
	int ldlu = n;
	int ldu = n;
	int ldl = n;

	dtype alpha_one = 1.0;
	dtype alpha_mone = -1.0;
	dtype beta_one = 1.0;
	dtype beta_zero = 0.0;
	int ione = 1;
	int mione = -1;
	double norm = 0;
	int info;
	int pivot = 0;

#ifdef PRINT
	printf("filling the matrix...\n");
#endif

#if 1
	Hilbert6(n, n, Hinit, ldh);
	Hilbert6(n, n, Hc, ldh);
	Hilbert6(n, n, H, ldh);

	//Hilbert7LowRank(n, n, Hinit, ldh);
    //Hilbert7LowRank(n, n, Hc, ldh);
	//Hilbert7LowRank(n, n, H, ldh);
#else
	Hilbert8Unique(n, n, Hinit, ldh);
	Hilbert8Unique(n, n, Hc, ldh);
	Hilbert8Unique(n, n, H, ldh);
#endif

#if 0
	for (int i = 0; i < n - 2; i++)
	{
		Hinit[i + ldh * i] += 3.0;
		Hc[i + ldh * i] += 3.0;
		H[i + ldh * i] += 3.0;
	}
#endif

#ifdef PRINT
	printf("LowRank Compress");
#endif

	cmnode *HCstr = (cmnode*)malloc(sizeof(cmnode));
	double time = omp_get_wtime();
	LowRankApproxStruct(n, n, Hc, ldh, HCstr, eps, "SVD");
	time = omp_get_wtime() - time;
#ifdef PRINT
	printf(" time = %lf\nRank = %d\nLowRankToUnsymmHSS", time, HCstr->p);
#endif
	cumnode *HSSstr;
	time = omp_get_wtime();
	LowRankToUnsymmHSS(n, HCstr->p, HCstr->U, n, HCstr->VT, HCstr->p, HSSstr, smallsize);
	time = omp_get_wtime() - time;
#ifdef PRINT
	printf(" time = %lf\nSymLUfactLowRankStruct", time);
#endif
	time = omp_get_wtime();
	SymLUfactLowRankStruct(n, HSSstr, ipiv, smallsize, eps, method);
	time = omp_get_wtime() - time;
#ifdef PRINT
	printf(" time = %lf\nRestore", time);
#endif
	time = omp_get_wtime();
	UnsymmResRestoreStruct(n, HSSstr, LUrec, ldlu, smallsize);
	time = omp_get_wtime() - time;

#ifdef PRINT
	printf(" time = %lf\nCopy factors", time);
#endif
	time = omp_get_wtime();
	zlacpy("L", &n, &n, LUrec, &ldlu, L, &ldl);
	for (int i = 0; i < n; i++)
		L[i + ldl * i] = 1.0;

	for (int i = 0; i < n; i++)
	{
		if (ipiv[i] != i + 1) pivot++;
	}

#ifdef PRINT
	//	for (int i = 0; i < n; i++)
		//	if (ipiv[i] != i + 1) printf("GLOBAL IPIV: row %d with %d\n", i + 1, ipiv[i]);
#endif

	zlaswp(&n, L, &ldl, &ione, &n, ipiv, &mione);

	zlacpy("U", &n, &n, LUrec, &ldlu, U, &ldu);
	time = omp_get_wtime() - time;

#ifdef PRINT
	printf(" time = %lf\n", time);
	printf("LU MKL");
#endif
	time = omp_get_wtime();
	zgetrf(&n, &n, H, &ldh, ipiv2, &info);
	time = omp_get_wtime() - time;

#ifdef PRINT
	for (int i = 0; i < n; i++)
		if (ipiv2[i] != i + 1) printf("ROW interchange\n");
#endif

#ifdef PRINT
	printf(" time = %lf\nZGEMM", time);
#endif
	time = omp_get_wtime();
	zgemm("No", "No", &n, &n, &n, &alpha_one, L, &ldl, U, &ldu, &beta_zero, Hc, &ldh);
	time = omp_get_wtime() - time;

#ifdef PRINT
	printf(" time = %lf\n", time);
#endif

	norm = rel_error_complex(n, n, Hc, Hinit, ldh, eps);
	sprintf(str, "Struct: sz = %d, n = %d ", smallsize, n);

#ifndef PRINT
	AssertLess(norm, eps, str);
#endif

#//define PRINT
#ifdef PRINT
	if (norm < eps) printf("Norm %10.8e < eps %10.8e: PASSED for size n = %d and sz = %d rank = %d pivot = %d info = %d\n", norm, eps, n, smallsize, HCstr->p, pivot, info);
	else printf("Norm %10.8lf > eps %10.8e : FAILED for size n = %d and sz = %d rank = %d pivot = %d info = %d\n", norm, eps, n, smallsize, HCstr->p, pivot, info);
#endif

	FreeUnsymmNodes(n, HSSstr, smallsize);
	free_arr(HCstr->U);
	free_arr(HCstr->VT);
	free_arr(HCstr);
	free_arr(H);
	free_arr(Hc);
	free_arr(Hinit);
	free_arr(L);
	free_arr(U);
	free_arr(LUrec);

}


void Test_LowRankCholeskyStruct(int n, int p, double eps, char* method, int smallsize)
{
#define PRINT
#undef PRINT
//#define DEBUG
#ifdef PRINT
	printf("----Test LowRankCholesky n = %d-----\n", n);
#endif
	dtype *H = alloc_arr<dtype>(n * n);
	dtype *Hinit = alloc_arr<dtype>(n * n);
	dtype *Hc = alloc_arr<dtype>(n * n);
	dtype *LLrec = alloc_arr<dtype>(n * n);
	dtype *BlockedLL = alloc_arr<dtype>(n * n);
	dtype *L = alloc_arr<dtype>(n * n);
	dtype *W = alloc_arr<dtype>(n * p); int ldw = n;
	dtype *Diag = alloc_arr<dtype>(n);
	char str[255];

	int ldh = n;
	int ldlu = n;
	int ldu = n;
	int ldl = n;

	dtype alpha_one = 1.0;
	dtype alpha_mone = -1.0;
	dtype beta_one = 1.0;
	dtype beta_zero = 0.0;
	int ione = 1;
	int mione = -1;
	double norm = 0;
	int info;
	double time;

#ifdef PRINT
	printf("filling the matrix...\n");
#endif

	Hilbert5(n, p, W, ldh);

	// low part dense matrix
	zsyrk("Low", "No", &n, &p, &alpha_one, W, &ldw, &beta_zero, Hinit, &ldh);
	zlacpy("Low", &n, &n, Hinit, &ldh, Hc, &ldh);
	zlacpy("Low", &n, &n, Hinit, &ldh, H, &ldh);
	zlacpy("Low", &n, &n, Hinit, &ldh, BlockedLL, &ldh);

#if 1
#pragma omp parallel for simd schedule(static)
	for (int i = 0; i < n; i++)
	{
		Diag[i] = 10.0;
		Hinit[i + ldh * i] += Diag[i];
		H[i + ldh * i] += Diag[i];
		BlockedLL[i + ldh * i] += Diag[i];
	}
#endif

#ifdef DEBUG
	printf("matrix init:\n");
	PrintMat(n, n, H, ldh);
#endif

#ifdef PRINT
	printf("LowRank Compress");
#endif

#ifdef PRINT
	//printf(" time = %lf\nRank = %d\nLowRankToUnsymmHSS", time, p);
#endif
	cmnode *HSSstr;
	time = omp_get_wtime();
	LowRankToSymmHSS(n, p, W, ldw, HSSstr, smallsize);
	time = omp_get_wtime() - time;
#ifdef PRINT
	printf(" time = %lf\nAddSymmHSSDiag", time);
#endif
	// D + W * W^T
	time = omp_get_wtime();
	AddSymmHSSDiag(n, HSSstr, Diag, smallsize);
	time = omp_get_wtime() - time;

#ifdef PRINT
	printf(" time = %lf\nSymLUfactLowRankStruct", time);
#endif

	dtype *work = alloc_arr2<dtype>(HSSstr->p * HSSstr->p);
	time = omp_get_wtime();
	LowRankCholeskyFact(n, HSSstr, work, smallsize, eps, method);
	time = omp_get_wtime() - time;
#ifdef PRINT
	printf(" time = %lf\nRestore", time);
#endif
	time = omp_get_wtime();
	SymResRestoreStruct(n, HSSstr, LLrec, ldlu, smallsize);
	time = omp_get_wtime() - time;

#ifdef PRINT
	printf(" time = %lf\nCopy factors", time);
#endif

#if 1
	time = omp_get_wtime();
	zlacpy("All", &n, &n, LLrec, &ldlu, L, &ldl);
	time = omp_get_wtime() - time;
#endif

#ifdef PRINT
	printf(" time = %lf\n", time);
	printf("Cholesky MKL");
#endif
	time = omp_get_wtime();
	zpotrf("L", &n, H, &ldh, &info);
	time = omp_get_wtime() - time;
	if (info != 0) printf("!!! global zpotrf error n = %d\n", n);

#ifdef PRINT
	printf(" time = %lf\n", time);
#endif

#if 0
	printf("ZSYRK", time);
	time = omp_get_wtime();
	zsyrk("L", "N", &n, &n, &alpha_one, L, &ldl, &beta_zero, Hc, &ldh);
	time = omp_get_wtime() - time;

#ifdef PRINT
	printf(" time = %lf\n", time);
#endif
#endif

	//CholeskyFact(n, BlockedLL, n, smallsize, eps, method);

#ifdef DEBUG
	printf("full dense cholesky:\n");
	PrintMat(n, n, H, ldh);
#endif

	//PrintMat(n, n, BlockedLL, ldh);
	
	//norm = RelErrorPart(zlange, 'L', n, n, BlockedLL, ldh, H, ldh, eps);
	norm = RelErrorPart(zlange, 'L', n, n, LLrec, ldh, H, ldh, eps);
	//norm = RelErrorPart(zlange, 'L', n, n, LLrec, ldh, Hinit, ldh, eps);
	sprintf(str, "Struct: sz = %d, n = %d, p = %d", smallsize, n, p);

#ifndef PRINT
	AssertLess(norm, eps, str);
#endif

//#define PRINT
#ifdef PRINT
	if (norm < eps) printf("Norm %10.8e < eps %10.8e: PASSED for size n = %d and sz = %d rank = %d info = %d\n", norm, eps, n, smallsize, p, info);
	else printf("Norm %10.8lf > eps %10.8e : FAILED for size n = %d and sz = %d rank = %d info = %d\n", norm, eps, n, smallsize, p, info);
#endif

	FreeNodes(n, HSSstr, smallsize);
	free_arr(H);
	free_arr(Hc);
	free_arr(Hinit);
	free_arr(L);
	free_arr(W);
	free_arr(LLrec);
}
# if 0
void Test_SymLUfact(int n, double eps, char* method, int smallsize)
{
	//#define PRINT
#ifdef PRINT
	printf("----Test LU fact n = %d-----\n", n);
#endif
	dtype *H = alloc_arr<dtype>(n * n);
	dtype *Hinit = alloc_arr<dtype>(n * n);
	dtype *Hc = alloc_arr<dtype>(n * n);
	dtype *LUrec = alloc_arr<dtype>(n * n);
	dtype *U = alloc_arr<dtype>(n * n);
	dtype *L = alloc_arr<dtype>(n * n);
	int *ipiv = alloc_arr<int>(n);
	int *ipiv2 = alloc_arr<int>(n);
	char str[255];

	int ldh = n;
	int ldlu = n;
	int ldu = n;
	int ldl = n;

	dtype alpha_one = 1.0;
	dtype alpha_mone = -1.0;
	dtype beta_one = 1.0;
	dtype beta_zero = 0.0;
	int ione = 1;
	int mione = -1;
	double norm = 0;
	int info;
	int pivot = 0;

#ifdef PRINT
	printf("filling the matrix...\n");
#endif

	Hilbert(n, n, Hinit, ldh);
	Hilbert(n, n, Hc, ldh);

	for (int i = 0; i < n; i++)
	{
		Hinit[i + ldh * i] += 3.0;
		Hc[i + ldh * i] += 3.0;
	}


	cmnode *HCstr;
#ifdef PRINT
	printf("Compress");
#endif
	double time = omp_get_wtime();
	SymRecCompressStruct(n, Hc, ldh, HCstr, smallsize, eps, method);
	//	UnsymmPrintRanksInWidthList(HCstr);
	time = omp_get_wtime() - time;
#ifdef PRINT
	printf(" time = %lf\nLU", time);
#endif
	time = omp_get_wtime();
	SymLUfact(n, HCstr, ipiv, smallsize, eps, method);
	//	UnsymmPrintRanksInWidthList(HCstr);
	time = omp_get_wtime() - time;
#ifdef PRINT
	printf(" time = %lf\nRestore", time);
#endif
	time = omp_get_wtime();
	SymResRestoreStruct(n, HCstr, LUrec, ldlu, smallsize);
	time = omp_get_wtime() - time;

#ifdef PRINT
	printf(" time = %lf\nCopy factors", time);
#endif
	time = omp_get_wtime();
	zlacpy("L", &n, &n, LUrec, &ldlu, L, &ldl);
	for (int i = 0; i < n; i++)
		L[i + ldl * i] = 1.0;

	for (int i = 0; i < n; i++)
	{
		if (ipiv[i] != i + 1) pivot++;
	}

#ifdef PRINT
	//	for (int i = 0; i < n; i++)
		//	if (ipiv[i] != i + 1) printf("GLOBAL IPIV: row %d with %d\n", i + 1, ipiv[i]);
#endif

	zlaswp(&n, L, &ldl, &ione, &n, ipiv, &mione);

	zlacpy("U", &n, &n, LUrec, &ldlu, U, &ldu);
	time = omp_get_wtime() - time;

#ifdef PRINT
	printf(" time = %lf\n", time);
	printf("LU MKL");
#endif
	time = omp_get_wtime();
	zgetrf(&n, &n, H, &ldh, ipiv2, &info);
	time = omp_get_wtime() - time;

#ifdef PRINT
	for (int i = 0; i < n; i++)
		if (ipiv2[i] != i + 1) printf("ROW interchange\n");
#endif

#ifdef PRINT
	printf(" time = %lf\nZGEMM", time);
#endif
	time = omp_get_wtime();
	zgemm("No", "No", &n, &n, &n, &alpha_one, L, &ldl, U, &ldu, &beta_zero, Hc, &ldh);
	time = omp_get_wtime() - time;

#ifdef PRINT
	printf(" time = %lf\n", time);
#endif

	norm = rel_error_complex(n, n, Hc, Hinit, ldh, eps);
	sprintf(str, "Struct: sz = %d, n = %d ", smallsize, n);
	AssertLess(norm, eps, str);

#ifdef PRINT
	if (norm < eps) printf("Norm %10.8e < eps %10.8e: PASSED for size n = %d and sz = %d pivot = %d\n", norm, eps, n, smallsize, pivot);
	else printf("Norm %10.8lf > eps %10.8e : FAILED for size n = %d and sz = %d pivot = %d\n", norm, eps, n, smallsize, pivot);
#endif

	FreeNodes(n, HCstr, smallsize);
	free_arr(H);
	free_arr(Hc);
	free_arr(L);
	free_arr(U);
	free_arr(LUrec);

}
#endif
#endif
void TestRowInterchange(int n, int m, double eps)
{
	dtype *A = alloc_arr<dtype>(n * n); int lda = n;
	dtype *P = alloc_arr<dtype>(n * n); int ldp = n;
	int *ipiv = alloc_arr<int>(n);
	int ione = 1;
	int mione = -1;

	Hilbert3(n, m, A, lda);

	print(n, m, A, lda, "A");
	
	ipiv[0] = 2;
	ipiv[1] = 4;
	ipiv[2] = 3;
	ipiv[3] = 4;
	ipiv[4] = 5;

	zlaswp(&m, A, &lda, &ione, &n, ipiv, &ione);

	print(n, m, A, lda, "Aperm");

	zlaswp(&m, A, &lda, &ione, &n, ipiv, &ione);

	print(n, m, A, lda, "AbackReverse");
}

void Test_DirFactFastDiagStructOnlineHODLR(size_m x, size_m y, cmnode** Gstr, dtype *B, dtype kwave2, double eps, int smallsize)
{
	printf("Testing factorization...\n");
	int n = x.n;
	int size = n * y.n;
	int nbr = y.n;
	char bench[255] = "No";
	dtype *DD = alloc_arr<dtype>(n * n); int lddd = n;
	dtype *DR = alloc_arr<dtype>(n * n); int lddr = n;
	dtype *alpX = alloc_arr<dtype>(n + 2);
	dtype *alpY = alloc_arr<dtype>(n + 2);
	double norm = 0;

	double timer = 0;
	timer = omp_get_wtime();

	GenerateDiagonal1DBlockHODLR(0, x, y, DD, lddd, kwave2);

	cmnode *DCstr;
	SymCompRecInvStruct(n, Gstr[0], DCstr, smallsize, eps, "SVD");
	SymResRestoreStruct(n, DCstr, DR, lddr, smallsize);

//	print(n, n, DR, lddd, "DR");
//	print(n, n, DD, lddd, "DD");

	norm = rel_error_complex(n, n, DR, DD, lddd, eps);
//	print(n, n, DR, lddd, "DR_DIFF");
/*	for (int i = 0; i < n; i++)
	{
		printf("%d ", i);
		for (int j = 0; j < n; j++)
		{
			printf("%14.12lf %14.12lf\n", DR[i + lddd*j].real(), DR[i + lddd*j].imag());
		}
		printf("\n");
	}*/

	//printf("%s\n", mess);

	if (norm > eps) printf("Block %d. Norm %12.10e > eps %12.10lf : FAILED\n", 0, norm, eps);

	free_arr(DR);
	free_arr(DD);
	FreeNodes(n, DCstr, smallsize);

	for (int k = 1; k < nbr; k++)
	{
		dtype *DR = alloc_arr<dtype>(n * n); int lddr = n;
		dtype *HR = alloc_arr<dtype>(n * n); int ldhr = n;
		dtype *DD = alloc_arr<dtype>(n * n); int lddd = n;
		cmnode *DCstr, *Hstr;

		SymCompRecInvStruct(n, Gstr[k], DCstr, smallsize, eps, "SVD");
		SymResRestoreStruct(n, DCstr, DR, lddr, smallsize);

		CopyStruct(n, Gstr[k - 1], Hstr, smallsize);
		DiagMultStruct(n, Hstr, &B[ind(k - 1, n)], smallsize);
		SymResRestoreStruct(n, Hstr, HR, ldhr, smallsize);

#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
#pragma omp simd
			for (int i = 0; i < n; i++)
				HR[i + ldhr * j] = HR[i + ldhr * j] + DR[i + lddr * j];

		GenerateDiagonal1DBlockHODLR(k, x, y, DD, lddd, kwave2);


	//	print(n, n, HR, lddd, "DR");
	//	print(n, n, DD, lddd, "DD");

		norm = rel_error_complex(n / 2, n / 2, &HR[n / 2 + ldhr * n / 2], &DD[n / 2 + lddd * n / 2], lddd, eps);

		if (norm > eps) printf("Block %d. Norm %12.10e > eps %12.10lf : FAILED\n", k, norm, eps);
		else printf("Block %d. Norm %12.10e > eps %12.10lf : PASSED\n", k, norm, eps);

	//	system("pause");

		FreeNodes(n, DCstr, smallsize);
		FreeNodes(n, Hstr, smallsize);
		free_arr(DR);
		free_arr(HR);
		free_arr(DD);
	}
	timer = omp_get_wtime() - timer;
	printf("Time: %lf\n", timer);

}

void Test_DirFactFastDiagStructOnline(size_m x, size_m y, cmnode** Gstr, dtype *B, double eps, int smallsize)
{
	printf("Testing factorization...\n");
	int n = x.n;
	int size = n * y.n;
	int nbr = y.n;
	char bench[255] = "No";
	dtype *DD = alloc_arr<dtype>(n * n); int lddd = n;
	dtype *DR = alloc_arr<dtype>(n * n); int lddr = n;
	dtype *alpX = alloc_arr<dtype>(n + 2);
	dtype *alpY = alloc_arr<dtype>(n + 2);
	double norm = 0;

	double timer = 0;
	timer = omp_get_wtime();

	SetPml(0, x, y, n, alpX, alpY);
	GenerateDiagonal1DBlock(0, x, y, DD, lddd, alpX, alpY);

	cmnode *DCstr;
	SymCompRecInvStruct(n, Gstr[0], DCstr, smallsize, eps, "SVD");
	SymResRestoreStruct(n, DCstr, DR, lddr, smallsize);

	//	print(n, n, DR, lddd, "DR");
	//	print(n, n, DD, lddd, "DD");

	norm = rel_error_complex(n, n, DR, DD, lddd, eps);
	//	print(n, n, DR, lddd, "DR_DIFF");
	/*	for (int i = 0; i < n; i++)
		{
			printf("%d ", i);
			for (int j = 0; j < n; j++)
			{
				printf("%14.12lf %14.12lf\n", DR[i + lddd*j].real(), DR[i + lddd*j].imag());
			}
			printf("\n");
		}*/

		//printf("%s\n", mess);

	if (norm > eps) printf("Block %d. Norm %12.10e > eps %12.10lf : FAILED\n", 0, norm, eps);

	free_arr(DR);
	free_arr(DD);
	FreeNodes(n, DCstr, smallsize);

	for (int k = 1; k < nbr; k++)
	{
		dtype *DR = alloc_arr<dtype>(n * n); int lddr = n;
		dtype *HR = alloc_arr<dtype>(n * n); int ldhr = n;
		dtype *DD = alloc_arr<dtype>(n * n); int lddd = n;
		cmnode *DCstr, *Hstr;

		SymCompRecInvStruct(n, Gstr[k], DCstr, smallsize, eps, "SVD");
		SymResRestoreStruct(n, DCstr, DR, lddr, smallsize);

		CopyStruct(n, Gstr[k - 1], Hstr, smallsize);
		DiagMultStruct(n, Hstr, &B[ind(k - 1, n)], smallsize);
		SymResRestoreStruct(n, Hstr, HR, ldhr, smallsize);

#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
#pragma omp simd
			for (int i = 0; i < n; i++)
				HR[i + ldhr * j] = HR[i + ldhr * j] + DR[i + lddr * j];

		SetPml(k, x, y, n, alpX, alpY);
		GenerateDiagonal1DBlock(k, x, y, DD, lddd, alpX, alpY);


		//	print(n, n, HR, lddd, "DR");
		//	print(n, n, DD, lddd, "DD");

		norm = rel_error_complex(n / 2, n / 2, &HR[n / 2 + ldhr * n / 2], &DD[n / 2 + lddd * n / 2], lddd, eps);

		if (norm > eps) printf("Block %d. Norm %12.10e > eps %12.10lf : FAILED\n", k, norm, eps);
		else printf("Block %d. Norm %12.10e > eps %12.10lf : PASSED\n", k, norm, eps);

		//	system("pause");

		FreeNodes(n, DCstr, smallsize);
		FreeNodes(n, Hstr, smallsize);
		free_arr(DR);
		free_arr(HR);
		free_arr(DD);
	}
	timer = omp_get_wtime() - timer;
	printf("Time: %lf\n", timer);

}

void Test_TransferBlock3Diag_to_CSR(int n1, int n2, zcsr* Dcsr, dtype* x_orig, dtype *f, double eps)
{
	int n = n1;
	int size = n * n2;
	double RelRes = 0;
	dtype *g = alloc_arr2<dtype>(size);
	ResidCSR(n1, n2, Dcsr, x_orig, f, g, RelRes);

	if (RelRes < eps) printf("Norm %10.8e < eps %10.8lf: PASSED\n", RelRes, eps);
	else printf("Norm %10.8lf > eps %10.8e : FAILED\n", RelRes, eps);

	free_arr(g);
}


void Test_DirSolveFactDiagStructBlockRanks(size_m x, size_m y, cmnode** Gstr)
{
	printf("----------Trees information-----------\n");
	int *size = alloc_arr<int>(y.n);
	int *depth = alloc_arr<int>(y.n);

	double time = omp_get_wtime();
#pragma omp parallel
	{
#pragma omp single
		for (int i = 0; i < y.n; i++)
		{
			size[i] = TreeSize(Gstr[i]);
			depth[i] = MaxDepth(Gstr[i]);
		}
	}
	double result = omp_get_wtime() - time;
	printf("Computational time of TreeSize and MaxDepth for all %d trees: %lf\n", y.n, result);

	for (int i = 0; i < y.n; i++)
	{
		printf("For block %2d. Size: %d, MaxDepth: %d, Ranks: ", i, size[i], depth[i]);
		PrintRanksInWidthList(Gstr[i]);
		printf("\n");
	}

	free(size);
	free(depth);

}

void Test_NonZeroElementsInFactors(size_m x, size_m y, cmnode **Gstr, dtype* B, double thresh, int smallsize)
{
	long long non_zeros_exact = 0;
	long long non_zeros_HSS = 0;

	//non_zeros_exact = (x.n * x.n) * y.n + 2 * x.n * (y.n - 1);
	non_zeros_exact = (x.n * x.n) * y.n;

	for(int k = 0; k < y.n; k++)
		non_zeros_HSS += CountElementsInMatrixTree(x.n, Gstr[k]);

	//non_zeros_HSS += 2 * (y.n - 1) * x.n;

	int compr_size = ceil((double)x.n / smallsize);
	int compr_level = ceil(log(compr_size) / log(2));
	int loc_size = x.n;
	long long zeros_ideal = 0;

	printf("Compression level: %d, Compression size: %d\n", compr_level, compr_size);
	for (int j = 0; j < compr_level; j++)
	{
		loc_size = ceil(loc_size / 2.0);
		compr_size = ceil((double)x.n / loc_size);
		zeros_ideal += (loc_size * loc_size * compr_size) * y.n;
		printf("loc_size: %d, compr_size: %d, zeros_ideal: %d\n", loc_size, compr_size, zeros_ideal);
	}

	printf("Compression level: %d, Compression size: %d\n", compr_level, compr_size);
	printf("non_zeros_exact: %ld\nnon_zeros_HSS: %ld\n", non_zeros_exact, non_zeros_HSS);
	printf("coefficient of compression: %lf (ideal: %lf)\n",  (double)non_zeros_exact / non_zeros_HSS, (double)non_zeros_exact/(non_zeros_exact - zeros_ideal));
	
}

void Test_UnsymmLUfact2(int n, double eps, char* method, int smallsize)
{
	printf("----Test LU fact-----\n");
	dtype *H = alloc_arr<dtype>(n * n);
	dtype *Hinit = alloc_arr<dtype>(n * n);
	dtype *Hc = alloc_arr<dtype>(n * n);
	dtype *LUrec = alloc_arr<dtype>(n * n);
	dtype *U = alloc_arr<dtype>(n * n);
	dtype *L = alloc_arr<dtype>(n * n);
	int *ipiv = alloc_arr<int>(n);
	int *ipiv2 = alloc_arr<int>(n);
	char str[255];

	int ldh = n;
	int ldlu = n;
	int ldu = n;
	int ldl = n;

	int nbl = n / 2;

	dtype alpha_one = 1.0;
	dtype alpha_mone = -1.0;
	dtype beta_one = 1.0;
	dtype beta_zero = 0.0;
	int ione = 1;
	int mione = -1;
	double norm = 0;
	int info;

	Hilbert3(n, n, Hinit, ldh);
	Hilbert3(n, n, Hc, ldh);

	// for stability
	for (int i = 0; i < n; i++)
	{
		Hinit[i + ldh * i] -= 5.0;
		Hc[i + ldh * i] -= 5.0;
	}

	print(n, n, Hinit, ldh, "A");

	cumnode *HCstr;
	printf("Compress\n");
	UnsymmRecCompressStruct(n, Hc, ldh, HCstr, smallsize, eps, method);
	//UnsymmPrintRanksInWidthList(HCstr);
	printf("LU\n");
	UnsymmLUfact(n, HCstr, ipiv, smallsize, eps, method);
	printf("Restore\n");
	UnsymmResRestoreStruct(n, HCstr, LUrec, ldlu, smallsize);

	print(n, n, LUrec, ldlu, "LU HSS after update A22");
		
	//MyLU(n, Hinit, ldh, ipiv2);
	MyLURec(n, Hinit, ldh, ipiv2, smallsize);

	print(n, n, Hinit, ldh, "MY LU after update A22");

//	zgetrf(&n, &n, H, &ldh, ipiv2, &info);
//	for (int i = 0; i < n; i++)
//		if (ipiv2[i] != i + 1) printf("MKL LU: ROW interchange\n");

//	print(n, n, H, ldh, "L + U MKL");
//	printf("\n");

//	printf("Gemm A:= A - LU\n");
	// A = A - Lrec * Urec
//	zgemm("No", "No", &n, &n, &n, &alpha_mone, L, &ldl, U, &ldu, &beta_one, Hinit, &ldh);
//	zgemm("No", "No", &n, &n, &n, &alpha_one, L, &ldl, U, &ldu, &beta_zero, Hc, &ldh);

//	print(n, n, Hc, ldlu, "L*U");
//	printf("\n");

	//for (int i = 0; i < n; i++)
		//for (int j = 0; j < n; j++)
			//Hinit[i + ldh * j] -= LUrec[i + ldlu * j];

	int n1 = 10;
	int n2 = 5;

	norm = rel_error_complex(n, n, LUrec, Hinit, ldh, eps);
	print(n, n, LUrec, ldlu, "A - LU");

	sprintf(str, "Struct: n = %d", n);
	//AssertLess(norm, eps, str);

	if (norm < eps) printf("Norm %10.8e < eps %10.8e: PASSED\n", norm, eps);
	else printf("Norm %10.8lf > eps %10.8e : FAILED\n", norm, eps);

	norm = zlange("Frob", &n, &n, &Hinit[0 + ldh * 0], &ldh, NULL);

	FreeUnsymmNodes(n, HCstr, smallsize);
	free_arr(H);
	free_arr(Hc);
	free_arr(L);
	free_arr(U);
	free_arr(LUrec);

}

void Test_MyLURecFact(int n, double eps, char* method, int smallsize)
{
//	printf("----Test LU fact-----\n");
	dtype *H = alloc_arr<dtype>(n * n);
	dtype *Hinit = alloc_arr<dtype>(n * n);
	dtype *Hc = alloc_arr<dtype>(n * n);
	dtype *LUrec = alloc_arr<dtype>(n * n);
	dtype *U = alloc_arr<dtype>(n * n);
	dtype *L = alloc_arr<dtype>(n * n);
	int *ipiv = alloc_arr<int>(n);
	int *ipiv2 = alloc_arr<int>(n);
	char str[255];

	int ldh = n;
	int ldlu = n;
	int ldu = n;
	int ldl = n;

	int nbl = n / 2;

	dtype alpha_one = 1.0;
	dtype alpha_mone = -1.0;
	dtype beta_one = 1.0;
	dtype beta_zero = 0.0;
	int ione = 1;
	int mione = -1;
	double norm = 0;
	int info;
	int pivot = 0;

	Hilbert3(n, n, Hinit, ldh);
	Hilbert3(n, n, H, ldh);

	// for stability
	for (int i = 0; i < n; i++)
	{
		Hinit[i + ldh * i] -= 3.0;
		H[i + ldh * i] -= 3.0;
	}

//	print(n, n, Hinit, ldh, "A");

	MyLURec(n, Hinit, ldh, ipiv, smallsize);

	for (int i = 0; i < n; i++)
	{
		if (ipiv[i] != i + 1) pivot++;
	}

#ifdef PRINT
	print(n, n, Hinit, ldh, "MY LU");
#endif

	zlacpy("L", &n, &n, Hinit, &ldh, L, &ldl);

	for (int i = 0; i < n; i++)
		L[i + ldl * i] = 1.0;

	zlaswp(&n, L, &ldl, &ione, &n, ipiv, &mione);

	zlacpy("U", &n, &n, Hinit, &ldh, U, &ldu);

	zgemm("No", "No", &n, &n, &n, &alpha_one, L, &ldl, U, &ldu, &beta_zero, Hc, &ldh);

	int n1 = 10;
	int n2 = 5;

	norm = rel_error_complex(n, n, Hc, H, ldh, eps);
	//print(n, n, LUrec, ldlu, "A - LU");

	sprintf(str, "Struct: n = %d", n);
	AssertLess(norm, eps, str);

//	if (norm < eps) printf("Norm %10.8e < eps %10.8e: PASSED for n = %d sz = %d pivot = %d\n", norm, eps, n, smallsize, pivot);
//	else printf("Norm %10.8lf > eps %10.8e : FAILED for n = %d sz = %d pivot = %d\n", norm, eps, n, smallsize, pivot);

	norm = zlange("Frob", &n, &n, &Hinit[0 + ldh * 0], &ldh, NULL);

	free_arr(H);
	free_arr(Hc);
	free_arr(L);
	free_arr(U);
	free_arr(LUrec);

}


void Test_LowRankToUnsymmHSS(int n, double eps, char* method, int smallsize)
{
	dtype *H = alloc_arr<dtype>(n * n);
	dtype *Hinit = alloc_arr<dtype>(n * n);
	dtype *Hrec = alloc_arr<dtype>(n * n);

	char str[255];

	int ldh = n;
	double norm = 0;

	Hilbert3(n, n, H, ldh);
	Hilbert3(n, n, Hinit, ldh);

	cmnode *HCstr;
	HCstr = (cmnode*)malloc(sizeof(cmnode));
	LowRankApproxStruct(n, n, H, ldh, HCstr, eps, "SVD");

	cumnode *HChss;
	LowRankToUnsymmHSS(n, HCstr->p, HCstr->U, n, HCstr->VT, HCstr->p, HChss, smallsize);

	UnsymmResRestoreStruct(n, HChss, Hrec, ldh, smallsize);

	norm = rel_error_complex(n, n, Hrec, Hinit, ldh, eps);
	sprintf(str, "Struct: sz = %d, n = %d ", smallsize, n);
	AssertLess(norm, eps, str);


	FreeUnsymmNodes(n, HChss, smallsize);
	free(HCstr->U);
	free(HCstr->VT);
	free(HCstr);
	free(Hinit);
	free(H);
	free(Hrec);
}




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


