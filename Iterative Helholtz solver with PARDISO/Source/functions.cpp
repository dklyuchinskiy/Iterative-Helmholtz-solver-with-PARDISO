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
			printf("%5.3lf ", u[i + ldu*j].imag());
		}
		printf("\n");
	}

	printf("\n");
}

void Eye(int n, dtype *H, int ldh)
{
	for (int j = 0; j < n; j++)
//#pragma omp simd
		for (int i = 0; i < n; i++)
			if (j == i) H[i + ldh * j] = 1.0;
			else H[i + ldh * j] = 0.0;
}


void Hilbert(int n, dtype *H, int ldh)
{
#pragma omp parallel for schedule(static)
	for (int j = 0; j < n; j++)
//#pragma omp simd
		for (int i = 0; i < n; i++)
			H[i + ldh * j] = 1.0 / (i + j + 1);
}

void Mat_Trans(int m, int n, dtype *H, int ldh, dtype *Hcomp_tr, int ldhtr)
{
#pragma omp parallel for schedule(static)
	for (int i = 0; i < m; i++)
//#pragma omp simd
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
//#pragma omp simd
			for (int i = 0; i < m; i++)
				C[i + ldc * j] = alpha * A[i + lda * j];
	}
	else if (alpha == dzero)
	{
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
//#pragma omp simd
			for (int i = 0; i < m; i++)
				C[i + ldc * j] = beta * B[i + ldb * j];
	}
	else
	{
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
//#pragma omp simd
			for (int i = 0; i < m; i++)
				C[i + ldc * j] = alpha * A[i + lda * j] + beta * B[i + ldb * j];
	}
}

void Clear(int m, int n, dtype* A, int lda)
{
#pragma omp parallel for schedule(runtime)
	for (int j = 0; j < n; j++)
//#pragma omp simd
		for (int i = 0; i < m; i++)
			A[i + lda * j] = 0.0;
}

void GenerateDiagonal1DBlock(double w, int part_of_field, size_m x, size_m y, size_m z, dtype *DD, int lddd,
	   dtype* alpX, dtype* alpY, dtype* alpZ)
{
	int n = x.n;
	double k = (double)kk;

	Clear(n, n, DD, lddd);
	
	// diagonal blocks in dense format
#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++)
	{
		 DD[i + lddd * i] = -2.0 * (1.0 / (y.h * y.h) + 1.0 / (z.h * z.h)) - dtype{ w , 0 };
		//DD[i + lddd * i] = -alpX[i + 1] * (alpX[i + 2] + 2.0 * alpX[i + 1] + alpX[i]) / (2.0 * x.h * x.h)
		//    			   -alpY[i + 1] * (alpY[i + 2] + 2.0 * alpY[i + 1] + alpY[i]) / (2.0 * y.h * y.h)
		//				   -alpZ[i + 1] * (alpZ[i + 2] + 2.0 * alpZ[i + 1] + alpZ[i]) / (2.0 * z.h * z.h)
		//-dtype{ w, 0 };
#ifdef HELMHOLTZ
		DD[i + lddd * i] += dtype{ k * k, 0 };
#endif
		if (i > 0) DD[i + lddd * (i - 1)] = 1.0 / (y.h * y.h);
		if (i < n - 1) DD[i + lddd * (i + 1)] = 1.0 / (y.h * y.h);
		//if (i > 0) DD[i + lddd * (i - 1)] = alpX[i + 1] * (alpX[i + 1] + alpX[i]) / (2.0 * x.h * x.h);
		//if (i < n - 1) DD[i + lddd * (i + 1)] = alpX[i + 1] * (alpX[i + 1] + alpX[i]) / (2.0 * x.h * x.h);
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

	int Nx, Ny, Nz;

	Nx = x.n - 2 * x.pml_pts;
	Ny = y.n - 2 * y.pml_pts;
	Nz = z.n - 2 * z.pml_pts;

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
	for (int k = 0; k < Nx; k++)
			u1D[k] = x_sol_prd[w + k * Ny * Nz];
#endif
}


void GenRHS2DandSolutionSyntetic(int i, size_m y, size_m z, dcsr* D2csr, dtype* u2Dsynt, dtype *f2D)
{

}

// v[i] = D[i] * v[i]
void DenseDiagMult(int n, dtype *diag, dtype *v, dtype *f)
{
#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++)
		f[i] = diag[i] * v[i];
}


void take_coord3D(int n1, int n2, int n3, int l, int& i, int& j, int& k)
{
	i = 0, j = 0, k = 0;
	k = l / (n1 * n2);
	j = (l - k * n1 * n2) / n1;
	i = l - n1 * n2 * k - n1 * j;

//	printf("l = %d i = %d j = %d k = %d\n", l, i, j, k);
}

void take_coord2D(int n1, int n2, int l, int& i, int& j)
{
	i = 0, j = 0;
	j = l / n1;
	i = l - n1 * j;

	//	printf("l = %d i = %d j = %d k = %d\n", l, i, j, k);
}

void reducePML3D(size_m x, size_m y, size_m z, int size1, dtype *vect, int size2, dtype *vect_red)
{
	int i = 0, j = 0, k = 0;
	int numb = 0;

	if (size1 == size2)
	{
		printf("There is no PML reduction\n");

#pragma omp parallel for schedule(static)
		for (int i = 0; i < size1; i++)
			vect_red[i] = vect[i];

		return;
	}

	for (int l = 0; l < size1; l++)
	{
		take_coord3D(x.n, y.n, z.n, l, i, j, k);
		if(i >= x.pml_pts && j >= y.pml_pts && k >= z.pml_pts && i < (x.n - x.pml_pts) && j < (y.n - y.pml_pts) && k < (z.n - z.pml_pts)) vect_red[numb++] = vect[l];
	}

	if (numb != size2) printf("ERROR of reducing PML: %d != %d\n", numb, size2);
	else printf("PML is reduced successfully!\n");
}

void reducePML2D(size_m x, size_m y, int size1, dtype *vect, int size2, dtype *vect_red)
{
	int i = 0, j = 0, k = 0;
	int numb = 0;

	if (size1 == size2)
	{
		printf("There is no PML 2D reduction\n");

#pragma omp parallel for schedule(static)
		for (int i = 0; i < size1; i++)
			vect_red[i] = vect[i];

		return;
	}

	for (int l = 0; l < size1; l++)
	{
		take_coord2D(x.n, y.n, l, i, j);
		if (i >= x.pml_pts && j >= y.pml_pts && i < (x.n - x.pml_pts) && j < (y.n - y.pml_pts)) vect_red[numb++] = vect[l];
	}

	if (numb != size2) printf("ERROR of reducing PML 2D: %d != %d\n", numb, size2);
	else printf("PML 2D is reduced successfully!\n");
}


void GenRHSandSolution(size_m x, size_m y, size_m z, /* output */ dtype *u, dtype *f)
{
	int n = x.n * y.n;
	int size = n * z.n;

	double center = (double)(x.l / 2.0);
	//double center = 0;

	point source = { center, center, center };

	//printf("SOURCE AT x = %lf, y = %lf, z = %lf\n", source.x, source.y, source.z);

	// approximation of exact right hand side (inner grid points)
#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < z.n; k++)
		for (int j = 0; j < y.n; j++)
//#pragma omp simd
			for (int i = 0; i < x.n; i++)
			{
				f[k * n + j * x.n + i] = F_ex_complex(x, y, z, (i + 1) * x.h, (j + 1) * y.h, (k + 1) * z.h, source);
			//	printf("%d\n", k * n + j * x.n + i);
			}

	//system("pause");
	// УБРАТЬ ВСЕ ПЕРЕБРОСКИ В ПРАВУЮ ЧАСТЬ
	// ТАК КАК КРАЕВЫЕ УСЛОВИЯ НУЛЕВЫЕ
	// А НЕ КАКАЯ-ТО ФУНКЦИЯ ФИ

#if 0

	// for boundaries z = 0 and z = Lz we distract blocks B0 and Bm from the RHS
#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < y.n; j++)
//#pragma omp simd
		for (int i = 0; i < x.n; i++)
		{
			f[ind(0, n) + ind(j, x.n) + i] -= u_ex_complex(x, y, z, (i + 1) * x.h, (j + 1) * y.h, 0, source) / (z.h * z.h); // u|z = 0
			f[ind(z.n - 1, n) + ind(j, x.n) + i] -= u_ex_complex(z, y, z, (i + 1)  * x.h, (j + 1) * y.h, z.l, source) / (z.h * z.h); // u|z = h
		}


	// for each boundary 0 <= z <= Lz
	// we distract 4 known boundaries f0, fl, g0, gL from right hand side
#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < z.n; k++)
	{
//#pragma omp simd
		for (int i = 0; i < x.n; i++)
		{
			f[k * n + 0 * x.n + i] -= u_ex_complex(x, y, z, (i + 1) * x.h, 0, (k + 1) * z.h, source) / (y.h * y.h);
			f[k * n + (y.n - 1) * x.n + i] -= u_ex_complex(x, y, z, (i + 1) * x.h, y.l, (k + 1) * z.h, source) / (y.h * y.h);
		}
		for (int j = 0; j < y.n; j++)
		{
			f[k * n + j * x.n + 0] -= u_ex_complex(x, y, z, 0, (j + 1) * y.h, (k + 1) * z.h, source) / (x.h * x.h);
			f[k * n + j * x.n + x.n - 1] -= u_ex_complex(x, y, z, x.l, (j + 1) * y.h, (k + 1) * z.h, source) / (x.h * x.h);
		}
	}

#endif

	// approximation of inner points values
#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < z.n; k++)
		for (int j = 0; j < y.n; j++)
//#pragma omp simd
			for (int i = 0; i < x.n; i++)
				u[k * n + j * x.n + i] = u_ex_complex(x, y, z, (i + 1) * x.h, (j + 1) * y.h, (k + 1) * z.h, source);


	printf("RHS and solution are constructed\n");
}

double d(double x)
{
	//printf("x = %lf\n", x);

	const double C = 50;
	return C * pow(x, 4);
}

dtype alph(size_m size, int xl, int xr, int i)
{
	double x = 0;
	if (i < xl || i >= (size.n + 2 - xr))
	{
		if (i < xl) x = ((xl - i) * size.h) / (size.pml_pts);
		else if (i >= (size.n + 2 - xr)) x = ((size.n + 3 - xr - i) * size.h) / (size.pml_pts);
		return { omega * omega / (omega * omega + d(x) * d(x)), omega * d(x) / (omega * omega + d(x) * d(x)) };
	}
	else
		return 1.0;
}

void SetPml3D(int blk3D, size_m x, size_m y, size_m z, int n, dtype* alpX, dtype* alpY, dtype* alpZ)
{

	for (int blk2D = 0; blk2D < y.n; blk2D++)
	{
		SetPml2D(blk3D, blk2D, x, y, z, n, alpX, alpY, alpZ);
	}

}

void SetPml2D(int blk3D, int blk2D, size_m x, size_m y, size_m z, int n, dtype* alpX, dtype* alpY, dtype *alpZ)
{
	if (blk2D < y.pml_pts || blk2D >= (y.n - y.pml_pts)) // pml to first Nx + pml strings
	{
		// from 0 to 1 including boundaries
#pragma omp parallel for schedule(runtime)
		for (int i = 0; i < y.n + 2; i++) // n + 1 points = 2 bound + (n  - 1) inside domain
		{
			alpX[i] = alph(x, x.pml_pts, x.pml_pts, i);     // a(x) != 1 only in the pml section
			alpY[i] = alph(y, y.n - 1, y.n - 1, i); // a(y) != 1 in the whole domain
		}
		if (blk3D < z.pml_pts || blk3D >= (z.n - z.pml_pts))
		{
			for (int i = 0; i < y.n + 2; i++)
				//alpZ[i] = alph(z, y.n - 1, y.n - 1, i);  // a(z) !=  1 in the whole domain
				alpZ[i] = alph(z, 0, 0, i);
		}
		else
		{
			for (int i = 0; i < y.n + 2; i++)
				alpZ[i] = alph(z, 0, 0, i);  // a(z) == 1 in the whole domain
		}
	}
	else
	{
#pragma omp parallel for schedule(runtime)
		for (int i = 0; i < y.n + 2; i++) // n + 1 points = 2 bound + (n  - 1) inside domain
		{
			alpX[i] = alph(x, x.pml_pts, x.pml_pts, i);   // a(x) != 1 only in the pml section
			alpY[i] = alph(y, 0, 0, i);       // a(y) == 1 in the whole domain
		}

		if (blk3D < z.pml_pts || blk3D >= (z.n - z.pml_pts))
		{
			for (int i = 0; i < y.n + 2; i++)
				//alpZ[i] = alph(z, y.n - 1, y.n - 1, i);  // a(z) !=  1 in the whole domain
				alpZ[i] = alph(z, 0, 0, i);
		}
		else
		{
			for (int i = 0; i < y.n + 2; i++)
				alpZ[i] = alph(z, 0, 0, i);  // a(z) == 1 in the whole domain
		}
	}
}

void GenerateDiagonal2DBlock(char* problem, int blk3D, size_m x, size_m y, size_m z, dtype *DD, int lddd, dtype *alpX, dtype *alpY, dtype *alpZ)
{
	int n = x.n * y.n;
	int size = n * z.n;
	double k = (double)kk;

	// diagonal blocks in dense format
	//#pragma omp parallel for simd schedule(simd:static)
	for (int blk2D = 0; blk2D < y.n; blk2D++)
	{
		SetPml2D(blk3D, blk2D, x, y, z, n, alpX, alpY, alpZ);
		GenerateDiagonal1DBlock(0, blk2D, x, y, z, &DD[blk2D * x.n + lddd * (blk2D * x.n)], lddd, alpX, alpY, alpZ);

		for (int i = 0; i < x.n; i++)
		{
			if (blk2D >= 1) DD[(i + blk2D * x.n) + lddd * (i) + lddd * (blk2D * x.n - x.n)] = alpY[i + 1] * (alpY[i + 1] + alpY[i]) / (2.0 * y.h * y.h);
			if (blk2D <= y.n - 2)  DD[(i + blk2D * x.n) + lddd * (i) + lddd * (blk2D * x.n + x.n)] = alpY[i + 1] * (alpY[i + 1] + alpY[i]) / (2.0 * y.h * y.h);
		}
	}

	//DD[i + lddd * i] = -2.0 * (1.0 / (x.h * x.h) + 1.0 / (y.h * y.h) + 1.0 / (z.h * z.h));
	//	DD[i + lddd * i] = -alpX[i + 1] * (alpX[i + 2] + 2.0 * alpX[i + 1] + alpX[i]) / (2.0 * x.h * x.h)
	//	/		  - alpY[i + 1] * (alpY[i + 2] + 2.0 * alpY[i + 1] + alpY[i]) / (2.0 * y.h * y.h);
	//		   -alpZ[i + 1] * (alpZ[i + 2] + 2.0 * alpZ[i + 1] + alpZ[i]) / (2.0 * z.h * z.h);
	//	if (i > 0) DD[i + lddd * (i - 1)] = 1.0 / (x.h * x.h);
	//	if (i < n - 1) DD[i + lddd * (i + 1)] = 1.0 / (x.h * x.h);
	//#ifdef HELMHOLTZ
	//	DD[i + lddd * i] += k * k;
	//#endif

	//	for (int i = 0; i < n; i++)
	//	{
	//		if (i % x.n == 0 && i > 0)
	//		{
	//			DD[i - 1 + lddd * i] = 0;
	//			DD[i + lddd * (i - 1)] = 0;
	//		}
	//	}

}


void GenSparseMatrixOnline3D(size_m x, size_m y, size_m z, dtype* B, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, ccsr* Acsr)
{
	int n = x.n * y.n;
	int size = n * z.n;
	int non_zeros_on_prev_level = 0;
	map<vector<int>, dtype> CSR;

	//Diag(n, BL, ldbl, dtype{ 1.0 / (z.h * z.h), 0 });
	//Diag(n, BR, ldbr, dtype{ 1.0 / (z.h * z.h), 0 });

	dtype *alpX = alloc_arr<dtype>(x.n + 2);
	dtype *alpY = alloc_arr<dtype>(y.n + 2);
	dtype *alpZ = alloc_arr<dtype>(z.n + 2);


	for (int blk3D = 0; blk3D < z.n; blk3D++)
	{
		if (blk3D < z.pml_pts || blk3D >= (z.n - z.pml_pts))
		{
			for (int i = 0; i < z.n + 2; i++)
				//alpZ[i] = alph(z, z.n - 1, z.n - 1, i);  // a(z) !=  1 in the whole domain
				alpZ[i] = alph(z, 0, 0, i);
		}
		else
		{
			for (int i = 0; i < z.n + 2; i++)
				alpZ[i] = alph(z, 0, 0, i);  // a(z) == 1 in the whole domain
		}
		// Set vector B
		if (blk3D < z.n - 1)
		{
//#pragma omp simd
			for (int j = 0; j < y.n; j++)
				for(int i = 0; i < y.n; i++)
				{
				B[blk3D * n + (x.n * j + i)] = alpZ[i + 1] * (alpZ[i + 2] + alpZ[i + 1]) / (2.0 * z.h * z.h);
				//	printf("%d %lf\n", i + blk * n, B[ind(blk, n) + i].real());
				}
		}
		DiagVec(n, BL, ldbl, B); // B тоже должен меняться в зависимости от уровня blk
		DiagVec(n, BR, ldbr, B);

		GenerateDiagonal2DBlock("3D", blk3D, x, y, z, A, lda, alpX, alpY, alpZ);
		CSR = BlockRowMat_to_CSR(blk3D, x.n, y.n, z.n, BL, ldbl, A, lda, BR, ldbr, Acsr, non_zeros_on_prev_level); // ВL, ВR and A - is 2D dimensional matrices (n x n)
																												 //	print_map(CSR);
		
		//printf("3D Block: %d\n", blk3D);
		//print(n, n, A, lda, "A");
		//	system("pause");
	//	if (blk3D == 0) print(n, n, A, lda, "A");
	//	if (blk3D == 0) print(n, n, BL, ldbl, "A");
	//	printf("Non_zeros inside row_block %d: %d\n", blk3D, non_zeros_on_prev_level);
	}
	printf("Non_zeros inside generating function: %d\n", non_zeros_on_prev_level);

}

dtype alpha(size_m xyz, int i)
{
	double x = 0;
	if (i == -1 || i == xyz.n)
	{
		// bound case
		return 1.0;
	}
	else if (i < xyz.pml_pts || i >= (xyz.n - xyz.pml_pts))
	{
		//printf("PMLLLLLLLLLLL: %d < %d || %d >= %d\n", i, xyz.pml_pts, i, xyz.n - xyz.pml_pts);
		if (i < xyz.pml_pts) x = (xyz.pml_pts - i) * xyz.h / (xyz.pml_pts);
		else if (i >= (xyz.n - xyz.pml_pts)) x = (i - xyz.n + xyz.pml_pts + 1) * xyz.h / (xyz.pml_pts);
		return { omega * omega / (omega * omega + d(x) * d(x)), omega * d(x) / (omega * omega + d(x) * d(x)) };
	}
	else return 1.0;
}

dtype beta3D(size_m x, size_m y, size_m z, int diag_case, int i, int j, int k)
{
	if (diag_case == 0)
	{
		dtype value;
		
		value = -alpha(x, i) * (alpha(x, i + 1) + 2.0 * alpha(x, i) + alpha(x, i - 1)) / (2.0 * x.h * x.h) 
			    -alpha(y, j) * (alpha(y, j + 1) + 2.0 * alpha(y, j) + alpha(y, j - 1)) / (2.0 * y.h * y.h)
			    -alpha(z, k) * (alpha(z, k + 1) + 2.0 * alpha(z, k) + alpha(z, k - 1)) / (2.0 * z.h * z.h);

		int l = k * x.n * y.n + j * x.n + i;

		//printf("l = %d : i = %d j = %d k = %d value = %lf %lf\n", l, i, j, k, value.real(), value.imag());

		return value;
	}
	else if (diag_case == -1 || diag_case == 1)
	{
		return alpha(x, i) * (alpha(x, i + 1) + alpha(x, i)) / (2.0 * x.h * x.h);
	}
	else if (diag_case == -2 || diag_case == 2)
	{
		return alpha(y, j) * (alpha(y, j + 1) + alpha(y, j)) / (2.0 * y.h * y.h);
	}
	else if (diag_case == -3 || diag_case == 3)
	{
		return alpha(z, k) * (alpha(z, k + 1) + alpha(z, k)) / (2.0 * z.h * z.h);
	}
	return 0;
}

dtype beta2D(size_m x, size_m y, int diag_case, int i, int j)
{
	if (diag_case == 0)
	{
		dtype value;

		value = -alpha(x, i) * (alpha(x, i + 1) + 2.0 * alpha(x, i) + alpha(x, i - 1)) / (2.0 * x.h * x.h)
			    -alpha(y, j) * (alpha(y, j + 1) + 2.0 * alpha(y, j) + alpha(y, j - 1)) / (2.0 * y.h * y.h);

		int l = j * x.n + i;

		//printf("l = %d : i = %d j = %d k = %d value = %lf %lf\n", l, i, j, k, value.real(), value.imag());

		return value;
	}
	else if (diag_case == -1 || diag_case == 1)
	{
		return alpha(x, i) * (alpha(x, i + 1) + alpha(x, i)) / (2.0 * x.h * x.h);
	}
	else if (diag_case == -2 || diag_case == 2)
	{
		return alpha(y, j) * (alpha(y, j + 1) + alpha(y, j)) / (2.0 * y.h * y.h);
	}

	return 0;
}

void GenSparseMatrixOnline3DwithPML(size_m x, size_m y, size_m z, dtype* B, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, ccsr* Acsr, double eps)
{
	int n = x.n * y.n;
	int size = x.n * y.n * z.n;
	int size2 = (n - x.n) * z.n;
	int size3 = (n - x.n) * z.n;
	int size4 = size - n;
	int non_zeros_in_3Dblock3diag = size + size2 * 2 + size3 * 2 + size4 * 2;
	double RelRes = 0;
	double k = (double)kk;

	printf("Number k = %lf\n", k);

	printf("analytic non_zeros in PML function: %d\n", non_zeros_in_3Dblock3diag);

	// All elements

	dtype *diag = alloc_arr<dtype>(size); // 0
	dtype *subXdiag = alloc_arr<dtype>(size2); // -1
	dtype *supXdiag = alloc_arr<dtype>(size2); // 1
	dtype *subYdiag = alloc_arr<dtype>(size3); // -2
	dtype *supYdiag = alloc_arr<dtype>(size3); // 2
	dtype *subZdiag = alloc_arr<dtype>(size4); // -3
	dtype *supZdiag = alloc_arr<dtype>(size4); // 3

	int i1, j1, k1;
	int i2, j2, k2;
	int c[7] = { 0 };
	int cc[7] = { 0 };

	//for (int i = 0; i < size3; i++)
	//	printf("%lf %lf %lf %lf\n", subXdiag[i].real(), subXdiag[i].imag(), supXdiag[i].real(), supXdiag[i].imag());
	//RelRes = rel_error(zlange, size2, 1, subXdiag, supXdiag, size2, eps);
	//printf("RelRes Xdiag: %e\n", RelRes);
	//RelRes = rel_error(zlange, size3, 1, subYdiag, supYdiag, size3, eps);
	//printf("RelRes Ydiag: %e\n", RelRes);
	//RelRes = rel_error(zlange, size4, 1, subZdiag, supZdiag, size4, eps);
	//printf("RelRes Zdiag: %e\n", RelRes);

	// Generate Dcsr by using BL, BR and A
	int non_zeros_on_prev_level = 0;
#if 0
	for (int l1 = 0; l1 < size; l1++)
		for (int l2 = 0; l2 < size; l2++)
		{
			take_coord3D(x.n, y.n, z.n, l1, i1, j1, k1);
			take_coord3D(x.n, y.n, z.n, l2, i2, j2, k2);

			if (l1 == l2) diag[c[0]++] = beta(x, y, z, 0, i1, j1, k1);
			else if (l1 == l2 - 1 && (l1 + 1) % x.n != 0) supXdiag[c[1]++] = beta(x, y, z, 1, i1, j1, k1); // right
			else if (l1 == l2 + 1 && l1 % x.n != 0) subXdiag[c[2]++] = beta(x, y, z, -1, i1, j1, k1); // left
			else if (l1 == l2 - x.n && (l1 + x.n) % n >= x.n) supYdiag[c[3]++] = beta(x, y, z, 2, i1, j1, k1); // right
			else if (l1 == l2 + x.n && l1 % n >= x.n) subYdiag[c[4]++] = beta(x, y, z, -2, i1, j1, k1); // left
			else if (l1 == l2 - x.n * y.n) supZdiag[c[5]++] = beta(x, y, z, 3, i1, j1, k1);
			else if (l1 == l2 + x.n * y.n) subZdiag[c[6]++] = beta(x, y, z, -3, i1, j1, k1);
		}

	/*
	printf("%d %d %d\n", 0, c[0], size);
	printf("%d %d %d\n", 1, c[1], size2);
	printf("%d %d %d\n", 2, c[2], size2);
	printf("%d %d %d\n", 3, c[3], size3);
	printf("%d %d %d\n", 4, c[4], size3);
	printf("%d %d %d\n", 5, c[5], size4);
	printf("%d %d %d\n", 6, c[6], size4);


	TestEqual(c[0], size, "c[0] != size");
	TestEqual(c[1], size2, "c[1] != size2");
	TestEqual(c[2], size2, "c[2] != size2");
	TestEqual(c[3], size3, "c[3] != size3");
	TestEqual(c[4], size3, "c[4] != size3");
	TestEqual(c[5], size4, "c[3] != size4");
	TestEqual(c[6], size4, "c[4] != size4");*/


	map<vector<int>, dtype> CSR;
	for (int blk3D = 0; blk3D < z.n; blk3D++)
	{
		Clear(n, n, A, lda);
		Clear(n, n, BL, ldbr);
		Clear(n, n, BR, ldbl);

		if (blk3D < z.n - 1)
		{
			for (int j = 0; j < y.n; j++)
				for (int i = 0; i < x.n; i++)
				{
					B[blk3D * n + (x.n * j + i)] = supZdiag[blk3D * n + (x.n * j + i)];
					//	printf("%d %lf %lf\n", blk3D * n + (x.n * j + i), B[blk3D * n + (x.n * j + i)].real(), B[blk3D * n + (x.n * j + i)].imag());
				}
		}

		if (blk3D == 0) DiagVec(n, BR, ldbr, &B[blk3D * n]); // B тоже должен меняться в зависимости от уровня blk
		else if (blk3D == z.n - 1) DiagVec(n, BL, ldbl, &B[(blk3D - 1) * n]);
		else
		{
			DiagVec(n, BL, ldbl, &B[(blk3D - 1) * n]);
			DiagVec(n, BR, ldbr, &B[blk3D * n]);
		}


		// Construction of A
		for (int i = 0; i < n; i++)
		{
			int block2D = i / x.n;
			A[i + lda * i] = diag[cc[0]++];
#ifdef HELMHOLTZ
			A[i + lda * i] += dtype{ k * k, 0 };
#endif
			//printf("diag: %10.8lf %10.8lf\n", A[i + lda * i].real(), A[i + lda * i].imag());
			if ((i + 1) % x.n != 0) A[i + lda * (i + 1)] = supXdiag[cc[1]++]; // условие на диагональ
			if (i % x.n != 0) A[i + lda * (i - 1)] = subXdiag[cc[2]++];
			if (block2D <= y.n - 2) A[i + lda * (i + x.n)] = supYdiag[cc[3]++]; // условие на первый и последний блок
			if (block2D >= 1) A[i + lda * (i - x.n)] = subYdiag[cc[4]++];
		}
		//print(n, n, A, lda, "A gen");
		CSR = BlockRowMat_to_CSR(blk3D, x.n, y.n, z.n, BL, ldbl, A, lda, BR, ldbr, Acsr, non_zeros_on_prev_level); // A - 2D Block
		
	}
	printf("Non_zeros inside generating PML: %d\n", non_zeros_on_prev_level);
#else
	int count = 0;

	for (int l1 = 0; l1 < size; l1++)
	{
		Acsr->ia[l1] = count + 1;
		for (int l2 = 0; l2 < size; l2++)
		{
			take_coord3D(x.n, y.n, z.n, l1, i1, j1, k1);
			take_coord3D(x.n, y.n, z.n, l2, i2, j2, k2);

			if (l1 == l2)
			{
				Acsr->ja[count] = l2 + 1;
#ifdef HELMHOLTZ
				Acsr->values[count] = dtype{ k * k, 0 };
				Acsr->values[count++] += beta3D(x, y, z, 0, i1, j1, k1);
#else
				Acsr->values[count++] = beta3D(x, y, z, 0, i1, j1, k1);
#endif
			}
			else if (l1 == l2 - 1 && (l1 + 1) % x.n != 0)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta3D(x, y, z, 1, i1, j1, k1); // right
			}
			else if (l1 == l2 + 1 && l1 % x.n != 0)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta3D(x, y, z, -1, i1, j1, k1); // left
			}
			else if (l1 == l2 - x.n && (l1 + x.n) % n >= x.n)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta3D(x, y, z, 2, i1, j1, k1); // right
			}
			else if (l1 == l2 + x.n && l1 % n >= x.n)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta3D(x, y, z, -2, i1, j1, k1); // left
			}
			else if (l1 == l2 - x.n * y.n)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta3D(x, y, z, 3, i1, j1, k1);
			}
			else if (l1 == l2 + x.n * y.n)
			{	
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta3D(x, y, z, -3, i1, j1, k1);
			}

		}
	}
	
	if (non_zeros_in_3Dblock3diag != count) printf("Failed generation!");
	else printf("Successed genration!\n");
	printf("Non_zeros inside generating PML function: %d\n", count);
#endif

	free_arr(diag);
	free_arr(subXdiag);
	free_arr(supXdiag);
	free_arr(subYdiag);
	free_arr(supYdiag);
	free_arr(subZdiag);
	free_arr(supZdiag);

}

void GenSparseMatrixOnline2DwithPML(int w, size_m y, size_m z, ccsr* Acsr)
{
	int n = y.n;
	int n2 = n / 2;
	int size = y.n * z.n;
	int size2 = size - z.n;
	int size3 = size - y.n;
	int non_zeros_in_2Dblock3diag = size + size2 * 2 + size3 * 2;
	double RelRes = 0;
	double k = (double)kk;
	double kww = 4 * PI * PI * (w - n2) * (w - n2) / (y.l * y.l);

	//printf("Number k = %lf\n", k);

	//printf("analytic non_zeros in PML function: %d\n", non_zeros_in_2Dblock3diag);


	if (non_zeros_in_2Dblock3diag != Acsr->non_zeros) printf("ERROR! Uncorrect value of non_zeros inside 2D: %d != %d\n", non_zeros_in_2Dblock3diag, Acsr->non_zeros);

	printf("Gen 2D matrix for frequency w = %d\n", w - n2);

	// All elements

	dtype *diag = alloc_arr<dtype>(size); // 0
	dtype *subXdiag = alloc_arr<dtype>(size2); // -1
	dtype *supXdiag = alloc_arr<dtype>(size2); // 1
	dtype *subYdiag = alloc_arr<dtype>(size3); // -2
	dtype *supYdiag = alloc_arr<dtype>(size3); // 2

	int j1, k1;
	int j2, k2;

	int count = 0;

	for (int l1 = 0; l1 < size; l1++)
	{
		Acsr->ia[l1] = count + 1;
		for (int l2 = 0; l2 < size; l2++)
		{
			take_coord2D(y.n, z.n, l1, j1, k1);
			take_coord2D(y.n, z.n, l2, j2, k2);

			if (l1 == l2)
			{
				Acsr->ja[count] = l2 + 1;
#ifdef HELMHOLTZ
				Acsr->values[count] = -dtype{ kww, 0 };
				Acsr->values[count] += dtype{ k * k, 0 };
				Acsr->values[count++] += beta2D(y, z, 0, j1, k1);
#else
				Acsr->values[count++] = beta2D(y, z, 0, j1, k1);
#endif
				
			}
			else if (l1 == l2 - 1 && (l1 + 1) % y.n != 0)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta2D(y, z, 1, j1, k1); // right
			}
			else if (l1 == l2 + 1 && l1 % y.n != 0)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta2D(y, z, -1, j1, k1); // left
			}
			else if (l1 == l2 - y.n)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta2D(y, z, 2, j1, k1); // right
			}
			else if (l1 == l2 + y.n)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta2D(y, z, -2, j1, k1); // left
			}

		}
	}

	if (non_zeros_in_2Dblock3diag != count) printf("FAILED generation!!! %d != %d\n", non_zeros_in_2Dblock3diag, count);
	else
	{
		printf("SUCCESSED generation!\n");
		printf("Non_zeros inside generating PML function: %d\n", count);
	}

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
	//	print(n, n, BR, ldar, "BR gen");
		construct_block_row(zlacpy, n, n, (dtype*)NULL, ldbl, A, lda, BR, ldbr, Arow, ldar);

		CSR_A = dense_to_CSR(n, 2 * n, Arow, ldar, &Acsr->ia[0], &Acsr->ja[0], &Acsr->values[0]);
		non_zeros_on_prev_level = CSR_A.size();
	//	print_map(CSR_A);
	//	system("pause");
	}
	else if (blk == n3 - 1)
	{
		//print(n, n, BL, ldar, "BL gen");
		construct_block_row(zlacpy, n, n, BL, ldbl, A, lda, (dtype*)NULL, ldbr, Arow, ldar);
	//	print(n, 2 * n, Arow, ldar, "Arowlast");
		CSR_A = dense_to_CSR(n, 2 * n, Arow, ldar, &Acsr->ia[ind(blk, n)], &Acsr->ja[non_zeros_on_prev_level], &Acsr->values[non_zeros_on_prev_level]);
		shift_values(n, &Acsr->ia[ind(blk, n)], non_zeros_on_prev_level, CSR_A.size(), &Acsr->ja[non_zeros_on_prev_level], n * (blk - 1));
		non_zeros_on_prev_level += CSR_A.size();
	}
	else
	{
		construct_block_row(zlacpy, n, n, BL, ldbl, A, lda, BR, ldbr, Arow, ldar);
		//print(n, 3 * n, Arow, ldar, "Arow_middle");
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

void GenSparseMatrixOnline2D(char *str, int w, size_m x, size_m y, size_m z, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, ccsr* Acsr)
{
	int n = y.n;
	int n2 = n / 2;
	int size = n * z.n;
	int non_zeros_on_prev_level = 0;
	map<vector<int>, dtype> CSR;

	Diag(n, BL, ldbl, dtype{ 1.0 / (y.h * y.h), 0 });
	Diag(n, BR, ldbr, dtype{ 1.0 / (y.h * y.h), 0 });

	dtype *alpX = alloc_arr<dtype>(y.n + 2);
	dtype *alpY = alloc_arr<dtype>(y.n + 2);
	dtype *alpZ = alloc_arr<dtype>(z.n + 2);

	double kww = 4 * PI * PI * (w - n2) * (w - n2) / (y.l * y.l);
	printf("Gen 2D matrix for frequency w = %d\n", w - n2);
	for (int blk2D = 0; blk2D < z.n; blk2D++)
	{
		//printf("Blk: %d\n", blk);

		GenerateDiagonal1DBlock(kww, blk2D, x, y, z, A, lda, alpX, alpY, alpZ);
		CSR = Block1DRowMat_to_CSR(blk2D, y.n, z.n, BL, ldbl, A, lda, BR, ldbr, Acsr, non_zeros_on_prev_level); // ВL, ВR and A - is 2D dimensional matrices (n x n)
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

	// Pardiso initialization 
	pardisoinit(pt, &mtype, iparm);

	// Note: it is very important that the pointer PT is initialised with zero 
	// before the first call of PARDISO. After the first call you should NEVER modify
	// the pointer, because it could cause a serious memory leak or a crash.

	int maxfct = 1;
	int mnum = 1;
	int phase = 13;
	int rhs = 1;
	int msglvl = 0;
	int error = 0;

	sparse_struct *my_check = (sparse_struct*)malloc(sizeof(sparse_struct));

	sparse_matrix_checker_init(my_check);

	my_check->n = size;
	my_check->csr_ia = Dcsr->ia;
	my_check->csr_ja = Dcsr->ja;
	my_check->indexing = MKL_ONE_BASED;
	my_check->matrix_structure = MKL_GENERAL_STRUCTURE;
	my_check->matrix_format = MKL_CSR;
	my_check->message_level = MKL_PRINT;
	my_check->print_style = MKL_C_STYLE;

	int ERROR_RESULT = sparse_matrix_checker(my_check);

	printf("ERROR_RESULT_CHECK_CSR: %d\n", ERROR_RESULT);

	pardiso(pt, &maxfct, &mnum, &mtype, &phase, &size, Dcsr->values, Dcsr->ia, Dcsr->ja, perm, &rhs, iparm, &msglvl, f, x_pard, &error);
	printf("PARDISO_ERROR: %d\n", error);

	free_arr(iparm);
	free_arr(perm);
	free_arr(pt);
}

void print_csr(int n, dcsr* A)
{
	for (int i = 0; i < n; i++)
		printf("i = %d j = %d value = %lf\n", A->ia[i], A->ja[i], A->values[i]);
}

void shift_values(int rows, int *ia, int shift_non_zeros, int non_zeros, int *ja, int shift_columns)
{
#pragma omp parallel for schedule(static)
	for (int i = 0; i < rows; i++)
		ia[i] += shift_non_zeros;

#pragma omp parallel for schedule(static)
	for (int i = 0; i < non_zeros; i++)
		ja[i] += shift_columns;
}

void DiagVec(int n, dtype *H, int ldh, dtype *value)
{
	int i = 0, j = 0;
//#pragma omp parallel private(i,j)
	{
//#pragma omp for schedule(static)
		for (j = 0; j < n; j++)
			for (i = 0; i < n; i++)
			{
				if (i == j) H[i + ldh * j] = value[j];
				else H[i + ldh * j] = 0.0;
			}
	}
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

dtype u_ex_complex(size_m xx, size_m yy, size_m zz, double x, double y, double z, point source)
{

#ifdef HELMHOLTZ
	x -= source.x;
	y -= source.y;
	z -= source.z;

	double r = sqrt(x * x + y * y + z * z);

	if (r == 0) r = 0.05;
	
	double arg = -(double)(kk * r);

	return my_exp(arg) / (4.0 * PI * r);

/*	double ksi1, ksi2, ksi3;

	ksi1 = ksi2 = ksi3 = sqrt(1.0 / 3.0);

	return my_exp(omega / c_z * (ksi1 * x + ksi2 * y + ksi3 * z));*/

#else
	return x * y * z * (x - xx.l) * (y - yy.l) * (z - zz.l);
#endif
}

dtype F_ex_complex(size_m xx, size_m yy, size_m zz, double x, double y, double z, point source)
{
//	return 0;


	//printf("x = %lf, y = %lf, z = %lf\n", x, y, z);
#ifdef HELMHOLTZ
	if (x == source.x && y == source.y && z == source.z)
	{
		printf("SOURCE AT x = %lf, y = %lf, z = %lf\n", x, y, z);
		return -1.0 / (xx.h * yy.h * zz.h);
	}
	else
		return 0;
#else
	return 2.0 * (x * (x - xx.l) * z * (z - zz.l) + y * (y - yy.l) * z * (z - zz.l) + x * (x - xx.l) * y * (y - yy.l));
#endif
}

void print_map(const map<vector<int>, dtype>& SD)
{
	cout << "SD size = " << SD.size() << endl;
	for (const auto& item : SD)
		//cout << "m = " << item.first[0] << " n = " << item.first[1] << " value = " << item.second.real() << " " << item.second.imag() << endl;
		printf("m = %d n = %d value = %lf %lf\n", item.first[0], item.first[1], item.second.real(), item.second.imag());

}

double rel_error(int n, int k, double *Hrec, double *Hinit, int ldh, double eps)
{
	double norm = 0;

	// Norm of residual
#pragma omp parallel for schedule(static)
	for (int j = 0; j < k; j++)
//#pragma omp simd
		for (int i = 0; i < n; i++)
			Hrec[i + ldh * j] = Hrec[i + ldh * j] - Hinit[i + ldh * j];

	norm = dlange("Frob", &n, &k, Hrec, &ldh, NULL);
	norm = norm / dlange("Frob", &n, &k, Hinit, &ldh, NULL);

	return norm;

	//if (norm < eps) printf("Norm %12.10e < eps %12.10lf: PASSED\n", norm, eps);
	//else printf("Norm %12.10lf > eps %12.10lf : FAILED\n", norm, eps);
}

void ResidCSR(size_m x, size_m y, size_m z, ccsr* Dcsr, dtype* x_sol, dtype *f, dtype* g, double &RelRes)
{
	int n = x.n * y.n;
	int size = n * z.n;
	int size_no_pml = (x.n - 2 * x.pml_pts) * (y.n - 2 * y.pml_pts) * (z.n - 2 * z.pml_pts);
	dtype *f1 = alloc_arr<dtype>(size);
	int ione = 1;

	// Multiply matrix A in CSR format by vector x_sol to obtain f1
	mkl_zcsrgemv("No", &size, Dcsr->values, Dcsr->ia, Dcsr->ja, x_sol, f1);

#pragma omp parallel for schedule(static)
	for (int i = 0; i < size; i++)
		g[i] = f[i] - f1[i];

	//	for (int i = 0; i < size; i++)
	//		printf("%lf %lf\n", f[i], f1[i]);

#ifdef DEBUG
	print_vec(size, f, g, "f and g");
#endif

	dtype *g_no_pml = alloc_arr<dtype>(size_no_pml);
	dtype *f_no_pml = alloc_arr<dtype>(size_no_pml);
	reducePML3D(x, y, z, size, g, size_no_pml, g_no_pml);
	reducePML3D(x, y, z, size, f, size_no_pml, f_no_pml);

	//RelRes = zlange("Frob", &size, &ione, g, &size, NULL);
	//RelRes = RelRes / zlange("Frob", &size, &ione, f, &size, NULL);

	RelRes = zlange("Frob", &size_no_pml, &ione, g_no_pml, &size_no_pml, NULL);
	RelRes = RelRes / zlange("Frob", &size_no_pml, &ione, f_no_pml, &size_no_pml, NULL);

	printf("End resid\n");

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


void output(char *str, bool pml_flag, size_m x, size_m y, size_m z, dtype* x_orig, dtype* x_pard)
{
	char name[255];
	int Nx, Ny, Nz;

	if (pml_flag == true)
	{
		Nx = x.n - 2 * x.pml_pts;
		Ny = y.n - 2 * y.pml_pts;
		Nz = z.n - 2 * z.pml_pts;
	}
	else
	{
		Nx = x.n;
		Ny = y.n;
		Nz = z.n;
	}

	for (int k = 0; k < Nz; k++)
	{
		sprintf(name, "%s_%02d.dat", str, k);
		FILE *file = fopen(name, "w");
		for (int j = 0; j < Ny; j++)
			for (int i = 0; i < Nx; i++)
				fprintf(file, "%lf %12.10lf %12.10lf %12.10lf %12.10lf %12.10lf %12.10lf\n", i * x.h, j * y.h, k * z.h,
					x_orig[i + j * Nx + k * Ny * Nx].real(), x_orig[i + j * Nx + k * Ny * Nx].imag(),
					x_pard[i + j * Nx + k * Ny * Nx].real(), x_pard[i + j * Nx + k * Ny * Nx].imag());
		fclose(file);
	}
}


void gnuplot(char *splot, char *sout, bool pml_flag, int col, size_m x, size_m y, size_m z)
{
	char *str;
	str = alloc_arr<char>(255);
	int Nx, Ny, Nz;
	if (pml_flag == true)
	{
		Nx = x.n - 2 * x.pml_pts;
		Ny = y.n - 2 * y.pml_pts;
		Nz = z.n - 2 * z.pml_pts;
	}
	else
	{
		Nx = x.n;
		Ny = y.n;
		Nz = z.n;
	}

	FILE* file1;
	//sprintf(str, "run.plt", numb++);
	if (col == 4) str = "run_ex.plt";
	else str = "run_pard.plt";

	file1 = fopen(str, "w");

	fprintf(file1, "reset\nclear\nset term png font \"Times-Roman, 16\"\n");
	//fprintf(file, "set view map\n");
	fprintf(file1, "set xrange[0:%d]\nset yrange[0:%d]\n", (int)LENGTH, (int)LENGTH);
	fprintf(file1, "set pm3d\n");
	fprintf(file1, "set palette\n");

	for (int k = 0; k < Nz; k++)
	{
		//fprintf(file, "set cbrange[%6.4lf:%6.4lf]\n", x_orig[x.n - 1 + (y.n - 1) * x.n + k * y.n * x.n].real(), x_orig[0 + 0 * x.n + k * y.n * x.n].real());
	//	if (col == 4) fprintf(file, "set cbrange[%12.10lf:%12.10lf]\n", x_orig[0 + 0 * Nx + k * Ny * Nx].real(), x_orig[(Nx - 1 + (Ny - 1) * Nx) / 2 + k * Ny * Nx].real());
	//	else fprintf(file, "set cbrange[%12.10lf:%12.10lf]\n", x_sol[0 + 0 * Nx + k * Ny * Nx].real(), x_sol[(Nx - 1 + (Ny - 1) * Nx) / 2 + k * Ny * Nx].real());
		//printf("k = %d\nleft: %d  %lf \nright: %d %lf \n\n", k, 0 + 0 * x.n, x_orig[0 + 0 * Nx + k * Ny * Nx].real(), (Nx - 1 + (Ny - 1) * Nx) / 2, x_orig[(Nx - 1 + (Ny - 1) * Nx) / 2 + k * Ny * Nx].real());
		fprintf(file1, "set output '%s_z_%4.2lf.png'\n", sout, k * z.h);
		fprintf(file1, "splot '%s_%02d.dat' u 2:1:%d w linespoints pt 7 palette pointsize 1 notitle\n\n", splot, k, col);
	}

	fclose(file1);
	system(str);
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