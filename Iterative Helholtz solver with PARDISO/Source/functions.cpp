#include "definitions.h"
#include "templates.h"
#include "TemplatesForMatrixConstruction.h"
#include "TestSuite.h"

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
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
#pragma omp simd
			for (int i = 0; i < m; i++)
				C[i + ldc * j] = alpha * A[i + lda * j] + beta * B[i + ldb * j];
}

void Clear(int m, int n, dtype* A, int lda)
{
#pragma omp parallel for schedule(runtime)
	for (int j = 0; j < n; j++)
//#pragma omp simd
		for (int i = 0; i < m; i++)
			A[i + lda * j] = 0.0;
}

void NormalizeVector(int size, dtype* v, dtype* out, double &norm)
{
	int ione = 1;
	norm = dznrm2(&size, v, &ione);

	// Compute 
#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < size; i++)
		out[i] = v[i] / norm;
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
		for (int k = 0; k < x.n * y.n; k++)
			f2D[k] = f[z.n * k + w];
#endif
}

void GenSol1DBackward(int w, size_m x, size_m y, size_m z, dtype* x_sol_prd, dtype *u1D)
{
	int l = 0;

	int Nx, Ny, Nz;

	//Nx = x.n - 2 * x.pml_pts;
	//Ny = y.n - 2 * y.pml_pts;
	//Nz = z.n - 2 * z.spg_pts;

	Nx = x.n;
	Ny = y.n;
	Nz = z.n;

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
	for (int k = 0; k < Nz; k++)
			u1D[k] = x_sol_prd[w + k * Nx * Ny];
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
	k = l / (n1 * n2);
	j = (l - k * n1 * n2) / n1;
	i = l - n1 * n2 * k - n1 * j;
}

void take_coord2D(int n1, int n2, int l, int& i, int& j)
{
	j = l / n1;
	i = l - n1 * j;
}

void reducePML3D(size_m x, size_m y, size_m z, int size1, dtype *vect, int size2, dtype *vect_red)
{
	int i = 0, j = 0, k = 0;
	int numb = 0;

	if (size1 == size2)
	{
		printf("There is no PML 3D reduction\n");

#pragma omp parallel for schedule(static)
		for (int i = 0; i < size1; i++)
			vect_red[i] = vect[i];

		return;
	}

	for (int l = 0; l < size1; l++)
	{
		take_coord3D(x.n, y.n, z.n, l, i, j, k);
		if(i >= x.pml_pts && j >= y.pml_pts && k >= z.spg_pts && i < (x.n - x.pml_pts) && j < (y.n - y.pml_pts) && k < (z.n - z.spg_pts)) vect_red[numb++] = vect[l];
	}

	if (numb != size2) printf("ERROR of reducing PML 3D: %d != %d\n", numb, size2);
	else printf("PML 3D is reduced successfully!\n");
}

void reducePML3D_FT(size_m x, size_m y, size_m z, int size1, dtype *vect, int size2, dtype *vect_red)
{
	int i = 0, j = 0, k = 0;
	int numb = 0;
	int size2D = x.n * y.n;

	if (size1 == size2)
	{
		printf("There is no PML 3D reduction after FT\n");

#pragma omp parallel for schedule(static)
		for (int i = 0; i < size1; i++)
			vect_red[i] = vect[i];

		return;
	}

	for (int l = 0; l < size2D; l++)
	{
		take_coord2D(x.n, y.n, l, i, j);
		if (i >= x.pml_pts && j >= y.pml_pts && i < (x.n - x.pml_pts) && j < (y.n - y.pml_pts))
		{
			for (int k = z.spg_pts; k < (z.n - z.spg_pts); k++)
				vect_red[numb++] = vect[l * z.n + k];
		}
	}

	if (numb != size2) printf("ERROR of reducing PML 3D after FT: %d != %d\n", numb, size2);
	else printf("PML 3D after FT is reduced successfully!\n");
}


void extendPML3D(size_m x, size_m y, size_m z, int size1, dtype *vect, int size2, dtype *vect_ext)
{
	int i = 0, j = 0, k = 0;
	int numb = 0;

	if (size1 == size2)
	{
		printf("There is no PML 3D extention\n");

#pragma omp parallel for schedule(static)
		for (int i = 0; i < size1; i++)
			vect_ext[i] = vect[i];

		return;
	}

	for (int l = 0; l < size2; l++)
	{
		take_coord3D(x.n, y.n, z.n, l, i, j, k);
		if (i >= x.pml_pts && j >= y.pml_pts && k >= z.spg_pts && i < (x.n - x.pml_pts) && j < (y.n - y.pml_pts) && k < (z.n - z.spg_pts)) vect_ext[l] = vect[numb++];
		else vect_ext[l] = 0;
	}

	if (numb != size1) printf("ERROR of extension PML 3D: %d != %d\n", numb, size1);
	else printf("PML 3D is extended successfully!\n");
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

void reducePML1D(size_m x, int size1, dtype *vect, int size2, dtype *vect_red)
{
	int i = 0, j = 0, k = 0;
	int numb = 0;

	if (size1 == size2)
	{
		printf("There is no PML 1D reduction\n");

#pragma omp parallel for schedule(static)
		for (int i = 0; i < size1; i++)
			vect_red[i] = vect[i];

		return;
	}

	for (int l = 0; l < size1; l++)
	{
		if (l >= x.pml_pts && l < (x.n - x.pml_pts)) vect_red[numb++] = vect[l];
	}

	if (numb != size2) printf("ERROR of reducing PML 1D: %d != %d\n", numb, size2);
	else printf("PML 1D is reduced successfully!\n");
}

void check_norm_result(int n1, int n2, int n3, dtype* x_orig_nopml, dtype* x_sol_nopml)
{
	printf("------------ACCURACY CHECK---------\n");

	int size2D = n1 * n2;

	double *x_orig_re = alloc_arr<double>(size2D);
	double *x_orig_im = alloc_arr<double>(size2D);

	double *x_sol_re = alloc_arr<double>(size2D);
	double *x_sol_im = alloc_arr<double>(size2D);

	double eps1p = 0.01;

	for (int k = 0; k < n3; k++)
	{
#pragma omp parallel for simd schedule(simd:static)
		for (int i = 0; i < size2D; i++)
		{
			x_orig_re[i] = x_orig_nopml[i + k * size2D].real();
			x_orig_im[i] = x_orig_nopml[i + k * size2D].imag();
			x_sol_re[i] = x_sol_nopml[i + k * size2D].real();
			x_sol_im[i] = x_sol_nopml[i + k * size2D].imag();
		}

		double norm = RelError(zlange, size2D, 1, &x_sol_nopml[k * size2D], &x_orig_nopml[k * size2D], size2D, 0.001);
		double norm_re = rel_error(dlange, size2D, 1, x_sol_re, x_orig_re, size2D, eps1p);
		double norm_im = rel_error(dlange, size2D, 1, x_sol_im, x_orig_im, size2D, eps1p);
		printf("i = %d norm = %lf norm_re = %lf norm_im = %lf\n", k, norm, norm_re, norm_im);
	}

	free_arr(x_sol_re);
	free_arr(x_sol_im);
	free_arr(x_orig_re);
	free_arr(x_orig_im);

}

void check_norm_result2(int n1, int n2, int n3, double ppw, double spg, dtype* x_orig_nopml, dtype* x_sol_nopml,
	double* x_orig_re, double* x_orig_im, double *x_sol_re, double *x_sol_im)
{
	printf("------------ACCURACY CHECK---------\n");

	int size2D = n1 * n2;
	int size = size2D * n3;

	double eps1p = 0.01;
	double *norms_re = alloc_arr<double>(n3);
	double *norms_im = alloc_arr<double>(n3);

#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < size; i++)
	{
		x_orig_re[i] = x_orig_nopml[i].real();
		x_orig_im[i] = x_orig_nopml[i].imag();
		x_sol_re[i] = x_sol_nopml[i].real();
		x_sol_im[i] = x_sol_nopml[i].imag();
	}

	for (int k = 0; k < n3; k++)
	{
		double norm = RelError(zlange, size2D, 1, &x_sol_nopml[k * size2D], &x_orig_nopml[k * size2D], size, 0.001);
		norms_re[k] = RelError(dlange, size2D, 1, &x_sol_re[k * size2D], &x_orig_re[k * size2D], size, eps1p);
		norms_im[k] = RelError(dlange, size2D, 1, &x_sol_im[k * size2D], &x_orig_im[k * size2D], size, eps1p);
		printf("i = %d norm = %lf norm_re = %lf norm_im = %lf\n", k, norm, norms_re[k], norms_im[k]);
	}

	FILE* fout;
	char str[255];
	sprintf(str, "NORMS_N%d_FREQ%d_PPW%4.2lf_SPG%lf.dat", n1, (int)omega, ppw, spg);
	fout = fopen(str, "w");

	for (int k = 0; k < n3; k++)
		fprintf(fout, "%d %lf %lf\n", k, norms_re[k], norms_im[k]);

	fclose(fout);
}

void check_norm_circle(size_m xx, size_m yy, size_m zz, dtype* x_orig, dtype* x_sol, point source, double thresh)
{
	int n1 = xx.n;
	int n2 = yy.n;
	int n3 = zz.n;

	int size2D = n1 * n2;
	int size = size2D * n3;
	
	double x, y, z;
	double r0 = 160;
	double r;
	double r_max = xx.l;
	double norm;

	dtype* x_sol_circ = alloc_arr<dtype>(xx.n * yy.n * zz.n);
	dtype* x_orig_circ = alloc_arr<dtype>(xx.n * yy.n * zz.n);

	for(int k = 0; k < n3; k++)
		for(int j = 0; j < n2; j++)
			for (int i = 0; i < n1; i++)
			{
				x = (i + 1) * xx.h - source.x;
				y = (j + 1) * yy.h - source.y;
				z = (k + 1) * zz.h - source.z;
				r = sqrt(x * x + y * y + z * z);
				if (r >= r0 || r <= r_max)
				{
					x_sol_circ[i + n1 * j + size2D * k] = x_sol[i + n1 * j + size2D * k];
					x_orig_circ[i + n1 * j + size2D * k] = x_orig[i + n1 * j + size2D * k];
				}
			}

	norm = RelError(zlange, size, 1, x_sol_circ, x_orig_circ, size, thresh);

	printf("Norm in circle: r0 < r < r_max: %lf\n", norm);
}


void GenRHSandSolution(size_m x, size_m y, size_m z, /* output */ dtype *u, dtype *f, point source, int &src)
{
	int n = x.n * y.n;
	int size = n * z.n;


	//printf("SOURCE AT x = %lf, y = %lf, z = %lf\n", source.x, source.y, source.z);

	// approximation of exact right hand side (inner grid points)
#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < z.n; k++)
		for (int j = 0; j < y.n; j++)
//#pragma omp simd
			for (int i = 0; i < x.n; i++)
			{
				f[k * n + j * x.n + i] = F3D_ex_complex(x, y, z, (i + 1) * x.h, (j + 1) * y.h, (k + 1) * z.h, source, src);
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

void GenRHSandSolutionViaSound3D(size_m x, size_m y, size_m z, /* output */ dtype *u, dtype *f, point source)
{
	int n = x.n * y.n;
	int size = n * z.n;
	int l = 0;

	SetRHS3D(x, y, z, f, source, l);

	// approximation of inner points values
#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < z.n; k++)
		for (int j = 0; j < y.n; j++)
			for (int i = 0; i < x.n; i++)
				u[k * n + j * x.n + i] = u_ex_complex_sound3D(x, y, z, (i + 1) * x.h, (j + 1) * y.h, (k + 1) * z.h, source);


	printf("RHS and solution are constructed\n");
}

double d(double x)
{
	//printf("x = %lf\n", x);

	const double C = 100;
	return C * pow(x, 2);

	// 10 * x ^ 2, проверить , что значения 0 и С на границах
	// - 15 h < x < 0
	// кси , n !=  (1, 0)
}

dtype MakeSound2D(size_m xx, size_m yy, double x, double y, point source)
{
	return MakeSound2D(xx, yy, x, y, source) * sqrt(dtype{ 1, -beta_eq } / (1 + beta_eq * beta_eq));
}

dtype MakeSound3D(size_m xx, size_m yy, size_m zz, double x, double y, double z, point source)
{
	return c_z;
}

void SetSoundSpeed3D(size_m x, size_m y, size_m z, dtype* sound3D, point source)
{
	int n = x.n * y.n;
	int Nx = x.n;
	int Ny = y.n;
	int Nz = z.n;
	printf("z.spg_pts = %d\n", z.spg_pts);

	for (int k = 0; k < Nz; k++)
	{
		if (k >= z.spg_pts && k < (Nz - z.spg_pts))
		{
			for (int j = 0; j < Ny; j++)
				for (int i = 0; i < Nx; i++)
					sound3D[k * n + j * Nx + i] = MakeSound3D(x, y, z, (i + 1) * x.h, (j + 1) * y.h, (k + 1) * z.h, source);
		}
		else
		{
			for (int j = 0; j < Ny; j++)
				for (int i = 0; i < Nx; i++)
					sound3D[k * n + j * Nx + i] = MakeSound3D(x, y, z, (i + 1) * x.h, (j + 1) * y.h, (k + 1) * z.h, source) * sqrt(dtype{ 1, -beta_eq } / (1 + beta_eq * beta_eq));
		}
	}
}

void SetSoundSpeed2D(size_m x, size_m y, size_m z, dtype* sound3D, dtype* sound2D, point source)
{
	int n = x.n * y.n;
	int Nx = x.n;
	int Ny = y.n;
	int Nz = z.n;
	dtype c0_max;
	dtype c0_min;

#if 1
	for (int j = 0; j < Ny; j++)
		for (int i = 0; i < Nx; i++)
		{
			c0_max = sound3D[z.spg_pts * n + j * Nx + i];
			c0_min = sound3D[z.spg_pts * n + j * Nx + i];

			for (int k = 0; k < Nz; k++)
			{
				if (k >= z.spg_pts && k < (Nz - z.spg_pts))
				{
					if (abs(sound3D[k * n + j * Nx + i]) > abs(c0_max)) c0_max = sound3D[k * n + j * Nx + i];
					if (abs(sound3D[k * n + j * Nx + i]) < abs(c0_min)) c0_min = sound3D[k * n + j * Nx + i];
				}
			}

			sound2D[j * Nx + i] = 0.5 * (c0_max + c0_min);
		}
#else
#pragma omp parallel for schedule(dynamic)
		for (int j = 0; j < Ny; j++)
			for (int i = 0; i < Nx; i++)
				sound2D[j * Nx + i] = MakeSound2D(x, y, (i + 1) * x.h, (j + 1) * y.h, source);
#endif
	
}

void GenerateDeltaL(size_m x, size_m y, size_m z, dtype* sound3D, dtype* sound2D, dtype* deltaL)
{
	int size2D = x.n * y.n;
	int nx = x.n;
	int ny = y.n;
	int nz = z.n;
	int ijk;
	int ij;

#if 0

#ifndef TRIVIAL
	for (int k = 0; k < nz; k++)
	{
		// k(x,y,z) - k(x,y) * (1 - i * beta)  - inside D'
		if (k >= z.spg_pts || k < (nz - z.spg_pts))
		{
			for (int j = 0; j < ny; j++)
				for (int i = 0; i < nx; i++)
				{
					ij = i + j * nx;
					ijk = ij + k * size2D;
					deltaL[ijk] = omega * omega * ( dtype{ 1.0, beta_eq } / (sound2D[ij] * sound2D[ij]) -  1.0 / (sound3D[ijk] * sound3D[ijk]));
				}
		}
		else // sponge zone: ( k(x,y,z) - k(x,y) ) * (1 + i * beta)
		{
			for (int j = 0; j < ny; j++)
				for (int i = 0; i < nx; i++)
				{
					ij = i + j * nx;
					ijk = ij + k * size2D;
					deltaL[ijk] = omega * omega * (1.0 / (sound3D[ijk] * sound3D[ijk]) - 1.0 / (sound2D[ij] * sound2D[ij])) * dtype{ 1.0, -beta_eq };
				}
		}
	}
#else
	for (int k = 0; k < nz; k++)
		for(int j = 0; j < ny; j++)
			for (int i = 0; i < nx; i++)
			{
				ij = i + j * nx;
				ijk = ij + k * size2D;
				deltaL[ijk] = 0;
			}
#endif
#endif

	for (int k = 0; k < nz; k++)
		for (int j = 0; j < ny; j++)
			for (int i = 0; i < nx; i++)
			{
				ij = i + j * nx;
				ijk = ij + k * size2D;
				deltaL[ijk] = omega * omega * (dtype{ 1.0, beta_eq } / (sound2D[ij] * sound2D[ij]) - 1.0 / (sound3D[ijk] * sound3D[ijk]));
			}


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

dtype alpha(size_m xyz, double i)
{
	double x = 0;
	double h = 1.0 / xyz.pml_pts;

	if (i < 0 || i > xyz.n - 1)
	{
		// bound case
		x = 1.0 - h / 2.0;
		return dtype{ 0, -omega } / dtype{ d(x), -omega };
	}
	else if (i < xyz.pml_pts || i >= (xyz.n - xyz.pml_pts))
	{
		if (i < xyz.pml_pts)
		{
			x = (double)(xyz.pml_pts - i) * h;

		//	if ((abs(x - 0) < EPS_ZERO) || (abs(x - 1) < EPS_ZERO)) printf("left: i = %d x = %lf\n", i, x);
		}
		else if (i >= (xyz.n - xyz.pml_pts))
		{
			//x = (double)(i - xyz.n + xyz.pml_pts + 1) / (xyz.pml_pts);
			x = (double)(i - xyz.n + xyz.pml_pts + 1) * h;
			
			//if ((abs(x - 0) < EPS_ZERO) || (abs(x - 1) < EPS_ZERO)) printf("right: i = %d x = %lf\n", i, x);
		}

		return dtype{ 0, -omega } / dtype{ d(x), -omega };
	}
	else return 1.0;
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
		if (blk3D < z.spg_pts || blk3D >= (z.n - z.spg_pts))
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

		if (blk3D < z.spg_pts || blk3D >= (z.n - z.spg_pts))
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

dtype Hankel(double x)
{
	return { j0(x), y0(x) };
}

dtype Hankel(dtype z)
{
	void *vmblock = NULL;

	int n = 1; // number of functions
	int nz;

	//memory allocation for cyr, cyi, cwr, cwi
	vmblock = vminit();
	REAL *res_real = (REAL *)vmalloc(vmblock, VEKTOR, n + 1, 0); //index 0 not used
	REAL *res_imag = (REAL *)vmalloc(vmblock, VEKTOR, n + 1, 0);
	REAL *CWRKR = (REAL *)vmalloc(vmblock, VEKTOR, n + 1, 0);
	REAL *CWRKI = (REAL *)vmalloc(vmblock, VEKTOR, n + 1, 0);

	if (!vmcomplete(vmblock)) {
		LogError("No Memory", 0, __FILE__, __LINE__);
		return 0;
	}

	int fnu = 0; // first function
	int kode = 1; // no scaling
	int ierr = 0;
	int m = 1; // kind of hankel function

#if 0

	ZBESJ(z.real(), z.imag(), fnu, kode, n, res_real, res_imag, &nz, &IERR);
	printf("ZBESJ IERROR: %d\n", IERR);
	dtype res1 = { res_real[1], res_imag[1] };

	ZBESY(z.real(), z.imag(), fnu, kode, n, res_real, res_imag, &nz, CWRKR, CWRKI, &IERR);
	printf("ZBESY IERROR: %d\n", IERR);
	dtype res2 = { res_real[1], res_imag[1] };
	res2 = { -res2.imag(), res2.real() };

	return res1 + res2;
#else

	// Hankel function of the 1 kind
	ZBESH(z.real(), z.imag(), fnu, kode, m, n, res_real, res_imag, &nz, &ierr);

	return { res_real[1], res_imag[1] };

#endif
}

dtype set_exact_2D_Hankel(double x, double y, dtype k, point source)
{
	x -= source.x;
	y -= source.y;

	double r = sqrt(x * x + y * y);

	if (r == 0) r = 0.005;


	return Hankel(k * r);
}

void get_exact_2D_Hankel(int Nx, int Ny, size_m x, size_m y, dtype* x_sol_ex, dtype k, point source)
{
	for (int j = 0; j < Ny; j++)
		for (int i = 0; i < Nx; i++)
		{
			x_sol_ex[j * Nx + i] = set_exact_2D_Hankel((i + 1) * x.h, (j + 1) * y.h, k, source);

			x_sol_ex[j * Nx + i] *= -0.25;
			x_sol_ex[j * Nx + i] = dtype{ -x_sol_ex[j * Nx + i].imag(), x_sol_ex[j * Nx + i].real() };
		}
}

void normalization_of_exact_sol(int n1, int n2, size_m x, size_m y, dtype *x_sol_ex, dtype alpha_k)
{
	for (int i = 0; i < n1 * n2; i++)
		x_sol_ex[i] *= alpha_k;
}

void check_exact_sol_Hankel(dtype alpha_k, double k2, size_m x, size_m y, dtype* x_sol_prd, double eps)
{
	char str1[255], str2[255];

	int Nx = x.n - 2 * x.pml_pts;
	int Ny = y.n - 2 * y.pml_pts;


	double l_nopml = (double)LENGTH;
	double eps1p = 0.01;
	bool pml_flag = true;

	sprintf(str1, "Charts2D/model_ft_kwave2_%lf", k2);

	
		int size = Nx * Ny;
		dtype* x_sol_ex = alloc_arr<dtype>(size);
		dtype* x_sol_cpy = alloc_arr<dtype>(size);

		double *x_sol_ex_re = alloc_arr<double>(size);
		double *x_sol_prd_re = alloc_arr<double>(size);

		double *x_sol_ex_im = alloc_arr<double>(size);
		double *x_sol_prd_im = alloc_arr<double>(size);
		int ione = 1;

		int i1, j1;
		
		dtype k = 0;

		if (k2 > 0) k = sqrt(k2);
		else k = { 0 , sqrt(abs(k2)) };

		point source = { l_nopml / 2.0, l_nopml / 2.0 };

		printf("SOURCE AT 2D PROBLEM AT x = %lf y = %lf\n", source.x, source.y);

		get_exact_2D_Hankel(Nx, Ny, x, y, x_sol_ex, k, source);


		//x_sol_ex[Nx * Ny / 2 - Nx / 2] = x_sol_prd[Nx * Ny / 2 - Nx / 2] = 0;

		normalization_of_exact_sol(Nx, Ny, x, y, x_sol_ex, alpha_k);

		zlacpy("All", &size, &ione, x_sol_prd, &size, x_sol_cpy, &size);

		for (int l = 0; l < size; l++)
		{
			take_coord2D(Nx, Ny, l, i1, j1);
			if (i1 == j1 && ((i1 + 1) * y.h == source.y))
			{
				//printf("i = j = %d val = %lf %lf\n", i1, j1, x_sol_ex[l].real(), x_sol_ex[l].imag());
				x_sol_cpy[l] = x_sol_prd[l] = x_sol_ex[l] = 0;
			}
		}

		double norm = rel_error(zlange, size, 1, x_sol_cpy, x_sol_ex, size, eps1p);

		for (int i = 0; i < size; i++)
		{
			x_sol_ex_re[i] = x_sol_ex[i].real();
			x_sol_ex_im[i] = x_sol_ex[i].imag();
			x_sol_prd_re[i] = x_sol_prd[i].real();
			x_sol_prd_im[i] = x_sol_prd[i].imag();
		}

		double norm_re = rel_error(dlange, size, 1, x_sol_prd_re, x_sol_ex_re, size, eps1p);
		double norm_im = rel_error(dlange, size, 1, x_sol_prd_im, x_sol_ex_im, size, eps1p);

		if (k2 > 0)
		{
			printf("k2 = %lf > 0, CHECK H0(kr): \n", k2);
			if (norm < eps1p) printf("Norm %12.10e < eps %12.10lf: PASSED  ", norm, eps1p);
			else printf("Norm %12.10lf > eps %12.10lf : FAILED  ", norm, eps1p);
			printf("norm_re: %12.10lf, norm_im: %12.10lf\n", norm_re, norm_im);

			sprintf(str2, "Charts2D/model_ex_2D_kwave2_%lf", k2);
			output2D(str1, pml_flag, x, y, x_sol_ex, x_sol_prd);
			gnuplot2D(str1, str2, pml_flag, 3, x, y);

			sprintf(str2, "Charts2D/model_prd_2D_kwave2_%lf", k2);
			gnuplot2D(str1, str2, pml_flag, 5, x, y);
		}
		else
		{
			printf("k2 = %lf < 0, CHECK H0(i * kr): \n", k2);
			if (norm < eps1p) printf("Norm %12.10e < eps %12.10lf: PASSED  ", norm, eps1p);
			else printf("Norm %12.10lf > eps %12.10lf : FAILED  ", norm, eps1p);
			printf("norm_re: %12.10lf, norm_im: %12.10lf\n", norm_re, norm_im);
			//printf("NO CHECK: k2 < 0\n");
		}


		free_arr(x_sol_ex);
		free_arr(x_sol_cpy);
		free_arr(x_sol_ex_re);
		free_arr(x_sol_ex_im);
		free_arr(x_sol_prd_re);
		free_arr(x_sol_prd_im);
}

double resid_2D_Hankel(size_m x, size_m y, ccsr* D2csr, dtype* x_sol_ex, dtype* f2D, point source)
{
	int n = x.n;
	int size = n * y.n;
	double RelRes = 0;
	dtype *g = alloc_arr<dtype>(size);

	ResidCSR2D(x, y, D2csr, x_sol_ex, f2D, g, source, RelRes);

	free_arr(g);
	return RelRes;
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
		if (blk3D < z.spg_pts || blk3D >= (z.n - z.spg_pts))
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

dtype beta3D(size_m x, size_m y, size_m z, int diag_case, int i, int j, int k)
{
	if (diag_case == 0)
	{
		dtype value;

#ifdef PML
	//	value = -alpha(x, i) * (alpha(x, i + 1) + 2.0 * alpha(x, i) + alpha(x, i - 1)) / (2.0 * x.h * x.h) 
	//		    -alpha(y, j) * (alpha(y, j + 1) + 2.0 * alpha(y, j) + alpha(y, j - 1)) / (2.0 * y.h * y.h)
	//		    -alpha(z, k) * (alpha(z, k + 1) + 2.0 * alpha(z, k) + alpha(z, k - 1)) / (2.0 * z.h * z.h);

		value = -alpha(x, i) * (alpha(x, i + 0.5) + alpha(x, i - 0.5)) / (x.h * x.h)
			    -alpha(y, j) * (alpha(y, j + 0.5) + alpha(y, j - 0.5)) / (y.h * y.h)
				-alpha(z, k) * (alpha(z, k + 0.5) + alpha(z, k - 0.5)) / (z.h * z.h);
#else
		dtype c1 = -2.0 / (x.h * x.h);
		dtype c2 = -2.0 / (y.h * y.h);
		dtype c3 = -2.0 / (z.h * z.h);
		if (i == 0 || i == x.n - 1) c1 = dtype{ -1.0 / (x.h * x.h), -(double)kk / x.h };
		if (j == 0 || j == y.n - 1) c2 = dtype{ -1.0 / (y.h * y.h), -(double)kk / y.h };
		if (k == 0 || k == z.n - 1) c3 = dtype{ -1.0 / (z.h * z.h), -(double)kk / z.h };
		
		value = c1 + c2 + c3;

	/*	if (i == 0 || i == x.n - 1 || j == 0 || j == y.n - 1 || k == 0 || k == z.n - 1)
		{
			printf("I = %d J = %d K = %d\n", i, j, k);
		}*/
#endif 

		int l = k * x.n * y.n + j * x.n + i;

		//printf("l = %d : i = %d j = %d k = %d value = %lf %lf\n", l, i, j, k, value.real(), value.imag());

		return value;
	}
	else if (diag_case == 1)
	{
		return alpha(x, i) * alpha(x, i + 0.5) / (x.h * x.h);
	}
	else if (diag_case == -1)
	{
		return alpha(x, i) * alpha(x, i - 0.5) / (x.h * x.h);
	}
	else if (diag_case == 2)
	{
		return alpha(y, j) * alpha(y, j + 0.5) / (y.h * y.h);
	}
	else if (diag_case == -2)
	{
		return alpha(y, j) * alpha(y, j - 0.5) / (y.h * y.h);
	}
	else if (diag_case == 3)
	{
		return alpha(z, k) * alpha(z, k + 0.5) / (z.h * z.h);
	}
	else if (diag_case == -3)
	{
		return alpha(z, k) * alpha(z, k - 0.5) / (z.h * z.h);
	}
	return 0;
}

dtype beta1D(size_m x, int diag_case, double k2, int i)
{
	if (diag_case == 0)
	{
		dtype value;

#if 0
		value = -alpha(x, i) * (alpha(x, i + 0.5) + alpha(x, i - 0.5)) / (x.h * x.h);
#else
		// sponge
		if (i >= x.pml_pts && i < (x.n - x.pml_pts))
		{
			value = -2.0 / (x.h * x.h) + k2;
		}
		else
		{
			value = -2.0 / (x.h * x.h) + dtype{ 1, beta_eq } * k2;
		}
#endif 

		return value;
	}
	else if (diag_case == 1)
	{
#if 0
		return alpha(x, i) * alpha(x, i + 0.5) / (x.h * x.h);
#else
		return 1.0 / (x.h * x.h);
#endif
	}
	else if (diag_case == -1)
	{
#if 0
		return alpha(x, i) * alpha(x, i - 0.5) / (x.h * x.h);
#else
		return 1.0 / (x.h * x.h);
#endif
	}

	return 0;
}

dtype beta2D_pml(size_m x, size_m y, int diag_case, dtype kwave_beta2, int i, int j)
{
	if (diag_case == 0)
	{
		dtype value;

#ifdef PML
	//	value = -alpha(x, i) * (alpha(x, i + 1) + 2.0 * alpha(x, i) + alpha(x, i - 1)) / (2.0 * x.h * x.h)
	//		    -alpha(y, j) * (alpha(y, j + 1) + 2.0 * alpha(y, j) + alpha(y, j - 1)) / (2.0 * y.h * y.h);

		value = -alpha(x, i) * (alpha(x, i + 0.5) + alpha(x, i - 0.5)) / (x.h * x.h)
				-alpha(y, j) * (alpha(y, j + 0.5) + alpha(y, j - 0.5)) / (y.h * y.h);

		value += kwave_beta2;

#else 
		dtype c1 = -2.0 / (x.h * x.h);
		dtype c2 = -2.0 / (y.h * y.h);
		if (i == 0 || i == x.n - 1) c1 = dtype{ -1.0 / (x.h * x.h), -(double)kk / x.h };
		if (j == 0 || j == y.n - 1) c2 = dtype{ -1.0 / (y.h * y.h), -(double)kk / y.h };

		value = c1 + c2;
#endif

		int l = j * x.n + i;

		//printf("l = %d : i = %d j = %d k = %d value = %lf %lf\n", l, i, j, k, value.real(), value.imag());

		return value;
	}
	else if (diag_case == 1)
	{
		//return alpha(x, i) * (alpha(x, i + 1) + alpha(x, i)) / (2.0 * x.h * x.h);

		return alpha(x, i) * alpha(x, i + 0.5) / (x.h * x.h);
	}
	else if (diag_case == -1)
	{
		return alpha(x, i) * alpha(x, i - 0.5) / (x.h * x.h);
	}
	else if (diag_case == 2)
	{
		//return alpha(y, j) * (alpha(y, j + 1) + alpha(y, j)) / (2.0 * y.h * y.h);

		return alpha(y, j) * alpha(y, j + 0.5) / (y.h * y.h);
	}
	else if (diag_case == -2)
	{
		return alpha(y, j) * alpha(y, j - 0.5) / (y.h * y.h);
	}

	return 0;
}

dtype beta2D_spg(size_m x, size_m y, int diag_case, double k2, int i, int j)
{
	if (diag_case == 0)
	{
		dtype value;

		// sponge
		if (i >= x.pml_pts && j >= y.pml_pts && i < (x.n - x.pml_pts) && j < (y.n - y.pml_pts))
		{
			value = -2.0 / (x.h * x.h) - 2.0 / (y.h * y.h) + k2;
		}
		else
		{
			value = -2.0 / (x.h * x.h) - 2.0 / (y.h * y.h) + k2 * dtype{ 1, beta_eq };
		}

		return value;
	}
	else if (diag_case == 1)
	{
		return 1.0 / (x.h * x.h);
	}
	else if (diag_case == -1)
	{
		return 1.0 / (x.h * x.h);
	}
	else if (diag_case == 2)
	{
		return 1.0 / (y.h * y.h);
	}
	else if (diag_case == -2)
	{
		return 1.0 / (y.h * y.h);
	}

	return 0;
}

void FGMRES(size_m x, size_m y, size_m z, const point source, dtype *x_sol, const dtype *f, double thresh)
{
	printf("-------------FGMRES-----------\n");

	// We need to solve iteratively: (I - \delta L * L_0^{-1})w = g

	int size = x.n * y.n * z.n;
	int size2D = x.n * y.n;
	int size_nopml = x.n_nopml * y.n_nopml * z.n_nopml;
	int m = 2;
	int iterCount = m;
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

	dtype *deltaL = alloc_arr<dtype>(size);
	dtype *sound3D = alloc_arr<dtype>(size);
	dtype *sound2D = alloc_arr<dtype>(size2D);
	dtype* work;

	printf("-----Step 0. Set sound speed and deltaL-----\n");
	//	SetSoundSpeed3D(x, y, z, sound3D, source);
	//	SetSoundSpeed2D(x, y, z, sound3D, sound2D, source);
#if 1
	// Gen velocity of sound in 3D domain
	SetSoundSpeed3D(x, y, z, sound3D, source);

	// Gen velocity of sound in 3D domain
	SetSoundSpeed2D(x, y, z, sound3D, sound2D, source);

	char str1[255] = "sound_speed2D";
	//output(str1, false, x, y, z, sound3D, deltaL);
	//output2D(str1, false, x, y, sound2D, sound2D);

	// Gen DeltaL function
	GenerateDeltaL(x, y, z, sound3D, sound2D, deltaL);

	free_arr(sound2D);
	free_arr(sound3D);

	char str2[255] = "sound_speed_deltaL";
	//output(str2, false, x, y, z, sound3D, deltaL);
	
	printf("-----Step 0. Memory allocation-----\n");
	// init cond
	dtype *x0 = alloc_arr<dtype>(size);
	dtype *x_init = alloc_arr<dtype>(size);

	// matrix of Krylov basis
	dtype* V = alloc_arr<dtype>(size * (m + 1)); int ldv = size;
	dtype* w = alloc_arr<dtype>(size);

	// residual vector
	dtype *r0 = alloc_arr<dtype>(size);

	// additional vector
	dtype *Ax0 = alloc_arr<dtype>(size);
	dtype *Ax0_nopml = alloc_arr<dtype>(size_nopml);

	// Hessenberg matrix
	dtype *H = alloc_arr<dtype>((m + 1) * m); int ldh = m + 1;
	dtype *Hgels = alloc_arr<dtype>((m + 1) * m);

	// the vector of right-hand side for the system with Hessenberg matrix
	dtype *eBeta = alloc_arr<dtype>(m + 1); int ldb = m + 1;

	// vars
	dtype calpha;
	double dalpha;

#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < size; i++)
		x_init[i] = 0;

	for (int restart = 0; restart < 1; restart++)
	{
		printf("------RESTART = %d------\n", restart);
		// 1. First step. Compute r_0 and its norm
		printf("-----Step 1-----\n");

		zcopy(&size, x_init, &ione, x0, &ione);

		// Multiply matrix A in CSR format by vector x_0 to obtain f1
		ApplyCoeffMatrixA(x, y, z, x0, deltaL, Ax0, thresh);

		norm = dznrm2(&size, f, &ione);
		printf("norm ||f|| = %lf\n", norm);

		//Add_dense(size, ione, 1.0, f, size, -1.0, Ax0, size, r0, size);
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
			// Compute w[j] := A * v[j]
			ApplyCoeffMatrixA(x, y, z, &V[ldv * j], deltaL, w, thresh);

			for (int i = 0; i <= j; i++)
			{
				// H[i + ldh * j] = (w_j * v_i) 
				zdotc(&H[i + ldh * j], &size, w, &ione, &V[ldv * i], &ione);
				//H[i + ldh * j] = zdot(size, w, &V[ldv * i]);

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
			//MultVectorConst(size, w, 1.0 / H[j + 1 + ldh * j], &V[ldv * (j + 1)]);
			zcopy(&size, w, &ione, &V[ldv * (j + 1)], &ione);
			dalpha = 1.0 / H[j + 1 + ldh * j].real();
			zdscal(&size, &dalpha, &V[ldv * (j + 1)], &ione);

			TestNormalizedVector(size, &V[ldv * (j + 1)], thresh);
			for (int i = 0; i <= j; i++)
			{
				TestOrtogonalizedVectors(size, &V[ldv * (j + 1)], &V[ldv * i], thresh);
			}
		}

			{
				// 3. Solving least squares problem to compute y_k
				// for x_k = x_0 + V_k * y_k
				printf("-----Step 3. LS problem-----\n");

				printf("size of basis: %d\n", iterCount);

				// Set eBeta
#pragma omp parallel for simd schedule(simd:static)
				for (int i = 0; i < m + 1; i++)
					eBeta[i] = 0;

				eBeta[0] = beta;

				// Set working H because it is destroyed after GELS
				for (int i = 0; i < m * (m + 1); i++)
					Hgels[i] = H[i];

				// Query
				lwork = -1;
				//row_min = j + 2;
				//col_min = j + 1;

				row_min = m + 1;
				col_min = m;

				zgels("no", &row_min, &col_min, &nrhs, Hgels, &ldh, eBeta, &ldb, &work_size, &lwork, &info);

				lwork = (int)work_size.real();
				work = alloc_arr<dtype>(lwork);
				// Run
				zgels("no", &row_min, &col_min, &nrhs, Hgels, &ldh, eBeta, &ldb, work, &lwork, &info);
				free_arr(work);

				// 4. Multiplication x_k = x_0 + V_k * y_k
				printf("-----Step 4. Computing x_k-----\n");

				zgemv("no", &size, &col_min, &done, V, &ldv, eBeta, &ione, &done, x0, &ione);

				// 5. Check |(I - deltaL * L^{-1}) * x_k - f|

				ApplyCoeffMatrixA(x, y, z, x0, deltaL, Ax0, thresh);
				zaxpy(&size, &mone, f, &ione, Ax0, &ione); // Ax0: = Ax0 - f

				RelRes = dznrm2(&size, Ax0, &ione);
				printf("-----------\n");
				printf("Residual in 3D with PML |(I - deltaL * L^{-1}) * x_sol - f| = %lf\n", RelRes);
				printf("-----------\n");

				reducePML3D(x, y, z, size, Ax0, size_nopml, Ax0_nopml);

				RelRes = dznrm2(&size_nopml, Ax0_nopml, &ione);

				printf("-----------\n");
				printf("Residual in 3D phys domain |(I - deltaL * L^{-1}) * x_sol - f| = %e\n", RelRes);
				printf("-----------\n");


				// 6. Solve L_0 * x_sol = x_gmres
				printf("-----Step 5. Solve the last system-----\n");
				Solve3DSparseUsingFT(x, y, z, x0, x_sol, thresh);
			}

			// For the next step
			zcopy(&size, x_init, &ione, x0, &ione);

#endif

			dtype *f_rsd = alloc_arr<dtype>(size);
			dtype *f_rsd_nopml = alloc_arr<dtype>(size_nopml);

			ComputeResidual(x, y, z, (double)kk, x_sol, f, f_rsd, RelRes);

			printf("-----------\n");
			printf("Residual in 3D with PML |A * x_sol - f| = %e\n", RelRes);
			printf("-----------\n");

			reducePML3D(x, y, z, size, f_rsd, size_nopml, f_rsd_nopml);

			RelRes = dznrm2(&size_nopml, f_rsd_nopml, &ione);

			printf("-----------\n");
			printf("Residual in 3D phys domain |A * x_sol - f| = %e\n", RelRes);
			printf("-----------\n");

#ifdef OUTPUT
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

			free_arr(f_rsd);
			free_arr(f_rsd_nopml);
		

		zcopy(&size, x_sol, &ione, x_init, &ione);
	}

	free_arr(H);
	free_arr(w);
	free_arr(V);
	free_arr(Ax0);
	free_arr(r0);
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

#ifdef GEN_BLOCK_CSR
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
	int count_bound = 0;
	bool right_value = 0;

	for (int l1 = 0; l1 < size; l1++)
	{
		printf("l1 = %d\n", l1);
		Acsr->ia[l1] = count + 1;
		for (int l2 = 0; l2 < size; l2++)
		{
			take_coord3D(x.n, y.n, z.n, l1, i1, j1, k1);
			take_coord3D(x.n, y.n, z.n, l2, i2, j2, k2);

			if (l1 == l2)
			{
				Acsr->ja[count] = l2 + 1;
#ifdef HELMHOLTZ
				Acsr->values[count] = beta3D(x, y, z, 0, i1, j1, k1);
				Acsr->values[count++] += dtype{ k * k, 0 };
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
				right_value = true;
			}
			else if (l1 == l2 + x.n * y.n)
			{	
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta3D(x, y, z, -3, i1, j1, k1);
			}

			if (right_value == true) break;
		}
		right_value = false;

	//	if ((i1 == 0 || j1 == 0 || k1 == 0 || i1 == x.n - 1 || j1 == y.n -1 || k1 == z.n - 1) && Acsr->ja[count - 1] != 0) count_bound++;
	}

//	printf("Count bound: %d\n", count_bound);
	
	if (non_zeros_in_3Dblock3diag != count) printf("Failed generation!");
	else printf("Successed generation!\n");
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

void GenSparseMatrixOnline2DwithPML(int w, size_m x, size_m y, size_m z, ccsr* Acsr, dtype kwave_beta2, int* freqs)
{
	int size = x.n * y.n;
	int size2 = size - y.n;
	int size3 = size - y.n;
	int non_zeros_in_2Dblock3diag = size + size2 * 2 + size3 * 2;
	double RelRes = 0;
	int j1, k1;
	int j2, k2;

	int count = 0;
	int count2 = 0;


	//double k = (double)kk;
	//double kww = 4.0 * PI * PI * (w - n2) * (w - n2) / (y.l * y.l);
	//double kww = 4 * PI * PI * (w - n2) * (w - n2);

	//printf("Number k = %lf\n", k);

	//printf("analytic non_zeros in PML function: %d\n", non_zeros_in_2Dblock3diag);


	if (non_zeros_in_2Dblock3diag != Acsr->non_zeros) printf("ERROR! Uncorrect value of non_zeros inside 2D: %d != %d\n", non_zeros_in_2Dblock3diag, Acsr->non_zeros);
//	else printf("Gen 2D matrix for frequency w = %d, k^2 - ky^2 = (%lf %lf)\n", w, kwave_beta2.real(), kwave_beta2.imag());

	if (w == -1)
	{

		for (int l1 = 0; l1 < size; l1++)
		{
			Acsr->ia[l1] = count + 1;
			for (int l2 = 0; l2 < size; l2++)
			{
				take_coord2D(x.n, y.n, l1, j1, k1);
				take_coord2D(x.n, y.n, l2, j2, k2);

				if (l1 == l2)
				{
					Acsr->ja[count] = l2 + 1;
					freqs[count2++] = count;
#ifdef HELMHOLTZ
					//Acsr->values[count] = dtype{ kwave2, 0 };
					//Acsr->values[count] += dtype{ 0, k * k * beta_eq };
					//Acsr->values[count] = dtype{ k * k, 0 };
					//Acsr->values[count] -= dtype{ kww, 0 };
					Acsr->values[count++] = beta2D_pml(x, y, 0, kwave_beta2, j1, k1);
#else
					Acsr->values[count++] = beta2D(x, y, 0, j1, k1);
#endif

				}
				else if (l1 == l2 - 1 && (l1 + 1) % x.n != 0)
				{
					Acsr->ja[count] = l2 + 1;
					Acsr->values[count++] = beta2D_pml(x, y, 1, kwave_beta2, j1, k1); // right
				}
				else if (l1 == l2 + 1 && l1 % x.n != 0)
				{
					Acsr->ja[count] = l2 + 1;
					Acsr->values[count++] = beta2D_pml(x, y, -1, kwave_beta2, j1, k1); // left
				}
				else if (l1 == l2 - x.n)
				{
					Acsr->ja[count] = l2 + 1;
					Acsr->values[count++] = beta2D_pml(x, y, 2, kwave_beta2, j1, k1); // right
				}
				else if (l1 == l2 + x.n)
				{
					Acsr->ja[count] = l2 + 1;
					Acsr->values[count++] = beta2D_pml(x, y, -2, kwave_beta2, j1, k1); // left
				}

			}
		}

		if (non_zeros_in_2Dblock3diag != count || size != count2) printf("FAILED generation!!! %d != %d\n", non_zeros_in_2Dblock3diag, count);
		else
		{
			printf("SUCCESSED BASIC 2D generation!\n");
			printf("Non_zeros inside generating PML function: %d\n", count);
		}
	}
	else
	{
#pragma omp parallel for simd schedule(simd:static)
		for (int i = 0; i < size; i++)
			Acsr->values[freqs[i]] += kwave_beta2;
	
		//printf("SUCCESSED 2D generation for frequency %d!\n", w);
	}

}

void GenSparseMatrixOnline2DwithSPONGE(int w, size_m x, size_m y, size_m z, ccsr* Acsr, double kwave2)
{
	int size = x.n * y.n;
	int size2 = size - y.n;
	int size3 = size - y.n;
	int non_zeros_in_2Dblock3diag = size + size2 * 2 + size3 * 2;
	double RelRes = 0;
	//double k = (double)kk;
	//double kww = 4.0 * PI * PI * (w - n2) * (w - n2) / (y.l * y.l);
	//double kww = 4 * PI * PI * (w - n2) * (w - n2);

	//printf("Number k = %lf\n", k);

	//printf("analytic non_zeros in PML function: %d\n", non_zeros_in_2Dblock3diag);


	if (non_zeros_in_2Dblock3diag != Acsr->non_zeros) printf("ERROR! Uncorrect value of non_zeros inside 2D: %d != %d\n", non_zeros_in_2Dblock3diag, Acsr->non_zeros);

	printf("Gen 2D matrix for frequency w = %d, k^2 - ky^2 = %lf\n", w - z.n / 2, kwave2);

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
			take_coord2D(x.n, y.n, l1, j1, k1);
			take_coord2D(x.n, y.n, l2, j2, k2);

			if (l1 == l2)
			{
				Acsr->ja[count] = l2 + 1;
#ifdef HELMHOLTZ
				//Acsr->values[count] = dtype{ kwave2, 0 };
				//Acsr->values[count] += dtype{ 0, k * k * beta_eq };
				//Acsr->values[count] = dtype{ k * k, 0 };
				//Acsr->values[count] -= dtype{ kww, 0 };
				Acsr->values[count++] = beta2D_spg(x, y, 0, kwave2, j1, k1);
#else
				Acsr->values[count++] = beta2D(x, y, 0, j1, k1);
#endif

			}
			else if (l1 == l2 - 1 && (l1 + 1) % x.n != 0)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta2D_spg(x, y, 1, kwave2, j1, k1); // right
			}
			else if (l1 == l2 + 1 && l1 % x.n != 0)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta2D_spg(x, y, -1, kwave2, j1, k1); // left
			}
			else if (l1 == l2 - x.n)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta2D_spg(x, y, 2, kwave2, j1, k1); // right
			}
			else if (l1 == l2 + x.n)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta2D_spg(x, y, -2, kwave2, j1, k1); // left
			}

		}
	}

	if (non_zeros_in_2Dblock3diag != count) printf("FAILED generation!!! %d != %d\n", non_zeros_in_2Dblock3diag, count);
	else
	{
		printf("SUCCESSED 2D generation!\n");
		printf("Non_zeros inside generating PML function: %d\n", count);
	}

}


void GenSparseMatrixOnline1DwithPML(int w, size_m x, size_m y, size_m z, ccsr* Acsr, double kwave2)
{
	int size = x.n;
	int size2 = x.n - 1;
	int non_zeros_in_1D3diag = size + size2 * 2;
	double RelRes = 0;
	double k = (double)kk;
	//double kww = 4.0 * PI * PI * (w - n2) * (w - n2) / (y.l * y.l);
	//double kww = 4 * PI * PI * (w - n2) * (w - n2);

	//printf("Number k = %lf\n", k);

	//printf("analytic non_zeros in PML function: %d\n", non_zeros_in_2Dblock3diag);


	if (non_zeros_in_1D3diag != Acsr->non_zeros) printf("ERROR! Uncorrect value of non_zeros inside 1D: %d != %d\n", non_zeros_in_1D3diag, Acsr->non_zeros);

	printf("Gen1D matrix for frequency w = %d, k^2 - ky^2 = %lf\n", w - z.n / 2, kwave2);

	// All elements

	dtype *diag = alloc_arr<dtype>(size); // 0
	dtype *subXdiag = alloc_arr<dtype>(size2); // -1
	dtype *supXdiag = alloc_arr<dtype>(size2); // 1

	int count = 0;

	for (int l1 = 0; l1 < size; l1++)
	{
		Acsr->ia[l1] = count + 1;
		for (int l2 = 0; l2 < size; l2++)
		{
			if (l1 == l2)
			{
				Acsr->ja[count] = l2 + 1;
				//Acsr->values[count] = dtype{ kwave2, 0 };
				//Acsr->values[count] += dtype{ 0, k * k * beta_eq };
				Acsr->values[count++] = beta1D(x, 0, kwave2, l1);
			}
			else if (l1 == l2 - 1)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta1D(x, 1, kwave2, l1); // right
			}
			else if (l1 == l2 + 1)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta1D(x, -1, kwave2, l1); // left
			}

		}
	}

	if (non_zeros_in_1D3diag != count) printf("FAILED generation!!! %d != %d\n", non_zeros_in_1D3diag, count);
	else
	{
		printf("SUCCESSED 1D generation!\n");
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

dtype EulerExp(dtype val)
{
	return exp(val.real()) * dtype { cos(val.imag()), sin(val.imag()) };
}

dtype my_exp(double val)
{
	return dtype { cos(val), sin(val) };
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

	if (r == 0) r = 0.005;
	
	double arg = (double)(kk) * r;

	return my_exp(arg) / (4.0 * PI * r);

/*	double ksi1, ksi2, ksi3;

	ksi1 = ksi2 = ksi3 = sqrt(1.0 / 3.0);

	return my_exp(omega / c_z * (ksi1 * x + ksi2 * y + ksi3 * z));*/

#else
	//return x * y * z * (x - xx.l) * (y - yy.l) * (z - zz.l);
	double c1 = 2 * PI / xx.l;
	double c2 = 2 * PI / yy.l;
	double c3 = 2 * PI / zz.l;

	return -sin(c1 * x) * sin(c2 * y) * sin(c3 * z) / 
		(c1 * c1 + c2 * c2 + c3 * c3 + 1.0);
#endif
}

dtype u_ex_complex_sound3D(size_m xx, size_m yy, size_m zz, double x, double y, double z, point source)
{
	x -= source.x;
	y -= source.y;
	z -= source.z;

	double r = sqrt(x * x + y * y + z * z);

	if (r == 0) r = 0.005;

	dtype kk_loc = 2.0 * PI * (double)omega / MakeSound3D(xx, yy, zz, x, y, z, source);

	dtype arg = kk_loc * r;
	
	// i * k * r
	arg = { -arg.imag() , arg.real() };

	// e ^ {i k r} / (4 Pi r)

	return EulerExp(arg) / (4.0 * PI * r);
}

dtype F3D_ex_complex(size_m xx, size_m yy, size_m zz, double x, double y, double z, point source, int &l)
{
#ifdef HELMHOLTZ
	if (x == source.x && y == source.y && z == source.z)
	{
		printf("SOURCE AT x = %lf, y = %lf, z = %lf\n", x, y, z);
		l = x / xx.h - 1 + xx.n * (y / yy.h - 1) + xx.n * yy.n * (z / zz.h - 1);

		double volume = -1.0 / (xx.h * yy.h * zz.h);
		printf("volume = %lf\n", fabs(volume));

		return volume;
	}
	else
	{
		return 0;
	}
#else
	//return 2.0 * (x * (x - xx.l) * z * (z - zz.l) + y * (y - yy.l) * z * (z - zz.l) + x * (x - xx.l) * y * (y - yy.l));
	return sin(2 * PI * x / xx.l) * sin(2 * PI * y / yy.l) * sin(2 * PI * z / zz.l);
#endif
}

dtype F2D_ex_complex(size_m xx, size_m yy, double x, double y, point source, int &l)
{
	if (x == source.x && y == source.y)
	{
		printf("SOURCE AT x = %lf, y = %lf\n", x, y);
		l = x / xx.h - 1 + xx.n * (y / yy.h - 1);
		return 1.0 / (xx.h * yy.h);
	}
	else
	{
		return 0;
	}
}

dtype F1D_ex_complex(size_m xx, double x, point source, int &l)
{
	if (x == source.x)
	{
		printf("SOURCE AT x = %lf\n", x);
		l = x / xx.h - 1;
		return 1.0 / (xx.h);
	}
	else
	{
		return 0;
	}
}

void SetRHS1D(size_m xx, dtype* f, point source, int& l)
{
	for (int i = 0; i < xx.n; i++)
		f[i] = F1D_ex_complex(xx, (i + 1) * xx.h, source, l);
}

void SetRHS2D(size_m xx, size_m yy, dtype* f, point source, int& l)
{
	for (int j = 0; j < yy.n; j++)
		for (int i = 0; i < xx.n; i++)
			f[i + xx.n * j] = F2D_ex_complex(xx, yy, (i + 1) * xx.h, (j + 1) * yy.h, source, l);
}

void SetRHS3D(size_m xx, size_m yy, size_m zz, dtype* f, point source, int& l)
{
	for (int k = 0; k < zz.n; k++)
		for (int j = 0; j < yy.n; j++)
			for (int i = 0; i < xx.n; i++)
				f[i + xx.n * j + xx.n * yy.n * k] = F3D_ex_complex(xx, yy, zz, (i + 1) * xx.h, (j + 1) * yy.h, (k + 1) * zz.h, source, l);
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
	int size_nopml = x.n_nopml * y.n_nopml * z.n_nopml;
	dtype *f1 = alloc_arr<dtype>(size);
	dtype *g_nopml = alloc_arr<dtype>(size_nopml);
	dtype *f_nopml = alloc_arr<dtype>(size_nopml);
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

	reducePML3D(x, y, z, size, g, size_nopml, g_nopml);
	reducePML3D(x, y, z, size, f, size_nopml, f_nopml);

	//RelRes = zlange("Frob", &size, &ione, g, &size, NULL);
	//RelRes = RelRes / zlange("Frob", &size, &ione, f, &size, NULL);

	RelRes = zlange("Frob", &size_nopml, &ione, g_nopml, &size_nopml, NULL);
	RelRes = RelRes / zlange("Frob", &size_nopml, &ione, f_nopml, &size_nopml, NULL);

	printf("End resid\n");

	free_arr(f1);
	free_arr(g_nopml);
	free_arr(f_nopml);
}

void ResidCSR2D(size_m x, size_m y, ccsr* Dcsr, dtype* x_sol, dtype *f, dtype* g, point source, double &RelRes)
{
	int n = x.n;
	int size = n * y.n;
	int size_nopml = (x.n - 2 * x.pml_pts) * (y.n - 2 * y.pml_pts);
	dtype *f1 = alloc_arr<dtype>(size);
	int ione = 1;

	struct safe_val
	{
		int i;
		dtype val;
	} p1;


	// Multiply matrix A in CSR format by vector x_sol to obtain f1
	mkl_zcsrgemv("No", &size, Dcsr->values, Dcsr->ia, Dcsr->ja, x_sol, f1);

//#pragma omp parallel for schedule(static)
	for (int i = 0; i < size; i++)
	{
		g[i] = f[i] - f1[i];
		//if (abs(f[i]) != 0) printf("%12.10lf %12.10lf %12.10lf %12.10lf\n", f[i].real(), f[i].imag(), f1[i].real(), f1[i].imag());

		if (abs(f[i]) != 0)
		{
			p1.i = i;
			p1.val = f[i];
			printf("non_zero value in f[%d] = (%lf, %lf)\n", i, f[i].real(), f[i].imag());
			f[i] = f1[i] = g[i] = 0;
		}
	}

	//	for (int i = 0; i < size; i++)
	//		printf("%lf %lf\n", f[i], f1[i]);

#ifdef DEBUG
	print_vec(size, f, g, "f and g");
#endif

	dtype *g_nopml = alloc_arr<dtype>(size_nopml);
	dtype *f_nopml = alloc_arr<dtype>(size_nopml);
	reducePML2D(x, y, size, g, size_nopml, g_nopml);
	reducePML2D(x, y, size, f, size_nopml, f_nopml);

	RelRes = zlange("Frob", &size_nopml, &ione, g_nopml, &size_nopml, NULL);

	f[p1.i] = p1.val;

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
	double real, imag;
	srand((unsigned int)time(0));
	for (int i = 0; i < size; i++)
	{
		real = random(0.0, 1.0);
		imag = random(0.0, 1.0);
		vector[i] = { real, imag };
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
		Nz = z.n - 2 * z.spg_pts;
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
				fprintf(file, "%lf %12.10lf %12.10lf %12.10lf %12.10lf %12.10lf %12.10lf %12.10lf\n", i * x.h, j * y.h, k * z.h,
					x_orig[i + j * Nx + k * Nx * Ny].real(), x_orig[i + j * Nx + k * Nx * Ny].imag(),
					x_pard[i + j * Nx + k * Nx * Ny].real(), x_pard[i + j * Nx + k * Nx * Ny].imag(),
					x_orig[i + j * Nx + k * Nx * Ny].real() / x_pard[i + j * Nx + k * Nx * Ny].real());
		fclose(file);
	}
}

void output2D(char *str, bool pml_flag, size_m x, size_m y, dtype* x_orig, dtype* x_pard)
{
	char name[255];
	int Nx, Ny;

	if (pml_flag == true)
	{
		Nx = x.n - 2 * x.pml_pts;
		Ny = y.n - 2 * y.pml_pts;
	}
	else
	{
		Nx = x.n;
		Ny = y.n;
	}

		sprintf(name, "%s.dat", str);
		FILE *file = fopen(name, "w");
		for (int j = 0; j < Ny; j++)
			for (int i = 0; i < Nx; i++)
				fprintf(file, "%lf %12.10lf %12.10lf %12.10lf %12.10lf %12.10lf %12.10lf %12.10lf\n", i * x.h, j * y.h,
					x_orig[i + j * Nx].real(), x_orig[i + j * Nx].imag(),
					x_pard[i + j * Nx].real(), x_pard[i + j * Nx].imag(),
					x_orig[i + j * Nx].real() / x_pard[i + j * Nx].real(),
					x_orig[i + j * Nx].imag() / x_pard[i + j * Nx].imag());
		fclose(file);

}

void output1D(char *str, bool pml_flag, size_m x, dtype* x_orig, dtype* x_pard)
{
	char name[255];
	int Nx;

	if (pml_flag == true)
	{
		Nx = x.n - 2 * x.pml_pts;
	}
	else
	{
		Nx = x.n;
	}

	sprintf(name, "%s.dat", str);
	FILE *file = fopen(name, "w");
	for (int i = 0; i < Nx; i++)
		fprintf(file, "%lf %12.10lf %12.10lf %12.10lf %12.10lf\n", i * x.h,
			x_orig[i].real(), x_orig[i].imag(),
			x_pard[i].real(), x_pard[i].imag());
	fclose(file);

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
		Nz = z.n - 2 * z.spg_pts;
	}
	else
	{
		Nx = x.n;
		Ny = y.n;
		Nz = z.n;
	}

	FILE* file1;
	//sprintf(str, "run.plt", numb++);
	if (col == 4 || col == 5) str = "run_ex.plt";
	else str = "run_pard.plt";

	file1 = fopen(str, "w");

	fprintf(file1, "set term png font \"Times-Roman, 16\"\n");
	//fprintf(file, "set view map\n");
	fprintf(file1, "set xrange[0:%d]\nset yrange[0:%d]\n", (int)LENGTH_X, (int)LENGTH_Y);
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

	fprintf(file1, "exit\n");

	fclose(file1);
	system(str);
}

void gnuplot2D(char *splot, char *sout, bool pml_flag, int col, size_m x, size_m y)
{
	char *str;
	str = alloc_arr<char>(255);
	int Nx, Ny;
	if (pml_flag == true)
	{
		Nx = x.n - 2 * x.pml_pts;
		Ny = y.n - 2 * y.pml_pts;
	}
	else
	{
		Nx = x.n;
		Ny = y.n;
	}

	FILE* file1;
	//sprintf(str, "run.plt", numb++);
	if (col == 3) str = "run_ex.plt";
	else str = "run_pard.plt";

	file1 = fopen(str, "w");

	//fprintf(file1, "reset\nclear\n");
	fprintf(file1, "set term png font \"Times-Roman, 16\"\n");
	//fprintf(file, "set view map\n");
	fprintf(file1, "set xrange[0:%lf]\nset yrange[0:%lf]\n", x.l, y.l);
	fprintf(file1, "set pm3d\n");
	fprintf(file1, "set palette\n");


	fprintf(file1, "set output '%s_re.png'\n", sout);
	fprintf(file1, "splot '%s.dat' u 2:1:%d w linespoints pt 7 palette pointsize 1 notitle\n\n", splot, col);

	fprintf(file1, "set output '%s_im.png'\n", sout);
	fprintf(file1, "splot '%s.dat' u 2:1:%d w linespoints pt 7 palette pointsize 1 notitle\n\n", splot, col + 1);

	fprintf(file1, "exit\n");
	

	fclose(file1);
	system(str);
}

void gnuplot1D(char *splot, char *sout, bool pml_flag, int col, size_m x)
{
	char *str;
	str = alloc_arr<char>(255);
	int Nx;
	if (pml_flag == true)
	{
		Nx = x.n - 2 * x.pml_pts;
	}
	else
	{
		Nx = x.n;
	}

	FILE* file1;
	//sprintf(str, "run.plt", numb++);
	if (col == 3) str = "run_ex.plt";
	else str = "run_pard.plt";

	file1 = fopen(str, "w");

	//fprintf(file1, "reset\nclear\n");
	fprintf(file1, "set term png font \"Times-Roman, 16\"\n");
	//fprintf(file, "set view map\n");
	fprintf(file1, "set xrange[0:%lf]\n", x.l);

	fprintf(file1, "set output '%s_re.png'\n", sout);
	fprintf(file1, "plot '%s.dat' u 1:%d w linespoints pt 7 pointsize 1 notitle\n\n", splot, col);

	fprintf(file1, "set output '%s_im.png'\n", sout);
	fprintf(file1, "plot '%s.dat' u 1:%d w linespoints pt 7 pointsize 1 notitle\n\n", splot, col + 1);

	fprintf(file1, "exit\n");


	fclose(file1);
	system(str);
}

void ApplyCoeffMatrixA(size_m x, size_m y, size_m z, const dtype *w, const dtype* deltaL, dtype* g, double thresh)
{
	// Function for applying (I - deltaL * L_0 ^{-1}) * w = g
	int size = x.n * y.n * z.n;

	// Solve the preconditioned system: L_0 ^ {-1} * w = g
	Solve3DSparseUsingFT(x, y, z, w, g, thresh);

	// Multiply point-to-point deltaL * g
	OpTwoMatrices(size, 1, g, deltaL, g, size, '*');

	// g:= w - g
	OpTwoMatrices(size, 1, w, g, g, size, '-');
}

void Solve3DSparseUsingFT(size_m x, size_m y, size_m z, const dtype *f, dtype* x_sol, double thresh)
{

//#define PRINT

#ifdef PRINT
	printf("Solving %d x %d x %d Laplace equation using FFT's\n", x.n, y.n, z.n);
	printf("Reduce the problem to set of %d systems of size %d x %d\n", z.n, x.n * y.n, x.n * y.n);
#endif

	int size = x.n * y.n * z.n;
	int size2D = x.n * y.n;
	MKL_LONG status;
	double norm = 0;

	dtype *x_sol_prd = alloc_arr<dtype>(size);
	//dtype *x_sol_fft_nopml = alloc_arr<dtype>(x.n_nopml * y.n_nopml * z.n);
	dtype *f_FFT = alloc_arr<dtype>(size);
	dtype *f2D = alloc_arr<dtype>(size2D);
	dtype* u1D = alloc_arr<dtype>(z.n);
	dtype* u1D_BFFT = alloc_arr<dtype>(z.n);
	//dtype *x_sol_ex = alloc_arr<dtype>(size2D);
	
	
	// f(x,y,z) -> fy(x,z) 
	DFTI_DESCRIPTOR_HANDLE my_desc1_handle;
	DFTI_DESCRIPTOR_HANDLE my_desc2_handle;

	// Create 1D FFT of COMPLEX DOUBLE case
	status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, z.n);
	status = DftiSetValue(my_desc1_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	status = DftiSetValue(my_desc1_handle, DFTI_BACKWARD_SCALE, 1.0 / z.n);
//	status = DftiSetValue(my_desc1_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	status = DftiCommitDescriptor(my_desc1_handle);

	// We make n2 * n3 FFT's for one dimensional direction x with n1 grid points

#ifdef PRINT
	printf("Applying 1D Fourier transformation for 3D RHS\n");
#endif
	for (int k = 0; k < size2D; k++)
	{
		status = DftiComputeForward(my_desc1_handle, (void*)&f[z.n * k], &f_FFT[z.n * k]);
		//MyFFT1D_ForwardComplexSin(n1, &f[n1 * k], &f_FFT[n1 * k]);
		//MyFT1D_ForwardComplex(z.n, z, &f[z.n * k], &f_FFT[z.n * k]);
	}

#ifdef PRINT
	printf("check right-hand-side f\n");
	for (int i = 0; i < size; i++)
		if (abs(f_FFT[i]) != 0) printf("f_FFT[%d] = %lf %lf\n", i, f_FFT[i].real(), f_FFT[i].imag());
#endif


	// Calling the solver
	int mtype = 13;
	int *iparm = alloc_arr<int>(64);
	int *perm = alloc_arr<int>(size2D);
	size_t *pt = alloc_arr<size_t>(64);

	//printf("pardisoinit...\n");
	pardisoinit(pt, &mtype, iparm);

	int maxfct = 1;
	int mnum = 1;
	int phase = 13;
	int rhs = 1;
	int msglvl = 0;
	int error = 0;

	int n_nopml = x.n_nopml * y.n_nopml;
	int size_nopml = n_nopml * z.n_nopml;
	int size2D_nopml = x.n_nopml * y.n_nopml;

	// Memory for 2D CSR matrix
	ccsr *D2csr;
	int non_zeros_in_2Dblock3diag = (x.n + (x.n - 1) * 2) * y.n + 2 * (size2D - x.n);
	D2csr = (ccsr*)malloc(sizeof(ccsr));
	D2csr->values = alloc_arr<dtype>(non_zeros_in_2Dblock3diag);
	D2csr->ia = alloc_arr<int>(size2D + 1);
	D2csr->ja = alloc_arr<int>(non_zeros_in_2Dblock3diag);
	D2csr->ia[size2D] = non_zeros_in_2Dblock3diag + 1;
	D2csr->non_zeros = non_zeros_in_2Dblock3diag;

#ifdef PRINT
	printf("Non-zeros in 2D block-diagonal: %d\n", non_zeros_in_2Dblock3diag);
	printf("----------Generating 2D matrix and rhs + solving by pardiso-------\n");
	printf("Size of system: %d x %d with PML %d on each direction\n", x.n, y.n, 2 * x.pml_pts);
#endif

	point sourcePML = { x.l / 2.0, y.l / 2 };

	printf("SOURCE in 2D WITH PML AT: (%lf, %lf)\n", sourcePML.x, sourcePML.y);
	double k = (double)kk;
	double k2 = k * k;
	int nhalf = z.n / 2;
	double kww;
	int src;

	char *str1, *str2, *str3;
	str1 = alloc_arr<char>(255);
	str2 = alloc_arr<char>(255);
	str3 = alloc_arr<char>(255);
	bool pml_flag = false;

	int* freqs = alloc_arr<int>(size);

	GenSparseMatrixOnline2DwithPML(-1, x, y, z, D2csr, 0, freqs);

	for (int k = 0; k < z.n; k++)
	{
		int count = 0;
//		printf("-------------Iter: %d------------------\n", i);

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
		double kwave2 = k2 - kww;

		if (kww > 2.5 * k2) continue;


		dtype kwave_beta2 = k2 * dtype{ 1, beta_eq } - kww;

		dtype alpha_k;

		double nu = 1;
		double c = 300;
		double omega_loc = 2.0 * PI * nu;

		//kwave2 = 0.005;
		//kwave2 = omega_loc * omega_loc / (c * c);

		//double ppw = c / nu / x.h;

		double ppw = 1.0 / (sqrt(abs(kwave2)) / (2.0 * PI)) / z.h;

#ifdef PRINT
		printf("ppw: %lf\n", ppw);
		printf("frequency: %lf\n", sqrt(abs(kwave2)) * c_z / (2.0 * PI));
#endif


		// источник в каждой задаче в середине 
		//GenSparseMatrixOnline2D("FT", i, x, y, z, Bc_mat, n1, Dc, n1, Bc_mat, n1, D2csr);

		GenSparseMatrixOnline2DwithPML(k, x, y, z, D2csr, kwave_beta2, freqs);

		GenRhs2D(k, x, y, z, f_FFT, f2D);

		// normalization of rhs
		alpha_k = f2D[size2D/2] / (1.0 / (x.h * y.h));


		//	GenRHSandSolution2D_Syntetic(y, z, D2csr, &u2Dsynt[i * size2D], f2D);
		pardiso(pt, &maxfct, &mnum, &mtype, &phase, &size2D, D2csr->values, D2csr->ia, D2csr->ja, perm, &rhs, iparm, &msglvl, f2D, &x_sol_prd[k * size2D], &error);
		if (error != 0) printf("Error: PARDISO!!!\n");
		//		norm = rel_error(zlange, n2 * n3, 1, &u2Dsynt[i * size2D], &x_sol_prd[i * size2D], n2 * n3, thresh);

#pragma omp parallel for simd schedule(simd:static)
		for (int i = 0; i < size; i++)
			D2csr->values[freqs[i]] -= kwave_beta2;

		double eps = 0.01; // 1 percent

#ifdef PRINT
		if (norm < eps) printf("Resid 2D Hankel norm %12.10e < eps %12.10lf: PASSED\n\n", norm, eps);
		else printf("Resid 2D Hankel norm %12.10lf > eps %12.10lf : FAILED\n\n", norm, eps);
#endif
		//sprintf(str1, "ChartsPML/model_pml_%lf", kwave2);
		//sprintf(str2, "ChartsPML/model_pml_ex_%lf", kwave2);
		//sprintf(str3, "ChartsPML/model_pml_pard_%lf", kwave2);

		//sprintf(str1, "ChartsSPONGE/model_pml_%lf", kwave2);
		//sprintf(str2, "ChartsSPONGE/model_pml_ex_%lf", kwave2);
		//sprintf(str3, "ChartsSPONGE/model_pml_pard_%lf", kwave2);

//#define CHECK_ACCURACY

#ifdef CHECK_ACCURACY
		if (kwave2 > 0)
		{
			get_exact_2D_Hankel(x.n, y.n, x, y, x_sol_ex, sqrt(kwave2), sourcePML);

			normalization_of_exact_sol(x.n, y.n, x, y, x_sol_ex, alpha_k);

			norm = resid_2D_Hankel(x, y, D2csr, x_sol_ex, f2D, sourcePML);
			printf("resid ||A * x_sol - kf|| 2D Hankel: %lf\n", norm);

			output2D(str1, pml_flag, x, y, x_sol_ex, &x_sol_prd[i * size2D]);
			gnuplot2D(str1, str2, pml_flag, 3, x, y);
			gnuplot2D(str1, str3, pml_flag, 5, x, y);
		}
		else // kwave2 < 0
		{
			get_exact_2D_Hankel(x.n, y.n, x, y, x_sol_ex, { 0, sqrt(abs(kwave2)) }, sourcePML);

			normalization_of_exact_sol(x.n, y.n, x, y, x_sol_ex, alpha_k);

			norm = resid_2D_Hankel(x, y, D2csr, x_sol_ex, f2D, sourcePML);
			printf("resid ||A * x_sol - kf|| 2D Hankel: %lf\n", norm);

			output2D(str1, pml_flag, x, y, x_sol_ex, &x_sol_prd[i * size2D]);

			//gnuplot2D(str1, str3, pml_flag, 5, y, z);
		}

		reducePML2D(x, y, size2D, &x_sol_prd[i * size2D], size2D_nopml, &x_sol_fft_nopml[i * size2D_nopml]);
		check_exact_sol_Hankel(alpha_k, kwave2, x, y, &x_sol_fft_nopml[i * size2D_nopml], thresh);
#endif

		//printf("End for w = %d\n", k);
	}

	//("Reducing PML after taking a solution\n");

#ifdef PRINT
	printf("Backward 1D FFT's of %d x %d times to each point of 2D solution\n", x.n_nopml, y.n_nopml);
#endif
	for (int w = 0; w < size2D; w++)
	{
		GenSol1DBackward(w, x, y, z, x_sol_prd, u1D);		 // new u1D on each iteration

#if 0
		status = DftiComputeBackward(my_desc1_handle, u1D, &x_sol[k * z.n]);
#else
		status = DftiComputeBackward(my_desc1_handle, u1D, u1D_BFFT);

		for (int k = 0; k < z.n; k++)
			x_sol[w + k * size2D] = u1D_BFFT[k];
#endif
	
	//	MyFFT1D_BackwardComplexSin(n1, u1D, &x_sol[k * n1]);
	//	MyFT1D_BackwardComplex(z.n, z, u1D, &x_sol[k * z.n]);
	}

	status = DftiFreeDescriptor(&my_desc1_handle);
	printf("------------- The end of algorithm ----------------------\n");


	int ione = 1;
	//zlacpy("All", &size_nopml, &ione, x_sol, &size_nopml, x_pard_nopml_cpy, &size_nopml);

	free_arr(x_sol_prd);
	//free_arr(x_sol_fft_nopml);
	free_arr(f_FFT);
	free_arr(u1D);
	free_arr(u1D_BFFT);
	free_arr(iparm);
	free_arr(perm);
	free_arr(pt);
	free_arr(freqs);
	free_arr(f2D);
	//free_arr(x_sol_ex);

}

void GenRHSandSolution1D(size_m x, dtype* u_ex1D, dtype* f1D, double k, point sourcePML, int &src)
{
	// Set RHS
	SetRHS1D(x, f1D, sourcePML, src);

	printf("src = %d\n", src);

	// exact 1D Helmholtz
	GenExact1DHelmholtz(x.n, x, u_ex1D, k, sourcePML);
}

void GenRHSandSolution2D(size_m x, size_m y, ccsr* D2csr, dtype* u_ex2D, dtype* f2D, double kwave2, point sourcePML, int &src)
{
	double norm = 0;

	// Set RHS
	SetRHS2D(x, y, f2D, sourcePML, src);

	printf("src 2D = %d\n", src);

	dtype alpha_k = f2D[src] / (1.0 / (x.h * y.h));

	// exact 1D Helmholtz

	if (kwave2 > 0)
	{
		get_exact_2D_Hankel(x.n, y.n, x, y, u_ex2D, sqrt(kwave2), sourcePML);

		normalization_of_exact_sol(x.n, y.n, x, y, u_ex2D, alpha_k);

		norm = resid_2D_Hankel(x, y, D2csr, u_ex2D, f2D, sourcePML);
		printf("resid ||A * x_sol - kf|| 2D Hankel: %lf\n", norm);

	}
	else // kwave2 < 0
	{
		get_exact_2D_Hankel(x.n, y.n, x, y, u_ex2D, { 0, sqrt(abs(kwave2)) }, sourcePML);

		normalization_of_exact_sol(x.n, y.n, x, y, u_ex2D, alpha_k);

		norm = resid_2D_Hankel(x, y, D2csr, u_ex2D, f2D, sourcePML);
		printf("resid ||A * x_sol - kf|| 2D Hankel: %lf\n", norm);
	}
}

dtype uf(size_m& x, size_m& y, size_m& z, int i, int j, int k, const dtype* u)
{
	int size2D = x.n * y.n;
	if (i < 0 || j < 0 || k < 0 || i > x.n - 1 || j > y.n - 1 || k > z.n - 1)
	{
		return 0;
	}
	else
	{
		return u[i + x.n * j + size2D * k];
	}
}

void ComputeResidual(size_m x, size_m y, size_m z, double kw, const dtype* u, const dtype* f, dtype *f_res, double &RelRes)
{
	int size = x.n * y.n * z.n;
	int size2D = x.n * y.n;
	int i, j, k;
	int count = 0;
	int ione = 1;

	for (int l = 0; l < size; l++)
	{
		take_coord3D(x.n, y.n, z.n, l, i, j, k);
		if (k > 1 && k < z.n - 2)
		{
			f_res[l] =
				(uf(x, y, z, i - 1, j, k, u) - 2.0 * uf(x, y, z, i, j, k, u) + uf(x, y, z, i + 1, j, k, u)) / (x.h * x.h) + 
				(uf(x, y, z, i, j - 1, k, u) - 2.0 * uf(x, y, z, i, j, k, u) + uf(x, y, z, i, j + 1, k, u)) / (y.h * y.h) +
				(uf(x, y, z, i, j, k - 1, u) - 2.0 * uf(x, y, z, i, j, k, u) + uf(x, y, z, i, j, k + 1, u)) / (z.h * z.h) +
				dtype{ kw * kw , 0 } * uf(x, y, z, i, j, k, u);
			count++;

			//	(u[i - 1 + x.n * j + size2D * k] - 2.0 * u[i + x.n * j + size2D * k] + u[i + 1 + x.n * j + size2D * k]) / (x.h * x.h) +
			//	(u[i + x.n * (j - 1) + size2D * k] - 2.0 * u[i + x.n * j + size2D * k] + u[i + x.n * (j + 1) + size2D * k]) / (y.h * y.h) +
			//	(u[i + x.n * j + size2D * (k - 1)] - 2.0 * u[i + x.n * j + size2D * k] + u[i + x.n * j + size2D * (k + 1)]) / (z.h * z.h) +
		}
		else
		{
			f_res[l] = 0;
			count++;
		}
	}

	if (count != size) printf("FAIL!!! Uncorrect multiplication f: = A * u\n");
	else printf("Successed residual\n");

#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < size; i++)
		f_res[i] = f_res[i] - f[i];

	int i1, j1, k1;

	for (int k = 0; k < z.n; k++)
	{
		f_res[k * size2D + size2D / 2] = 0;

		int l = k * size2D + size2D / 2;

		take_coord3D(x.n, y.n, z.n, l, i1, j1, k1);
	//	printf("i = %d j = %d k = %d\n", i1, j1, k1);
	}

	for (int k = 0; k < z.n; k++)
	{
		int src = size2D / 2;
		NullifySource2D(x, y, &f_res[k * size2D], src, 5);
	}

	RelRes = dznrm2(&size, f_res, &ione);

}

void Solve1DSparseHelmholtz(size_m x, size_m y, size_m z, dtype *f1D, dtype *x_sol_prd, double thresh)
{
	printf("-----------Test 1D Helmholtz--------\n");
	ccsr *D1csr;
	int size1D = x.n;
	int size1D_nopml = x.n_nopml;
	int non_zeros_in_1D3diag = size1D + (size1D - 1) * 2;
	D1csr = (ccsr*)malloc(sizeof(ccsr));
	D1csr->values = alloc_arr<dtype>(non_zeros_in_1D3diag);
	D1csr->ia = alloc_arr<int>(size1D + 1);
	D1csr->ja = alloc_arr<int>(non_zeros_in_1D3diag);
	D1csr->ia[size1D] = non_zeros_in_1D3diag + 1;
	D1csr->non_zeros = non_zeros_in_1D3diag;

#ifdef PRINT
	printf("Non-zeros in 2D block-diagonal: %d\n", non_zeros_in_2Dblock3diag);
	printf("----------Generating 2D matrix and rhs + solving by pardiso-------\n");
	printf("Size of system: %d x %d with PML %d on each direction\n", x.n, y.n, 2 * x.pml_pts);
#endif

	point sourcePML = { x.l / 2.0 };
	
	printf("L = %lf\n", x.l);
	printf("PML = %lf\n", x.pml_pts * x.h);
	printf("SOURCE in 2D WITH PML AT: (%lf)\n", sourcePML.x);
	double k = (double)kk;

	char *str1, *str2, *str3;
	str1 = alloc_arr<char>(255);
	str2 = alloc_arr<char>(255);
	str3 = alloc_arr<char>(255);
	bool pml_flag = false;

	// Calling the solver
	int mtype = 13;
	int *iparm = alloc_arr<int>(64);
	int *perm = alloc_arr<int>(size1D);
	size_t *pt = alloc_arr<size_t>(64);

	printf("pardisoinit...\n");
	pardisoinit(pt, &mtype, iparm);

	int maxfct = 1;
	int mnum = 1;
	int phase = 13;
	int rhs = 1;
	int msglvl = 0;
	int error = 0;
	int src =  0;


	int count = 0;
	double kwave2 = k * k;

	dtype alpha_k;
	dtype *x_sol_ex = alloc_arr<dtype>(size1D);

	dtype *x_sol_ex_nopml = alloc_arr<dtype>(size1D_nopml);
	dtype *x_sol_prd_nopml = alloc_arr<dtype>(size1D_nopml);

	double ppw = 1.0 / (sqrt(abs(kwave2)) / (2.0 * PI)) / z.h;

	// источник в каждой задаче в середине 

	GenSparseMatrixOnline1DwithPML(0, x, y, z, D1csr, kwave2);

	// Gen RHS and exact solution
	GenRHSandSolution1D(x, x_sol_ex, f1D, sqrt(kwave2), sourcePML, src);

	pardiso(pt, &maxfct, &mnum, &mtype, &phase, &size1D, D1csr->values, D1csr->ia, D1csr->ja, perm, &rhs, iparm, &msglvl, f1D, x_sol_prd, &error);
	//		norm = rel_error(zlange, n2 * n3, 1, &u2Dsynt[i * size2D], &x_sol_prd[i * size2D], n2 * n3, thresh);

	double eps = 0.01; // 1 percent

	sprintf(str1, "Charts1D/model_1D_kwave2_%lf", kwave2); // .dat file
	sprintf(str2, "Charts1D/model_ex_1D_kwave2_%lf", kwave2);
	sprintf(str3, "Charts1D/model_prd_1D_kwave2_%lf", kwave2);

	pml_flag = false;

	output1D(str1, pml_flag, x, x_sol_ex, x_sol_prd);

	gnuplot1D(str1, str2, pml_flag, 2, x);
	gnuplot1D(str1, str3, pml_flag, 4, x);

	x_sol_ex[src] = x_sol_prd[src] = 0;

	reducePML1D(x, size1D, x_sol_ex, size1D_nopml, x_sol_ex_nopml);
	reducePML1D(x, size1D, x_sol_prd, size1D_nopml, x_sol_prd_nopml);

	double norm = RelError(zlange, size1D_nopml, 1, x_sol_prd_nopml, x_sol_ex_nopml, size1D_nopml, thresh);
	printf("Norm 1D solution ||x_sol - x_ex|| / ||x_ex|| = %lf\n", norm);


}

dtype SetExact1DHelmholtz(double x, double k, point sourcePML)
{
	x -= sourcePML.x;

	double r = sqrt(x * x);

	dtype arg = k * r;

	// i * k * r
	arg = { -arg.imag(), arg.real() };

	// e ^ {i k r} / (4 Pi r)

	return EulerExp(arg) / dtype{ 0, 2.0 * k };
}

void GenExact1DHelmholtz(int n, size_m x, dtype *x_sol_ex, double k, point sourcePML)
{
	for (int i = 0; i < n; i++)
	{
		x_sol_ex[i] = SetExact1DHelmholtz((i + 1) * x.h, k, sourcePML);
	}
}


void NullifySource2D(size_m x, size_m y, dtype *u, int src, int npoints)
{
	int isrc, jsrc;
	jsrc = src / x.n;
	isrc = src - x.n * jsrc;

	for (int l = 0; l <= npoints; l++)
	{
		u[isrc - l + x.n * jsrc] = 0;
		u[isrc + l + x.n * jsrc] = 0;

		u[isrc + x.n * (jsrc - l)] = 0;
		u[isrc + x.n * (jsrc + l)] = 0;
	}
}

void Solve2DSparseHelmholtz(size_m x, size_m y, size_m z, dtype *f2D, dtype *x_sol_prd, double thresh)
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
	D2csr = (ccsr*)malloc(sizeof(ccsr));
	D2csr->values = alloc_arr<dtype>(non_zeros_in_2Dblock3diag);
	D2csr->ia = alloc_arr<int>(size2D + 1);
	D2csr->ja = alloc_arr<int>(non_zeros_in_2Dblock3diag);
	D2csr->ia[size2D] = non_zeros_in_2Dblock3diag + 1;
	D2csr->non_zeros = non_zeros_in_2Dblock3diag;

	point sourcePML = { x.l / 2.0, y.l / 2.0 };

	printf("Lx = %lf, Ly = %lf\n", x.l, y.l);
	printf("PML_x = %lf, PML_y = %lf\n", x.pml_pts * x.h, y.pml_pts * y.h);
	printf("SOURCE in 2D WITH PML AT: (%lf, %lf)\n", sourcePML.x, sourcePML.y);
	double k = (double)kk;
	int nhalf = z.n / 2;
	int src = 0;

	char *str1, *str2, *str3;
	str1 = alloc_arr<char>(255);
	str2 = alloc_arr<char>(255);
	str3 = alloc_arr<char>(255);
	bool pml_flag = false;


	int count = 0;
	int i = 50;

	double kww = 4.0 * PI * PI * (i - nhalf) * (i - nhalf) / (z.l * z.l);
	double kwave2 = k * k - kww;

	dtype alpha_k;
	dtype *x_sol_ex = alloc_arr<dtype>(size2D);

	dtype *x_sol_ex_nopml = alloc_arr<dtype>(size2D_nopml);
	dtype *x_sol_prd_nopml = alloc_arr<dtype>(size2D_nopml);


	double nu = 1;
	double c = 300;
	double omega_loc = 2.0 * PI * nu;
	double norm = 0;

	//double ppw = c / nu / x.h;


	// источник в каждой задаче в середине 

	GenSparseMatrixOnline2DwithSPONGE(i, x, y, z, D2csr, kwave2);

	// Gen RHS and exact solution; check residual |A * u - f|
	GenRHSandSolution2D(x, y, D2csr, x_sol_ex, f2D, kwave2, sourcePML, src);

	// normalization of rhs

	pardiso(pt, &maxfct, &mnum, &mtype, &phase, &size2D, D2csr->values, D2csr->ia, D2csr->ja, perm, &rhs, iparm, &msglvl, f2D, x_sol_prd, &error);

	double eps = 0.01; // 1 percent


	sprintf(str1, "Charts2D_v2/model_pml_%lf", kwave2);
	sprintf(str2, "Charts2D_v2/model_pml_ex_%lf", kwave2);
	sprintf(str3, "Charts2D_v2/model_pml_pard_%lf", kwave2);

	pml_flag = false;

#if 0
	for (int j = 0; j < y.n; j++)
		for (int i = 0; i < x.n; i++)
			if (x_sol_ex[i + x.n * j].real() < -1.0) printf("i = %d j = %d, l = %d, lx = %lf, ly = %lf\n",
				i, j, i + x.n * j, i * x.h, j * y.h);
#endif

	NullifySource2D(x, y, x_sol_ex, src, 1);
	NullifySource2D(x, y, x_sol_prd, src, 1);
	x_sol_ex[src] = x_sol_prd[src] = 0;

	output2D(str1, pml_flag, x, y, x_sol_ex, x_sol_prd);

	gnuplot2D(str1, str2, pml_flag, 3, x, y);
	gnuplot2D(str1, str3, pml_flag, 5, x, y);

	reducePML2D(x, y, size2D, x_sol_ex, size2D_nopml, x_sol_ex_nopml);
	reducePML2D(x, y, size2D, x_sol_prd, size2D_nopml, x_sol_prd_nopml);

	norm = RelError(zlange, size2D_nopml, 1, x_sol_prd_nopml, x_sol_ex_nopml, size2D_nopml, thresh);
	printf("Norm 1D solution ||x_sol - x_ex|| / ||x_ex|| = %lf\n", norm);

	check_exact_sol_Hankel(alpha_k, kwave2, x, y, x_sol_prd_nopml, thresh);

}


void OpTwoMatrices(int m, int n, const dtype *Y1, const dtype *Y2, dtype *Yres, int ldy, char sign)
{
	if (sign == '+')
	{
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
#pragma omp simd
			for (int i = 0; i < m; i++)
				Yres[i + ldy * j] = Y1[i + ldy * j] + Y2[i + ldy * j];
	}
	else if (sign == '-')
	{
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
#pragma omp simd
			for (int i = 0; i < m; i++)
				Yres[i + ldy * j] = Y1[i + ldy * j] - Y2[i + ldy * j];
	}
	else if (sign == '*')
	{
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
#pragma omp simd
			for (int i = 0; i < m; i++)
				Yres[i + ldy * j] = Y1[i + ldy * j] * Y2[i + ldy * j];
	}
	else if (sign == '/')
	{
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; j++)
#pragma omp simd
			for (int i = 0; i < m; i++)
				Yres[i + ldy * j] = Y1[i + ldy * j] / Y2[i + ldy * j];
	}
	else
	{
		printf("Incorrect sign\n");
	}
}

dtype zdot(int size, dtype* v1, dtype* v2)
{
	dtype *temp = alloc_arr<dtype>(size);
	dtype res = 0;

	OpTwoMatrices(size, 1, v1, v2, temp, size, '*');

	for (int i = 0; i < size; i++)
		res += temp[i];

	free_arr(temp);
	
	return res;
}

void AddDenseVectors(int n, double alpha, double *a, double beta, double *b, double *c)
{
#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < n; i++)
		c[i] = alpha * a[i] + beta * b[i];
}

void AddDenseVectorsComplex(int n, dtype alpha, dtype *a, dtype beta, dtype *b, dtype *c)
{
#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < n; i++)
		c[i] = alpha * a[i] + beta * b[i];
}

void MultVectorConst(int n, dtype* v1, dtype alpha, dtype* v2)
{
#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < n; i++)
		v2[i] = v1[i] * alpha;
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