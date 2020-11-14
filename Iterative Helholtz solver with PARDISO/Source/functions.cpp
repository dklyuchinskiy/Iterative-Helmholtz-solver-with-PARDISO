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
			printf("%5.3lf ", u[i + ldu*j].real());
		}
		printf("\n");
	}

	printf("\n");
}

void fprint(int m, int n, dtype *u, int ldu, char *mess)
{
	FILE *out = fopen("LaplacePML_real.dat", "w");
	for (int i = 0; i < m; i++)
	{
		//fprintf(out, "%d ", i);
		for (int j = 0; j < n; j++)
		{
			fprintf(out, "%4.2lf ", u[i + ldu * j].real());
		}
		fprintf(out, "\n");
	}

	fprintf(out,"\n");

	fclose(out);

	out = fopen("LaplacePML_imag.dat", "w");
	for (int i = 0; i < m; i++)
	{
		//fprintf(out, "%d ", i);
		for (int j = 0; j < n; j++)
		{
			fprintf(out, "%4.2lf ", u[i + ldu * j].imag());
		}
		fprintf(out, "\n");
	}

	fprintf(out, "\n");

	fclose(out);
}

void Eye(int n, dtype *H, int ldh)
{
	Clear(n, n, H, ldh);

	for (int j = 0; j < n; j++)
		for (int i = 0; i < n; i++)
			H[i + ldh * i] = 1.0;
}


void Hilbert(int n, dtype *H, int ldh)
{
	Clear(n, n, H, ldh);

#pragma omp parallel for schedule(static)
	for (int j = 0; j < n; j++)
#pragma omp simd
		for (int i = 0; i < n; i++)
			H[i + ldh * j] = 1.0 / (i + j + 1);
}


/* (m x n) matrix -> to (n x m) matrix */
void Mat_Trans(int m, int n, dtype *H, int ldh, dtype *Hcomp_tr, int ldhtr)
{
#pragma omp parallel for schedule(static)
	for (int i = 0; i < m; i++)
#pragma omp simd
		for (int j = 0; j < n; j++)
			Hcomp_tr[j + ldhtr * i] = H[i + ldh * j];
}

void Clear(int m, int n, dtype* A, int lda)
{
#pragma omp parallel for schedule(static)
	for (int j = 0; j < n; j++)
#pragma omp simd
		for (int i = 0; i < m; i++)
			A[i + lda * j] = 0.0;
}

void NormalizeVector(int size, dtype* v, dtype* out, double &norm)
{
	int ione = 1;
	double loc_norm = dznrm2(&size, v, &ione);

	// Compute 
#pragma omp parallel for simd schedule(static)
	for (int i = 0; i < size; i++)
		out[i] = v[i] / loc_norm;

	norm = loc_norm;
}

void GenerateDiagonal1DBlock(double w, int part_of_field, size_m x, size_m y, size_m z, dtype *DD, int lddd,
	   dtype* alpX, dtype* alpY, dtype* alpZ)
{
	int n = x.n;
	double k = double(kk);

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

void reducePML3D(size_m x, size_m y, size_m z, int size1, const dtype *vect, int size2, dtype *vect_red)
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
#ifdef PRINT
	if (numb != size2) printf("ERROR of reducing PML 3D: %d != %d\n", numb, size2);
	else printf("PML 3D is reduced successfully!\n");
#endif
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
#ifdef PRINT_TEST
		printf("There is no PML 2D reduction\n");
#endif

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

#ifdef PRINT_TEST
	if (numb != size2) printf("ERROR of reducing PML 2D: %d != %d\n", numb, size2);
	else printf("PML 2D is reduced successfully!\n");
#endif
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
#ifdef PRINT
	printf("------------ACCURACY CHECK---------\n");
#endif

	int size2D = n1 * n2;

	double *x_orig_re = alloc_arr<double>(size2D);
	double *x_orig_im = alloc_arr<double>(size2D);

	double *x_sol_re = alloc_arr<double>(size2D);
	double *x_sol_im = alloc_arr<double>(size2D);

	double eps1p = 0.01;

	for (int k = 0; k < n3; k++)
	{
#pragma omp parallel for simd schedule(static)
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
#ifdef PRINT
		printf("i = %d norm = %lf norm_re = %lf norm_im = %lf\n", k, norm, norm_re, norm_im);
#endif
	}

	free_arr(x_sol_re);
	free_arr(x_sol_im);
	free_arr(x_orig_re);
	free_arr(x_orig_im);

}

void check_norm_result2(int n1, int n2, int n3, int niter, double ppw, double spg, const dtype* x_orig_nopml, const dtype* x_sol_nopml,
	double* x_orig_re, double* x_orig_im, double *x_sol_re, double *x_sol_im)
{
	printf("------------ACCURACY CHECK---------\n");

	int size2D = n1 * n2;
	int size = size2D * n3;

	double eps1p = 0.01;
	double *norms_re = alloc_arr<double>(n3);
	double *norms_im = alloc_arr<double>(n3);

#pragma omp parallel for simd schedule(static)
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
		//if (niter == 11)
			printf("i = %d norm = %lf norm_re = %lf norm_im = %lf\n", k, norm, norms_re[k], norms_im[k]);
	}

#if 0
	FILE* fout;
	char str[255];
	sprintf(str, "Nit%d_N%d_Lx%d_FREQ%d_PPW%4.2lf_SPG%6.lf_BETA%5.3lf.dat", niter, n1, (int)LENGTH_X, (int)nu, ppw, spg, beta_eq);
	fout = fopen(str, "w");

	for (int k = 0; k < n3; k++)
		fprintf(fout, "%d %lf %lf\n", k, norms_re[k], norms_im[k]);

	fclose(fout);
#endif

	free_arr(norms_re);
	free_arr(norms_im);
}



void PrintProjection1D(size_m x, size_m y, dtype *x_ex, dtype *x_prd, int freq)
{
	char *str1 = alloc_arr<char>(255);
	char *str2 = alloc_arr<char>(255);
	bool pml_flag = false;

	dtype* x_sol1D_ex = alloc_arr<dtype>(x.n);
	dtype* x_sol1D_prd = alloc_arr<dtype>(x.n);

	// output 1D - X direction
	for(int j = 0; j < y.n; j++)
	{
		if (j == y.n / 2)
		{
			sprintf(str1, "Charts3D/projX/model_pml1Dx_freq%d_sec%d_h%d", freq, j, (int)x.h);
			sprintf(str2, "Charts3D/projX/model_pml1Dx_diff_freq%d_sec%d_h%d", freq, j, (int)x.h);

			output1D(str1, pml_flag, x, &x_ex[x.n * j], &x_prd[x.n * j]);
			gnuplot1D(str1, str2, pml_flag, 0, x);
		}
	}

	// output 1D - Y direction
	sprintf(str1, "Charts3D/projY/model_pml1Dy_freq%d_sec%d_h%d", freq, x.n / 2, (int)x.h);
	sprintf(str2, "Charts3D/projY/model_pml1Dy_diff_freq%d_sec%d_h%d", freq, x.n / 2, (int)x.h);

	for (int j = 0; j < y.n; j++)
	{
		x_sol1D_ex[j] = x_ex[x.n / 2 + j * x.n];
		x_sol1D_prd[j] = x_prd[x.n / 2 + j * x.n];
	}

	output1D(str1, pml_flag, y, x_sol1D_ex, x_sol1D_prd);
	gnuplot1D(str1, str2, pml_flag, 0, y);

	free_arr(x_sol1D_ex);
	free_arr(x_sol1D_prd);
}

double check_norm_circle_3D(size_m xx, size_m yy, size_m zz, int start_x, int end_x, const dtype* x_orig, const dtype* x_sol, point source, double thresh)
{
	int n1 = xx.n;
	int n2 = yy.n;
	int n3 = zz.n;

	int size2D = n1 * n2;
	int size = size2D * n3;
	
	double x, y, z;
	//double r0 = 5 * xx.h;
	double r;
//	double r_max = xx.l - 10 * xx.h;
	double norm;

	double r0 = start_x * xx.h;
	double r_max = end_x * xx.h;

	dtype* x_sol_circ = alloc_arr<dtype>(size);
	dtype* x_orig_circ = alloc_arr<dtype>(size);

	for(int k = zz.spg_pts; k < n3 - zz.spg_pts; k++)
		for(int j = yy.pml_pts; j < n2 - yy.pml_pts; j++)
			for (int i = xx.pml_pts; i < n1 - xx.pml_pts; i++)
			{
				x = (i + 1) * xx.h - source.x;
				y = (j + 1) * yy.h - source.y;
				z = (k + 1) * zz.h - source.z;
				r = sqrt(x * x + y * y + z * z);
				if (r >= r0 && r <= r_max)
				{
					x_sol_circ[i + n1 * j + size2D * k] = x_sol[i + n1 * j + size2D * k];
					x_orig_circ[i + n1 * j + size2D * k] = x_orig[i + n1 * j + size2D * k];
				}
			}

	norm = RelError(zlange, size, 1, x_sol_circ, x_orig_circ, size, thresh);

	printf("Square: 0 < x < %lf, 0 < y < %lf, 0 < z < %lf.\n", xx.l, yy.l, zz.l);
	printf("Norm in circle: %lf < r < %lf: %lf\n", r0, r_max, norm);

	free_arr(x_sol_circ);
	free_arr(x_orig_circ);

	return norm;
}

double check_norm_circle_3D_nopml(size_m xx, size_m yy, size_m zz, int start_x, int end_x, const dtype* x_orig_nopml, const dtype* x_sol_nopml, point source, double thresh)
{
	int n1 = xx.n_nopml;
	int n2 = yy.n_nopml;
	int n3 = zz.n_nopml;

	int size2D = n1 * n2;
	int size = size2D * n3;

	double x, y, z;
	//double r0 = 5 * xx.h;
	double r;
	//	double r_max = xx.l - 10 * xx.h;
	double norm;

	double r0 = start_x * xx.h;
	double r_max = end_x * xx.h;

	dtype* x_sol_circ = alloc_arr<dtype>(size);
	dtype* x_orig_circ = alloc_arr<dtype>(size);

	for (int k = 0; k < n3; k++)
		for (int j = 0; j < n2; j++)
			for (int i = 0; i < n1; i++)
			{
				x = (i + 1) * xx.h - source.x;
				y = (j + 1) * yy.h - source.y;
				z = (k + 1) * zz.h - source.z;
				r = sqrt(x * x + y * y + z * z);
				if (r >= r0 && r <= r_max)
				{
					x_sol_circ[i + n1 * j + size2D * k] = x_sol_nopml[i + n1 * j + size2D * k];
					x_orig_circ[i + n1 * j + size2D * k] = x_orig_nopml[i + n1 * j + size2D * k];
				}
			}

	norm = RelError(zlange, size, 1, x_sol_circ, x_orig_circ, size, thresh);

	printf("Square: 0 < x < %lf, 0 < y < %lf, 0 < z < %lf.\n", xx.l, yy.l, zz.l);
	printf("Norm in circle: %lf < r < %lf: %lf\n", r0, r_max, norm);

	free_arr(x_sol_circ);
	free_arr(x_orig_circ);

	return norm;
}

double check_norm_circle2D(size_m xx, size_m yy, int start, int end, dtype* x_orig, dtype* x_sol, point source, double thresh)
{
	int n1 = xx.n;
	int n2 = yy.n;

	int size2D = n1 * n2;

	double x, y;
	double r;
	//double r0 = (5 + xx.pml_pts) * xx.h - source.x;
	//double r_max = xx.l - (5 + xx.pml_pts) * xx.h - source.x;
	//double r_max = xx.l - (25 + xx.pml_pts) * xx.h - source.x;
	double norm;

	double r0 = start * xx.h;
	double r_max = end * xx.h;

	int pts_inside_circle = 0;
	int pts_all_domain = 0;

	dtype* x_sol_circ = alloc_arr<dtype>(size2D);
	dtype* x_orig_circ = alloc_arr<dtype>(size2D);

	for (int j = yy.pml_pts; j < yy.n - yy.pml_pts; j++)
		for (int i = xx.pml_pts; i < xx.n - xx.pml_pts; i++)
		{
			x = (i + 1) * xx.h - source.x;
			y = (j + 1) * yy.h - source.y;
			r = sqrt(x * x + y * y);
			if (r >= r0 && r <= r_max)
			{
				x_sol_circ[i + n1 * j] = x_sol[i + n1 * j];
				x_orig_circ[i + n1 * j] = x_orig[i + n1 * j];
				pts_inside_circle++;
			}
			pts_all_domain++;
		}

	norm = RelError(zlange, size2D, 1, x_sol_circ, x_orig_circ, size2D, thresh);

	//printf("Square: 0 < x < %lf, 0 < y < %lf\n", xx.l, yy.l);
	printf("RelError in circle: %lf < r < %lf: %e\n", r0, r_max, norm);
	printf("Points inside circle = %d. All phycial domain = %d\n", pts_inside_circle, pts_all_domain);

#if 0
	char str[255]; sprintf(str, "CircleSolOrig_N%d.dat", xx.n);
	FILE *out = fopen(str, "w");
	for (int i = 0; i < size2D; i++)
		fprintf(out, "%d %lf %lf\n", i, x_sol_circ[i].real(), x_orig_circ[i].imag());

	fclose(out);
#endif

	free_arr(x_sol_circ);
	free_arr(x_orig_circ);

	return norm;
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
	int size2D = x.n * y.n;
	int size = size2D * z.n;
	int l = 0;

#ifndef TEST_HELM_1D
	SetRHS3D(x, y, z, f, source, l);

	// approximation of inner points values
#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < z.n; k++)
		for (int j = 0; j < y.n; j++)
			for (int i = 0; i < x.n; i++)
				u[k * size2D + j * x.n + i] = u_ex_complex_sound3D(x, y, z, (i + 1) * x.h, (j + 1) * y.h, (k + 1) * z.h, source);
#else
	SetRHS3DForTest(x, y, z, f, source, l);

	dtype *u_ex1D = alloc_arr<dtype>(z.n);

	GenExact1DHelmholtz(z.n, z, u_ex1D, kk, source);

	for (int k = 0; k < z.n; k++)
		for (int w = 0; w < size2D; w++) // for each point (x,z) on 2D domain we construct solution in direction Z
		{
			u[w + k * size2D] = u_ex1D[k];
		}
#endif


	printf("RHS and solution are constructed\n");
}

void GenRHSandSolutionViaSound3D_Hetero(char *file, size_m x, size_m y, size_m z, /* output */ dtype *u, dtype *f, point source)
{
	int size2D = x.n * y.n;
	int size = size2D * z.n;
	int l = 0;

	SetRHS3D(x, y, z, f, source, l);

	FILE *in = fopen(file, "r");

	// approximation of inner points values
#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < z.n; k++)
		for (int j = 0; j < y.n; j++)
			for (int i = 0; i < x.n; i++)
				fscanf(in, "%lf\n", u[k * size2D + j * x.n + i]);

	fclose(in);
	printf("RHS and solution are constructed\n");
}

void check_test_3Dsolution_in1D(int n1, int n2, int n3, dtype* u_sol, dtype *u_ex, double thresh)
{
	int size2D = n1 * n2;
	int size = size2D * n3;
	int l = 0;
	double norm;

	dtype *u_ex1D = alloc_arr<dtype>(n3);
	dtype *u_sol1D = alloc_arr<dtype>(n3);

	Test1DHelmholtz(n1, n2, n3, u_ex, thresh, "EXACT");
	Test1DHelmholtz(n1, n2, n3, u_sol, thresh, "NUM");	

	FILE *out = fopen("TEST_HELM_1D.dat", "w");

	for (int w = 0; w < size2D; w++) // for each point (x,z) on 2D domain we construct solution in direction Z
	{
		for (int k = 0; k < n3; k++)
		{
			u_ex1D[k] = u_ex[w + k * size2D];
			u_sol1D[k] = u_sol[w + k * size2D];
			if (w == size2D / 4) fprintf(out, "%d %lf %lf %lf %lf\n", k, u_ex1D[k].real(), u_ex1D[k].imag(), u_sol1D[k].real(), u_sol1D[k].imag());
		}

		norm = RelError(zlange, n3, 1, u_sol1D, u_ex1D, n3, thresh);
	//	printf("Norm %d = %lf\n", w, norm);
	}

	fclose(out);
}

double d(double x)
{
	//printf("x = %lf\n", x);
	// 100
	const double C = 1000;
	return C * x * x;

	//const double C = 1.25;
	//return C * pow(x, 4);

	// 10 * x ^ 2, проверить , что значения 0 и С на границах
	// - 15 h < x < 0
	// кси , n !=  (1, 0)
}

dtype MakeSound2D(size_m xx, size_m yy, double x, double y, point source, double beta_eq)
{
	return MakeSound2D(xx, yy, x, y, source, beta_eq) * sqrt(dtype{ 1, -beta_eq } / (1 + beta_eq * beta_eq));
}

dtype MakeSound3D(size_m xx, size_m yy, size_m zz, double x, double y, double z, point source)
{
	dtype c1 = (double)C1;
	dtype c2 = (double)C2;
	dtype c3 = (double)C3;
#ifdef HOMO
	return c_z;
#else
	//return c_z + c1 * x + c2 * y + c3 * z;
	return c1 * x + c2 * y + c3 * z;
#endif
}

double Runge(size_m x1, size_m x2, size_m x3,
	size_m y1, size_m y2, size_m y3,
	size_m z1, size_m z2, size_m z3,
	char *str1, char *str2, char *str3, int dim)
{
	printf("Count RUNGE order...\n");
	// the lowest size (50 from line 50, 100, 200)
	int size1, size2, size3;

	if (dim == 3)
	{
		x1.n_nopml = y1.n_nopml = z1.n_nopml = 49;
		x2.n_nopml = y2.n_nopml = z2.n_nopml = 99;
		x3.n_nopml = y3.n_nopml = z3.n_nopml = 199;

		size1 = x1.n_nopml * y1.n_nopml * z1.n_nopml;
		size2 = x2.n_nopml * y2.n_nopml * z2.n_nopml;
		size3 = x3.n_nopml * y3.n_nopml * z3.n_nopml;
	}
	else if (dim == 2)
	{
		size1 = x1.n_nopml * y1.n_nopml;
		size2 = x2.n_nopml * y2.n_nopml;
		size3 = x3.n_nopml * y3.n_nopml;
	}
	else
	{
		size1 = x1.n_nopml;
		size2 = x2.n_nopml;
		size3 = x3.n_nopml;
	}
	FILE* file1, *file2, *file3;
	file1 = fopen(str1, "r");
	file2 = fopen(str2, "r");
	file3 = fopen(str3, "r");
	double a, b;
	double f1, f2, f3;
	int ione = 1;

	dtype *sol2h = alloc_arr<dtype>(size1); int lda2h = x1.n_nopml;
	dtype *solh = alloc_arr<dtype>(size2); int ldah = x2.n_nopml;
	dtype *solh2 = alloc_arr<dtype>(size3); int ldah2 = x3.n_nopml;

	for (int i = 0; i < size1; i++)
	{
		fscanf(file1, "%lf %lf\n", &a, &b);
		sol2h[i] = dtype{ a, b };
	}
	//printf("LAST 1 = %lf", sol2h[size1 - 1].real(), sol2h[size1 - 1].imag());

	for (int i = 0; i < size2; i++)
	{
		fscanf(file2, "%lf  %lf\n", &a, &b);
		solh[i] = dtype{ a, b };
	}
	//printf("LAST 2 = %lf", solh[size2 - 1].real(), solh[size2 - 1].imag());

	for (int i = 0; i < size3; i++)
	{
		fscanf(file3, "%lf %lf\n", &a, &b);
		solh2[i] = dtype{ a, b };
	}
	//printf("LAST 3 = %lf", solh2[size3 - 1].real(), solh2[size3 - 1].imag());

	f1 = dznrm2(&size1, sol2h, &ione);
	f2 = dznrm2(&size2, solh, &ione);
	f3 = dznrm2(&size3, solh2, &ione);

	//printf("Runge (||u1|| - ||u2||) / (||u2|| - ||u3||) = %lf\n", (f1 - f2) / (f2 - f3));

#if 0
	int step1 = (x2.n + 1) / (x1.n + 1);
	int step2 = (x3.n + 1) / (x1.n + 1);

	printf("step1 = %d, step2 = %d\n", step1, step2);

	dtype *hlp2h = alloc_arr<dtype>(size1);
	dtype *hlph = alloc_arr<dtype>(size1);

	for (int j = 0; j < y1.n_nopml; j++)
		for (int i = 0; i < x1.n_nopml; i++)
		{
			hlp2h[i + lda2h * j] = sol2h[i + lda2h * j] - solh2[step2 * (i + ldah2 * j)];
			hlph[i + lda2h * j] = solh[step1 * (i + ldah * j)] - solh2[step2 * (i + ldah2 * j)];
		}
	f1 = dznrm2(&size1, hlp2h, &ione) * sqrt(x1.h);
	f2 = dznrm2(&size1, hlph, &ione) * sqrt(x2.h);
#else
	
	int step1 = (x3.n_nopml + 1) / (x1.n_nopml + 1);
	int step2 = (x3.n_nopml + 1) / (x2.n_nopml + 1);

	printf("step1 = %d, step2 = %d\n", step1, step2);

	dtype *hlp2h = alloc_arr<dtype>(size1);
	dtype *hlph = alloc_arr<dtype>(size2);

#if 0
	for (int j = 0; j < y1.n; j++)
		for (int i = 0; i < x1.n; i++)
			hlp2h[i + lda2h * j] = sol2h[i + lda2h * j] - solh2[step1 * (i + 1) - 1 + ldah2 * (step1 * (j + 1) - 1)];
	// 0, 1, 2, 3, 4, 5, 6 vs 3, 7, 11

	printf("size1 = %d, (i + lda2h * j)last = %d, (i + ldah2 * j) * step1 = %d\n", size1, x1.n - 1 + lda2h * (y1.n - 1), step1 * (x1.n) - 1 + ldah2 * (step1 * (y1.n) - 1));

	for (int j = 0; j < y2.n; j++)
		for (int i = 0; i < x2.n; i++)
			hlph[i + ldah * j] = solh[i + ldah * j] - solh2[step2 * (i + 1) - 1 + ldah2 * (step2 * (j + 1) - 1)];
	// 0, 1, 2, 3, 4, 5, 6 vs 1, 3, 5, 7, 9, 11

	printf("size2 = %d, (i + ldah * j)last = %d, (i + ldah2 * j) * step2 = %d\n", size2, x2.n - 1 + ldah * (y2.n - 1), step2 * (x2.n) - 1 + ldah2 * (step2 * y2.n - 1));

	//f1 = dznrm2(&size1, hlp2h, &ione) * sqrt(x1.h);
	//f2 = dznrm2(&size2, hlph, &ione) * sqrt(x2.h);
	f1 = zlange("F", &size1, &ione, hlp2h, &lda2h, NULL);// *sqrt(x1.h);
	f2 = zlange("F", &size2, &ione, hlph, &ldah, NULL);// *sqrt(x2.h);
#else
	double eps = 1e-6;
	if (dim == 2)
	{
		for (int j = 0; j < y1.n_nopml; j++)
			for (int i = 0; i < x1.n_nopml; i++)
				hlp2h[i + lda2h * j] = solh2[step1 * (i + 1) - 1 + ldah2 * (step1 * (j + 1) - 1)];
		// 0, 1, 2, 3, 4, 5, 6 vs 3, 7, 11

		printf("size1 = %d, (i + lda2h * j)last = %d, (i + ldah2 * j) * step1 = %d\n", size1, x1.n - 1 + lda2h * (y1.n - 1), step1 * (x1.n) - 1 + ldah2 * (step1 * (y1.n) - 1));

		for (int j = 0; j < y2.n_nopml; j++)
			for (int i = 0; i < x2.n_nopml; i++)
				hlph[i + ldah * j] = solh2[step2 * (i + 1) - 1 + ldah2 * (step2 * (j + 1) - 1)];

		f1 = RelError2(zlange, x1.n_nopml, y1.n_nopml, sol2h, lda2h, hlp2h, lda2h, eps);
		f2 = RelError2(zlange, x2.n_nopml, y2.n_nopml, solh, ldah, hlph, ldah, eps);
	}
	else if (dim == 3)
	{
		for (int k = 0; k < z1.n_nopml; k++)
			for (int j = 0; j < y1.n_nopml; j++)
				for (int i = 0; i < x1.n_nopml; i++)
					hlp2h[i + x1.n_nopml * j + x1.n_nopml * y1.n_nopml * k] = solh2[step1 * (i + 1) - 1 + x3.n_nopml * (step1 * (j + 1) - 1) + x3.n_nopml * y3.n_nopml * (step1 * (k + 1) - 1)];
		// 0, 1, 2, 3, 4, 5, 6 vs 3, 7, 11

		//printf("size1 = %d, (i + lda2h * j)last = %d, (i + ldah2 * j) * step1 = %d\n", size1, x1.n - 1 + lda2h * (y1.n - 1), step1 * (x1.n) - 1 + ldah2 * (step1 * (y1.n) - 1));

		for (int k = 0; k < z2.n_nopml; k++)
			for (int j = 0; j < y2.n_nopml; j++)
				for (int i = 0; i < x2.n_nopml; i++)
					hlph[i + x2.n_nopml * j + x2.n_nopml * y2.n_nopml * k] = solh2[step2 * (i + 1) - 1 + x3.n_nopml * (step2 * (j + 1) - 1) + x3.n_nopml * y3.n_nopml * (step2 * (k + 1) - 1)];

		// 0, 1, 2, 3, 4, 5, 6 vs 1, 3, 5, 7, 9, 11
#if 0
		f1 = RelError2(zlange, size1, 1, sol2h, size1, hlp2h, size1, eps);
		f2 = RelError2(zlange, size2, 1, solh, size2, hlph, size2, eps);
#else
		int cur_z_1 = (z1.n_nopml + 1) / 2;
		int cur_z_2 = cur_z_1 * 2 + 1;

		cur_z_1 = 1;
		cur_z_2 = 3;

		for (int k = 0; k < z1.n_nopml; k++)
			for (int j = 0; j < y1.n_nopml; j++)
				for (int i = 0; i < x1.n_nopml; i++)
				{
					if(1)
					//if (k == cur_z_1)
					{
						hlp2h[i + x1.n_nopml * j + x1.n_nopml * y1.n_nopml * k] -= sol2h[i + x1.n_nopml * j + x1.n_nopml * y1.n_nopml * k];
					}
					else
					{
						hlph[i + x1.n_nopml * j + x1.n_nopml * y1.n_nopml * k] = 0;
					}
				}

		for (int k = 0; k < z2.n_nopml; k++)
			for (int j = 0; j < y2.n_nopml; j++)
				for (int i = 0; i < x2.n_nopml; i++)
				{
					if (1)
					//if (k == cur_z_2)
					{
						hlph[i + x2.n_nopml * j + x2.n_nopml * y2.n_nopml * k] -= solh[i + x2.n_nopml * j + x2.n_nopml * y2.n_nopml * k];
					}
					else
					{
						hlph[i + x2.n_nopml * j + x2.n_nopml * y2.n_nopml * k] = 0;
					}
				}
		f1 = IntegralNorm3D(x1, y1, z1, hlp2h);
		f2 = IntegralNorm3D(x2, y2, z2, hlph);
#endif

	}
	// 0, 1, 2, 3, 4, 5, 6 vs 1, 3, 5, 7, 9, 11
#endif

	printf("size3 = %d\n", size3);
#endif

	free_arr(hlp2h);
	free_arr(hlph);
	free_arr(sol2h);
	free_arr(solh);
	free_arr(solh2);

	fclose(file1);
	fclose(file2);
	fclose(file3);

	printf("Runge ||u(2h) - u(h/2)|| = %e\n||u(h) - u(h/2)|| = %e\n", f1, f2);

	return f1 / f2;
}

double Beta3D(size_m x1, size_m x2,
	size_m y1, size_m y2,
	size_m z1, size_m z2,
	char *str1, char *str2, int dim, int grid)
{	
	printf("Count BETA_3D order for N = %d and N = %d \n", grid, 2 * grid);
	// the lowest size (50 from line 50, 100, 200)
	int size1, size2;
	point source = { 0.5, 0.5 };
	double thresh = 0.5;

	if (dim == 3)
	{
		x1.n_nopml = y1.n_nopml = z1.n_nopml = grid - 1;
		x2.n_nopml = y2.n_nopml = z2.n_nopml = 2 * grid - 1;

		size1 = x1.n_nopml * y1.n_nopml * z1.n_nopml;
		size2 = x2.n_nopml * y2.n_nopml * z2.n_nopml;
	}
	else if (dim == 2)
	{
		size1 = x1.n_nopml * y1.n_nopml;
		size2 = x2.n_nopml * y2.n_nopml;
	}
	else
	{
		size1 = x1.n_nopml;
		size2 = x2.n_nopml;
	}
	FILE* file1, *file2;
	file1 = fopen(str1, "r");
	file2 = fopen(str2, "r");
	double a, b;
	double f1, f2;
	int ione = 1;

	dtype *sol2h = alloc_arr<dtype>(size1); int lda2h = x1.n_nopml;
	dtype *solh = alloc_arr<dtype>(size2); int ldah = x2.n_nopml;

	for (int i = 0; i < size1; i++)
	{
		fscanf(file1, "%lf %lf\n", &a, &b);
		sol2h[i] = dtype{ a, b };
		//sol2h[i] = dtype{ 0, 0 };
	}
	//printf("LAST 1 = %lf", sol2h[size1 - 1].real(), sol2h[size1 - 1].imag());

	for (int i = 0; i < size2; i++)
	{
		fscanf(file2, "%lf  %lf\n", &a, &b);
		solh[i] = dtype{ a, b };
	}
	//printf("LAST 2 = %lf", solh[size2 - 1].real(), solh[size2 - 1].imag());

	f1 = dznrm2(&size1, sol2h, &ione);
	f2 = dznrm2(&size2, solh, &ione);

	//printf("Runge (||u1|| - ||u2||) / (||u2|| - ||u3||) = %lf\n", (f1 - f2) / (f2 - f3));

	int step1 = (x2.n_nopml + 1) / (x1.n_nopml + 1);
	int step2 = (x2.n_nopml + 1) / (x2.n_nopml + 1);

	printf("step1 = %d step2 = %d\n", step1, step2);

	dtype *sol2h_red = alloc_arr<dtype>(size1);
	dtype *solh_red = alloc_arr<dtype>(size1); 

		for (int k = 0; k < z1.n_nopml; k++)
			for (int j = 0; j < y1.n_nopml; j++)
				for (int i = 0; i < x1.n_nopml; i++)
					sol2h_red[i + x1.n_nopml * j + x1.n_nopml * y1.n_nopml * k] = sol2h[step2 * (i + 1) - 1 + x1.n_nopml * (step2 * (j + 1) - 1) + x1.n_nopml * y1.n_nopml * (step2 * (k + 1) - 1)];

		for (int k = 0; k < z1.n_nopml; k++)
			for (int j = 0; j < y1.n_nopml; j++)
				for (int i = 0; i < x1.n_nopml; i++)
					solh_red[i + x1.n_nopml * j + x1.n_nopml * y1.n_nopml * k] = solh[step1 * (i + 1) - 1 + x2.n_nopml * (step1 * (j + 1) - 1) + x2.n_nopml * y2.n_nopml * (step1 * (k + 1) - 1)];

		printf("compute_and_print_circle_norm_nopml\n");
		// 0, 1, 2, 3, 4, 5, 6 vs 1, 3, 5, 7, 9, 11
		compute_and_print_circle_norm_nopml(x1, y1, z1, solh_red, sol2h_red, source, thresh);

	free_arr(sol2h_red);
	free_arr(solh_red);
	free_arr(sol2h);
	free_arr(solh);

	fclose(file1);
	fclose(file2);

	//printf("Norm3D ||u(2h) - u(h/2)|| / || u(h/2) || = %e\n", f1);

	return f1 / f2;
}

void compute_and_print_circle_norm(size_m x, size_m y, size_m z, dtype *x_orig, dtype *x_sol, point source, double thresh)
{
	double *norm_arr = alloc_arr<double>(x.n);
	char str_norm[255];
	sprintf(str_norm, "beta_3D_norm2_Nx%dNy%d_SPG%d.dat", x.n, y.n, z.spg_pts);

	FILE *beta_norm2 = fopen(str_norm, "w");
	for (int i = 2; i < x.n - x.pml_pts; i++)
	{
		norm_arr[i] = check_norm_circle_3D(x, y, z, 2, i, x_orig, x_sol, source, thresh);
		fprintf(beta_norm2, "%lf %lf\n", i * x.h, norm_arr[i]);
	}
	fclose(beta_norm2);
	free(norm_arr);

	FILE *fp = fopen("beta_3D_norm2.plt", "w");
	fprintf(fp, "set term png font \"Times - Roman, 16\"\n \
		set output 'beta_3D_norm2_Nx%dNy%d_SPG%d.png' \n \
		plot \'%s\' u 1:2 w linespoints pt 7 pointsize 1 notitle\n\n \
		exit\n", x.n, y.n, z.spg_pts, str_norm);

	fclose(fp);
	system("beta_3D_norm2.plt");
}

void compute_and_print_circle_norm_nopml(size_m x, size_m y, size_m z, dtype *x_orig_nopml, dtype *x_sol_nopml, point source, double thresh)
{
	double *norm_arr = alloc_arr<double>(x.n_nopml);
	char str_norm[255];

	sprintf(str_norm, "beta_3D_norm2_Nx%dNy%d_SPG%d.dat", x.n_nopml, y.n_nopml, z.spg_pts);

	FILE *beta_norm2 = fopen(str_norm, "w");
	for (int i = 2; i < x.n_nopml; i++)
	{
		norm_arr[i] = check_norm_circle_3D_nopml(x, y, z, 2, i, x_orig_nopml, x_sol_nopml, source, thresh);
		fprintf(beta_norm2, "%lf %lf\n", i * x.h, norm_arr[i]);
	}
	fclose(beta_norm2);
	free(norm_arr);

	FILE *fp = fopen("beta_3D_norm2.plt", "w");
	fprintf(fp, "set term png font \"Times - Roman, 16\"\n \
		set output 'beta_3D_norm2_Nx%dNy%d_SPG%d.png' \n \
		plot \'%s\' u 1:2 w linespoints pt 7 pointsize 1 notitle\n\n \
		exit\n", x.n_nopml, y.n_nopml, z.spg_pts, str_norm);

	fclose(fp);
	system("beta_3D_norm2.plt");
}

double IntegralNorm2D(size_m x, size_m y, char type, double* v)
{
	double norm = 0;
	
	if (type == 'I')
	{
		for (int i = 0; i < x.n; i++)
			for (int j = 0; j < y.n; j++)
				norm += v[i + x.n * j];

		norm *= x.h / 2;
	}
	else if (type == '1')
	{
		for (int i = 0; i < x.n; i++)
			for (int j = 0; j < y.n; j++)
				norm += abs(v[i + x.n * j]);

		norm *= x.h;
	}
	else if (type == 'F')
	{
		for (int i = 0; i < x.n; i++)
			for (int j = 0; j < y.n; j++)
				norm += v[i + x.n * j] * v[i + x.n * j];

		norm = sqrt(norm);
	}

	return norm;
}

double IntegralNorm3D(size_m x, size_m y, size_m z, dtype* v)
{
	double norm = 0;

	for (int k = 0; k < z.n_nopml; k++)
		for (int j = 0; j < y.n_nopml; j++)
			for (int i = 0; i < x.n_nopml; i++)
				norm += abs(v[i + x.n_nopml * j + x.n_nopml * y.n_nopml * k]);

	norm *= x.h;

	return norm;
}

void SetSoundSpeed3D(size_m x, size_m y, size_m z, dtype* sound3D, point source)
{
	int n = x.n * y.n;
	int Nx = x.n;
	int Ny = y.n;
	int Nz = z.n;
	printf("z.spg_pts = %d\n", z.spg_pts);

#if 0
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
#else
	for (int k = 0; k < Nz; k++)
			for (int j = 0; j < Ny; j++)
				for (int i = 0; i < Nx; i++)
					sound3D[k * n + j * Nx + i] = MakeSound3D(x, y, z, (i + 1) * x.h, (j + 1) * y.h, (k + 1) * z.h, source);
#endif

#if 0
	char str[255]; sprintf(str, "SoundSpeed3D_N%d.dat", x.n);
	FILE* file = fopen(str, "w");

	for (int k = z.spg_pts; k < Nz - z.spg_pts; k++)
		for (int j = y.pml_pts; j < Ny - y.pml_pts; j++)
			for (int i = x.pml_pts; i < Nx - x.pml_pts; i++)
				fprintf(file, "sound[%d][%d][%d] = %lf\n", i, j, k, sound3D[k * n + j * Nx + i].real());

	fclose(file);
#endif
}

void SetSoundSpeed2D(size_m x, size_m y, size_m z, dtype* sound3D, dtype* sound2D, point source)
{
	int n = x.n * y.n;
	int Nx = x.n;
	int Ny = y.n;
	int Nz = z.n;
	dtype c0_max;
	dtype c0_min;

	char str[255]; sprintf(str, "SoundSpeed2D_N%d.dat", x.n);
	FILE *file = fopen(str, "w");

#if 1
	for (int j = y.pml_pts; j < Ny - y.pml_pts; j++)
		for (int i = x.pml_pts; i < Nx - x.pml_pts; i++)
		{
			c0_max = sound3D[z.spg_pts * n + j * Nx + i];
			c0_min = sound3D[z.spg_pts * n + j * Nx + i];

			for (int k = z.spg_pts; k < Nz - z.spg_pts; k++)
			{
					if (abs(sound3D[k * n + j * Nx + i]) > abs(c0_max)) c0_max = sound3D[k * n + j * Nx + i];
					if (abs(sound3D[k * n + j * Nx + i]) < abs(c0_min)) c0_min = sound3D[k * n + j * Nx + i];
			}

			sound2D[j * Nx + i] = 0.5 * (c0_max + c0_min);
			fprintf(file, "sound[%d][%d] = %lf\n", i, j, sound2D[j * Nx + i].real());
		}
#else
#pragma omp parallel for schedule(dynamic)
		for (int j = 0; j < Ny; j++)
			for (int i = 0; i < Nx; i++)
				sound2D[j * Nx + i] = MakeSound2D(x, y, (i + 1) * x.h, (j + 1) * y.h, source);
#endif
	
		fclose(file);
}

void GenerateDeltaL(size_m x, size_m y, size_m z, dtype* sound3D, dtype* sound2D, dtype* deltaL, double beta_eq)
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

	double sigma = 0.3;
	double zp = 0;
	double z0 = z.spg_pts * z.h;
	double zN = (z.n - z.spg_pts) * z.h;
	double L0 = z0;
	double Ln = z.l - zN;

	for (int k = 0; k < nz; k++)
		for (int j = 0; j < ny; j++)
			for (int i = 0; i < nx; i++)
			{
				ij = i + j * nx;
				ijk = ij + k * size2D;
				deltaL[ijk] = double(omega) * double(omega) * (dtype{ 1.0, beta_eq } / (sound2D[ij] * sound2D[ij]) - 1.0 / (sound3D[ijk] * sound3D[ijk]));

				zp = k * z.h;
				if (k < z.spg_pts)
				{
					deltaL[ijk] *= exp(-(zp - z0) * (zp - z0) / (L0 * L0) / (sigma * sigma));
				}
				else if (k >= z.n - z.spg_pts)
				{
					deltaL[ijk] *= exp(-(zp - zN) * (zp - zN) / (Ln * Ln) / (sigma * sigma));
				}
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
	double h;

	if (xyz.pml_pts == 0) return 1.0;
	else h = 1.0 / xyz.pml_pts;

	if (i < xyz.pml_pts || i >= (xyz.n - xyz.pml_pts))
	{
		if (i < xyz.pml_pts)
		{
			//x = (double)(xyz.pml_pts - i) * h;
			x = 1.0 - i * h;

			// i = 0 ,       x = 1.0
			// i = pml - 1 , x = h 

			//if (x < 0 || x > 1)  printf("left: i = %lf x = %lf\n", i, x);
			//if ((abs(x - 0) < EPS_ZERO) || (abs(x - 1) < EPS_ZERO)) printf("left: i = %lf x = %lf\n", i, x);
		}
		else if (i >= (xyz.n - xyz.pml_pts))
		{
			//x = (double)(i - xyz.n + 1 + xyz.pml_pts) * h;

			x = 1.0 + (i - xyz.n + 1) * h;


			// i = n - pml , x = h
			// i = n - 1   , x = 1.0

			//if (x < 0 || x > 1)  printf("right: i = %lf x = %lf\n", i, x);
			//if ((abs(x - 0) < EPS_ZERO) || (abs(x - 1) < EPS_ZERO)) printf("right: i = %lf x = %lf\n", i, x);
		}

		return dtype{ 0, -double(omega) } / dtype{ d(x), -double(omega) };
		//return dtype{ double(omega) * double(omega), -double(omega) * d(x) } / (double(omega) * double(omega) + d(x) * d(x));
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
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
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
	int nz = 0;

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
	if (nz != 0) printf("!!! UNDERFLOW IN HANKEL !!!\n");
	if (ierr != 0) printf("!!! ERROR IN HANKEL !!!\n");

	vmfree(vmblock);

	return { res_real[1], res_imag[1] };

#endif
}

dtype set_exact_2D_Hankel(double x, double y, dtype k, point source)
{
	x -= source.x;
	y -= source.y;

	double r = sqrt(x * x + y * y);

	if (abs(r) < EPS)
	{
		r = 0.005;
		//printf("MIIIIIIIIIIIIIIIIIIIIID\n");
	}


	return Hankel(k * r);
}

void get_exact_2D_Hankel(int Nx, int Ny, size_m x, size_m y, dtype* x_sol_ex, dtype k, point source)
{
	for (int j = 0; j < Ny; j++)
		for (int i = 0; i < Nx; i++)
		{
			x_sol_ex[j * Nx + i] = set_exact_2D_Hankel((i + 1) * x.h, (j + 1) * y.h, k, source);
			x_sol_ex[j * Nx + i] *= dtype{0, -0.25};
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

double resid_2D_Hankel(size_m x, size_m y, zcsr* D2csr, dtype* x_sol_ex, dtype* f2D, point source)
{
	int n = x.n;
	int size = n * y.n;
	double RelRes = 0;
	dtype *g = alloc_arr<dtype>(size);

	ResidCSR2DHelm(x, y, D2csr, x_sol_ex, f2D, g, source, RelRes);

	free_arr(g);
	return RelRes;
}

void GenerateDiagonal2DBlock(char* problem, int blk3D, size_m x, size_m y, size_m z, dtype *DD, int lddd, dtype *alpX, dtype *alpY, dtype *alpZ)
{
	int n = x.n * y.n;
	int size = n * z.n;
	double k = double(kk);

	// diagonal blocks in dense format
	//#pragma omp parallel for simd schedule(static)
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


void GenSparseMatrixOnline3D(size_m x, size_m y, size_m z, dtype* B, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, zcsr* Acsr)
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

dtype beta1D(size_m x, int diag_case, double k2, int i, double beta_eq)
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

	//	value = -alpha(x, i + 1) * (alpha(x, i + 1.5) + alpha(x, i + 0.5)) / (x.h * x.h)
	//			-alpha(y, j + 1) * (alpha(y, j + 1.5) + alpha(y, j + 0.5)) / (y.h * y.h);

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

dtype beta2D_pml_9pts(size_m x, size_m y, int diag_case, dtype kwave_beta2, int i, int j, double sigma)
{
	if (diag_case == 0)
	{
		dtype value;

#ifdef PML
		value = -alpha(y, j) * (alpha(y, j + 0.5) + alpha(y, j - 0.5)) / (y.h * y.h)
				-(1 - 2.0 * sigma) * alpha(x, i) * (alpha(x, i + 0.5) + alpha(x, i - 0.5)) / (x.h * x.h);

		value += kwave_beta2;

#else 
		dtype c1 = -2.0 / (x.h * x.h);
		dtype c2 = -2.0 / (y.h * y.h);
		if (i == 0 || i == x.n - 1) c1 = dtype{ -1.0 / (x.h * x.h), -(double)kk / x.h };
		if (j == 0 || j == y.n - 1) c2 = dtype{ -1.0 / (y.h * y.h), -(double)kk / y.h };

		value = c1 + c2;
#endif

		int l = j * x.n + i;

		return value;
	}
	else if (diag_case == 1)
	{
		return (1 - 2.0 * sigma) * alpha(x, i) * alpha(x, i + 0.5) / (x.h * x.h);
	}
	else if (diag_case == -1)
	{
		return (1 - 2.0 * sigma) * alpha(x, i) * alpha(x, i - 0.5) / (x.h * x.h);
	}
	else if (diag_case == 3)
	{
		return -sigma * alpha(x, i) * (alpha(x, i + 0.5) + alpha(x, i - 0.5)) / (x.h * x.h)
			+ alpha(y, j) * alpha(y, j + 0.5) / (y.h * y.h);
	}
	else if (diag_case == -3)
	{
		return -sigma * alpha(x, i) * (alpha(x, i + 0.5) + alpha(x, i - 0.5)) / (x.h * x.h)
			+ alpha(y, j) * alpha(y, j - 0.5) / (y.h * y.h);
	}
	else if (diag_case == 2)
	{
		return sigma * alpha(x, i) * alpha(x, i + 0.5) / (x.h * x.h);
	}
	else if (diag_case == -2)
	{
		return sigma * alpha(x, i) * alpha(x, i - 0.5) / (x.h * x.h);
	}
	else if (diag_case == 4)
	{
		return sigma * alpha(x, i) * alpha(x, i + 0.5) / (x.h * x.h);
	}
	else if (diag_case == -4)
	{
		return sigma * alpha(x, i) * alpha(x, i - 0.5) / (x.h * x.h);
	}

	return 0;
}

dtype beta2D_pml_9pts_opt(size_m x, size_m y, int diag_case, dtype kwave_beta2, int i, int j, double a, double c, double d)
{
	if (diag_case == 0)
	{
		dtype value;

		value = -a * alpha(y, j) * (alpha(y, j + 0.5) + alpha(y, j - 0.5)) / (y.h * y.h)
			    -a * alpha(x, i) * (alpha(x, i + 0.5) + alpha(x, i - 0.5)) / (x.h * x.h)
			    -(1.0-a) * alpha(y, j) * (alpha(y, j + 0.5) + alpha(y, j - 0.5)) / (2.0 * y.h * y.h)
			    -(1.0-a) * alpha(x, i) * (alpha(x, i + 0.5) + alpha(x, i - 0.5)) / (2.0 * x.h * x.h)
				+ kwave_beta2 * c;

		return value;
	}
	else if (diag_case == 1)
	{
		return a * alpha(x, i) * alpha(x, i + 0.5) / (x.h * x.h) + kwave_beta2 * d;
	}
	else if (diag_case == -1)
	{
		return a * alpha(x, i) * alpha(x, i - 0.5) / (x.h * x.h) + kwave_beta2 * d;
	}
	else if (diag_case == 3)
	{
		return a * alpha(y, j) * alpha(y, j + 0.5) / (y.h * y.h) + kwave_beta2 * d;
	}
	else if (diag_case == -3)
	{
		return a * alpha(y, j) * alpha(y, j - 0.5) / (y.h * y.h) + kwave_beta2 * d;
	}
	else if (diag_case == 2)
	{
		return (1.0 - a) * alpha(y, j) * alpha(y, j + 0.5) / (2.0 * y.h * y.h) + (1.0 - c - 4.0 * d) / 4.0 * kwave_beta2;
	}
	else if (diag_case == -2)
	{
		return (1.0 - a) * alpha(y, j) * alpha(y, j - 0.5) / (2.0 * y.h * y.h) + (1.0 - c - 4.0 * d) / 4.0 * kwave_beta2;
	}
	else if (diag_case == 4)
	{
		return (1.0 - a) * alpha(x, i) * alpha(x, i + 0.5) / (2.0 * x.h * x.h) + (1.0 - c - 4.0 * d) / 4.0 * kwave_beta2;
	}
	else if (diag_case == -4)
	{
		return (1.0 - a) * alpha(x, i) * alpha(x, i - 0.5) / (2.0 * x.h * x.h) + (1.0 - c - 4.0 * d) / 4.0 * kwave_beta2;
	}

	return 0;
}

dtype beta2D_pml_13pts(size_m x, size_m y, DIAG13 diag_case, dtype kwave_beta2, int i, int j)
{
#ifdef ORDER4
	double g1 = 9.0 / 8;
	double g2 = -1.0 / 24;
#else
	double g1 = 1.0;
	double g2 = 0.0;
#endif

	dtype value;
	int l;

	switch (diag_case)
	{
	case DIAG13::zero:

		value = -alpha(x, i) * (g1 * g1 * (alpha(x, i + 0.5) + alpha(x, i - 0.5)) + g2 * g2 * (alpha(x, i + 1.5) + alpha(x, i - 1.5))) / (x.h * x.h)
			    -alpha(y, j) * (g1 * g1 * (alpha(y, j + 0.5) + alpha(y, j - 0.5)) + g2 * g2 * (alpha(y, j + 1.5) + alpha(y, j - 1.5))) / (y.h * y.h);
		value += kwave_beta2;


		l = j * x.n + i;

		return value;

	case DIAG13::one:
		return alpha(x, i) * (g1 * g1 * alpha(x, i + 0.5) - g1 * g2 * (alpha(x, i - 0.5) + alpha(x, i + 1.5))) / (x.h * x.h);

	case DIAG13::m_one:
		return alpha(x, i) * (g1 * g1 * alpha(x, i - 0.5) - g1 * g2 * (alpha(x, i + 0.5) + alpha(x, i - 1.5))) / (x.h * x.h);

	case DIAG13::two:
		return alpha(x, i) * g1 * g2 * (alpha(x, i + 0.5) + alpha(x, i + 1.5)) / (x.h * x.h);

	case DIAG13::m_two:
		return alpha(x, i) * g1 * g2 * (alpha(x, i - 0.5) + alpha(x, i - 1.5)) / (x.h * x.h);

	case DIAG13::three:
		return alpha(x, i) * g2 * g2 * alpha(x, i + 1.5) / (x.h * x.h);

	case DIAG13::m_three:
		return alpha(x, i) * g2 * g2 * alpha(x, i - 1.5) / (x.h * x.h);

	case DIAG13::four:
		return alpha(y, j) * (g1 * g1 * alpha(y, j + 0.5) - g1 * g2 * (alpha(y, j - 0.5) + alpha(y, j + 1.5))) / (y.h * y.h);

	case DIAG13::m_four:
		return alpha(y, j) * (g1 * g1 * alpha(y, j - 0.5) - g1 * g2 * (alpha(y, j + 0.5) + alpha(y, j - 1.5))) / (y.h * y.h);

	case DIAG13::five:
		return alpha(y, j) * g1 * g2 * (alpha(y, j + 0.5) + alpha(y, j + 1.5)) / (y.h * y.h);

	case DIAG13::m_five:
		return alpha(y, j) * g1 * g2 * (alpha(y, j - 0.5) + alpha(y, j - 1.5)) / (y.h * y.h);

	case DIAG13::six:
		return alpha(y, j) * g2 * g2 * alpha(y, j + 1.5) / (y.h * y.h);

	case DIAG13::m_six:
		return alpha(y, j) * g2 * g2 * alpha(y, j - 1.5) / (y.h * y.h);

	default:
		return 0;
	}
	return 0;
}


dtype beta2D_spg(size_m x, size_m y, int diag_case, double k2, int i, int j, double beta_eq)
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

void Copy2DCSRMatrix(int size2D, int nonzeros, zcsr* &A, zcsr* &B)
{
	size2D++;
	
	MultVectorConst<int>(size2D, A->ia, 1, B->ia);
	MultVectorConst<int>(nonzeros, A->ja, 1, B->ja);
	MultVectorConst<dtype>(nonzeros, A->values, 1.0, B->values);

	B->non_zeros = A->non_zeros;
	B->solve = A->solve;
}

void HeteroSoundSpeed2DExtensionToPML(size_m x, size_m y, dtype *sound2D)
{
	int ii = 0, jj = 0;

	// left x PML
	for (int j = 0; j < y.n; j++)
	{
		ii = x.pml_pts; jj = j;
		for (int i = 0; i < x.pml_pts; i++)
			sound2D[i + x.n * j] = sound2D[ii + x.n * jj];
	}

	// right x PML
	for (int j = 0; j < y.n; j++)
	{
		ii = x.n - x.pml_pts - 1; jj = j;
		for (int i = x.n - x.pml_pts; i < x.n; i++)
			sound2D[i + x.n * j] = sound2D[ii + x.n * jj];
	}

	// upper y PML
	for (int i = 0; i < x.n; i++)
	{
		jj = y.pml_pts; ii = i;
		for (int j = 0; j < y.pml_pts; j++)
			sound2D[i + x.n * j] = sound2D[ii + x.n * jj];
	}

	// lower y PML
	for (int i = 0; i < x.n; i++)
	{
		jj = y.n - y.pml_pts - 1; ii = i;
		for (int j = y.n - y.pml_pts; j < y.n; j++)
			sound2D[i + x.n * j] = sound2D[ii + x.n * jj];
	}

}

dtype IntegralNorm(int size, dtype *vec, double h)
{
	dtype norm = 0;
	for (int i = 0; i < size; i++)
	{
		norm += vec[i];
	}
	norm *= h;

	return sqrt(norm);
}

dtype ratio_dot(int size, dtype alpha, dtype beta, dtype *s1, dtype *s2, dtype *s3)
{
	// return alpha / beta * (s1, s2) / (s1, s3)
	int ione = 1;
	dtype val = alpha / beta;
	dtype sc1, sc2;
	zdotc(&sc1, &size, s1, &ione, s2, &ione);
	zdotc(&sc2, &size, s1, &ione, s3, &ione);

	return val * sc1 / sc2;
}

void ModifyNumericalSolution(int size_nopml, dtype *x_sol_nopml, dtype* x_orig_nopml)
{
	int ione = 1;
	dtype s1, s2;
	zdotc(&s1, &size_nopml, x_sol_nopml, &ione, x_orig_nopml, &ione);
	zdotc(&s2, &size_nopml, x_orig_nopml, &ione, x_orig_nopml, &ione);

	dtype s = s1 / s2;

	printf("Scalar (x_n, x_t) / (x_t, x_t) = (%lf, %lf)\n", s.real(), s.imag());
	zscal(&size_nopml, &s, x_sol_nopml, &ione);
}


void ReadSolution(int size_nopml, dtype *x_sol, char *name)
{
	FILE *fin = fopen(name, "r");
	int i = 0;
	double real, imag;

	while (!feof(fin))
	{
		fscanf(fin, "%lf %lf", &real, &imag);
		x_sol[i++] = dtype{ real, imag };
	}

	if (i != size_nopml) printf("Reading error! Read = %d, Needed = %d", i, size_nopml);

	fclose(fin);
}
#include <complex>
void ReadBinSolution(long int nmodel, stype *sol, char *title)
{
	/*! \brief Save solution in the Phisical domain.
* \param [in] handle pointer to the handle structure
* \param [in] sol pointer to the solution
* \param [in] f dummy argument
* \param [in] title prefix of output info
*
* \return 0 on success
*/
	FILE *fp;
	char fname_bin[200];
	double fdum = -1;
	int lb;
	int n1, n2, n3;
	float hx, hy, hz;
	//=== save solution =============================

	sprintf(fname_bin, 
		"C:/Users/dklyuchi/Documents/Visual Studio 2017/Projects/Iterative Helholtz solver with PARDISO/x64/Release/hss_sol_%s.xyz.float.bin", title);

	//printf("FileName = %s\n", fname_bin);
	fp = fopen(fname_bin, "rb");
	//=== save mesh parameters ===============
	fread(&(n1), sizeof(int), 1, fp);
	fread(&(n2), sizeof(int), 1, fp);
	fread(&(n3), sizeof(int), 1, fp);
	fread(&(lb), sizeof(int), 1, fp);
//	printf("n1 = %d n2 = %d n3 = %d\n", n1, n2, n3);
	//=== save solution ===============
//	printf("printf sizeof stype = %d\n", sizeof(stype));
	int nmodel_loc = n1 * n2 * n3;

	
	fread(sol, sizeof(stype), nmodel_loc, fp);

#if 0
	float real, imag;
	int i, j, k;
	FILE *file = fopen("results.dat", "w");
	for (int l = 0; l < nmodel_loc; l++)
	{
		fread(&real, sizeof(float), 1, fp);
		fread(&imag, sizeof(float), 1, fp);
		//printf("sol[%d] = %10.8f %10.8f\n", i, real, imag);
		fprintf(file, "%d %10.8f %10.8f\n", l, real, imag);

		take_coord3D(n1, n2, n3, l, i, j, k);
		sol[i + j * n1 + k * n1 * n2] = stype{ real, imag };
		//real = 0;
		//imag = 0;
	}
	fclose(file);
#endif

#if 0
	char str[255];
	sprintf(str, "results_%s.dat", fname_bin);
	FILE *file = fopen(str, "w");
	for (int k = 0; k < n3; k++)
	{
		for (int j = 0; j < n2; j++)
			for (int i = 0; i < n1; i++)
			{

				fprintf(file, "%10.8f %10.8f\n",
					sol[i + j * n1 + k * n1 * n2].real(), sol[i + j * n1 + k * n1 * n2].imag());
				
			}
	}
	fclose(file);
#endif
	//for (int i = 0; i < nmodel_loc; i++)
		//printf("sol[%d] = %f %f\n", i, sol[i].real(), sol[i].imag());
	//=== save mesh parameters ===============
	fread(&(hx), sizeof(float), 1, fp);
	fread(&(hy), sizeof(float), 1, fp);
	fread(&(hz), sizeof(float), 1, fp);
//	printf("hx = %f hy = %f hz = %f\n", hx, hy, hz);
	//=== save Frequency ===============
	float freq_re, freq_im;
	fread(&(freq_re), sizeof(float), 1, fp);
	fread(&(freq_im), sizeof(float), 1, fp);
	stype freq = { freq_re, freq_im };

//	printf("freq = (%f, %f)\n", freq.real(), freq.imag());

//	printf("total kilo bytes = %d\n", (4 * sizeof(int) + nmodel_loc * sizeof(stype) + 5 * sizeof(float)) / 1024);

#if 0
	if (x.cPi != NULL) {
		fwrite(&(x.cPi[0]), sizeof(double), 1, fp);
		//    printf("x.cPi[0]=%f\n", x.cPi[0]);
	}
	else {
		fwrite(&fdum, sizeof(double), 1, fp);
	}
	if (x.cSi != NULL) {
		fwrite(&(x.cSi[0]), sizeof(double), 1, fp);
		//    printf("x.cSi[0]=%f\n", x.cPi[0]);
	}
	else {
		fwrite(&fdum, sizeof(double), 1, fp);
	}
	if (x.rhoi != NULL) {
		fwrite(&(x.rhoi[0]), sizeof(double), 1, fp);
		//    printf("x.rhoi[0]=%f\n", x.cPi[0]);
	}
	else {
		fwrite(&fdum, sizeof(double), 1, fp);
	}
#endif
	fclose(fp);
}

void GenSparseMatrixOnline3DwithPML(size_m x, size_m y, size_m z, dtype* B, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, zcsr* Acsr, double eps)
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

DIAG5 CheckDiag5Pts(size_m x, size_m y, int l1, int l2)
{
	if (l1 == l2)
	{
		return DIAG5::zero;
	}
	else if (l1 == l2 - 1 && (l1 + 1) % x.n != 0)
	{
		return DIAG5::one;
	}
	else if (l1 == l2 + 1 && l1 % x.n != 0)
	{
		return DIAG5::m_one;
	}
	else if (l1 == l2 - x.n)
	{
		return DIAG5::two;
	}
	else if (l1 == l2 + x.n)
	{
		return DIAG5::m_two;
	}
	else return DIAG5::not_a_diag;
}

void GenSparseMatrixOnline2DwithPML(int w, size_m x, size_m y, zcsr* Acsr, dtype kwave_beta2, int* freqs)
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
	//Acsr->values[count] = dtype{ kwave2, 0 };
	//Acsr->values[count] += dtype{ 0, k * k * beta_eq };
	//Acsr->values[count] = dtype{ k * k, 0 };
	//Acsr->values[count] -= dtype{ kww, 0 };


	if (non_zeros_in_2Dblock3diag != Acsr->non_zeros) printf("ERROR! Uncorrect value of non_zeros inside 2D: %d != %d\n", non_zeros_in_2Dblock3diag, Acsr->non_zeros);
//	else printf("Gen 2D matrix for frequency w = %d, k^2 - ky^2 = (%lf %lf)\n", w, kwave_beta2.real(), kwave_beta2.imag());

	if (w == -1)
	{
		for (int l1 = 0; l1 < size; l1++)
		{
			Acsr->ia[l1] = count + 1;
			take_coord2D(x.n, y.n, l1, j1, k1);
			for (int l2 = 0; l2 < size; l2++)
			{		
				take_coord2D(x.n, y.n, l2, j2, k2);
#ifdef SYMMETRY
				dtype alp = alpha(x, j1) * alpha(y, k1);
#else
				dtype alp = 1;
#endif
				DIAG5 diag = CheckDiag5Pts(x, y, l1, l2);

				if (diag == DIAG5::zero) freqs[count2++] = count;

				if (diag != DIAG5::not_a_diag)
				{
					Acsr->ja[count] = l2 + 1;
					Acsr->values[count++] = beta2D_pml(x, y, static_cast<int>(diag), kwave_beta2, j1, k1) / alp;

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
		for (int l = 0; l < size; l++)
		{
#ifdef SYMMETRY
			take_coord2D(x.n, y.n, l, j1, k1);
			dtype alp = alpha(x, j1) * alpha(y, k1);
#else
			dtype alp = 1;
#endif
			Acsr->values[freqs[l]] += kwave_beta2 / alp;		
		}
	
		//printf("SUCCESSED 2D generation for frequency %d!\n", w);
	}

	//print_2Dcsr_mat_to_file(x, y, Acsr, "2Dmat.dat");
}

void GenSparseMatrixOnline2DwithPMLFast(int w, size_m x, size_m y, zcsr* Acsr, dtype kwave_beta2, int* freqs)
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
		int l2;
		for (int l1 = 0; l1 < size; l1++)
		{
			Acsr->ia[l1] = count + 1;

			l2 = l1 - x.n;
			take_coord2D(x.n, y.n, l2, j1, k1);
			if (l2 >= 0)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta2D_pml(x, y, -2, kwave_beta2, j1, k1);
			}

			l2 = l1 - 1;
			take_coord2D(x.n, y.n, l2, j1, k1);
			if (l2 >= 0 && l1 % x.n != 0)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta2D_pml(x, y, -1, kwave_beta2, j1, k1);
			}

			l2 = l1;
			take_coord2D(x.n, y.n, l2, j1, k1);
			if (1)
			{
				Acsr->ja[count] = l2 + 1;
				freqs[count2++] = count;
				Acsr->values[count++] = beta2D_pml(x, y, 0, kwave_beta2, j1, k1);
			}

			l2 = l1 + 1;
			take_coord2D(x.n, y.n, l2, j1, k1);
			if (l2 < size && l2 % x.n != 0)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta2D_pml(x, y, 1, kwave_beta2, j1, k1);
			}


			l2 = l1 + x.n;
			take_coord2D(x.n, y.n, l2, j1, k1);
			if (l2 < size)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta2D_pml(x, y, 2, kwave_beta2, j1, k1);
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
#pragma omp parallel for simd schedule(static)
		for (int i = 0; i < size; i++)
			Acsr->values[freqs[i]] += kwave_beta2;

		//printf("SUCCESSED 2D generation for frequency %d!\n", w);
	}

	//print_2Dcsr_mat_to_file(x, y, Acsr, "2Dmat.dat");
}


DIAG9 CheckDiag9Pts(size_m x, size_m y, int l1, int l2)
{
	if (l1 == l2)
	{
		return DIAG9::zero;
	}
	else if (l1 == l2 - 1 && (l1 + 1) % x.n != 0)
	{
		return DIAG9::one;
	}
	else if (l1 == l2 + 1 && l1 % x.n != 0)
	{
		return DIAG9::m_one;
	}
	else if (l1 == l2 - x.n)
	{
		return DIAG9::three;
	}
	else if (l1 == l2 - x.n - 1 && (l1 + 1) % x.n != 0)
	{
		return DIAG9::four;
	}
	else if (l1 == l2 - x.n + 1 && l1 % x.n != 0)
	{
		return DIAG9::two;
	}
	else if (l1 == l2 + x.n)
	{
		return DIAG9::m_three;
	}
	else if (l1 == l2 + x.n + 1 && l1 % x.n != 0)
	{
		return DIAG9::m_four;
	}
	else if (l1 == l2 + x.n - 1 && (l1 + 1) % x.n != 0)
	{
		return DIAG9::m_two;
	}
	else return DIAG9::not_a_diag;
}

void GenSparseMatrixOnline2DwithPMLand9Points(int w, size_m x, size_m y, size_m z, zcsr* Acsr, dtype kwave_beta2, int* freqs, double sigma)
{
	int size = x.n * y.n;
	int size2 = size - y.n;
	int size3 = size - y.n;
	int size4 = size - x.n - y.n + 1;
	int non_zeros_in_2Dblock9diag = size + 2 * size2 * 2 + size4 * 4;
	double RelRes = 0;
	int j1, k1;
	int j2, k2;

	int count = 0;
	int count2 = 0;

#ifdef TEST_HELM_1D
	non_zeros_in_2Dblock3diag += y.n * 2;
	non_zeros_in_2Dblock3diag += x.n * 2;
#endif

	//double k = (double)kk;
	//double kww = 4.0 * PI * PI * (w - n2) * (w - n2) / (y.l * y.l);
	//double kww = 4 * PI * PI * (w - n2) * (w - n2);

	//printf("Number k = %lf\n", k);

	//printf("analytic non_zeros in PML function: %d\n", non_zeros_in_2Dblock3diag);


	if (non_zeros_in_2Dblock9diag != Acsr->non_zeros) printf("ERROR! Uncorrect value of non_zeros inside 2D 9diag: %d != %d\n", non_zeros_in_2Dblock9diag, Acsr->non_zeros);
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

					Acsr->values[count++] = beta2D_pml_9pts(x, y, 0, kwave_beta2, j1, k1, sigma);

				}
				else if (l1 == l2 - 1 && (l1 + 1) % x.n != 0)
				{
					Acsr->ja[count] = l2 + 1;
					Acsr->values[count++] = beta2D_pml_9pts(x, y, 1, kwave_beta2, j1, k1, sigma); // right
				}
				else if (l1 == l2 + 1 && l1 % x.n != 0)
				{
					Acsr->ja[count] = l2 + 1;
					Acsr->values[count++] = beta2D_pml_9pts(x, y, -1, kwave_beta2, j1, k1, sigma); // left
				}
				else if (l1 == l2 - x.n)
				{
					Acsr->ja[count] = l2 + 1;
					Acsr->values[count++] = beta2D_pml_9pts(x, y, 3, kwave_beta2, j1, k1, sigma); // right
				}
				else if (l1 == l2 - x.n - 1 && (l1 + 1) % x.n != 0)
				{
					Acsr->ja[count] = l2 + 1;
					Acsr->values[count++] = beta2D_pml_9pts(x, y, 4, kwave_beta2, j1, k1, sigma); // right
				}
				else if (l1 == l2 - x.n + 1 && l1 % x.n != 0)
				{
					Acsr->ja[count] = l2 + 1;
					Acsr->values[count++] = beta2D_pml_9pts(x, y, 2, kwave_beta2, j1, k1, sigma); // right
				}
				else if (l1 == l2 + x.n)
				{
					Acsr->ja[count] = l2 + 1;
					Acsr->values[count++] = beta2D_pml_9pts(x, y, -3, kwave_beta2, j1, k1, sigma); // left
				}
				else if (l1 == l2 + x.n + 1 && l1 % x.n != 0)
				{
					Acsr->ja[count] = l2 + 1;
					Acsr->values[count++] = beta2D_pml_9pts(x, y, -4, kwave_beta2, j1, k1, sigma); // left
				}
				else if (l1 == l2 + x.n - 1 && (l1 + 1) % x.n != 0)
				{
					Acsr->ja[count] = l2 + 1;
					Acsr->values[count++] = beta2D_pml_9pts(x, y, -2, kwave_beta2, j1, k1, sigma); // left
				}
			}

	//		printf("l1 = %d\n", l1);
	//		print_2Dcsr_mat(x, y, Acsr);

	//		system("pause");
		}

		if (non_zeros_in_2Dblock9diag != count || size != count2) printf("FAILED generation!!! %d != %d\n", non_zeros_in_2Dblock9diag, count);
		else
		{
			printf("SUCCESSED BASIC 2D generation!\n");
			printf("Non_zeros inside generating PML function: %d\n", count);
		}
	}
	else
	{
#pragma omp parallel for simd schedule(static)
		for (int i = 0; i < size; i++)
			Acsr->values[freqs[i]] += kwave_beta2;

		//printf("SUCCESSED 2D generation for frequency %d!\n", w);
	}

	//print_2Dcsr_mat_to_file(x, y, Acsr, "2Dmat.dat");
}


void GenSparseMatrixOnline2DwithPMLand9Pts(int w, size_m x, size_m y, zcsr* Acsr, dtype kwave_beta2, int* freqs)
{
	int size = x.n * y.n;
	int size2 = size - y.n;
	int size3 = size - y.n;
	int size4 = size - x.n - y.n + 1;
	int non_zeros_in_2Dblock9diag = size + 2 * size2 * 2 + size4 * 4;
	double RelRes = 0;
	int j1, k1;
	int j2, k2;

	int count = 0;
	int count2 = 0;

	//double a = 1.0;
	//double c = 0.85;
	//double d = 0.0;

	double a = 1.0;
    double c = 0.85;
    double d = 0.0;   // 0.1597, 0.038, 0.011

	//double a = 0.5461;
	//double c = 0.6248;
	//double d = 0.09381;

	//double a = 0.9556;
	//double c = 0.1808;
	//double d = 1.2768 / 4.0; //0.27, 0.046, 0.011

	//double a = 0.9622;
	//double a = 1.0;
	//double c = 0.5940;
	//double d = 0.4504 / 4.0;

	//double a = 0.588887;
	//double c = 0.616928;
	//double d = 0.409708 / 4.0;   // 0.16, 0.039, 0.009

	//double a = 0.6322;
	//double c = 0.6865;
	//double d = 0.2702 / 4;  // 0.16, 0.037, 0.0093

	// 5 points
	//double a = 0.6576;
	//double c = 0.7149;
	//double d = 0.2137 / 4;  //0.16, 0.037, 0.0091

	//double a = 0.6574;
	//double c = 0.6789;
	//double d = 0.3033 / 4; //0.165,  0.037, 0.0092

	//double a = 0.6618;
	//double c = 0.8208;
	//double d = 0.0202 / 4; // 0.17, 0.038, 0.092

	//double a = 0.7116;
	//double c = 0.8866;
	//double d = -0.1116 / 4; // 0.17, 0.036, 0.0089
		
	//double a = 1.0;
	//double c = 1.0;
	//double d = 0.0; //0.12

	//a = x.ta;
	//c = x.tc;
	//d = x.td;

#ifdef TEST_HELM_1D
	non_zeros_in_2Dblock3diag += y.n * 2;
	non_zeros_in_2Dblock3diag += x.n * 2;
#endif

	//double k = (double)kk;
	//double kww = 4.0 * PI * PI * (w - n2) * (w - n2) / (y.l * y.l);
	//double kww = 4 * PI * PI * (w - n2) * (w - n2);

	//printf("Number k = %lf\n", k);

	//printf("analytic non_zeros in PML function: %d\n", non_zeros_in_2Dblock3diag);


	if (non_zeros_in_2Dblock9diag != Acsr->non_zeros) printf("ERROR! Uncorrect value of non_zeros inside 2D 9diag: %d != %d\n", non_zeros_in_2Dblock9diag, Acsr->non_zeros);
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

				DIAG9 diag = CheckDiag9Pts(x, y, l1, l2);

				if (diag == DIAG9::zero) freqs[count2++] = count;

				if (diag != DIAG9::not_a_diag)
				{
					Acsr->ja[count] = l2 + 1;
					Acsr->values[count++] = beta2D_pml_9pts_opt(x, y, static_cast<int>(diag), kwave_beta2, j1, k1, a, c, d);
				}
			}

			//		printf("l1 = %d\n", l1);
			//		print_2Dcsr_mat(x, y, Acsr);

			//		system("pause");
		}

		if (non_zeros_in_2Dblock9diag != count || size != count2) printf("FAILED generation!!! %d != %d\n", non_zeros_in_2Dblock9diag, count);
		else
		{
#ifdef PRINT_TEST
			printf("SUCCESSED BASIC 2D generation!\n");
			printf("Non_zeros inside generating PML function: %d\n", count);
#endif
		}
	}
	else
	{
#pragma omp parallel for simd schedule(static)
		for (int i = 0; i < size; i++)
			Acsr->values[freqs[i]] += kwave_beta2;

		//printf("SUCCESSED 2D generation for frequency %d!\n", w);
	}

	//print_2Dcsr_mat_to_file(x, y, Acsr, "2Dmat.dat");
}
#if 0
void GenSparseMatrixOnline2DwithPMLand13Pts(int w, size_m x, size_m y, zcsr* Acsr, dtype kwave_beta2, int* freqs)
{
	int size = x.n * y.n;
	int size2 = size - 1 * y.n;
	int size3 = size - 2 * y.n;
	int size4 = size - 3 * y.n;
	int size5 = size - 1 * x.n;
	int size6 = size - 2 * x.n;
	int size7 = size - 3 * x.n;
	int non_zeros_in_2Dblock13diag = size + 2 * size2 + 2 * size3 + 2 * size4 + 2 * size5 + 2 * size6 + 2 * size7;
	double RelRes = 0;
	int j1, k1;
	int j2, k2;

	int count = 0;
	int count2 = 0;

#ifdef TEST_HELM_1D
	non_zeros_in_2Dblock3diag += y.n * 2;
	non_zeros_in_2Dblock3diag += x.n * 2;
#endif


	if (non_zeros_in_2Dblock13diag != Acsr->non_zeros) printf("ERROR! Uncorrect value of non_zeros inside 2D 13diag: %d != %d\n", non_zeros_in_2Dblock13diag, Acsr->non_zeros);
	//	else printf("Gen 2D matrix for frequency w = %d, k^2 - ky^2 = (%lf %lf)\n", w, kwave_beta2.real(), kwave_beta2.imag());


//#define DEBUG

	if (w == -1)
	{
		// строки
		for (int l1 = 0; l1 < size; l1++)
		{
			Acsr->ia[l1] = count + 1;
			// столбцы
			for (int l2 = 0; l2 < size; l2++)
			{
				take_coord2D(x.n, y.n, l1, j1, k1);
				take_coord2D(x.n, y.n, l2, j2, k2);

				if (l1 == l2)
				{
					Acsr->ja[count] = l2 + 1;
					freqs[count2++] = count;
#ifdef DEBUG
					Acsr->values[count++] = static_cast<int>(DIAG13::zero);
#else
					Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::zero, kwave_beta2, j1, k1);
#endif

				}
				else if (l1 == l2 - 1 && (l1 + 1) % x.n != 0)
				{
					Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
					Acsr->values[count++] = static_cast<int>(DIAG13::one);
#else
					Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::one, kwave_beta2, j1, k1); // right
#endif
				}
				else if (l1 == l2 + 1 && l1 % x.n != 0)
				{
					Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
					Acsr->values[count++] = static_cast<int>(DIAG13::m_one);
#else
					Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::m_one, kwave_beta2, j1, k1); // left
#endif
				}
				else if (l1 == l2 - 2 && (l1 + 1) % x.n != 0 && (l1 + 2) % x.n != 0)
				{
					// не пишем, если номер строки + 1, 2, 3 строки кратен Nx
					Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
					Acsr->values[count++] = static_cast<int>(DIAG13::two);
#else
					Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::two, kwave_beta2, j1, k1); // right
#endif
				}
				else if (l1 == l2 + 2 && l1 % x.n != 0 && (l1 - 1) % x.n != 0)
				{
					Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
					Acsr->values[count++] = static_cast<int>(DIAG13::m_two);
#else
					Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::m_two, kwave_beta2, j1, k1); // left
#endif
				}
				else if (l1 == l2 - 3 && (l1 + 1) % x.n != 0 && (l1 + 2) % x.n != 0 && (l1 + 3) % x.n != 0)
				{
					Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
					Acsr->values[count++] = static_cast<int>(DIAG13::three);
#else
					Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::three, kwave_beta2, j1, k1); // right
#endif
				}
				else if (l1 == l2 + 3 && l1 % x.n != 0 && (l1 - 1) % x.n != 0 && (l1 - 2) % x.n != 0)
				{
					// не пишем, если номер строки -1, 2 строки кратен Nx
					Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
					Acsr->values[count++] = static_cast<int>(DIAG13::m_three);
#else
					Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::m_three, kwave_beta2, j1, k1); // left
#endif
				}
				else if (l1 == l2 - x.n)
				{
					Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
					Acsr->values[count++] = static_cast<int>(DIAG13::four);
#else
					Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::four, kwave_beta2, j1, k1); // right
#endif
				}
				else if (l1 == l2 + x.n)
				{
					Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
					Acsr->values[count++] = static_cast<int>(DIAG13::m_four);
#else
					Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::m_four, kwave_beta2, j1, k1); // left
#endif
				}
				else if (l1 == l2 - 2 * x.n)
				{
					Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
					Acsr->values[count++] = static_cast<int>(DIAG13::five);
#else
					Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::five, kwave_beta2, j1, k1); // right
#endif
				}
				else if (l1 == l2 + 2 * x.n)
				{
					Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
					Acsr->values[count++] = static_cast<int>(DIAG13::m_five);
#else
					Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::m_five, kwave_beta2, j1, k1); // left
#endif
				}
				else if (l1 == l2 - 3 * x.n)
				{
					Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
					Acsr->values[count++] = static_cast<int>(DIAG13::six);
#else
					Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::six, kwave_beta2, j1, k1); // right
#endif
				}
				else if (l1 == l2 + 3 * x.n)
				{
					Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
					Acsr->values[count++] = static_cast<int>(DIAG13::m_six);
#else
					Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::m_six, kwave_beta2, j1, k1); // left
#endif	
				}
			}
#ifdef DEBUG
			//		printf("l1 = %d\n", l1);
			//		print_2Dcsr_mat(x, y, Acsr);

			//		system("pause");
#endif
		}

		if (non_zeros_in_2Dblock13diag != count || size != count2) printf("FAILED generation!!! %d != %d\n", non_zeros_in_2Dblock13diag, count);
		else
		{
			printf("SUCCESSED BASIC 2D generation!\n");
			printf("Non_zeros inside generating PML function: %d\n", count);
		}
	}
	else
	{
#pragma omp parallel for simd schedule(static)
		for (int i = 0; i < size; i++)
			Acsr->values[freqs[i]] += kwave_beta2;

	//	printf("SUCCESSED 2D generation for frequency %d!\n", w);
	}

	print_2Dcsr_mat_to_file(x, y, Acsr, "2Dmat13pts_old.dat");
#undef DEBUG
}
#else
void GenSparseMatrixOnline2DwithPMLand13Pts(int w, size_m x, size_m y, zcsr* Acsr, dtype kwave_beta2, int* freqs)
{
	int size = x.n * y.n;
	int size2 = size - 1 * y.n;
	int size3 = size - 2 * y.n;
	int size4 = size - 3 * y.n;
	int size5 = size - 1 * x.n;
	int size6 = size - 2 * x.n;
	int size7 = size - 3 * x.n;
	int non_zeros_in_2Dblock13diag = size + 2 * size2 + 2 * size3 + 2 * size4 + 2 * size5 + 2 * size6 + 2 * size7;
	double RelRes = 0;
	int j1, k1;

	int count = 0;
	int count2 = 0;
	int l2;


	if (non_zeros_in_2Dblock13diag != Acsr->non_zeros) printf("!!!ERROR!!! Uncorrect value of non_zeros inside 2D 13diag: %d != %d\n", non_zeros_in_2Dblock13diag, Acsr->non_zeros);
	//	else printf("Gen 2D matrix for frequency w = %d, k^2 - ky^2 = (%lf %lf)\n", w, kwave_beta2.real(), kwave_beta2.imag());

//#define DEBUG

	if (w == -1)
	{
		// строки
		for (int l1 = 0; l1 < size; l1++)
		{
			Acsr->ia[l1] = count + 1;
			// столбцы

			l2 = l1 - 3 * x.n;
			take_coord2D(x.n, y.n, l1, j1, k1);
			if (l2 >= 0)
			{
				Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
				Acsr->values[count++] = static_cast<int>(DIAG13::m_six);
#else
				Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::m_six, kwave_beta2, j1, k1); // left
#endif	
			}

			l2 = l1 - 2 * x.n;
			take_coord2D(x.n, y.n, l1, j1, k1);
			if (l2 >= 0)
			{
				Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
				Acsr->values[count++] = static_cast<int>(DIAG13::m_five);
#else
				Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::m_five, kwave_beta2, j1, k1); // left
#endif
			}

			l2 = l1 - x.n;
			take_coord2D(x.n, y.n, l1, j1, k1);
			if (l2 >= 0)
			{
				Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
				Acsr->values[count++] = static_cast<int>(DIAG13::m_four);
#else
				Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::m_four, kwave_beta2, j1, k1); // left
#endif
			}

			l2 = l1 - 3;
			take_coord2D(x.n, y.n, l1, j1, k1);
			if (l2 >= 0 && (l2 + 1) % x.n != 0 && (l2 + 2) % x.n != 0 && (l2 + 3) % x.n != 0)
			{
				// не пишем, если номер строки -1, 2 строки кратен Nx
				Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
				Acsr->values[count++] = static_cast<int>(DIAG13::m_three);
#else
				Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::m_three, kwave_beta2, j1, k1); // left
#endif
			}

			l2 = l1 - 2;
			take_coord2D(x.n, y.n, l1, j1, k1);
			if (l2 >= 0 && (l1 - 1) % x.n != 0 && l1 % x.n != 0)
			{
				Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
				Acsr->values[count++] = static_cast<int>(DIAG13::m_two);
#else
				Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::m_two, kwave_beta2, j1, k1); // left
#endif
			}

			l2 = l1 - 1;
			take_coord2D(x.n, y.n, l1, j1, k1);
			if (l2 >= 0 && l1 % x.n != 0)
			{
				Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
				Acsr->values[count++] = static_cast<int>(DIAG13::m_one);
#else
				Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::m_one, kwave_beta2, j1, k1); // left
#endif
			}

			l2 = l1;
			take_coord2D(x.n, y.n, l1, j1, k1);
			if (1)
			{
				Acsr->ja[count] = l2 + 1;
				freqs[count2++] = count;
#ifdef DEBUG
				Acsr->values[count++] = static_cast<int>(DIAG13::zero);
#else
				Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::zero, kwave_beta2, j1, k1);
#endif
			}

			l2 = l1 + 1;
			take_coord2D(x.n, y.n, l1, j1, k1);
			if (l2 < size && l2 % x.n != 0)
			{
				Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
				Acsr->values[count++] = static_cast<int>(DIAG13::one);
#else
				Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::one, kwave_beta2, j1, k1); // right
#endif
			}

			l2 = l1 + 2;
			take_coord2D(x.n, y.n, l1, j1, k1);
			if (l2 < size && (l2 - 1) % x.n != 0 && l2 % x.n != 0)
			{
				// не пишем, если номер строки + 1, 2, 3 строки кратен Nx
				Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
				Acsr->values[count++] = static_cast<int>(DIAG13::two);
#else
				Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::two, kwave_beta2, j1, k1); // right
#endif
			}

			l2 = l1 + 3;
			take_coord2D(x.n, y.n, l1, j1, k1);
			if (l2 < size && (l2 - 2) % x.n != 0 && (l2 - 1) % x.n != 0 && l2 % x.n != 0)
			{
				Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
				Acsr->values[count++] = static_cast<int>(DIAG13::three);
#else
				Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::three, kwave_beta2, j1, k1); // right
#endif
			}

			l2 = l1 + x.n;
			take_coord2D(x.n, y.n, l1, j1, k1);
			if (l2 < size)
			{
				Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
				Acsr->values[count++] = static_cast<int>(DIAG13::four);
#else
				Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::four, kwave_beta2, j1, k1); // right
#endif
			}

			l2 = l1 + 2 * x.n;
			take_coord2D(x.n, y.n, l1, j1, k1);
			if (l2 < size)
			{
				Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
				Acsr->values[count++] = static_cast<int>(DIAG13::five);
#else
				Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::five, kwave_beta2, j1, k1); // right
#endif
			}

			l2 = l1 + 3 * x.n;
			take_coord2D(x.n, y.n, l1, j1, k1);
			if (l2 < size)
			{
				Acsr->ja[count] = l2 + 1;
#ifdef DEBUG
				Acsr->values[count++] = static_cast<int>(DIAG13::six);
#else
				Acsr->values[count++] = beta2D_pml_13pts(x, y, DIAG13::six, kwave_beta2, j1, k1); // right
#endif
			}

#ifdef DEBUG
			//		printf("l1 = %d\n", l1);
			//		print_2Dcsr_mat(x, y, Acsr);

			//		system("pause");
#endif
		}

		if (non_zeros_in_2Dblock13diag != count || size != count2) printf("FAILED generation!!! %d != %d\n", non_zeros_in_2Dblock13diag, count);
		else
		{
			printf("SUCCESSED BASIC 2D generation!\n");
			printf("Non_zeros inside generating PML function: %d\n", count);
		}
	}
	else
	{
#pragma omp parallel for simd schedule(static)
		for (int i = 0; i < size; i++)
			Acsr->values[freqs[i]] += kwave_beta2;

		//	printf("SUCCESSED 2D generation for frequency %d!\n", w);
	}

	print_2Dcsr_mat_to_file(x, y, Acsr, "2Dmat13pts_new.dat");
#undef DEBUG
}
#endif

void GenSparseMatrixOnline2DwithSPONGE(int w, size_m x, size_m y, size_m z, zcsr* Acsr, double kwave2, double beta_eq)
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
				Acsr->values[count++] = beta2D_spg(x, y, 0, kwave2, j1, k1, beta_eq);
#else
				Acsr->values[count++] = beta2D(x, y, 0, j1, k1);
#endif

			}
			else if (l1 == l2 - 1 && (l1 + 1) % x.n != 0)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta2D_spg(x, y, 1, kwave2, j1, k1, beta_eq); // right
			}
			else if (l1 == l2 + 1 && l1 % x.n != 0)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta2D_spg(x, y, -1, kwave2, j1, k1, beta_eq); // left
			}
			else if (l1 == l2 - x.n)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta2D_spg(x, y, 2, kwave2, j1, k1, beta_eq); // right
			}
			else if (l1 == l2 + x.n)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta2D_spg(x, y, -2, kwave2, j1, k1, beta_eq); // left
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


void GenSparseMatrixOnline1DwithPML(int w, size_m x, size_m y, size_m z, zcsr* Acsr, double kwave2, double beta_eq)
{
	int size = x.n;
	int size2 = x.n - 1;
	int non_zeros_in_1D3diag = size + size2 * 2;
	double RelRes = 0;
	double k = double(kk);

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
				Acsr->values[count++] = beta1D(x, 0, kwave2, l1, beta_eq);
			}
			else if (l1 == l2 - 1)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta1D(x, 1, kwave2, l1, beta_eq); // right
			}
			else if (l1 == l2 + 1)
			{
				Acsr->ja[count] = l2 + 1;
				Acsr->values[count++] = beta1D(x, -1, kwave2, l1, beta_eq); // left
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

map<vector<int>, dtype> BlockRowMat_to_CSR(int blk, int n1, int n2, int n3, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, zcsr* Acsr, int& non_zeros_on_prev_level)
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

map<vector<int>, dtype> Block1DRowMat_to_CSR(int blk, int n1, int n2, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, zcsr* Acsr, int& non_zeros_on_prev_level)
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

void GenSparseMatrixOnline2D(char *str, int w, size_m x, size_m y, size_m z, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, zcsr* Acsr)
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

void SolvePardiso3D(size_m x, size_m y, size_m z, zcsr* Dcsr, dtype* x_pard, dtype* f, double thresh)
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
// note - smth wrong with parallelization
//#pragma omp parallel private(i,j)
	{
//#pragma omp for schedule(static)
		for (j = 0; j < n; j++)
#pragma omp simd
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
	
	double arg = double(kk) * r;

	return my_exp(arg) / (4.0 * double(PI) * r);

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

	dtype kk_loc = double(omega) / MakeSound3D(xx, yy, zz, x, y, z, source);

	dtype arg = kk_loc * r;
	
	// i * k * r
	arg = { -arg.imag() , arg.real() };

	// e ^ {i k r} / (4 Pi r)

	return EulerExp(arg) / (4.0 * double(PI) * r);
}

dtype F3D_ex_complex(size_m xx, size_m yy, size_m zz, double x, double y, double z, point source, int &l)
{
#ifdef HELMHOLTZ
	if (x == source.x && y == source.y && z == source.z)
	{
		printf("SOURCE AT x = %lf, y = %lf, z = %lf\n", x, y, z);
		Get3DSrc(xx, yy, zz, source, l);

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

void Get1DSrc(size_m xx, point source, int &l)
{
	double x = source.x;
	l = x / xx.h - 1;
}

void Get2DSrc(size_m xx, size_m yy, point source, int &l)
{
	double x = source.x;
	double y = source.y;
	l = x / xx.h - 1 + xx.n * (y / yy.h - 1);
}

void Get3DSrc(size_m xx, size_m yy, size_m zz, point source, int &l)
{
	double x = source.x;
	double y = source.y;
	double z = source.z;
	l = x / xx.h - 1 + xx.n * (y / yy.h - 1) + xx.n * yy.n * (z / zz.h - 1);
}

dtype F2D_ex_complex(size_m xx, size_m yy, double x, double y, point source, int &l)
{
	if (x == source.x && y == source.y)
	{
#ifdef PRINT_TEST
		printf("SET SOURCE AT x = %lf, y = %lf\n", x, y);
#endif
		Get2DSrc(xx, yy, source, l);
		return -2.0 / (xx.h * yy.h);
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
		Get1DSrc(xx, source, l);
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

void SetRHS3DForTest(size_m xx, size_m yy, size_m zz, dtype* f, point source, int& l)
{
	for(int k = 0; k < zz.n; k++)
		for (int j = 0; j < yy.n; j++)
			for (int i = 0; i < xx.n; i++)
				if (k == (zz.n / 2))
				{
					f[i + xx.n * j + xx.n * yy.n * k] = 1.0 / (xx.h);
				}
				else
				{
					f[i + xx.n * j + xx.n * yy.n * k] = 0;
				}
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

void ResidCSR(size_m x, size_m y, size_m z, zcsr* Dcsr, dtype* x_sol, dtype *f, dtype* g, double &RelRes)
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

#pragma omp parallel for simd schedule(static)
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

void ResidCSR2DHelm(size_m x, size_m y, zcsr* Dcsr, dtype* x_sol, dtype *f, dtype* g, point source, double &RelRes)
{
	int n = x.n;
	int size = n * y.n;
	int size_nopml = (x.n - 2 * x.pml_pts) * (y.n - 2 * y.pml_pts);
	dtype *f1 = alloc_arr<dtype>(size);
	int ione = 1;
	double thresh = 1e-8;

	struct safe_val
	{
		int i;
		dtype val;
	} p1;

	// Multiply matrix A in CSR format by vector x_sol to obtain f1
	mkl_zcsrgemv("No", &size, Dcsr->values, Dcsr->ia, Dcsr->ja, x_sol, f1);

	for (int i = 0; i < size; i++)
	{
		g[i] = f[i] - f1[i];
		//f[i] = f1[i];
		//if (abs(f[i]) != 0) printf("%12.10lf %12.10lf %12.10lf %12.10lf\n", f[i].real(), f[i].imag(), f1[i].real(), f1[i].imag());

#if 1
		if (abs(f[i]) != 0)
		{
			p1.i = i;
			p1.val = f[i];
#ifdef PRINT_TEST
			printf("non_zero value in f[%d] = (%lf, %lf)\n", i, f[i].real(), f[i].imag());
#endif
			f[i] = f1[i] = g[i] = 0;
		}
#endif
	}


#ifdef DEBUG
	print_vec(size, f, g, "f and g");
#endif

	dtype *g_nopml = alloc_arr<dtype>(size_nopml);
	dtype *f_nopml = alloc_arr<dtype>(size_nopml);
	dtype *f1_nopml = alloc_arr<dtype>(size_nopml);
	reducePML2D(x, y, size, g, size_nopml, g_nopml);
	reducePML2D(x, y, size, f, size_nopml, f_nopml);
	reducePML2D(x, y, size, f1, size_nopml, f1_nopml);

	//RelRes = RelError(zlange, size, 1, f1_nopml, f_nopml, size, thresh);
	RelRes = zlange("Frob", &size_nopml, &ione, g_nopml, &size_nopml, NULL);
	//RelRes /= p1.val.real();

	//printf("resid in circle\n");
	//double norm = check_norm_circle2D(x, y, x.pml_pts, x.n - x.pml_pts, f, f1, source, thresh);

	f[p1.i] = p1.val;

	char str[255];
	sprintf(str, "ResidHelmholtz2D_N%d.dat", x.n);
	FILE *out = fopen(str, "w");
	for (int i = 0; i < size_nopml; i++)
		fprintf(out, "%d %23.18lf %23.18lf\n", i, g_nopml[i].real(), g_nopml[i].imag());

	fclose(out);

	printf("End resid\n");

	free_arr(f1);
	free_arr(g_nopml);
	free_arr(f_nopml);
	free_arr(f1_nopml);
}

void ResidCSR2D(size_m x, size_m y, zcsr* Dcsr, dtype* x_sol, dtype *f, dtype* g, double &RelRes)
{
	int size = x.n * y.n;
	int size_nopml = x.n_nopml * y.n_nopml;
	dtype *f1 = alloc_arr<dtype>(size);
	dtype *g_nopml = alloc_arr<dtype>(size_nopml);
	dtype *f_nopml = alloc_arr<dtype>(size_nopml);
	dtype *f1_nopml = alloc_arr<dtype>(size_nopml);
	int ione = 1;

	FILE *out = fopen("ResidTestCSR2D.dat", "w");

	// Multiply matrix A in CSR format by vector x_sol to obtain f1
	mkl_zcsrgemv("No", &size, Dcsr->values, Dcsr->ia, Dcsr->ja, x_sol, f1);

#pragma omp parallel for schedule(static)
	for (int i = 0; i < size; i++)
		g[i] = f[i] - f1[i];


#ifdef DEBUG
	print_vec(size, f, g, "f and g");
#endif

	RelRes = zlange("Frob", &size, &ione, g, &size, NULL);
	RelRes = RelRes / zlange("Frob", &size, &ione, f, &size, NULL);

	printf("RelRes |Au_ex - f| with PML zone = %e\n", RelRes);

	reducePML2D(x, y, size, g, size_nopml, g_nopml);
	reducePML2D(x, y, size, f, size_nopml, f_nopml);
	reducePML2D(x, y, size, f1, size_nopml, f1_nopml);

#if 0
	for (int i = 0; i < size_nopml; i++)
		fprintf(out, "%d %lf %lf\n", i, f_nopml[i].real(), f1_nopml[i].real());
#else
	for (int i = 0; i < size; i++)
		fprintf(out, "%d %lf %lf\n", i, f[i].real(), f1[i].real());
#endif

#if 0
	RelRes = zlange("Frob", &size, &ione, g, &size, NULL);
	RelRes = RelRes / zlange("Frob", &size, &ione, f, &size, NULL);
#else
	RelRes = zlange("Frob", &size_nopml, &ione, g_nopml, &size_nopml, NULL);
	RelRes = RelRes / zlange("Frob", &size_nopml, &ione, f_nopml, &size_nopml, NULL);
#endif
	fclose(out);

#if 0
#pragma omp parallel for schedule(static)
	for (int i = 0; i < size; i++)
		f[i] = f1[i];
#endif

	free_arr(f1);
	free_arr(g_nopml);
	free_arr(f_nopml);
	free_arr(f1_nopml);
}



void GenRHSandSolution2D_Syntetic(size_m x, size_m y, zcsr *Dcsr, dtype *u, dtype *f)
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
		//real = random(0.0, 1.0);
		//imag = random(0.0, 1.0);
		//vector[i] = { real, imag };

		vector[i] = random(0.0, 1.0);
		//	printf("%lf\n", vector[i].real());
	}
}

double random(double min, double max)
{
	return (double)(rand()) / RAND_MAX * (max - min) + min;
}


void output(char *str, bool pml_flag, size_m x, size_m y, size_m z, dtype* x_orig, dtype* x_pard, double diff_sol)
{
	char name[255], name2[255];
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

	double rel_real, rel_imag;
//#define BINARY
#ifndef BINARY
	for (int k = 0; k < Nz; k++)
	{
		sprintf(name, "%s_%02d.dat", str, k + 1);
		FILE *file = fopen(name, "w");
		for (int j = 0; j < Ny; j++)
			for (int i = 0; i < Nx; i++)
			{
				rel_real = (x_pard[i + j * Nx + k * Nx * Ny].real() - x_orig[i + j * Nx + k * Nx * Ny].real()) / x_orig[i + j * Nx + k * Nx * Ny].real();
				rel_imag = (x_pard[i + j * Nx + k * Nx * Ny].imag() - x_orig[i + j * Nx + k * Nx * Ny].imag()) / x_orig[i + j * Nx + k * Nx * Ny].imag();

//				if (fabs(rel_real) > 2)
				{
					fprintf(file, "%lf %12.10lf %12.10lf %12.10lf %12.10lf %12.10lf %12.10lf %12.10lf %12.10lf\n", i * x.h, j * y.h, k * z.h,
						x_orig[i + j * Nx + k * Nx * Ny].real(), x_orig[i + j * Nx + k * Nx * Ny].imag(),
						x_pard[i + j * Nx + k * Nx * Ny].real(), x_pard[i + j * Nx + k * Nx * Ny].imag(),
						fabs(rel_real), fabs(rel_imag));
				}
			}
		fclose(file);
	}

//#define MATLAB 
#ifdef MATLAB
	double lambda = double(c_z) / nu;
	double ppw = lambda / x.h;
	sprintf(name, "%s_3D.dat", str);
	FILE *file2 = fopen(name, "w");
	fprintf(file2, "Nx %d, Nz %d, Nz %d\n", Nx, Ny, Nz);
	fprintf(file2, "hx %.2lf, hy %.2lf, hz %.2lf\n", x.h, y.h, z.h);
	fprintf(file2, "Lx %.2lf, Ly %.2lf, Lz %.2lf\n", x.l, y.l, z.l);
	fprintf(file2, "frequency nu %d\n", nu);
#ifdef HOMO
	fprintf(file2, "ppw %.2lf\n", ppw);
#else
	fprintf(file2, "sound speed %.1lf * x + %.1lf * y + %.1lf * z\n", C1, C2, C3);
#endif
	fprintf(file2, "x_pml %d, y_pml %d, z_sponge %d\n", x.pml_pts, y.pml_pts, z.spg_pts);
	fprintf(file2, "niter %d\n", NITER * 3);
#ifdef HOMO
	fprintf(file2, "||u_num-u_ex|| / ||u_ex|| %11.4e\n", diff_sol);
#else
	fprintf(file2, "||u_{k+1}-u_{k}|| / ||u_k|| %11.4e\n", diff_sol);
#endif
	fprintf(file2, "i j k Re(u) Im(u)\n");
	for (int k = 0; k < Nz; k++)
		for (int j = 0; j < Ny; j++)
			for (int i = 0; i < Nx; i++)
				fprintf(file2, "%d %d %d %11.4e %11.4e\n", i, j, k, x_pard[i + j * Nx + k * Nx * Ny].real(), x_pard[i + j * Nx + k * Nx * Ny].imag());
				
	fclose(file2);
#endif
#else

#if 0

#if 1
	for (int k = 0; k < 1; k++)
	{
		sprintf(name, "%s_%02d.bin", str, k);
		FILE *file = fopen(name, "w");
		for (int j = 0; j < 1; j++)
			for (int i = 0; i < Nx; i++)
			{
				//struct package pack = {i * x.h, j * y.h, k * z.h, x_pard[i + j * Nx + k * Nx * Ny].real(), x_pard[i + j * Nx + k * Nx * Ny].imag()} ;
				struct package2 pack = { i * x.h, j * y.h, x_pard[i + j * Nx + k * Nx * Ny] };
				fwrite(&pack, sizeof(struct package2), 1, file);
				if (fwrite == 0) printf("error writing file !\n");
				//fprintf(file, "%12.10lf %12.10lf %12.10lf %12.10lf\n", pack.x, pack.y, pack.ureal, pack.uimag);
			}
		fclose(file);
	}
#if 1
	for (int k = 0; k < 1; k++)
	{
		sprintf(name, "%s_%02d.bin", str, k);
		sprintf(name2, "%s_%02d.dat", str, k);
		FILE *file2 = fopen(name, "r");
		FILE *file3 = fopen(name2, "w");

		if (file2 == NULL)
		{
			fprintf(stderr, "\nError opend file\n");
			//exit(1);
		}
			
				struct package2 pack1;
				printf("begin read\n");

				while (fread(&pack1, sizeof(struct package2), 1, file2))
				{
					//printf("new string!\n");
					//fprintf(file2, "%12.10lf %12.10lf %12.10lf %12.10lf %12.10lf\n", pack.x, pack.y, pack.z, pack.ureal, pack.uimag);
					printf("%12.10lf %12.10lf %12.10lf %12.10lf\n", pack1.x, pack1.y, pack1.u.real(), pack1.u.imag());
				}
				printf("end read\n");
			
		fclose(file2);
		fclose(file3);
	}
#endif
#endif
#endif

#if 0
	FILE *outfile;

	// open file for writing 
	outfile = fopen("person.dat", "w");
	if (outfile == NULL)
	{
		fprintf(stderr, "\nError opend file\n");
		exit(1);
	}

	struct person input1 = { 1.0, 3.0, "rohan", "sharma" };
	struct person input2 = { 2.0, 6.0, "mahendra", "dhoni" };

	// write struct to file 
	fwrite(&input1, sizeof(struct person), 1, outfile);
	fwrite(&input2, sizeof(struct person), 1, outfile);

	if (fwrite != 0)
		printf("contents to file written successfully !\n");
	else
		printf("error writing file !\n");

	// close file 
	fclose(outfile);

	FILE *infile;
	struct person input;

	// Open person.dat for reading 
	infile = fopen("person.dat", "r");
	if (infile == NULL)
	{
		fprintf(stderr, "\nError opening file\n");
		exit(1);
	}

	// read file contents till end of file 
	while (fread(&input, sizeof(struct person), 1, infile))
		printf("id = %lf x = %lf name = %s %s\n", input.id, input.x,
			input.fname, input.lname);

	// close file 
	fclose(infile);
#endif

	for (int k = 0; k < Nz; k++)
	{
		sprintf(name, "%s_%02d.bin", str, k);
		FILE *file = fopen(name, "w");
		for (int j = 0; j < Ny; j++)
			for (int i = 0; i < Nx; i++)
			{
				struct package3 pack = { i * x.h, j * y.h, x_pard[i + j * Nx + k * Nx * Ny].real() };
				fwrite(&pack, sizeof(struct package3), 1, file);
				if (fwrite == 0) printf("error writing file !\n");
			}
		fclose(file);
	}

#if 0
	for (int k = 0; k < Nz; k++)
	{
		sprintf(name, "%s_%02d.bin", str, k);
		sprintf(name2, "%s_%02d.dat", str, k);
		FILE *file = fopen(name, "r");
		FILE *file2 = fopen(name2, "w");

		if (file == NULL)
		{
			fprintf(stderr, "\nError opend file\n");
			//exit(1);
		}

		struct package3 pack;

		while (fread(&pack, sizeof(struct package3), 1, file))
		{
			fprintf(file2, "%12.10lf %12.10lf %12.10lf\n", pack.x, pack.y, pack.sol);
		}

		fclose(file);
		fclose(file2);

	}
#endif
#endif

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

void output2D_float(char *str, bool pml_flag, size_m x, size_m y, stype* x_orig, stype* x_pard)
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
			fprintf(file, "%lf %lf %f %f %f %f\n", i * x.h, j * y.h,
				x_orig[i + j * Nx].real(), x_orig[i + j * Nx].imag(),
				x_pard[i + j * Nx].real(), x_pard[i + j * Nx].imag());
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

void output1D(char *str, bool pml_flag, size_m x, double* x_re, double* x_im)
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
		fprintf(file, "%lf %12.10lf %12.10lf\n", i * x.h, x_re[i], x_im[i]);
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
	//fprintf(file1, "set zrange[-0.03:0.03]\n");
	fprintf(file1, "set pm3d\n");
	fprintf(file1, "set palette\n");

	for (int k = 0; k < Nz; k++)
	{
		//fprintf(file, "set cbrange[%6.4lf:%6.4lf]\n", x_orig[x.n - 1 + (y.n - 1) * x.n + k * y.n * x.n].real(), x_orig[0 + 0 * x.n + k * y.n * x.n].real());
	//	if (col == 4) fprintf(file, "set cbrange[%12.10lf:%12.10lf]\n", x_orig[0 + 0 * Nx + k * Ny * Nx].real(), x_orig[(Nx - 1 + (Ny - 1) * Nx) / 2 + k * Ny * Nx].real());
	//	else fprintf(file, "set cbrange[%12.10lf:%12.10lf]\n", x_sol[0 + 0 * Nx + k * Ny * Nx].real(), x_sol[(Nx - 1 + (Ny - 1) * Nx) / 2 + k * Ny * Nx].real());
		//printf("k = %d\nleft: %d  %lf \nright: %d %lf \n\n", k, 0 + 0 * x.n, x_orig[0 + 0 * Nx + k * Ny * Nx].real(), (Nx - 1 + (Ny - 1) * Nx) / 2, x_orig[(Nx - 1 + (Ny - 1) * Nx) / 2 + k * Ny * Nx].real());
		fprintf(file1, "set output '%s_z_%4.2lf.png'\n", sout, (k + 1) * z.h);
		fprintf(file1, "splot '%s_%02d.dat' u 2:1:%d w linespoints pt 7 palette pointsize 1 notitle\n\n", splot, k + 1, col);
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

	//x.l = LENGTH_X;
	//y.l = LENGTH_Y;

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
	char *str = alloc_arr<char>(255);
	int Nx;
	double L;
	if (pml_flag == true)
	{
		Nx = x.n - 2 * x.pml_pts;
	}
	else
	{
		Nx = x.n;
	}

	L = Nx * x.h;
	FILE* file1;
	//sprintf(str, "run.plt", numb++);

	if (col != 0)
	{
		if (col == 3) str = "run_ex.plt";
		else str = "run_pard.plt";

		file1 = fopen(str, "w");

		//fprintf(file1, "reset\nclear\n");
		fprintf(file1, "set term png font \"Times-Roman, 16\"\n");
		//fprintf(file, "set view map\n");
		fprintf(file1, "set xrange[0:%lf]\n", L);
		fprintf(file1, "set yrange[-50:50]\n");

		fprintf(file1, "set output '%s_re.png'\n", sout);
		fprintf(file1, "plot '%s.dat' u 1:%d w linespoints pt 7 pointsize 1 notitle\n\n", splot, col);

		fprintf(file1, "set output '%s_im.png'\n", sout);
		fprintf(file1, "plot '%s.dat' u 1:%d w linespoints pt 7 pointsize 1 notitle\n\n", splot, col + 1);

		fprintf(file1, "exit\n");
		fclose(file1);
	}
	else
	{
		str = "run_diff.plt";
		file1 = fopen(str, "w");

		fprintf(file1, "set term png font \"Times-Roman, 16\"\n");
		//fprintf(file, "set view map\n");
		fprintf(file1, "set xrange[0:%lf]\n", L);

		fprintf(file1, "set output '%s_re.png'\n", sout);
		fprintf(file1, "plot '%s.dat' u 1:2 w linespoints pt 7 pointsize 1 title \"exact\", '%s.dat' u 1:4 w linespoints pt 7 pointsize 1 title \"num\" \n\n", splot, splot);

		fprintf(file1, "set output '%s_im.png'\n", sout);
		fprintf(file1, "plot '%s.dat' u 1:3 w linespoints pt 7 pointsize 1 title \"exact\", '%s.dat' u 1:5 w linespoints pt 7 pointsize 1 title \"num\" \n\n", splot, splot);


		fprintf(file1, "exit\n");
		fclose(file1);
	}

	system(str);
}

void gnuplot1D_simple(char *splot, char *sout, bool pml_flag, int col, size_m x)
{
	char *str = alloc_arr<char>(255);
	int Nx;
	double L;
	if (pml_flag == true)
	{
		Nx = x.n - 2 * x.pml_pts;
	}
	else
	{
		Nx = x.n;
	}

	L = Nx * x.h;
	FILE* file1;
	//sprintf(str, "run.plt", numb++);

	str = "run_ex.plt";

	file1 = fopen(str, "w");

	//fprintf(file1, "reset\nclear\n");
	fprintf(file1, "set term png font \"Times-Roman, 16\"\n");
	//fprintf(file, "set view map\n");
	fprintf(file1, "set xrange[0:%lf]\n", L);
	//fprintf(file1, "set yrange[-50:50]\n");

	fprintf(file1, "set output '%s_re.png'\n", sout);
	fprintf(file1, "plot '%s.dat' u 1:%d w linespoints pt 7 pointsize 1 notitle\n\n", splot, 2);

	fprintf(file1, "set output '%s_im.png'\n", sout);
	fprintf(file1, "plot '%s.dat' u 1:%d w linespoints pt 7 pointsize 1 notitle\n\n", splot, 3);

	fprintf(file1, "exit\n");
	fclose(file1);

	system(str);
}

void ApplyCoeffMatrixA_CSR(size_m x, size_m y, size_m z, int *iparm, int *perm, size_t *pt, zcsr** &D2csr, const dtype *w, const dtype* deltaL, dtype* g, double beta_eq, double thresh)
{
	// Function for applying (I - deltaL * L_0 ^{-1}) * w = g
	int size = x.n * y.n * z.n;

#if 0
	printf("check right-hand-side f\n");
	for (int i = 0; i < size; i++)
		if (abs(w[i]) != 0) printf("f_FFT[%d] = %lf %lf\n", i, w[i].real(), w[i].imag());

	system("pause");
#endif

	// Solve the preconditioned system: L_0 ^ {-1} * w = g
	Solve3DSparseUsingFT_CSR(x, y, z, iparm, perm, pt, D2csr, w, g, beta_eq, thresh);

	// Multiply point-to-point deltaL * g
	OpTwoMatrices(size, 1, g, deltaL, g, size, '*');

	// g:= w - g
	OpTwoMatrices(size, 1, w, g, g, size, '-');

}

#ifdef HODLR
void ApplyCoeffMatrixA_HODLR(size_m x, size_m y, size_m z, cmnode* **Gstr, dtype *B, bool *solves, const dtype *w, const dtype* deltaL, dtype* g, double thresh, int smallsize)
{
	// Function for applying (I - deltaL * L_0 ^{-1}) * w = g
	int size = x.n * y.n * z.n;

#if 0
	printf("check right-hand-side f\n");
	for (int i = 0; i < size; i++)
		if (abs(w[i]) != 0) printf("f_FFT[%d] = %lf %lf\n", i, w[i].real(), w[i].imag());

	system("pause");
#endif

	// Solve the preconditioned system: L_0 ^ {-1} * w = g
	Solve3DSparseUsingFT_HODLR(x, y, z, Gstr, B, solves, w, g, thresh, smallsize);

	// Multiply point-to-point deltaL * g
	OpTwoMatrices(size, 1, g, deltaL, g, size, '*');

	// g:= w - g
	OpTwoMatrices(size, 1, w, g, g, size, '-');

}
#endif

void print_2Dcsr_mat(size_m x, size_m y, zcsr* D2csr)
{
	int count1 = 0;
	int count2 = 0;
	int size = x.n * y.n;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
			if (j == D2csr->ja[count1] - 1)
			{
				count1++;
				printf("%2.0lf ", D2csr->values[count2++].real());
			}
			else
			{
				printf("%2.0lf ", 0);
			}
		printf("\n");
	}
	
	printf("count1 = %d count2 = %d valuesN = %d\n", count1, count2, D2csr->non_zeros);
}


void print_2Dcsr_mat_to_file(size_m x, size_m y, zcsr* D2csr, char *s)
{
	int count1 = 0;
	int count2 = 0;
	int size = x.n * y.n;

	FILE* fout;
	fout = fopen(s, "w");
#if 0
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
			if (j == D2csr->ja[count1] - 1)
			{
				count1++;
				fprintf(fout, "%6.3lf ", D2csr->values[count2++].real());
			}
			else
			{
				//fprintf(fout, "%6.3lf ", 0);
			}
		fprintf(fout, "          elems: %d\n", count2);
	}

	fprintf(fout, "count1 = %d count2 = %d valuesN = %d\n", count1, count2, D2csr->non_zeros);

#else
	int vals_count = 0;

	for (int i = 0; i < size; i++)
	{
		int vals_in_row = D2csr->ia[i + 1] - D2csr->ia[i];
		for (int j = 0; j < vals_in_row; j++)
		{
			fprintf(fout, "%8.5lf ", D2csr->values[vals_count + j].real());
		}
		vals_count += vals_in_row;
		fprintf(fout, "\n");
	}
	fprintf(fout, "n1 = %d, count = %d, valuesN = %d\n", x.n, vals_count, D2csr->non_zeros);
#endif

	fclose(fout);
}

void print_2Dcsr_mat2(size_m x, size_m y, zcsr* D2csr)
{
	int count1 = 0;
	int count2 = 0;
	int size = x.n * y.n;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
			if (j == D2csr->ja[count1] - 1)
			{
				count1++;
				printf("%6.4lf ", D2csr->values[count2++].real());
			}
		printf("\n");
	}

	printf("count1 = %d count2 = %d valuesN = %d\n", count1, count2, D2csr->non_zeros);
}

void AdjustRHS(size_m x, size_m y, dtype *f)
{
	int j, k;
	int size = x.n * y.n;

	for (int l1 = 0; l1 < size; l1++)
	{
		take_coord2D(x.n, y.n, l1, j, k);
		dtype alp = alpha(x, j) * alpha(y, k);
		f[l1] /= alp;
	}
}

void SetFrequency(char *str, size_m x, size_m y, size_m z, int i, dtype &kwave_beta2, double beta_eq)
{
	int nhalf = y.n / 2;
	double kww;
	double k = (double)kk;
	int n = 7;

	if (!strcmp(str, "MKL_FFT"))
	{
		if (i < z.n / 2 - 1)
		{
			kww = 4.0 * double(PI) * double(PI) * i * i / (z.l * z.l);
		}
		else
		{
			kww = 4.0 * double(PI) * double(PI) * (z.n - i) * (z.n - i) / (z.l * z.l);
		}
	}
	else
	{
		kww = 4.0 * double(PI)* double(PI) * (i - nhalf) * (i - nhalf) / (y.l * y.l);
	}

	kwave_beta2 = dtype{ k * k - kww, k * k * beta_eq };

	double lambda = (double)(c_z) / nu;
	double ppw = lambda / x.h;

#ifdef PRINT_TEST
	printf("ppw = %lf\n", ppw);
#endif
}

void Solve3DSparseUsingFT_CSR(size_m x, size_m y, size_m z, int *iparm, int *perm, size_t *pt, zcsr** &D2csr, const dtype *f, dtype* x_sol, double beta_eq, double thresh)
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
	dtype *f_FFT = alloc_arr<dtype>(size);
	dtype* u1D = alloc_arr<dtype>(z.n);
	dtype* u1D_BFFT = alloc_arr<dtype>(z.n);
	
	// f(x,y,z) -> fy(x,z) 
	DFTI_DESCRIPTOR_HANDLE my_desc_handle;

	MKL_LONG strides_in[2] = { 0, size2D };
	MKL_LONG strides_out[2] = { 0, size2D };

	// Create 1D FFT of COMPLEX DOUBLE case
	status = DftiCreateDescriptor(&my_desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, z.n);
	status = DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	status = DftiSetValue(my_desc_handle, DFTI_BACKWARD_SCALE, 1.0 / z.n);
	status = DftiSetValue(my_desc_handle, DFTI_INPUT_STRIDES, strides_in);
	status = DftiSetValue(my_desc_handle, DFTI_OUTPUT_STRIDES, strides_out);
	status = DftiCommitDescriptor(my_desc_handle);

	// We make n1 * n2 FFT's for one dimensional direction z with n3 grid points

#ifdef PRINT
	printf("Applying 1D Fourier transformation for 3D RHS\n");
#endif
	for (int w = 0; w < size2D; w++)
	{
		status = DftiComputeForward(my_desc_handle, (void*)&f[w], &f_FFT[w]);
		
		//status = DftiComputeForward(my_desc_handle, f_FFT_in, &f_FFT_out[z.n * w]);
	}

//#define PRINT

#if 0
	printf("check right-hand-side f\n");
	for (int i = 0; i < size; i++)
		if (abs(f_FFT[i]) != 0) printf("f_FFT[%d] = %lf %lf\n", i, f_FFT[i].real(), f_FFT[i].imag());
	system("pause");
#endif
	
//#undef PRINT

	// Calling the solver
	int mtype = 13;
	int maxfct = 1;
	int mnum = 1;
	int phase = 33;
	int rhs = 1;
	int msglvl = 0;
	int error = 0;

	int n_nopml = x.n_nopml * y.n_nopml;
	int size_nopml = n_nopml * z.n_nopml;
	int size2D_nopml = x.n_nopml * y.n_nopml;

#ifdef PRINT_DEBUG
	printf("Non-zeros in 2D block-diagonal: %d\n", non_zeros_in_2Dblock3diag);
	printf("----------Generating 2D matrix and rhs + solving by pardiso-------\n");
	printf("Size of system: %d x %d with PML %d on each direction\n", x.n, y.n, 2 * x.pml_pts);
#endif

	char *str1, *str2, *str3;
	str1 = alloc_arr<char>(255);
	str2 = alloc_arr<char>(255);
	str3 = alloc_arr<char>(255);
	bool pml_flag = false;
	double time;

#ifdef PRINT
	printf("Solving set of 2D problems...\n");
#endif
	int count = 0;
	point sourcePML = { x.l / 2, y.l / 2 };
	int src = 0;

	// Get source point coordinates
	Get2DSrc(x, y, sourcePML, src);


	time = omp_get_wtime();

	for (int k = 0; k < z.n; k++)
	{
		if (D2csr[k]->solve == 1)
		{
			count++;
			int non_zeros_rhs = 0;
#ifdef CHECK_ACCURACY
			dtype alpha_k = f_FFT[k * size2D + src] / (1.0 / (x.h * y.h));
			for (int i = 0; i < size2D; i++)
			{
				if (abs(f_FFT[k * size2D + i]) > 0) non_zeros_rhs++;
				//if (abs(f_FFT[k * size2D + i]) > 0) printf("src = %d, f_FFT[%d] = (%lf, %lf)\n", src, i, f_FFT[k * size2D + i].real(), f_FFT[k * size2D + i].imag());
			}
#endif
#ifdef SYMMETRY
			AdjustRHS(x, y, &f_FFT[k * size2D]);
#endif
			pardiso(&pt[k * 64], &maxfct, &mnum, &mtype, &phase, &size2D, D2csr[k]->values, D2csr[k]->ia, D2csr[k]->ja, &perm[k * size2D], &rhs, &iparm[k * 64], &msglvl, &f_FFT[k * size2D], &x_sol_prd[k * size2D], &error);

			if (error != 0) printf("!!!Error: PARDISO %d!!!\n", error);

#ifdef PRINT_DEBUG
			double eps = 0.01; // 1 percent
			if (norm < eps) printf("Resid 2D Hankel norm %12.10e < eps %12.10lf: PASSED\n\n", norm, eps);
			else printf("Resid 2D Hankel norm %12.10lf > eps %12.10lf : FAILED\n\n", norm, eps);
			//sprintf(str1, "ChartsPML/model_pml_%lf", kwave2);
			//sprintf(str2, "ChartsPML/model_pml_ex_%lf", kwave2);
			//sprintf(str3, "ChartsPML/model_pml_pard_%lf", kwave2);

			//sprintf(str1, "ChartsSPONGE/model_pml_%lf", kwave2);
			//sprintf(str2, "ChartsSPONGE/model_pml_ex_%lf", kwave2);
			//sprintf(str3, "ChartsSPONGE/model_pml_pard_%lf", kwave2);
#endif

#ifdef CHECK_ACCURACY
		// normalization of rhs

			dtype kwave2;
			SetFrequency("MKL_FFT", x, y, z, k, kwave2, beta_eq);

			dtype *x_sol_ex = alloc_arr<dtype>(size2D);
			dtype *x_num_prd = alloc_arr<dtype>(size2D);
			dtype *x_sol_ex_nopml = alloc_arr<dtype>(size2D_nopml);
			dtype *x_sol_prd_nopml = alloc_arr<dtype>(size2D_nopml);

			int ione = 1;
			zcopy(&size2D, &x_sol_prd[k * size2D], &ione, x_num_prd, &ione);

		//	printf("kwave2 = (%lf, %lf)\n", kwave2.real(), kwave2.imag());
		//	printf("alpha = (%lf, %lf)\n", alpha_k.real(), alpha_k.imag());
			
			// Get exact solution
			get_exact_2D_Hankel(x.n, y.n, x, y, x_sol_ex, sqrt(kwave2), sourcePML);
			normalization_of_exact_sol(x.n, y.n, x, y, x_sol_ex, alpha_k);

			// Nullify source point
			NullifySource2D(x, y, x_sol_ex, src, 1);
			NullifySource2D(x, y, x_num_prd, src, 1);

			//norm = resid_2D_Hankel(x, y, D2csr[k], x_sol_ex, &f_FFT[k * size2D], sourcePML);
			//printf("resid ||A * x_sol - kf|| 2D Hankel: %lf\n", norm);

			//output2D(str1, pml_flag, x, y, x_sol_ex, &x_sol_prd[k * size2D]);
			//gnuplot2D(str1, str2, pml_flag, 3, x, y);
			//gnuplot2D(str1, str3, pml_flag, 5, x, y);

			// check norm solution
			reducePML2D(x, y, size2D, x_sol_ex, size2D_nopml, x_sol_ex_nopml);
			reducePML2D(x, y, size2D, x_num_prd, size2D_nopml, x_sol_prd_nopml);

			printf("non_zeros_rhs = %d of %d\n", non_zeros_rhs, size2D);
			norm = RelError(zlange, size2D_nopml, 1, x_sol_prd_nopml, x_sol_ex_nopml, size2D_nopml, thresh);
			printf("Norm 2D solution for k = %d is ||x_sol - x_ex|| / ||x_ex|| = %lf\n", k, norm);
		//	system("pause");

			free(x_sol_ex);
			free(x_num_prd);
			free(x_sol_ex_nopml);
			free(x_sol_prd_nopml);
#endif
		}
	}

#ifdef PRINT
	printf("Solved: %d of %d\nMissed: %d of %d\n", count, z.n, z.n - count, z.n);
#endif

	time = omp_get_wtime() - time;

#ifdef PRINT
	printf("time elapsed for 2D problems: %lf sec\n", time);
	printf("Backward 1D FFT's of %d x %d times to each point of 2D solution\n", x.n_nopml, y.n_nopml);
#endif
	for (int w = 0; w < size2D; w++)
	{
		status = DftiComputeBackward(my_desc_handle, &x_sol_prd[w], &x_sol[w]);

		// status = DftiComputeBackward(my_desc_handle, u1D, u1D_BFFT);
	}

	status = DftiFreeDescriptor(&my_desc_handle);
#ifdef PRINT
	printf("------------- The end of algorithm ----------------------\n");
#endif

	free(str1);
	free(str2);
	free(str3);

	free_arr(u1D);
	free_arr(u1D_BFFT);
	free_arr(x_sol_prd);
	free_arr(f_FFT);
}

#ifdef HODLR
void Solve3DSparseUsingFT_HODLR(size_m x, size_m y, size_m z, cmnode* **Gstr, dtype* B, bool *solves, const dtype *f, dtype* x_sol, double thresh, int smallsize)
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
	int lwork = size2D + x.n;

	dtype *x_sol_hss = alloc_arr2<dtype>(size);
	dtype *f_FFT = alloc_arr2<dtype>(size);
	dtype *work = alloc_arr2<dtype>(lwork);

	// f(x,y,z) -> fy(x,z) 
	DFTI_DESCRIPTOR_HANDLE my_desc_handle;

	MKL_LONG strides_in[2] = { 0, size2D };
	MKL_LONG strides_out[2] = { 0, size2D };

	// Create 1D FFT of COMPLEX DOUBLE case
	status = DftiCreateDescriptor(&my_desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, z.n);
	status = DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	status = DftiSetValue(my_desc_handle, DFTI_BACKWARD_SCALE, 1.0 / z.n);
	status = DftiSetValue(my_desc_handle, DFTI_INPUT_STRIDES, strides_in);
	status = DftiSetValue(my_desc_handle, DFTI_OUTPUT_STRIDES, strides_out);
	status = DftiCommitDescriptor(my_desc_handle);

	// We make n1 * n2 FFT's for one dimensional direction z with n3 grid points

#ifdef PRINT
	printf("Applying 1D Fourier transformation for 3D RHS\n");
#endif
	for (int w = 0; w < size2D; w++)
	{
		status = DftiComputeForward(my_desc_handle, (void*)&f[w], &f_FFT[w]);
	}

	//#define PRINT

#ifdef PRINT
	printf("check right-hand-side f\n");
	for (int i = 0; i < size; i++)
		if (abs(f_FFT[i]) != 0) printf("f_FFT[%d] = %lf %lf\n", i, f_FFT[i].real(), f_FFT[i].imag());
	system("pause");
#endif

#undef PRINT

	int n_nopml = x.n_nopml * y.n_nopml;
	int size_nopml = n_nopml * z.n_nopml;
	int size2D_nopml = x.n_nopml * y.n_nopml;

#ifdef PRINT
	printf("Non-zeros in 2D block-diagonal: %d\n", non_zeros_in_2Dblock3diag);
	printf("----------Generating 2D matrix and rhs + solving by pardiso-------\n");
	printf("Size of system: %d x %d with PML %d on each direction\n", x.n, y.n, 2 * x.pml_pts);
#endif

	char *str1, *str2, *str3;
	str1 = alloc_arr<char>(255);
	str2 = alloc_arr<char>(255);
	str3 = alloc_arr<char>(255);
	bool pml_flag = false;
	double time;

	printf("Solving set of 2D problems with HODLR...\n");
	int count = 0;
	point sourcePML = { x.l / 2, y.l / 2 };
	int src = 0;

	// Get source point coordinates
	Get2DSrc(x, y, sourcePML, src);

	time = omp_get_wtime();

	for (int k = 0; k < z.n; k++)
	{

		if (solves[k])
		{
			count++;
#ifdef CHECK_ACCURACY
			dtype alpha_k = f_FFT[k * size2D + src] / (1.0 / (x.h * y.h));
			for (int i = 0; i < size2D; i++)
			{
				if (abs(f_FFT[k * size2D + i]) > 0) printf("src = %d, f_FFT[%d] = (%lf, %lf)\n", src, i, f_FFT[k * size2D + i].real(), f_FFT[k * size2D + i].imag());
			}
#endif
#ifdef SYMMETRY
			AdjustRHS(x, y, &f_FFT[k * size2D]);
#endif
			DirSolveFastDiagStruct(x.n, y.n, Gstr[k], &B[k * (size2D - x.n)], &f_FFT[k * size2D], &x_sol_hss[k * size2D], work, lwork, thresh, smallsize);

#ifdef PRINT
			double eps = 0.01; // 1 percent
			if (norm < eps) printf("Resid 2D Hankel norm %12.10e < eps %12.10lf: PASSED\n\n", norm, eps);
			else printf("Resid 2D Hankel norm %12.10lf > eps %12.10lf : FAILED\n\n", norm, eps);
			//sprintf(str1, "ChartsPML/model_pml_%lf", kwave2);
			//sprintf(str2, "ChartsPML/model_pml_ex_%lf", kwave2);
			//sprintf(str3, "ChartsPML/model_pml_pard_%lf", kwave2);

			//sprintf(str1, "ChartsSPONGE/model_pml_%lf", kwave2);
			//sprintf(str2, "ChartsSPONGE/model_pml_ex_%lf", kwave2);
			//sprintf(str3, "ChartsSPONGE/model_pml_pard_%lf", kwave2);
#endif

#ifdef CHECK_ACCURACY
		// normalization of rhs

			dtype kwave2;
			SetFrequency("MKL_FFT", x, y, z, k, kwave2, beta_eq);

			dtype *x_sol_ex = alloc_arr<dtype>(size2D);
			dtype *x_sol_ex_nopml = alloc_arr<dtype>(size2D_nopml);
			dtype *x_sol_prd_nopml = alloc_arr<dtype>(size2D_nopml);

			printf("kwave2 = (%lf, %lf)\n", kwave2.real(), kwave2.imag());
			printf("alpha = (%lf, %lf)\n", alpha_k.real(), alpha_k.imag());

			// Get exact solution
			get_exact_2D_Hankel(x.n, y.n, x, y, x_sol_ex, sqrt(kwave2), sourcePML);
			normalization_of_exact_sol(x.n, y.n, x, y, x_sol_ex, alpha_k);

			// Nullify source point
			NullifySource2D(x, y, x_sol_ex, src, 1);
			NullifySource2D(x, y, &x_sol_hss[k * size2D], src, 1);

			//norm = resid_2D_Hankel(x, y, D2csr[k], x_sol_ex, &f_FFT[k * size2D], sourcePML);
			//printf("resid ||A * x_sol - kf|| 2D Hankel: %lf\n", norm);

			//output2D(str1, pml_flag, x, y, x_sol_ex, &x_sol_prd[k * size2D]);
			//gnuplot2D(str1, str2, pml_flag, 3, x, y);
			//gnuplot2D(str1, str3, pml_flag, 5, x, y);

			// check norm solution
			reducePML2D(x, y, size2D, x_sol_ex, size2D_nopml, x_sol_ex_nopml);
			reducePML2D(x, y, size2D, &x_sol_hss[k * size2D], size2D_nopml, x_sol_prd_nopml);

			norm = RelError(zlange, size2D_nopml, 1, x_sol_prd_nopml, x_sol_ex_nopml, size2D_nopml, thresh);
			printf("Norm 2D solution ||x_sol - x_ex|| / ||x_ex|| = %lf\n", norm);
			system("pause");

			free(x_sol_ex);
			free(x_sol_ex_nopml);
			free(x_sol_prd_nopml);
#endif
		}
	}

	printf("Solved: %d of %d\nMissed: %d of %d\n", count, z.n, z.n - count, z.n);

	time = omp_get_wtime() - time;

	printf("time elapsed for 2D problems: %lf sec\n", time);

#ifdef PRINT
	printf("Backward 1D FFT's of %d x %d times to each point of 2D solution\n", x.n_nopml, y.n_nopml);
#endif
	for (int w = 0; w < size2D; w++)
	{
		status = DftiComputeBackward(my_desc_handle, &x_sol_hss[w], &x_sol[w]);
	}

	status = DftiFreeDescriptor(&my_desc_handle);
	printf("------------- The end of algorithm ----------------------\n");

	free(str1);
	free(str2);
	free(str3);

	free_arr(x_sol_hss);
	free_arr(f_FFT);
	free_arr(work);
}
#endif

void Multiply3DSparseUsingFT(size_m x, size_m y, size_m z, int *iparm, int *perm, size_t *pt, zcsr** &D2csr, const dtype *u, dtype* f_sol, double thresh)
{
	int size = x.n * y.n * z.n;
	int size2D = x.n * y.n;
	MKL_LONG status;
	int error = 0;
	double norm = 0;
	int count = 0;
	double time = 0;

	DFTI_DESCRIPTOR_HANDLE my_desc_handle;

	MKL_LONG strides_in[2] = { 0, size2D };
	MKL_LONG strides_out[2] = { 0, size2D };

	// Create 1D FFT of COMPLEX DOUBLE case
	status = DftiCreateDescriptor(&my_desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, z.n);
	status = DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	status = DftiSetValue(my_desc_handle, DFTI_BACKWARD_SCALE, 1.0 / z.n);
	status = DftiSetValue(my_desc_handle, DFTI_INPUT_STRIDES, strides_in);
	status = DftiSetValue(my_desc_handle, DFTI_OUTPUT_STRIDES, strides_out);
	status = DftiCommitDescriptor(my_desc_handle);


	dtype *f_sol_gemv = alloc_arr<dtype>(size);
	dtype *u_FFT = alloc_arr<dtype>(size);

	// Direct Fourier to right-hand side
	for (int w = 0; w < size2D; w++)
	{
		status = DftiComputeForward(my_desc_handle, (void*)&u[w], &u_FFT[w]);

		//status = DftiComputeForward(my_desc_handle, f_FFT_in, &f_FFT_out[z.n * w]);
	}

	time = omp_get_wtime();

	for (int k = 0; k < z.n; k++)
	{
		if (D2csr[k]->solve == 1)
		{
			count++;
			//pardiso(&pt[k * 64], &maxfct, &mnum, &mtype, &phase, &size2D, D2csr[k]->values, D2csr[k]->ia, D2csr[k]->ja, &perm[k * size2D], &rhs, &iparm[k * 64], &msglvl, &f_FFT[k * size2D], &x_sol_prd[k * size2D], &error);
			mkl_zcsrgemv("no", &size2D, D2csr[k]->values, D2csr[k]->ia, D2csr[k]->ja, &u_FFT[k * size2D], &f_sol_gemv[k * size2D]);
		}
	}
	time = omp_get_wtime() - time;

#ifdef PRINT
	printf("Solved: %d of %d\nMissed: %d of %d\n", count, z.n, z.n - count, z.n);
	printf("time elapsed for multiplication of 2D problems: %lf sec\n", time);
	printf("Backward 1D FFT's of %d x %d times to each point of 2D solution\n", x.n_nopml, y.n_nopml);
#endif
	for (int w = 0; w < size2D; w++)
	{
		status = DftiComputeBackward(my_desc_handle, &f_sol_gemv[w], &f_sol[w]);

		// status = DftiComputeBackward(my_desc_handle, u1D, u1D_BFFT);
	}

	status = DftiFreeDescriptor(&my_desc_handle);
#ifdef PRINT
	printf("------------- The end of algorithm ----------------------\n");
#endif
	free_arr(f_sol_gemv);
	free_arr(u_FFT);
}

void GenRHSandSolution1D(size_m x, dtype* u_ex1D, dtype* f1D, double k, point sourcePML, int &src)
{
	// Set RHS
	SetRHS1D(x, f1D, sourcePML, src);

	printf("src = %d\n", src);

	// exact 1D Helmholtz
	GenExact1DHelmholtz(x.n, x, u_ex1D, k, sourcePML);
}

void GenRHSandSolution2DComplexWaveNumber(size_m x, size_m y, zcsr* D2csr, dtype* u_ex2D, dtype* f2D, dtype kwave2, point sourcePML, int &src)
{
	double norm = 0;

	// Set RHS
	SetRHS2D(x, y, f2D, sourcePML, src);

	dtype alpha_k = f2D[src] / (1.0 / (x.h * y.h));

#ifdef PRINT_TEST
	printf("src 2D = %d\n", src);
	printf("alpha_k = (%lf, %lf)\n", alpha_k.real(), alpha_k.imag());
	printf("kwave2 = (%lf, %lf)\n", kwave2.real(), kwave2.imag());
#endif

	// exact 1D Helmholtz
	{
		//H(sqrt(a + ib))
		get_exact_2D_Hankel(x.n, y.n, x, y, u_ex2D, sqrt(kwave2), sourcePML);

		normalization_of_exact_sol(x.n, y.n, x, y, u_ex2D, alpha_k);
#if 0
		norm = resid_2D_Hankel(x, y, D2csr, u_ex2D, f2D, sourcePML);
		norm /= x.h * x.h;
#ifdef PRINT_TEST
		printf("resid ||A * x_sol - kf|| 2D Hankel: %lf\n", norm);
#endif
#endif
	}
}

void GenRHSandSolution2DComplexWaveNumberHSS(size_m x, size_m y, dtype* u_ex2D, dtype* f2D, dtype kwave2, point sourcePML, int &src)
{
	double norm = 0;

	// Set RHS
	SetRHS2D(x, y, f2D, sourcePML, src);

	printf("src 2D = %d\n", src);

	dtype alpha_k = f2D[src] / (1.0 / (x.h * y.h));

	printf("alpha_k = (%lf %lf)\n", alpha_k.real(), alpha_k.imag());

	// exact 1D Helmholtz
	{
		//H(sqrt(a + ib))
		get_exact_2D_Hankel(x.n, y.n, x, y, u_ex2D, sqrt(kwave2), sourcePML);

		normalization_of_exact_sol(x.n, y.n, x, y, u_ex2D, alpha_k);

		//norm = resid_2D_Hankel(x, y, D2csr, u_ex2D, f2D, sourcePML);
		//norm *= x.h * x.h;
		//printf("resid ||A * x_sol - kf|| 2D Hankel: %lf\n", norm);

	}

	// Adjust RHS
#ifdef SYMMETRY
	AdjustRHS(x, y, f2D);
#endif

}

void GenRHSandSolution2D(size_m x, size_m y, zcsr* D2csr, dtype* u_ex2D, dtype* f2D, double kwave2, point sourcePML, int &src)
{
	double norm = 0;

	// Set RHS
	SetRHS2D(x, y, f2D, sourcePML, src);

	printf("src 2D = %d\n", src);

	dtype alpha_k = f2D[src] / (1.0 / (x.h * y.h));

	printf("alpha_k = (%lf, %lf)\n", alpha_k.real(), alpha_k.imag());

	// exact 1D Helmholtz

	if (kwave2 > 0)
	{
		// H(k)
		get_exact_2D_Hankel(x.n, y.n, x, y, u_ex2D, sqrt(kwave2), sourcePML);

		normalization_of_exact_sol(x.n, y.n, x, y, u_ex2D, alpha_k);

		norm = resid_2D_Hankel(x, y, D2csr, u_ex2D, f2D, sourcePML);
		printf("resid ||A * x_sol - kf|| 2D Hankel: %lf\n", norm);

	}
	else // kwave2 < 0
	{
		// H(ik)
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

#pragma omp parallel for simd schedule(static)
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

void Solve1DSparseHelmholtz(size_m x, size_m y, size_m z, dtype *f1D, dtype *x_sol_prd, double thresh, double beta_eq)
{
	// Init condtitions: N = 1200, ppw = 26, omega = 4, sponge = 200 -  4 %
	printf("-----------Test 1D Helmholtz--------\n");
	zcsr *D1csr;
	int size1D = x.n;
	int size1D_nopml = x.n_nopml;
	int non_zeros_in_1D3diag = size1D + (size1D - 1) * 2;
	D1csr = (zcsr*)malloc(sizeof(zcsr));
	D1csr->values = alloc_arr<dtype>(non_zeros_in_1D3diag);
	D1csr->ia = alloc_arr<int>(size1D + 1);
	D1csr->ja = alloc_arr<int>(non_zeros_in_1D3diag);
	D1csr->ia[size1D] = non_zeros_in_1D3diag + 1;
	D1csr->non_zeros = non_zeros_in_1D3diag;

#ifdef PRINT
	printf("Non-zeros in 2D block-diagonal: %d\n", non_zeros_in_1D3diag);
	printf("----------Generating 2D matrix and rhs + solving by pardiso-------\n");
	printf("Size of system: %d x %d with PML %d on each direction\n", x.n, y.n, 2 * x.pml_pts);
#endif

	point sourcePML = { x.l / 2.0 };
	
	printf("L = %lf\n", x.l);
	printf("PML = %lf\n", x.pml_pts * x.h);
	printf("SOURCE in 2D WITH PML AT: (%lf)\n", sourcePML.x);
	double k = double(kk);

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

	double ppw = 1.0 / (sqrt(abs(kwave2)) / (2.0 * double(PI))) / z.h;
	printf("ppw = %lf\n", ppw);

	// источник в каждой задаче в середине 

	GenSparseMatrixOnline1DwithPML(0, x, y, z, D1csr, kwave2, beta_eq);

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

void check_norm_result2_1D(int n1, int n2, int niter, double ppw, double spg, const dtype* x_orig_nopml, const dtype* x_sol_nopml,
	double* x_orig_re, double* x_orig_im, double *x_sol_re, double *x_sol_im)
{
	printf("------------ACCURACY CHECK---------\n");

	int size1D = n1;
	int size = size1D * n2;

	double eps1p = 0.01;
	double *norms_re = alloc_arr<double>(n2);
	double *norms_im = alloc_arr<double>(n2);

#pragma omp parallel for simd schedule(static)
	for (int i = 0; i < size; i++)
	{
		x_orig_re[i] = x_orig_nopml[i].real();
		x_orig_im[i] = x_orig_nopml[i].imag();
		x_sol_re[i] = x_sol_nopml[i].real();
		x_sol_im[i] = x_sol_nopml[i].imag();
	}

	for (int k = 0; k < n2; k++)
	{
		double norm = RelError(zlange, size1D, 1, &x_sol_nopml[k * size1D], &x_orig_nopml[k * size1D], size, 0.001);
		norms_re[k] = RelError(dlange, size1D, 1, &x_sol_re[k * size1D], &x_orig_re[k * size1D], size, eps1p);
		norms_im[k] = RelError(dlange, size1D, 1, &x_sol_im[k * size1D], &x_orig_im[k * size1D], size, eps1p);
		//if (niter == 11) printf("i = %d norm = %lf norm_re = %lf norm_im = %lf\n", k, norm, norms_re[k], norms_im[k]);
		printf("i = %d norm = %lf norm_re = %lf norm_im = %lf\n", k, norm, norms_re[k], norms_im[k]);
	}

#if 0
	FILE* fout;
	char str[255];
	sprintf(str, "Nit%d_N%d_Lx%d_FREQ%d_PPW%4.2lf_SPG%6.lf_BETA%5.3lf.dat", niter, n1, (int)LENGTH_X, (int)nu, ppw, spg, beta_eq);
	fout = fopen(str, "w");

	for (int k = 0; k < n3; k++)
		fprintf(fout, "%d %lf %lf\n", k, norms_re[k], norms_im[k]);

	fclose(fout);
#endif

	free_arr(norms_re);
	free_arr(norms_im);
}

void Solve2DSparseHelmholtz(size_m x, size_m y, size_m z, dtype *f2D, dtype *x_sol_prd, double thresh, double beta_eq)
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

	beta_eq = 0.05;

	int maxfct = 1;
	int mnum = 1;
	int phase = 13;
	int rhs = 1;
	int msglvl = 0;
	int error = 0;

	// Memory for 2D CSR matrix
	zcsr *D2csr;
	int non_zeros_in_2Dblock9diag = (x.n + (x.n - 1) * 2) * y.n + 2 * (size2D - x.n) + 4 * (x.n - 1) * (y.n - 1);
	int non_zeros_in_2Dblock3diag = (x.n + (x.n - 1) * 2) * y.n + 2 * (size2D - x.n);
	
	int non_zeros = non_zeros_in_2Dblock3diag;
	
	D2csr = (zcsr*)malloc(sizeof(zcsr));
	D2csr->values = alloc_arr<dtype>(non_zeros);
	D2csr->ia = alloc_arr<int>(size2D + 1);
	D2csr->ja = alloc_arr<int>(non_zeros);
	D2csr->ia[size2D] = non_zeros + 1;
	D2csr->non_zeros = non_zeros;

	point sourcePML = { x.l / 2.0, y.l / 2.0 };

	printf("Lx = %lf, Ly = %lf\n", x.l, y.l);
	printf("PML_x = %lf, PML_y = %lf\n", x.pml_pts * x.h, y.pml_pts * y.h);
	printf("SOURCE in 2D WITH PML AT: (%lf, %lf)\n", sourcePML.x, sourcePML.y);
	double k = double(kk);
	int nhalf = z.n / 2;
	int src = 0;

	char *str1, *str2, *str3;
	str1 = alloc_arr<char>(255);
	str2 = alloc_arr<char>(255);
	str3 = alloc_arr<char>(255);
	bool pml_flag = false;


	int count = 0;
	int i = nhalf;

	//double kww = 4.0 * double(PI) * double(PI) * (i - nhalf) * (i - nhalf) / (z.l * z.l);
	double kww = 0;
	double kwave2 = k * k - kww;

	dtype alpha_k;
	dtype *x_sol_ex = alloc_arr<dtype>(size2D);

	dtype *x_sol_ex_nopml = alloc_arr<dtype>(size2D_nopml);
	dtype *x_sol_prd_nopml = alloc_arr<dtype>(size2D_nopml);


	double omega_loc = 2.0 * double(PI) * nu;
	double norm = 0;

	//double ppw = c / nu / x.h;


	// источник в каждой задаче в середине 

	GenSparseMatrixOnline2DwithSPONGE(i, x, y, z, D2csr, kwave2, beta_eq);
	//int *freqs = alloc_arr<int>(size2D_nopml);
	//GenSparseMatrixOnline2DwithPMLand9Pts(-1, x, y, D2csr, 0, freqs);

	// Gen RHS and exact solution; check residual |A * u - f|
	GenRHSandSolution2D(x, y, D2csr, x_sol_ex, f2D, kwave2, sourcePML, src);

	// normalization of rhs

	pardiso(pt, &maxfct, &mnum, &mtype, &phase, &size2D, D2csr->values, D2csr->ia, D2csr->ja, perm, &rhs, iparm, &msglvl, f2D, x_sol_prd, &error);

	double eps = 0.01; // 1 percent


	sprintf(str1, "Charts2D/model_pml_%lf", kwave2);
	sprintf(str2, "Charts2D/model_pml_ex_%lf", kwave2);
	sprintf(str3, "Charts2D/model_pml_pard_%lf", kwave2);

	pml_flag = false;

#if 0
	for (int j = 0; j < y.n; j++)
		for (int i = 0; i < x.n; i++)
			if (x_sol_ex[i + x.n * j].real() < -1.0) printf("i = %d j = %d, l = %d, lx = %lf, ly = %lf\n",
				i, j, i + x.n * j, i * x.h, j * y.h);
#endif

	NullifySource2D(x, y, x_sol_ex, src, 3);
	NullifySource2D(x, y, x_sol_prd, src, 3);
	x_sol_ex[src] = x_sol_prd[src] = 0;

	output2D(str1, pml_flag, x, y, x_sol_ex, x_sol_prd);

	gnuplot2D(str1, str2, pml_flag, 3, x, y);
	gnuplot2D(str1, str3, pml_flag, 5, x, y);

	reducePML2D(x, y, size2D, x_sol_ex, size2D_nopml, x_sol_ex_nopml);
	reducePML2D(x, y, size2D, x_sol_prd, size2D_nopml, x_sol_prd_nopml);

	double *x_orig_re = alloc_arr<double>(size2D_nopml);
	double *x_sol_re = alloc_arr<double>(size2D_nopml);
	double *x_orig_im = alloc_arr<double>(size2D_nopml);
	double *x_sol_im = alloc_arr<double>(size2D_nopml);
	check_norm_result2_1D(x.n_nopml, y.n_nopml, 0, 0, 2 * z.spg_pts * z.h, x_sol_ex_nopml, x_sol_prd_nopml, x_orig_re, x_orig_im, x_sol_re, x_sol_im);

	norm = RelError(zlange, size2D_nopml, 1, x_sol_prd_nopml, x_sol_ex_nopml, size2D_nopml, thresh);
	printf("Norm 2D solution ||x_sol - x_ex|| / ||x_ex|| = %lf\n", norm);

//	check_exact_sol_Hankel(alpha_k, kwave2, x, y, x_sol_prd_nopml, thresh);

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
#pragma omp parallel for simd schedule(static)
	for (int i = 0; i < n; i++)
		c[i] = alpha * a[i] + beta * b[i];
}

void AddDenseVectorsComplex(int n, dtype alpha, dtype *a, dtype beta, dtype *b, dtype *c)
{
#pragma omp parallel for simd schedule(static)
	for (int i = 0; i < n; i++)
		c[i] = alpha * a[i] + beta * b[i];
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

#pragma omp parallel for simd schedule(static)
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

#pragma omp parallel for simd schedule(static)
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

#pragma omp parallel for simd schedule(static)
	for (int i = 0; i < n; i++)
		tb[i] = f[i];

	for (int k = 1; k < nbr; k++)
	{
		RecMultL(n, 1, &G[ind(k - 1, n) + ldg * 0], ldg, &tb[ind(k - 1, n)], size, y, n, smallsize);	
		DenseDiagMult(n, &B[ind(k - 1, n)], y, y);

#pragma omp parallel for simd schedule(static)
		for (int i = 0; i < n; i++)
			tb[ind(k, n) + i] = f[ind(k, n) + i] - y[i];

	}

	RecMultL(n, 1, &G[ind(nbr - 1, n) + ldg * 0], ldg, &tb[ind(nbr - 1, n)], size, &x[ind(nbr - 1, n)], size, smallsize);

	for (int k = nbr - 2; k >= 0; k--)
	{
		DenseDiagMult(n, &B[ind(k, n)], &x[ind(k + 1, n)], y);

#pragma omp parallel for simd schedule(static)
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
#pragma omp parallel for simd schedule(static)
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
#pragma omp parallel for simd schedule(static)
		for (int j = 0; j < n1; j++)
			for (int i = 0; i < n2; i++)
				A[i + n1 + lda * (0 + j)] *= d[n1 + i]; // вторая часть массива D

		// VT * D - каждый j-ый столбец умножается на элемент вектора d[j]
#pragma omp parallel for simd schedule(static)
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