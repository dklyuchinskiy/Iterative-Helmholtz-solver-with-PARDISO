#include "definitions.h"
#include "templates.h"
#include "TestSuite.h"
//#define B
#define FT

void MyFFT1D_ForwardReal(int N, double* f, dtype *f_MYFFT)
{

#ifdef B
	for (int n = 0; n < N; n++)
	{
		f_MYFFT[n] = 0;
		for (int k = 0; k < N; k++)
			f_MYFFT[n] += dtype{ f[k] * cos(2 * PI * n * k / N),  -f[k] * sin(2 * PI * n *  k / N) };

		//f_MYFFT[n] /= N;
	}
#endif
#ifdef C
	for (int n = 0; n < N; n++)
		for (int k = 0; k < N; k++)
			f_MYFFT[n] += f[n] * sin (PI * (k + 1) * (n + 1) / (N + 1));
#endif

#ifdef A
	for (int k = 0; k < N; k++)
	{
		f_MYFFT[k] = 0;
		for (int j = 0; j < N; j++)
			f_MYFFT[k] += dtype{ 2 * f[j] * sin(PI * (j + 1) * (k + 1) / (N + 1)),  0};
	}
#endif
}

void MyFT1D_ForwardReal(int N, double h, double* f, dtype *f_MYFFT)
{
	int n2 = N / 2;

	for (int k = 0; k < N; k++)
	{
		f_MYFFT[k] = 0;
		for (int j = 0; j < N - 1; j++)
		{
			f_MYFFT[k] += dtype
			{
				f[j + 1] * cos(2 * PI * (j + 1) * (k - n2) * h) + f[j] * cos(2 * PI * j * (k - n2) * h),
					-(f[j + 1] * sin(2 * PI * (j + 1) * (k - n2) * h) + f[j] * sin(2 * PI * j *  (k - n2) * h))
			};

		//	if (k == 2) printf("f_MYFFT[k]: %lf %lf\n", f_MYFFT[k].real(), f_MYFFT[k].imag());
		}

		f_MYFFT[k] *= h / 2.0;

	//	f_MYFFT[k] /= N;
	}

}

dtype my_exp(double val)
{
	return dtype{ cos(val), sin(val) };
}

void MyFT1D_ForwardComplex(int N, size_m x, dtype* f, dtype *f_MYFFT)
{
	int n2 = N / 2;

	for (int k = 0; k < N; k++)
	{
		f_MYFFT[k] = 0;
		for (int j = 0; j < N - 1; j++)
		{
			f_MYFFT[k] += (f[j + 1] * my_exp(-2 * PI * (j + 1) * (k - n2) * x.h / x.l) + f[j] * my_exp(-2 * PI * j * (k - n2) * x.h / x.l));

			//	if (k == 2) printf("f_MYFFT[k]: %lf %lf\n", f_MYFFT[k].real(), f_MYFFT[k].imag());
		}

		f_MYFFT[k] *= x.h / 2.0;
		//	f_MYFFT[k] /= N;
	}
}

void MyFT1D_BackwardComplex(int N, size_m x, dtype* f_MYFFT, dtype *f)
{
	int n2 = N / 2;
	double L;

//	if (x.l == 1) L = 1.0;
//	else L = (double)LENGTH;

	for (int i = 0; i < N; i++)
	{
		f[i] = 0;
		for (int k = 0; k < N; k++)
			f[i] += f_MYFFT[k] * my_exp(2 * PI * (k - n2) * i * x.h / x.l);
	}
}

void MyFT1D_BackwardReal(int N, double h, dtype* f_MYFFT, double *f)
{
	int n2 = N / 2;

	for (int i = 0; i < N; i++)
	{
		f[i] = 0;
		for (int k = 0; k < N; k++)
			f[i] += (f_MYFFT[k].real() * cos(2 * PI * (k - n2) * i * h) - f_MYFFT[k].imag() * sin(2 * PI * (k - n2) * i * h));
	}
}

void MyFFT1D_BackwardReal(int N, dtype *f_MYFFT, double* f) /*Inverse transformation*/
{
#ifdef B
	for (int n = 0; n < N; n++)
	{
		f[n] = 0;
		for (int k = 0; k < N; k++)
			f[n] += f_MYFFT[k].real() * cos(2 * PI * n * k / N) - f_MYFFT[k].imag() * sin(2 * PI * n * k / N);
	
		f[n] /= N;
	}
#endif
#ifdef A
	for (int k = 0; k < N; k++)
	{
		f[k] = 0;
		for (int j = 0; j < N; j++)
			f[k] += (2 * f_MYFFT[j].real() * sin(PI * (j + 1) * (k + 1) / (N + 1)));

		f[k] /= 2 * (N + 1);
	}

#endif
}

void MyFFT1D_ForwardComplex(int N, dtype* f, dtype *f_MYFFT)
{
	for (int n = 0; n < N; n++)
	{
		f_MYFFT[n] = 0;
		for (int k = 0; k < N; k++)
			f_MYFFT[n] += f[k] * dtype{ cos(2 * PI * n * k / N),  -sin(2 * PI * n *  k / N) };
	}
}

void MyFFT1D_BackwardComplex(int N, dtype *f_MYFFT, dtype* f) /*Inverse transformation*/
{
	for (int n = 0; n < N; n++)
	{
		f[n] = 0;
		for (int k = 0; k < N; k++)
			f[n] += dtype{ f_MYFFT[k].real() * cos(2 * PI * n * k / N) - f_MYFFT[k].imag() * sin(2 * PI * n * k / N),
			f_MYFFT[k].real() * sin(2 * PI * n *  k / N) + f_MYFFT[k].imag() * cos(2 * PI * n * k / N) };

		f[n] /= N;
	}
}

void MyFFT1D_ForwardComplexSin(int N, dtype* f, dtype *f_MYFFT)
{
	for (int k = 0; k < N; k++)
	{
		f_MYFFT[k] = 0;
		for (int j = 0; j < N; j++)
			f_MYFFT[k] += dtype{ 2 * f[j].real() * sin(PI * (j + 1) * (k + 1) / (N + 1)),  0 };
	}
}

void MyFFT1D_BackwardComplexSin(int N, dtype* f, dtype *f_MYFFT)
{
	for (int k = 0; k < N; k++)
	{
		f[k] = 0;
		for (int j = 0; j < N; j++)
			f[k] += dtype{ 2 * f_MYFFT[j].real() * sin(PI * (j + 1) * (k + 1) / (N + 1)), 0 };

		f[k] /= 2 * (N + 1);
	}
}