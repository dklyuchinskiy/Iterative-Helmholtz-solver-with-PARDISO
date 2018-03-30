#include "definitions.h"
#include "templates.h"
#include "TestSuite.h"

void MyFFT1D_Forward(int N, double* f, dtype *f_MYFFT)
{
	for (int n = 0; n < N; n++)
		for (int k = 0; k < N; k++)
			f_MYFFT[n] +=  dtype{f[k] * cos(2 * PI * n * k / N),  -f[k] * sin(2 * PI * n *  k / N) };
}

void MyFFT1D_Backward(int N, dtype *f_MYFFT, dtype* f) /*Inverse transformation*/
{
	for (int n = 0; n < N; n++)
	{
		for (int k = 0; k < N; k++)
			f[n] += dtype{ f_MYFFT[k].real() * cos(2 * PI * n * k / N) - f_MYFFT[k].imag() * sin(2 * PI * n * k / N),
								f_MYFFT[k].real() * sin(2 * PI * n *  k / N) + f_MYFFT[k].imag() * cos(2 * PI * n * k / N) };
	
		f[n] /= N;
	}
}

void MyFFT1D_ForwardComplex(int N, dtype* f, dtype *f_MYFFT)
{
	for (int n = 0; n < N; n++)
		for (int k = 0; k < N; k++)
			f_MYFFT[n] += f[k] * dtype{cos(2 * PI * n * k / N),  -sin(2 * PI * n *  k / N) };

}