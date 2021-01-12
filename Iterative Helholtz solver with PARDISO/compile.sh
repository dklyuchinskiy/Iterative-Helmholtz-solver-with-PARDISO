module purge
module load intel/2017.4.196 compilers/intel/2017.4.196
icc -qopenmp -O2 -xHost -std=c++11 -mkl -o a.out Source/functions.cpp Source/main.cpp Source/FFT.cpp Source/FGMRES.cpp Source/BcGSTAB.cpp Source/Tests.cpp Source/source_bessel/basis_r.cpp Source/source_bessel/cbess00.cpp Source/source_bessel/cbess0.cpp Source/source_bessel/cbess1.cpp Source/source_bessel/cbess2.cpp Source/source_bessel/cbess3.cpp Source/source_bessel/complex.cpp Source/source_bessel/tzbesi.cpp Source/source_bessel/tzbesj.cpp Source/source_bessel/tzbesk.cpp Source/source_bessel/tzbesy.cpp Source/source_bessel/vmblock.cpp

