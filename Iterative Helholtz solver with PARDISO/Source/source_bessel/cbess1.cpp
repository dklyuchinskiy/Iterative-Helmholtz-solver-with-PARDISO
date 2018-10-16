/*************************************************************************
*            Functions used By programs TZBESJ, TZBESK, TZBESY           *
*    (Evalute Bessel Functions with complex argument, 1st to 3rd kind)   *
* ---------------------------------------------------------------------- *
* Reference:  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES, 1983.       *
*                                                                        *
*                         C++ Release By J-P Moreau, Paris (07/05/2005). *
*                                     (www.jpmoreau.fr)                  *
*************************************************************************/
#include "../definitions.h"

#include "complex.h"

  //Function defined below
  void ZWRSK(REAL ZRR, REAL ZRI, REAL FNU, int KODE, int N, REAL *YR, REAL *YI,
             int *NZ, REAL *CWR, REAL *CWI, REAL TOL, REAL ELIM, REAL ALIM);

  //Functions defined in CBess0.cpp
  void ZSERI(REAL, REAL, REAL, int, int, REAL *, REAL *, int *, REAL, REAL, REAL);

  void ZASYI(REAL, REAL, REAL, int, int, REAL *, REAL *, int *, REAL, REAL, REAL, REAL);

  void ZMLRI(REAL, REAL, REAL, int, int, REAL *, REAL *, int *, REAL);

  void ZS1S2(REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, int *, REAL, REAL, int *);

  void ZSHCH(REAL, REAL, REAL *, REAL *, REAL *, REAL *);

  REAL DGAMLN(REAL, int *);

  void ZUCHK(REAL, REAL, int *, REAL, REAL);

  void ZUOIK(REAL, REAL, REAL, int, int, int, REAL *, REAL *, int *, REAL, REAL, REAL);

  void ZRATI(REAL, REAL, REAL, int, REAL *, REAL *, REAL);

  //Function defined in CBess0
  void ZBKNU(REAL, REAL, REAL, int, int, REAL *, REAL *, int *, REAL, REAL, REAL);

  //Function defined inCBess2.cpp
  void ZBUNI(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *CYR, REAL *CYI, 
	         int *NZ, int NUI, int *NLAST, REAL FNUL, REAL TOL, REAL ELIM, REAL ALIM);

  //for debug only
  void test(REAL, REAL, REAL, int, int, REAL *, REAL *, int *, REAL, REAL, REAL);


void ZBINU(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *CYR, REAL *CYI,
           int *NZ, REAL RL, REAL FNUL, REAL TOL, REAL ELIM, REAL ALIM) 
{
/***BEGIN PROLOGUE  ZBINU
!***REFER TO  ZBESH,ZBESI,ZBESJ,ZBESK,ZAIRY,ZBIRY

!   ZBINU COMPUTES THE I FUNCTION IN THE RIGHT HALF Z PLANE

!***ROUTINES CALLED  ZABS,ZASYI,ZBUNI,ZMLRI,ZSERI,ZUOIK,ZWRSK
!***END PROLOGUE  ZBINU */
//Labels: e10,e20,e30,e40,e50,e60,e70,e80,e100,e110,e120,e130

      REAL AZ, DFNU, ZEROI, ZEROR;
      int I, INW, NLAST, NN, NUI, NW;
      REAL *CWR, *CWI;
	  void *vmblock = NULL;

      ZEROR=0.0; ZEROI=0.0;

//    Initialize CWR, CWI
      vmblock = vminit();  
      CWR = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      CWI = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      if (! vmcomplete(vmblock)) {
        LogError ("No Memory", 0, __FILE__, __LINE__);
        return;
	  } 	 

      *NZ = 0;
      AZ = ZABS(ZR,ZI);
      NN = N;
      DFNU = FNU + 1.0*(N-1);
      if (AZ <= 2.0) goto e10;
      if (AZ*AZ*0.25 > DFNU+1.0) goto e20;
/*----------------------------------------------------------------------
!     POWER SERIES
!---------------------------------------------------------------------*/
e10:  ZSERI(ZR, ZI, FNU, KODE, NN, CYR, CYI, &NW, TOL, ELIM, ALIM);
//	  test(ZR, ZI, FNU, KODE, NN, CYR, CYI, &NW, TOL, ELIM, ALIM);
      INW = ABS(NW);
      *NZ = *NZ + INW;
      NN = NN - INW;
      if (NN == 0) {
        vmfree(vmblock);
	    return;
      }
      if (NW >= 0) goto e120;
      DFNU = FNU + 1.0*(NN-1);
e20:  if (AZ < RL) goto e40;
      if (DFNU <= 1.0) goto e30;
      if (AZ+AZ < DFNU*DFNU) goto e50;
/*----------------------------------------------------------------------
!     ASYMPTOTIC EXPANSION FOR LARGE Z
!---------------------------------------------------------------------*/
e30:  ZASYI(ZR, ZI, FNU, KODE, NN, CYR, CYI, &NW, RL, TOL, ELIM, ALIM);
      if (NW < 0) goto e130;
      goto e120;
e40:  if (DFNU <= 1.0) goto e70;
/*----------------------------------------------------------------------
!     OVERFLOW AND UNDERFLOW TEST ON I SEQUENCE FOR MILLER ALGORITHM
!---------------------------------------------------------------------*/
e50:  ZUOIK(ZR, ZI, FNU, KODE, 1, NN, CYR, CYI, &NW, TOL, ELIM, ALIM);
      if (NW < 0) goto e130;
      *NZ = *NZ + NW;
      NN = NN - NW;
      if (NN == 0) {
        vmfree(vmblock);
	    return;
      }
      DFNU = FNU+1.0*(NN-1);
      if (DFNU > FNUL) goto e110;
      if (AZ > FNUL) goto e110;
e60:  if (AZ > RL) goto e80;
/*----------------------------------------------------------------------
!     MILLER ALGORITHM NORMALIZED BY THE SERIES
!---------------------------------------------------------------------*/
e70:  ZMLRI(ZR, ZI, FNU, KODE, NN, CYR, CYI, &NW, TOL);
      if (NW < 0) goto e130;
      goto e120;
/*----------------------------------------------------------------------
!     MILLER ALGORITHM NORMALIZED BY THE WRONSKIAN
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!     OVERFLOW TEST ON K FUNCTIONS USED IN WRONSKIAN
!---------------------------------------------------------------------*/
e80:  ZUOIK(ZR, ZI, FNU, KODE, 2, 2, CWR, CWI, &NW, TOL, ELIM, ALIM);
      if (NW >= 0) goto e100;
      *NZ = NN;
      for (I=1; I<=NN; I++) {
        CYR[I] = ZEROR;
        CYI[I] = ZEROI;
      }
	  vmfree(vmblock);
      return;
e100: if (NW > 0) goto e130;
      ZWRSK(ZR, ZI, FNU, KODE, NN, CYR, CYI, &NW, CWR, CWI, TOL, ELIM, ALIM);
      if (NW < 0) goto e130;
      goto e120;
/*----------------------------------------------------------------------
!     INCREMENT FNU+NN-1 UP TO FNUL, COMPUTE AND RECUR BACKWARD
!---------------------------------------------------------------------*/
e110: NUI = (int) (floor(FNUL-DFNU) + 1);
      NUI = IMAX(NUI,0);
      ZBUNI(ZR, ZI, FNU, KODE, NN, CYR, CYI, &NW, NUI, &NLAST, FNUL,TOL,ELIM,ALIM);
      if (NW < 0) goto e130;
      *NZ = *NZ + NW;
      if (NLAST == 0) goto e120;
      NN = NLAST;
      goto e60;
e120: vmfree(vmblock);
	  return;
e130: *NZ = -1;
      if (NW == -2)  *NZ = -2;
	  vmfree(vmblock);
} //ZBINU()


void ZWRSK(REAL ZRR, REAL ZRI, REAL FNU, int KODE, int N, REAL *YR, REAL *YI,
           int *NZ, REAL *CWR, REAL *CWI, REAL TOL, REAL ELIM, REAL ALIM)  {
/***BEGIN PROLOGUE  ZWRSK
!***REFER TO  ZBESI,ZBESK
!
!     ZWRSK COMPUTES THE I BESSEL FUNCTION FOR RE(Z) >= 0.0 BY
!     NORMALIZING THE I FUNCTION RATIOS FROM ZRATI BY THE WRONSKIAN
!
!***ROUTINES CALLED  D1MACH,ZBKNU,ZRATI,ZABS
!***END PROLOGUE  ZWRSK
!     COMPLEX CINU,CSCL,CT,CW,C1,C2,RCT,ST,Y,ZR */
//Labels: e10,e20,e30,e50

      REAL ACT, ACW, ASCLE, CINUI, CINUR, CSCLR, CTI, CTR, C1I, C1R,
      C2I, C2R, PTI, PTR, RACT, STI, STR;
      int I, NW;
/*----------------------------------------------------------------------
!     I(FNU+I-1,Z) BY BACKWARD RECURRENCE FOR RATIOS
!     Y(I)=I(FNU+I,Z)/I(FNU+I-1,Z) FROM CRATI NORMALIZED BY THE
!     WRONSKIAN WITH K(FNU,Z) AND K(FNU+1,Z) FROM CBKNU.
!---------------------------------------------------------------------*/
      *NZ = 0;
      ZBKNU(ZRR, ZRI, FNU, KODE, 2, CWR, CWI, &NW, TOL, ELIM, ALIM);
      if (NW != 0) goto e50;
      ZRATI(ZRR, ZRI, FNU, N, YR, YI, TOL);
/*----------------------------------------------------------------------
!     RECUR FORWARD ON I(FNU+1,Z) = R(FNU,Z)*I(FNU,Z),
!     R(FNU+J-1,Z)=Y(J),  J=1,...,N
!---------------------------------------------------------------------*/
      CINUR = 1.0;
      CINUI = 0.0;
      if (KODE == 1) goto e10;
      CINUR = COS(ZRI);
      CINUI = SIN(ZRI);
/*----------------------------------------------------------------------
!     ON LOW EXPONENT MACHINES THE K FUNCTIONS CAN BE CLOSE TO BOTH
!     THE UNDER AND OVERFLOW LIMITS AND THE NORMALIZATION MUST BE
!     SCALED TO PREVENT OVER OR UNDERFLOW. CUOIK HAS DETERMINED THAT
!     THE RESULT IS ON SCALE.
!---------------------------------------------------------------------*/
e10:  ACW = ZABS(CWR[2],CWI[2]);
      ASCLE = 1000*D1MACH(1)/TOL;
      CSCLR = 1.0;
      if (ACW > ASCLE) goto e20;
      CSCLR = 1.0/TOL;
      goto e30;
e20:  ASCLE = 1.0/ASCLE;
      if (ACW < ASCLE) goto e30;
      CSCLR = TOL;
e30:  C1R = CWR[1]*CSCLR;
      C1I = CWI[1]*CSCLR;
      C2R = CWR[2]*CSCLR;
      C2I = CWI[2]*CSCLR;
      STR = YR[1];
      STI = YI[1];
/*----------------------------------------------------------------------
!     CINU=CINU*(CONJG(CT)/CABS(CT))*(1.0D0/CABS(CT) PREVENTS
!     UNDER- OR OVERFLOW PREMATURELY BY SQUARING CABS(CT)
!---------------------------------------------------------------------*/
      PTR = STR*C1R - STI*C1I;
      PTI = STR*C1I + STI*C1R;
      PTR = PTR + C2R;
      PTI = PTI + C2I;
      CTR = ZRR*PTR - ZRI*PTI;
      CTI = ZRR*PTI + ZRI*PTR;
      ACT = ZABS(CTR,CTI);
      RACT = 1.0/ACT;
      CTR = CTR*RACT;
      CTI = -CTI*RACT;
      PTR = CINUR*RACT;
      PTI = CINUI*RACT;
      CINUR = PTR*CTR - PTI*CTI;
      CINUI = PTR*CTI + PTI*CTR;
      YR[1] = CINUR*CSCLR;
      YI[1] = CINUI*CSCLR;
      if (N == 1) return;
      for (I=2; I<=N; I++) {
        PTR = STR*CINUR - STI*CINUI;
        CINUI = STR*CINUI + STI*CINUR;
        CINUR = PTR;
        STR = YR[I];
        STI = YI[I];
        YR[I] = CINUR*CSCLR;
        YI[I] = CINUI*CSCLR;
      }
      return;
e50:  *NZ = -1;
      if (NW = -2) *NZ=-2;
} //ZWRSK()

//end of file cbess1.cpp
