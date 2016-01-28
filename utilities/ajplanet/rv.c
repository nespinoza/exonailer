#include<stdio.h>
#include<math.h>
#include<gsl/gsl_math.h>
#include<stdlib.h>

#include "planet.h"

double pl_rv(double t, double v0, double K, double w, double e,  double t0, double P){

  double f,M, resid, mod;

  mod = fmod((t-t0)/P, 1.0);

  if (mod < 0)
    mod += 1;

  /*  printf(" %9.3f ",mod); */

  f = pl_true_anomaly(t, t0, e, P);

  /*if (mod > 0.5)
    f  = 2 * M_PI - f;*/
  
  return ( v0 + K * ( cos(w + f) + e * cos(w) ) ); 

}

void pl_rv_array(double *t, double v0, double K, double w, double e,  double t0, double P, int n, double * out){

  int i;

  for (i=0; i<n; i++)
    out[i] = pl_rv(t[i], v0, K, w, e, t0, P);

}


