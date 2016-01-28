#include<stdio.h>
#include<math.h>
#include<gsl/gsl_math.h>

#include "planet.h"

/* This file contains a function that returns the orbit of an exoplanet 
   */

void pl_Orbit(double t, double t0, double P, double a, double e, double w, double i, double Omega, double * x, double * y, double * z){
  
  double f, r,ang_arg;
  
  /*mod = fmod((t-t0)/P, 1.0);
  
  if (mod < 0)
  mod += 1; */
  
  /*  printf(" %9.3f ",mod); */
  
  f = pl_true_anomaly(t, t0, e, P);
  
  /*if (mod > 0.5)
    f  = 2 * M_PI - f; */
  
  /* Now compute x [0], y [1] and z [2] components of orbit as seen projected in plane of the sky*/
  
  r = a * (1 - e * e) / (1 + e * cos(f));
  
  ang_arg = (f + w);

  *x = r * (cos(Omega) * cos(ang_arg) - sin(Omega) * sin(ang_arg) * cos(i));
  *y = r * (sin(Omega) * cos(ang_arg) + cos(Omega) * sin(ang_arg) * cos(i));
  *z = r * sin (ang_arg) * sin(i);
}

void pl_Orbit_array(double *t, double t0, double P, double a, double e, double w, double inc, double Omega, double * x, double * y, double * z, int n){

  int i;

  for (i=0; i<n; i++)
    pl_Orbit(t[i], t0, P, a, e, w, inc, Omega, x+i, y+i, z+i);
  
}
   
