#include<stdio.h>
#include<math.h>
#include<gsl/gsl_math.h>

#include "planet.h"

/* This file contains a subroutine that returns the times
   of transit (primary and secondary given the orbital elements */

/* t_1 happens when f + w = PI/2; t_2 when f + w = 1.5 PI */

double pl_t_from_f(double e, double f, double t0,  double period){

  double ecc_anomaly, M, f_mod, t;

  f_mod = fmod(f, 2 * M_PI);

  if (f_mod < 0)
    f_mod = 2 * M_PI + f_mod;

  /* first, figure eccentric anomaly given f 
     tan f/2 = sqrt(1+e / 1-e) tan E/2 */

  if (f == M_PI) {
    ecc_anomaly = M_PI;
  }
  else{
    ecc_anomaly = 2 * atan( pow((1-e) / (1+e),0.5) * tan (f_mod/2) );
    
    /*printf("\nEA%9.3f\n",ecc_anomaly);    */
    
    if (f_mod > M_PI){

      ecc_anomaly = 2.0 * M_PI + ecc_anomaly;
      
    }
    
  }
  
  /* now solve for mean anomaly and therefore t */
  
  M = ecc_anomaly - e * sin(ecc_anomaly); 

  t = t0 + period * M / (2 * M_PI); 

  /* printf("%9.3f %9.3f %9.3f %9.3f\n",f_mod, ecc_anomaly, M, t); */

  return t;
}

void pl_transit_times(double e, double w, double t0,  double period,
		   double * t1, double *t2){
  
  *t1 = pl_t_from_f(e, 0.5 * M_PI - w, t0, period);
  *t2 = pl_t_from_f(e, 1.5 * M_PI - w, t0, period);
}
