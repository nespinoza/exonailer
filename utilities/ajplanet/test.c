#include<stdio.h>
#include<math.h>
#include<gsl/gsl_math.h>
#include "planet.h"

int main (void){

  double res = 0.0, res2 = 0.0;
  double P = 4.0;
  int i;
  double phase;
  

  res = pl_true_anomaly(2e0, 0.0e0, 0.99e0, P);


  for(i = 0; i<= 20; i++){
    phase = ((double) i / 20.0) * 37.18;
    res = pl_rv(phase, 0.98e0, 37.18e0, 0.755 + 5. * M_PI / 180., 0.361, 0.0e0, 37.18e0);
    res2 =   pl_true_anomaly(phase, 0.0, 0.361, 37.18e0);
    printf("%10.7f %10.7f %10.7f \n",res, res2, phase/37.18);
  }
  printf("\n");

  
  res = pl_t_from_f(0., 0.99 * M_PI, 0.0, P);
  printf("\n->%9.3f\n",res);
  res = pl_t_from_f(0., 1.01 * M_PI, 0.0, P);
  printf("\n->%9.3f\n",res);

  P = 1.0;
  pl_transit_times(0.5, 1.75 *  M_PI, 0.0, P, &res, &res2);
  printf("%f %f\n",res,res2);

  res = fmod(-0.1, 2 * M_PI);
  printf("fmod: %9.3f\n",res);

}
