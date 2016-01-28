
double pl_true_anomaly(double, double, double, double);

double pl_rv(double, double, double, double, double, double, double);

void pl_rv_array(double *, double , double , double , double ,  double , double , int ,double *);

double pl_tE(double, void *);
double pl_tE_deriv(double, void *);
void pl_tE_fdf(double, void *, double *, double *);

void pl_Orbit(double, double , double , double , double , double , double , double , double * , double * , double *); 

void pl_Orbit_array(double *, double , double , double , double , double , double , double , double * , double * , double *, int); 
	
double pl_t_from_f(double , double , double ,  double );
void pl_transit_times(double , double, double,  double , double * , double *);		 
double pl_gau(double, double, double);
void pl_Rconvolv(double *, double *, long, double, double, double);
void pl_convolv(double *, double *, long, double, double, double, double);
