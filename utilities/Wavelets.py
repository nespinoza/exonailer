import numpy as np
from math import sqrt,log
import FWT
def getDWT(data,wavelet='daub4'):
 d_length=len(data)
 fdatalen=0
 M=0
 # First we search for the optimal data length...
 for i in range(d_length):
     min_2=2**i
     max_2=2**(i+1)
     if(min_2< d_length and max_2>d_length):
        fdatalen = max_2
        M=i+1
     elif(min_2 == d_length):
        fdatalen = d_length
        M=i
     if(fdatalen!=0):
        break
 # Now we form our zero-padded vector (we padd the zeroes to the end of the vector)...
 data_vector = np.double(np.arange(fdatalen))
 for i in range(fdatalen):
     if(i<d_length):
       data_vector[i]=data[i]
     else:
       data_vector[i]=0.0
 if(wavelet=='daub4'):
   C=np.double(np.arange(4))
   C[0]=abs((1.0+sqrt(3.0))/(4.0*sqrt(2.0)))
   C[1]=abs((3.0+sqrt(3.0))/(4.0*sqrt(2.0)))
   C[2]=abs((3.0-sqrt(3.0))/(4.0*sqrt(2.0)))
   C[3]=-abs((1.0-sqrt(3.0))/(4.0*sqrt(2.0)))
 #print C
 c_A,coeff=PerformWaveletTransform(data_vector,C,M)
 return c_A,coeff,M
 
def getIDWT(wavelet,scaling):
 w='daub4'
 data_vector=np.append(wavelet,scaling)
 M=int(log(len(data_vector))/log(2))
 if(w=='daub4'):
   C=np.double(np.arange(4))
   C[2]=(1.0+sqrt(3.0))/(4.0*sqrt(2.0))
   C[1]=(3.0+sqrt(3.0))/(4.0*sqrt(2.0))
   C[0]=(3.0-sqrt(3.0))/(4.0*sqrt(2.0))
   C[3]=(1.0-sqrt(3.0))/(4.0*sqrt(2.0))
 #print C
 Signal=PerformInverseWaveletTransform(data_vector,C,M)
 return Signal

def PerformWaveletTransform(data_vector,C,M):
# print C
 Result1,Result2=FWT.getWC(data_vector,C,len(data_vector),len(C),M)
# print Result1
# print Result2
 FinalMatrix1=np.asarray(Result1)
 FinalMatrix2=np.asarray(Result2)
 #n=len(data_vector)/2  
 #FinalMatrix1.resize(M,n)
 return FinalMatrix1,FinalMatrix2
 
def PerformInverseWaveletTransform(data_vector,C,M):
# print C
 Result=FWT.getSignal(data_vector,C,len(data_vector),len(C),M)
# print Result1
# print Result2
 Signal=np.asarray(Result)
 #n=len(data_vector)/2  
 #FinalMatrix1.resize(M,n)
 return Signal

