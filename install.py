######################## Install Script v.1.0. #########################
#                                                                      #
#            This script installs exonailer in your computer           #
#                                                                      #
########################################################################

p_name = "exonailer"

import glob
import sys
import os
import shutil
import subprocess
import tarfile
import urllib

def CheckLibraries():
    try:
      import numpy
    except ImportError:
      print "     ----------------------------------------------------------"
      print '     ERROR: '+p_name+' will not be installed in your system because'
      print '             numpy is not installed in your system.'
      print '             To install it, go to: http://www.numpy.org/\n\n'
      sys.exit(1)
    print "     > Numpy is ok!"
    try:
      import scipy
    except ImportError:
      print "     ----------------------------------------------------------"
      print '     ERROR: '+p_name+' will not be installed in your system because'
      print '             scipy is not installed in your system.'
      print '             To install it, go to: http://www.scipy.org/\n\n'
      sys.exit(1)
    print "     > Scipy is ok!"
    try:
      import astropy
    except ImportError:
      print "     ----------------------------------------------------------"
      print '     ERROR: '+p_name+' will not be installed in your system because'
      print '            astropy is not installed in your system.'
      print '            To install it, go to: http://www.astropy.org/ \n\n'
      sys.exit(1)
    print "     > Astropy is ok!"

    try:
      import batman
    except ImportError:
      print "     ----------------------------------------------------------"
      print '     ERROR: '+p_name+' will not be installed in your system because'
      print '            batman is not installed in your system.'
      print '            To install it, go to: http://astro.uchicago.edu/~kreidberg/batman/ \n\n'
      sys.exit(1)
    print "     > batman is ok!"    

    try:
      import emcee
    except ImportError:
      print "     ----------------------------------------------------------"
      print '     ERROR: '+p_name+' will not be installed in your system because'
      print '            emcee is not installed in your system.'
      print '            To install it, go to: http://dan.iel.fm/emcee/current/ \n\n'
      sys.exit(1)
    print "     > emcee is ok!"  

    try:
      import radvel
    except ImportError:
      print "     ----------------------------------------------------------"
      print '     ERROR: '+p_name+' will not be installed in your system because'
      print '            radvel is not installed in your system.'
      print '            To install it, go to: http://radvel.readthedocs.io/en/latest/ \n\n'
      sys.exit(1)
    print "     > radvel is ok!" 

    try:
      import radvel
    except ImportError:
      print "     ----------------------------------------------------------"
      print '     ERROR: '+p_name+' will not be installed in your system because'
      print '            radvel is not installed in your system.'
      print '            To install it, go to: http://radvel.readthedocs.io/en/latest/ \n\n'
      sys.exit(1)
    print "     > radvel is ok!" 

def Build(directory):
    # We obtain al files and folders of the current directory...
    files_and_folders = glob.glob(directory+'/*')
    CFileFound = False
    SetupFileFound = False
    # ...and we check each folder or file:
    for cf in files_and_folders:
        # We search for files named Proceso_f2py, or for the setup.py - file.c
        # combination. If present, we build the process.
        pf2py = 'Proceso_f2py'
        stp = 'setup.py'
        if( cf[-len(pf2py):] == pf2py ):
           print "     > Fortran code found in directory "+directory+". Building..."
           cwd = os.getcwd()
           os.chdir(directory)
           subprocess.Popen('chmod u+wrx Proceso_f2py',shell = True).wait()
           subprocess.Popen('chmod g+wrx Proceso_f2py',shell = True).wait()
           subprocess.Popen('chmod o+wrx Proceso_f2py',shell = True).wait()
           p = subprocess.Popen('./Proceso_f2py',stdout = subprocess.PIPE, stderr = subprocess.PIPE,shell = True)
           p.wait()
           if(p.returncode != 0 and p.returncode != None):
             print "     ----------------------------------------------------------"
             print "     > ERROR: "+p_name+" couldn't be installed."
             print "     > Problem building code in "+directory+". The error was:\n"
             out, err = p.communicate()
             print err
             print "     > If you can't solve the problem, please communicate"
             print "     > with the "+p_name+" team for help.\n\n"
             os.chdir(cwd)
             sys.exit()
           os.chdir(cwd)
           print "     >...done!"
        elif( cf[-len(stp):] == stp ):
           SetupFileFound = True
        elif( cf[-2:] == '.c' ):
           CFileFound = True
        if( SetupFileFound and CFileFound ):
           print "     > C code found in directory "+directory+". Building..."
           cwd = os.getcwd()
           os.chdir(directory)
           p = subprocess.Popen('python setup.py build',stdout = subprocess.PIPE, stderr = subprocess.PIPE,shell = True)
           p.wait()
           if(p.returncode != 0 and p.returncode != None):
             print "     ----------------------------------------------------------"
             print "     > ERROR: "+p_name+" couldn't be installed."
             print "     > Problem building code in "+directory+". The error was:\n"
             out, err = p.communicate()
             print spaced(err,"\t \t")
             print "     > If you can't solve the problem, please communicate"
             print "     > with the "+p_name+" team for help.\n \n"
             os.chdir(cwd)
             sys.exit()
           libfolder = getDirs('build/.')
           for name in libfolder:
               if(name[0:3]=='lib'):
                 filename = glob.glob('build/'+name+'/*')
                 shutil.copy2(filename[0],'../.')
           shutil.rmtree('build')
           os.chdir(cwd)
           print '     >...done!'
           break

def getDirs(foldername):
    return os.walk(foldername).next()[1]

def spaced(input,space):
    fixed = False
    i = 0
    input = space+input
    while(not fixed):
        if(input[i:i+1] == '\n'):
           input = input[0:i+1]+space+input[i+1:]
           i = i + len(space)
        i = i + 1
        if(i == len(input)-1):
          fixed = True
    return input

CheckLibraries()
Build('utilities/flicker-noise')
