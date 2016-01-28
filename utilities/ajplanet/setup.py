

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('',parent_package,top_path)
    config.add_extension('ajplanet',sources=['ajplanet.pyf','convlib.c','true_anomaly.c','rv.c','orbit.c','occultquad.f','occultnl.f'],include_dirs=['', '/usr/local/include'],library_dirs=['/usr/local/lib'], libraries=['gsl','gslcblas','m'])
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

