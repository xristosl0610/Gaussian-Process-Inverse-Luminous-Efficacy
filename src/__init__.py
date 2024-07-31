from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline


ROOTDIR = Path(__file__).parent.parent
CONFIGDIR = ROOTDIR.joinpath('config')
SRCDIR = ROOTDIR.joinpath("src")
DATADIR = ROOTDIR.joinpath("data")
OUTPUTDIR = ROOTDIR.joinpath("output")

# Plotting settings
CM = 1/2.54
FSIZE = 11
TSIZE = 18
TDIR = 'in'
MAJOR = 5.0
MINOR = 3.0
STYLE = 'default'

BENCHMARK_MODELS = {
    'linear': LinearRegression(),
    'polynomial2': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    'polynomial3': make_pipeline(PolynomialFeatures(degree=3), LinearRegression()),
    'random_forest': RandomForestRegressor(),
}