from pathlib import Path

ROOTDIR = Path(__file__).parent.parent
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
