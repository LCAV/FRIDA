#!/bin/bash

# This will run all the scripts to reproduce the figures of the paper
# A fast Hadamard transform for signals sparse in the tranform domain
# by Robin Scheibler, Saeid Haghighatshoar, and Martin Vetterli
# 
# This bash file was written by Robin Scheibler, October 2016
# License: MIT

# Config
########

# Enable test mode. This will run a single loop of each
# script. This can be used to test if the configuration
# is correct
ENABLE_TEST=0

# Enable serial mode. This deactivate the use of parallel
# workers. The code runs in a straight loop.
ENABLE_SERIAL=0

# This sets the number of nodes in the ipyparallel cluster
# If no cluster is used, this can be set to zero to run
# in serial mode (super slow though...)
# This number can be set to the number of available threads of
# your CPU minus 1. Usually, the number of threads is twice
# the number of cores.
PARALLEL_WORKERS=0

# Show help function
####################
function show_help {
  echo "$1 [OPTS]"
  echo "Options:"
  echo "  -t    Runs a single loop only for test purpose"
  echo "  -s    Runs all the code in a simple for loop. No parallelism"
  echo "  -n x  Runs the loops in parallel using x workers. This option is ignored if -s is used"
}

# Process arguments
###################

# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

while getopts "h?tsn:" opt; do
  case "$opt" in
    h|\?)
      show_help $0
      exit 0
      ;;
    t)  ENABLE_TEST=1
      ;;
    s)  ENABLE_SERIAL=1
      ;;  
    n)  PARALLEL_WORKERS=$OPTARG
      ;;
  esac
done

shift $((OPTIND-1))

[ "$1" = "--" ] && shift

# Process SERIAL flag
if [ $ENABLE_SERIAL -eq 1 ]; then
  PARALLEL_WORKERS=0
fi

# Check that all necessary packages are installed
##################

python check_requirements.py
if [ $? -ne 0 ]; then
  echo "Some dependency is not met. Please check you have the packages listed in requirements.txt installed."
  echo "This can be done by running: python ./check_requirements.py"
  exit 1
fi

# Run all the scripts
#####################

# Prepare parallel processing
if [ $PARALLEL_WORKERS -gt 0 ]; then
  echo ""
  echo "Starting ${PARALLEL_WORKERS} ipyparallel workers."
  echo ""
  ipcluster start -n ${PARALLEL_WORKERS} --daemonize
  echo ""
  echo "Wait for 30 seconds to give time to engines to start..."
  echo ""
  sleep 30
  SERIAL_FLAG=
else
  SERIAL_FLAG=-s
  echo ""
  echo "Running the scripts in serial mode (no parallelism)"
  echo ""
fi

# Process test flag
if [ $ENABLE_TEST -eq 1 ]; then
  TEST_FLAG=-t
  echo "Running the script in testing mode"
else
  TEST_FLAG=
fi

if [ $ENABLE_TEST -ne 1 ] && [ $PARALLEL_WORKERS -eq 0 ]; then
  echo "#### You are about to run a very long simulation without parallel processing ####"
  echo ""
  echo "  You might want to take a look at the option -t for a quick test, or -n x to"
  echo "  use parallel processing (requires ipyparallel module)."
  echo ""
  read -n 1 -p "Press any key to go ahead."
  echo ""
fi

# Make some folders
mkdir -p figures
mkdir -p data

FLAGS="${SERIAL_FLAG} ${TEST_FLAG}"
echo "Running with flags ${FLAGS}"

# Run all the scripts and get the output data file name
echo 'Processing experiment data...'
FILE1=`python figure_doa_experiment.py ${FLAGS} | grep 'Saved data to file:' | awk '{ print $5 }'`
echo 'Running Monte-Carlo SNR simulation...'
FILE2=`python figure_doa_synthetic.py ${FLAGS} | grep 'Saved data to file:' | awk '{ print $5 }'`
echo 'Running Monte-Carlos source resolution simulation...'
FILE3=`python figure_doa_separation.py ${FLAGS} | grep 'Saved data to file:' | awk '{ print $5 }'`
echo 'Processing experiment with more loudspeakers than microphones...'
FILE4=`python figure_doa_9_mics_10_src.py | grep 'Saved data to file:' | awk '{ print $5 }'`

echo "All processing done! The data was saved in files:"
echo "  ${FILE1}"
echo "  ${FILE2}"
echo "  ${FILE3}"
echo "  ${FILE4}"

# Now produce all the plots
echo 'Creating all figures...'
python figure_doa_experiment_plot.py -f $FILE1
python figure_doa_synthetic_plot.py -f $FILE2
python figure_doa_separation_plot.py -f $FILE3
python figure_doa_9_mics_10_src_plot.py -f $FILE4

if [ $PARALLEL_WORKERS -gt 0 ]; then
  echo 'Stopping parallel processing now.'
  ipcluster stop
fi

echo 'All done. See you soon...'

