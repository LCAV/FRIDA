#!/bin/bash

# This will run all the scripts to reproduce the figures of the paper
# FRIDA: FRI-based DOA estimation with arbitrary array geometry
# by Hanjie Pan, Robin Scheibler, Eric Bezzam, Ivan Dokmanic, and Martin Vetterli
# 
# This bash file was written by Robin Scheibler, September 2016
# License: CC-BY-SA 4.0

# Config
########

# Enable test mode. This will run a single loop of each
# script. This can be used to test if the configuration
# is correct
ENABLE_TEST=1

# This sets the number of nodes in the ipyparallel cluster
# If no cluster is used, this can be set to zero to run
# in serial mode (super slow though...)
# This number can be set to the number of available threads of
# your CPU minus 1. Usually, the number of threads is twice
# the number of cores.
PARALLEL_WORKERS=4


# Run all the scripts
#####################

# Prepare parallel processing
if [ $PARALLEL_WORKERS -gt 0 ]; then
  echo "Starting ${PARALLEL_WORKERS} ipyparallel workers."
  ipcluster start -n ${PARALLEL_WORKERS} --daemonize
  echo "Wait for 30 seconds to give time to engines to start..."
  sleep 30
  SERIAL_FLAG=
else
  SERIAL_FLAG=-s
fi

# Process test flag
if [ $ENABLE_TEST -eq 1 ]; then
  TEST_FLAG=-t
else
  TEST_FLAG=
fi

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

