#!/usr/bin/env bash

echo "Spliting files..."

OUTDIR=./segmented
SLICE_DIR=./slices
SAMPLERATE=16000

# Create folders if needed
mkdir -p ${OUTDIR}/one_speaker
mkdir -p ${OUTDIR}/two_speakers
mkdir -p ${OUTDIR}/three_speakers

# Slice all files
for file in `ls ${SLICE_DIR}/speech*.slices` ; do
	echo "Slice ${file}"
	CMD="python slice_files.py -f ${file} -r ${SAMPLERATE} -o ${OUTDIR}/one_speaker"
  echo $CMD
  $CMD
done

# Move things to correct folders
mv ${OUTDIR}/one_speaker/*-*-* ${OUTDIR}/three_speakers/
mv ${OUTDIR}/one_speaker/*-* ${OUTDIR}/two_speakers/

# extract silence segment
python slice_files.py -f ${SLICE_DIR}/silence.slices -r ${SAMPLERATE} -o ${OUTDIR}

echo "Done!"

