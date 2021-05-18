#!/usr/bin/tcsh

date +"%T"

cd ~/../../../../../../elhanan/SCRATCH/OrS/FBA_v2.0

pwd

echo Running job: $1
echo Command:     $2

@ sleep_time = 20 * $1
sleep $sleep_time

date +"%T"

matlab -nodisplay -nojvm -nosplash -r "cd ../cobratoolbox; initCobraToolbox; cd ../FBA_v2.0;  $2"
