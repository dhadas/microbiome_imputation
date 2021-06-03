#!/usr/bin/tcsh

date +"%T"

cd ~/../../../../../../elhanan/PROJECTS/MAT_IMPUTE_DH

pwd

echo Running job: $1
echo Command:     $2

@ sleep_time = 20 * $1
sleep $sleep_time

date +"%T"


python distributed_RF.py "$2"


#matlab -nodisplay -nojvm -nosplash -r "cd ../cobratoolbox; initCobraToolbox; cd ../FBA_v2.0;  $2"
