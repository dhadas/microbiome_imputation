#!/usr/bin/tcsh

date +"%T"

cd ~/../../../../../../elhanan/PROJECTS/MAT_IMPUTE_DH

pwd

echo Running job: $4
echo Command:     distributed_RF.py $1 $2 $3 $4

@ sleep_time = 20 * $1
sleep $sleep_time

date +"%T"


python printpy.py


#matlab -nodisplay -nojvm -nosplash -r "cd ../cobratoolbox; initCobraToolbox; cd ../FBA_v2.0;  $2"
