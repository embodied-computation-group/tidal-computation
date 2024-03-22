#! /bin/bash

nmcmc=10000
burn=2000
thin=2
chains=4

START=1


for (( c=$START; c<=$chains; c++ ))
do
    nohup python run_RDM_model_base_DDM.py $nmcmc $burn $thin $chains $c &
done