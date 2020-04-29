#!/bin/bash

mkdir logs &> /dev/null
mkdir fout-results &> /dev/null
mkdir fout-visual &> /dev/null
mkdir ../agents/fqf/configs/gins &> /dev/null
n=0
#iqn_fqf-ws-sticky-0" "iqn_fqf-ws-sticky-0"
declare -a games=("Centipede")
declare -a seeds=(0 1 2)
declare -a factors=(0.00001)
declare -a ents=(0.0001)
declare -a optimizers=('rmsprop')
for game in "${games[@]}"
do
    for opt in "${optimizers[@]}"
    do
        for seed in "${seeds[@]}"
        do
            for factor in "${factors[@]}"
            do
                for ent in "${ents[@]}"
                do
                    d="iqn_fqf-ws-${opt}-f${factor}-e${ent}-s${seed}"
                    sed -e "s!GAME!${game}!" -e "s!RUNTYPE!$d!" -e "s!FQFFACTOR!${factor}!" -e "s!FQFENT!${ent}!" ../agents/fqf/configs/fqf.gin > ../agents/fqf/configs/gins/${d}_${game}.gin
                    CUDA_VISIBLE_DEVICES=$n nohup python train.py --base_dir=/tmp/${d}-${game} --gin_files="../agents/fqf/configs/gins/${d}_${game}.gin" >& logs/output_${game}_${d} &
                    echo "$i, $n"
                    n=$((($n+1) % 4))
                    sleep 2
                done
            done
        done
    done
done
