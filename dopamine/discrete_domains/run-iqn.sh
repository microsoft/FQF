#!/bin/bash

mkdir logs &> /dev/null
mkdir fout-results &> /dev/null
mkdir fout-visual &> /dev/null
mkdir ../agents/implicit_quantile/configs/gins &> /dev/null
n=0
declare -a games=("Centipede")
declare -a seeds=(0 1 2)
for game in "${games[@]}"
do
    for seed in "${seeds[@]}"
    do
        d="iqn-s${seed}"
        sed -e "s!GAME!${game}!" -e "s!RUNTYPE!$d!" ../agents/implicit_quantile/configs/implicit_quantile_icml.gin > ../agents/implicit_quantile/configs/gins/${d}_icml_${game}.gin
        CUDA_VISIBLE_DEVICES=$n nohup python train.py --base_dir=/tmp/${d}-${game} --gin_files="../agents/implicit_quantile/configs/gins/${d}_icml_${game}.gin" >& logs/output_${game}_${d} &
        echo "$i, $n"
        n=$(($n+1))
        sleep 2
    done
done

