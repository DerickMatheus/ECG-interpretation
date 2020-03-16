for f in $(cat ids)
do
    python run.py with sim=499 id_ecg="$f" real='../plots/real'"$f" noise='../plots/noise'"$f" output_name='../results_pkdd/all/' output_name_mean='../results_pkdd/mean/id_'"$f"
done
