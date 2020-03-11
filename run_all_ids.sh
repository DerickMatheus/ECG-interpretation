for f in $(cat ids)
do
    python run.py with sim=499 id_ecg="$f" model=tensorflow_resnet real='plots/real'"$f" noise='plots/noite'"$f"
done
