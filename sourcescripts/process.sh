
# run all the processing from sourcescripts

export PYTHONDONTWRITEBYTECODE=1
python3 -B /processing/process.py
python3 -B ./processing/graphdata.py


# finetune word2vec from the same directory sourcescripts
