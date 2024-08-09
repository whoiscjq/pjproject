# How to run

# PRP
- Before running, you need to change the dataset filepath (input_path) in PRP/code/run.py  manually
```
cd llm
python PRP/code/run.py
```

# SC
- Before running, you need to change the dataset file path (input_path) in SC/main.py and SC/datawrite.py manually

```
cd llm
bash SC/run.sh
```


# SC_prp
- Before running, you need to change the dataset file path (input_path) in SC_prp/main.py and SC_prp/datawrite.py manually
you may increase the max_length of the model input
```
cd llm
bash SC_prp/myrun.sh
```