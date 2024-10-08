# Get an A in Math: Progressive Rectification Prompting (AAAI 2024)
PRP iterates a verify-then-rectify process to progressively identify incorrect answers and rectify the reasoning paths.

Check out our [22-page paper](https://arxiv.org/abs/2312.06867) for more information.

![image](https://github.com/wzy6642/PRP/blob/main/img/framework.PNG)

## Requirements

```python
Python==3.9.12
pip install -r requirements.txt
```

## Quick Start

```python
python run.py --data_index 0
```

## Experimental Results

![image](https://github.com/wzy6642/PRP/blob/main/img/experiment.png)

## Citing PRP
```markdown
@inproceedings{wu2024prp,
  title={Get an A in Math: Progressive Rectification Prompting},
  author={Wu, Zhenyu and Jiang, Meng and Shen, Chao},
  booktitle={The Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI 2024)},
  year={2024}
}
```

## License

This project is licensed under the Apache-2.0 License.
