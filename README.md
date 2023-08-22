# Competitive_network

Code for the paper "Efficient computation by molecular competition networks".

## Introduction


## Install

You need to install Python 3.9, Pytorch 1.12.1, then

```
pip install -r requirements.txt
```

## Usage

1. Generate the data.

   ```
   python generate_data.py
   ```
2. Train the model.
   Modify configs/xxx.yaml content run the desired expriment with appropriate hyperparameters. After running, the results will be saved in `results_dir`.

   ```
   python train.py -c configs/xxx.yaml
   ```
3. Visualize the result.

   Change the `file_path` in plot_demo.py to the path generated in the previous step, then

   ```
   python plot_demo.py
   ```

## Citation

If you find this repo useful, please cite our paper:

```
@article{Cai2023Efficient,
  title={Efficient computation by molecular competition networks},
  author={Cai, Haoxiao and Wei, lei and Zhang, Xiaoran and Qiao, Rong and Wang, Xiaowo},
  journal={bioRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License

This project is licensed under the terms of the MIT license.
