# DPASyn

Hi, this is the code of our paper "DPASyn: Mechanism-Aware Drug Synergy Prediction via Dual Attention and Precision-Aware Quantization" accepted by IEEE BIBM 2025. Our paper is available [here](https://ieeexplore.ieee.org/abstract/document/11356358).


### Prepare the Data

Refer to the `Data.md` file located in the `data/` directory for detailed instructions on how to download and preprocess the dataset. The default data path used by the script is `data/data.pt`. Make sure your processed data is placed accordingly.

### Run the Script

```bash
python main.py
```

By default, the script will train the model using the configuration specified in the code. You can modify hyperparameters directly in `main.py` or extend it to support command-line arguments if needed. 

### ✍Citation
```
@article{nie2025dpasyn,
  title={DPASyn: Mechanism-Aware Drug Synergy Prediction via Dual Attention and Precision-Aware Quantization},
  author={Nie, Yuxuan and Song, Yutong and Yang, Jinjie and Song, Yupeng and Zhou, Yujue and Peng, Hong},
  journal={arXiv preprint arXiv:2505.19144},
  year={2025}
}
```
