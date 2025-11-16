# DPASyn

Hi, this is the code of our paper "DPASyn: Mechanism-Aware Drug Synergy Prediction via Dual Attention and Precision-Aware Quantization" accepted by IEEE BIBM 2025. Our paper is available [here](https://arxiv.org/abs/2505.19144).


### **Prepare the Data** 

Refer to the `Data.md` file located in the `data/` directory for detailed instructions on how to download and preprocess the dataset. The default data path used by the script is `data/data.pt`. Make sure your processed data is placed accordingly.

### **Run the Script** 

```bash
python main.py
```

By default, the script will train the model using the configuration specified in the code. You can modify hyperparameters directly in `main.py` or extend it to support command-line arguments if needed. 
