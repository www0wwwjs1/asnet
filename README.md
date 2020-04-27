To create exceutable environment with anaconda:
```
conda env create -f conda.yaml
conda activate reid
```

To install required python packages with pip:
```
pip install -r requirements.txt
```

To train and evaluate AsNet model on Market1501 dataset:
```
python exp_as_net.py --config-file ./configs/market.yaml --transforms random_flip random_erase --root /path/to/your/data
```

To train and evaluate AsNet model on CUHK03_Detected dataset:
```
python exp_as_net.py --config-file ./configs/cuhk03_detected.yaml --transforms random_flip random_erase --root /path/to/your/data
```

To train and evaluate AsNet model on CUHK03_Labeled dataset:
```
python exp_as_net.py --config-file ./configs/cuhk03_labeled.yaml --transforms random_flip random_erase --root /path/to/your/data
```

To train and evaluate AsNet model on DukeMTMC dataset:
```
python exp_as_net.py --config-file ./configs/market.yaml --transforms random_flip random_erase --root /path/to/your/data
```

To train and evaluate AsNet model on MARS dataset:
```
python exp_as_net.py --config-file ./configs/mars.yaml --transforms random_flip random_erase --gpu-devices 2 --root /path/to/your/data
```

All records and log file are in the `./log/` directory.