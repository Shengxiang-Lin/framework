# framework

## Quick Start
1. install pytorch and other dependencies
```bash
conda create -n PBAT python=3.8   
conda activate PBAT      
pip install -r requirements.txt
pip install "jsonargparse[signatures]"
```
2. run the model with a `yaml` configuration file like following:
```bash
python run.py fit --config src/configs/retail.yaml
```

## Dataset
Due to file size limitations, we have not uploaded all of the data. You can download the datasets from [releases](https://github.com/Shengxiang-Lin/PBAT/releases).

## Log View   
```bash
pip install tensorboard==2.14.0
pip install protobuf==3.20.3.3
cd logs/yelp/full/lightning_logs/version_0   
tensorboard --logdir=.
```
Then open http://localhost:6006/    