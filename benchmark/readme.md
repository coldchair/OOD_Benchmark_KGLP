# Benchmark Moudule

Pytorch implementation of KGEMs based on pykeen.

  - [x] TransE
  - [x] DistMult
  - [x] RotatE
  - [ ] TuckER
  - [ ] ComplEx


## Usage

### Training

```bash
python run.py --config_path ./config/TransE.yaml \
              --dataset_dir_path ./dataset/FB15K-237 \
              --train # do training \
              --output_dir_path ./results
```

### Evaluation Only

```bash
python run.py --config_path ./config/TransE.yaml \
              --dataset_dir_path ./dataset/FB15K-237 \
              --model_ckpt_path ./results/model.bin \
              --output_dir_path ./test-results
```


## Config file

```yaml
seed: 233 # random seed

model:
  name: TransE # model name
  embedding_dim: 400
  scoring_fct_norm: 1
  regularizer: LpRegularizer
  regularizer_kwargs:
    p: 2.0
    weight: 1.0e-4

train:
  batch_count: 5  
  # batch_size: 200  batch_size = len(dataset) / batch_count
  lr: 1.0e-4 # learning rate
  loss: multiclass_nll # loss_function in utils/loss.py
  loss_kwargs: {} # loss_function args
  eta: 30 # negative sampling number
  epochs: 4000 
  mode: ht # negative sampling mode, "ht" for head and tail, "h" for head only, "t" for tail only

eval:
  batch_size: 64 # evaluation batch_size
  ranking_strategy: worst
```