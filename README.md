# COMP5331 Project: Knowledge Discovery in Time Series with Application to Financial Trading (Group 10)

This is the github repository containing all the code for this course project.

## File Description
**model/baseline.py**: baseline models contain CNN, MLP, LSTM \
**model/transformer.py**: relation-aware model \
**model/model.py**: main file for model training\
**model/model.py**: code to reproduce non subset results with model saving\


## Model Setting
There are a set of hyerparameter settings at the top of **model/model.py**

**k**: number of date in a batch  \
**l**: context window size\
**num_feature**: numer of feature for stock price. (open, high, low and close prices)\
**numBatches**: batch size\
**numStocksInSubset**: number of stocks in a batch \
**trainInvestmentLength**: observation window for training\
**numTrainEpisodes**: total number of examples for training\
**tranCostRate**: transaction cost rate\
**testInvestmentLength**: observation window for testing\
**numTestEpisodes**: total number of examples for testing\
**eval_interval**: evaluate the model for every eval_interval iterations during training\
**lr**: learning rate\
**weight_decay**: weight decay for optimizer\
**MODEL**: model selection (LSTM, CNN, MLP, transformer) \
### RAT Hyperparameters
**tran_n_layer**: number of encoder and decoder layr\
**n_head** : number of head \
**d_model**: hidden dimension of transformer\
**context_attn**: add context attention\
**relation_aware**: add relation aware layer\
**rat_b**: remove leverage operation in decision making layer\


### LSTM Hyperparameters
**lstm_hid**: hidden dimension of LSTM\
**lstm_layer**: number of LSTM laye\

### CNN Hyperparameters
**cnn_hid**: hidden dimension of CNN

### MLP Hyperparameters
**mlp_hid**: hidden dimension of MLP


## Model Running
After setting the hypaparameters in **model/model.py**, to trian the model, please run the command
```
python model.py
```
The training approximately takes 5 hours for 1000 iterations with a GTX 1080  

Example Output During Training

```
lr 0.01
relation_aware True
context_attn True
rat_b False
output Path: ./Nov27122301
We are using transformer
Adam
  0%|▏                                                                                                                          | 19/10000 [03:34<31:04:20, 11.21s/it]training loss: -0.31551554799079895

 testing 20

APVs mean: 1.241657018661499  std: 0.16542647778987885
SRs mean: 0.05145053192973137  std: 0.024767430499196053
CRs mean: 8.376821517944336  std: 2.5465805530548096
  0%|▏                                                                                                                          | 20/10000 [04:49<83:33:27, 30.14s/it]
```
