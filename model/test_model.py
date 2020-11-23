bTrain = False
drive = '/content/drive/MyDrive/Colab Notebooks/'
model_dir = drive+'/transformer/Nov21062056/model_bestEval.pkl'#drive+'transformer/model_1 (2).pkl'
checkpoint_dir = ''#'/content/drive/MyDrive/Colab Notebooks/transformer/Nov21053944/model_1_checkpoint'
k = 30                     #number of date in a batch
l = 5                      #context window size
num_feature = 4
numBatches = 2#128          #batch size
numStocksInSubset = 422     #num of stocks in a batch 
trainInvestmentLength = 30      
numTrainEpisodes = numBatches*10000
tranCostRate = 0.0025

testInvestmentLength = 466-k-1#184 - k - 1 # len(testDates) - k - 1
numTestEpisodes = numBatches*1
eval_interval = 1

lr = 1e-3
weight_decay=1e-7

# selection of model 
MODEL = "transformer"             #LSTM, CNN, MLP, transformer  

# transformer architecture
tran_n_layer = 1
n_head = 2
d_model = 12

# lstm archi
lstm_hid = 50
lstm_layer = 2

# cnn archi
cnn_hid = 50

# mlp
mlp_hid = 50



# %%
import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
import time


import torch
import torch.optim as optim

import sys
sys.path.append(drive)
from transformer import RATransformer
from baseline import CNN, LSTM, MLP

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
df = pd.read_csv(drive+'whole_selected.csv')
df = df[["Ticker", "Date", "Open", "High", "Low", "Close"]].sort_values(by=["Ticker", "Date"])

# %%
numTickers = len(df["Ticker"].unique()) # m
# print("Number of tickers: " + str(numTickers))

datesValueCounts = df["Date"].value_counts()
validDates = datesValueCounts.loc[datesValueCounts == max(datesValueCounts)].index
validDates = list(validDates.sort_values())
# print("Number of valid dates: " + str(len(validDates)))

# %%
trainDates = [date for date in validDates if date >= "2013" and date < "2017"]
testDates = [date for date in validDates if date >= "2017" and date < "2019"]
# print("Number of dates for training: " + str(len(trainDates)))
# print("Number of dates for testing: " + str(len(testDates)))


save_folder = drive + MODEL + '/' + str(time.strftime('%b%d%H%M%S', time.localtime()))
os.makedirs(save_folder)
print('output Path:', save_folder)
save_file = save_folder + '/output.txt'
config_file = save_folder + '/config.txt'


with open(config_file, 'a') as fw:
           print('k,l,num_feature,numBatches,numStocksInSubset,trainInvestmentLength,numTrainEpisodes,tranCostRate,numTestEpisodes,eval_interval,lr, weight_decay,MODEL,tran_n_layer,n_head,d_model,lstm_hid ,lstm_layer ,cnn_hid ,mlp_hid \n',
              k,
              l ,
              num_feature,
              numBatches,
              numStocksInSubset,
              trainInvestmentLength,
              numTrainEpisodes,
              tranCostRate,
              numTestEpisodes,
              eval_interval,
              lr,
              weight_decay,
              MODEL,
              tran_n_layer,
              n_head,
              d_model,
              lstm_hid ,
              lstm_layer ,
              cnn_hid ,
              mlp_hid , file=fw)
            
            

# %%
def generateInputs(df, dates):
    numDates = len(dates)
    prices = df[df["Date"].isin(dates)][["Open", "High", "Low", "Close"]].to_numpy()
    pricesArrays = prices.reshape((numTickers, numDates, 4)) # shape: (numStocks: m, numDates, numFeatures)

    pricesArrays = pricesArrays / pricesArrays[0]

    pricesArraysTransposed = pricesArrays.T # shape: (numFeatures, numDates, numStocks: m)
    pricesArraysClosingPrices = pricesArraysTransposed[3] # shape: (numDates, numStocks: m)
    inflations = np.array([pricesArraysClosingPrices[i + 1] / pricesArraysClosingPrices[i] for i in range(len(pricesArraysClosingPrices) - 1)]).T # shape: (numStocks: m, numDates-1)
    
    return pricesArrays, inflations

# %%
priceArraysTrain, inflationsTrain = generateInputs(df, trainDates)
priceArraysTest, inflationsTest = generateInputs(df, testDates)
assert priceArraysTrain.shape == (numTickers, len(trainDates), 4)
assert inflationsTrain.shape == (numTickers, len(trainDates)-1)
assert priceArraysTest.shape == (numTickers, len(testDates), 4)
assert inflationsTest.shape == (numTickers, len(testDates)-1)

# %%
def getTotalLosses(ys, actions):
    assert actions.shape == (numBatches, trainInvestmentLength, numStocksInSubset + 1)
    assert ys.shape == actions.shape

    losses = []

    for subsetYs, subsetActions in zip(ys, actions):
        reward = 1
        originalWeights = subsetActions
        inflatedWeights = []
        inflatedValues = []
        updatedWeights = [torch.zeros(len(subsetActions[0])).to(device)]
        for index, currWeights in enumerate(subsetActions):
            inflatedWeights.append(currWeights * subsetYs[index])
            inflatedValues.append(inflatedWeights[-1].sum())
            updatedWeights.append(inflatedWeights[-1] / inflatedValues[-1])

        for index in range(trainInvestmentLength):
            tranCost = tranCostRate * abs(originalWeights[index][1:] - updatedWeights[index][1:]).sum()
            #reward += torch.log(inflatedValues[index] * (1 - tranCost))
            reward = reward * inflatedValues[index] * (1 - tranCost)
        
        #reward /= trainInvestmentLength
        reward = reward.unsqueeze(0)

        losses.append(-torch.log(reward))
    
    #print(losses)
    
    return torch.cat(losses).sum(), reward[0]

# %%
def evaluatePortfolios(ys, actions):
    assert actions.shape == (numBatches, testInvestmentLength, numStocksInSubset+1)
    assert ys.shape == actions.shape

    APVs = []
    SRs = []
    CRs = []

    for subsetYs, subsetActions in zip(ys, actions):
        originalWeights = subsetActions
        inflatedWeights = []
        inflatedValues = []
        updatedWeights = [torch.zeros(len(subsetActions[0])).to(device)]
        aggInflatedValues = [1]
        for index, currWeights in enumerate(subsetActions):
            inflatedWeights.append(currWeights * subsetYs[index])
            inflatedValues.append(inflatedWeights[-1].sum())
            updatedWeights.append(inflatedWeights[-1] / inflatedValues[-1])

        for index in range(testInvestmentLength):
            tranCost = tranCostRate * abs(originalWeights[index][1:] - updatedWeights[index][1:]).sum()
            aggInflatedValues.append(aggInflatedValues[-1] * inflatedValues[index] * (1 - tranCost))
        aggInflatedValues = aggInflatedValues[1:]

        APVs.append(aggInflatedValues[-1].item())
        SRs.append((torch.mean(torch.Tensor(inflatedValues)-1) / torch.std(torch.Tensor(inflatedValues)-1)).item())

        maxAggInflatedValueIndex = 0
        minGainRatio = 1
        for index in range(testInvestmentLength):
            if aggInflatedValues[index] / aggInflatedValues[maxAggInflatedValueIndex] < minGainRatio:
                minGainRatio = aggInflatedValues[index] / aggInflatedValues[maxAggInflatedValueIndex]
            if aggInflatedValues[index] > aggInflatedValues[maxAggInflatedValueIndex]:
                maxAggInflatedValueIndex = index
        CRs.append((aggInflatedValues[-1]/(1 - minGainRatio)).item())

    return APVs, SRs, CRs

# %%
def Evaluation(model, epoch, maxPortVal):
    APVs = []
    SRs = []
    CRs = []
    model.eval()
    with torch.no_grad():
        #for _ in range(int(numTestEpisodes/numBatches)):
            #print(f"\r testing progress {_}/{int(numTestEpisodes/numBatches)}", end='')
        startDate = k
        randomSubsets = [[i for i in range(len(priceArraysTest))] for j in range(numBatches)]#= [random.sample(range(len(priceArraysTest)), numStocksInSubset) for _ in range(numBatches)] # shape: (numBatches, numStocksInSubset)

        ys = [inflationsTest.T[startDate:startDate+testInvestmentLength].T[randomSubset].T for randomSubset in randomSubsets] # shape: (numBatches, testInvestmentLength, numStocksInSubset)
        ys = torch.Tensor(ys)
        ys = torch.cat([torch.ones(size=(numBatches, testInvestmentLength, 1)), ys], 2) # shape: (numBatches, testInvestmentLength, numStocksInSubset+1)

        actions = [torch.ones(size=(numBatches, numStocksInSubset + 1)).unsqueeze(-1)/(numStocksInSubset + 1)] # shape after for loop: (testInvestmentLength, numBatches, numStocksInSubset+1, 1) average assignment
        for i in range(startDate, startDate + testInvestmentLength):
            encInput = [[priceSeries[i-k:i] for priceSeries in priceArraysTest[randomSubset]] for randomSubset in randomSubsets] # shape: (numBatches, numStocksInSubset+1, priceSeriesLength: k, numFeatures)
            encInput = torch.Tensor(encInput)
            decInput = [[priceSeries[i-l:i] for priceSeries in priceArraysTest[randomSubset]] for randomSubset in randomSubsets] # shape: (numBatches, numStocksInSubset+1, localContextLength: l, numFeatures)
            decInput = torch.Tensor(decInput)
            actions.append(runModel(modelInstance, encInput.to(device), decInput.to(device), actions[-1].to(device)))
        actions = torch.stack(actions[1:]).permute([1, 0, 2, 3]).squeeze(-1) # shape: (numBatches, testInvestmentLength, numStocksInSubset+1)

        tempAPVs, tempSRs, tempCRs = evaluatePortfolios(ys.to(device), actions.to(device))
        APVs += tempAPVs
        SRs += tempSRs
        CRs += tempCRs


        tmp = torch.mean(torch.tensor(APVs)).item()
        print("\nAPVs mean:", torch.mean(torch.tensor(APVs)).item(), ' std:', torch.std(torch.tensor(APVs)).item())
        print("SRs mean:", torch.mean(torch.tensor(SRs)).item(), ' std:', torch.std(torch.tensor(SRs)).item())
        print("CRs mean:", torch.mean(torch.tensor(CRs)).item(), ' std:', torch.std(torch.tensor(CRs)).item())
        
        if tmp > maxPortVal:
            maxPortVal = tmp 
            torch.save(modelInstance,save_folder+'/model_bestEval.pkl')
            print('model saved: '+save_folder+'/model_bestEval.pkl')
            print('max port val: '+str(maxPortVal))

        with open(save_file, 'a') as fw:
            print(epoch, torch.mean(torch.tensor(APVs)).item(),
                  torch.std(torch.tensor(APVs)).item(), 
                  torch.mean(torch.tensor(SRs)).item(),
                  torch.std(torch.tensor(SRs)).item(),
                  torch.mean(torch.tensor(CRs)).item(), 
                  torch.std(torch.tensor(CRs)).item(), file=fw)
        return maxPortVal

    
# %%
def runModel(modelInstance, encInput, decInput, prevAction, model=MODEL):
    assert encInput.shape == (numBatches, numStocksInSubset, k, 4)
    assert decInput.shape == (numBatches, numStocksInSubset, l, 4)
    assert prevAction.shape == (numBatches, numStocksInSubset + 1, 1)
    # return torch.ones(size=(numBatches, numStocksInSubset + 1), requires_grad=True).unsqueeze(-1)/(numStocksInSubset + 1)
    
    if model=="transformer":
        return modelInstance.forward(encInput, decInput, prevAction)
    else:
        return modelInstance.forward(encInput)

# %%
def count_parameters(model):
    temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model architecture:\n\n', model)
    print(f'\nThe model has {temp:,} trainable parameters')

# %%
if MODEL=="transformer":
    modelInstance = RATransformer(tran_n_layer, k, num_feature, d_model, n_head, l).to(device)
elif MODEL=="CNN":
    modelInstance = CNN(num_feature, cnn_hid).to(device)
elif MODEL=="LSTM":
    modelInstance = LSTM(num_feature, lstm_hid, lstm_layer).to(device)
elif MODEL=="MLP":
    modelInstance = MLP(k, num_feature, mlp_hid).to(device)
else:
    print("invalid model selection")
    
print(f"We are using {MODEL}")
count_parameters(modelInstance)

optimizer = optim.Adam(modelInstance.parameters(),lr=lr, weight_decay=weight_decay)


batchStart = 0
if (bTrain):
  maxPortVal = 1.0
  if checkpoint_dir != '':
      checkpoint = torch.load(checkpoint_dir)
      modelInstance.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      batchStart = checkpoint['batchIndex']
      print('checkpoint loaded: '+checkpoint_dir)
      print(batchStart)
  minLoss = 0    
        
  for batchIndex in tqdm(range(batchStart,int(numTrainEpisodes/numBatches))):
      modelInstance.train()
      #print("\r traning progress " , str(batchIndex) , '/' , str(int(numTrainEpisodes/numBatches)), end='')
      randomStartDate = random.randint(k, len(priceArraysTrain[0]) - 1 - trainInvestmentLength)
      randomSubsets = [[i for i in range(len(priceArraysTest))] for j in range(numBatches)]#randomSubsets = [random.sample(range(len(priceArraysTrain)), numStocksInSubset) for _ in range(numBatches)] # shape: (numBatches, numStocksInSubset)

      ys = [inflationsTrain.T[randomStartDate:randomStartDate+trainInvestmentLength].T[randomSubset].T for randomSubset in randomSubsets] # shape: (numBatches, trainInvestmentLength, numStocksInSubset)
      ys = torch.Tensor(ys)
      ys = torch.cat([torch.ones(size=(numBatches, trainInvestmentLength, 1)), ys], 2) # shape: (numBatches, trainInvestmentLength, numStocksInSubset+1)

      actions = [torch.ones(size=(numBatches, numStocksInSubset + 1)).unsqueeze(-1)/(numStocksInSubset + 1)] # shape after for loop: (trainInvestmentLength, numBatches, numStocksInSubset+1, 1)  average assignment
      for i in range(randomStartDate, randomStartDate + trainInvestmentLength):
          encInput = [[priceSeries[i-k:i] for priceSeries in priceArraysTrain[randomSubset]] for randomSubset in randomSubsets] # shape: (numBatches, numStocksInSubset+1, priceSeriesLength: k, numFeatures)
          encInput = torch.Tensor(encInput)
          decInput = [[priceSeries[i-l:i] for priceSeries in priceArraysTrain[randomSubset]] for randomSubset in randomSubsets] # shape: (numBatches, numStocksInSubset+1, localContextLength: l, numFeatures)
          decInput = torch.Tensor(decInput)
          actions.append(runModel(modelInstance, encInput.to(device), decInput.to(device), actions[-1].to(device)))
      actions = torch.stack(actions[1:]).permute([1, 0, 2, 3]).squeeze(-1) # shape: (numBatches, trainInvestmentLength, numStocksInSubset+1)

      totalLosses, portVal = getTotalLosses(ys.to(device), actions.to(device))
      print(f"training loss: {totalLosses}")
      print(f"portfolio value: {portVal}")

      if totalLosses < minLoss:
          minLoss = totalLosses
          torch.save(modelInstance,save_folder+'/model_1.pkl')
          print('model saved: '+save_folder+'/model_1.pkl')
          checkpoint = {'batchIndex': batchIndex+1, 'state_dict': modelInstance.state_dict(),'optimizer': optimizer.state_dict()}
          torch.save(checkpoint,save_folder+'/model_1_checkpoint')
          print('checkpoint saved: '+save_folder+'/model_1_checkpoint')

      optimizer.zero_grad()
      totalLosses.backward()
      optimizer.step()
      if (batchIndex+1)%eval_interval==0:
        print(f"\n testing {batchIndex+1}")
        maxPortVal = Evaluation(modelInstance, batchIndex+1,maxPortVal)
else:
    modelInstance=torch.load(model_dir)
    Evaluation(modelInstance, 1000000,1000000)
