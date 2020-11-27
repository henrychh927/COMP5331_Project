k = 30                     #number of date in a batch
l = 5                      #context window size
num_feature = 4
numBatches = 128          #batch size
numStocksInSubset = 11     #num of stocks in a batch 
trainInvestmentLength = 40      
numTrainEpisodes = numBatches*10000
tranCostRate = 0.0025

testInvestmentLength = 466 - k - 1 # len(testDates) - k - 1
numTestEpisodes = numBatches*1
eval_interval = 20

lr = 1e-2
weight_decay=1e-7

# selection of model 
MODEL = "transformer"             #LSTM, CNN, MLP, transformer  
context_attn = True
relation_aware = True
rat_b = False

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



print('lr', lr)
print('relation_aware', relation_aware)
print('context_attn', context_attn)
print('rat_b', rat_b)


# %%
import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
import time


import torch
import torch.optim as optim
 

from transformer import RATransformer
from baseline import CNN, LSTM, MLP
from torch_optimizer import RAdam

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# %%
df = pd.read_csv("whole_selected.csv")
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


save_folder = './' + str(time.strftime('%b%d%H%M%S', time.localtime()))
os.makedirs(save_folder)
print('output Path:', save_folder)
save_file = save_folder + '/output.txt'
config_file = save_folder + '/config.txt'


with open(config_file, 'a') as fw:
           print('k,l,num_feature,numBatches,numStocksInSubset,trainInvestmentLength,numTrainEpisodes,tranCostRate,numTestEpisodes,eval_interval,lr, weight_decay,MODEL,tran_n_layer,n_head,d_model,lstm_hid ,lstm_layer ,cnn_hid ,mlp_hid, context_attn, relation_aware, rat_b \n',
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
              mlp_hid ,
              context_attn ,
                relation_aware,
                 rat_b,
                 file=fw)
            
            

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
        reward = 0
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
            reward += torch.log(inflatedValues[index] * (1 - tranCost))
        
        reward /= trainInvestmentLength
        reward = reward.unsqueeze(0)

        losses.append(-reward)
    
    #print(losses)
    
    return torch.cat(losses).sum()

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
def Evaluation(model, epoch):
    APVs = []
    SRs = []
    CRs = []
    model.eval()
    with torch.no_grad():
        for _ in range(int(numTestEpisodes/numBatches)):
            #print(f"\r testing progress {_}/{int(numTestEpisodes/numBatches)}", end='')
            startDate = k
            randomSubsets = [random.sample(range(len(priceArraysTest)), numStocksInSubset) for _ in range(numBatches)] # shape: (numBatches, numStocksInSubset)

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


        print("\nAPVs mean:", torch.mean(torch.tensor(APVs)).item(), ' std:', torch.std(torch.tensor(APVs)).item())
        print("SRs mean:", torch.mean(torch.tensor(SRs)).item(), ' std:', torch.std(torch.tensor(SRs)).item())
        print("CRs mean:", torch.mean(torch.tensor(CRs)).item(), ' std:', torch.std(torch.tensor(CRs)).item())
        
        with open(save_file, 'a') as fw:
            print(epoch, torch.mean(torch.tensor(APVs)).item(),
                  torch.std(torch.tensor(APVs)).item(), 
                  torch.mean(torch.tensor(SRs)).item(),
                  torch.std(torch.tensor(SRs)).item(),
                  torch.mean(torch.tensor(CRs)).item(), 
                  torch.std(torch.tensor(CRs)).item(), file=fw)

    
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
    modelInstance = RATransformer(tran_n_layer, k, num_feature, d_model, n_head, l, 
                                 context_attn=context_attn, rat_b=rat_b,
                                  relation_aware=relation_aware).to(device)
elif MODEL=="CNN":
    modelInstance = CNN(num_feature, cnn_hid).to(device)
elif MODEL=="LSTM":
    modelInstance = LSTM(num_feature, lstm_hid, lstm_layer).to(device)
elif MODEL=="MLP":
    modelInstance = MLP(k, num_feature, mlp_hid).to(device)
else:
    print("invalid model selection")
    
print(f"We are using {MODEL}")
# count_parameters(modelInstance)

# print('RAdam')
# optimizer = RAdam(modelInstance.parameters(), lr=lr, weight_decay=weight_decay)

print('Adam')
optimizer = optim.Adam(modelInstance.parameters(), lr=lr, weight_decay=weight_decay)


    
    
    
for batchIndex in tqdm(range(int(numTrainEpisodes/numBatches))):
    modelInstance.train()
    #print("\r traning progress " , str(batchIndex) , '/' , str(int(numTrainEpisodes/numBatches)), end='')
    randomStartDate = random.randint(k, len(priceArraysTrain[0]) - 1 - trainInvestmentLength)
    randomSubsets = [random.sample(range(len(priceArraysTrain)), numStocksInSubset) for _ in range(numBatches)] # shape: (numBatches, numStocksInSubset)

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

    totalLosses = getTotalLosses(ys.to(device), actions.to(device))

    optimizer.zero_grad()
    totalLosses.backward()
    torch.nn.utils.clip_grad_norm_(modelInstance.parameters(), 1.0)
    optimizer.step()
    
    if (batchIndex+1)%eval_interval==0:
        print(f"training loss: {totalLosses}")
        print(f"\n testing {batchIndex+1}")
        
        
        Evaluation(modelInstance, batchIndex+1)
        

