k = 30                     #number of date in a batch
l = 5                      #context window size
num_feature = 4
numBatches = 128          #batch size
numStocksInSubset = 11     #num of stocks in a batch 
investmentLength = 40      
numTrainEpisodes = numBatches*10000
tranCostRate = 0.0025

numTestEpisodes = numBatches*5
eval_interval = 10

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

import torch
import torch.optim as optim


from transformer import RATransformer
from baseline import CNN, LSTM, MLP

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

# %%
def generateInputs(df, dates):
    numDates = len(dates)
    prices = df[df["Date"].isin(dates)][["Open", "High", "Low", "Close"]].to_numpy()
    pricesArrays = prices.reshape((numTickers, numDates, 4)) # shape: (numStocks: m, numDates, numFeatures)

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
    assert actions.shape == (numBatches, investmentLength, numStocksInSubset + 1)
    assert ys.shape == actions.shape

    losses = []

    for subsetYs, subsetActions in zip(ys, actions):
        reward = 0
        originalWeights = subsetActions
        inflatedWeights = []
        inflatedValues = []
        updatedWeights = [torch.zeros(len(subsetActions[0])).cuda()]
        for index, currWeights in enumerate(subsetActions):
            inflatedWeights.append(currWeights * subsetYs[index])
            inflatedValues.append(inflatedWeights[-1].sum())
            updatedWeights.append(inflatedWeights[-1] / inflatedValues[-1])

        for index in range(investmentLength):
            tranCost = tranCostRate * abs(originalWeights[index][1:] - updatedWeights[index][1:]).sum()
            reward += torch.log(inflatedValues[index] * (1 - tranCost))
        
        reward /= investmentLength
        reward = reward.unsqueeze(0)

        losses.append(-reward)
    
    #print(losses)
    
    return torch.cat(losses).sum()

# %%
def evaluatePortfolios(ys, actions):
    assert actions.shape == (numBatches, investmentLength, numStocksInSubset+1)
    assert ys.shape == actions.shape

    APVs = []
    SRs = []
    CRs = []

    for subsetYs, subsetActions in zip(ys, actions):
        originalWeights = subsetActions
        inflatedWeights = []
        inflatedValues = []
        updatedWeights = [torch.zeros(len(subsetActions[0])).cuda()]
        aggInflatedValues = [1]
        for index, currWeights in enumerate(subsetActions):
            inflatedWeights.append(currWeights * subsetYs[index])
            inflatedValues.append(inflatedWeights[-1].sum())
            updatedWeights.append(inflatedWeights[-1] / inflatedValues[-1])

        for index in range(investmentLength):
            tranCost = tranCostRate * abs(originalWeights[index][1:] - updatedWeights[index][1:]).sum()
            aggInflatedValues.append(aggInflatedValues[-1] * inflatedValues[index] * (1 - tranCost))
        aggInflatedValues = aggInflatedValues[1:]

        APVs.append(aggInflatedValues[-1].item())
        SRs.append((torch.mean(torch.Tensor(inflatedValues)-1) / torch.std(torch.Tensor(inflatedValues)-1)).item())

        maxAggInflatedValueIndex = 0
        minGainRatio = 1
        for index in range(investmentLength):
            if aggInflatedValues[index] / aggInflatedValues[maxAggInflatedValueIndex] < minGainRatio:
                minGainRatio = aggInflatedValues[index] / aggInflatedValues[maxAggInflatedValueIndex]
            if aggInflatedValues[index] > aggInflatedValues[maxAggInflatedValueIndex]:
                maxAggInflatedValueIndex = index
        CRs.append((aggInflatedValues[-1]/(1 - minGainRatio)).item())

    return APVs, SRs, CRs

# %%
def Evaluation(model):
    APVs = []
    SRs = []
    CRs = []

    for _ in range(int(numTestEpisodes/numBatches)):
        #print(f"\r testing progress {_}/{int(numTestEpisodes/numBatches)}", end='')
        randomStartDate = random.randint(k, len(priceArraysTest[0]) - 1 - investmentLength)
        randomSubsets = [random.sample(range(len(priceArraysTest)), numStocksInSubset) for _ in range(numBatches)] # shape: (numBatches, numStocksInSubset)

        ys = [inflationsTest.T[randomStartDate:randomStartDate+investmentLength].T[randomSubset].T for randomSubset in randomSubsets] # shape: (numBatches, investmentLength, numStocksInSubset)
        ys = torch.Tensor(ys)
        ys = torch.cat([torch.ones(size=(numBatches, investmentLength, 1)), ys], 2) # shape: (numBatches, investmentLength, numStocksInSubset+1)

        actions = [torch.ones(size=(numBatches, numStocksInSubset + 1)).unsqueeze(-1)/(numStocksInSubset + 1)] # shape after for loop: (investmentLength, numBatches, numStocksInSubset+1, 1) average assignment
        for i in range(randomStartDate, randomStartDate + investmentLength):
            encInput = [[priceSeries[i-k:i] for priceSeries in priceArraysTest[randomSubset]] for randomSubset in randomSubsets] # shape: (numBatches, numStocksInSubset+1, priceSeriesLength: k, numFeatures)
            encInput = torch.Tensor(encInput)
            decInput = [[priceSeries[i-l:i] for priceSeries in priceArraysTest[randomSubset]] for randomSubset in randomSubsets] # shape: (numBatches, numStocksInSubset+1, localContextLength: l, numFeatures)
            decInput = torch.Tensor(decInput)
            actions.append(runModel(modelInstance, encInput.cuda(), decInput.cuda(), actions[-1].cuda()))
        actions = torch.stack(actions[1:]).permute([1, 0, 2, 3]).squeeze(-1) # shape: (numBatches, investmentLength, numStocksInSubset+1)

        tempAPVs, tempSRs, tempCRs = evaluatePortfolios(ys.cuda(), actions.cuda())
        APVs += tempAPVs
        SRs += tempSRs
        CRs += tempCRs



    print("\nAPVs mean:", torch.mean(torch.tensor(APVs)).item(), ' std:', torch.std(torch.tensor(APVs)).item())
    print("SRs mean:", torch.mean(torch.tensor(SRs)).item(), ' std:', torch.mean(torch.tensor(SRs)).item())
    print("CRs mean:", torch.mean(torch.tensor(CRs)).item(), ' std:', torch.mean(torch.tensor(CRs)).item())
    
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
    modelInstance = RATransformer(tran_n_layer, k, num_feature, d_model, n_head, l).cuda()
elif MODEL=="CNN":
    modelInstance = CNN(num_feature, cnn_hid).cuda()
elif MODEL=="LSTM":
    modelInstance = LSTM(num_feature, lstm_hid, lstm_layer).cuda()
elif MODEL=="MLP":
    modelInstance = MLP(k, num_feature, mlp_hid).cuda()
else:
    print("invalid model selection")
    
print(f"We are using {MODEL}")
count_parameters(modelInstance)

optimizer = optim.Adam(modelInstance.parameters(),lr=lr, weight_decay=weight_decay)
for batchIndex in tqdm(range(int(numTrainEpisodes/numBatches))):
    #print("\r traning progress " , str(batchIndex) , '/' , str(int(numTrainEpisodes/numBatches)), end='')
    randomStartDate = random.randint(k, len(priceArraysTrain[0]) - 1 - investmentLength)
    randomSubsets = [random.sample(range(len(priceArraysTrain)), numStocksInSubset) for _ in range(numBatches)] # shape: (numBatches, numStocksInSubset)

    ys = [inflationsTrain.T[randomStartDate:randomStartDate+investmentLength].T[randomSubset].T for randomSubset in randomSubsets] # shape: (numBatches, investmentLength, numStocksInSubset)
    ys = torch.Tensor(ys)
    ys = torch.cat([torch.ones(size=(numBatches, investmentLength, 1)), ys], 2) # shape: (numBatches, investmentLength, numStocksInSubset+1)

    actions = [torch.ones(size=(numBatches, numStocksInSubset + 1)).unsqueeze(-1)/(numStocksInSubset + 1)] # shape after for loop: (investmentLength, numBatches, numStocksInSubset+1, 1)  average assignment
    for i in range(randomStartDate, randomStartDate + investmentLength):
        encInput = [[priceSeries[i-k:i] for priceSeries in priceArraysTrain[randomSubset]] for randomSubset in randomSubsets] # shape: (numBatches, numStocksInSubset+1, priceSeriesLength: k, numFeatures)
        encInput = torch.Tensor(encInput)
        decInput = [[priceSeries[i-l:i] for priceSeries in priceArraysTrain[randomSubset]] for randomSubset in randomSubsets] # shape: (numBatches, numStocksInSubset+1, localContextLength: l, numFeatures)
        decInput = torch.Tensor(decInput)
        actions.append(runModel(modelInstance, encInput.cuda(), decInput.cuda(), actions[-1].cuda()))
    actions = torch.stack(actions[1:]).permute([1, 0, 2, 3]).squeeze(-1) # shape: (numBatches, investmentLength, numStocksInSubset+1)

    totalLosses = getTotalLosses(ys.cuda(), actions.cuda())

    optimizer.zero_grad()
    totalLosses.backward()
    optimizer.step()
    
    if (batchIndex+1)%eval_interval==0:
        print(f"\n testing {batchIndex+1}")
        Evaluation(modelInstance)
        

