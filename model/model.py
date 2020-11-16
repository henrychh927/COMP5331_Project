# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
k = 30
l = 5
numBatches = 60
numStocksInSubset = 11
investmentLength = 60
numTrainEpisodes = 1024
tranCostRate = 0.0025

testPerc = 0.2
numTestEpisodes = 256
eval_interval = 10



# %%
import pandas as pd
import numpy as np
import random

import torch
import torch.optim as optim
import os

from transformer import RATransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
df = pd.read_csv("whole_selected.csv")
print(df)


# %%
df = df[["Ticker", "Date", "Open", "High", "Low", "Close"]]


# %%
tickers = df["Ticker"].unique()
numTickers = len(tickers)
print("Number of tickers: " + str(numTickers))
tickersDict = {}
for index, ticker in enumerate(tickers):
    tickersDict[ticker] = index

df["Ticker"] = df["Ticker"].apply(lambda ticker: tickersDict[ticker])
print(df)


# %%
datesValueCounts = df["Date"].value_counts()
validDates = datesValueCounts.loc[datesValueCounts == max(datesValueCounts)].index
validDates = list(validDates.sort_values())
print("Number of valid dates: " + str(len(validDates)))


# %%
print(validDates[:100])
validDates = validDates[5:]


# %%
df = df[df["Date"].isin(validDates)]


# %%
dates = df["Date"].unique()
numDates = len(dates)
print("Number of valid dates: " + str(numDates))
datesDict = {}
for index, date in enumerate(dates):
    datesDict[date] = index

df["Date"] = df["Date"].apply(lambda date: datesDict[date])
print(df)


# %%
df = df.sort_values(by=["Ticker", "Date"])
print(df)


# %%
entries = df[["Open", "High", "Low", "Close"]].to_numpy()
entryArrays = entries.reshape((numTickers, numDates, 4)) # shape: (numStocks: m, numDates: T, numFeatures)
print(entryArrays)


# %%
entryArraysTransposed = entryArrays.T # shape: (numFeatures, numDates: T, numStocks: m)
entryArraysClosingPrices = entryArraysTransposed[3] # shape: (numDates: T, numStocks: m)
inflations = np.array([entryArraysClosingPrices[i + 1] / entryArraysClosingPrices[i] for i in range(len(entryArraysClosingPrices) - 1)]) # shape: (numDates-1: T-1, numStocks: m)
print(inflations) # percentage change from period i to (i+1)


# %%
def getTotalLosses(ys, actions):
    assert actions.shape == (numBatches, investmentLength, numStocksInSubset)
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
            tranCost = tranCostRate * abs(originalWeights[index] - updatedWeights[index]).sum()
            reward += torch.log(inflatedValues[index] * (1 - tranCost))
        
        reward /= investmentLength
        reward = reward.unsqueeze(0)

        losses.append(-reward)
    
    #print(losses)
    
    return torch.cat(losses).sum()

def evaluatePortfolios(ys, actions):
    assert actions.shape == (numBatches, investmentLength, numStocksInSubset)
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
            tranCost = tranCostRate * abs(originalWeights[index] - updatedWeights[index]).sum()
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


def Evaluation(model, entryArraysTest):
    APVs = []
    SRs = []
    CRs = []

    for _ in range(int(numTestEpisodes/numBatches)):
        print(f"\r testing progress {_}/{int(numTestEpisodes/numBatches)}", end='')
        randomStartDate = random.randint(k, numDates - 1 - investmentLength)
        randomSubsets = [random.sample(range(len(entryArraysTest)), numStocksInSubset) for _ in range(numBatches)] # shape: (numBatches, numStocksInSubset)

        ys = [inflations[randomStartDate:randomStartDate+investmentLength].T[randomSubset].T for randomSubset in randomSubsets] # shape: (numBatches, investmentLength, numStocksInSubset)
        actions = [torch.zeros(size=(numBatches, numStocksInSubset)).unsqueeze(-1)] # shape after for loop: (investmentLength, numBatches, numStocksInSubset, 1)

        for i in range(randomStartDate, randomStartDate + investmentLength):
            encInput = [[priceSeries[i-k:i] for priceSeries in entryArraysTest[randomSubset]] for randomSubset in randomSubsets] # shape: (numBatches, numStocksInSubset, priceSeriesLength: k, numFeatures)
            encInput = torch.Tensor(encInput)
            decInput = [[priceSeries[i-l:i] for priceSeries in entryArraysTest[randomSubset]] for randomSubset in randomSubsets] # shape: (numBatches, numStocksInSubset, localContextLength: l, numFeatures)
            decInput = torch.Tensor(decInput)
            actions.append(runModel(modelInstance, encInput.cuda(), decInput.cuda(), actions[-1].cuda()))

        actions = torch.stack(actions[1:]).permute([1, 0, 2, 3]).squeeze(-1) # shape: (numBatches, investmentLength, numStocksInSubset)
        ys = torch.Tensor(ys)
        tempAPVs, tempSRs, tempCRs = evaluatePortfolios(ys.cuda(), actions.cuda())
        APVs += tempAPVs
        SRs += tempSRs
        CRs += tempCRs


    # %%

    print("APVs mean:", torch.mean(torch.tensor(APVs)), ' std:', torch.std(torch.tensor(APVs)))
    print("SRs mean:", torch.mean(torch.tensor(SRs)), ' std:', torch.mean(torch.tensor(SRs)))
    print("CRs mean:", torch.mean(torch.tensor(CRs)), ' std:', torch.mean(torch.tensor(CRs)))
    
# %%
def runModel(modelInstance, encInput, decInput, prevAction):
    assert encInput.shape == (numBatches, numStocksInSubset, k, 4)
    assert decInput.shape == (numBatches, numStocksInSubset, l, 4)
    assert prevAction.shape == (numBatches, numStocksInSubset, 1)
    # return torch.ones(size=(numBatches, numStocksInSubset), requires_grad=True).unsqueeze(-1)/numStocksInSubset
    return modelInstance.forward(encInput, decInput, prevAction)


# %%
# Train-test split
np.random.shuffle(entryArrays)
entryArraysTrain = entryArrays[:int(testPerc * numTickers)]
entryArraysTest = entryArrays[int(testPerc * numTickers):]

# %%
modelInstance = RATransformer(1, k, 4, 12, 2, l).cuda()
optimizer = optim.Adam(modelInstance.parameters(),lr=1e-2)
for _ in range(int(numTrainEpisodes/numBatches)):
    print("\r traning progress " , str(_) , '/' , str(int(numTrainEpisodes/numBatches)), end='')
    randomStartDate = random.randint(k, numDates - 1 - investmentLength)
    randomSubsets = [random.sample(range(len(entryArraysTrain)), numStocksInSubset) for _ in range(numBatches)] # shape: (numBatches, numStocksInSubset)

    ys = [inflations[randomStartDate:randomStartDate+investmentLength].T[randomSubset].T for randomSubset in randomSubsets] # shape: (numBatches, investmentLength, numStocksInSubset)
    actions = [torch.zeros(size=(numBatches, numStocksInSubset)).unsqueeze(-1)] # shape after for loop: (investmentLength, numBatches, numStocksInSubset, 1)

    for i in range(randomStartDate, randomStartDate + investmentLength):
        encInput = [[priceSeries[i-k:i] for priceSeries in entryArraysTrain[randomSubset]] for randomSubset in randomSubsets] # shape: (numBatches, numStocksInSubset, priceSeriesLength: k, numFeatures)
        encInput = torch.Tensor(encInput)
        decInput = [[priceSeries[i-l:i] for priceSeries in entryArraysTrain[randomSubset]] for randomSubset in randomSubsets] # shape: (numBatches, numStocksInSubset, localContextLength: l, numFeatures)
        decInput = torch.Tensor(decInput)
        actions.append(runModel(modelInstance, encInput.cuda(), decInput.cuda(), actions[-1].cuda()))

    actions = torch.stack(actions[1:]).permute([1, 0, 2, 3]).squeeze(-1) # shape: (numBatches, investmentLength, numStocksInSubset)
    ys = torch.Tensor(ys)
    totalLosses = getTotalLosses(ys.cuda(), actions.cuda())

    optimizer.zero_grad()
    totalLosses.backward()
    optimizer.step()
    
    if _%eval_interval==0:
        print("testing")
        Evaluation(modelInstance, entryArraysTest)
        


# %%



# %%



# %%



