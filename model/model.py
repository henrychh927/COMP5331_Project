# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
k = 30
l = 5
numBatches = 128
numStocksInSubset = 11
numEpisodes = 1
tranCostRate = 0.0025


# %%
import pandas as pd
import numpy as np
import random

import torch
import torch.optim as optim

from transformer import RATransformer


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
    assert actions.shape == (numBatches, numDates-k-1, numStocksInSubset)
    assert ys.shape == actions.shape

    losses = []

    for subsetYs, subsetActions in zip(ys, actions):
        reward = 0

        originalWeights = subsetActions
        inflatedWeights = []
        updatedWeights = [torch.zeros(len(subsetActions[0]))]
        for index, currWeights in enumerate(subsetActions):
            inflatedWeights.append(currWeights * subsetYs[index])
            updatedWeights.append(inflatedWeights[-1] / (currWeights @ subsetYs[index]))

        for index in range(numDates-k-1):
            tranCost = tranCostRate * abs(originalWeights[index] - updatedWeights[index]).sum()
            reward += torch.log(inflatedWeights[index] * (1 - tranCost))
        
        reward /= numDates-k-1

        losses.append(-reward)
    
    return torch.cat(losses).sum()


# %%
def runModel(modelInstance, encInput, decInput, prevAction):
    assert encInput.shape == (numBatches, numStocksInSubset, k, 4)
    assert decInput.shape == (numBatches, numStocksInSubset, l, 4)
    assert prevAction.shape == (numBatches, numStocksInSubset, 1)
    # return torch.zeros(size=(numBatches, numStocksInSubset)).unsqueeze(-1)
    return modelInstance.forward(encInput, decInput, prevAction)


# %%
modelInstance = RATransformer(1, k, 4, 12, 2, l)
optimizer = optim.Adam(modelInstance.parameters(),lr=1e-2)
for _ in range(numEpisodes):
    randomSubsets = [random.sample(range(numTickers), numStocksInSubset) for _ in range(numBatches)] # shape: (numBatches, numStocksInSubset)
    ys = [inflations[k:].T[randomSubset].T for randomSubset in randomSubsets] # shape: (numBatches, numDates-k-1: T-k-1, numStocksInSubset)
    actions = [torch.zeros(size=(numBatches, numStocksInSubset)).unsqueeze(-1)] # shape after for loop: (numDates-k-1: T-k-1, numBatches, numStocksInSubset, 1)
    for i in range(k, numDates - 1):
        encInput = [[priceSeries[i-k:i] for priceSeries in entryArrays[randomSubset]] for randomSubset in randomSubsets] # shape: (numBatches, numStocksInSubset, priceSeriesLength: k, numFeatures)
        encInput = torch.Tensor(encInput)
        decInput = [[priceSeries[i-l:i] for priceSeries in entryArrays[randomSubset]] for randomSubset in randomSubsets] # shape: (numBatches, numStocksInSubset, localContextLength: l, numFeatures)
        decInput = torch.Tensor(decInput)
        actions.append(runModel(modelInstance, encInput, decInput, actions[-1]))

    actions = torch.stack(actions[1:]).permute([1, 0, 2, 3]).squeeze(-1) # shape: (numBatches, numDates-k-1: T-k-1, numStocksInSubset)
    ys = torch.Tensor(ys)
    totalLosses = getTotalLosses(ys, actions)

    optimizer.zero_grad()
    totalLosses.backward()
    optimizer.step()


# %%



