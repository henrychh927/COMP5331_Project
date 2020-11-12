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
df


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
df


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
df


# %%
df = df.sort_values(by=["Ticker", "Date"])
df


# %%
entries = df[["Open", "High", "Low", "Close"]].to_numpy()
entryArrays = entries.reshape((numTickers, numDates, 4)) # shape: (numStocks: m, numDates: T, numFeatures)
entryArrays


# %%
entryArraysTransposed = entryArrays.T # shape: (numFeatures, numDates: T, numStocks: m)
entryArraysClosingPrices = entryArraysTransposed[3] # shape: (numDates: T, numStocks: m)
inflations = np.array([entryArraysClosingPrices[i + 1] / entryArraysClosingPrices[i] for i in range(len(entryArraysClosingPrices) - 1)]) # shape: (numDates-1: T-1, numStocks: m)
inflations # percentage change from period i to (i+1)


# %%
def getTotalReward(ys, actions):
    reward = 0
    n = len(actions)

    assert len(ys) == n
    assert len(actions[0]) == numStocksInSubset
    assert len(ys) == numStocksInSubset

    originalWeights = actions
    inflatedWeights = []
    updatedWeights = [np.zeros(len(actions[0]))]
    for index, currWeights in enumerate(actions):
        inflatedWeights.append(currWeights * ys[index])
        updatedWeights.append(inflatedWeights[-1] / (currWeights @ ys[index]))

    for index in range(n):
        tranCost = tranCostRate * abs(originalWeights[index] - updatedWeights[index]).sum()
        reward += np.log(inflatedWeights[index] * (1 - tranCost))
    
    reward /= n

    return reward


# %%
def runModel(modelInstance, encInput, decInput, prevAction):
    return modelInstance.forward(encInput, decInput, prevAction)


# %%
modelInstance = RATransformer(1, k, 4, 12, 2, l)
optimizer = optim.Adam(modelInstance.parameters(),lr=1e-2)
for _ in range(numEpisodes):
    randomSubsets = [random.sample(range(numTickers), numStocksInSubset) for _ in range(numBatches)]
    ys = [stockInflations[k:] for stockInflations in inflations[randomSubsets]] # shape: (numDates-1: T-1, numStocks: m)
    actions = [np.zeros(4)]
    for i in range(numDates - k):
        print(i)
        encInput = [priceSeries[i:i+k] for priceSeries in entryArrays[randomSubsets]] # shape: (numBatches, numStocksInSubset, priceSeriesLength: k, numFeatures)
        encInput = torch.Tensor(encInput)
        decInput = [priceSeries[i+k-l:i+k] for priceSeries in entryArrays[randomSubsets]] # shape: (numBatches, numStocksInSubset, localContextLength: l, numFeatures)
        decInput = torch.Tensor(decInput)
        actions.append(runModel(modelInstance, encInput, decInput, actions[-1]))
    totalReward = getTotalReward(ys, actions)

    optimizer.zero_grad()
    totalReward.backward()
    optimizer.step()


# %%



