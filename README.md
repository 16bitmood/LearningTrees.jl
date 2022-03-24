# LearningTrees.jl

Simple Julia implementation of Decision Trees, Random Forest, etc.

## Usage
```julia
using DataFrames
using CSV

using LearningTrees

df = CSV.File("data.csv") |> DataFrame
topredict = :target

dftrain, dftest = traintestsplit(df, 0.33)

clf = DecisionTree(;maxdepth = 15, criterion = gini)
fit!(clf, dftrain, topredict)
predict(clf, dftest)

clf = RandomForest(;ntrees = 20, criterion = entropy)
fit!(clf, dftrain, topredict)
predict(clf, dftest)
```