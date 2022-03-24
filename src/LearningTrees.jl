module LearningTrees
    using Random
    using DataFrames, CSV, StatsBase

    export
        traintestsplit,
        gini,
        entropy,
        confusionmatrix,
        accuracyscore,
        recallscore,
        precisionscore,
        f1score,

        DecisionTree,
        RandomForest,
        fit!,
        predict

    abstract type Classifier end

    include("utils.jl")
    include("tree.jl")
    include("decisiontree.jl")
    include("randomforest.jl")
end
