using Pkg.Artifacts
using Random
using Test

using DataFrames
using CSV

using LearningTrees

RNGSEED = 53
Random.seed!(RNGSEED)

loadiris() = begin
    artifact_toml = joinpath(@__DIR__, "Artifacts.toml")
    iris_hash = artifact_hash("iris", artifact_toml)
    if iris_hash === nothing || !artifact_exists(iris_hash)
        iris_hash = create_artifact() do artifact_dir
            iris_url_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris"
            download("$(iris_url_base)/iris.data", joinpath(artifact_dir, "iris.data"))
            download("$(iris_url_base)/bezdekIris.data", joinpath(artifact_dir, "bezdekIris.csv"))
            download("$(iris_url_base)/iris.names", joinpath(artifact_dir, "iris.names"))
        end
        bind_artifact!(artifact_toml, "iris", iris_hash)
    end
    artifact_path(iris_hash)
end

testiris() = begin
    datasetpath = loadiris()
    names = [:sepal_length, :sepal_width, :petal_length, :petal_width, :class]
    df = CSV.File(joinpath(datasetpath, "iris.data"); header = names) |> DataFrame
    topredict = :class

    dftrain, dftest = traintestsplit(df, 0.33)

    c = true
    for i in 2:(length(propertynames(df))-1)
        clf = DecisionTree(;maxdepth = i, criterion = gini)
        fit!(clf, dftrain, topredict)
        score = accuracyscore(dftest[:, topredict], predict(clf, dftest))
        c = c && score >= 0.9
    end

    for i in [2,4,8,16,32,64]
        clf = RandomForest(;ntrees = i, criterion = gini)
        fit!(clf, dftrain, topredict)
        score = accuracyscore(dftest[:, topredict], predict(clf, dftest))
        c = c && score >= 0.9
    end

    c
end

@test testiris()