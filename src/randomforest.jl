mutable struct RandomForest <: Classifier
    trees     :: Union{Nothing, Vector{DecisionTree}}
    topredict
    ntrees    :: UInt
    criterion
    attrfilter
end

RandomForest(;ntrees = 20, criterion = gini) = begin
    RandomForest(nothing, nothing, ntrees, criterion, nothing)
end

fit!(rf::RandomForest, df::DataFrame, topredict) = begin
    rf.topredict = topredict

    dfrows = [sample(1:nrow(df), nrow(df), replace = true) for i = 1:rf.ntrees]
    dfs = map(rows -> df[rows, :], dfrows)

    m = Int(ceil(1 + sqrt(size(df, 2) - 1)))
    rf.attrfilter = attrs -> sample(attrs, min(m, length(attrs)), replace = false)

    trees = Dict()
    Threads.@threads for i = 1:rf.ntrees
        id = Threads.threadid()
        t = DecisionTree(
            maxdepth = typemax(Int), 
            criterion = rf.criterion, 
            attrfilter = rf.attrfilter
        )
        fit!(t, dfs[i], topredict)
        trees[id] = [get(trees, id, []); [t]]
    end
    rf.trees = reduce((x,y) -> [x;y], values(trees))
    rf
end

predict(rf::RandomForest, df) = findmajority(map(t -> predict(t, df), rf.trees))