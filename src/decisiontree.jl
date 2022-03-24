mutable struct DecisionTree <: Classifier
    root      :: Union{Nothing, Node}
    topredict
    maxdepth  :: UInt
    criterion 
    attrfilter
end

DecisionTree(;maxdepth = typemax(Int), criterion = gini, attrfilter = identity) = begin
    DecisionTree(nothing, nothing, maxdepth, criterion, attrfilter)
end

fit!(dt::DecisionTree, df::DataFrame, topredict) = begin
    dt.topredict = topredict
    dt.root = Node(df, setdiff(propertynames(df), [topredict]), 0, dt)
    dt
end

predict(dt::DecisionTree, df) = predict(dt.root, df)