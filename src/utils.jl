# Counter
struct Counter
    counts::Dict{Any, Any}
end

Counter(xs::AbstractVector) = begin
    counts = Dict()
    for x in xs
        counts[x] = 1 + get(counts, x, 0)
    end
    Counter(counts)
end

Base.getindex(cnt::Counter, i) = get(cnt.counts, i, 0)
Base.setindex!(cnt::Counter, v, i) = begin cnt.counts[i] = v end

Base.keys(cnt::Counter)   = Base.keys(cnt.counts)
Base.values(cnt::Counter) = Base.values(cnt.counts)
total(cnt::Counter)  = sum(Base.values(cnt))

# Helpers
weightedsum(xs, ws) = foldl(((s,(x,w)) -> s + w*x), zip(xs, ws); init = 0)

findmajority(xs::AbstractVector) = findmajority(Counter(xs))

findmajority(cnt::Counter) = begin
    reduce((x, y) -> cnt[x] >= cnt[y] ? x : y, keys(cnt))
end

traintestsplit(df::DataFrame, testfraction = 0.3) = begin
    df = DataFrame(shuffle(eachrow(df)))
    k = Int(ceil(testfraction * nrow(df)))
    df[k:end, :], df[1:k, :]
end

# Impurity Measures
gini(xs::AbstractVector) = gini(Counter(xs))

gini(cnt::Counter) = begin
    n = total(cnt)
    1 - sum((c/n)^2 for c in values(cnt))
end

entropy(xs::AbstractArray) = entropy(Counter(xs))

entropy(cnt::Counter) = begin
    n = total(cnt)
    -sum((c/n)*log(2, (c/n)) for c in values(cnt))
end

confusionmatrix(clf, df, topredict, trueclass) = begin
    ytrue = map(x -> x == trueclass, df[:,topredict])
    ypred = map(x -> x == trueclass, predict(clf, df))

    cm = Counter(collect(zip(ytrue, ypred)))
    cm[(true, true)], cm[(true, false)], cm[(false, true)], cm[(false, false)]
end

precisionscore(confusionmatrix) = begin
    tp, fn, fp, tn = confusionmatrix
    tp/(tp+fp)
end

recallscore(confusionmatrix) = begin
    tp, fn, fp, tn = confusionmatrix
    tp/(tp+fn)
end

accuracyscore(confusionmatrix) = begin
    tp, fn, fp, tn = confusionmatrix
    (tp+tn)/(tp+fn+fp+tn)
end

accuracyscore(ytrue, ypred) = begin
    sum(a == b for (a,b) in zip(ytrue, ypred))/length(ytrue)
end

f1score(confusionmatrix) = begin
    tp, fn, fp, tn = confusionmatrix
    tp/(tp + 0.5*(fp + fn))
end
