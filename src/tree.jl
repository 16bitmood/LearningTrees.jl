abstract type Node end

struct LeafNode <: Node
    value
end

struct CategoricalNode <: Node
    attribute
    children  :: Dict{Any, Node}
    majority  :: LeafNode
end

struct NumericNode <: Node
    attribute
    split     :: Number
    left      :: Node
    right     :: Node
end

# Calculate Impurity
impurity(df, current, clf) = begin
    n = nrow(df)

    if eltype(df[!, current]) <: Number
        df = sort!(df, current)

        countl, countr = Counter(df[1:1, clf.topredict]), Counter(df[2:end, clf.topredict])
        minimp, splitat = weightedsum(map(clf.criterion, [countl, countr]), [1/n, 1-1/n]), 1

        for i = 2:(n-1)
            x = df[i, clf.topredict]
            countl[x] += 1
            countr[x] -= 1
            imp = weightedsum(map(clf.criterion, [countl, countr]), [i/n, 1-i/n])

            if imp <= minimp
                minimp, splitat = imp, i
            end
        end
        splitval = (df[splitat, current] + df[splitat+1, current])/2
        minimp, (splitat, splitval)
    else
        sdfs = values(groupby(df, current))
        imps = [(nrow(sdf)/n)*clf.criterion(sdf[!, clf.topredict]) for sdf in sdfs]
        sum(imps), nothing
    end
end

# Building a Tree
Node(df, attributes, depth, clf) = begin
    if (length(Set(df[:, clf.topredict])) == 1 || depth >= clf.maxdepth  ||
        length(attributes) == 0  || nrow(df) <= 2)
        return LeafNode(findmajority(df[:, clf.topredict]))
    end

    minimp, mincurrent, minsplit = Inf, nothing, nothing
    for current in attributes
        imp, split = impurity(df, current, clf)
        if imp <= minimp
            minimp, mincurrent, minsplit = imp, current, split
        end
    end
    current, split = mincurrent, minsplit

    if eltype(df[!, current]) <: Number 
        df = sort!(df, current)
        splitat, splitval = split
        dfl, dfr = df[1:splitat, :], df[splitat:end, :]
        left  = Node(dfl, setdiff(attributes, [current]), depth+1, clf)
        right = Node(dfr, setdiff(attributes, [current]), depth+1, clf)
        NumericNode(current, splitval, left, right)
    else 
        children = Dict()
        for ((c,), sdf) in pairs(groupby(df, current))
            children[c] = Node(sdf, setdiff(attributes, [current]), depth+1, clf)
        end
        majority = LeafNode(findmajority(df[:, clf.topredict]))
        CategoricalNode(current, children, majority)
    end
end

# Predicting
predict(node::Node, df::DataFrame) = [predict(node, x) for x in eachrow(df)]

predict(node::LeafNode, x::DataFrameRow) = node.value

predict(node::CategoricalNode, x::DataFrameRow) = begin
    category = x[node.attribute]
    switchto = get(node.children, category, node.majority)
    predict(switchto, x)
end

predict(node::NumericNode, x::DataFrameRow) = begin
    if x[node.attribute] <= node.split
        predict(node.left, x)
    else
        predict(node.right, x)
    end
end