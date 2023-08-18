using DataFrames
using StatsBase
# Structure de l'arbre de décision
struct Node
    feature::String
    children::Dict
end

# Fonction pour créer un échantillon bootstrap
function bootstrap_sample(data, n_samples)
    n = size(data, 1)
    indices = rand(1:n, n_samples)
    return data[indices, :]
end

function Forest_RI(data, ntree, n_samples, attributes, target)
    trees = DataFrame()
    trees[!,"tree"] = Vector{Any}(missing, ntree)
    n = size(data, 1)
    for i in 1:ntree
        data_2 = bootstrap_sample(data, n_samples)
        trees[i,"tree"] = cart(data_2, target, attributes)
    end
    return trees
end

# Fonction de prédiction à partir d'une forêt aléatoire
function predict_rf(trees, example)
    ntree = size(trees,1)
    pred = zeros(Int, ntree)
    
    for i in 1:ntree
        pred[i] = predict_bi(trees[i,1], example)
    end
    
    return StatsBase.mode(pred)
end

