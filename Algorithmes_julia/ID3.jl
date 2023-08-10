
using DataFrames
using StatsBase
# Structure de l'arbre de décision
struct Node
    feature::String
    children::Dict
end

# Fonction pour construire l'arbre de décision
function ID3(data, attributes, target)
    if size(unique(data[:,target]),1) == 1
        # Cas de base : si tous les exemples ont la même classe
        # retourner un nœud avec cette classe
        return data[1,target]
    end
    
    if length(attributes) == 0
        # Cas de base : si toutes les caractéristiques ont été utilisées
        # retourner un nœud avec la classe majoritaire
        return StatsBase.mode(data[:, target])
    end
    
    # Sélection de la meilleure caractéristique pour la division
    best_feature = ""
    best_score = Inf
    for feature in attributes
        score = score_e(data, feature, target)
        if best_score > score
            best_score = score
            best_feature = feature
        end
    end

    if isempty(best_feature)
        return StatsBase.mode(data[:,target])
    end

    # Création du nœud de décision avec la meilleure caractéristique
    tree = Node(best_feature, Dict())
    subset_attributes = filter(x -> x != best_feature, attributes)
    
    # Séparation des exemples en fonction des valeurs de la meilleure caractéristique
    feature_values = sort(unique(data[:,best_feature]))
    for value in feature_values
        sub = filter(best_feature => ==(value), data)
        subset = select(sub, Not(best_feature))
        # Construction récursive de l'arbre pour le sous-ensemble
        tree.children[value] = ID3(subset, subset_attributes, target)
    end    
    return tree
end

function log_2(x)
    if x == 0
        return x
    else
        return log2(x)
    end
end

# Fonction pour calculer l'entropie
function entropy(data, target)
    b = combine(groupby(data, target), nrow) # proportion des différentes classes dans data
    probabilities = b[:,2] ./ size(data)[1] # probabilité de chaque classe
    entropy = -sum(probabilities .* log_2.(probabilities))
    return entropy
end

# Fonction pour calculer le score E
function score_e(data, feature, target)
    feature_values = unique(data[:,feature]) # les différentes valeurs de l'attribut
    weighted_entropy = 0.0
    for value in feature_values
        subset = filter(feature => ==(value), data) # sous-ensemble de data 
        subset_entropy = entropy(subset, target)
        weighted_entropy += (size(subset)[1] / size(data)[1]) * subset_entropy
    end
    return weighted_entropy
end

# Fonction pour prédire la classe d'un exemple en utilisant l'arbre de décision
function predict_tree(tree, example)
    if typeof(tree)==Int
        return tree
    else
        value = example[tree.feature]
        k = []
        for i in keys(tree.children)
            push!(k,i)
        end
        i = length(k)
        k = sort(k)
        while value < k[i] && i!=1
            i-=1
        end
        subtree = tree.children[k[i]]
        return predict_tree(subtree, example)
    end
end

