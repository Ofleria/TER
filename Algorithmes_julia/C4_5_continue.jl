
using DataFrames
using StatsBase
# Structure de l'arbre de décision
struct Node
    feature::String
    children::Dict
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

# Fonction pour calculer le gain d'information
function information_gain(data, feature, target)
    total_entropy = entropy(data, target)
    score = score_e(data, feature, target)
    return total_entropy - score
end

# Fonction pour la recherche du meilleur seuil
function Threshold1(data, attr, target)
    A = sort(unique(data[:,attr]))
    n = length(A)
    S = zeros(n-1,1)
    for i in 1:n-1
        # Calcul de chaque seuil
        S[i] = (A[i] + A[i+1])/2 
    end
    
    best_info_gain = 0.0
    best_threshold = -Inf
    total_entropy = entropy(data, target)
    attr_entropy = entropy(data, attr)
    for i in 1:n-1
        # Scission des données en deux 
        left_subset = filter(attr => <=(S[i]), data)
        right_subset = filter(attr => >(S[i]), data)
        # Entropie de chaque sous-ensemble
        left_entropy = entropy(left_subset, target)
        right_entropy = entropy(right_subset, target)
        
        weighted_entropy = (size(left_subset,1) / size(data,1) * left_entropy) + (size(right_subset,1) / size(data,1) * right_entropy)
        
        # Calcule du gain d'information
        info_gain = total_entropy - weighted_entropy

        # Calcule du gain ratio
        gain_ratio = info_gain / attr_entropy
        
        if gain_ratio > best_info_gain
            best_info_gain = gain_ratio
            best_threshold = S[i]
        end
    end
    return best_threshold, best_info_gain
end

# C4.5 pour des données continues
function C4_5_con(data, target, attributes, max_depth=Inf, min_samples=1, depth=0)
    
    # Vérifier les conditions d'arrêt (profondeur maximale ou nombre minimal d'échantillons)
    if depth >= max_depth || nrow(data) <= min_samples
        # Créer un nœud feuille avec la classe majoritaire dans l'ensemble de données
        return StatsBase.mode(data[:, target])
    end

    if length(unique(data[:,target])) == 1
        # Cas de base : si tous les exemples ont la même classe
        # retourner un nœud avec cette classe
        return data[1,target]
    end

    # Sélection de la meilleure caractéristique pour la division
    best_info_gain = 0.0
    best_attribute = ""
    best_threshold = -Inf
    for attribute in attributes
        (threshold, gain_ratio) = Threshold1(data, attribute, target)
        if gain_ratio > best_info_gain
            best_info_gain = gain_ratio
            best_attribute = attribute
            best_threshold = threshold
        end
    end
    
    if isempty(best_attribute)
        return StatsBase.mode(data[:,target])
    end

    # Partitionner les exemples en fonction du seuil
    left_subset = filter(best_attribute => <=(best_threshold), data)
    right_subset = filter(best_attribute => >(best_threshold), data)
    value = sort(unique(right_subset[:,best_attribute]))[1]
    
    # Création du nœud de décision avec la meilleure caractéristique
    tree = Node(best_attribute, Dict())
    depth += 1

    # Séparation des exemples en fonction des valeurs du meilleur seuil
    tree.children[best_threshold] = C4_5_con(left_subset, target, attributes, max_depth, min_samples, depth)
    tree.children[value] = C4_5_con(right_subset, target, attributes, max_depth, min_samples, depth)
    
    return tree
end

# Fonction pour prédire la classe d'un exemple en utilisant l'arbre de décision
function predict_bi(tree, example)
    if typeof(tree)==Int
        return tree
    else
        k = []
        for i in keys(tree.children)
            push!(k,i)
        end
        k = sort(k)
        if example[tree.feature] <= k[1]
            return predict_bi(tree.children[k[1]], example)
        else
            return predict_bi(tree.children[k[2]], example)
        end
    end
end