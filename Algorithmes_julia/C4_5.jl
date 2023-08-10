
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

# Fonction pour trouver le seuil optimal d'un attribut continu
function Threshold(data, attr, order, target)
    A = sort(unique(data[:,attr]))
    n = length(A)
    S = zeros(n-1,1)
    if order == "c"
        for i in 1:n-1
            S[i] = (A[i] + A[i+1])/2
        end
    elseif order == "d"
        for i in n:-1:2
            S[n-i+1] = (A[i] + A[i-1])/2
        end
    end
    
    best_info_gain = 0.0
    best_threshold = -Inf
    total_entropy = entropy(data, target)
    attr_entropy = entropy(data, attr)
    for i in 1:n-1
        left_subset = filter(attr => <=(S[i]), data)
        right_subset = filter(attr => >(S[i]), data)
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
    return best_threshold
end

# Fonction pour construire l'arbre de décision
function C4_5(data, target, attributes, max_depth=Inf, min_samples=1, depth=0)
    
    # Vérifier les conditions d'arrêt (profondeur maximale ou nombre minimal d'échantillons)
    if depth >= max_depth || nrow(data) <= min_samples
        # Créer un nœud feuille avec la classe majoritaire dans l'ensemble de données
        return StatsBase.mode(data[:, target])
    end

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
    best_info_gain = 0.0
    best_attribute = ""
    
    for attribute in attributes
        # Calcule du gain d'information
        info_gain = information_gain(data, attribute, target)

        # Calcule de l'entropie de l'attribut
        attr_entropy = entropy(data, attribute)

        # Calcule du gain ratio
        gain_ratio = info_gain / attr_entropy

        if gain_ratio > best_info_gain
            best_info_gain = gain_ratio
            best_attribute = attribute
        end
    end
    
    if isempty(best_attribute)
        return StatsBase.mode(data[:,target])
    end

    # Création du nœud de décision avec la meilleure caractéristique
    tree = Node(best_attribute, Dict())
    depth += 1
    subset_attributes = filter(x -> x != best_attribute, attributes)

    # Séparation des exemples en fonction des valeurs de la meilleure caractéristique
    feature_values = sort(unique(data[:,best_attribute]))
    for value in feature_values
        sub = filter(best_attribute => ==(value), data)
        subset = select(sub, Not(best_attribute))
        # Construction récursive de l'arbre pour le sous-ensemble
        tree.children[value] = C4_5(subset, target, subset_attributes, max_depth, min_samples, depth)
    end
    
    return tree
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