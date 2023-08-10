
using DataFrames
using StatsBase
# Structure de l'arbre de décision
struct Node
    feature::String
    children::Dict
end

# Fonction pour construire l'arbre de décision
function cart(data, target, attributes, max_depth=Inf, min_samples=1, depth=0)
    # Vérifier les conditions d'arrêt (profondeur maximale ou nombre minimal d'échantillons)
    if depth >= max_depth || nrow(data) <= min_samples
        # Créer un nœud feuille avec la classe majoritaire dans l'ensemble de données
        return StatsBase.mode(data[:, target])
    end
    
    # Vérifier si tous les exemples appartiennent à la même classe
    classes = unique(data[:,target])
    if size(classes,1) == 1
        return classes[1]
    end
    
    # Sélectionner l'attribut et la valeur de découpe qui maximisent le gain de Gini
    best_attribute, best_value, best_gini = "", 0, Inf
    for attribute in attributes
        values = sort(unique(data[:,attribute]))
        for value in values
            gini = gini_score(data, target, attribute, value)
            if best_gini > gini
                best_attribute = attribute
                best_value = value
                best_gini = gini
            end
        end
    end

    # Vérifier si les attributs sont vides
    if isempty(best_attribute)
        return StatsBase.mode(data[:,target])
    end
    
    # Créer le noeud de décision
    node = Node(string(best_attribute), Dict())
    depth += 1
    
    # Partitionner les exemples en fonction de la valeur de découpe sélectionnée
    left_subset = filter(best_attribute => <=(best_value), data)
    right_subset = filter(best_attribute => >(best_value), data)
    
    # Construction récursive des sous-arbres gauche et droit
    if isempty(left_subset) || isempty(right_subset)
        return StatsBase.mode(data[:,target])
    else
        # Créer le noeud de décision
        node = Node(string(best_attribute), Dict())
        value = sort(unique(right_subset[:,best_attribute]))[1]
        depth += 1
        node.children[best_value] = cart(left_subset, target, attributes, max_depth, min_samples, depth)
        node.children[value] = cart(right_subset, target, attributes, max_depth, min_samples, depth)
    end
    
    return node
end

# Fonction pour calculer l'indice de Gini 
function gini_index(data, target)
    b = combine(groupby(data, target), nrow)
    probabilities = b[:,2] ./ size(data,1)
    gini = 1.0 - sum(probabilities .^2)
    return gini
end

# Fonction pour calculer le score de Gini 
function gini_score(data, target, feature, value)
    left_subset = filter(feature => <=(value), data)
    index_left = gini_index(left_subset, target)

    right_subset = filter(feature => >(value), data)
    index_right = gini_index(right_subset, target)

    gini = (index_left*size(left_subset,1) + index_right*size(right_subset,1))/size(data,1)
    return gini
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