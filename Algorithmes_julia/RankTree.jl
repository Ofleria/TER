
using DataFrames
using StatsBase
# Structure de l'arbre de décision
struct Node
    feature::String
    children::Dict
end

function RankTree(data, target, attributes, max_depth=Inf, min_samples=1, depth=0)
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
    best_attribute = ""
    best_value = 0
    best_decrease = Inf
    for attribute in attributes
        values = sort(unique(data[:,attribute]))
        for value in values[1:end-1]

            left_child = filter(attribute => <=(value), data)
            left_I = ranking_impurity(left_child, target)

            right_child = filter(attribute => >(value), data)
            right_I = ranking_impurity(right_child, target)

            score_I = left_I + right_I

            if score_I < best_decrease
                best_decrease = score_I
                best_attribute = attribute
                best_value = value   
            end
        end
    end
    
    if isempty(best_attribute)
        return StatsBase.mode(data[:,target])
    end
    
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
        node.children[best_value] = RankTree(left_subset, target, attributes, max_depth, min_samples, depth)
        node.children[value] = RankTree(right_subset, target, attributes, max_depth, min_samples, depth)
    end
   
    return node
end

function ranking_impurity(data, target)
    N = combine(groupby(sort(data,target), target), nrow)
    n = size(N,1)
    impurity = 0.0
    for j in 1:n
        for i in 1:j
            impurity += (j-i)*N[j,2]*N[i,2]
        end
    end
    return impurity
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