
using DataFrames
using StatsBase
# Structure de l'arbre de décision
struct Node
    feature::String
    children::Dict
end

function mon_cart(data, target, attributes, path, branch, R=1, max_depth=Inf, min_samples=1, depth=0)
    # Vérifier les conditions d'arrêt (profondeur maximale ou nombre minimal d'échantillons)
    if depth >= max_depth || nrow(data) <= min_samples
        majority_class = StatsBase.mode(data[:,target])
        current_path = [path..., majority_class]
        push!(branch, [current_path])
        # Créer un nœud feuille avec la classe majoritaire dans l'ensemble de données
        return majority_class
    end
    # Vérifier si tous les exemples appartiennent à la même classe
    classes = unique(data[:,target])
    if length(classes) == 1
        current_path = [path..., classes[1]]
        push!(branch, [current_path])
        return classes[1]
    end
       
    if !isempty(branch)
        tab = data_tree(branch, attributes, target)
        push!(tab, Vector{Any}(missing, size(tab,2)))
    else
        tab = DataFrame()
        for value in attributes
            tab[!,value] = Vector{Any}(missing, 1)
        end
        tab[!, target] = Vector{Any}(missing, 1)
    end
    
    # Sélectionner l'attribut et la valeur de découpe qui maximisent le score d'ambiguïté totale
    best_attribute = ""
    best_value = 0
    best_score = Inf
    for attribute in attributes
        T = copy(tab)
        push!(T, Vector{Any}(missing, size(T,2)))
        if !isempty(path)
            m = length(path)
            for j in 1:Int((m)/2)
                T[end-1:end, path[2*j-1]] .= path[2*j]
            end
        end
        values = sort(unique(data[:,attribute]))
        n = size(values,1)
        for i in 1:n-1
            T[end-1,attribute] = values[i]
            left_child = filter(attribute => <=(values[i]), data)
            T[end-1,target] = StatsBase.mode(left_child[:,target])

            right_child = filter(attribute => >(values[i]), data)
            T[end,attribute] = values[i+1]
            T[end,target] = StatsBase.mode(right_child[:,target])
            
            index_left = gini_index(left_child, target)
            index_right = gini_index(right_child, target)

            gini = (index_left*size(left_child,1) + index_right*size(right_child,1))/size(data,1)
            INM = NMI(T)
            score = gini + R*score_a(INM)
            if best_score > score
                best_attribute = attribute
                best_value = values[i]
                best_score = score
            end
        end
    end
    
    if isempty(best_attribute)
        majority_class = StatsBase.mode(data[:,target])
        current_path = [path..., majority_class]
        push!(branch, [current_path])
        return majority_class
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
        current_path = [path..., best_attribute]
        node.children[best_value] = mon_cart(left_subset, target, attributes, [current_path..., best_value], branch, R, max_depth, min_samples, depth)
        node.children[value] = mon_cart(right_subset, target, attributes, [current_path..., value], branch, R, max_depth, min_samples, depth)
    end
    
    return node
end

function NMI(tab)
    (n,m) = size(tab)
    W = 0
    for i in 1:n-1
        for j in i+1:n
            inf = 0
            sup = 0
            eq = 0
            for k in 1:m-1
                if (!ismissing(tab[i,k]) && !ismissing(tab[j,k]))
                    if tab[i,k] < tab[j,k]
                        inf += 1
                    elseif tab[i,k] > tab[j,k]
                        sup += 1
                    else
                        eq += 1
                    end
                end
            end
            if (sup==0 && tab[i,m] > tab[j,m])
                W += 1
            elseif (inf == 0 && tab[i,m] < tab[j,m])
                W += 1
            elseif (eq==m-1 && tab[i,m] != tab[j,m])
                W += 1
            end
        end
    end
    return W*2 / (n^2-n)
end

function score_a(INM)
    if INM == 0
        return 0
    else
        return -(log2(INM))^(-1)
    end
end

# Fonction récursive pour extraire les différentes branches de l'arbre
function branchs(node, path, paths)
    if typeof(node)==Int
        current_path = [path..., node]
        push!(paths, [current_path])
    else
        current_path = [path..., node.feature]
        for (value, child_node) in pairs(node.children)
            branchs(child_node, [current_path..., value], paths)
        end
    end
end

# Fonction pour extraire les éléments des branches dans un tableau 
function data_tree(paths, attributes, target)

    n = size(paths,1)
    tab = DataFrame()
    for value in attributes
        tab[!,value] = Vector{Any}(missing, n)
    end
    tab[!, target] = Vector{Any}(missing, n)
    
    for i in 1:n
        m = length(paths[i,1])
        for j in 1:Int((m-1)/2)
            tab[i, paths[i,1][2*j-1]] = paths[i,1][2*j]
        end
        tab[i,target] = paths[i,1][m]
    end

    return tab
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