
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
function MON_C4_5(data, target, attributes, path, branch, R=1, max_depth=Inf, min_samples=1, depth=0)
    
    # Vérifier les conditions d'arrêt (profondeur maximale ou nombre minimal d'échantillons)
    if depth >= max_depth || nrow(data) <= min_samples
        current_path = [path..., StatsBase.mode(data[:, target])]
        push!(branch, [current_path])
        # Créer un nœud feuille avec la classe majoritaire dans l'ensemble de données
        return StatsBase.mode(data[:, target])
    end

    if size(unique(data[:,target]),1) == 1
        # Cas de base : si tous les exemples ont la même classe
        current_path = [path..., data[1,target]]
        push!(branch, [current_path])
        # retourner un nœud avec cette classe
        return data[1,target]
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

    # Sélection de la meilleure caractéristique pour la division
    best_info_gain = 0.0
    best_attribute = ""
    total_entropy = entropy(data, target)
    for attribute in attributes
        values = unique(data[:,attribute])
        n = size(values,1)
        if n != 1
            T = copy(tab)
            for i in 1:n-1
                push!(T, Vector{Any}(missing, size(T,2)))
            end
            if !isempty(path)
                m = length(path)
                for j in 1:Int((m)/2)
                    T[end-n+1:end, path[2*j-1]] .= path[2*j]
                end
            end
            score = 0.0
            for i in 1:n
                T[end-i+1,attribute] = values[i]
                subset = filter(attribute => ==(values[i]), data)
                T[end-i+1,target] = StatsBase.mode(subset[:,attribute])
                score += (size(subset,1) / size(data,1)) * entropy(subset, target)
            end
        
            INM = NMI(T)
            # Calcule du gain d'information
            info_gain = total_entropy - (score + R*score_a(INM))
        
            # Calcule de l'entropie de l'attribut
            attr_entropy = entropy(data, attribute)

            # Calcule du gain ratio
            gain_ratio = info_gain / attr_entropy
        
            if gain_ratio > best_info_gain
                best_info_gain = gain_ratio
                best_attribute = attribute
            end
        end
    end

    if isempty(best_attribute)
        majority_class = StatsBase.mode(data[:,target])
        current_path = [path..., majority_class]
        push!(branch, [current_path])
        return majority_class
    end

    # Création du nœud de décision avec la meilleure caractéristique
    tree = Node(best_attribute, Dict())
    depth += 1
    current_path = [path..., best_attribute]

    # Séparation des exemples en fonction des valeurs de la meilleure caractéristique
    feature_values = sort(unique(data[:,best_attribute]))
    for value in feature_values
        subset = filter(best_attribute => ==(value), data)
        # Construction récursive de l'arbre pour le sous-ensemble
        tree.children[value] = MON_C4_5(subset, target, attributes, [current_path..., value], branch, R, max_depth, min_samples, depth)
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