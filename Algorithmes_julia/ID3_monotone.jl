
using DataFrames
using StatsBase
# Structure de l'arbre de décision
struct Node
    feature::String
    children::Dict
end

# Fonction pour construire l'arbre de décision
function MON_ID3(data, target, attributes, path, branch, R=1, max_depth=Inf, min_samples=1, depth=0)
    # Vérifier les conditions d'arrêt (profondeur maximale ou nombre minimal d'échantillons)
    if depth >= max_depth || nrow(data) <= min_samples
        current_path = [path..., StatsBase.mode(data[:, target])]
        push!(branch, [current_path])
        # Créer un nœud feuille avec la classe majoritaire dans l'ensemble de données
        return StatsBase.mode(data[:, target])
    end

    if  size(unique(data[:,target]),1) == 1
        # Cas de base : si tous les exemples ont la même classe
        current_path = [path..., data[1,target]]
        push!(branch, [current_path])
        # retourner un nœud avec cette classe
        return data[1,target]
    end
    
    if length(attributes) == 0
        # Cas de base : si toutes les caractéristiques ont été utilisées
        current_path = [path..., StatsBase.mode(data[:, target])]
        push!(branch, [current_path])
        # retourner un nœud avec la classe majoritaire
        return StatsBase.mode(data[:, target])
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
    best_feature = ""
    best_score = Inf
    for feature in attributes
        values = unique(data[:,feature])
        n = length(values)
        if n != 1
            T = copy(tab)
            for i in 1:n-1
                push!(T, zeros(size(T,2)))
            end
            if !isempty(path)
                m = length(path)
                for j in 1:Int((m)/2)
                    T[end-n+1:end, path[2*j-1]] .= path[2*j]
                end
            end
            score = 0.0
            for i in 1:n
                T[end-i+1,feature] = values[i]
                subset = filter(feature => ==(values[i]), data)
                T[end-i+1,target] = StatsBase.mode(subset[:,feature])
                score += (size(subset,1) / size(data,1)) * entropy(subset, target)
            end
        
            INM = NMI(T)
            score += R*score_a(INM)
            if best_score > score
                best_score = score
                best_feature = feature
            end
        end
    end

    if isempty(best_feature)
        majority_class = StatsBase.mode(data[:,target])
        current_path = [path..., majority_class]
        push!(branch, [current_path])
        return majority_class
    end

    # Création du nœud de décision avec la meilleure caractéristique
    tree = Node(best_feature, Dict())
    depth += 1
    current_path = [path..., best_feature]
    
    # Séparation des exemples en fonction des valeurs de la meilleure caractéristique
    feature_values = sort(unique(data[:,best_feature]))
    for value in feature_values
        subset = filter(best_feature => ==(value), data)
        # Construction récursive de l'arbre pour le sous-ensemble
        tree.children[value] = MON_ID3(subset, target, attributes, [current_path..., value], branch, R, max_depth, min_samples, depth)
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