
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

# Fonction pour trouver le seuil optimal
function Threshold2(data, attr, target, T)
    A = sort(unique(data[:,attr]))
    n = length(A)
    S = zeros(n,1) 
    for i in 1:n-1
        S[i] = (A[i] + A[i+1])/2 # les différents seuils
    end
    S[n] = A[n]
    best_info_gain = Inf
    best_threshold = -Inf
    total_entropy = entropy(data, target)
    attr_entropy = entropy(data, attr)
    for i in 1:n-1
        # Les branches en construction
        T[end-1,attr] = S[i]
        T[end,attr] = S[i+1]
        # Les sous-ensembles
        left_subset = filter(attr => <=(S[i]), data)
        right_subset = filter(attr => >(S[i]), data)
        # Le noeud feuille des branches en construction
        T[end-1,target] = StatsBase.mode(left_subset[:,target])
        T[end,target] = StatsBase.mode(right_subset[:,target])
        
        left_entropy = entropy(left_subset, target)
        right_entropy = entropy(right_subset, target)
        # Calcul de l'indice de non-monotonie
        INM = NMI(T)
        
        weighted_entropy = (size(left_subset,1) / size(data,1) * left_entropy) + (size(right_subset,1) / size(data,1) * right_entropy)
        # Calcule du gain d'information
        info_gain = weighted_entropy + R*score_a(INM)
        # Calcule du gain ratio
        gain_ratio = info_gain / attr_entropy
        
        if best_info_gain > gain_ratio
            best_info_gain = gain_ratio
            best_threshold = S[i]
        end
    end
    return best_threshold, best_info_gain
end

# Fonction pour construire l'arbre de décision
function MON_C4_5_con(data, target, attributes, path, branch, R=1, max_depth=Inf, min_samples=1, depth=0)
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
        # Transformation de l'arbre déjà construit en tableau de branches
        tab = data_tree(branch, attributes, target)
        push!(tab, Vector{Any}(missing, size(tab,2)))
    else
        # Création d'un tableau de branches
        tab = DataFrame()
        for value in attributes
            tab[!,value] = Vector{Any}(missing,1)
        end
        tab[!, target] =  Vector{Any}(missing,1)
    end

    # Sélection de la meilleure caractéristique pour la division
    best_info_gain = Inf
    best_attribute = ""
    best_threshold = -Inf
    for attribute in attributes
        
        T = copy(tab)
        push!(T, Vector{Any}(missing, size(T,2)))
        # Ajout des branches en construction dans le tableau de branches
        if !isempty(path)
            m = length(path)
            for j in 1:Int((m)/2)
                T[end-1:end, path[2*j-1]] .= path[2*j]
            end
        end
        # Calcul du meilleur seuil
        (threshold, gain_ratio) = Threshold2(data, attribute, target, T)
        if best_info_gain > gain_ratio
            best_info_gain = gain_ratio
            best_attribute = attribute
            best_threshold = threshold
        end
    end

    if isempty(best_attribute)
        # Aucun attribut n'est sélectionné
        majority_class = StatsBase.mode(data[:,target])
        current_path = [path..., majority_class]
        push!(branch, [current_path])
        return majority_class
    end

    # Création du nœud de décision avec la meilleure caractéristique
    tree = Node(best_attribute, Dict())
    depth += 1
    current_path = [path..., best_attribute]

    # Séparation des exemples en fonction du seuil
    left_subset = filter(best_attribute => <=(best_threshold), data)
    right_subset = filter(best_attribute => >(best_threshold), data)
    if isempty(left_subset) || isempty(right_subset)
        return StatsBase.mode(data[:,target])
    else
        value = sort(unique(right_subset[:,best_attribute]))[1]
        tree.children[best_threshold] = MON_C4_5_con(left_subset, target, attributes, [current_path..., best_threshold], branch, R, max_depth, min_samples, depth)
        tree.children[value] = MON_C4_5_con(right_subset, target, attributes, [current_path..., value], branch, R, max_depth, min_samples, depth)
    end
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