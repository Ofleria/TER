using DataFrames

# Fonction pour créer un échantillon bootstrap
function bootstrap_sample(data, n_trees)
    n = size(data, 1)
    indices = rand(1:n, n_trees)
    return data[indices, :]
end


function Mon_Forest_RI(data, ntree, n_samples, R_limit, T, target, attributes, max_depth=Inf, min_samples=1)
    M_RF = DataFrame()
    M_RF[!,"tree"] = Vector{Any}(missing, ntree)
    M_RF[!,"INM"] = zeros(ntree)
    n = size(data, 1)
    for i in 1:ntree
        data_2 = bootstrap_sample(data, n_samples) #échantillonnage
        R = rand(1:R_limit) #sélection du facteur d'importance
        # Construction de l'arbre de décision
        M_RF[i,"tree"] = mon_cart(data_2, target, attributes, [], DataFrame(col=Vector()), R, max_depth, min_samples, 0)
        # Extration des branches
        paths = DataFrame(col=Vector())
        branchs(M_RF[i,"tree"], [], paths)
        tab = data_tree(paths, attributes, target)
        # Calcule de l'indice de non-monotonie
        M_RF[i,"INM"] = NMI(tab)
    end
    # Classement par ordre croissant
    sort!(M_RF, "INM")
    # Elagage
    TREE = M_RF[1:Int(round(ntree*T)),:]
    return TREE
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

function predict_rf(trees, example)
    ntree = size(trees,1)
    pred = zeros(Int, ntree)
    
    for i in 1:ntree
        pred[i] = predict_bi(trees[i,1], example)
    end
    
    return StatsBase.mode(pred)
end
