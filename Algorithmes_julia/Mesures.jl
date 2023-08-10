
#mesures d'évaluation
function kendall_tau(r1, r2)
    n = length(r1)
    if n < 2
        return 1
    end
    tau = 0.0
    for i = 1:n-1
        for j=i+1:n
            if (r1[i]-r1[j]) * (r2[i]-r2[j]) > 0
                tau += 1
            elseif r1[i]-r1[j]==0 && r2[i]-r2[j]==0
            end
        end
    end
    2 * tau / (n * (n-1))
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

function MZE(X, Y)
    n=length(X)
    r = 0
    for i in 1:n
        if X[i] != Y[i]
            r += 1
        end
    end
    return r/n
end



function MAE(X, Y)
    n=length(X)
    r = 0
    for i in 1:n
        r += abs(X[i]-Y[i])
    end
    return r/n
end

function leaves_count(tree, attributes, target)
    paths = DataFrame(col=Vector())
    branchs(tree, [], paths)
    tab = data_tree(paths, attributes, target)
    leaves = size(tab,1)

    return leaves
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
