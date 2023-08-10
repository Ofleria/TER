using Combinatorics, JuMP, GLPK
K = 3
PT = [2 3 2; 3 3 2; 1 3 2; 2 2 1; 3 1 3;
      1 2 4; 1 1 3; 2 1 4; 3 2 4; 3 2 1]

C = [3,3,2,1,2,2,1,1,3,2]
n, m = size(PT)

P = collect(powerset(collect(1:m),1))
sP = length.(P)
np = length(P)

D = permutedims(permutedims(repeat(PT,1,1,n),(3,2,1)) .>= PT, (3,1,2)) # dominance i x j for any criterion k

DP = [all(D[:,:,p], dims = 3) for p in P]

LAge = cat(
[
    reshape(
        permutedims(
            all(
                repeat(.!DP[i], 1, 1, K) .|| (repeat(DP[i] .* C, 1, 1, K) .>= reshape(collect(1:K),1,1,K))
                , dims = 1)
            , (3,2,1))
        , K, n)
    for i=1:np
]..., dims = 3) # lower approximation for >=

UAge = cat(
[
    reshape(
        permutedims(
            any(
                repeat(DP[i] .* C', 1, 1, K) .>= reshape(collect(1:K),1,1,K)
                , dims = 2)
            , (3,1,2))
        , K, n)
    for i=1:np
]..., dims = 3) # upper approximation for >=

BRge = UAge .&& .!LAge # boundary for >=

LAle = cat(
[
    reshape(
        permutedims(
            all(
                repeat(DP[i] .* C', 1, 1, K) .<= reshape(collect(1:K),1,1,K)
                , dims = 2)
            , (3,1,2))
        , K, n)
    for i=1:np
]..., dims = 3) # lower approximation for <=

UAle = cat(
[
    reshape(
        permutedims(
            any(
                repeat(DP[i], 1, 1, K) .&& (repeat(DP[i] .* C, 1, 1, K) .<= reshape(collect(1:K),1,1,K))
                , dims = 1)
            , (3,2,1))
        , K, n)
    for i=1:np
]..., dims = 3) # upper approximation for <=

BRle = UAle .&& .!LAle # boundary for <=

indices = []
cover = falses(n,0)
cost = []
for k = 1:K
    if k == 1
        X = vec(any(LAle[k,:,:], dims = 1))
        for i=1:np
            if X[i]
                Y = vec(all(LAle[k,:,:] .== LAle[k,:,i], dims = 1))
                if minimum(sP[Y]) == sP[i]
                    global cover = hcat(cover, LAle[k,:,i:i])
                    push!(indices, (k,[],P[i],true))
                    push!(cost, 1)
                end
            end
        end
    elseif k == K
        X = vec(any(LAge[k,:,:], dims = 1))
        for i=1:np
            if X[i]
                Y = vec(all(LAge[k,:,:] .== LAge[k,:,i], dims = 1))
                if minimum(sP[Y]) == sP[i]
                    global cover = hcat(cover, LAge[k,:,i:i])
                    push!(indices, (k,P[i],[],true))
                    push!(cost, 1)
                end
            end
        end
    else
        Z = repeat(LAge[k,:,:], 1, 1, np) .&& permutedims(repeat(LAle[k,:,:], 1, 1, np), (1, 3, 2))
        X = reshape(permutedims(any(Z, dims = 1), (2,3,1)), np, np)
        for i=1:np
            for j=1:np
                if X[i,j]
                    Yi = vec(all(Z[:,:,j] .== Z[:,i,j], dims = 1))
                    Yj = vec(all(Z[:,i,:] .== Z[:,i,j], dims = 1))
                    if minimum(sP[Yi]) == sP[i] && minimum(sP[Yj]) == sP[j]
                        global cover = hcat(cover, Z[:,i,j])
                        push!(indices, (k,P[i],P[j],true))
                        push!(cost, 1)
                    end
                end
            end
        end
    end
end

for k = 1:K-1
    for i=1:np
        if any(BRle[k,:,i])
            global cover = hcat(cover, BRle[k,:,i])
            push!(indices, (k,P[i],P[i],false))
            push!(cost, n+1)
        end
    end
end

v = length(indices)

model = Model(GLPK.Optimizer)

@variable(model, x[1:v], Bin)

@constraint(model,[i=1:n], sum(x .* cover[i,:]) >= 1)

@objective(model, Min, sum(x[j] * cover[i,j] * cost[j] for i=1:n for j=1:v))

optimize!(model)

r = round.(Int,value.(x)) .== 1
cover[:,r] .* C
indices[r]

for j=1:v
    if r[j]
        X = cover[:,j]
        k, pge, ple, certain = indices[j]
        if certain
            if k == 1
                println("If ",join([join(["c$i"," <= ",maximum(PT[cover[:,j],i])]) for i in ple] ," and ")," then category $k")
            elseif k == K
                println("If ",join([join(["c$i"," >= ",minimum(PT[cover[:,j],i])]) for i in pge] ," and ")," then category $k")
            else
                println("If ",join(vcat([join(["c$i"," <= ",maximum(PT[cover[:,j],i])]) for i in ple],[join(["c$i"," >= ",minimum(PT[cover[:,j],i])]) for i in pge]) ," and ")," then category $k")
            end
        else
            #println("If ",join([join(["c$i"," <= ",maximum(PT[cover[:,j],i])]) for i in ple if all(PT[:,i])] ," and ")," then category ",join(["$l" for l=k:k+1]," or "))
            println("If ",join([join(["c$i"," <= ",maximum(PT[cover[:,j],i])]) for i in ple] ," and ")," then category ",join(["$l" for l=k:k+1]," or "))
        end
    end
end