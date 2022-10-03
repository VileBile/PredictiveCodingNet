using IterTools
using LinearAlgebra 
using Statistics
using MLDatasets 
using Plots
using Random
using MultivariateStats
using Distributions
using SimpleTools
using SparseArrays
using Flux
using LightGraphs

include("afuns.jl"), include("opts.jl"),include("DataPrep.jl"),include("ModelBuilder.jl")
rng = MersenneTwister(1234)


# Alghoritm

F(x,fx,W,P) = (transpose(x-W*fx))*P*(x-W*fx) - log(det(P)) +0.5*norm(W) +0.5*norm(P) 
dF(x,fx,W,P) = gradient(F,x,fx,W,P)

function validate(ValidSet,ols,W,P,T,fbs,lx)
    acc = 0
    cer = 0
    err = 0
    normnum = length(ValidSet)
    for d in ValidSet
        i, t = d
        x,X,E = cool(i,W,P,T,fbs,lx)
        err += sum(E[end].^2)/length(E[end])
        o = softmax(x[ols])
        c, pred = findmax(o)
        cer += c
        if t[pred] == 1
            acc += 1
        end
    end
    acc = acc/normnum
    cer = cer/normnum
    err = err/normnum
    return acc, cer, err
end




function cool(fixed,W,P,T,groups,lx,isfix=true)
    E = []
    X = [] #diagnostitcs
    s, i = first.(fixed), last.(fixed)
    x = Float32.(zeros(size(W)[1]))
    x[i] = s
    if !isfix
        i = []
    end
    activ = [j for j in 1:length(x) if !(j in i)]
    e = Float32.(zero(x))
    μ = Float32.(zero(x))
    for t in 1:T
        fx = Float32.(zeros(length(x)))
        dfx = Float32.(zeros(length(x)))
        for temp in groups
            g, afun = temp
            fx[g], dfx[g] = afun(x[g])
        end
        mul!(μ,W,fx)
        e = P*(x-μ)
        push!(E,e)
        dx = (lx*(-e + dfx.*(W'*e)))
        x[activ] += dx[activ]
        h = copy(x)
        push!(X,h)
    end
    return x,X,E
end

function pruneNgrow(P,Pmask,fbs,num,tresh)
    #lW = copy(W)
    #lWmask = copy(Wmask)
    lP = copy(P)
    lP = .5*(lP+lP')
    lPmask = copy(Pmask)
    absP = copy(abs.(P))
    #absW = copy(abs.(W))
    for node in 1:size(absP)[1]
            pro = absP[node,:]
            #wro = absW[i,:]
            pis = sortperm(pro)
            #wis = sortperm(wro)
            smolp = findfirst(!iszero,pro[pis])
            #smolw = findfirst(!iszero,wro[wis])
            if !isnothing(smolp) && smolp != length(pro)
                deadps = pis[smolp:min(smolp+(num-1),length(pis))]
                lP[node,deadps] .= 0
                lPmask[node,deadps] .= 0
            end
            #=if !isnothing(smolw)
                deadws = wis[smolw:smolw+(num-1)]
                lW[i,deadws] .= 0
                lWmask[i,deadws] .= 0
            end=#
        end
        #=
        global groups = []
        components = connected_components(SimpleGraph(Pmask[clust,clust]))
        if length(components) > 1 && minimum(length.(components)) > 20
            print("New partition in $clust")
            push!(groups,zip(components,clust))
        end
        =#
        lP[lP.!=lP'] .= 0
        lPmask[lPmask.!=lPmask'] .= 0
    end
    return lP, lPmask
end

function BCor(Buds,D)
    BudCor = zeros(length(Buds),length(Buds))
    for i1 in 1:length(Buds)
        for i2 in 1:length(Buds)
            BudCor[i1,i2] = mean((D[Buds[i1],Buds[i2]]))
        end
    end
    BudCor[diagind(BudCor)] .= 0
    return BudCor
end

function MakeBuds(bg)
    temp = zeros(length(bg),length(bg))
    temp = one(temp)
    Buds = [Bool.(temp[i,:]) for i in 1:length(bg)]
end

function e_merge(Data,meps,eps,fbs,growtresh,W,P,Wmask,Pmask)
    try 
        for _ in 1:meps
            bg = fbs[end-1][1]
            Buds = MakeBuds(bg)
            nneur = size(P)[1]
            numkids = 0
            while length(Buds) > 0
                #W, P, = learn(S,Sv,params)
                D = abs.(P[bg,bg])
                BudCor = BCor(Buds,D)
                BudsMask = zero(BudCor)
                BudsMask[0 .!= BudCor] .= 1
                BudCor, BudsMask = pruneNgrow(Buds,Budmask,1)
                li = rand(1:length(Buds))
                locus = Buds[li]
                closei = findmax(BudCor[li,:])[2]
                closeone = Buds[closei]
                try
                    global NuBuds = Bool.(locus+closeone)
                catch e
                    return locus, closeone, Buds
                end
                filter!(x-> !(x  in [locus,closeone]),Buds)            
                
                if sum(NuBuds) >= growtresh
                    print("Birth!!")
                    numkids += 1
                    #Growing W
                    tl = zeros(bg[1]-1) ; hd = zeros(nneur-(bg[end])) ;nucol = [tl;Int.(NuBuds);hd]
                    nuro = zeros(nneur+1)' ; nuro[ols] .= 1
                    Wmask = [[Wmask nucol];nuro]
                    W = [[W rand(-1e-1:1e-2:1e-1,size(nucol)).*nucol]; rand(-1e-1:1e-2:1e-1,size(nuro)).*nuro]
                    #Growing P
                    nuPmcol = [zeros(nneur-(numkids-1));ones(numkids-1)] ;nuPmro = [zeros((nneur+1)-numkids);ones(numkids)]'
                    Pmask = [[Pmask nuPmcol];nuPmro]
                    P = [[P zeros(nneur)];[zeros(nneur);1]']
                    
                    nneur += 1
                    #Giving new node activation function
                    if numkids == 1
                        insert!(fbs,length(fbs),[nneur:nneur, hardtanh])
                    else
                        fbs[end-1][1] = nneur-(numkids-1):nneur
                    end
                else
                    push!(Buds,NuBuds)
                end

                if length(Buds) == 1
                    print("done with oprhans")
                    break
                end
                
            end
        end
    catch err
        return Buds
    end
        
    return W, P, Wmask, Pmask, fbs
end



function learn(dataset,validset,ols,W,Wmask,P,Pmask,T,groups,lx,lw,lp,a,epochs,firstprune,prunevery)
    #try
        global Es = []
        global mdP = zeros(size(P))
        global mdW = zeros(size(W))
        vP = zero(P)
        vW = zero(W)
        BestModel = (copy((W)),copy(P))
        BestAcc = 0
        for ep in 1:epochs
            shuffle!(dataset)
            E = []
            ngW = []
            ngP = []
            for d in dataset
                s, i = first.(d), last.(d) 
                s[1:784] += 1e-1rand(-1:1e-2:1,784)
                d = zip(s,i)
                #push!(Ps,P)
                global x,= cool(d,W,P,T,groups,lx)
                fx, dfx = zero(x), zero(x)
                for temp in groups
                    g, afun = temp 
                    fx[g], dfx[g] = afun(x[g])
                end
                if det(P) <= 0
                    print("Neg det!")
                    return W, P, Es, BestModel
                end
                push!(E,F(x,fx,W,P))
                dx, dfx, dW, dP = dF(x,fx,W,P)
                normdP = norm(dP)
                normdW = norm(dW)
                push!(ngW,normdW)
                push!(ngP,normdP)
                if any(isnan, dP)
                    print("NaN in dP, abort!")
                    return W, P, Es, BestModel
                end
                if any(isnan, dW)
                    print("NaN in dW, abort!")
                    return W, P, Es, BestModel
                end
                #=
                if normdP > cP
                    dP = cP*(dP/normdP)
                    print("clipped dP")
                end
                if normdW > cW
                    dW = cW*(dW/normdW)
                    print("clipped dW")
                end
                =#
                vP = momentum(dP,vP,a)
                vW = momentum(dW,vW,a)
                W -= Wmask.*(lw*vW) #+ 10^-3*W
                P -= Pmask.*(lp*vP) 
                P = (1/2)*(P+P')
                print("'")
            end
            acc, cer, validerr = validate(validset,ols,W,P,T,fbs,lx)
            if  acc >= BestAcc
                BestAcc = acc
                print("\nNew *\n")
                global BestModel = (copy((W)),copy(P))
            end
            
            e = sum(E)/length(E) 
            ngw = mean(ngW)
            ngp = mean(ngP)
            push!(Es,e)   
            print("\nepoch: "*string(ep)*"\nErr:  "*string(e)*"  \n") 
            print("ngw= $ngw")
            print("\n ngp=$ngp\n")
            print("detP:"*string(det(P))*"\n")
            print("validation:"*"cer = "*string(cer)*" acc = "*string(acc)*" err = "*string(validerr)*"\n")
            if ep>=firstprune    
                if  ep%prunevery == 0
                    print("is murder time\n")
                    global P, Pmask, components = pruneNgrow(P,Pmask,fbs,20,20)
                    if !isempty(components)
                        print("kids")
                        return W, P, Es, components
                    end
                end 
            end
        
        end
        
    #=catch err
        print(err)
        return W, P, Es, BestModel
    end=#
    return W, P, Es, BestModel
end






#=
n = [3,1]

D = []
dist = MvNormal([2,3,4],zeros(3,3)+I)
indis = [1,2,3]
for i in 1:1000
    p = rand(dist)
    d = zip(p,indis)
    push!(D,d)
end
=#


#=
#Full Net
n = 1500
nneur = 1500
Wmask = ones(n,n) - I
Pmask = copy(Wmask)
ow = 0.003
W = (Wmask.*(rand(-ow:ow/1000:ow,size(Wmask))))
P = zeros(n,n) + I 
ols = nneur-9:nneur
=#




#shuffle!(tsu)
T = 100
epochs = 100


#Inverted FF net
n = [28*28,10]
nneur = sum(n)
ols = nneur-(10-1):nneur

afuns = [hardtanh,hardtanh,id]
fbs = funkbands(n,afuns)

TS = GenMNISTset(nneur,c_tr_x,ctr_y_onehot)
VS = GenMNISTsetValid(c_tr_x,ctr_y_onehot)
#TSu = GenMNISTsetUnsup(tr_x,tr_y_onehot)
ts = [TS[0][1:10];TS[1][1:10];TS[9][1:10]]#;TS[6][1:15]]
#tsu = [TSu[0][30:40];TSu[1][30:40];TS[9][30:40]]
vs = [VS[0][30:50];VS[1][30:50];VS[9][30:50]]#;VS[6][30:40]]
shuffle!(ts)




#W, P, Es, BestModel= learn(ts,vs,ols,Float32.(W),Float32.(Wmask),Float32.(P),Float32.(Pmask),100,fbs,1e-1,1e-3,1e-5,0.7,100,1,1) 




