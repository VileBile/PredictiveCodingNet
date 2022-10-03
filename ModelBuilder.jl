function funkbands(n,fs,o=0)
    G = []
    push!(G,[1+o:n[1]+o,fs[1]])
    for i in 2:length(n)
        g = sum(n[1:i-1])+o+1:sum(n[1:i-1])+n[i]+o
        push!(G,[g,fs[i]])
    end
    return G
end 

function GenVm(mask)
    Vm = zero(mask) + I
    n = size(mask)[1]
    ols = []
    for i in 1:n
        kids = findall(isone,mask[:,i])
        pars = findall(isone,mask[i,:])
        if length(pars) == 0
            push!(ols,i)
        end
        indis = CartesianIndex.(vec(collect(product(kids,kids))))
        Vm[indis] .= 1
    end
    #indis = CartesianIndex.(vec(collect(product(ols,ols))))
    #Vm[indis] .= 1 
    return Vm, ols
end


function GenVm1(n)
   Vm =  dcat(ones.(n,n)...)
   return Vm
end

function GenFFMask(n::Vector{Int64},sparse = 1)
    mask = Float32.(zeros(sum(n),sum(n)))
    indis = []
    push!(indis,CartesianIndices((n[1]+1:sum(n[1:2]),1:n[1])))
    for i in 2:length(n)-1
        rowbot = sum(n[1:i])+1
        rowtop = sum(n[1:i+1])
        colbot = sum(n[1:i-1])+1
        coltop = sum(n[1:i])
        global indi = CartesianIndices((rowbot:rowtop,colbot:coltop))
        push!(indis,indi)
    end
    indis = collect(Iterators.flatten(indis))
    mask[indis] .= 1
    t = rand(size(mask)[1],size(mask)[1])
    mask[t .< (1-sparse)] .= 0
    return mask
end

function LatLay(n,laynum,mask)
    loc = sum(n[1:laynum-1])+1:sum(n[1:laynum-1])+laynum
    mask[loc,loc] = ones(length(loc),length(loc)) - LinearAlgebra.I
    return mask, loc
end

function AddNode(kids,ols,W,Wmask) 
    Wmrow = zeros(1,size(W)[1])
    Wmrow[ols] .= 1 
    Wmcol = zeros(size(W)[1]+1,1)
    Wmcol[kids] .= 1
    Wmask = [[Wmask; Wmrow] Wmcol]
    Wrow = Wmrow.*(10^-2*rand(-0.03:0.0001:0.03,size(Wmrow)))
    Wcol = Wmcol.*(10^-2*rand(-0.03:0.0001:0.03,size(Wmcol)))
    W = [[W ;Wrow] Wcol]
    return W, Wmask
end

