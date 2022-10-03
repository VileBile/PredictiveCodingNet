using IterTools
using LinearAlgebra 
using Statistics
using MLDatasets
using Plots
using Random
include("afuns.jl"), include("opts.jl"),include("DataPrep.jl")
rng = MersenneTwister(1234)

function funkbands(n,fs,o=0)
    G = []
    push!(G,[1+o:n[1]+o,fs[1]])
    for i in 2:length(n)
        g = sum(n[1:i-1])+o+1:sum(n[1:i-1])+n[i]+o
        push!(G,[g,fs[i]])
    end
    return G
end 


function GenPC(nneur,mask=Float32.(ones(nneur,nneur)-I)) 
    x = Float32.(zeros(nneur))
    W = Float32.(10^-2*(rand(-0.3:0.0001:0.3,(nneur,nneur)).*mask))
    return W, mask
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

function BinocMask(nl,nr,n,lat = true)
    numl = sum(nl)
    numr = sum(nr)
    numt = sum(n)
    mask = Float32.(zeros(numl+numr+numt,numl+numr+numt))
    L = GenFFMask(nl)
    R = GenFFMask(nr)
    T = GenFFMask(n)
    if lat
        for i in 1:length(n)
            T, = LatLay(n,i,T)
        end
    end
    mask[1:numl,1:numl] = L
    mask[numl+1:numl+numr,numl+1:numl+numr] = R
    mask[numl+numr+1:numl+numr+numt,numl+numr+1:numl+numr+numt] = T
    mask[numl+numr+1:numl+numr+n[1],sum(nl[1:end-1])+1:sum(nl[1:end-1])+nl[end]] .= 1
    mask[numl+numr+1:numl+numr+n[1],numl+sum(nr[1:end-1])+1:numl+sum(nr[1:end-1])+nr[end]] .= 1
 
    return mask

end

function AsMask(nneur,p)
    mask = rand(nneur,nneur)
    mask[mask .< (1-p)] .= 0
    mask[mask .> (1-p)] .= 1
    return mask
end


function LatLay(n,laynum,mask)
    loc = sum(n[1:laynum-1])+1:sum(n[1:laynum-1])+laynum
    mask[loc,loc] = ones(length(loc),length(loc)) - LinearAlgebra.I
    return mask, loc
end

# Alghoritm


function validate(W,vs_x,vs_y,T,afun)
    acc = 0
    cer = 0
    fixed_y = last.(vs_y[1])
    setlen = length(vs_x)
    for i in 1:setlen
        e, x, E= cool(vs_x[i],Float32.(zeros(shape(W)[1])),W,T,afun,lx)
        o = exp.(x[fixed_y])
        o = o./sum(o)
        c, pred = firts(findmax(o)), last(findmax(o))
        cer += c
        if fisrt.(vs_y[i])[pred] == 1
            acc += 1
        end
    end
    acc /= setlen
    cer /= setlen
    return acc, cer
end




function cool(fixed,W,T,groups,lx,isfix=true)
    E = []
    X = []
    Dx = []    
    #diagnostitcs
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
        e = x-μ
        push!(E,e)
        dx = (lx*(-e + dfx.*(W'*e)))
        push!(Dx,dx)
        x[activ] += dx[activ]
        h = copy(x)
        push!(X,h)
    end
    return e, x, X, E, Dx
end

function learn(dataset,W,mask,T,groups,lx,lw,epochs,)
    try
        global Es = []
        global gnorms = []
        activ = findall(!iszero,mask)
        m = Float32.(zeros(length(activ)))
        #v = Float32.(zeros(length(activ)))
        for ep in 1:epochs
            shuffle!(dataset)
            global E = []
            for d in dataset
                fx = Float32.(zeros(size(W)[1]))
                dfx = Float32.(zeros(size(W)[1]))
                global e, x, = cool(d,W,T,groups,lx)
                push!(E,e)
                for temp in groups
                    g, afun = temp 
                    fx[g], dfx[g] = afun(x[g])
                end
                gw = (e*fx')[activ] - (10^-2)*W[activ]
                gnorm = norm(gw)
                push!(gnorms,gnorm)
                m = momentum(gw,m,0.4)
                W[activ] += lw*m #
                print("'")
            end
        errs = [sqrt(sum(e.^2)/length(e)) for e in E]
        e = mean(errs) 
        print("epoch: "*string(ep)*"  Err:  "*string(e)*"  ")    
        end
        return W, E, gnorms
    catch err
        print(err)
        return W, E, gnorms
    end
end

n = [784,512,10]
nneur = sum(n)
mask = GenFFMask(n)'
W, mask = GenPC(nneur,mask)
TS = GenMNISTset(nneur,tr_x,tr_y_onehot)
ts = [TS[0][1:5];TS[1][1:5];TS[9][1:5];TS[8][1:5]]
#ts1 = [TS[0][1:5];TS[1][1:5];TS[9][1:5];TS[6][1:5]]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 = learn(tsB,Float32.(zeros(size(W)[1])),W,mask,50,hardtanh,0.1,10^-2,300) 
shuffle!(ts)
#shuffle!(ts1)
T = 100
epochs = 100
afuns = [hardtanh,hardtanh,hardtanh,hardtanh]
fbs = funkbands(n,afuns)
W, Es, gnorms = learn(ts,W,mask,50,fbs,0.1,10^-4,300) 



#=
n = [784,512]
nt = [256,10]
nneur = sum([n;n;nt])
TSb = GenBinocMNISTset(n,n,nt,tr_x,tr_y_onehot)
tsB = [TSb[0][1:3];TSb[1][1:3]]
#Rivs = RivSet(TSb[0][50:100],TSb[1][50:100])
#=T0 = RivSet(TSb[0][1:1],TSb[0][1:1])
T1 = RivSet(TSb[1][1:1],TSb[1][1:1])
Ts0 = zip([first.(T0[1]);[1,0]],[last.(T0[1]);[nneur-1,nneur]])
Ts1 = zip([first.(T1[1]);[0,1]],[last.(T1[1]);[nneur-1,nneur]])
TSb2 = [Ts0;Ts1]
t = [T0[1:1];T1[1:1]]=#
mask = BinocMask(n,n,nt,true) 
W, mask = GenPC(nneur,mask) 
fs = [hardtanh,hardtanh]
fst = [hardtanh,kWTstrict]
fs1 = funkbands(n,fs)
fs2 = funkbands(n,fs,sum(n))
fs3 = funkbands(nt,fst,2*sum(n))
fbs = [fs1;fs2;fs3]
W, Es, gnorms = learn(tsB,W,mask,50,fbs,0.1,10^-2,300) 
=#
#=
IMS = []
for x in X 
im1 = x[1:784]
im2 = x[sum(n)+1:sum(n)+784]
im = [reshape(im1,28,28) reshape(im2,28,28)]
push!(IMS,im)
=#


#im2 = x[sum(n)+1:sum(n)+784]

#=
## Gratings
n = [121,10]
nt = [5,2]
nneur = sum([n;n;nt])
a = zeros(11,11)
for i in 1:2:11
    a[:,i] .= 1
end
b = transpose(a)
oh_a = [1,0]
oh_b = [0,1]
xs = cat(a,b,dims=3)
ys = [oh_a,oh_b]
S = GenBinocMNISTset(n,n,nt,xs,ys)
Ts = [S[0][1:1];S[1][1:1]]
t0 = RivSet(S[0][1:1],S[0][1:1],n,2)[1]
t1 = RivSet(S[1][1:1],S[1][1:1],n,2)[1]
r01 = RivSet(S[0][1:1],S[1][1:1],n,2)[1] 
r10 = RivSet(S[1][1:1],S[0][1:1],n,2)[1] 

fb = [hardtanh,hardtanh]
ft = [hardtanh,hardtanh]
fl = funkbands(n,fb)
fr = funkbands(n,fb,sum(n))
f = funkbands(nt,ft,sum(n)*2)
fbs = [fl;fr;f]

mask = BinocMask(n,n,nt,true)
W, mask = GenPC(sum([n;n;nt]),mask)
W, Es, gnorms = learn(Ts,W,mask,50,fbs,1,10^-1,300)

=#



