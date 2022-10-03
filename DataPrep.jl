using MLDatasets 
using Statistics

tr_x, tr_y = MNIST(split=:train)[:]
tr_x = Float32.(tr_x)
tr_y = Float32.(tr_y)
tr_y_onehot = []
for i in tr_y
    a = Float32.(zeros(10))
    a[Int(i)+1] = 1
    push!(tr_y_onehot,a)
end
te_x, te_y = MNIST.(split=:test)[:]

trainset = []


ctr_x, ctr_y = CIFAR10(split=:train)[:]
ctr_x = Float32.(ctr_x)
ctr_y = Float32.(ctr_y)
ctr_y_onehot = []
for i in ctr_y
    a = Float32.(zeros(10))
    a[Int(i)+1] = 1
    push!(ctr_y_onehot,a)
end
Cte_x, Cte_y = CIFAR10(split=:test)[:]

c_tr_x = zeros(size(ctr_x)[1],size(ctr_x)[2],size(ctr_x)[4])
for i in 1:size(ctr_x)[4]
    c_tr_x[:,:,i] = 0.299*ctr_x[:,:,1,i] + 0.587*ctr_x[:,:,2,i] +0.114*ctr_x[:,:,3,i]
end

function GenMNISTset(nneur,s_x,s_y,valid=false)
    TS = Dict(0=>[],1=>[],2=>[],3=>[],4=>[],5=>[],6=>[],7=>[],8=>[],9=>[])
    if valid
    end
    x_len = length(vec(s_x[:,:,1]))
    y_len = length(s_y[1])
    for i in 1:length(s_y)
    s = [vec(s_x[:,:,i]);s_y[i]]
    indi = vcat(collect(1:x_len),collect(nneur-y_len+1:nneur))
    d = zip(s,indi)
    push!(TS[findall(isone,s_y[i])[1]-1],d)
    end 
    return TS
end

function GenMNISTsetUnsup(s_x,s_y)
    TS = Dict(0=>[],1=>[],2=>[],3=>[],4=>[],5=>[],6=>[],7=>[],8=>[],9=>[])
    x_len = length(vec(s_x[:,:,1]))
    for i in 1:length(s_y)
    s = vec(s_x[:,:,i])
    indi = collect(1:x_len)
    d = zip(s,indi)
    push!(TS[findall(isone,s_y[i])[1]-1],d)
    end 
    return TS
end

function GenMNISTsetValid(s_x,s_y)
    TS = Dict(0=>[],1=>[],2=>[],3=>[],4=>[],5=>[],6=>[],7=>[],8=>[],9=>[])
    x_len = length(vec(s_x[:,:,1]))
    for i in 1:length(s_y)
    s = vec(s_x[:,:,i])
    indi = collect(1:x_len)
    t = zip(s,indi)
    d = [t,s_y[i]]
    push!(TS[findall(isone,s_y[i])[1]-1],d)
    end 
    return TS
end


function  GenBinocMNISTset(nl,nr,n,s_x,s_y)
    nneur = sum([nl;nr;n])
    numl = sum(nl)
    TS = Dict(0=>[],1=>[],2=>[],3=>[],4=>[],5=>[],6=>[],7=>[],8=>[],9=>[])
    x_len = length(vec(s_x[:,:,1]))
    y_len = length(s_y[1])
    for i in 1:length(s_y)
    s = [vec(s_x[:,:,i]);vec(s_x[:,:,i]);s_y[i]]
    indi = vcat(collect(1:x_len),collect(numl+1:numl+nr[1]),collect(nneur-y_len+1:nneur))
    d = zip(s,indi)
    push!(TS[findall(isone,s_y[i])[1]-1],d)
    end 
    return TS

    
end

function RivSet(Ls,Rs,n,osize)
    Rivs = []
    indis = last.(Ls[1])[1:end-osize]
    for i in 1:length(Ls)
        l, r = first.(Ls[i]), first.(Rs[i])
        s = [l[1:n[1]];r[1:n[1]]]
        S = zip(s,indis)
        push!(Rivs,S)
    end
    return Rivs
end

#=
for i in 1:size(tr_x)[3]
    tr_x[:,:,i] =  (tr_x[:,:,i].-mean( tr_x[:,:,i]))./(sqrt(var( tr_x[:,:,i])))
end
=# 

#tr_x = (tr_x.-mean(tr_x))/std(tr_x)