using Flux
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
using CUDA
include("ModelBuilder.jl")

n = [784,100,10]
nneur = sum(n)

n = [784,10]
nneur = sum(n)
ols = nneur-(n[end]-1):nneur




Wmask = GenFFMask(n)'
ow = 0.003
W = 10^-2*(Wmask.*(rand(-ow:ow/1000:ow,size(Wmask))))

Vmask = GenVm1(n)
Vmask[ols,ols] .= 0
#Vmask[1:784,1:784] .= 0
V = zero(Vmask) + I

P = V^-1
e = W*rand(nneur) 

err(P,x,m) = P*(x-m)
derr(P,x,m) = gardient(err,P,x,m)
F(e::Vector{Float64},P::Matrix{Float64}) = e'*P*e + log(abs(det(P)))
dF(e,P)= gradient(F,e,P)

#=
function learn(eps,D,W,Wmask,V,Vmask,lw,lv,(lx,T))
    for ep in eps
        shuffle!(D)
        for d in D
        end
    end

=#