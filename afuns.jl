function RELU(z)
    fz = max.(0,z)
    indis = findall(!iszero,fz)
    dfz = zero(fz)
    dfz[indis] .= 1
    return fz, dfz
end

function sig(z::Real)
    s = 1/(1+exp(-z))
    return s, (1-s)*s
end

function swish(z,α=1)
    sw = z./(exp.(-z).+1) 
    return sw, sw + ((exp.(-z).+1).^-1).*-(sw.+1)
    
end

function hardtanh(x)
    fx = copy(x)
    dfx = convert(typeof(x),ones(length(x)) .+ 0.1)
    fx[x.>1] .= 1  
    dfx[x.>1] .= .1
    fx[x.<-1] .= -1
    dfx[x.<-1] .= .1 
    return fx, dfx
end


function kWT(x,s=0.2,f=hardtanh)
    ffx,dffx = f(x)
    fx = zeros(length(x))
    dfx = zeros(length(x))
    num = maximum((Int(floor(length(x)*s)),1))
    indis = reverse(sortperm(x))[1:num]
    fx[indis] = ffx[indis]
    dfx[indis] = dffx[indis]
    return fx, dfx
end

function kWTstrict(x,k=1,f=id)
    ffx,dffx = f(x)
    fx = zeros(length(x))
    dfx = zeros(length(x))
    indis = reverse(sortperm(x))[1:k]
    fx[indis] = ffx[indis]
    dfx[indis] = dffx[indis]
    return fx, dfx
end

function id(z)
    return z, convert(typeof(z),ones(length(z)))
    
end