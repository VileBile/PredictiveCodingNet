function momentum(g,m,a=0.1)
    m = (1-a)*g + a*m
    return m
end

function Adam(g,m = Float32.(zeros(size(g))),v = Float32.(zeros(size(g))),bm =0.9,bv=0.999,ϵ = 10^-3)
    m = bm*m +(1-bm)*g
    v = bv*v + (1-bv)*(g.^2)
    delW = m./(sqrt.(v).+ϵ)
    return delW
end

