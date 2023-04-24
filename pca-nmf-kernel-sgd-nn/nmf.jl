using Printf
using Statistics
using LinearAlgebra
include("misc.jl")
include("findMin.jl")

function nmf(X,k)
    (n,d) = size(X)

    mu = mean(X, dims=1)

    W = randn(k,d)
    Z = randn(n,k)

    W[W.<0].=0
    Z[Z.<0].=0

    R = Z*W - X
    f = sum(R.^2)
    funObjZ(z) = pcaObjZ(z,X,W)
    funObjW(w) = pcaObjW(w,X,Z)
    for iter in 1:50
        fold = f

        # Update Z
        Z[:] = findMinNN(funObjZ,Z[:], verbose=false,maxIter=10)

        # Update W
        W[:] = findMinNN(funObjW,W[:], verbose=false,maxIter=10)

        R = Z*W - X
        f = (1/2)sum(R.^2)

        if (fold - f) / length(X) < 1e-2
            break
        end
    end