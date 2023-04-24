using Printf
using Statistics

# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])



# Fit a least squares model
include("leastSquares.jl")

sigma=10
model = leastSquaresGaussian(X,y,sigma)

# Evaluate training error
yhat = model.predict(X)
trainError = mean((yhat - y).^2)
@printf("Squared train Error with least squares: %.3f for sigma= %.3f\n",trainError,sigma)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean((yhat - ytest).^2)
@printf("Squared test Error with least squares: %.3f for sigma= %.3f\n",testError,sigma)
     
#Plot model
using Plots
scatter(X,y,legend=false,linestyle=:dot)
Xhat = minimum(X):.1:maximum(X)+0.2
yhat = model.predict(Xhat)
plot!(Xhat,yhat,legend=false)
gui()
