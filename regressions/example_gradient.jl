using Printf
using Statistics

# Load X and y variable
using JLD
data = load("outliersData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a least squares model
include("leastSquaresGradient.jl")
model = leastSquaresGradient(X,y)

# Evaluate training error
yhat = model.predict(X)
trainError = mean((yhat - y).^2)
@printf("Squared train Error with least squares: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean((yhat - ytest).^2)
@printf("Squared test Error with least squares: %.3f\n",testError)

# Plot model
using Plots
scatter(X,y,legend=false,linestyle=:dot)
Xhat = minimum(X):.01:maximum(X)
yhat = model.predict(Xhat)
plot!(Xhat,yhat,legend=false)
gui()
