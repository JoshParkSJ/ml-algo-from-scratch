using Printf

# Load X and y variable
using JLD
using Plots
data = load("basisData.jld")
(X,y) = (data["X"],data["y"])
(n,d) = size(X)

# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
nHidden = [3]
nParams = NeuralNet_nParams(d,nHidden)
w = randn(nParams,1)

# Train with stochastic gradient
maxIter = 10000
stepSize = 1e-4
for t in 1:maxIter

	# The stochastic gradient update:
	i = rand(1:n)
	(f,g) = NeuralNet_backprop(w,X[i,:],y[i],nHidden)
	global w = w - stepSize*g

	# Every few iterations, plot the data/model:
	if (mod(t-1,round(maxIter/50)) == 0)
		@printf("Training iteration = %d\n",t-1)
		xVals = -10:.05:10
		Xhat = zeros(length(xVals),1)
		Xhat[:] .= xVals
		yhat = NeuralNet_predict(w,Xhat,nHidden)
		scatter(X,y,legend=false,linestyle=:dot)
		plot!(Xhat,yhat,legend=false)
		gui()
		sleep(.1)
	end
end
plot!()
