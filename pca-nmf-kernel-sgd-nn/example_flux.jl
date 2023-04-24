# Load X and y variable
using JLD, Printf
data = load("mnist35.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])
y[y.==2] .= -1
ytest[ytest.==2] .= -1
(n,d) = size(X)

# Pick a random training example where we will compute NLL and gradient
i = rand(1:n)

# Compute the squared loss for a neural network with 1 hidden layer of 3 units
W = randn(3,d)
v = randn(3)
loss(x,y,W,v) = (1/2)*(v'*tanh.(W*x)-y)^2
f = loss(X[i,:],y[i],W,v)

# Compute loss and gradient using manually-written code from previous assignment
include("NeuralNet.jl")
nHidden  = [3]
w = [W[:];v[:]]
(f_manual,g_manual) = NeuralNet_backprop(w,X[i,:],y[i],nHidden)
gW_manual = reshape(g_manual[1:3*d],3,d) # Gradient with respect to weights in first layer
gv_manual = g_manual[3*d+1:end] # Gradient with respect to weights in seond layer

# Compute the gradient using Flux's automatic differentiation
using Flux
g_AD = gradient(loss,X[i,:],y[i],W,v) # Returns gradient of function 'loss' with respect to each argument
gW_AD = g_AD[3]
gv_AD = g_AD[4]

# Re-writing the objective using Flux's "Chain" and "Flux.params" functions
model = Chain(x -> W*x,z -> tanh.(z), a -> v'a)
loss2(x,y) = (1/2)*(model(x)-y)^2
f_chain = loss2(X[i,:],y[i])
g_chain = gradient(()->loss2(X[i,:],y[i]),Flux.params([W,v])) # Using the "no argument" function ()->loss2(...) "delays" executing the loss so AD can do its work
gW_chain = g_chain[W]
gv_chain = g_chain[v]

# An alternate syntax that Flux supports
g_chain2 = gradient(Flux.params([W,v])) do
	loss2(X[i,:],y[i])
end
gW_chain2 = g_chain2[W]
gv_chain2 = g_chain2[v]

# Re-writing the objective using Flux's pre-defined layer function
vt = reshape(v,1,3)
model2 = Chain(Dense(W,false,tanh),Dense(vt,false,identity))
loss3(x,y) = (1/2)*(model2(x)[1]-y)^2
f_layer = loss3(X[i,:],y[i])
g_layer = gradient(Flux.params(model2)) do
	loss3(X[i,:],y[i])
end
gW_layer = g_layer[Flux.params(model2)[1]]
gv_layer = g_layer[Flux.params(model2)[2]]


maxIter = 10000
stepSize = 1e-4
opt = Descent(stepSize) # For using the update! function
for t in 1:maxIter
	local i = rand(1:n)

	if false # Manual gradient code
		(~,g) = NeuralNet_backprop(w,X[i,:],y[i],nHidden)
		gW = reshape(g[1:3*d],3,d) # Gradient with respect to weights in first layer
		gv = g[3*d+1:end] # Gradient with respect to weights in seond layer
		global w = w - stepSize*g
	elseif true # Use AD
		g = gradient(loss,X[i,:],y[i],W,v)
		global W -= stepSize*g[3]
		global v -= stepSize*g[4]
	elseif false # Implicit parameter version (slow, not sure why)
		g = gradient(()->loss2(X[i,:],y[i]),Flux.params([W,v]))
		global W -= stepSize*g[W]
		global v -= stepSize*g[v]
	elseif false # Implicit parameter version (also slow)
		g = gradient(Flux.params([W,v])) do
			loss2(X[i,:],y[i])
		end
		global W -= stepSize*g[W]
		global v -= stepSize*g[v]
	elseif false # Using Flux's built-in layers (also slow!)
		g = gradient(Flux.params(model2)) do
			loss3(X[i,:],y[i])
		end
		for p in Flux.params(model2)
			Flux.update!(p,stepSize*g[p])
		end
	else # Version without the for loop
		g = gradient(Flux.params(model2)) do
			loss3(X[i,:],y[i])
		end
		Flux.Optimise.update!(opt,Flux.params(model2),g)
	end
end

# You can also import and use the "update!" function,
# to remove the "for" loop above and update all parameters with
# a selected optimizer
# see here: https://fluxml.ai/Flux.jl/stable/training/optimisers/
