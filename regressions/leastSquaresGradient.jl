using Printf
include("misc.jl")
include("findMin.jl")

function leastSquaresGradient(X,y)

	(n,d) = size(X)
    @show (n,d)

	# Initial guess
	w = zeros(d,1)
    @show 

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = leastSquaresObj(w,X,y)
    @show funObj(w)

	# This is how you compute the function and gradient:
	(f,g) = funObj(w)

	# Derivative check that the gradient code is correct:
	g2 = numGrad(funObj,w)

	if maximum(abs.(g-g2)) > 1e-4
		@printf("User and numerical derivatives differ:\n")
		@show([g g2])
	else
		@printf("User and numerical derivatives agree\n")
	end

	# Solve least squares problem
	w = findMin(funObj,w)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end

function leastSquaresObj(w,X,y)
	Xw = X*w
    epsilon = 1
    
    r= Xw-y
	matf = (1/2)*(Xw - y).^2
    matf[abs.(r).>epsilon] = epsilon*(abs.(r[abs.(r).>epsilon]).-0.5*epsilon)
    f=sum(matf)
    #f= (1/2)*sum((Xw - y).^2)
    
	matg = X.*(Xw - y)
    matg[abs.(r).>epsilon] = X[abs.(r).>epsilon]*epsilon.*sign.(r[abs.(r).>epsilon])
    g= [sum(matg)][:,:]
    #g= X'*(Xw - y)
    
	return (f,g)
end