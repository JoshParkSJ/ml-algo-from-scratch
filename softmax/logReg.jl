using Printf
include("misc.jl")
include("findMin.jl")

function logReg(X,y)

	(n,d) = size(X)
    @show (n,d)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = logisticObj(w,X,y)

	# Solve least squares problem
	w = findMin(funObj,w,derivativeCheck=true)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end

function logisticObj(w,X,y)
	yXw = y.*(X*w)
	f = sum(log.(1 .+ exp.(-yXw)))
	g = -X'*(y./(1 .+ exp.(yXw)))
    @show size(g)
	return (f,g)
end

function logRegL1(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = logisticObj(w,X,y)

	# Solve least squares problem
	w = findMinL1(funObj,w,1.0)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end

function logRegL2(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObjL2(w) = logisticObjL2(w,X,y)

	# Solve least squares problem
	w = findMin(funObjL2,w,derivativeCheck=true)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end

function logisticObjL2(w,X,y)
	yXw = y.*(X*w)
    lambda = 1.0
	f = sum(log.(1 .+ exp.(-yXw))) + (lambda/2 *(w'*w))[1]
	g = -X'*(y./(1 .+ exp.(yXw))) .+ lambda*w
	return (f,g)
end

# Variant where we use forward selection for feature selection
function logRegL0(X,y,lambda)
	(n,d) = size(X)

	# Define an objective that will operate on a subset of the data called Xs
	funObj(w) = logisticObj(w,Xs,y)

	# Start out just using the bias variable (assumed to be in first column),
	# and record 'score' which is the loss plus regularizer
	S = [1] # Candidate set of features
	Xs = X[:,S]
	w = zeros(length(S),1)
	w = findMin(funObj,w,verbose=false)
	(f,~) = funObj(w)
	score = f + lambda*length(S)
	minScore = score # Lowest score we've found
	minS = S # Best set of features we've found

	@show(minScore)
	@show(minS)

	# Greedily start adding the variable that improves the score the most
	oldScore = Inf
	while minScore != oldScore
		oldScore = minScore

		# Print out the variables we've selected so far
		@printf("Current set of selected variables (score = %f):\n",minScore)
		for j in 1:length(S)
			@printf("%d ",S[j])
		end
		@printf("\n")

		for j in setdiff(1:d,S)
			# Fit the model with 'j' added to the feature set 'S'
			# then compute the score and update 'minScore' and 'minS'
			Sj = [S;j]
			Xs = X[:,Sj]

			# PUT YOUR CODE HERE
            w = zeros(length(Sj),1)
            w = findMin(funObj,w,verbose=false)
            (f,~) = funObj(w)
            score = f + lambda*length(Sj)
            
            if score<minScore
                minScore = score
                minS = Sj
            end    
		end
		S = minS
	end

	# Construct final 'w' vector
	w = zeros(d,1)
	S = minS
	Xs = X[:,S]
	w[S] = findMin(funObj,zeros(length(S),1),verbose=false)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end


# Multi-class one-vs-all version (assumes y_i in {1,2,...,k})
function logRegOnevsAll(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each row of 'w' will be a logistic regression classifier
	W = zeros(k,d)

	for c in 1:k
		yc = ones(n,1) # Treat class 'c' as +1
		yc[y .!= c] .= -1 # Treat other classes as -1

		# Each binary objective has the same features but different lables
		funObj(w) = logisticObj(w,X,yc)

		W[c,:] = findMin(funObj,W[c,:],verbose=false)
	end

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W',dims=2)

	return LinearModel(predict,W)
end

function softmaxClassifier(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each row of 'w' will be a logistic regression classifier
	W = zeros(k,d)
    @show size(W)

	for c in 1:k
		yc = ones(n,1) # Treat class 'c' as +1
		yc[y .!= c] .= 0 # Treat other classes as 0

		# Each binary objective has the same features but different lables
		funObj(w) = softmaxObj(w,X,yc,c,k,d)
        w= reshape(W', (k*d,1))
        @show size(w)

		W[c,:] = findMin(funObj,w, verbose = false)
	end

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W',dims=2)

	return LinearModel(predict,W)
end

function softmaxObj(w,X,y,c,k,d)
    
    w = (reshape(w, (d,k)))'
	(n,d) = size(X)
    f=0;
    g= zeros(d,1)
    for i in 1:n
        f+= -(w[c,:])'*X[i,:]+log(sum(exp.(w*X[i,:])))
        g.+= -y[i].*X[i,:].+ (exp((w[c,:])'*X[i,:]))/(sum(exp.(w*X[i,:]))).*X[i,:]
    end
	return (f,g)
end

