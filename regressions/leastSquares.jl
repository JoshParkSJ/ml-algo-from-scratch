include("misc.jl")

function leastSquares(X,y)

	# Find regression weights minimizing squared error
	w = (X'*X)\(X'*y)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end

function leastSquaresBias(X,y)

    n = size(X,1)
    wo = ones(n)
    X = [X wo]
	# Find regression weights minimizing squared error
	w = (X'X)\(X'y)

	# Make linear prediction function
	predict(Xhat) = [Xhat wo]*w

	# Return model
	return GenericModel(predict)
end

function leastSquaresBasis(X,y,p)

    n = size(X,1)
    
    Z = polyBasis(X,p)
	# Find regression weights minimizing squared error
	w = (Z'Z)\(Z'y)

	# Make linear prediction function
	predict(Xhat) = polyBasis(Xhat,p)*w

	# Return model
	return GenericModel(predict)
end

function polyBasis(X,p)

    n = size(X,1)
	Z = ones(n)
    
    if(p==0)
        return Z
    end
    
    for i in 1:p
        addMatrix = X.^i
        Z = [Z addMatrix]
    end 
	

	# Return matrix
	return Z
end

function leastSquaresGaussian(X,y,sigma)

    n = size(X,1)
    
    Z = gaussianBasis(X, X, sigma)
	# Find regression weights minimizing squared error
	w = (Z'Z)\(Z'y)

	# Make linear prediction function
	predict(Xhat) = gaussianBasis(X,Xhat,sigma)*w

	# Return model
	return GenericModel(predict)
end

function gaussianBasis(X, Xhat,sigma)

    n = size(X,1)
    Z = ones(n)
    
    for i in 1:n
        addMatrix = exp.((-(Xhat.-X[i]).^2)/(2*sigma^2))
        
        if(i==1)
            Z = addMatrix
        else
            Z = [Z addMatrix]
        end 
    end 
	

	# Return matrix
	return Z
end

function weightedLeastSquares(X,y,v)

	# Find regression weights minimizing squared error
	w = (X'*v*X)\(X'*v*y)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end