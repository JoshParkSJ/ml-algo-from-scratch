# Define a "model" type, that just needs a predict function
mutable struct GenericModel
	predict # Function that makes predictions
end

# Function to compute the mode of a vector
function mode(x)
	# Returns mode of x
	# if there are multiple modes, returns the smallest
	x = sort(x[:]);
	
	commonVal = [];
	commonFreq = 0;
	x_prev = NaN;
	freq = 0;
	for i in 1:length(x)
		if(x[i] == x_prev)
			freq += 1;
		else
			freq = 1;
		end
		if(freq > commonFreq)
			commonFreq = freq;
			commonVal = x[i];
		end
		x_prev = x[i];
	end
	return commonVal
end

# Return element-wise log, but set log(0)=0
function log0(x)
	y = copy(x)
	y[y.==0] .= 1
	return log.(y)
end

# Return squared Euclidean distance all pairs of rows in X1 and X2
function distancesSquared(X1,X2)
	(n,d) = size(X1)
	(t,d2) = size(X2)
	@assert(d==d2)
	return X1.^2*ones(d,t) .+ ones(n,d)*(X2').^2 .- 2X1*X2'
end
