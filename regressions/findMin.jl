using Printf
using LinearAlgebra

function findMin(funObj,w,maxIter=100,epsilon=1e-2)
	# funObj: function that returns (objective,gradient)
	# w: initial guess
	# maxIter: maximum number of iterations
	# epsilon: stop if the gradient gets below this

	# Evalluate the intial objective and gradient
	(f,g) = funObj(w)

	# Set initial step-size so update has an L1-norm at most 1
	alpha = min(1,1/norm(g,1))

	for i in 1:maxIter

		# Try out the current step-size
		wNew = w - alpha*g
		(fNew,gNew) = funObj(wNew)

		# Decrease the step-size if we increased the function
		while fNew > f
			alpha /= 2

			# Try out the smaller step-size
			wNew = w - alpha*g
			(fNew,gNew) = funObj(wNew)
		end

		# Accept the new parameters/function/gradient
		w = wNew
		f = fNew
		g = gNew

		# Print out some diagnostics
		gradNorm = norm(g,Inf)
		@printf("%6d %15.5e %15.5e %15.5e\n",i,alpha,f,gradNorm)

		# We want to stop if the gradient is really small
		if gradNorm < epsilon
			@printf("Problem solved up to optimality tolerance\n")
			return w
		end
	end
@printf("Reached maximum number of iterations\n")
return w
end
