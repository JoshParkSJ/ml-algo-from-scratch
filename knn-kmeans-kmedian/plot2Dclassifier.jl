using Plots


function plot2Dclassifier(X,y,model)

	# Make predictions across range of plot
	increment = 100
	xmin = minimum(X[:,1])
	xmax = maximum(X[:,1])
	xDomain = range(xmin,stop=xmax,length=increment)
	xValues = repeat(xDomain',length(xDomain),1)

	ymin = minimum(X[:,2])
	ymax = maximum(X[:,2])
	yDomain = range(ymin,stop=ymax,length=increment)
	yValues = repeat(yDomain,1,length(yDomain))

	z = model.predict([xValues[:] yValues[:]])

	@assert(length(z) == length(xValues),"Size of model function's output is wrong");

	zValues = reshape(z,size(xValues))

	# Plot decision surface
	if all(zValues[:] .== 1)
    		cm = cgrad([RGB(0,0,.5),RGB(0,0,0.5)],[1,2])
	elseif all(zValues[:] .== 2)
    		cm = cgrad([RGB(.5,0,0),RGB(.5,0,0)],[1,2])
	else
    		cm = cgrad([RGB(0,0,.5),RGB(.5,0,0)],[1,2])
	end
	contour(xDomain,yDomain,zValues,fill=true,seriescolor=cm,legend=false)

	# Overlay data points
	scatter!(X[y.==1,1],X[y.==1,2],legend=false,markercolor=RGB(0,0,1),markershape=:rect)
	scatter!(X[y.==2,1],X[y.==2,2],legend=false,markercolor=RGB(1,0,0),markershape=:circle)

end
