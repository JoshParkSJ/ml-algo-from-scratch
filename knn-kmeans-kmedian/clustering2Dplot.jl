#using(PyPlot)
using Plots

function clustering2Dplot(X,y,W=[])
	k = length(unique(y))

	markers = [:circle
		:rect
		:star5
		:diamond
		:hexagon
		:cross
		:xcross
		:utriangle
		:dtriangle
		:rtriangle
		:ltriangle
		:pentagon
		:octagon
		:star4
		:star6
		:star7
		:star8
		:vline
		:hline
		:+
		:-]

	for c in 1:k
		if c == 1
			scatter(X[y.==1,1],X[y.==1,2],legend=false,markershape=markers[1],color=c)
		else
			scatter!(X[y.==c,1],X[y.==c,2],legend=false,markershape=markers[c],color=c)
		end

		if !isempty(W)
			scatter!([W[c,1]],[W[c,2]],legend=false,markershape=markers[c],color=c,markersize=12)
		end
	end
	gui()
end

function clustering2Dplot_old(X,y,W=[])
	(n,d) = size(X)

	k = length(unique(y))

	# Pick some symbols and colors for the clusters
	symbols = ["s","o","v","^","x","+","*","d","<",">","p"]
	colours = [(0,1,0)
			(1,0,0)
			(0,0,1)
			(1,0,1)
			(1,1,0)
			(0,1,1)
			(.1,.1,.1)
			(1,.5,0)
			(0,.5,0)
			(.5,.5,.5)
			(.5,.25,0)
			(.5,0,.5)
			(0,.5,1)]

	# Plot the points and the means
	clf()

	# Plot the points coloured by cluster
	for c in 1:k
		colour = (.75colours[c][1],.75colours[c][2],.75colours[c][3])
		plot(X[y.==c,1],X[y.==c,2],"o",marker=symbols[c],color=colour,markersize=5)

		if !isempty(W)			plot(W[c,1],W[c,2],marker=symbols[c],color=colours[c],markersize=12)
		end
	end

	# Plot the outliers in black
	if any(y.==0)
		plot(X[y.==0,1],X[y.==0,2],"o",color="black",markersize=5)
	end
end
