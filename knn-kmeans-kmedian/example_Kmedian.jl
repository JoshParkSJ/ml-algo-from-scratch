# Load data
using JLD
X = load("clusterData2.jld","X")

# K-means clustering
k = 4
#minArray = zeros(10)
include("kMedians.jl")

#for k in 1:10
let modelChosen = kMedians(X,k,doPlot=false), yFinal = modelChosen.predict(X), errVal = kMediansError(X,yFinal,modelChosen.W)

for i in 2:50
    model = kMedians(X,k,doPlot=false)
    y = model.predict(X)
    errVal_i=kMediansError(X,y,model.W)
    if(errVal_i< errVal)
        errVal = errVal_i
        modelChosen=model
        yFinal=y
    end
end    

#minArray[k] = errVal;
include("clustering2Dplot.jl")
clustering2Dplot(X,yFinal,modelChosen.W)
end
#end

#xCoord = 1:10
#plot(xCoord, minArray, title="Error values for k=1 to k=10 for clusterData2.jld")
#xlabel!("k values")
#ylabel!("kMediansError")