include("misc.jl") # Includes GenericModel typedef

function knn_predict(Xhat,X,y,k)
  (n,d) = size(X)
  (t,d) = size(Xhat)
  k = min(n,k) # To save you some debuggin
  prediction = zeros(t)
  allDistancesSquared = distancesSquared(Xhat, X)
  allDistances = sqrt.(abs.(allDistancesSquared))
  
  for i in 1:t
    distances_i = allDistances[i,:]
    p = sortperm(distances_i)
    
    neighbors = zeros(k)
    for j in 1:k
      neighbors[j] = y[p[j]]
    end
    
    mode_y = mode(neighbors)
    prediction[i] = mode_y
   
  end
    
  return prediction
end

function knn(X,y,k)
	# Implementation of k-nearest neighbour classifier
  predict(Xhat) = knn_predict(Xhat,X,y,k)
  return GenericModel(predict)
end
