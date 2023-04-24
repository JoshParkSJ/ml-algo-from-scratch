using DelimitedFiles

# Load data
dataTable = readdlm("animals.csv",',')
X = float(real(dataTable[2:end,2:end]))
(n,d) = size(X)

# Plot matrix as image
using Plots
heatmap(X)
