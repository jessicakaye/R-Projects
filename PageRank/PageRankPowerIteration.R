library(pracma)

# This is the initial connectivity matrix
H <- matrix(nrow=5, ncol=5, byrow=FALSE, dimnames = list(c("A", "B", "C", "D", "E"),c("A", "B", "C", "D", "E")))
H[1:5,1:5] <-  c(0, 1/2, 0, 1/2,0,1/2,0,0,1/2,0,0,0,0,1/2,1/2,0,0,0,0,0,1/3,0,1/3,1/3,0)
H
# This is the teleportation matrix
e <- matrix(nrow=5,ncol=5)
e[1:5,1:5] <- 1/5
e
# The given random surfer probability is 0.85
G <- (0.85 * H) + (0.15*e)
G

nodes <- c("A", "B", "C", "D", "E")

# These are the first iteration values of the PageRanks for the 5 pages
a1 <- round((rowSums(G)["A"] * 0.2),3)
b1 <- round((rowSums(G)["B"] * 0.2),3)
c1 <- round((rowSums(G)["C"] * 0.2),3)
d1 <- round((rowSums(G)["D"] * 0.2),3)
e1 <- round((rowSums(G)["E"] * 0.2),3)

# r will be the iterative schema for these pages
r <- c(a1,b1,c1,d1,e1)
r

# This matrix will show all of the iterations before tolerance is met
iterations <- matrix(ncol = 10, nrow = 5, dimnames = list(c("A", "B", "C", "D", "E"),c()))
# Here I assign the initialized PageRank for the pages
iterations[1:5,1] <- 0.2
# Here I assign the first iteration values of PageRank for the pages
iterations[1:5,2] <- r
iterations



# This loop iterates through the process of calculating the PageRank until the tolerance is reached (tolerance <=0.02)
errorexists <- TRUE
iter <- 3
while (errorexists == TRUE) {
  errorcount <- 0
  for (node in nodes) {
      error <- abs(iterations[node,iter-1] - iterations[node, iter-2])
      if (error >0.02) {
        errorcount <- errorcount +1
      }
        iterations[node,iter] <- round(dot(G[node,],r),3)
  }
  r <- iterations[1:5,iter]
  if (errorcount>0){
    iter <- iter + 1
  }
  if (errorcount == 0){
    errorexists <- FALSE
  }
}  

# Here are all the iterations of PageRank
iterations

# Here are the final importance values for each page
r
initialresults <- r

# Here are the pages in order of decreasing PageRank score!
sort(r, decreasing = TRUE)


# DOUBLE CHECK WITH OTHER METHODS

# Transition matrix A
A <- matrix(nrow=5, ncol=5, byrow=FALSE, dimnames = list(c("A", "B", "C", "D", "E"),c("A", "B", "C", "D", "E")))
A[1:5,1:5] <-  c(0, 1/2, 0, 1/2,0,1/2,0,0,1/2,0,0,0,0,1/2,1/2,0,0,0,0,0,1/3,0,1/3,1/3,0)
A

# Take Decay d = 0.85
d <- 0.85
B <- 0.85*A + (0.15/5)
r <- matrix(c(1/5,1/5,1/5,1/5,1/5),nrow=5)
r

# perform power iterations on B
iterations <- function(M, r, n) {
  Bn = diag(nrow(M)) 
  # Caculate B n times
  for (i in 1:n)
  {
    Bn = Bn %*% M
  }
  return (Bn %*% r)
}

# Different values of n, to see convergence
eig_vec1<-iterations(B, r, 10) #Converges
#iterations(B, r, 40)
eig_vec1

# Decomposing B
eigen_decomp <- eigen(B)

# Maximum eigen value we get is indeed 1
max_value<-which.max(eigen_decomp$values)
## Warning in which.max(eigen_decomp$values): imaginary parts discarded in
## coercion
# check this eigenvector has all positive entries and it sums to 1
eig_vec2 <- as.numeric((1/sum(eigen_decomp$vectors[,1]))*eigen_decomp$vectors[,1])
sum(eig_vec2)
## [1] 1
library(igraph)
graph_A <- graph.adjacency(t(A), weighted = TRUE, mode = "directed")
plot(graph_A)
eig_vec3 <- page.rank(graph_A)$vector
eig_vec3

#Let's compare all of the results
results <- cbind(initialresults, eig_vec1, eig_vec2, eig_vec3)
results
colnames(results) <- c('Initial Iteration', 'Power Iter', 'Eigen Decom', 'Igraph')
results
