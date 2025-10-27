library(class)
library(ggplot2)
library(dbscan)

# Chargement des données iris
data(iris)

# Suppression de la colonne Species pour appliquer les algorithmes
iris_data <- iris[, -5]
iris_labels <- iris$Species

#----------------------------K-Means--------------------------------------------

#### Appliquer k-means sur les données iris
set.seed(42)
kmeans_result <- kmeans(iris_data, centers = 3)

# Ajouter les clusters prévus dans les données
iris$Cluster_kmeans <- as.factor(kmeans_result$cluster)

#------------------------hierarchical clustering--------------------------------

dist_matrix <- dist(iris_data)  # Matrice de distances
hclust_result <- hclust(dist_matrix, method = "ward.D2")

# Couper l'arbre pour obtenir 3 clusters
hclust_clusters <- cutree(hclust_result, k = 3)
iris$Cluster_hclust <- as.factor(hclust_clusters)

#-----------------------------DBSCAN--------------------------------------------

# eps = rayon maximal d'un voisinage) et minPts (nombre minimum de voisins)
dbscan_result <- dbscan(iris_data, eps = 0.5, minPts = 5)

# Ajouter les clusters DBSCAN à la dataframe iris
iris$Cluster_dbscan <- as.factor(dbscan_result$cluster)

#-------------------Séparation des données en train et test---------------------

# Séparation des données en train et test
set.seed(42)
train_idx <- sample(1:nrow(iris_data), 0.7 * nrow(iris_data))
train_data <- iris_data[train_idx, ]
train_labels <- iris_labels[train_idx]
test_data <- iris_data[-train_idx, ]
test_labels <- iris_labels[-train_idx]

# Appliquer la méthode K-NN avec k = 3
knn_pred <- knn(train = train_data, test = test_data, cl = train_labels, k = 3)

#-------------------Visualisation des résultats---------------------------------
##K-means
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Cluster_kmeans)) +
  geom_point(size = 2) +
  labs(title = "K-means Clustering sur les données iris") +
  theme_minimal()
##Hierarchical Clustering
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Cluster_hclust)) +
  geom_point(size = 2) +
  labs(title = "Hierarchical Clustering sur les données iris") +
  theme_minimal()
##DBSCAN
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Cluster_dbscan)) +
  geom_point(size = 2) +
  labs(title = "DBSCAN Clustering sur les données iris") +
  theme_minimal()
##K-NN -  sur les données de test uniquement
test_results <- data.frame(Sepal.Length = test_data$Sepal.Length, 
                           Sepal.Width = test_data$Sepal.Width, 
                           Predicted = knn_pred)
ggplot(test_results, aes(x = Sepal.Length, y = Sepal.Width, color = Predicted)) +
  geom_point(size = 2) +
  labs(title = "K-NN method sur les données iris (test)") +
  theme_minimal()

#-----------------------Matrice de confusion------------------------------------

# Matrice de confusion pour K-NN
table(Predicted = knn_pred, Actual = test_labels)

# Matrice de confusion pour K-means
table(Kmeans = kmeans_result$cluster, Réel = iris_labels)

# Matrice de confusion pour DBSCAN
table(DBSCAN = dbscan_result$cluster, Réel = iris_labels)

# Matrice de confusion pour hclust
table(Hclust = hclust_clusters, Réel = iris_labels)
