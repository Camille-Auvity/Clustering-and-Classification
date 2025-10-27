library(class)
library(ggplot2)
library(dbscan)

# Charger les données
data <- read.csv("/Users/camilleauvity/Desktop/ECOLE/Ecole des mines/MAJEURE DATA/UP3 Machine Learning/Clustering/dataset_diabetes.csv", 
                 header = TRUE, sep = ",")

# Sélectionner les variables numériques pour le clustering
X_data <- data[, sapply(data, is.numeric)]

# Données brutes
X_raw_data <- X_data

# Données centrées et scalées
X_scaled_data <- scale(X_data)

# Assurez-vous que la colonne cible existe
y_data <- data$class  # Ajustez "class" au bon nom de colonne

#----------------------------K-Means--------------------------------------------

# K-Means sur les données brutes
set.seed(100)
kmeans_result_raw <- kmeans(X_raw_data, centers = 2)
data$Cluster_kmeans_raw <- as.factor(kmeans_result_raw$cluster)

# K-Means sur les données normalisées
kmeans_result_scaled <- kmeans(X_scaled_data, centers = 2)
data$Cluster_kmeans_scaled <- as.factor(kmeans_result_scaled$cluster)

#------------------------Hierarchical Clustering--------------------------------

# Hierarchical Clustering sur les données brutes
dist_matrix_raw <- dist(X_raw_data)  # matrice de distances pour les données brutes
hclust_result_raw <- hclust(dist_matrix_raw, method = "ward.D2")
hclust_clusters_raw <- cutree(hclust_result_raw, k = 2)
data$Cluster_hclust_raw <- as.factor(hclust_clusters_raw)

# Hierarchical Clustering sur les données normalisées
dist_matrix_scaled <- dist(X_scaled_data)  # matrice de distances pour les données scalées
hclust_result_scaled <- hclust(dist_matrix_scaled, method = "ward.D2")
hclust_clusters_scaled <- cutree(hclust_result_scaled, k = 2)
data$Cluster_hclust_scaled <- as.factor(hclust_clusters_scaled)

#-----------------------------DBSCAN--------------------------------------------

# DBSCAN sur les données scalées et centrées
dbscan_result_scaled <- dbscan(X_scaled_data, eps = 2.5, minPts = 15)
data$Cluster_dbscan_scaled <- as.factor(dbscan_result_scaled$cluster)

# Visualisation de l'eps avec KNNdistplot pour les données normalisées
kNNdistplot(X_scaled_data, k = 8)
abline(h = 2, col = "red")


#-------------------Séparation des données en train et test---------------------

# Séparer les données en train et test (sur les données scalées et centrées)
set.seed(100)
train_idx <- sample(1:nrow(X_scaled_data), 0.7 * nrow(X_scaled_data))
train_data <- X_scaled_data[train_idx, ]
train_labels <- y_data[train_idx]
test_data <- X_scaled_data[-train_idx, ]
test_labels <- y_data[-train_idx]

# Appliquer la méthode K-NN avec k = 2
knn_pred <- knn(train = train_data, test = test_data, cl = train_labels, k = 2)

#-------------------Visualisation des résultats---------------------------------

## K-Means sur les données brutes
ggplot(data, aes(x = pedigree, y = glucose, color = Cluster_kmeans_raw)) +
  geom_point(size = 2) +
  labs(title = "K-means Clustering sur les données brutes") +
  theme_minimal()

## K-Means sur les données normalisées
ggplot(data, aes(x = pedigree, y = glucose, color = Cluster_kmeans_scaled)) +
  geom_point(size = 2) +
  labs(title = "K-means Clustering sur les données normalisées") +
  theme_minimal()

## Hierarchical Clustering sur les données brutes
ggplot(data, aes(x = pedigree, y = glucose, color = Cluster_hclust_raw)) +
  geom_point(size = 2) +
  labs(title = "Hierarchical Clustering sur les données brutes") +
  theme_minimal()

## Hierarchical Clustering sur les données normalisées
ggplot(data, aes(x = pedigree, y = glucose, color = Cluster_hclust_scaled)) +
  geom_point(size = 2) +
  labs(title = "Hierarchical Clustering sur les données normalisées") +
  theme_minimal()

## DBSCAN sur les données normalisées
ggplot(data, aes(x = pedigree, y = glucose, color = Cluster_dbscan_scaled)) +
  geom_point(size = 2) +
  labs(title = "DBSCAN Clustering sur les données normalisées") +
  theme_minimal()

## K-NN sur les données de test
test_data <- as.data.frame(test_data)

test_results <- data.frame(pedigree = test_data$pedigree, 
                           glucose = test_data$glucose, 
                           Predicted = knn_pred)

ggplot(test_results, aes(x = pedigree, y = glucose, color = Predicted)) +
  geom_point(size = 2) +
  labs(title = "K-NN method sur les données (test)") +
  theme_minimal()

#-----------------------Matrices de Confusion-----------------------------------

# Matrice de confusion pour K-NN
table(Predicted = knn_pred, Actual = test_labels)

# Matrice de confusion pour K-means sur les données brutes (comparaison qualitative)
table(Kmeans_raw = data$Cluster_kmeans_raw, Réel = y_data)

# Matrice de confusion pour K-means sur les données normalisées (comparaison qualitative)
table(Kmeans_scaled = data$Cluster_kmeans_scaled, Réel = y_data)

# Matrice de confusion pour DBSCAN (comparaison qualitative)
table(DBSCAN_scaled = data$Cluster_dbscan_scaled, Réel = y_data)

# Matrice de confusion pour hclust sur les données brutes (comparaison qualitative)
table(Hclust_raw = data$Cluster_hclust_raw, Réel = y_data)

# Matrice de confusion pour hclust sur les données normalisées (comparaison qualitative)
table(Hclust_scaled = data$Cluster_hclust_scaled, Réel = y_data)
