
#--------------Importing the Data--------------------------

samplesData <- read.csv(file = "pheno.f.csv", header = TRUE, sep = ",")
genesData <- read.csv(file = "gene.f.with.entrez.csv", header = TRUE, sep = ",")
samplesGenesData <- read.csv(file = "mat.f.coding.csv", header = TRUE, sep = ",")

#--------------Arranging the Data--------------------------

brainSamples <- samplesData[which(samplesData$SMTS == "Brain"),]

# all samples in death level 1 + 2, we will check how much we have in each level
levelOne <- brainSamples[which(brainSamples$DTHHRDY == 1),]
levelTwo <- brainSamples[which(brainSamples$DTHHRDY == 2),]

#We changed the SampID in the levelOne and LeveTwo data frames because we needed it to match the sampIds in 
# the samplesGenesData data frame
# originally in levelOne/levelTwo: GTEX-1117F-0226-SM-5GZZ7
#in samplesGenesData: GTEX.1117F.0226.SM.5GZZ7
levelOne$SAMPID <- gsub("-",".",levelOne$SAMPID)
levelTwo$SAMPID <- gsub("-",".",levelTwo$SAMPID)

#combine all the data of level two and one 
samplesInfo <- rbind(levelOne, levelTwo)

#samples of brain in level 1+2
sampId <- samplesInfo$SAMPID


#--------------Exploration--------------------------
dev.off()
brainSampIdTwo <- levelTwo$SAMPID
brainSampIdOne <- levelOne$SAMPID 

#get rna levels of samples of each level
brainOne <- cbind(samplesGenesData$Name,select(samplesGenesData, all_of(brainSampIdOne)))
brainTwo <-  cbind(samplesGenesData$Name,select(samplesGenesData, all_of(brainSampIdTwo)))

# box plot according to death level in regard to the mean of the gene expression of each sample
geneMeansOne <- colMeans(brainOne[,-1])
geneMeansTwo <- colMeans(brainTwo[,-1])

#boxplot of samples
boxplot(geneMeansOne, geneMeansTwo, names=c("first level", "second level"), main = "Comperison of RNA levels in the Brain according to length of terminal state", xlab = "Death label", ylab = "Mean of genes per sample")

#--------------Primary Cleaning--------------------------

library('dplyr')

cleanData <- function(samplesGenesData, sampId) {
  
  #add the Names of the genes to the table
  genes <- cbind(samplesGenesData$Name,select(samplesGenesData, all_of(sampId)))
  names(genes)[names(genes) == 'samplesGenesData$Name'] <- 'Name'
  
  #delete genes with low values - with 80% of expression is 0.1.
  vec <- apply(genes[,-1], 1, function(x) length(which( x > log(0.1 + 1, 2))))
  row <- which(vec > (0.8*(ncol(genes[,-1]))))
  # leave just rows with expression per at least 80% of the samples
  filteredGenes <- genes[row, ]

  #delete genes with low variance
  varGenes <- apply(filteredGenes[,-1], 1, var) #generate variance of each row(gene)

  #delete genes with variance <=  the 0.05 quantile
  lowVarIndxs <- which(varGenes <= quantile(varGenes, probs = 0.05))
  if(length(lowVarIndxs) > 0)
  {
    data.free = filteredGenes
    #now we get smaller matrix, with no genes with variance <= the 0.05 quantile
    filteredGenes <- data.free[-lowVarIndxs,]
  }
  return (filteredGenes)
}

brainData <- cleanData(samplesGenesData, sampId)

#-------------------------PCA to detect unwanted parts of the brain------------------
library(ggfortify)
library(ggplot2)

## plot pc1 and pc2

pca <- prcomp(t(brainData[,-1]), scale=TRUE) 
autoplot(pca, data = samplesInfo, colour = 'SMTSD', main = "PCA according to the specific part in the Brain for levels one and two") +
  guides(colour = guide_legend(title = "Brain Parts")) 

#delete unwanted brain parts

samplesInfo <- samplesInfo[which(samplesInfo$SMTSD != 'Brain - Cerebellum' & samplesInfo$SMTSD != 'Brain - Cerebellar Hemisphere'),]

#clean the data again
brainSampId <- samplesInfo$SAMPID 
brainData <- cleanData(brainData, brainSampId)

#--------------Outliers and More Cleaning------------------

#We will use Hierarchical Clustering to find the outliers
sampleTree = hclust(dist(t(brainData[,-1])), method = "average")

par(cex = 0.3)
par(mar = c(0,4,2,0))
#plot the dendrogram with clustering
plot(sampleTree, hang = -1, cex = 1, main = '')
rect.hclust(sampleTree , h = 100, border = 2:6)

#decide of the height for the clustering - devide to groups
clusters <- cutree(sampleTree, h = 100)

#show the number of values in each cluster
as.data.frame(table(clusters))  

#get the name of the outliers - groups of one sample
outNames <- names(clusters[clusters == 13 |clusters == 9 | clusters == 2 | clusters == 12])

#plot the PCA with a colored outliers 
tB <- as.data.frame(t(brainData[,-1]))
newPca <- prcomp(tB, scale=TRUE) 
tB$outlier <- factor(ifelse(rownames(tB) %in% outNames, "Outlier", "Normal"))

#plot the PCA
autoplot(newPca, data = tB, colour = 'outlier', main = "PCA to check potential outliers") 

#we decided not to remove the outliers from the df, according to the pca every thing looks mixed well

#-----------------------remove unnecessary features from samplesInfo table----------------------

samplesInfo <- samplesInfo[c("SAMPID","SMCENTER","SMRIN", "SMTS", "SMTSD", "SMTSISCH", "SUBJID", "SEX", "AGE", "DTHHRDY")]

#export to csv
write.csv(x = brainData, file = 'brainData.csv')
write.csv(x = samplesInfo, file = 'samplesInfo.csv')
