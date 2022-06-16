#-------------------import data---------------------------

brainData <- read.csv(file = "brainData.csv", header = TRUE, sep = ",")
samplesInfo <- read.csv(file = "samplesInfo.csv", header = TRUE, sep = ",")

#remove non relevant columns
geneNames <- brainData[,2]
brainData <- brainData[,c(-1, -2)]
samplesInfo <- samplesInfo[,-1]

#For brainData, we'll make the row names as the gene names
rownames(brainData) <- geneNames


#---------Second OPTION: Reggresion + Quantile Normalization---------

#first step in doing the reggresion, we need to understand what features "act differently" according to their value in this feature
#so let's plot PCA

library(ggfortify)
library(ggplot2)

pca <- prcomp(t(brainData), scale=TRUE) 

#AGE
autoplot(pca, data = samplesInfo, colour = 'AGE', main = "PCA according to the age") +
  guides(colour = guide_legend(title = "Age")) 

#SEX
samplesInfo$SEX <- as.factor(samplesInfo$SEX)
autoplot(pca, data = samplesInfo, colour = 'SEX', main = "PCA according to the sex") +
  guides(colour = guide_legend(title = "Sex")) 

#CENTER
autoplot(pca, data = samplesInfo, colour = 'SMCENTER', main = "PCA according to the Centers") +
  guides(colour = guide_legend(title = "center")) 

#RIN(quality)
samplesInfo$GRPSMRIN <- as.factor(round(samplesInfo$SMRIN))
autoplot(pca, data = samplesInfo, colour = 'GRPSMRIN', main = "PCA according to the quality") +
    guides(colour = guide_legend(title = "quality")) + scale_color_manual(values = c("orange","blue","red", "yellow", "green", "#C7A6EA"))

#TIME
cuts <- cut(samplesInfo$SMTSISCH, c(seq(300, 1700, 280)), dig.lab = 5)
samplesInfo$GRPSMTSISCH <- cuts
autoplot(pca, data = samplesInfo, colour = 'GRPSMTSISCH', main = "PCA according to the time that passed from the death") +
  guides(colour = guide_legend(title = "SMTSISCH")) 

#first of all, for the regression we need to make sure that the sampIds synchronized  
#so lets order the tables according to the sampId.

oredredSamplesInfo <- samplesInfo[order(samplesInfo$SAMPID),]
orderedBrainData <- brainData[order(names(brainData))]

#check correlation between Rin and Time
cor.test(samplesInfo$SMRIN, samplesInfo$SMTSISCH, 
         method = "pearson")

#plot the correlation
plot(as.numeric(oredredSamplesInfo$SMTSISCH), as.numeric(oredredSamplesInfo$SMRIN), xlab = "Time", ylab = "Rin", main = "Rin according to time")
abline(lm(as.numeric(oredredSamplesInfo$SMRIN)~as.numeric(oredredSamplesInfo$SMTSISCH)))


#It looks like the features RIN and CENTER are dividing the points according to their values.
#We'll fix the RNA according to them.
#But we know that the time is supposed to effect the RNA values as well.
#let's do some box plots and regression lines. 

#RIN
par(mfrow = c(2,2))

plotRin <- function(i) {
  plot(as.numeric(oredredSamplesInfo$SMRIN), as.numeric(orderedBrainData[i,]), xlab = "RIN", ylab = "RNA value", main = sprintf("%s", row.names(orderedBrainData[i,])))
  mod <- lm(as.numeric(orderedBrainData[i,])~as.numeric(oredredSamplesInfo$SMRIN))
  abline(mod)
}

randGenes <- as.integer(runif(4, min=0, max=nrow(orderedBrainData)))

for (gene in randGenes) {
  plotRin(gene)
}


#TIME
plotTime <- function(i) {
  plot(as.numeric(oredredSamplesInfo$SMTSISCH), as.numeric(orderedBrainData[i,]), xlab = "Time", ylab = "RNA value", main = sprintf("%s", row.names(orderedBrainData[i,])))
  mod <- lm(as.numeric(orderedBrainData[i,])~as.numeric(oredredSamplesInfo$SMTSISCH))
  abline(mod)
}

for (gene in randGenes) {
  plotTime(gene)
}

par(mfrow = c(1,1))

# box plot according to Rin regard to the mean of the gene expression of each sample
geneMean <- colMeans(orderedBrainData)
boxplot(geneMean~oredredSamplesInfo$GRPSMRIN, main = "Comperison of RNA levels according to the quality", xlab = "RIN (quality)", ylab = "Mean of genes per sample")

#CENTER
boxplot(geneMean~oredredSamplesInfo$SMCENTER, main = "Comperison of RNA levels according to the centers", xlab = "Center", ylab = "Mean of genes per sample")

# box plot according to Rin regard to the mean of the gene expression of each sample
boxplot(geneMean~oredredSamplesInfo$GRPSMTSISCH, main = "Comperison of RNA levels according to the Time", xlab = "Time", ylab = "Mean of genes per sample")

#-------------------------------REGRESSION-------------------------------------

FixedBrainData <- orderedBrainData
#RIN + CENTER
for (i in 1:nrow(orderedBrainData)) {
  mod <- lm(as.numeric(orderedBrainData[i,])~as.numeric(oredredSamplesInfo$SMRIN) + oredredSamplesInfo$SMCENTER)
  FixedBrainData[i,] <- residuals(mod)
}

#-----------------quantile normalization-----------------------

#if (!require("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")

#BiocManager::install("preprocessCore")
library(preprocessCore)

#rows are genes, columns are samples
quantileNormalization <- function(ourData)
{
  normData <- preprocessCore::normalize.quantiles(as.matrix(ourData))
  rownames(normData) <- rownames(ourData)
  colnames(normData) <- colnames(ourData)
  return(normData)
}
#normalize for every level
normBrainData <- as.data.frame(quantileNormalization(FixedBrainData))

par(mfrow = c(1,2))
boxplot(FixedBrainData[,2:12], names = NULL, main = "Examples of genes before\n quantile normalization")
boxplot(normBrainData[,2:12], names = NULL, main = "Examples of genes after\n quantile normalization")


write.csv(FixedBrainData, "RegQuantBrainData.csv")

