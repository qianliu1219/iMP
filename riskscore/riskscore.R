rm(list = ls())

library(scales)
library(ggplot2)

#setwd("~/Desktop/projects/iMP/ictcf")
etwd("~/Desktop/projects/iMP/COVID-CT")
pCT <- read.csv("risk_score_pCT.txt",col.names = F)
pCT_scale <- as.data.frame(rescale(as.matrix(pCT), to = c(0, 100)))
normalize(as.vector(pCT), method = "standardize", range = c(0, 100), margin = 1L, on.constant = "quiet")
nCT <- read.csv("risk_score_nCT.txt",col.names = F)
nCT_scale <- as.data.frame(rescale(as.matrix(nCT), to = c(0, 100)))


t.test(pCT_scale,nCT_scale)


ggplot(nCT_scale) + geom_density(aes(x = FALSE.,fill="red"), alpha = 0.5) + 
  geom_density(data = pCT_scale,aes(x = FALSE.,fill="blue"), alpha = 0.5)
