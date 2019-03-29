if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("TCGAbiolinks", version = "3.8")

library(psycho)
library(SummarizedExperiment)
library(BiocManager)
library(TCGAbiolinks)
library(ggfortify)

# Reading in breast cancer data. 
query_breast <- GDCquery(project = c("TCGA-BRCA"),
                         data.category = "Transcriptome Profiling",  
                         #data.type = 'Gene Expression Quantification',
                         experimental.strategy = "RNA-Seq",
                         workflow.type = "HTSeq - Counts",
                         legacy = FALSE)

GDCdownload(query_breast)
breast_data <- GDCprepare(query_breast) 
                          #save=TRUE, 
                          #save.filename = 'BreastS4.rda', 
                          #summarizedExperiment = FALSE)
breastMatrix <- assay(breast_data)
breastMatrix <- t(breastMatrix)
breast_df = as.data.frame(breastMatrix)
write.csv(breast_df, file = "breast_exp.csv")



# Reading in ovary data
query_ovary <- GDCquery(project = c("TCGA-OV"),
                        data.category = "Transcriptome Profiling",  
                        data.type = 'Gene Expression Quantification',
                        experimental.strategy = "RNA-Seq",
                        workflow.type = "HTSeq - Counts",
                        legacy = FALSE)

GDCdownload(query_ovary)
data_ov <- GDCprepare(query_ovary)

ovMatrix <- assay(data_ov)
ovMatrix <- t(ovMatrix)
ovary_df = as.data.frame(ovMatrix)
write.csv(ovary_df, file = "ovary_exp.csv")
