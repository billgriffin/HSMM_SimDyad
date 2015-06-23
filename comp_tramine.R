library(TraMineR)

data <- read.csv(file="tmp.csv", header=TRUE)
seq <- seqdef(data)

res <- seqdist(seq, method="HAM")
write.table(res[2:2,1:1], quote=FALSE, file="HAM.csv", row.names=FALSE, col.names=FALSE)

res <- seqdist(seq, method="DHD")
write.table(res[2:2,1:1], quote=FALSE, file="DHD.csv", row.names=FALSE, col.names=FALSE)

res <- seqdist(seq, method="LCP")
write.table(res[2:2,1:1], quote=FALSE, file="LCP.csv", row.names=FALSE, col.names=FALSE)

res <- seqdist(seq, method="LCS")
write.table(res[2:2,1:1], quote=FALSE, file="LCS.csv", row.names=FALSE, col.names=FALSE)