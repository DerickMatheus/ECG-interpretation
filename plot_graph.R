library(easyGgplot2)
library(purrr)

source("/scratch/derickmath/interpret/results_pkdd/mean/id_24")
file_name = "~/123_result.pdf"

diagnosis = c("fdAVb", "RBBB", "LBBB", "SB", "AF", "ST")
derivations = c("D1", "D2", "D3", "AVL", "AVF", "AVR", "V1", "V2", "V3", "V4", "V5", "V6")
segs = c("p", "t", "pr", "qt", "A/V Rate", "st", "qrs", "axis", "rhythm", "random15", "random30", "random50", "random", "random2")
threshold = c(0.15, 0.1, 0.08, 0.33, 0.27, 0.20)
id = c(1,2,3,4,5,6)
names(threshold) = diagnosis

#AV_rate[3] = abs(AV_rate[3]) - abs(pr[3])  - abs(qt[3]) - abs(st[3]) - abs(qrs[3])
#pr[5] = pr[5] - p[5] - r[5]
#pr[3] = pr[3] - p[3] - r[3]

data_file = data.frame(p, t, pr, qt, AV_rate, st, qrs, axis, rhythm, random15, random30, random50, random, random2)
data_t = as.data.frame(t(as.matrix(data_file)))
rownames(data_t) = colnames(data)
colnames(data_t) = diagnosis

p1 = ggplot(data=data_t, aes(x= segs, y=abs(fdAVb), fill = ifelse(abs(fdAVb) < abs(threshold[id[1]]), "Fail", "Pass")))
p1 = p1 + geom_bar(stat="identity", width = 0.4)
p1 = p1 + coord_flip() + ylim(0, 1) + geom_hline(yintercept=abs(threshold[id[1]]), linetype="dashed", color = "red")
p1 = p1 + scale_fill_manual(name="Threshold", values = c("pink","orange"))
p1 = p1 + xlab("")
p1 = p1 + theme(axis.text=element_text(size=14), axis.title=element_text(size=14))
p1 = p1 + theme(axis.text.x = element_text(angle = 45,hjust = 1), legend.position = "bottom")
p1 = p1 + labs(y = " ", title = diagnosis[id[1]])

p2 = ggplot(data=data_t, aes(x= segs, y=abs(RBBB), fill = ifelse(abs(RBBB) < abs(threshold[id[2]]), "Fail", "Pass")))
p2 = p2 + geom_bar(stat="identity", width = 0.4)
p2 = p2 + coord_flip() + ylim(0, 1) + geom_hline(yintercept=abs(threshold[id[2]]), linetype="dashed", color = "red")
p2 = p2 + scale_fill_manual(name="Threshold", values = c("pink","orange"))
p2 = p2 + xlab("")
p2 = p2 + theme(axis.text=element_text(size=14), axis.title=element_text(size=14))
p2 = p2 + theme(axis.text.x = element_text(angle = 45,hjust = 1), legend.position = "bottom")
p2 = p2 + labs(y = " ", title = diagnosis[id[2]])

p3 = ggplot(data=data_t, aes(x= segs, y=abs(LBBB), fill = ifelse(abs(LBBB) < abs(threshold[id[3]]), "Fail", "Pass")))
p3 = p3 + geom_bar(stat="identity", width = 0.4)
p3 = p3 + coord_flip() + ylim(0, 1) + geom_hline(yintercept=abs(threshold[id[3]]), linetype="dashed", color = "red")
p3 = p3 + scale_fill_manual(name="Threshold", values = c("pink","orange"))
p3 = p3 + xlab("")
p3 = p3 + theme(axis.text=element_text(size=14), axis.title=element_text(size=14))
p3 = p3 + theme(axis.text.x = element_text(angle = 45,hjust = 1), legend.position = "bottom")
p3 = p3 + labs(y = " ", title = diagnosis[id[3]])

p4 = ggplot(data=data_t, aes(x= segs, y=abs(SB), fill = ifelse(abs(SB) < abs(threshold[id[4]]), "Fail", "Pass")))
p4 = p4 + geom_bar(stat="identity", width = 0.4)
p4 = p4 + coord_flip() + ylim(0, 1) + geom_hline(yintercept=abs(threshold[id[4]]), linetype="dashed", color = "red")
p4 = p4 + scale_fill_manual(name="Threshold", values = c("pink","orange"))
p4 = p4 + xlab("")
p4 = p4 + theme(axis.text=element_text(size=14), axis.title=element_text(size=14))
p4 = p4 + theme(axis.text.x = element_text(angle = 45,hjust = 1), legend.position = "bottom")
p4 = p4 + labs(y = " ", title = diagnosis[id[4]])

p5 = ggplot(data=data_t, aes(x= segs, y=abs(AF), fill = ifelse(abs(AF) < abs(threshold[id[5]]), "Fail", "Pass")))
p5 = p5 + geom_bar(stat="identity", width = 0.4)
p5 = p5 + coord_flip() + ylim(0, 1) + geom_hline(yintercept=abs(threshold[id[5]]), linetype="dashed", color = "red")
p5 = p5 + scale_fill_manual(name="Threshold", values = c("pink","orange"))
p5 = p5 + xlab("")
p5 = p5 + theme(axis.text=element_text(size=14), axis.title=element_text(size=14))
p5 = p5 + theme(axis.text.x = element_text(angle = 45,hjust = 1), legend.position = "bottom")
p5 = p5 + labs(y = " ", title = diagnosis[id[5]])

p6 = ggplot(data=data_t, aes(x= segs, y=abs(ST), fill = ifelse(abs(ST) < abs(threshold[id[6]]), "Fail", "Pass")))
p6 = p6 + geom_bar(stat="identity", width = 0.4)
p6 = p6 + coord_flip() + ylim(0, 1) + geom_hline(yintercept=abs(threshold[id[6]]), linetype="dashed", color = "red")
p6 = p6 + scale_fill_manual(name="Threshold", values = c("pink","orange"))
p6 = p6 + xlab("")
p6 = p6 + theme(axis.text=element_text(size=14), axis.title=element_text(size=14))
p6 = p6 + theme(axis.text.x = element_text(angle = 45,hjust = 1), legend.position = "bottom")
p6 = p6 + labs(y = " ", title = diagnosis[id[6]])

ggplot2.multiplot(p1,p2,p3,p4,p5,p6, cols=3)
#pdf(file_name)#, height = 3.5)
#ggplot2.multiplot(p1,p2,p3,p4,p5,p6, cols=3)
#dev.off()
