
############################################################
## Proteomics data preprocessing and feature filtering
############################################################
# ==========================================================
# 1. Load proteomics intensity matrix
#    Rows: proteins
#    Columns: samples
# ==========================================================
library(limma)
library(dplyr)
Proteomics.data=read.table(text = read_clip(), 
                           header = TRUE, sep = "\t", stringsAsFactors = FALSE)
rownames(Proteomics.data) <- Proteomics.data[, 1]
Proteomics.data <- Proteomics.data[, -1]
# ==========================================================
# 2. Missing value filtering
#    Retain proteins quantified in at least (1 - na_threshold)
#    proportion of samples
# ==========================================================
na_threshold <- 0.75   # maximum allowed missing rate per protein
data_filtered <- Proteomics.data %>%
  dplyr::filter(rowMeans(is.na(.)) <= na_threshold)
# ==========================================================
# 3. Missing value imputation
#    Remaining missing values are imputed using the global
#    minimum intensity, assuming left-censored measurements
# ==========================================================
min_intensity <- min(Proteomics.data, na.rm = TRUE)
data_filtered[is.na(data_filtered)] <- min_intensity
# ==========================================================
# 4. Log2 transformation
# ==========================================================
data_log2 <- log2(data_filtered)
# ==========================================================
# 5. Variability-based feature filtering
#    Proteins with low variability across samples are
#    removed based on coefficient of variation (CV)
# ==========================================================
cv_threshold <- 0.25
protein_cv <- apply(
  data_log2,
  1,
  function(x) sd(x, na.rm = TRUE) / mean(x, na.rm = TRUE)
)
data_var <- data_log2[protein_cv >cv_threshold, ]
cat("Number of proteins retained after CV filtering:", nrow(data_var), "\n")
# --> 2902 proteins retained for downstream analysis

############################################################
## Differential protein expression analysis using limma
############################################################
# ==========================================================
# 6. Experimental design
#    low-risk: 27 samples
#    high-risk: 33 samples
# ==========================================================
group <- c(rep("low-risk", 27), rep("high-risk", 33))
group <- factor(group, levels = c("low-risk", "high-risk"))
design <- model.matrix(~0 + group)
colnames(design) <- c("LowRisk", "HighRisk")

# ==========================================================
# 7. Linear modeling and empirical Bayes moderation
# ==========================================================
fit <- lmFit(data_var, design, method = "ls")

contrast_matrix <- makeContrasts(
  HighRisk_vs_LowRisk = HighRisk - LowRisk,
  levels = design
)

fit2 <- contrasts.fit(fit, contrast_matrix)
fit2 <- eBayes(fit2, trend = TRUE, robust = TRUE)

# ==========================================================
# 8. Extract differential proteins
# ==========================================================
de_proteins <- topTable(
  fit2,
  coef = "HighRisk_vs_LowRisk",
  number = nrow(data_var),
  adjust.method = "BH",
  confint = TRUE
)
# Add protein identifiers
de_proteins$ProteinID <- rownames(de_proteins)

# If multiple IDs are present, retain the primary identifier
de_proteins <- de_proteins %>%
  mutate(ProteinID_clean = sub(";.*", "", ProteinID))

# ==========================================================
# 9. Define risk-associated protein signatures
# ==========================================================
logFC_cutoff <- 1
fdr_cutoff   <- 0.1

de_signatures <- de_proteins %>%
  filter(abs(logFC) > logFC_cutoff,
         adj.P.Val < fdr_cutoff)

high_risk_signatures <- de_signatures %>%
  filter(logFC > logFC_cutoff)

low_risk_signatures <- de_signatures %>%
  filter(logFC < -logFC_cutoff)



















TS <-c(rep(1,27),rep(2,33)) 
TS<-ifelse(TS==2,"trearment","control")
design2 <- model.matrix(~0+TS)
colnames(design2) <- c( "control","treatment")

# ==========================================================
# 7. Linear modeling and empirical Bayes moderation
# ==========================================================
fit <- lmFit(data_var, design2, method = "ls")

contrast_matrix <- makeContrasts(
  HighRisk_vs_LowRisk = HighRisk - LowRisk,
  levels = design
)

fit2 <- contrasts.fit(fit, contrast_matrix)
fit2 <- eBayes(fit2, trend = TRUE, robust = TRUE)
# ==========================================================
# 8. Extract differential proteins
# ==========================================================
de_proteins <- topTable(
  fit2,
  coef = "HighRisk_vs_LowRisk",
  number = nrow(data_var),
  adjust.method = "BH",## p-value校准方法用BH
  confint = TRUE
)
# Add protein identifiers
de_proteins$ProteinID <- rownames(de_proteins)
# If multiple IDs are present, retain the primary identifier
de_proteins <- de_proteins %>%
  mutate(ProteinID_clean = sub(";.*", "", ProteinID))
# ==========================================================
# 9. Define risk-associated protein signatures
# ==========================================================
logFC_cutoff <- 1
fdr_cutoff   <- 0.1

de_signatures <- de_proteins %>%
  filter(abs(logFC) > logFC_cutoff,
         adj.P.Val < fdr_cutoff)

high_risk_signatures <- de_signatures %>%
  filter(logFC > logFC_cutoff)
low_risk_signatures <- de_signatures %>%
  filter(logFC < -logFC_cutoff)
# ==========================================================
# 10. Export signatures for downstream network analysis
# ==========================================================
write.csv(high_risk_signatures,"high_risk_signatures.csv")
write.csv(low_risk_signatures,"low_risk_signatures.csv")



group <- c(rep("low-risk", 27), rep("high-risk", 33))
group <- factor(group, levels = c("low-risk", "high-risk"))
design <- model.matrix(~0 + group)
colnames(design) <- c("LowRisk", "HighRisk")


rfit <- lmFit(data_var,design2,method="ls")
cont.matrix <- makeContrasts(contrast=treatment-control, levels=design2)
rfit <- contrasts.fit(rfit, cont.matrix)
rfit <- eBayes(rfit,trend=TRUE,robust = TRUE)
DE.protein<-topTable(rfit, coef="contrast", number=nrow(Proteomics.data),confint=TRUE,
                     adjust="BH")
##改变id
DE.protein$ID=rownames(DE.protein)
DE.protein <- DE.protein %>%
  mutate(ID_clean = sub(";.*", "", ID))
## 宣传差异蛋白质
DE.final=subset(DE.protein,abs(logFC)>1&adj.P.Val<0.1)
high.risk.signatures= subset(DE.final,logFC>1)
low.risk.signatures= subset(DE.final,logFC<(-1))
write.csv(high.risk.signatures,"high.risk.signatures.csv")
write.csv(low.risk.signatures,"low.risk.signatures.csv")

### 输出的signature，投入到string database中，获得PPI相互作用网络.tsv格式文件，构建subclusters 根据前面的格式，修改注释，如果觉得代码不够高级的地方，请一并修改。








### 采用limma 软件对差异蛋白质进行差异分析
##low-risk 包含27，high-risk 包含33例样本
library(limma)
library(dplyr)
TS <-c(rep(1,27),rep(2,33)) 
TS<-ifelse(TS==2,"trearment","control")
design2 <- model.matrix(~0+TS)
colnames(design2) <- c( "control","treatment")
rfit <- lmFit(data_var,design2,method="ls")
cont.matrix <- makeContrasts(contrast=treatment-control, levels=design2)
rfit <- contrasts.fit(rfit, cont.matrix)
rfit <- eBayes(rfit,trend=TRUE,robust = TRUE)
DE.protein<-topTable(rfit, coef="contrast", number=nrow(Proteomics.data),confint=TRUE,
                     adjust="BH")
##改变id
DE.protein$ID=rownames(DE.protein)
DE.protein <- DE.protein %>%
  mutate(ID_clean = sub(";.*", "", ID))
## 宣传差异蛋白质
DE.final=subset(DE.protein,abs(logFC)>1&adj.P.Val<0.1)
high.risk.signatures= subset(DE.final,logFC>1)
low.risk.signatures= subset(DE.final,logFC<(-1))
write.csv(high.risk.signatures,"high.risk.signatures.csv")
write.csv(low.risk.signatures,"low.risk.signatures.csv")

### 输出的signature，投入到string database中，获得PPI相互作用网络.tsv格式文件，构建subclusters



library(dplyr)
library(igraph)
library(readr)
ppi_raw <- read_tsv("string_interactions_short.tsv", show_col_types = FALSE)
colnames(ppi_raw)[1]="node1"

ppi <- ppi_raw %>%
  dplyr::select(node1, node2, combined_score) 
colnames(ppi) <- c("protein1", "protein2", "score")

########################################
# 2. 完成Walktrap clustering
########################################
g <- graph_from_data_frame(ppi[,1:2], directed = FALSE)
E(g)$weight <- ppi$score
# Walktrap clustering
cl <- cluster_walktrap(g, weights = E(g)$weight)
V(g)$cluster <- membership(cl)

table(V(g)$cluster)

cluster_df <- data.frame(
  gene = V(g)$name,
  cluster = V(g)$cluster
)

########################################
# 2. 每个 cluster 的基因提取出来
########################################
clusters <- split(cluster_df$gene, cluster_df$cluster)
# 去除小模块（节点太少无意义）
clusters <- clusters[sapply(clusters, length) >= 8]
length(clusters)
########################################
# 3. 对每个 cluster 做 pathway 富集（Reactome + GO BP + GO MF）
########################################
library(clusterProfiler)
library(stringr)

enrich_cluster <- function(gene_vector){
  
  # 不做任何过滤，把完整结果拿出来
  res_reactome <- enricher(
    gene_vector,
    TERM2GENE = final.pathway,
    pvalueCutoff = 1,
    qvalueCutoff = 1,
    minGSSize = 4
  )@result
  
  res_gobp <- enricher(
    gene_vector,
    TERM2GENE = GOBP.pathway,
    pvalueCutoff = 1,
    qvalueCutoff = 1,
    minGSSize = 4
  )@result
  
  res_gomf <- enricher(
    gene_vector,
    TERM2GENE = GOMF.pathway,
    pvalueCutoff = 1,
    qvalueCutoff = 1,
    minGSSize = 4
  )@result
  
  # 合并
  df <- bind_rows(
    res_reactome %>% mutate(DB = "Reactome"),
    res_gobp %>% mutate(DB = "GO_BP"),
    res_gomf %>% mutate(DB = "GO_MF")
  )
  
  # 手动根据 p.adjust 过滤
  df %>%
    filter(Count >= 4) %>%
    filter(!is.na(p.adjust)) %>%
    filter(p.adjust < 0.05)
}


cluster_path_list <- lapply(clusters, enrich_cluster)

all_cluster_pathways_full <- bind_rows(cluster_path_list, .id = "cluster_id")
########################################
# 3. 每个 cluster 内做 pathway 去冗余（Jaccard + hclust）
########################################
reduce_pathways <- function(df){
  
  if(nrow(df) < 2) return(df)
  
  geneSets <- lapply(df$geneID, function(x) str_split(x, "/")[[1]])
  names(geneSets) <- df$Description
  
  # 计算 Jaccard 矩阵
  jaccard <- function(a, b) length(intersect(a, b)) / length(union(a, b))
  n <- length(geneSets)
  mat <- matrix(0, n, n)
  for (i in 1:n){
    for (j in 1:n){
      mat[i,j] <- jaccard(geneSets[[i]], geneSets[[j]])
    }
  }
  
  dist_mat <- as.dist(1 - mat)
  
  hc <- hclust(dist_mat, method = "average")
  clusters <- cutree(hc, h = 0.4)
  
  df$cluster <- clusters
  
  df_sel <- df %>%
    group_by(cluster) %>%
    slice_min(order_by = p.adjust, n = 1) %>%
    ungroup()
  
  df_sel
}

cluster_path_sel <- lapply(cluster_path_list, reduce_pathways)

########################################
# 3. 构建一个全局 PPI + Pathway 融合网络
########################################

library(tidygraph)
library(ggraph)
library(tidyr)

# 合并所有 selected pathways
df_sel_all <- bind_rows(cluster_path_sel, .id = "cluster_id")

gene_path_edges <- df_sel_all %>%
  separate_rows(geneID, sep = "/") %>%
  rename(pathway = Description,
         gene = geneID)

# 所有基因节点
node_genes <- V(g)$name
node_pathways <- unique(gene_path_edges$pathway)

nodes <- data.frame(
  name = c(node_genes, node_pathways),
  type = c(
    rep("gene", length(node_genes)),
    rep("pathway", length(node_pathways))
  )
)

ppi_edges <- ppi %>% select(from = protein1, to = protein2) %>% mutate(type = "ppi")
gene_path_edges2 <- gene_path_edges %>% rename(from = gene, to = pathway) %>% mutate(type = "membership")

edges <- bind_rows(ppi_edges, gene_path_edges2)

g_all <- tbl_graph(nodes = nodes, edges = edges, directed = FALSE)
########################################
# 3. 专业可发表级别的 PPI–Pathway 综合网络图
########################################

set.seed(123)
ggraph(g_all, layout = "fr") +
  
  geom_edge_link(aes(color = type), alpha = 0.3) +
  
  geom_node_point(aes(color = type, size = ifelse(type=="gene", 3, 8)),
                  alpha = 0.9) +
  
  geom_node_text(
    data = function(x) x %>% filter(type=="pathway"),
    aes(label = name),
    repel = TRUE,
    size = 4,
    fontface = "bold"
  ) +
  
  geom_node_text(
    data = function(x) x %>%
      filter(type=="gene") %>%
      filter(centrality_degree() > quantile(centrality_degree(), 0.95)),
    aes(label = name),
    repel = TRUE,
    size = 3
  ) +
  
  scale_color_manual(values = c(
    gene = "#2C7BB6",
    pathway = "#D7191C",
    ppi = "grey70",
    membership = "#FDAE61"
  )) +
  
  theme_void() +
  ggtitle("Integrated PPI–Pathway Network by Cluster")

########################################
# PPI + Pathway 综合网络（Cluster 着色）
########################################

set.seed(123)

ggraph(g_all, layout = "fr") +
  
  geom_edge_link(aes(color = type), alpha = 0.3) +
  
  geom_node_point(aes(color = type, size = ifelse(type=="gene", 3, 8)),
                  alpha = 0.9) +
  
  geom_node_text(
    data = function(x) x %>% filter(type=="pathway"),
    aes(label = name),
    repel = TRUE,
    size = 4,
    fontface = "bold"
  )+
  
  
  scale_color_manual(
    values = c(
      "pathway" = "#D7191C",
      "1" = "#1b9e77",
      "2" = "#d95f02",
      "3" = "#7570b3",
      "4" = "#e7298a",
      "5" = "#66a61e",
      "6" = "#e6ab02",
      "7" = "#a6761d",
      "8" = "#666666"
    )
  ) +
  
  theme_void() +
  ggtitle("Integrated PPI–Pathway Network (Gene colored by Cluster)")

library(dplyr)
library(tidyr)

# 计算热图矩阵
heatmap_df <- all_cluster_pathways_full %>%
  mutate(cluster_id = paste0("Cluster_", cluster_id)) %>%
  mutate(pathway = Description) %>%
  mutate(value = -log10(p.adjust)) %>%   # 热图信号
  select(cluster_id, pathway, value) %>%
  spread(cluster_id, value, fill = 0)    # 没有富集则填 0

mat <- as.matrix(heatmap_df[,-1])
rownames(mat) <- heatmap_df$pathway

pheatmap(mat,
         color = colorRampPalette(c("white", "#2c7bb6", "#d7191c"))(100),
         border_color = NA,
         fontsize = 10,
         fontsize_row = 8,
         clustering_distance_rows = "euclidean",
         clustering_method = "average",
         main = "Cluster-specific Enriched Pathways")


library(ComplexHeatmap)
Heatmap(mat,
        name = "-log10(adj p)",
        col = colorRampPalette(c("white", "#fee8c8", "#fdae6b", "#e6550d"))(100),
        cluster_rows = TRUE,
        cluster_columns = TRUE,
        show_row_names = TRUE,
        show_column_names = TRUE,
        row_names_gp = grid::gpar(fontsize = 8),
        column_names_gp = grid::gpar(fontsize = 10),
        heatmap_legend_param = list(
          title = "-log10(p.adjust)",
          at = c(0, 1, 2, 3, 4),
          labels = c("0", "1", "2", "3", "4+")
        ))



########################################
# 2. 过滤低可信 PPI（非常关键）
########################################
ppi <- ppi %>%
  filter(!is.na(protein1) & !is.na(protein2)) %>%
  filter(protein1 != "" & protein2 != "") %>%
  filter(protein1 != protein2) %>%     # 去除自互作
  distinct(protein1, protein2, score)  # 去重

ppi <- ppi %>% filter(score >= 0.5)

cat("Nodes before graph build:", length(unique(c(ppi$protein1, ppi$protein2))), "\n")
cat("Edges after filtering:", nrow(ppi), "\n")
g <- graph_from_data_frame(ppi[, 1:2], directed = FALSE)

# 添加 STRING 置信度作为边权重
E(g)$weight <- ppi$score
cat("Graph nodes:", vcount(g), "\n")
cat("Graph edges:", ecount(g), "\n")

cl <- cluster_walktrap(g, weights = E(g)$weight)

# 提取模块标签
module <- membership(cl)
V(g)$module <- module
cat("Module sizes:\n")
print(table(module))

module_df <- data.frame(
  gene   = V(g)$name,
  module = V(g)$module
)

tg <- as_tbl_graph(g) %>%
  mutate(module = as.factor(module),
         degree = centrality_degree())

########################################
# 4. ggraph 绘制 PPI 网络
########################################

set.seed(123)   # 确保布局可复现

p <- 
  pdf("x.pdf",12,10)
ggraph(tg, layout = "fr") +   # Fruchterman–Reingold 布局（常用）
  # edges
  geom_edge_link(aes(alpha = ..index..),
                 colour = "grey70",
                 show.legend = FALSE) +
  # nodes
  geom_node_point(aes(color = module,
                      size = degree),
                  alpha = 0.9) +
  # 隐藏小节点标签，保留 hub genes
  geom_node_text(aes(label = ifelse(degree > quantile(degree, 0.90),
                                    name, "")),
                 repel = TRUE,
                 size = 3) +
  scale_size_continuous(range = c(1, 10)) +
  scale_color_brewer(palette = "Dark2") +
  theme_void(base_size = 14) +
  ggtitle("PPI Network Colored by Walktrap Modules")
dev.off()

print(p)

















Proteomics.data=proteomics.data1

library(ggplot2)
library(ggpubr) 
library(pheatmap)
###计算各样本的蛋白质组数目，并画图
non_na_counts <- colSums(!is.na(Proteomics.data))
non_na_df <- data.frame(
  Sample = names(non_na_counts),
  Non_NA_Count = as.numeric(non_na_counts)
)
non_na_df$subtype= c(rep("L", 27), rep("H", 33))
#####
non_na_df$Sample=factor(non_na_df$Sample,levels=non_na_df$Sample)
pdf("number.bar.plot.pdf",10,5)
ggplot(non_na_df, aes(x = Sample, y = Non_NA_Count, fill = subtype)) +
  geom_bar(stat = "identity") +
  labs(
    title = "Non-NA Counts per Sample",
    x = "Sample",
    y = "Non-NA Count"
  ) +
  scale_fill_manual(values = c("L" = "#3D3BF3", "H" = "#F8766D"))+
  theme_classic()+
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),  # 调整 x 轴标签角度
    legend.position = "none"  # 不显示图例
  )
dev.off()

######### boxplot
non_na_df$subtype=factor(non_na_df$subtype,levels=c("L","H"))
p <- ggplot(non_na_df, aes(x =subtype, y = Non_NA_Count, fill = subtype)) +
  geom_violin(trim = FALSE, alpha = 0.6) +  # 小提琴图
  geom_boxplot(width = 0.2, outlier.shape = NA, alpha = 0.5) +  # 添加箱线图
  geom_jitter(shape = 21, size = 2, width = 0.1, aes(fill = subtype),  stroke = 0.4, color = "white" , alpha = 0.7) +  # 添加散点图
  scale_fill_manual(values = c("L" = "#3D3BF3", "H" = "#F8766D")) +  # 自定义颜色
  theme_classic() +  # 使用经典主题
  labs(x = "Group", y = "Gene Expression", title = "SPP1 Expression in High- and Low-Risk Groups") +  # 轴标签和标题
  theme(legend.position = "none", text = element_text(size = 14)) +  # 隐藏图例 & 设置字体大小
  stat_compare_means(method = "wilcox.test", label = "p.signif", comparisons = list(c("L", "H"))) +  # 添加 t-test 显著性
  stat_compare_means(method = "wilcox.test", aes(label = paste0("p = ", ..p.format..)), label.x = 1.5, size = 5)  # 直接显示 p 值

pdf("nubmer.boxplot.pdf",5,5)
print(p) 
dev.off()

#### 
data_binary <- !is.na(data.raw.arrange)
# TRUE 表示有数值，FALSE 表示NA

# 初始化变量
cumulative_set <- c()  # 存储累计鉴定的蛋白
increment <- numeric(ncol(data_binary))  # 存储每个样本的增量

# 逐步累积计算增量
for (i in 1:ncol(data_binary)) {
  current_proteins <- rownames(data_binary)[data_binary[, i]]  # 当前样本鉴定的蛋白
  new_proteins <- setdiff(current_proteins, cumulative_set)  # 计算相对前面所有样本新增的蛋白
  increment[i] <- length(new_proteins)  # 记录增量
  cumulative_set <- union(cumulative_set, current_proteins)  # 更新累计蛋白集合
}
cumulative_increment <- cumsum(increment)


### intensity plot
log2.data=log2(Proteomics.data)
long_data <- melt(log2.data, variable.name = "Sample", value.name = "Log2_Intensity", na.rm = TRUE)
# 设置颜色分组（前26个蓝色，后34个红色）
long_data$Color <- ifelse(as.numeric(as.factor(long_data$Sample)) <= 27, "#3D3BF3", "#F8766D")

# 绘制 Boxplot
pdf("box.plot.pdf",10,3.5)
ggplot(long_data, aes(x = Sample, y = Log2_Intensity, fill = Color)) +ylim(c(10,25))+
  geom_boxplot(outlier.shape = NA, alpha = 0.7,col="gray70") +  # 箱线图，不显示离群值
  scale_fill_manual(values = c("#3D3BF3", "#F8766D")) +  # 自定义颜色
  labs(title = "Protein Log2 Expression Boxplot",
       x = "Sample",
       y = "Log2 Intensity") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 
dev.off()


data_filled <- log2.data
# 创建示例数据

# 定义一个函数，用于两两样本之间的相关性计算
calculate_pairwise_correlation <- function(data) {
  n_samples <- ncol(data)
  corr_matrix <- matrix(NA, nrow = n_samples, ncol = n_samples)
  colnames(corr_matrix) <- colnames(data)
  rownames(corr_matrix) <- colnames(data)
  
  for (i in 1:n_samples) {
    for (j in 1:n_samples) {
      if (i == j) {
        # 对角线上的值为1（样本与自身的相关性）
        corr_matrix[i, j] <- 1
      } else {
        # 提取两个样本的数据
        sample1 <- data[, i]
        sample2 <- data[, j]
        
        # 找到两个样本中至少有一个非缺失值的蛋白
        non_missing <- !(is.na(sample1) & is.na(sample2))
        sample1_non_missing <- sample1[non_missing]
        sample2_non_missing <- sample2[non_missing]
        
        # 对于只有一个样本缺失的蛋白，用两个样本的最小值填充
        min_value <- min(c(sample1_non_missing, sample2_non_missing), na.rm = TRUE)
        sample1_non_missing[is.na(sample1_non_missing)] <- min_value
        sample2_non_missing[is.na(sample2_non_missing)] <- min_value
        
        # 计算相关性
        corr_matrix[i, j] <- cor(sample1_non_missing, sample2_non_missing, 
                                 method = "spearman")
      }
    }
  }
  
  return(corr_matrix)
}

# 计算相关性矩阵
corr_matrix <- calculate_pairwise_correlation(data_filled)

corr_matrix <- cor(data_filled, use = "pairwise.complete.obs", method = "spearman")

print("相关性矩阵：")
print(corr_matrix)



corr_matrix_lower <- corr_matrix
corr_matrix_lower[upper.tri(corr_matrix_lower)] <- NA
corr_matrix_lower[diag(corr_matrix_lower)] <- NA
diag(corr_matrix_lower)<-NA


display_matrix <- round(corr_matrix_lower, 2)
display_matrix[is.na(display_matrix)] <- ""


pdf("cor1.pdf",10,8)
pheatmap(corr_matrix_lower, 
         display_numbers = display_matrix, 
         number_format = "%.2f",  # 设置数值显示格式
         color = colorRampPalette(c("blue", "white", "red"))(50),
         cluster_rows = FALSE,  # 不聚类行
         cluster_cols = FALSE,  # 不聚类列
         na_col = "white",
         main = "Lower Triangle Correlation Heatmap",
         display_numbers_color = "white",
         border_color = NA,
         fontsize_number=4)
dev.off()






# 手动修改字体颜色
draw <- function() {
  grid.draw(p$gtable)  # 先绘制热图
  grid.text(display_matrix, 
            x = rep(p$gtable$layout$l[4:length(p$gtable$layout$l)], each = ncol(corr_matrix_lower)) / max(p$gtable$layout$r),
            y = rep(rev(p$gtable$layout$t[4:length(p$gtable$layout$t)]), nrow(corr_matrix_lower)) / max(p$gtable$layout$b),
            gp = gpar(col = "white", fontsize = 10))  # 手动设置字体为白色
}

grid.newpage()
draw()




####
Proteomics.data2=proteomics.data1
na_threshold <- 0.75
data_filtered <- Proteomics.data[rowMeans(is.na(Proteomics.data)) <= na_threshold, ]
data_filtered[is.na(data_filtered)]=min(Proteomics.data,na.rm = TRUE)
data_filtered=log2(data_filtered)
cv <- apply(data_filtered, 1, function(x) sd(x, na.rm=TRUE) / mean(x, na.rm=TRUE))
data_var <- data_filtered[cv > 0.25, ]  # 仅保留CV > 0.2 的蛋白
cat("变异度过滤后剩余蛋白数量:", nrow(data_var), "\n")
library(limma)
TS <-c(rep(1,27),rep(2,33)) 
TS<-ifelse(TS==2,"trearment","control")
design2 <- model.matrix(~0+TS)
colnames(design2) <- c( "control","treatment")
rfit <- lmFit(data_var,design2,method="ls")
cont.matrix <- makeContrasts(contrast=treatment-control, levels=design2)
rfit <- contrasts.fit(rfit, cont.matrix)
rfit <- eBayes(rfit,trend=TRUE,robust = TRUE)
DE.protein<-topTable(rfit, coef="contrast", number=nrow(Proteomics.data),confint=TRUE,
                     adjust="BH")
DE.final=subset(DE.protein,abs(logFC)>1&adj.P.Val<0.1)

DE.protein$ID=rownames(DE.protein)
library(dplyr)
DE.protein <- DE.protein %>%
  mutate(ID_clean = sub(";.*", "", ID))
df2 <- DE.protein %>%
  left_join(ID, by = c("ID_clean" = "ID"))

save(list=ls(),file="proteomics.analysis.2025.11.22.RData")
rm(list=ls())

################3################3################3################3################3 pathway for PPI analysis

library(dplyr)
library(igraph)
library(readr)
ppi_raw <- read_tsv("string_interactions_short.tsv", show_col_types = FALSE)
colnames(ppi_raw)[1]="node1"

ppi <- ppi_raw %>%
  dplyr::select(node1, node2, combined_score) 
colnames(ppi) <- c("protein1", "protein2", "score")

########################################
# 2. 完成Walktrap clustering
########################################
g <- graph_from_data_frame(ppi[,1:2], directed = FALSE)
E(g)$weight <- ppi$score
# Walktrap clustering
cl <- cluster_walktrap(g, weights = E(g)$weight)
V(g)$cluster <- membership(cl)

table(V(g)$cluster)

cluster_df <- data.frame(
  gene = V(g)$name,
  cluster = V(g)$cluster
)

########################################
# 2. 每个 cluster 的基因提取出来
########################################
clusters <- split(cluster_df$gene, cluster_df$cluster)
# 去除小模块（节点太少无意义）
clusters <- clusters[sapply(clusters, length) >= 8]
length(clusters)
########################################
# 3. 对每个 cluster 做 pathway 富集（Reactome + GO BP + GO MF）
########################################
library(clusterProfiler)
library(stringr)

enrich_cluster <- function(gene_vector){
  
  # 不做任何过滤，把完整结果拿出来
  res_reactome <- enricher(
    gene_vector,
    TERM2GENE = final.pathway,
    pvalueCutoff = 1,
    qvalueCutoff = 1,
    minGSSize = 4
  )@result
  
  res_gobp <- enricher(
    gene_vector,
    TERM2GENE = GOBP.pathway,
    pvalueCutoff = 1,
    qvalueCutoff = 1,
    minGSSize = 4
  )@result
  
  res_gomf <- enricher(
    gene_vector,
    TERM2GENE = GOMF.pathway,
    pvalueCutoff = 1,
    qvalueCutoff = 1,
    minGSSize = 4
  )@result
  
  # 合并
  df <- bind_rows(
    res_reactome %>% mutate(DB = "Reactome"),
    res_gobp %>% mutate(DB = "GO_BP"),
    res_gomf %>% mutate(DB = "GO_MF")
  )
  
  # 手动根据 p.adjust 过滤
  df %>%
    filter(Count >= 4) %>%
    filter(!is.na(p.adjust)) %>%
    filter(p.adjust < 0.05)
}


cluster_path_list <- lapply(clusters, enrich_cluster)

all_cluster_pathways_full <- bind_rows(cluster_path_list, .id = "cluster_id")
########################################
# 3. 每个 cluster 内做 pathway 去冗余（Jaccard + hclust）
########################################
reduce_pathways <- function(df){
  
  if(nrow(df) < 2) return(df)
  
  geneSets <- lapply(df$geneID, function(x) str_split(x, "/")[[1]])
  names(geneSets) <- df$Description
  
  # 计算 Jaccard 矩阵
  jaccard <- function(a, b) length(intersect(a, b)) / length(union(a, b))
  n <- length(geneSets)
  mat <- matrix(0, n, n)
  for (i in 1:n){
    for (j in 1:n){
      mat[i,j] <- jaccard(geneSets[[i]], geneSets[[j]])
    }
  }
  
  dist_mat <- as.dist(1 - mat)
  
  hc <- hclust(dist_mat, method = "average")
  clusters <- cutree(hc, h = 0.4)
  
  df$cluster <- clusters
  
  df_sel <- df %>%
    group_by(cluster) %>%
    slice_min(order_by = p.adjust, n = 1) %>%
    ungroup()
  
  df_sel
}

cluster_path_sel <- lapply(cluster_path_list, reduce_pathways)

########################################
# 3. 构建一个全局 PPI + Pathway 融合网络
########################################

library(tidygraph)
library(ggraph)
library(tidyr)

# 合并所有 selected pathways
df_sel_all <- bind_rows(cluster_path_sel, .id = "cluster_id")

gene_path_edges <- df_sel_all %>%
  separate_rows(geneID, sep = "/") %>%
  rename(pathway = Description,
         gene = geneID)

# 所有基因节点
node_genes <- V(g)$name
node_pathways <- unique(gene_path_edges$pathway)

nodes <- data.frame(
  name = c(node_genes, node_pathways),
  type = c(
    rep("gene", length(node_genes)),
    rep("pathway", length(node_pathways))
  )
)

ppi_edges <- ppi %>% select(from = protein1, to = protein2) %>% mutate(type = "ppi")
gene_path_edges2 <- gene_path_edges %>% rename(from = gene, to = pathway) %>% mutate(type = "membership")

edges <- bind_rows(ppi_edges, gene_path_edges2)

g_all <- tbl_graph(nodes = nodes, edges = edges, directed = FALSE)
########################################
# 3. 专业可发表级别的 PPI–Pathway 综合网络图
########################################

set.seed(123)
ggraph(g_all, layout = "fr") +
  
  geom_edge_link(aes(color = type), alpha = 0.3) +
  
  geom_node_point(aes(color = type, size = ifelse(type=="gene", 3, 8)),
                  alpha = 0.9) +
  
  geom_node_text(
    data = function(x) x %>% filter(type=="pathway"),
    aes(label = name),
    repel = TRUE,
    size = 4,
    fontface = "bold"
  ) +
  
  geom_node_text(
    data = function(x) x %>%
      filter(type=="gene") %>%
      filter(centrality_degree() > quantile(centrality_degree(), 0.95)),
    aes(label = name),
    repel = TRUE,
    size = 3
  ) +
  
  scale_color_manual(values = c(
    gene = "#2C7BB6",
    pathway = "#D7191C",
    ppi = "grey70",
    membership = "#FDAE61"
  )) +
  
  theme_void() +
  ggtitle("Integrated PPI–Pathway Network by Cluster")

########################################
# PPI + Pathway 综合网络（Cluster 着色）
########################################

set.seed(123)

ggraph(g_all, layout = "fr") +
  
  geom_edge_link(aes(color = type), alpha = 0.3) +
  
  geom_node_point(aes(color = type, size = ifelse(type=="gene", 3, 8)),
                  alpha = 0.9) +
  
  geom_node_text(
    data = function(x) x %>% filter(type=="pathway"),
    aes(label = name),
    repel = TRUE,
    size = 4,
    fontface = "bold"
  )+
  
  
  scale_color_manual(
    values = c(
      "pathway" = "#D7191C",
      "1" = "#1b9e77",
      "2" = "#d95f02",
      "3" = "#7570b3",
      "4" = "#e7298a",
      "5" = "#66a61e",
      "6" = "#e6ab02",
      "7" = "#a6761d",
      "8" = "#666666"
    )
  ) +
  
  theme_void() +
  ggtitle("Integrated PPI–Pathway Network (Gene colored by Cluster)")

library(dplyr)
library(tidyr)

# 计算热图矩阵
heatmap_df <- all_cluster_pathways_full %>%
  mutate(cluster_id = paste0("Cluster_", cluster_id)) %>%
  mutate(pathway = Description) %>%
  mutate(value = -log10(p.adjust)) %>%   # 热图信号
  select(cluster_id, pathway, value) %>%
  spread(cluster_id, value, fill = 0)    # 没有富集则填 0

mat <- as.matrix(heatmap_df[,-1])
rownames(mat) <- heatmap_df$pathway

pheatmap(mat,
         color = colorRampPalette(c("white", "#2c7bb6", "#d7191c"))(100),
         border_color = NA,
         fontsize = 10,
         fontsize_row = 8,
         clustering_distance_rows = "euclidean",
         clustering_method = "average",
         main = "Cluster-specific Enriched Pathways")


library(ComplexHeatmap)
Heatmap(mat,
        name = "-log10(adj p)",
        col = colorRampPalette(c("white", "#fee8c8", "#fdae6b", "#e6550d"))(100),
        cluster_rows = TRUE,
        cluster_columns = TRUE,
        show_row_names = TRUE,
        show_column_names = TRUE,
        row_names_gp = grid::gpar(fontsize = 8),
        column_names_gp = grid::gpar(fontsize = 10),
        heatmap_legend_param = list(
          title = "-log10(p.adjust)",
          at = c(0, 1, 2, 3, 4),
          labels = c("0", "1", "2", "3", "4+")
        ))



########################################
# 2. 过滤低可信 PPI（非常关键）
########################################
ppi <- ppi %>%
  filter(!is.na(protein1) & !is.na(protein2)) %>%
  filter(protein1 != "" & protein2 != "") %>%
  filter(protein1 != protein2) %>%     # 去除自互作
  distinct(protein1, protein2, score)  # 去重

ppi <- ppi %>% filter(score >= 0.5)

cat("Nodes before graph build:", length(unique(c(ppi$protein1, ppi$protein2))), "\n")
cat("Edges after filtering:", nrow(ppi), "\n")
g <- graph_from_data_frame(ppi[, 1:2], directed = FALSE)

# 添加 STRING 置信度作为边权重
E(g)$weight <- ppi$score
cat("Graph nodes:", vcount(g), "\n")
cat("Graph edges:", ecount(g), "\n")

cl <- cluster_walktrap(g, weights = E(g)$weight)

# 提取模块标签
module <- membership(cl)
V(g)$module <- module
cat("Module sizes:\n")
print(table(module))

module_df <- data.frame(
  gene   = V(g)$name,
  module = V(g)$module
)

tg <- as_tbl_graph(g) %>%
  mutate(module = as.factor(module),
         degree = centrality_degree())

########################################
# 4. ggraph 绘制 PPI 网络
########################################

set.seed(123)   # 确保布局可复现

p <- 
  pdf("x.pdf",12,10)
ggraph(tg, layout = "fr") +   # Fruchterman–Reingold 布局（常用）
  # edges
  geom_edge_link(aes(alpha = ..index..),
                 colour = "grey70",
                 show.legend = FALSE) +
  # nodes
  geom_node_point(aes(color = module,
                      size = degree),
                  alpha = 0.9) +
  # 隐藏小节点标签，保留 hub genes
  geom_node_text(aes(label = ifelse(degree > quantile(degree, 0.90),
                                    name, "")),
                 repel = TRUE,
                 size = 3) +
  scale_size_continuous(range = c(1, 10)) +
  scale_color_brewer(palette = "Dark2") +
  theme_void(base_size = 14) +
  ggtitle("PPI Network Colored by Walktrap Modules")
dev.off()

print(p)



方案 1（强烈推荐）🔵：按模块拆成多个子网络图（每个模块一张图）


modules <- unique(tg$module)

for (m in modules) {
  subg <- tg %>% filter(module == m)
  
  p <- ggraph(subg, layout = "kk") +
    geom_edge_link(alpha = 0.3, color = "grey70") +
    geom_node_point(aes(size = degree),
                    color = RColorBrewer::brewer.pal(8,"Dark2")[as.numeric(m)],
                    alpha = 0.9) +
    geom_node_text(aes(label = name), repel = TRUE, size = 3) +
    scale_size_continuous(range = c(3, 12)) +
    theme_void() +
    ggtitle(paste("Module", m))
  print(p)
}


方案 2：只绘制顶层骨架网络（去掉度小于阈值的节点）
用 degree 过滤掉 leaf nodes（让图更集中）：

tg2 <- tg %>% filter(degree >= 3)
ggraph(tg2, layout = "kk") +
  geom_edge_link(alpha = 0.4, color = "grey70") +
  geom_node_point(aes(color = module, size = degree), alpha = 0.9) +
  geom_node_text(aes(label = name), repel = TRUE, size = 3) +
  theme_void()

方案 3：取每个模块的 Top N hub genes 构成“meta network”
包括跨模块的 cross-talk（更有生物启发意义）

hub_genes <- tg %>%
  group_by(module) %>%
  slice_max(order_by = degree, n = 3) %>%  # 每个模块取 15 个 hub
  pull(name)

tg3 <- tg %>% filter(name %in% hub_genes)

ggraph(tg3, layout = "kk") +
  geom_edge_link(alpha = 0.4, color = "grey80") +
  geom_node_point(aes(color = module, size = degree), alpha = 0.95) +
  geom_node_text(aes(label = name), repel = TRUE, size = 3.5) +
  theme_void()



方案 4：使用“社区布局”（layout = 'graphopt' 或 'lgl'）
ggraph(tg, layout = "lgl") +
  geom_edge_link(color = "grey80", alpha = 0.4) +
  geom_node_point(aes(color = module, size = log1p(degree)), alpha = 0.9) +
  theme_void()












V(g)$module <- cl$membership
table(V(g)$module)



cl_walk  <- cluster_walktrap(g, weights = E(g)$weight)



cl_fast  <- cluster_fast_greedy(g, weights = E(g)$weight)
optimal <- cutatoptimal(cl_fast)
V(g)$module <- optimal
table(optimal)



cl_mcl   <- cluster_markov_clustering(g, inflation = 1.6, weights = E(g)$weight,addLoops = TRUE)

adj <- as_adjacency_matrix(g, attr = "weight",addLoops = TRUE)
mcl_res <- mcl(adj, inflation = 1.6)


table(membership(cl_walk))


table(membership(cl_fast))




table(cl_mcl$membership)







ppi <- ppi_raw[, c(1:2,13)]
colnames(ppi) <- c("protein1", "protein2", "score")
ppi <- ppi %>%
  filter(protein1 != protein2) %>%
  distinct(protein1, protein2, score)

library(igraph)

g <- graph_from_data_frame(ppi[, 1:2], directed = FALSE)
E(g)$weight <- ppi$score

# 你可以加权（如果 score 是数值）
# E(g)$weight <- as.numeric(ppi$score)
cl <- cluster_leiden(
  g,
  weights = E(g)$weight,               # 使用 STRING 置信度
  resolution_parameter = 1.0
)
V(g)$module <- cl$membership
table(V(g)$module)
module_df <- data.frame(
  gene = V(g)$name,
  module = V(g)$module
)



# 查看前几行确认
head(ppi_raw)
ppi <- ppi_raw %>%
  filter(protein1 != protein2) %>%     # 去掉自互作
  distinct(protein1, protein2, score)  # 去重
###############################
# 3. 构建 igraph 网络
###############################
g <- graph_from_data_frame(ppi[, 1:2], directed = FALSE)
cl <- cluster_leiden(g, resolution_parameter = 1.0)

V(g)$module <- cl$membership

# 打印每个 module 的大小
cat("Module sizes:\n")
print(table(V(g)$module))

###############################
# 5. 导出 module → 基因 的映射
###############################
module_df <- data.frame(
  gene = V(g)$name,
  module = V(g)$module
)

# 按 module 排序
module_df <- module_df %>% arrange(module)

head(module_df)
###############################
# 6. 筛选最小模块
###############################
min_module_size <- 10

valid_modules <- names(table(V(g)$module)[table(V(g)$module) >= min_module_size])

valid_genes <- module_df %>%
  filter(module %in% valid_modules)

cat("保留的 module 数：", length(valid_modules), "\n")
cat("保留的基因数：", nrow(valid_genes), "\n")






proteins=read.table(text = read_clip(), 
                    header = TRUE, sep = "\t", stringsAsFactors = FALSE)

genes_group3 <- Subgroups %>%
  dplyr::filter(Modules.membership ==4) %>%
  dplyr::pull(Modules.names)
genes_group3=c(genes_group3)

library(clusterProfiler)
library(tidyverse)
library(ComplexUpset)
library(enrichplot)
library(clusterProfiler)
library(enrichplot)
library(org.Hs.eg.db)
library(dplyr)
library(stringr)
library(simplifyEnrichment)   # ⭐ 用于去除通路冗余
library(AnnotationDbi)

pathway=enricher(genes_group3, pvalueCutoff = 0.05, pAdjustMethod = "BH", 
                 minGSSize = 4, maxGSSize = 500, qvalueCutoff = 0.05, TERM2GENE=final.pathway,
                 TERM2NAME = NA)
Down.pathway=pathway@result
pathway=enricher(genes_group3, pvalueCutoff = 0.05, pAdjustMethod = "BH", 
                 minGSSize = 4, maxGSSize = 500, qvalueCutoff = 0.2, TERM2GENE=GOBP.pathway,
                 TERM2NAME = NA)
Down.BP=pathway@result
pathway=enricher(genes_group3, pvalueCutoff = 0.05, pAdjustMethod = "BH", 
                 minGSSize = 4, maxGSSize = 500, qvalueCutoff = 0.2, TERM2GENE=GOMF.pathway,
                 TERM2NAME = NA)
Down.MF=pathway@result

pathway=rbind(Down.pathway,Down.BP)

df <- pathway %>% filter(Count >= 4)
# 将 geneID 转换为列表
df$geneID <- as.character(df$geneID)
geneSets <- lapply(df$geneID, function(x) str_split(x, "/")[[1]])
names(geneSets) <- df$Description
all_genes <- unique(unlist(geneSets))
gene2Symbol <- all_genes
names(gene2Symbol) <- all_genes
er <- new(
  "enrichResult",
  result = df %>%
    mutate(
      pvalue = p.adjust,   # 如没有 p.adjust，请改成对应列
      qvalue = p.adjust,
      geneID = geneID
    ),
  pvalueCutoff = 1,
  pAdjustMethod = "BH",
  qvalueCutoff = 1,
  geneSets = geneSets,
  gene2Symbol = gene2Symbol,
  organism = "human",
  keytype = "SYMBOL",
  readable = TRUE
)
cnetplot(er,
         showCategory = 6,
         circular = TRUE,
         colorEdge = TRUE) +
  theme_minimal()


############################################################################################################
genes <- read.table(text = read_clip(), 
                    header = TRUE, sep = "\t", stringsAsFactors = FALSE)
genes=genes[,1]
# ① Reactome or 自定义 final.pathway
reactome_res <- enricher(
  genes,
  TERM2GENE = final.pathway,
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.2,
  minGSSize = 4
)@result
# ② GO BP
gobp_res <- enricher(
  genes,
  TERM2GENE = GOBP.pathway,
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.2,
  minGSSize = 4
)@result
# ③ GO MF
gomf_res <- enricher(
  genes,
  TERM2GENE = GOMF.pathway,
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.2,
  minGSSize = 4
)@result

df_all <- bind_rows(
  reactome_res %>% mutate(DB = "Reactome"),
  gobp_res %>% mutate(DB = "GO_BP"),
  gomf_res %>% mutate(DB = "GO_MF")
)
df_all <- df_all %>% filter(Count >= 4&p.adjust<0.05)

geneSets <- lapply(df_all$geneID, function(x) str_split(x, "/")[[1]])
names(geneSets) <- df_all$Description
# -----------------------------
# 3. 计算 Jaccard 基因集相似度矩阵
# -----------------------------
jaccard <- function(a, b) length(intersect(a, b)) / length(union(a, b))
n <- length(geneSets)
mat <- matrix(0, n, n)
for (i in 1:n){
  for (j in 1:n){
    mat[i,j] <- jaccard(geneSets[[i]], geneSets[[j]])
  }
}
rownames(mat) <- colnames(mat) <- names(geneSets)
# 距离矩阵 = 1 - 相似度
dist_mat <- as.dist(1 - mat)
# -----------------------------
# 4. 层次聚类去冗余（核心）
# -----------------------------
hc <- hclust(dist_mat, method = "average")
# h 越小 → 越严格去冗余（0.3~0.5之间调节）
clusters <- cutree(hc, h = 0.7)
df_all$cluster <- clusters
# -----------------------------
# 5. 每个 cluster 选最显著 pathway
# -----------------------------
df_sel <- df_all %>%
  group_by(cluster) %>%
  slice_min(order_by = p.adjust, n = 1) %>%
  ungroup()
# -----------------------------
# 6. 构造 enrichResult（不会报错）
# -----------------------------
geneSets_sel <- lapply(df_sel$geneID, function(x) str_split(x, "/")[[1]])
names(geneSets_sel) <- df_sel$Description

all_genes <- unique(unlist(geneSets_sel))
gene2Symbol <- all_genes
names(gene2Symbol) <- all_genes

er <- new(
  "enrichResult",
  result = df_sel %>%
    mutate(
      pvalue = p.adjust,
      qvalue = p.adjust
    ),
  pvalueCutoff = 1,
  qvalueCutoff = 1,
  pAdjustMethod = "BH",
  geneSets = geneSets_sel,
  gene2Symbol = gene2Symbol,
  keytype = "SYMBOL",
  organism = "human",
  readable = TRUE
)
cnetplot(er,
         showCategory = 5,  # 展示所有代表 pathway
         circular = TRUE,
         colorEdge = TRUE) +
  theme_minimal(base_size = 14)















mat <- simplifyEnrichment::term_similarity(
  df_all$Description,
  method = "jaccard",
  term_gene = df_all$geneID
)
# 聚类
cl <- simplifyEnrichment::cluster_terms(mat)

# 取每个 cluster 的代表通路（代表性最高）
sel <- simplifyEnrichment::select_representative_terms(mat, cl)

df_sel <- df_all[sel, ]








# 创建 enrichResult 需要的格式
enrich_df <- df %>%
  mutate(
    pvalue = p.adjust,   # 如果没有 p.adjust，请写 df$pvalue
    qvalue = p.adjust,   # 同上
    geneID = geneID,
    Count = Count
  )

er <- new(
  "enrichResult",
  result = enrich_df,
  pvalueCutoff = 1,
  pAdjustMethod = "BH",
  qvalueCutoff = 1,
  geneSets = NULL,
  gene2Symbol = NULL,
  readable = TRUE
)








df <- ego %>% 
  as.data.frame() %>% 
  separate_rows(geneID, sep="/")

head(df)


