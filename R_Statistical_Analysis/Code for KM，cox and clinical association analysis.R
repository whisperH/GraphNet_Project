############################################################
## Basic statistical evaluation pipeline for G-TRS model ##
############################################################

############################################################
## Clinical association evaluation pipeline for G-TRS model ##
############################################################

## Load required libraries
library(survminer)
library(ROCR)
library(pROC)
library(survival)
library(timeROC)
library(clipr)
library(dplyr)
library(caret)
library(PRROC)

############################################################
## 1. Load training cohort (macOS clipboard input)
############################################################
## Read tab-delimited data copied to clipboard
Train.data=read.table(text = read_clip(), 
                                    header = TRUE, sep = "\t", stringsAsFactors = FALSE)

############################################################
## 2. Determine optimal cutoff for G-TRS using resampling
############################################################
## Use repeated random subsampling (80%) to stabilize cutoff
set.seed(123)
folds <-10000
thresholds <- numeric(folds)
for (i in 1:folds) {
  idx <- sample(seq_len(nrow(Train.data)), size = 0.8 * nrow(Train.data))
  roc_obj <- roc(Train.data$RFS_status[idx], Train.data$G.TRS[idx])
  roc_coords <- coords(roc_obj, "all", ret=c("threshold", "sensitivity", "specificity"))
  youden_index <- roc_coords$sensitivity + roc_coords$specificity - 1
  thresholds[i] <- roc_coords$threshold[which.max(youden_index)]
}
## Final cutoff defined as the mean of resampled thresholds
best_threshold <- mean(thresholds)
best_threshold
## Example output: 0.5358274

############################################################
## 3. Define G-TRS risk subtypes based on fixed cutoff
############################################################
## High-risk = 1, Low-risk = 0
Train.data$G.TRS_subtype=ifelse(Train.data$G.TRS<0.5358274,0,1)
table(Train.data$G.TRS_subtype)
## Example: low-risk = 144, high-risk = 100

############################################################
## 4. Binary classification performance (recurrence)
############################################################
## Convert yes/no outcomes to binary (1 = event)
Train.data2 <- Train.data %>%
  mutate(
    RFS_status = ifelse(RFS_status == "yes", 1, 0),
    OS_status = ifelse(OS_status == "yes", 1, 0)
  )

## Confusion matrix (positive class = recurrence)
cm <- confusionMatrix(as.factor(Train.data2$G.TRS_subtype), 
         as.factor(Train.data2$RFS_status), positive = "1")
## Extract precision, recall, and F1-score
precision <- cm$byClass["Precision"]
recall <- cm$byClass["Recall"]
f1 <- cm$byClass["F1"]
############################################################
## 5. Discrimination performance (continuous G-TRS)
############################################################
## AUROC using continuous G-TRS values
roc_obj <- roc(Train.data2$RFS_status, Train.data2$G.TRS)
auc_value <- auc(roc_obj)
## Area under precision–recall curve (AUPR)
AUPR <- pr.curve(scores.class0 = Train.data2$G.TRS[Train.data2$RFS_status == 1],
                 scores.class1 = Train.data2$G.TRS[Train.data2$RFS_status == 0],
                 curve = TRUE)
aupr <- AUPR$auc.integral
## Output performance metrics
cat("Precision:", precision, "\n") ### 0.75 
cat("Recall:", recall, "\n") ## 0.8522727
cat("F1-score:", f1, "\n") ## 0.7978723
cat("AUROC:", auc_value, "\n") ## 0.9227855
cat("AUPR:", aupr, "\n") ## 0.8626883 

############################################################
## 6. Survival analysis (Kaplan–Meier and log-rank test)
############################################################
## Recurrence-free survival (RFS)
RFS.fit = survfit(Surv(RFS_Time_Month,RFS_status)~factor(G.TRS_subtype), 
                  data=Train.data2,conf.type="log-log")

RFS.log.rank = survdiff(Surv(RFS_Time_Month, RFS_status) ~ factor(G.TRS_subtype),
                        data = Train.data2)
RFS.p.value=RFS.log.rank$pvalue
## Overall survival (OS)
OS.fit = survfit(Surv(OS_Time_Month, OS_status)~factor(G.TRS_subtype), 
                 data=Train.data2,conf.type="log-log")
OS.log.rank= survdiff(Surv(OS_Time_Month, OS_status)~factor(G.TRS_subtype), 
                      data=Train.data2)
OS.p.value=OS.log.rank$pvalue

############################################################
## 7. Subgroup survival analysis (Milan / UCSF criteria)
############################################################
## Milan-eligible patients
Milan_yes <- Train.data2 %>%
  filter(Milan == "yes")
Milan.yes.RFS.fit= survfit(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype), 
                           data=Milan_yes,conf.type="log-log")
Milan.yes.RFS.log.rank=survdiff(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype), 
                                data=Milan_yes)
## Milan-ineligible patients
Milan_no<- Train.data2 %>%
  filter(Milan == "no")
Milan.no.RFS.fit= survfit(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype), 
                          data=Milan_no,conf.type="log-log")
Milan.no.RFS.log.rank=survdiff(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype), 
                               data=Milan_no)
## UCSF-eligible patients
UCSF_yes<- Train.data2 %>%
  filter(UCSF == "yes")
UCSF.yes.RFS.fit = survfit(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype), data=UCSF_yes,conf.type="log-log")
UCSF.yes.RFS.log.rank = survdiff(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype), data=UCSF_yes)
## UCSF-ineligible patients
UCSF_no<- Train.data2 %>%
  filter(UCSF == "no")
UCSF.no.RFS.fit = survfit(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype), data=UCSF_no,conf.type="log-log")
UCSF.no.RFS.log.rank = survdiff(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype), data=UCSF_no)


############################################################
## External validation cohorts (YA / CY / HS-2)
############################################################
## NOTE:
## The same cutoff derived from the training cohort is applied
## to all external validation cohorts without re-optimization.
## The evaluation pipeline (classification, ROC/PR, survival)
## is identical to that used in the training cohort.

############################################################
## 1. Load training cohort (macOS clipboard input)
############################################################
## Load YA cohort from clipboard (same format as training data)
YA.data=read.table(text = read_clip(), 
                 header = TRUE, sep = "\t", stringsAsFactors = FALSE)
############################################################
## 2. Apply fixed G-TRS cutoff derived from training cohort
############################################################
## IMPORTANT:
## The cutoff (best_threshold) is fixed and NOT re-optimized
## in the validation cohort to avoid information leakage.
YA.data$G.TRS_subtype=ifelse(YA.data$G.TRS<0.5358274,0,1)
## Example: low-risk = 108, high-risk = 81
## Example: low-risk = 108, high-risk = 81

############################################################
## 3. Convert survival outcomes to binary format
############################################################
YA.data2 <- YA.data %>%
  mutate(
    RFS_status = ifelse(RFS_status == "yes", 1, 0),
    OS_status = ifelse(OS_status == "yes", 1, 0)
  )
############################################################
##4.Binary classification performance (recurrence)
############################################################
## Confusion matrix with recurrence as positive class
cm <- confusionMatrix(as.factor(YA.data2$G.TRS_subtype), 
                      as.factor(YA.data2$RFS_status), positive = "1")
## Extract performance metrics
precision <- cm$byClass["Precision"]
recall <- cm$byClass["Recall"]
f1 <- cm$byClass["F1"]
############################################################
##5. Discrimination performance using continuous G-TRS
############################################################
## AUROC
roc_obj <- roc(YA.data2$RFS_status, YA.data2$G.TRS)
auc_value <- auc(roc_obj)
## AUPR (Precision–Recall curve)
AUPR <- pr.curve(scores.class0 = YA.data2$G.TRS[YA.data2$RFS_status == 1],
                 scores.class1 = YA.data2$G.TRS[YA.data2$RFS_status == 0],
                 curve = TRUE)
aupr <- AUPR$auc.integral
## Output validation performance
cat("Precision:", precision, "\n") ###0.3580247 
cat("Recall:", recall, "\n") ## 0.8055556 
cat("F1-score:", f1, "\n") ##0.4957265 
cat("AUROC:", auc_value, "\n") ## 0.7632534 
cat("AUPR:", aupr, "\n") ## 0.4877594   

############################################################
## Survival analysis in YA cohort
############################################################
## Recurrence-free survival (RFS)
RFS.fit = survfit(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype),
                  data=YA.data2,conf.type="log-log")
RFS.log.rank = survdiff(Surv(RFS_Time_Month, RFS_status) ~ factor(G.TRS_subtype), 
                        data = YA.data2)
YA.RFS.pvalue=RFS.log.rank$pvalue
## Overall survival (OS)
OS.fit = survfit(Surv(OS_Time_Month, OS_status)~factor(G.TRS_subtype), 
                 data=YA.data2,conf.type="log-log")
OS.log.rank= survdiff(Surv(OS_Time_Month, OS_status)~factor(G.TRS_subtype), 
                      data=YA.data2)
YA.OS.pvalue=OS.log.rank$pvalue

############################################################
## Subgroup survival analysis in YA cohort
############################################################
## Milan-eligible patients
Milan_yes <- YA.data2 %>%
  filter(Milan == "yes")
Milan.yes.RFS.fit= survfit(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype), 
                           data=Milan_yes,conf.type="log-log")
Milan.yes.RFS.log.rank=survdiff(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype), 
                                data=Milan_yes)
## Milan-ineligible patients
Milan_no<- YA.data2 %>%
  filter(Milan == "no")
Milan.no.RFS.fit= survfit(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype), 
                          data=Milan_no,conf.type="log-log")
Milan.no.RFS.log.rank=survdiff(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype),
                               data=Milan_no)

## UCSF-eligible patients
UCSF_yes<- YA.data2 %>%
  filter(UCSF == "yes")
UCSF.yes.RFS.fit = survfit(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype), 
                           data=UCSF_yes,conf.type="log-log")
UCSF.yes.RFS.log.rank = survdiff(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype), 
                                 data=UCSF_yes)
## UCSF-ineligible patients
UCSF_no<- YA.data2 %>%
  filter(UCSF == "no")
UCSF.no.RFS.fit = survfit(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype), 
                          data=UCSF_no,conf.type="log-log")
UCSF.no.RFS.log.rank = survdiff(Surv(RFS_Time_Month, RFS_status)~factor(G.TRS_subtype), 
                                data=UCSF_no)

############################################################
## Notes for additional external cohorts (CY / HS-2)
############################################################

## CY and HS-2 cohorts should be processed using the same
## pipeline as YA:
## 1) Apply the fixed cutoff from the training cohort
## 2) Evaluate binary classification performance
## 3) Compute AUROC and AUPR using continuous G-TRS
## 4) Perform survival and subgroup analyses


############################################################
## Cox proportional hazards analysis for RFS and OS
## Univariate and multivariate models
############################################################
# ==========================================================
# 1. Load required packages
# ==========================================================
library(survival)
library(dplyr)
library(tidyr)
library(ggplot2)
library(forcats)

# ==========================================================
# 2. Read input dataset
#    Data are assumed to be tab-delimited and copied
#    to the clipboard.
# ==========================================================
Train.data=read.table(text = read_clip(), 
            header = TRUE, sep = "\t", stringsAsFactors = FALSE)
# ==========================================================
# 3. Recode outcome variables and covariates
#    All variables are dichotomized into binary form:
#      1 = adverse / high-risk category
#      0 = reference / low-risk category
#
#    This explicit binary encoding avoids implicit factor
#    reference levels and ensures direct interpretability
#    of hazard ratios (HRs).
# ==========================================================
Train.cox <- Train.data %>%
  mutate(
    RFS_status = ifelse( RFS_status== "yes", 1, 0),
    OS_status = ifelse(OS_status== "yes", 1, 0),
    G.TRS_subtype = ifelse(G.TRS_subtype == "high-risk", 1, 0),
    UCSF  = ifelse(UCSF  == "yes", 1, 0),
    Milan = ifelse(Milan == "yes", 1, 0),
    PVTT = ifelse(PVTT == "yes", 1, 0),
    AJCC = ifelse(AJCC=="T3-T4", 1, 0),
    MVI_status = ifelse(MVI_status == "yes", 1, 0),
    MTD_status = ifelse(MTD_status ==">5", 1, 0),
    Tumor_Number = ifelse(Tumor_Number == "Multi", 1, 0),
    AFP_status = ifelse(AFP_status== "＞200", 1, 0),
    PT_status = ifelse(PT_status== "＞13", 1, 0),
    TBil_status = ifelse(TBil_status== "＞23", 1, 0),
    ALB_status = ifelse(ALB_status== "＞35", 1, 0),
    PLT_status = ifelse(PLT_status== "≤100", 1, 0),
    Age_status = ifelse(Age_status== "＞60", 1, 0),
    Gender = ifelse(Gender == "male", 1, 0),
  )
# ==========================================================
# 4. Define covariates included in Cox regression
# ==========================================================
vars <- c("G.TRS_subtype", "UCSF", "Milan", "PVTT", "AJCC", 
          "MVI_status", "MTD_status", "Tumor_Number", "AFP_status",
          "PT_status", "Age_status",
          "ALB_status", "PLT_status", "Gender","TBil_status")

############################################################
## Part I: Recurrence-Free Survival (RFS)
############################################################
# ==========================================================
# 5. Univariate Cox analysis for RFS
#    Each covariate is evaluated independently.
# ==========================================================
univ_RFS <- lapply(vars, function(v){
  fml <- as.formula(paste("Surv(RFS_Time_Month, RFS_status) ~", v))
  fit <- coxph(fml, data = Train.cox)
  s <- summary(fit)
  data.frame(
    Variable = v,
    HR = s$coef[,"exp(coef)"],
    lower = s$conf.int[,"lower .95"],
    upper = s$conf.int[,"upper .95"],
    p = s$coef[,"Pr(>|z|)"]
  )
}) %>% bind_rows()
# Sort univariate results by ascending p value
univ_RFS <- univ_RFS %>%
  arrange(p)
# ==========================================================
# 6. Multivariate Cox analysis for RFS
#    Only covariates with p < 0.05 in univariate analysis
#    are included.
# ==========================================================
sig_vars <- univ_RFS %>%
  filter(p < 0.05) %>%
  pull(Variable)
sig_vars
multi_formula <- as.formula(
  paste("Surv(RFS_Time_Month, RFS_status) ~", 
        paste(sig_vars, collapse = " + "))
)
multi_formula
multi_cox <- coxph(multi_formula, data = Train.cox)
multi_sum <- summary(multi_cox)
multi_RFS <- data.frame(
  Variable = rownames(multi_sum$coef),
  HR       = multi_sum$coef[, "exp(coef)"],
  lower    = multi_sum$conf.int[, "lower .95"],
  upper    = multi_sum$conf.int[, "upper .95"],
  p        = multi_sum$coef[, "Pr(>|z|)"]
) %>%
  arrange(p)


############################################################
## Part II: Overall Survival (OS)
############################################################
# ==========================================================
# 7. Univariate Cox analysis for OS
# ==========================================================
univ_OS <- lapply(vars, function(v){
  fml <- as.formula(paste("Surv(OS_Time_Month, OS_status) ~", v))
  fit <- coxph(fml, data = Train.cox)
  s <- summary(fit)
  data.frame(
    Variable = v,
    HR = s$coef[,"exp(coef)"],
    lower = s$conf.int[,"lower .95"],
    upper = s$conf.int[,"upper .95"],
    p = s$coef[,"Pr(>|z|)"]
  )
}) %>% bind_rows()
# Sort by statistical significance
univ_OS <- univ_OS %>%
  arrange(p)

# ==========================================================
# 8. Multivariate Cox analysis for OS
# ==========================================================
sig_vars <- univ_OS %>%
  filter(p < 0.05) %>%
  pull(Variable)
multi_formula <- as.formula(
  paste("Surv(OS_Time_Month, OS_status) ~", 
        paste(sig_vars, collapse = " + "))
)
multi_cox <- coxph(multi_formula, data = Train.cox)
multi_sum <- summary(multi_cox)
multi_OS <- data.frame(
  Variable = rownames(multi_sum$coef),
  HR       = multi_sum$coef[,"exp(coef)"],
  lower    = multi_sum$conf.int[,"lower .95"],
  upper    = multi_sum$conf.int[,"upper .95"],
  p        = multi_sum$coef[,"Pr(>|z|)"]
) %>%
  arrange(p)

############################################################
## External validation cohorts (YA / CY / HS-2)
############################################################
## Note:The same preprocessing, variable dichotomization,
## univariate Cox screening, and multivariate Cox modeling
## strategies were applied to all external validation
## cohorts (YA, CY, and HS-2).
##
## For each cohort, identical variable definitions,
## cutoff values, and model construction procedures
## were used to ensure consistency and comparability
## of hazard ratio estimates across centers.
############################################################



############################################################
## Bar plot for clinical associations with G-TRS subtype
##
## This analysis evaluates the association between the
## G-TRS risk stratification (high-risk vs low-risk) and
## a panel of clinical and pathological variables.
##
## For each clinical variable, the relative proportions
## of G-TRS high- and low-risk cases are calculated within
## each variable level, and visualized using a stacked
## bar plot with paired (+ / −) categories.
############################################################

#### Load required packages
library(dplyr)
library(tidyr)
library(ggplot2)
library(purrr)
library(scales)

# ==========================================================
# Data source and variable definitions
# ==========================================================
# 'df' contains preprocessed clinical data with all
# covariates already dichotomized into binary form (0 / 1)
df <- Train.cox
# Binary risk stratification variable:
# 1 = high-risk G-TRS subtype, 0 = low-risk subtype
risk_var <- "G.TRS_subtype"

# Clinical variables included in the association analysis.
# All variables are binary and represent predefined
# clinically meaningful thresholds.
clin_vars <- c(
  "RFS_status", "OS_status", "Milan", "UCSF", "AJCC", "PVTT",
  "AFP_status", "MVI_status", "Tumor_Number", "MTD_status", 
  "Age_status", "Gender", "PT_status", "TBil_status",
  "ALB_status", "PLT_status"
)
# Retain only variables available in the current dataset
clin_vars <- intersect(clin_vars, names(df))
############################################################
## Function: compute proportions of High-/Low-risk groups
## within each level of a clinical variable, together with
## Fisher's exact test p value.
############################################################
compute_clin_assoc <- function(data, var, risk_var){
  # Exclude samples with missing values
  dat <- data %>%
    filter(
      !is.na(.data[[var]]),
      !is.na(.data[[risk_var]])
    )
  # Skip variables without sufficient variability
  if(length(unique(dat[[var]])) < 2) return(NULL)
  # Fisher's exact test for association
  tab <- table(dat[[risk_var]], dat[[var]])
  p_val <- suppressWarnings(fisher.test(tab)$p.value)
  # Calculate within-level proportions of G-TRS subtypes
  dat %>%
    count(!!sym(var), !!sym(risk_var)) %>%
    group_by(!!sym(var)) %>%
    mutate(
      prop = n / sum(n),
      feature = var,
      level = !!sym(var),          # 0 / 1
      sign  = ifelse(level == 1, "+", "−"),
      level_label = paste0(var, sign, " (n=", sum(n), ")"),
      p_value = p_val
    ) %>%
    ungroup()
}
############################################################
## Apply association analysis across all clinical variables
##
## The results from individual variables are combined
## into a single long-format data frame suitable for
## visualization.
############################################################
plot_df <- map_df(
  clin_vars,
  ~ compute_clin_assoc(df, .x, risk_var)
)
# Assign human-readable labels for G-TRS subtypes
plot_df <- plot_df %>%
  mutate(
    risk_label = ifelse(.data[[risk_var]] == 1,
                        "High-risk", "Low-risk")
  )
############################################################
## Construct x-axis with visual gaps between variables
##
## To improve readability, paired (+ / −) levels of each
## clinical variable are displayed together, with a small
## visual gap separating different variables along the
## x-axis.
############################################################
build_gap_axis <- function(plot_df, clin_vars){
  label_order <- c()
  for(v in clin_vars){
    lv <- unique(plot_df$level_label[plot_df$feature == v])
    lv <- sort(lv)  # ensure − then +
    label_order <- c(label_order, lv, paste0("gap_", v))
  }
  # Main data
  df_main <- plot_df %>%
    mutate(
      level_label = factor(level_label, levels = label_order),
      x_id = as.numeric(level_label)
    )
  # Gap placeholders
  gap_df <- data.frame(
    feature = clin_vars,
    level_label = factor(paste0("gap_", clin_vars),
                         levels = label_order),
    risk_label = NA,
    prop = 0,
    x_id = as.numeric(factor(paste0("gap_", clin_vars),
                             levels = label_order))
  )
  
  bind_rows(df_main, gap_df) %>%
    arrange(x_id)
}
plot_df2 <- build_gap_axis(plot_df, clin_vars)
############################################################
## Visualization: stacked bar plot
##
## The stacked bars represent the relative proportions
## of G-TRS high- and low-risk subtypes within each
## clinical variable level.
############################################################
ggplot(plot_df2, aes(x = x_id, y = prop, fill = risk_label)) +
  geom_bar(stat="identity", width = 0.9, color = NA) +
  geom_text(
    aes(label = ifelse(!is.na(risk_label),
                       percent(prop, accuracy = 1), "")),
    position = position_stack(vjust = 0.5),
    angle = 90,
    color = "white",
    size = 3.6,
    fontface = "bold"
  ) +
  scale_fill_manual(values = c("Low-risk"="#4C72B0",
                               "High-risk"="#D1495B"),
                    na.value = NA) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1.05)) +
  scale_x_continuous(
    breaks = plot_df2$x_id,
    labels = plot_df2$level_label
  ) +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 11),
    axis.title = element_text(size = 12, face ="bold"),
    legend.title = element_blank(),
    panel.grid = element_blank()
  ) +
  labs(
    y = "Percentage of G-TRS subtypes (high- and low-risk)",
    x = ""
  )

############################################################
## Application to external validation cohorts (CY / HS-2 / YA)
############################################################
## The same clinical association analysis pipeline was
## applied to all external validation cohorts, including
## CY, HS-2, and YA.
##
## For analyses stratified by post-transplant recurrence
## status, the stratification variable was set to
## RFS_status instead of G-TRS subtype:
##
##   risk_var <- "RFS_status"
##
## Accordingly, G-TRS subtype was treated as a clinical
## feature and included among the tested covariates:
##
##   clin_vars <- c(
##     "G.TRS_subtype", "OS_status", "Milan", "UCSF", "AJCC",
##     "PVTT", "AFP_status", "MVI_status", "Tumor_Number",
##     "MTD_status", "Age_status", "Gender", "PT_status",
##     "TBil_status", "ALB_status", "PLT_status"
##   )
##
## Apart from this role exchange between the stratification
## variable and clinical features, all preprocessing steps,
## proportion calculations, Fisher's exact tests, and
## visualization procedures were identical to those used
## in the discovery cohort.
##
## This unified strategy ensures consistency and
## comparability of clinical association patterns across
## cohorts and across different stratification schemes.
############################################################