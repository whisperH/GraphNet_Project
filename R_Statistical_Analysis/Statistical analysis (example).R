
############################################################
## Statistical tests for Table 1 and Figure 10I
##
## This script performs statistical comparisons of
## categorical and continuous variables across cohorts
## and between G-TRS subtypes.
############################################################


############################################################
## Part I. Chi-squared tests for Table 1
##
## Purpose:
## To compare the distribution of a categorical variable
## between the training cohort and each external validation
## cohort, as well as across all cohorts simultaneously.
############################################################

## Training vs HS-2 for gender
tbl1 <- matrix(c(221, 23,
                 206, 16),
               nrow = 2, byrow = TRUE)
# chi-squared test
chi_res <- chisq.test(tbl1)
chi_res$p.value
## Training vs YA for gender
tbl2 <- matrix(c(221, 23,
                 171, 18),
               nrow = 2, byrow = TRUE)
chi_res <- chisq.test(tbl2)
chi_res$p.value
## Training vs CY for gender
tbl3 <- matrix(c(221, 23,
                 90, 12),
               nrow = 2, byrow = TRUE)
chi_res <- chisq.test(tbl3)
chi_res$p.value

#### Overall test across all cohorts
## (training, HS-2, YA, CY)
tbl4 <- matrix(c(221, 23,
                 206, 16,
                 171, 18,
                 90, 12),
               nrow = 4, byrow = TRUE)
chi_res <- chisq.test(tbl4)
chi_res$p.value


############################################################
## Part II. Fisher's exact tests for Figure 10I
##
## Purpose:
## To compare categorical variables between groups when
## expected cell counts are small, where Fisher’s exact
## test is more appropriate than chi-squared test.
############################################################

tbl <- matrix(
  c(24, 47,
    41, 30),
  nrow = 2,
  byrow = TRUE
)

fisher_res <- fisher.test(tbl)
fisher_res$p.value

tbl <- matrix(
  c(43, 70,
    22, 7),
  nrow = 2,
  byrow = TRUE
)

fisher_res <- fisher.test(tbl)
fisher_res$p.value


############################################################
## Part III. Statistical analysis of multiplex
## immunofluorescence (mIF) markers
##
## Purpose:
## To compare quantitative mIF marker levels between
## G-TRS high-risk and low-risk subtypes.
############################################################
library(dplyr)
library(purrr)
library(broom)
mIF.data=read.table(text = read_clip(), 
                  header = TRUE, sep = "\t", stringsAsFactors = FALSE)
test_vars <- colnames(mIF.data)
test_vars =test_vars [-c(1:4)]

ttest_results <- map_dfr(test_vars, function(var){
  df_sub <-mIF.data %>%
    select(G.TRS.subtype, value = all_of(var)) %>%
    mutate(value = as.numeric(value)) %>%
    filter(!is.na(value),
          G.TRS.subtype %in% c("high-risk", "low-risk"))
  
  # 如果某一组样本数太少，直接跳过
  if(length(unique(df_sub$G.TRS.subtype)) < 2){
    return(NULL)
  }
  t_res <- t.test(value ~ G.TRS.subtype, data = df_sub)
  tidy(t_res) %>%
    mutate(
      variable = var,
      mean_high = mean(df_sub$value[df_sub$G.TRS.subtype == "high-risk"], na.rm = TRUE),
      mean_low  = mean(df_sub$value[df_sub$G.TRS.subtype == "low-risk"],  na.rm = TRUE),
      sd_high   = sd(df_sub$value[df_sub$G.TRS.subtype == "high-risk"],   na.rm = TRUE),
      sd_low    = sd(df_sub$value[df_sub$G.TRS.subtype == "low-risk"],    na.rm = TRUE)
    )
})


############################################################
## Discrimination analysis:
## Comparison of G-TRS with Milan and UCSF criteria
##
## Metrics:
##   - AUC for binary recurrence outcome
##   - Harrell's C-index for recurrence-free survival
##
## Cohorts:
##   - Training cohort
##   - CY validation cohort
##   - HS-2 validation cohort
##   - YA validation cohort
############################################################

############################################################
## Discrimination performance evaluation
## AUC (binary outcome) and C-index (survival outcome)
############################################################

## ---------------------------------------------------------
## 1. Load data
## ---------------------------------------------------------
Train.data <- read.table(
  text = read_clip(),
  header = TRUE,
  sep = "\t",
  stringsAsFactors = FALSE
)

library(pROC)
library(survival)

## ---------------------------------------------------------
## 2. Standardize outcome and risk-score encoding
##
## Principle:
##   Higher values always indicate higher recurrence risk
## ---------------------------------------------------------

# Binary recurrence outcome
Train.data$RFS_status <- ifelse(
  Train.data$RFS_status == "yes", 1, 0
)

# G-TRS risk score (model-derived)
Train.data$GTRS_score <- ifelse(
  Train.data$G.TRS_subtype == "high-risk", 1, 0
)

# Clinical criteria (protective -> reversed to risk score)
Train.data$Milan_risk <- ifelse(
  Train.data$Milan == "yes", 0, 1
)

Train.data$UCSF_risk <- ifelse(
  Train.data$UCSF == "yes", 0, 1
)

## ---------------------------------------------------------
## 3. AUC calculation for binary recurrence
## ---------------------------------------------------------
## AUC quantifies discrimination for the binary outcome
## (recurrence vs non-recurrence)

auc_results <- data.frame(
  Model = c("G-TRS", "Milan", "UCSF"),
  AUC = c(
    auc(roc(Train.data$RFS_status, Train.data$GTRS_score, quiet = TRUE)),
    auc(roc(Train.data$RFS_status, Train.data$Milan_risk, quiet = TRUE)),
    auc(roc(Train.data$RFS_status, Train.data$UCSF_risk,  quiet = TRUE))
  )
)

auc_results
# Expected:
# G-TRS  ~ 0.8460082
# Milan  ~ 0.6786657
# UCSF   ~0.6825147

## ---------------------------------------------------------
## 4. C-index calculation for recurrence-free survival
## ---------------------------------------------------------
## C-index evaluates time-to-event discrimination while
## accounting for censoring.

cindex_results <- data.frame(
  Model = c("G-TRS", "Milan", "UCSF"),
  C_index = c(
    concordance(
      coxph(Surv(RFS_Time_Month, RFS_status) ~ GTRS_score,
            data = Train.data)
    )$concordance,
    
    concordance(
      coxph(Surv(RFS_Time_Month, RFS_status) ~ Milan_risk,
            data = Train.data)
    )$concordance,
    
    concordance(
      coxph(Surv(RFS_Time_Month, RFS_status) ~ UCSF_risk,
            data = Train.data)
    )$concordance
  )
)

cindex_results
# Expected:
# G-TRS  ~ 0.7571772
# Milan  ~ 0.6457098
# UCSF   ~ 0.6437227

####
CY.data <- read.table(
  text = read_clip(),
  header = TRUE,
  sep = "\t",
  stringsAsFactors = FALSE
)
## ---------------------------------------------------------
## 2. Standardize outcome and risk-score encoding
##
## Principle:
##   Higher values always indicate higher recurrence risk
## ---------------------------------------------------------

# Binary recurrence outcome
CY.data$RFS_status <- ifelse(
  CY.data$RFS_status == "yes", 1, 0
)

# G-TRS risk score (model-derived)
CY.data$GTRS_score <- ifelse(
  CY.data$G.TRS_subtype == "high-risk", 1, 0
)

# Clinical criteria (protective -> reversed to risk score)
CY.data$Milan_risk <- ifelse(
  CY.data$Milan == "yes", 0, 1
)

CY.data$UCSF_risk <- ifelse(
  CY.data$UCSF == "yes", 0, 1
)

## ---------------------------------------------------------
## 3. AUC calculation for binary recurrence
## ---------------------------------------------------------
## AUC quantifies discrimination for the binary outcome
## (recurrence vs non-recurrence)

auc_results <- data.frame(
  Model = c("G-TRS", "Milan", "UCSF"),
  AUC = c(
    auc(roc(CY.data$RFS_status, CY.data$GTRS_score, quiet = TRUE)),
    auc(roc(CY.data$RFS_status, CY.data$Milan_risk, quiet = TRUE)),
    auc(roc(CY.data$RFS_status, CY.data$UCSF_risk,  quiet = TRUE))
  )
)

auc_results
# Expected:
# G-TRS  ~ 0.7737740
# Milan  ~ 0.6377399
# UCSF   ~0.6179104

## ---------------------------------------------------------
## 4. C-index calculation for recurrence-free survival
## ---------------------------------------------------------
## C-index evaluates time-to-event discrimination while
## accounting for censoring.

cindex_results <- data.frame(
  Model = c("G-TRS", "Milan", "UCSF"),
  C_index = c(
    concordance(
      coxph(Surv(RFS_Time_Month, RFS_status) ~ GTRS_score,
            data = CY.data)
    )$concordance,
    
    concordance(
      coxph(Surv(RFS_Time_Month, RFS_status) ~ Milan_risk,
            data = CY.data)
    )$concordance,
    
    concordance(
      coxph(Surv(RFS_Time_Month, RFS_status) ~ UCSF_risk,
            data = CY.data)
    )$concordance
  )
)

cindex_results
# Expected:
# G-TRS  ~ 0.7151746
# Milan  ~ 0.6314733
# UCSF   ~ 0.6083902

###################
HS2.data <- read.table(
  text = read_clip(),
  header = TRUE,
  sep = "\t",
  stringsAsFactors = FALSE
)
## ---------------------------------------------------------
## 2. Standardize outcome and risk-score encoding
##
## Principle:
##   Higher values always indicate higher recurrence risk
## ---------------------------------------------------------

# Binary recurrence outcome
HS2.data$RFS_status <- ifelse(
  HS2.data$RFS_status == "yes", 1, 0
)

# G-TRS risk score (model-derived)
HS2.data$GTRS_score <- ifelse(
  HS2.data$G.TRS_subtype == "high-risk", 1, 0
)

# Clinical criteria (protective -> reversed to risk score)
HS2.data$Milan_risk <- ifelse(
  HS2.data$Milan == "yes", 0, 1
)

HS2.data$UCSF_risk <- ifelse(
  HS2.data$UCSF == "yes", 0, 1
)

## ---------------------------------------------------------
## 3. AUC calculation for binary recurrence
## ---------------------------------------------------------
## AUC quantifies discrimination for the binary outcome
## (recurrence vs non-recurrence)

auc_results <- data.frame(
  Model = c("G-TRS", "Milan", "UCSF"),
  AUC = c(
    auc(roc(HS2.data$RFS_status, HS2.data$GTRS_score, quiet = TRUE)),
    auc(roc(HS2.data$RFS_status, HS2.data$Milan_risk, quiet = TRUE)),
    auc(roc(HS2.data$RFS_status, HS2.data$UCSF_risk,  quiet = TRUE))
  )
)

auc_results
# Expected:
# G-TRS  ~ 0.7859848
# Milan  ~0.6717172
# UCSF   ~0.6553030

## ---------------------------------------------------------
## 4. C-index calculation for recurrence-free survival
## ---------------------------------------------------------
## C-index evaluates time-to-event discrimination while
## accounting for censoring.

cindex_results <- data.frame(
  Model = c("G-TRS", "Milan", "UCSF"),
  C_index = c(
    concordance(
      coxph(Surv(RFS_Time_Month, RFS_status) ~ GTRS_score,
            data = HS2.data)
    )$concordance,
    
    concordance(
      coxph(Surv(RFS_Time_Month, RFS_status) ~ Milan_risk,
            data = HS2.data)
    )$concordance,
    
    concordance(
      coxph(Surv(RFS_Time_Month, RFS_status) ~ UCSF_risk,
            data = HS2.data)
    )$concordance
  )
)

cindex_results
# Expected:
# G-TRS  ~0.7848633
# Milan  ~ 0.6663163
# UCSF   ~ 0.6451764

####
YA.data <- read.table(
  text = read_clip(),
  header = TRUE,
  sep = "\t",
  stringsAsFactors = FALSE
)
## ---------------------------------------------------------
## 2. Standardize outcome and risk-score encoding
##
## Principle:
##   Higher values always indicate higher recurrence risk
## ---------------------------------------------------------

# Binary recurrence outcome
YA.data$RFS_status <- ifelse(
  YA.data$RFS_status == "yes", 1, 0
)

# G-TRS risk score (model-derived)
YA.data$GTRS_score <- ifelse(
  YA.data$G.TRS_subtype == "high-risk", 1, 0
)

# Clinical criteria (protective -> reversed to risk score)
YA.data$Milan_risk <- ifelse(
  YA.data$Milan == "yes", 0, 1
)

YA.data$UCSF_risk <- ifelse(
  YA.data$UCSF == "yes", 0, 1
)

## ---------------------------------------------------------
## 3. AUC calculation for binary recurrence
## ---------------------------------------------------------
## AUC quantifies discrimination for the binary outcome
## (recurrence vs non-recurrence)

auc_results <- data.frame(
  Model = c("G-TRS", "Milan", "UCSF"),
  AUC = c(
    auc(roc(YA.data$RFS_status, YA.data$GTRS_score, quiet = TRUE)),
    auc(roc(YA.data$RFS_status, YA.data$Milan_risk, quiet = TRUE)),
    auc(roc(YA.data$RFS_status, YA.data$UCSF_risk,  quiet = TRUE))
  )
)

auc_results
# Expected:
# G-TRS  ~ 0.7328431
# Milan  ~0.5281879
# UCSF   ~0.5052381

## ---------------------------------------------------------
## 4. C-index calculation for recurrence-free survival
## ---------------------------------------------------------
## C-index evaluates time-to-event discrimination while
## accounting for censoring.

cindex_results <- data.frame(
  Model = c("G-TRS", "Milan", "UCSF"),
  C_index = c(
    concordance(
      coxph(Surv(RFS_Time_Month, RFS_status) ~ GTRS_score,
            data = YA.data)
    )$concordance,
    
    concordance(
      coxph(Surv(RFS_Time_Month, RFS_status) ~ Milan_risk,
            data = YA.data)
    )$concordance,
    
    concordance(
      coxph(Surv(RFS_Time_Month, RFS_status) ~ UCSF_risk,
            data = YA.data)
    )$concordance
  )
)

cindex_results
# Expected:
# G-TRS  ~0.6770675
# Milan  ~ 0.5213624
# UCSF   ~ 0.5051061






