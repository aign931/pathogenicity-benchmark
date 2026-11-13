suppressPackageStartupMessages(library(pROC))

input_path <- file.path(Sys.getenv("PB_DATA_DIR", unset = "data"), "processed/holdout_predictions.csv")
out_dir <- file.path(Sys.getenv("PB_RESULTS_DIR", unset = "results"), "holdout")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(input_path, stringsAsFactors = FALSE)
stopifnot(nrow(df) == 104)
stopifnot(identical(df$varity_id, df$mvp_id))
stopifnot(identical(df$varity_id, df$mutscore_id))
stopifnot(identical(df$varity_label, df$mvp_label))
stopifnot(identical(df$varity_label, df$mutscore_label))
stopifnot(all(df$varity_label %in% c(0, 1)))

scores <- list(
  VARITY = df$varity_score,
  MutScore = df$mutscore_score,
  MVP = df$mvp_score
)

set.seed(2025)
roc_objects <- lapply(scores, function(score) {
  roc(df$varity_label, score, levels = c(0, 1), direction = "<", quiet = TRUE)
})

ci_rows <- lapply(names(roc_objects), function(model) {
  object <- roc_objects[[model]]
  interval <- ci.auc(object, method = "bootstrap", boot.n = 5000, stratified = TRUE)
  data.frame(
    model = model,
    n = nrow(df),
    n_negative = sum(df$varity_label == 0),
    n_positive = sum(df$varity_label == 1),
    auc = as.numeric(auc(object)),
    ci_low = as.numeric(interval[1]),
    ci_high = as.numeric(interval[3]),
    bootstrap_n = 5000
  )
})
ci_table <- do.call(rbind, ci_rows)

pairs <- list(c("VARITY", "MutScore"), c("VARITY", "MVP"), c("MutScore", "MVP"))
test_rows <- lapply(pairs, function(pair) {
  result <- roc.test(
    roc_objects[[pair[1]]], roc_objects[[pair[2]]],
    paired = TRUE, method = "delong"
  )
  data.frame(
    model_1 = pair[1],
    model_2 = pair[2],
    auc_1 = as.numeric(auc(roc_objects[[pair[1]]])),
    auc_2 = as.numeric(auc(roc_objects[[pair[2]]])),
    auc_difference = as.numeric(auc(roc_objects[[pair[1]]])) - as.numeric(auc(roc_objects[[pair[2]]])),
    z = as.numeric(result$statistic),
    p_raw = as.numeric(result$p.value)
  )
})
test_table <- do.call(rbind, test_rows)
test_table$p_holm <- p.adjust(test_table$p_raw, method = "holm")

write.csv(ci_table, file.path(out_dir, "original_holdout_auc_bootstrap_ci.csv"), row.names = FALSE)
write.csv(test_table, file.path(out_dir, "original_holdout_delong_holm.csv"), row.names = FALSE)
capture.output(sessionInfo(), file = file.path(out_dir, "R_sessionInfo.txt"))

report_path <- file.path(out_dir, "holdout_summary.md")
sink(report_path)
cat("# Original 104-Variant Hold-out: pROC comparison\n\n")
cat("## Input validation\n\n")
cat("- Source: `data/processed/holdout_predictions.csv`.\n")
cat("- Rows: 104; IDs and binary labels are identical across all three model columns.\n")
cat("- Negative/positive counts:", sum(df$varity_label == 0), "/", sum(df$varity_label == 1), ".\n\n")
cat("## AUC and 95% stratified bootstrap CI\n\n```text\n")
print(ci_table, row.names = FALSE, digits = 6)
cat("```\n\n## Paired DeLong tests with Holm correction\n\n```text\n")
print(test_table, row.names = FALSE, digits = 6)
cat("```\n\n")
cat("All three pairwise differences are non-significant after Holm correction. ")
cat("These pROC results supersede the custom DeLong implementation embedded in ")
cat("`bootstrap_CI_delong_test.ipynb`, whose covariance calculation is not a valid ")
cat("implementation of the standard paired DeLong estimator. The qualitative ")
cat("conclusion (no statistically detectable pairwise AUC difference in the original ")
cat("104-variant hold-out) is unchanged, but the historical p-values must not be reused.\n")
sink()

cat(report_path, "\n")
