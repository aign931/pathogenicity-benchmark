suppressPackageStartupMessages(library(pROC))

args <- commandArgs(trailingOnly = TRUE)
base_dir <- if (length(args) >= 1) args[[1]] else file.path(Sys.getenv("PB_WORKSPACE", unset = "."), "revision/04_analysis/loop05_corrected_common5fold")
boot_n <- if (length(args) >= 2) as.integer(args[[2]]) else 5000L
output_dir <- file.path(base_dir, "statistical_comparison")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

scenario_order <- c("full519", "reduced519", "palb2_526", "expert_panel", "leave_one_gene_out")
model_order <- c("VARITY", "MutScore", "MVP")
pair_order <- list(c("VARITY", "MutScore"), c("VARITY", "MVP"), c("MutScore", "MVP"))

read_aligned_predictions <- function(scenario) {
  path <- file.path(base_dir, scenario, "tables", "oof_predictions.csv")
  if (!file.exists(path)) return(NULL)
  pred <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("row_id", "fold", "label", "model", "prob_pathogenic")
  missing <- setdiff(required, names(pred))
  if (length(missing) > 0) stop(sprintf("%s missing columns: %s", path, paste(missing, collapse = ", ")))
  model_frames <- lapply(model_order, function(model_name) {
    part <- pred[pred$model == model_name, required, drop = FALSE]
    part <- part[order(part$row_id), , drop = FALSE]
    if (anyDuplicated(part$row_id)) stop(sprintf("%s has duplicate row_id for %s", scenario, model_name))
    part
  })
  names(model_frames) <- model_order
  reference <- model_frames[[1]]
  for (model_name in model_order[-1]) {
    current <- model_frames[[model_name]]
    if (!identical(reference$row_id, current$row_id) ||
        !identical(reference$fold, current$fold) ||
        !identical(reference$label, current$label)) {
      stop(sprintf("%s predictions are not paired across models", scenario))
    }
  }
  scores <- do.call(cbind, lapply(model_frames, function(part) part$prob_pathogenic))
  colnames(scores) <- model_order
  list(path = path, row_id = reference$row_id, fold = reference$fold,
       label = reference$label, scores = scores)
}

all_auc_rows <- list()
all_pair_rows <- list()
report <- c(
  "# Loop05 paired OOF AUC comparison",
  "",
  sprintf("Generated with R %s and pROC %s.", getRversion(), as.character(packageVersion("pROC"))),
  sprintf("Bootstrap replicates per model/scenario: %d (stratified).", boot_n),
  "",
  "These are paired comparisons of pooled out-of-fold predictions from fixed post-development recipes. They are not de-novo unbiased model-development estimates for VARITY because its hyperparameters were selected in the original benchmark workflow.",
  ""
)

for (scenario_index in seq_along(scenario_order)) {
  scenario <- scenario_order[[scenario_index]]
  aligned <- read_aligned_predictions(scenario)
  if (is.null(aligned)) next
  labels <- as.integer(aligned$label)
  if (!identical(sort(unique(labels)), c(0L, 1L))) stop(sprintf("%s labels are not binary 0/1", scenario))

  roc_objects <- list()
  scenario_auc_rows <- list()
  for (model_index in seq_along(model_order)) {
    model_name <- model_order[[model_index]]
    scores <- as.numeric(aligned$scores[, model_name])
    roc_object <- roc(response = labels, predictor = scores, levels = c(0, 1),
                      direction = "<", quiet = TRUE)
    set.seed(2025L + scenario_index * 100L + model_index)
    ci <- ci.auc(roc_object, conf.level = 0.95, method = "bootstrap",
                 boot.n = boot_n, boot.stratified = TRUE, progress = "none")
    roc_objects[[model_name]] <- roc_object
    scenario_auc_rows[[model_name]] <- data.frame(
      scenario = scenario, model = model_name, n = length(labels),
      n_negative = sum(labels == 0), n_positive = sum(labels == 1),
      auc = as.numeric(auc(roc_object)), ci_low = as.numeric(ci[[1]]),
      ci_high = as.numeric(ci[[3]]), bootstrap_n = boot_n,
      stringsAsFactors = FALSE
    )
  }
  scenario_auc <- do.call(rbind, scenario_auc_rows)
  rownames(scenario_auc) <- NULL

  scenario_pair_rows <- list()
  for (pair_index in seq_along(pair_order)) {
    pair <- pair_order[[pair_index]]
    first <- pair[[1]]
    second <- pair[[2]]
    test <- roc.test(roc_objects[[first]], roc_objects[[second]], paired = TRUE,
                     method = "delong", alternative = "two.sided")
    scenario_pair_rows[[pair_index]] <- data.frame(
      scenario = scenario, model_a = first, model_b = second,
      auc_a = as.numeric(auc(roc_objects[[first]])),
      auc_b = as.numeric(auc(roc_objects[[second]])),
      auc_difference_a_minus_b = as.numeric(auc(roc_objects[[first]]) - auc(roc_objects[[second]])),
      z = unname(as.numeric(test$statistic)), p_raw = as.numeric(test$p.value),
      method = as.character(test$method), stringsAsFactors = FALSE
    )
  }
  scenario_pairs <- do.call(rbind, scenario_pair_rows)
  scenario_pairs$p_holm <- p.adjust(scenario_pairs$p_raw, method = "holm")
  scenario_pairs$significant_holm_0_05 <- scenario_pairs$p_holm < 0.05

  write.csv(scenario_auc, file.path(output_dir, paste0(scenario, "_auc_bootstrap_ci.csv")), row.names = FALSE)
  write.csv(scenario_pairs, file.path(output_dir, paste0(scenario, "_delong_pairwise.csv")), row.names = FALSE)
  all_auc_rows[[scenario]] <- scenario_auc
  all_pair_rows[[scenario]] <- scenario_pairs

  report <- c(
    report, sprintf("## %s", scenario), "",
    "AUC and stratified bootstrap 95% CI:", "",
    paste(capture.output(print(scenario_auc[, c("model", "n", "n_negative", "n_positive", "auc", "ci_low", "ci_high")], row.names = FALSE)), collapse = "\n"),
    "", "Paired DeLong tests with within-scenario Holm correction:", "",
    paste(capture.output(print(scenario_pairs[, c("model_a", "model_b", "auc_difference_a_minus_b", "z", "p_raw", "p_holm", "significant_holm_0_05")], row.names = FALSE)), collapse = "\n"),
    ""
  )
}

combined_auc <- do.call(rbind, all_auc_rows)
combined_pairs <- do.call(rbind, all_pair_rows)
rownames(combined_auc) <- NULL
rownames(combined_pairs) <- NULL
write.csv(combined_auc, file.path(output_dir, "all_scenarios_auc_bootstrap_ci.csv"), row.names = FALSE)
write.csv(combined_pairs, file.path(output_dir, "all_scenarios_delong_pairwise.csv"), row.names = FALSE)

report <- c(
  report, "## Method audit", "",
  "The earlier notebook D:/thesis/scripts/bootstrap_CI_delong_test.ipynb was not used for final p-values. Its custom covariance calculation does not implement the standard cross-class structural components used by DeLong's correlated-ROC estimator. Final results therefore use the independently maintained pROC implementation. The notebook's stratified-bootstrap idea was retained, with a fixed seed and 5,000 replicates by default.",
  "", "## Interpretation boundary", "",
  "- DeLong tests compare paired OOF score vectors for each completed fixed-recipe sensitivity scenario.",
  "- Holm correction is applied to the three pairwise tests within each scenario.",
  "- The five outer folds are reused to construct one OOF vector; the tests do not turn a post-development design into de-novo external validation.",
  "- For leave-one-gene-out, pooled values combine heterogeneous held-out genes; gene-specific metrics and sample counts must be shown alongside them.",
  ""
)
writeLines(report, file.path(output_dir, "AUC_DELONG_REPORT.md"))
writeLines(capture.output(sessionInfo()), file.path(output_dir, "R_sessionInfo.txt"))
cat(sprintf("Wrote %d scenario comparisons to %s\n", length(all_auc_rows), output_dir))

