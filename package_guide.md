# Supporting materials

The Git repository contains analysis code, feature definitions, fold assignments, environment versions and documentation.

The supporting results are held separately: fitted models, preprocessing objects, held-out predictions, SHAP matrices, statistical tables and figures. Large generated files are not part of ordinary Git history.

For analysis, arrange a local copy as follows:

    data/
      processed/
        canonical519.csv
        pooled526.csv
        expert_panel.csv
        canonical519_aligned.csv
        vus.csv
    results/
      full519/
      reduced519/
      palb2_526/
      expert_panel/
      leave_one_gene_out/
      statistical_comparison/
      vus/

Each labelled scenario directory contains models/, preprocess/, predictions/, shap/ and tables/. VUS descriptive outputs belong directly in results/vus/.

The filenames used by the model scripts are unchanged within each scenario. The processed-input names and directory structure above are the public interface; local project-management directories are not required.

Follow REPRODUCIBILITY.md for the commands and DATA_SOURCES.md for source terms. A public archive of supporting results should be deposited separately after its contents and redistribution terms are confirmed. The manuscript's Data Availability statement should describe the files actually present in the cited archive.
