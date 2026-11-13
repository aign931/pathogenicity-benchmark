# Manuscript release checklist

The corresponding author will create the approved tag, for example v1.0.0-manuscript. Repository preparation does not create a tag or mint a DOI.

1. Review the corrected main branch, CITATION.cff author list/order, scientific audit and validation report.
2. Run python verify_package.py from a clean checkout. Review the three separate environment snapshots and the limits stated in REPRODUCIBILITY.md.
3. Assemble the large reproducibility archive outside Git. Start from the historical inventory, review redistribution rights, and record every included and omitted category. Exclude external dbNSFP/VEP resources. Generate a fresh manifest and verify the final archive.
4. Connect the repository to Zenodo before creating the approved GitHub release. Confirm the code metadata and release version; use CITATION.cff. No placeholder DOI should be committed as a real identifier.
5. Publish the approved GitHub release. Inspect the resulting Zenodo record and its actual file list. Do not assume separately attached GitHub release assets have been preserved merely because the source archive has a DOI.
6. If the model/evidence archive is absent, deposit the cleared archive explicitly in an appropriate Zenodo record and link it to the code record with related identifiers. A code DOI and a data/archive DOI may be separate.
7. Use the VERSION-SPECIFIC DOI for the exact manuscript version. Add actual issued identifiers and released version/date to citation metadata in the next appropriate repository update.
8. Check that Data Availability accurately lists what the DOI record(s) contain. If processed feature tables are omitted, state their external acquisition terms and reconstruction route.
9. Obtain all-author approval and use the verified DOI(s) in the Overleaf manuscript.

Suggested wording once both records exist:

The analysis code is available from the Pathogenicity Benchmark repository (https://github.com/aign931/pathogenicity-benchmark), with the manuscript version archived in Zenodo at [CODE VERSION DOI]. The supporting reproducibility archive is available at [ARCHIVE VERSION DOI] and contains [actual deposited categories]. External dbNSFP and VEP resources are not redistributed; their versions and provenance are documented. [Describe any additional omitted feature tables and their access conditions.]

If one verified record includes both code and supporting materials, combine the two DOI references. Replace every bracketed item before submission.

References:
- https://help.zenodo.org/docs/github/archive-software/github-upload/
- https://help.zenodo.org/docs/github/describe-software/citation-file/
- https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
