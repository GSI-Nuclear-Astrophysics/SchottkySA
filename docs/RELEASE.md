# Publishing the reviewed software version

Build the source and wheel, check metadata, inspect every archive and verify
installation outside the checkout:

```sh
python -m build
twine check dist/*
python scripts/check_release.py dist/*
```

Publishing runs manually through `release.yml`, with an explicit confirmation
and a selected package index. Creating a tag or GitHub release does not invoke
the PyPI workflow. TestPyPI is an external upload too, not a local dry run.
Configure the appropriate Trusted Publisher and protected GitHub environment
before running the workflow.

The existing Zenodo concept DOI is `10.5281/zenodo.21906687`. The first public
version has DOI `10.5281/zenodo.21906688`; it does not identify the new 1.1 source.
Archive the reviewed GitHub release as a new version in the existing software
record. Add the actual new version DOI and date after publication. Keep the
version, authors, license and repository URL consistent in `CITATION.cff`,
`.zenodo.json`, package metadata and the SoftwareX article.

When both citation files are present, Zenodo's GitHub integration uses
`.zenodo.json`. The concept DOI in `CITATION.cff` identifies the software family.

References: [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
and [Zenodo software metadata](https://help.zenodo.org/docs/github/describe-software/).
