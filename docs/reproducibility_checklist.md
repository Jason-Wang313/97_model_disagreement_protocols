# Reproducibility Checklist

- [x] `python -m py_compile src/run_experiment.py` passes.
- [x] `python src/run_experiment.py` regenerates the v5 evidence package.
- [x] `python scripts/generate_manuscript.py` regenerates `paper/main.tex` and `paper/references.bib`.
- [x] `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` builds `paper/main.pdf`.
- [x] Final PDF copied only to `C:/Users/wangz/Downloads/97.pdf`.
- [x] `python scripts/validate_submission_artifacts.py` validates row counts, page count, citation-box settings, SHA256, and Desktop absence.
- [x] Visual PDF QA rendered representative pages and found no blocking layout defects.

Validated PDF SHA256: `CC500D7CCC351CBAB82FC07B729A98CD13A7201487281C316711072E71320B39`.
