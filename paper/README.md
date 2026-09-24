# Paper

This directory contains the author-identified public preprint source and a
locally compiled comparison PDF.

## Build

A standard LaTeX installation can compile the source:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error routide-arxiv.tex
pdflatex -interaction=nonstopmode -halt-on-error routide-arxiv.tex
```

The two figure PDFs are committed next to the TeX source.

The public repository URL is intended for the arXiv/preprint version. The
double-anonymous ICPE review manuscript is deliberately **not** stored here or
linked from the anonymous review PDF.
