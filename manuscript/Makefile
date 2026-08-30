# Render the manuscript. Needs pandoc (+ a LaTeX engine for PDF).
#   choco install pandoc miktex     # Windows
#   apt-get install pandoc texlive  # Debian/Ubuntu

manuscript.pdf: manuscript.md references.bib
	pandoc manuscript.md \
	  --citeproc --bibliography=references.bib \
	  --metadata link-citations=true \
	  -o manuscript.pdf

manuscript.docx: manuscript.md references.bib
	pandoc manuscript.md --citeproc --bibliography=references.bib -o manuscript.docx

.PHONY: clean
clean:
	rm -f manuscript.pdf manuscript.docx
