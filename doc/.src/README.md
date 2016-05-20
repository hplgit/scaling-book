## Directory structure

The setup of this directory follows the best practice of the `doc/src/book`
directory in the [setup4book-doconce](https://hplgit.github.io/setup4book-doconce/doc/web/index.html) resource repository.

 * `book.do.txt`: main document to compile, basically includes
   mako code (`mako_code.txt`) and the book manuscript `scaling.do.txt`.
 * `scaling.do.txt`: all the source for the book.
 * `fig-scaling`: all figures for the book.
 * `src-scaling`: all program files for the book.
 * `mov-scaling`: all movies for the book.
 * `clean.sh`: clean up all redundant files.
 * `make.sh`: create LaTeX PDF books.
 * `make_html.sh`: create HTML books.
 * `papers.pub`: database for references in the book (used to generate
    BibTeX for instance).
 * `venues.list`: companion file for `papers.pub`.
 * `pack_Springer.sh`: pack all files in a subdirectory `langtangen`
   that can be transferred as a tarball to Springer.
 * `README_Springer_dir.txt`: help file for Springer.
 * `password.html`: help file for creating password protected HTML file
   (book with solutions for instance).


