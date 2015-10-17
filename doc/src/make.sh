#!/bin/bash -x
# Compile the book to LaTeX/PDF.
#
# Usage: make.sh [nospell]
#
# With nospell, spellchecking is skipped.

set -x

name=book
topicname=scaling

encoding="--encoding=utf-8"

CHAPTER=chapter
BOOK=book
APPENDIX=appendix

function system {
  "$@"
  if [ $? -ne 0 ]; then
    echo "make.sh: unsuccessful command $@"
    echo "abort!"
    exit 1
  fi
}

rm tmp_*

if [ $# -ge 1 ]; then
  spellcheck=$1
else
  spellcheck=spell
fi

# No spellchecking of local files here since book.do.txt just includes files.
# Spellcheck all *.do.txt files in each chapter.
if [ "$spellcheck" != 'nospell' ]; then
system doconce spellcheck -d .dict4spell.txt scaling.do.txt preface.do.txt
fi

system preprocess -DFORMAT=pdflatex newcommands_keep.p.tex > newcommands_keep.tex
doconce replace 'newcommand{\E}' 'renewcommand{\E}' newcommands_keep.tex
doconce replace 'newcommand{\I}' 'renewcommand{\I}' newcommands_keep.tex

# TASKS: generate book with solutions, also in the html version
# Make pdfnup with two-pages per sheet

opt1="CHAPTER=$CHAPTER BOOK=$BOOK APPENDIX=$APPENDIX"
opt2="--without_solutions --without_answers"
opt2=
devices="screen paper"

function compile {
    options="$@"
#system doconce format pdflatex $name $opt1 --device=$device --exercise_numbering=chapter   --latex_style=Springer_T4 --latex_title_layout=titlepage --latex_list_of_exercises=loe --latex_admon=mdfbox --latex_admon_color=1,1,1 --latex_table_format=left --latex_admon_title_no_period --latex_no_program_footnotelink "--latex_code_style=default:vrb-blue1@sys:vrb[frame=lines,label=\\fbox{{\tiny Terminal}},framesep=2.5mm,framerule=0.7pt,fontsize=\fontsize{9pt}{9pt}]" $options

system doconce format pdflatex $name $opt1 --device=$device --exercise_numbering=chapter   --latex_style=Springer_sv --latex_title_layout=std --latex_list_of_exercises=none --latex_admon=mdfbox --latex_admon_color=1,1,1 --latex_table_format=left --latex_admon_title_no_period --latex_no_program_footnotelink "--latex_code_style=default:vrb-blue1@sys:vrb[frame=lines,label=\\fbox{{\tiny Terminal}},framesep=2.5mm,framerule=0.7pt,fontsize=\fontsize{9pt}{9pt}]" $options

# Auto edits
system doconce replace 'linecolor=black,' 'linecolor=darkblue,' $name.tex
system doconce subst 'frametitlebackgroundcolor=.*?,' 'frametitlebackgroundcolor=blue!5,' $name.tex

rm -rf $name.aux $name.ind $name.idx $name.bbl $name.toc $name.loe

system pdflatex $name
system bibtex $name
system makeindex $name
system pdflatex $name
system pdflatex $name
system makeindex $name
system pdflatex $name
}

# Important: run with solutions first such that the .aux file
# for the book, referred to by other documents, uses the .aux
# file corresponding to a version without solutions.

# With solutions, password protected
compile --device=screen
newname=${topicname}-book-4screen-sol
password="d!e!cay"
pdftk $name.pdf output $newname.pdf owner_pw foo user_pw $password

compile --device=screen --without_solutions --without_answers
newname=${topicname}-book-4screen
cp $name.pdf $newname.pdf

#--latex_index_in_margin
compile --device=paper --without_solutions --without_answers
newname=${topicname}-book-4print
cp $name.pdf $newname.pdf
pdfnup --frame true --outfile ${newname}-2up.pdf $newname.pdf
cp $name.aux ${newname}.aux-final

# Report typical problems with the book (too long lines,
# undefined labels, etc.). Here we report lines that are more than 10pt
# too long.
doconce latex_problems $name.log 10

# Check grammar in MS Word:
# doconce spellcheck tmp_mako__book.do.txt
# load tmp_stripped_book.do.txt into Word

# Publish
repo=../pub
dest=${repo}/book
if [ ! -d $dest ]; then mkdir $dest; fi
if [ ! -d $dest/pdf ]; then mkdir $dest/pdf; fi
cp ${topicname}-book*.pdf $dest/pdf

src_dest=../../src
if [ ! -d $src_dest ]; then
echo "No directory $src_dest"
exit 1
fi
cd src-$topicname

src_decay=/home/hpl/vc/decay-book/doc/src/chapters/softeng/src-softeng
cp ${src_decay}/decay.py .
src_decay=/home/hpl/vc/decay-book/doc/src/chapters/genz/src-genz
cp ${src_decay}/decay_vc.py .
src_wave=/home/hpl/vc/deqbook/doc/src/chapters/wave/src-wave/wave1D
cp ${src_wave}/wave1D_u0.py .
cp ${src_wave}/wave1D_dn.py .
src_vib=/home/hpl/vc/deqbook/doc/src/chapters/vib/src-vib
cp ${src_vib}/vib.py .

if [ -f clean.sh ]; then
sh clean.sh
fi
cd ..
files=`find src-$topicname \( -name '*.py' -o -name '*.pyx' -o -name '*.f' -o -name '*.c' -o -name '*.cpp' -o -name 'make*.sh' \) -print`
if [ -f src-$topicname/README ]; then
  files="$files src-$topicname/README"
fi
tar cfz tmp.tar.gz $files
mv -f tmp.tar.gz $src_dest
cd $src_dest
rm -rf src-$topicname
tar xfz tmp.tar.gz
rm -f tmp.tar.gz
python ~/1/programs/rsync_git.py src-$topicname $topicname
cd -

cd $dest; git add .; cd -

# What about slides? One file here, but so far non-existing...
