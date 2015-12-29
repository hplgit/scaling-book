# Read output from pdflatex/latex, after doconce grab
# doconce grab --from- '\*File List\*' --to- '\*\*\*\*' tmp.txt > tmp.txt
# and find all styles files with full path

dont_copy = 'svmono', 't2', 'newcommands_keep'

import sys, commands
f = open(sys.argv[1], 'r')
lines = f.readlines()
paths = []
for line in lines:
    words = line.split()
    filename = words[0]
    if filename.endswith('.def') or \
       filename.endswith('.tex') or \
       filename.endswith('.sty') or \
       filename.endswith('.cls') or \
       filename.endswith('.clo') or \
       filename.endswith('.cfg') or \
       filename.endswith('.dfu'):

        if sum(filename.startswith(name) for name in dont_copy) > 0:
            continue
        failure, output = commands.getstatusoutput('kpsewhich %s' % filename)
        if not failure:
            paths.append(output)

# Write copy script
f = open('tmpcp.sh', 'w')
for path in paths:
    f.write('cp %s .\n' % path)
f.close()


