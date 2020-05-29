import docopt
args=docopt.docopt("""
        Usage: mc_extract_vocabs <train_file> <vocab_prefix> [options]

        Options:
            --min_count n  [default: 20]
""")
from collections import *
minc = int(args['--min_count'])
maxV = -1
vocabs = defaultdict(Counter)
for line in file(args['<train_file>']):
    group_a, group_b = line.strip().split(" ||| ")
    group_a = group_a.split()
    group_b = group_b.split()
    va = int(group_a[0])
    vb = int(group_b[0])
    maxV = max([maxV, va, vb])
    vocabs[va].update(group_a[1:])
    vocabs[vb].update(group_b[1:])

for i in xrange(maxV+1):
    fout = file("%s%d" % (args['<vocab_prefix>'], i),'w')
    for w,c in vocabs[i].iteritems():
        if c >= minc:
            print >> fout, w, c
    fout.close()
print "num_vocabs:", maxV+1

