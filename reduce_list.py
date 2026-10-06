import random, sys, collections

src, dst, max_per_class = sys.argv[1], sys.argv[2], int(sys.argv[3])
max_classes = int(sys.argv[4]) if len(sys.argv) > 4 else None
rng = random.Random(1)

by_class = collections.defaultdict(list)
for line in open(src):
    line = line.strip()
    if line:
        by_class[line.split('/')[0]].append(line)

classes = sorted(by_class)
if max_classes:
    classes = sorted(rng.sample(classes, min(max_classes, len(classes))))

with open(dst, 'w') as f:
    for c in classes:
        vids = by_class[c]
        keep = rng.sample(vids, min(max_per_class, len(vids)))
        f.write('\n'.join(sorted(keep)) + '\n')
