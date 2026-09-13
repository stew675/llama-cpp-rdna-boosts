import sys, hashlib
def extract(path):
    s = open(path, encoding='utf-8', errors='replace').read()
    out = []
    for ch in s:
        if ch == '\b':
            if out: out.pop()
        else:
            out.append(ch)
    lines = ''.join(out).split('\n')
    # generated text = lines after the first "> " prompt echo, before the "[ Prompt:" footer
    i = next((k for k, l in enumerate(lines) if l.startswith('> ')), None)
    if i is None:
        return None
    j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith('[ Prompt:')), len(lines))
    text = '\n'.join(lines[i + 1:j]).strip()
    return text
p = sys.argv[1]
t = extract(p)
if t is None:
    print("EXTRACT-FAILED"); sys.exit(1)
print("%d chars  sha=%s" % (len(t), hashlib.sha256(t.encode()).hexdigest()[:12]))
open(p + '.txt', 'w').write(t)
