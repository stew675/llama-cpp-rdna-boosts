import random
random.seed(1234)
words = ("the expert router selects a small subset of tokens from a large bank of learned parameters "
         "and the result is combined with a shared path before it reaches the next layer").split()
out = []
n = 0
target = 900_000  # characters, ~200k tokens
while n < target:
    s = " ".join(random.choice(words) for _ in range(20))
    out.append(s.capitalize() + ".")
    n += len(s) + 2
open("prompt_long.txt", "w").write("\n".join(out) + "\n")
open("prompt_short.txt", "w").write("Explain in detail how a mixture of experts layer routes tokens.\n")
print("long chars", n)
