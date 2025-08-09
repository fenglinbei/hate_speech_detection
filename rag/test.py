unique_texts = ["test"] * 9
resorted_indices = [0] * len(unique_texts)
l = 0
r = len(unique_texts) - 1
for i in range(len(unique_texts)):
    if i % 2 == 0:
        resorted_indices[l] = i
        l += 1
    else:
        resorted_indices[r] = i
        r -= 1

print(resorted_indices)