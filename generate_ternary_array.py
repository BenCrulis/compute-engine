

def set_instruction(w0, w1, w2, w3):
    return f"_mm_set_ps({w0}, {w1}, {w2}, {w3})"


def unpack_ternary(w):
    w0 = w & 3
    w1 = (w >> 2) & 3
    w2 = (w >> 4) & 3
    w3 = (w >> 6) & 3

    return w0 - 1, w1 - 1, w2 - 1, w3 - 1


cases = []

for i in range(256):
    # binary count
    b = bin(i)[2:]

    ws = unpack_ternary(i)
    if 2 not in ws:
        print(f"case {i}:", b)
        print(ws)
        generated_instr = set_instruction(*ws)
        cases.append(generated_instr)
    else:
        generated_instr = set_instruction(0.0, 0.0, 0.0, 0.0)
        cases.append(generated_instr)


print("array of cases:\n")

print(",\n".join(cases))


