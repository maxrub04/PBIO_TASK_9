"""
Album number: 31880
Date: 09.05.2026
Short description: Generates a random DNA sequence in FASTA format (80 nt per line),
optionally with custom A/C/G/T percentages, motif search, and extra FASTA records for complement, reverse complement, and mRNA.
Inserts the user's name as lowercase at a random position without affecting printed statistics.
"""


import random
from typing import Optional


def validate_positive_int(
    prompt: str, min_val: int = 1, max_val: int = 100_000
) -> int:
    """Gets an integer from the user in a range. In case of an error, repeats the question."""
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
            if min_val <= value <= max_val:
                return value
        except ValueError:
            pass
        print(f"Error: value must be an integer in the range [{min_val}, {max_val}].")


def validate_seq_id(prompt: str) -> str:
    """FASTA ID: non-empty, no whitespace."""
    while True:
        seq_id = input(prompt).strip()
        if not seq_id:
            print("Error: sequence ID cannot be empty.")
            continue
        if any(ch.isspace() for ch in seq_id):
            print("Error: ID cannot contain whitespace.")
            continue
        return seq_id


def generate_sequence(length: int) -> str:
    """Returns a random DNA sequence of the specified length."""
    return "".join(random.choices("ACGT", k=length))


def calculate_stats(sequence: str) -> dict:
    """Returns a dictionary of sequence statistics.

    Keys: "A", "C", "G", "T" (float values, %), "GC" (float value, %).
    """
    n = len(sequence)
    if n == 0:
        return {"A": 0.0, "C": 0.0, "G": 0.0, "T": 0.0, "GC": 0.0}

    counts = {"A": 0, "C": 0, "G": 0, "T": 0}
    for ch in sequence.upper():
        if ch in counts:
            counts[ch] += 1

    stats = {base: round(100.0 * counts[base] / n, 2) for base in "ACGT"}
    stats["GC"] = round(100.0 * (counts["G"] + counts["C"]) / n, 2)
    return stats


def insert_name(sequence: str, name: str) -> str:
    """Inserts a name at a random position in the sequence.

    Name written in lowercase letters.
    """
    if not name.strip():
        return sequence
    piece = name.strip().lower()
    pos = random.randint(0, len(sequence))
    return sequence[:pos] + piece + sequence[pos:]


def format_fasta(
    seq_id: str, description: str, sequence: str, line_width: int = 80
) -> str:
    """Returns a formatted FASTA record as a string."""
    desc = description.strip()
    header = f">{seq_id} {desc}".rstrip() if desc else f">{seq_id}"
    lines = [header]
    for start in range(0, len(sequence), line_width):
        lines.append(sequence[start : start + line_width])
    return "\n".join(lines) + "\n"


def read_nucleotide_distribution() -> Optional[dict[str, float]]:
    """Optional custom A,C,G,T %. Press Enter for uniform. Re-prompts until sum is 100%."""
    tip = (
        "Optional base %% — enter four numbers A C G T summing to 100 "
        "(or press Enter for equal 25%% each): "
    )
    while True:
        line = input(tip).strip()
        if not line:
            return None
        parts = line.replace(",", " ").split()
        if len(parts) != 4:
            print("Error: need exactly four numbers (A C G T).")
            continue
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            print("Error: numbers only.")
            continue
        total = sum(vals)
        if abs(total - 100.0) > 0.001:
            print(f"Error: percentages must sum to 100 (got {total}).")
            continue
        return {"A": vals[0], "C": vals[1], "G": vals[2], "T": vals[3]}


def sequence_from_distribution(length: int, pct: dict[str, float]) -> str:
    """Random DNA with given nucleotide probabilities."""
    bases = ["A", "C", "G", "T"]
    weights = [pct[b] for b in bases]
    return "".join(random.choices(bases, weights=weights, k=length))




def find_motif_positions(sequence: str, motif: str) -> list[int]:
    """All 1-based start positions of motif (may overlap)."""
    pat = motif.upper().strip()
    if not pat:
        return []
    hay = sequence.upper()
    out: list[int] = []
    i = 0
    while True:
        j = hay.find(pat, i)
        if j == -1:
            break
        out.append(j + 1)
        i = j + 1
    return out



_COMP = str.maketrans("ACGT", "TGCA")


def dna_complement(sequence: str) -> str:
    """Complementary DNA strand."""
    return sequence.upper().translate(_COMP)


def dna_reverse_complement(sequence: str) -> str:
    """Reverse complement."""
    return dna_complement(sequence)[::-1]


def dna_to_mrna(sequence: str) -> str:
    """Replace T by U on the coding strand."""
    return sequence.upper().replace("T", "U")


def main() -> None:
    """Single-sequence flow from the exercise + four optional features."""
    length = validate_positive_int("Enter sequence length: ")
    seq_id = validate_seq_id("Enter sequence ID: ")
    description = input("Enter a description of the sequence: ").rstrip()
    dist = read_nucleotide_distribution()
    name = input("Enter your name: ").strip()
    motif_q = input("Motif to search after generation (empty to skip): ").strip()

    if dist is None:
        dna = generate_sequence(length)
    else:
        dna = sequence_from_distribution(length, dist)

    stats = calculate_stats(dna)
    fasta_seq = insert_name(dna, name)

    records = [
        (seq_id, description, fasta_seq),
        (f"{seq_id}_comp", "complement", dna_complement(dna)),
        (f"{seq_id}_revcomp", "reverse complement", dna_reverse_complement(dna)),
        (f"{seq_id}_mrna", "mRNA T->U", dna_to_mrna(dna)),
    ]

    path = f"{seq_id}.fasta"
    with open(path, "w", encoding="ascii") as fh:
        for sid, desc, seq in records:
            fh.write(format_fasta(sid, desc, seq))

    print(f"\nSequence saved to file: {path}")
    print(f"\nSequence statistics (n={length}):")
    for b in "ACGT":
        print(f"{b}: {stats[b]:.2f}%")
    print(f"GC- content : {stats['GC']:.2f}%")

    if motif_q:
        hits = find_motif_positions(dna, motif_q)
        if hits:
            print("Motif positions (1-based):", ", ".join(map(str, hits)))
        else:
            print("Motif not found.")


if __name__ == "__main__":
    main()
