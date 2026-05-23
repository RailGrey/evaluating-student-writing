DISCOURSE_TYPES = [
    "Lead",
    "Position",
    "Claim",
    "Counterclaim",
    "Rebuttal",
    "Evidence",
    "Concluding Statement",
]

BIO_LABELS = ["O"]
for dt in DISCOURSE_TYPES:
    BIO_LABELS.append(f"B-{dt}")
    BIO_LABELS.append(f"I-{dt}")

LABEL2ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(BIO_LABELS)
