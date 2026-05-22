import nltk
from nltk.tokenize import sent_tokenize

DISCOURSE_TYPES = [
    "Lead",
    "Position",
    "Claim",
    "Counterclaim",
    "Rebuttal",
    "Evidence",
    "Concluding Statement",
]

CLASSES = DISCOURSE_TYPES + ["Other"]


def ensure_nltk_data() -> None:
    try:
        sent_tokenize("test")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)


def split_sentences(text: str) -> list[str]:
    ensure_nltk_data()
    return sent_tokenize(text)


def split_words(text: str) -> list[str]:
    return text.split()


def char_to_word_index(text: str, char_pos: int) -> int:
    return len(text[:char_pos].split())


def word_range_for_text(text: str) -> list[tuple[int, int]]:
    words = split_words(text)
    ranges: list[tuple[int, int]] = []
    for i, word in enumerate(words):
        start = text.find(word, sum(len(w) for w in words[:i]) + i)
        end = start + len(word)
        ranges.append((start, end))
    return ranges


def assign_sentence_labels(
    essay_text: str,
    sentences: list[str],
    discourse_annotations: list[dict],
) -> list[str]:
    word_ranges = word_range_for_text(essay_text)
    sentence_word_ranges: list[tuple[int, int]] = []
    char_cursor = 0
    for sent in sentences:
        sent_start = essay_text.find(sent, char_cursor)
        if sent_start == -1:
            sent_start = char_cursor
        sent_end = sent_start + len(sent)
        first_word = char_to_word_index(essay_text, sent_start)
        last_word = char_to_word_index(essay_text, sent_end)
        if last_word > first_word:
            last_word -= 1
        sentence_word_ranges.append((first_word, last_word))
        char_cursor = sent_end

    labels = ["Other"] * len(sentences)
    for ann in discourse_annotations:
        ann_start = ann["discourse_start"]
        ann_end = ann["discourse_end"]
        ann_first_word = char_to_word_index(essay_text, ann_start)
        ann_last_word = char_to_word_index(essay_text, ann_end) - 1
        for i, (s_start, s_end) in enumerate(sentence_word_ranges):
            if s_end < ann_first_word:
                continue
            if s_start > ann_last_word:
                break
            overlap_start = max(s_start, ann_first_word)
            overlap_end = min(s_end, ann_last_word)
            overlap = overlap_end - overlap_start + 1
            sentence_len = s_end - s_start + 1
            if overlap >= sentence_len * 0.3:
                labels[i] = ann["discourse_type"]
    return labels


def get_sentence_word_ranges(
    essay_text: str, sentences: list[str]
) -> list[tuple[int, int]]:
    char_cursor = 0
    ranges: list[tuple[int, int]] = []
    for sent in sentences:
        sent_start = essay_text.find(sent, char_cursor)
        if sent_start == -1:
            sent_start = char_cursor
        sent_end = sent_start + len(sent)
        first_word = char_to_word_index(essay_text, sent_start)
        last_word = char_to_word_index(essay_text, sent_end)
        if last_word > first_word:
            last_word -= 1
        ranges.append((first_word, last_word))
        char_cursor = sent_end
    return ranges


def merge_segments(
    sentence_labels: list[str],
    sentence_word_ranges: list[tuple[int, int]],
) -> list[dict]:
    if not sentence_labels:
        return []
    segments: list[dict] = []
    current_class = sentence_labels[0]
    current_start = sentence_word_ranges[0][0]

    for i in range(1, len(sentence_labels)):
        if sentence_labels[i] != current_class:
            current_end = sentence_word_ranges[i - 1][1]
            if current_class != "Other":
                word_indices = list(range(current_start, current_end + 1))
                segments.append(
                    {
                        "class": current_class,
                        "predictionstring": " ".join(map(str, word_indices)),
                    }
                )
            current_class = sentence_labels[i]
            current_start = sentence_word_ranges[i][0]

    if current_class != "Other":
        last_end = sentence_word_ranges[-1][1]
        word_indices = list(range(current_start, last_end + 1))
        segments.append(
            {
                "class": current_class,
                "predictionstring": " ".join(map(str, word_indices)),
            }
        )
    return segments
