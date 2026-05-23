import logging

logger = logging.getLogger(__name__)


def predictions_to_spans(word_preds: list[str], min_span_length: int) -> list[dict]:
    spans = []
    j = 0
    while j < len(word_preds):
        cls = word_preds[j]
        if cls == "O":
            j += 1
            continue

        cls_normalized = cls.replace("B-", "I-")
        end = j + 1
        while end < len(word_preds) and word_preds[end] == cls_normalized:
            end += 1

        span_length = end - j
        if span_length >= min_span_length:
            class_name = cls_normalized.replace("I-", "")
            word_indices = list(range(j, end))
            spans.append(
                {
                    "class": class_name,
                    "predictionstring": " ".join(map(str, word_indices)),
                }
            )
        j = end

    return spans
