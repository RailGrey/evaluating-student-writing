import pandas as pd


def _word_set(predictionstring: str) -> set[int]:
    if not predictionstring or pd.isna(predictionstring):
        return set()
    return set(map(int, str(predictionstring).split()))


def _overlap(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(b)


def _match_group(
    gt_sets: list[set[int]], pred_sets: list[set[int]]
) -> tuple[list[tuple[int, int]], int, int, int]:
    candidates = []
    for gi, gt in enumerate(gt_sets):
        for pi, pred in enumerate(pred_sets):
            o1 = _overlap(gt, pred)
            o2 = _overlap(pred, gt)
            if o1 >= 0.5 and o2 >= 0.5:
                candidates.append((o1, o2, gi, pi))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    tp = 0
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matched_pairs: list[tuple[int, int]] = []
    for _, _, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        tp += 1
        used_gt.add(gi)
        used_pred.add(pi)
        matched_pairs.append((gi, pi))

    fn = len(gt_sets) - tp
    fp = len(pred_sets) - tp
    return matched_pairs, tp, fp, fn


def _validate_cols(df: pd.DataFrame, name: str) -> None:
    required = {"id", "class", "predictionstring"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _build_per_class(class_tp: dict, class_fp: dict, class_fn: dict) -> dict:
    from collections import defaultdict

    classes = sorted(set(class_tp) | set(class_fp) | set(class_fn))
    per_class = {}
    for cls in classes:
        tp = class_tp[cls]
        fp = class_fp[cls]
        fn = class_fn[cls]
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return per_class


def aggregate_averages(per_class: dict) -> dict:
    classes = sorted(per_class.keys())
    f1_scores = [per_class[c]["f1"] for c in classes]
    precisions = [per_class[c]["precision"] for c in classes]
    recalls = [per_class[c]["recall"] for c in classes]
    accuracies = [per_class[c]["accuracy"] for c in classes]
    supports = [per_class[c]["support"] for c in classes]
    total_support = sum(supports)

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    macro_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

    micro_tp = sum(per_class[c]["tp"] for c in classes)
    micro_fp = sum(per_class[c]["fp"] for c in classes)
    micro_fn = sum(per_class[c]["fn"] for c in classes)
    micro_precision = (
        micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    )
    micro_recall = (
        micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )
    micro_accuracy = (
        micro_tp / (micro_tp + micro_fp + micro_fn)
        if (micro_tp + micro_fp + micro_fn) > 0
        else 0.0
    )

    if total_support > 0:
        weighted_precision = (
            sum(p * s for p, s in zip(precisions, supports)) / total_support
        )
        weighted_recall = sum(r * s for r, s in zip(recalls, supports)) / total_support
        weighted_f1 = sum(f * s for f, s in zip(f1_scores, supports)) / total_support
        weighted_accuracy = (
            sum(a * s for a, s in zip(accuracies, supports)) / total_support
        )
    else:
        weighted_precision = weighted_recall = weighted_f1 = weighted_accuracy = 0.0

    return {
        "micro": {
            "precision": round(micro_precision, 4),
            "recall": round(micro_recall, 4),
            "f1": round(micro_f1, 4),
            "accuracy": round(micro_accuracy, 4),
        },
        "macro": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4),
            "accuracy": round(macro_accuracy, 4),
        },
        "weighted": {
            "precision": round(weighted_precision, 4),
            "recall": round(weighted_recall, 4),
            "f1": round(weighted_f1, 4),
            "accuracy": round(weighted_accuracy, 4),
        },
    }


def _example_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_data = {
        "id": ["423A1CA112E2"] * 10,
        "class": [
            "Lead",
            "Position",
            "Evidence",
            "Evidence",
            "Claim",
            "Evidence",
            "Evidence",
            "Claim",
            "Evidence",
            "Concluding Statement",
        ],
        "predictionstring": [
            "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44",
            "45 46 47 48 49 50 51 52 53 54 55 56 57 58 59",
            "60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75",
            "76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138",
            "139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162",
            "163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210",
            "211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281",
            "282 283 284 285 286 287 288 289 290 291 292 293 294 295 296",
            "297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354",
            "355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378",
        ],
    }

    pred_data = {
        "id": ["423A1CA112E2"] * 9,
        "class": [
            "Lead",
            "Position",
            "Evidence",
            "Evidence",
            "Claim",
            "Evidence",
            "Evidence",
            "Concluding Statement",
            "Claim",
        ],
        "predictionstring": [
            "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42",
            "46 47 48 49 50 51 52 53 54 55 56 57 58 59 60",
            "60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75",
            "76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120",
            "139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162",
            "163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190",
            "211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281",
            "355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378",
            "290 291 292 293 294 295 296",
        ],
    }
    return pd.DataFrame(gt_data), pd.DataFrame(pred_data)
