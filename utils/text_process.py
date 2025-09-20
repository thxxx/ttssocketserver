import re
from difflib import SequenceMatcher
from typing import List, Tuple

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\sA-Za-z0-9]")

def _tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    toks = []
    for m in TOKEN_RE.finditer(text):
        raw = m.group(0)
        norm = raw.lower().strip(",.?!;:\"“”‘’()[]{}")
        if norm == "":
            continue
        toks.append((norm, m.start(), m.end()))
    return toks

def _longest_head_tail_overlap(old_norm, new_norm, max_window: int) -> int:
    max_k = min(max_window, len(old_norm), len(new_norm))
    for k in range(max_k, 0, -1):
        if old_norm[-k:] == new_norm[:k]:
            return k
    return 0

def _fuzzy_overlap(old_norm, new_norm) -> int:
    a = " ".join(old_norm)
    b = " ".join(new_norm)
    sm = SequenceMatcher(None, a, b, autojunk=False)
    best = 0
    for blk in sm.get_matching_blocks():
        if blk.b == 0:  # new의 head에 닿는 블록 선호
            best = max(best, blk.size // 2)
    return best

def text_pr(
    old: str,
    new: str,
    *,
    tail_window_tokens: int = 40,   # 최근 5초면 여유 있게 40~80 권장
    skip_new_head_tokens: int = 1,  # ← 경계 오염 회피: new의 맨 앞 N 토큰 제외
    skip_old_tail_tokens: int = 1,  # ← (옵션) old 꼬리도 1토큰 제외
    fuzzy_min_tokens: int = 3       # 정확 겹침 없을 때 최소 허용 토큰수
) -> str:
    if not new:
        return old or ""

    old_toks = _tokenize_with_spans(old)
    new_toks = _tokenize_with_spans(new)
    if not new_toks:
        return old or ""
    if not old_toks:
        # new의 앞 N 토큰을 버리고 시작 (오염 제거)
        keep_from = new_toks[min(skip_new_head_tokens, len(new_toks)-1)][1] if len(new_toks) > 1 else new_toks[0][1]
        tail = new[keep_from:]
        return tail if not old else (old + (" " if (old and tail and not old.endswith((" ","\n")) and not tail.startswith((" ","\n"))) else "") + tail)

    old_norm = [t[0] for t in old_toks]
    new_norm = [t[0] for t in new_toks]

    # 매칭용 윈도우 추출 (old 꼬리 일부만)
    old_start_idx = max(0, len(old_norm) - tail_window_tokens)
    old_tail_norm = old_norm[old_start_idx:]
    old_tail_toks = old_toks[old_start_idx:]

    # 경계 오염 방지: old 꼬리 마지막 몇 토큰 제외
    if skip_old_tail_tokens > 0 and len(old_tail_norm) > skip_old_tail_tokens:
        old_tail_norm = old_tail_norm[:-skip_old_tail_tokens]
        old_tail_toks = old_tail_toks[:-skip_old_tail_tokens]

    # 경계 오염 방지: new 머리 토큰 제외
    if skip_new_head_tokens > 0 and len(new_norm) > skip_new_head_tokens:
        new_head_offset = skip_new_head_tokens
    else:
        new_head_offset = 0

    new_head_norm = new_norm[new_head_offset:]
    new_head_toks = new_toks[new_head_offset:]

    # 1) 정확 겹침 우선
    overlap = _longest_head_tail_overlap(old_tail_norm, new_head_norm, max_window=tail_window_tokens)

    # 2) 정확 겹침 없으면 약한 정렬
    if overlap == 0:
        overlap = _fuzzy_overlap(old_tail_norm, new_head_norm)
        if overlap < fuzzy_min_tokens:
            overlap = 0

    # new에서 붙일 시작 문자 인덱스
    if overlap >= len(new_head_toks):
        # new의 (헤드 제외) 전부가 겹침 → 아무 것도 안 붙임
        append_char_start = None
    elif overlap > 0:
        append_char_start = new_head_toks[overlap][1]
    else:
        append_char_start = new_head_toks[0][1] if new_head_toks else None

    if append_char_start is None:
        return old

    # 원문 그대로 tail 붙이기
    new_tail_raw = new[append_char_start:]

    # 공백 보정
    if old and new_tail_raw and (not old.endswith((" ", "\n"))) and (not new_tail_raw.startswith((" ", "\n"))):
        return old + " " + new_tail_raw
    return old + new_tail_raw
