import re
import unicodedata
from typing import Dict, Optional

class TranscriptStabilizer:
    """
    - 앞부분은 '정규화 토큰' 기준으로 안정화 (대소문자/구두점/공백 차이 무시)
    - 마지막 N 단어는 새 결과로 교체 허용 (끊긴 단어 보정)
    - 출력은 가능하면 새 가설(new_text)의 원문 포맷을 보존
    - '완전히 새로운 문장'처럼 앞부분이 크게 달라지면 전체 교체(RESET)
    """

    def __init__(
        self,
        replace_last_n: int = 2,       # 뒤에서 교체 허용할 단어 개수
        min_append_tokens: int = 1,    # 안정적으로 붙일 최소 토큰(너무 잦은 깜빡임 억제)
        require_prefix_match: bool = True,  # 안정 구간(prefix)이 어긋나면 RESET
    ):
        self.replace_last_n = max(0, replace_last_n)
        self.min_append_tokens = max(1, min_append_tokens)
        self.require_prefix_match = require_prefix_match

        self._committed_text: str = ""        # 화면에 보여준 확정 텍스트(원문 포맷)
        self._committed_norm_tokens: list[str] = []  # 정규화 토큰(비교용)

    # ---------- 내부 유틸 ----------

    def _normalize(self, s: str) -> str:
        # 1) 소문자
        s = s.lower()
        # 2) 유니코드 구두점/기호 → 공백 치환
        cleaned = []
        for ch in s:
            cat = unicodedata.category(ch)
            if cat.startswith("P") or cat.startswith("S"):
                cleaned.append(" ")
            else:
                cleaned.append(ch)
        s = "".join(cleaned)
        # 3) 공백 정규화
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _tokens(self, s: str) -> list[str]:
        # 유니코드 단어(\w) 기준 토큰화
        return re.findall(r"\w+", s, flags=re.UNICODE)

    def _word_spans(self, s: str) -> list[tuple[int, int]]:
        # 단어의 (start, end) 문자 인덱스 span 리스트 (원문 포맷 복원용)
        return [m.span() for m in re.finditer(r"\w+", s, flags=re.UNICODE)]

    # ---------- 공개 API ----------

    def reset(self) -> None:
        self._committed_text = ""
        self._committed_norm_tokens = []

    @property
    def committed(self) -> str:
        return self._committed_text

    def update(self, new_text: str) -> Dict[str, Optional[str]]:
        """
        새 가설(new_text)을 받아 상태를 갱신하고 결과를 반환.
        returns:
            {
              "committed": 최종 확정 텍스트,
              "appended":  이번에 실제로 새로 붙인 원문 (RESET이면 new_text 전체),
              "action":    "append" | "replace" | "noop"
            }
        """
        # 초기 상태
        if not self._committed_text:
            self._committed_text = new_text
            self._committed_norm_tokens = self._tokens(self._normalize(new_text))
            return {"committed": self._committed_text, "appended": new_text, "action": "replace"}

        # 정규화 토큰 비교
        old_norm = self._committed_norm_tokens
        new_norm = self._tokens(self._normalize(new_text))

        # 공통 prefix 길이
        lcp = 0
        while lcp < len(old_norm) and lcp < len(new_norm) and old_norm[lcp] == new_norm[lcp]:
            lcp += 1

        # 안정적으로 고정할 토큰 수 (마지막 N 단어는 교체 허용)
        stable_count = max(0, len(old_norm) - self.replace_last_n)

        # 앞부분이 크게 달라졌다면(특히 안정 구간이 불일치) → RESET
        if self.require_prefix_match and lcp < stable_count:
            self._committed_text = new_text
            self._committed_norm_tokens = new_norm
            return {"committed": self._committed_text, "appended": new_text, "action": "replace"}

        # 새로 추가할 정규화 토큰 수
        new_norm_count = len(new_norm) - stable_count
        if new_norm_count < self.min_append_tokens:
            # 아직 붙일 만큼 안정되지 않음 → NOOP
            return {"committed": self._committed_text, "appended": None, "action": "noop"}

        # ===== 원문 포맷으로 안전하게 '뒤쪽만' 붙이기 =====
        # new_text에서 "stable_count번째 단어 이후"를 원문 그대로 잘라 append
        spans = self._word_spans(new_text)
        if stable_count < len(spans):
            start_char = spans[stable_count][0]
        else:
            start_char = len(new_text)  # 새 단어가 없다면 끝에서 붙이기(= 빈 append)

        appended = new_text[start_char:]
        if not appended:
            return {"committed": self._committed_text, "appended": None, "action": "noop"}

        # 커밋 갱신
        self._committed_text += appended
        self._committed_norm_tokens = new_norm[: stable_count + new_norm_count]

        return {"committed": self._committed_text, "appended": appended, "action": "append"}
