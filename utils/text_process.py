import re

# def text_pr(old, new):
#     # old, new 둘 다 소문자로 변환
#     o = old.lower()
#     n = new.lower()

#     # 공백과 콤마 제거
#     o_clean = re.sub(r"[ ,]", "", o)
#     n_clean = re.sub(r"[ ,]", "", n)

#     # 공통 prefix 찾기
#     i = 0
#     while i < len(o_clean) and i < len(n_clean) and o_clean[i] == n_clean[i]:
#         i += 1

#     # old는 공통 부분까지만, 나머지는 new에서 가져오기
#     return new[:i] + new[i:]

def text_pr(old, new):
    # old, new 둘 다 소문자로 변환
    o = old.lower()
    n = new.lower()

    # 공백과 콤마 제거
    o_clean = re.sub(r"[ ,.]", "", o)
    n_clean = re.sub(r"[ ,.]", "", n)

    # 공통 prefix 찾기
    i = 0
    while i < len(o_clean) and i < len(n_clean) and o_clean[i] == n_clean[i]:
        i += 1

    # old는 공통 부분까지만, 나머지는 new에서 가져오기
    return old[:i] + new[i:]