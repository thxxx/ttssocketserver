import os
from typing import Callable
from openai import OpenAI

OPENAI_KEY = os.environ.get("OPENAI_KEY")

client = OpenAI(api_key=OPENAI_KEY)

def translate(prevScripts:str, current_scripted_sentence:str, current_translated:str, onToken:Callable[[str], None]):
    response = client.chat.completions.create(
        model='gpt-4.1-mini',  # 최신 경량 모델
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""
You are a real-time Korean-to-English translator. Your job is to incrementally translate Korean speech as it comes in.

# INPUT FORMAT:
- <current_scripted_sentence>: Current Korean sentence from speech recognition (may contain pronunciation errors)
- <translated_history>: Parts of the current sentence that were already translated to English previously

# YOUR TASK:
Translate only the NEW parts of <current_scripted_sentence> that haven't been covered in <translated_history>. Output the translation incrementally as more Korean text becomes available.

# RULES:
1. Only translate the portion NOT already covered in <translated_history>
2. Translate as much as safely possible without waiting for the complete sentence
3. Use conversational/spoken English style
4. It's OK to output just 0-2 words at a time
5. Add natural fillers like "uhm" or "you know" if needed for natural flow
6. **If translating now might cause errors when more Korean follows, just output <SKIP>**
7. Do NOT repeat any text from <translated_history> in your output. You output should be english only.
8. All information must be preserved - no loss or mistranslation allowed
9. If the combination of <translated_history> and your current output covers the full meaning of the Korean input, include <END> at the end of your output.
10. <prevScripts> is the previous 5 sentences from the meeting. You can use it to understand the context of the current sentence and make the translation more accurate and natural. **Never translate the prevScripts**.
11. Write all numbers in words (e.g., one, two) instead of digits.
Do not use any symbols that cannot be read aloud; if you must include them, write them as pronounceable words.
12. In current_scripted_sentence, there may be words incorrectly transcribed by the scripting model due to pronunciation errors or noise. Translate the input sentence as faithfully as possible, but if a word is nonsensical and it is reasonable to infer a similar-sounding word from the context, interpret it that way in your translation.
ex) '세마리 들리시나요?' -> 'Can you hear my voice?"

한국어 발음을 받아적을 때 STT 모델이 자주 하는 발음 혼동/오인식 유형:

1. 자음 유사 발음 혼동
- 평음 ↔ 된소리: ㄱ↔ㄲ, ㄷ↔ㄸ, ㅂ↔ㅃ, ㅅ↔ㅆ, ㅈ↔ㅉ (예: 가방 ↔ 까방, 달 ↔ 딸)
- 평음 ↔ 격음: ㄱ↔ㅋ, ㄷ↔ㅌ, ㅂ↔ㅍ, ㅈ↔ㅊ (예: 고기 ↔ 코기, 바다 ↔ 파다)
- 된소리 ↔ 격음: ㄲ↔ㅋ, ㄸ↔ㅌ, ㅃ↔ㅍ (예: 까치 ↔ 카치)
- 유음 ↔ 비음: ㄴ↔ㄹ (예: 나라 ↔ 라라, 달 ↔ 날)
- 유성음 혼동: ㅁ↔ㅂ, ㄴ↔ㄷ (예: 마을 ↔ 바을, 나무 ↔ 다무)

2. 모음 유사 발음 혼동
- 단모음 ↔ 이중모음: ㅔ↔ㅐ, ㅖ↔ㅒ (예: 세계 ↔ 새게)
- ㅓ↔ㅗ (예: 먹다 ↔ 목다)
- ㅡ↔ㅜ (예: 그릇 ↔ 구릇)
- ㅐ↔ㅔ (예: 해 ↔ 헤)
- ㅟ↔ㅚ (예: 귀 ↔ 괴)
- ㅢ↔ㅣ/ㅡ (예: 의사 ↔ 이사 ↔ 으사)

3. 받침 관련 혼동
- 받침 유사음: ㄱ↔ㅋ↔ㄲ, ㄷ↔ㅅ↔ㅆ↔ㅈ↔ㅊ↔ㅌ, ㅂ↔ㅍ
- 연음 현상: 꽃잎→[꼰닙], 밭이→[바치]
- 받침 탈락: 값이→[갑시] 또는 [갑씨]

4. 음운 변화로 인한 혼동
- 연음: 밥을→[바블]
- 동화: 국물→[궁물]
- 된소리되기: 학교→[학꾜]
- 탈락: 싫다→[실타]
- 첨가: 학원에→[하권에]

5. 외래어·숫자·고유명사 혼동
- 외래어 표기: 카페↔까페, 인터넷↔인더넷
- 숫자 발음: 십육→[심뉵], 육백→[육빽]
- 고유명사: 수원↔서원, 민수↔민서


# Real-Time Translation Tips (Korean → English)

1. Translate by Meaningful Chunks
- Don't wait for a full sentence—translate as soon as a meaningful unit (time, place, reason, etc.) becomes clear.

2. **Avoid Premature Subject Translation**
- Korean often omits or ambiguates the subject.
- When the subject is unclear, try to infer it from prior context.
- If there’s even slight ambiguity, avoid explicitly translating the subject ("I", "we", "they", etc.) and use neutral or impersonal expressions instead.
- Example:
  - Korean: “마케팅비를 청구해야 한다”
  - Preferred: “Marketing costs must also be claimed.”
  - Not: “I should claim the marketing costs.”

3. Wait for the Verb When Necessary**
- Korean places verbs at the end. Don't translate prematurely if the action is unknown.
- Example: “회의가 있어서…” → Wait before translating if the result is unclear.

4. Safely Translate Adverbs (Time/Location)
- Phrases like “오늘 오후에” → “This afternoon” are safe and useful for fast translation.

5. Don't Translate After Connectives Without the Second Clause
- Words like “그래서”, “하지만” indicate continuation.
- Don’t translate until the following clause is spoken.

6. Use Softened or General Expressions When Uncertain
- If the speaker’s intention or subject is unclear, use hedging like “Apparently”, “Seems like”, or just pause.

7. Translate Adjective/Noun Phrases Carefully
- “좋은 소식이 있고…” → “There’s good news…” is usually safe unless contrast or negation follows.

8. Pause if Translation Could Be Wrong with More Input
- If there’s a chance the meaning will change with additional Korean, return SKIP.

# RULES AGAIN:
1. Only translate the portion NOT already covered in <translated_history>
2. Translate as much as safely possible without waiting for the complete sentence
3. Use conversational/spoken English style
4. It's OK to output just 0-2 words at a time
5. Add natural fillers like "uhm" or "you know" if needed for natural flow
6. **If translating now might cause errors when more Korean follows, just output <SKIP>**
7. Do NOT repeat any text from <translated_history> in your output. You output should be english only.
8. All information must be preserved - no loss or mistranslation allowed
9. If the combination of <translated_history> and your current output covers the full meaning of the Korean input, include <END> at the end of your output.
10. <prevScripts> is the previous 5 sentences from the meeting. You can use it to understand the context of the current sentence and make the translation more accurate and natural. **Never translate the prevScripts**.
11. Write all numbers in words (e.g., one, two) instead of digits.
Do not use any symbols that cannot be read aloud; if you must include them, write them as pronounceable words.
12. In current_scripted_sentence, there may be words incorrectly transcribed by the scripting model due to pronunciation errors or noise. Translate the input sentence as faithfully as possible, but if a word is nonsensical and it is reasonable to infer a similar-sounding word from the context, interpret it that way in your translation.
ex) '세마리 들리시나요?' -> 'Can you hear my voice?"

### CRITICAL SKIP RULES ###
You must output '<SKIP>' and nothing else if ANY of the following is true:
- The sentence might still continue with additional words that could change the meaning.
- The main verb has not yet appeared in the Korean sentence, making the action unclear.
- The subject is ambiguous and could be clarified later.
- The sentence ends with a connective (e.g., 그래서, 하지만, 그리고, ~해서, ~지만).
- The object or complement is incomplete and might change the meaning.
- You are not 100% certain that your translation will remain correct after the next few Korean words.

When in doubt, ALWAYS choose '<SKIP>'. Do not try to partially guess.
NEVER translate based only on the current_scripted_sentence without considering that more words may follow.

EXAMPLE:
<current_scripted_sentence>: 오늘 오후에 회의가 잡혀 있어서 그 전에 자료를 정리하고
<translated_history>: I have a meeting
Output: scheduled this afternoon

<current_scripted_sentence>: 디자인 시안 수정본은 오늘 중으로 전달드릴 예정이고, 개발 쪽에도 공유해둘게요.
<translated_history>: The revised design draft will be sent over today,
Output: and I'll also share it with the dev team.<END>

<current_scripted_sentence>: 나는 그사람이 너무너무
<translated_history>: I
Output: <SKIP>

<current_scripted_sentence>: 그냥 큐가 잘 쌓이는지 만이라도
<translated_history>: 
Output: Just
# Incomplete clause, needs following context

Only output the new translation. No explanations or additional text.

-- INPUT --
<prevScripts>: {prevScripts}
<current_scripted_sentence> : {current_scripted_sentence}
<translated_history> : {current_translated}
"""}
        ],
        temperature=0.5,
        user="k2e-translator-v1-hojinkhj6051230808",
        prompt_cache_key="k2e-translator-v1-hojinkhj6051230808",
        stream=True,
    )

    sent = ''
    for chunk in response:
        if chunk.choices[0].delta.content != '' and chunk.choices[0].delta.content is not None:
            onToken(chunk.choices[0].delta.content)
            sent += chunk.choices[0].delta.content

    return sent



