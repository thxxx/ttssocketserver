import os
from typing import Callable
from openai import OpenAI

OPENAI_KEY = os.environ.get("OPENAI_KEY")

client = OpenAI(api_key=OPENAI_KEY)

def translate_simple(prevScripts:str, current_scripted_sentence:str, current_translated:str, onToken:Callable[[str], None]):
    hist = "\n".join([f" me:{x}," for x in prevScripts])
    
    response = client.chat.completions.create(
        model='gpt-4.1-mini',  # 최신 경량 모델
        messages=[
            {"role": "system", "content": "You are a professional translator specializing in [English] → [Korean] translation. Your job is to incrementally translate English speech as it comes in."},
            {"role": "user", "content": f"""
아래의 <me_speaking>를 한글로 번역해줘. 근데 사람이 말하는걸 바로 기록한거라 잘못 알아들어서 오타가 섞여있을 수 있어. 알아서 감안한 뒤 번역해줘.
Only return translated korean of <me_speaking>, Do not include anything else.
<history>는 내가 <me_speaking> 이전에 말하고 있던 문장들을 의미해. 제대로된 번역을 위한 맥락으로 사용해줘.

-- INPUT --
<history>{hist}
<me_speaking> : {current_scripted_sentence}
"""}
        ],
        temperature=0.1,
        user="k2e-translator-v1-hojinkhj6051230808",
        prompt_cache_key="k2e-translator-v1-hojinkhj6051230808",
        stream=True,
        stream_options={"include_usage": True},
    )

    sent = ''

    pt = 0
    pt_cached = 0
    ct = 0

    for chunk in response:
        if chunk.usage and chunk.usage is not None:
            u = chunk.usage;
            pt += u.prompt_tokens
            pt_cached += u.prompt_tokens_details.cached_tokens
            ct += u.completion_tokens
        else:
            if chunk.choices[0].delta.content != '' and chunk.choices[0].delta.content is not None:
                onToken(chunk.choices[0].delta.content)
                sent += chunk.choices[0].delta.content

    return {
        "text": sent,
        "prompt_tokens": pt,
        "prompt_tokens_cached": pt_cached,
        "completion_tokens": ct,
    }

def translate(prevScripts:str, current_scripted_sentence:str, current_translated:str, onToken:Callable[[str], None]):
    hist = "\n".join([f" me:{x}," for x in prevScripts])
    
    response = client.chat.completions.create(
        model='gpt-4.1-mini',  # 최신 경량 모델
        messages=[
            {"role": "system", "content": "You are a professional translator specializing in [Korean] → [English] translation. Your job is to incrementally translate Korean speech as it comes in."},
            {"role": "user", "content": f"""
Your task is to translate the input text into **natural, conversational spoken [English]**.

# INPUT FORMAT:
- <previous utterances>: Previous 5 sentences What user said. You can use it to understand the context of the current sentence and make the translation more accurate and natural, mitigating scripting errors. **Never translate the prevScripts**.
- <transcript>: Current Korean sentence from speech recognition (may contain pronunciation errors)
- <current translation>: Parts of the current sentence that were already translated to English previously

# YOUR TASK:
Translate only the NEW parts of <transcript> that haven't been covered in <current translation>. Output the translation incrementally as more Korean text becomes available.

# RULES:
1. Only translate the portion NOT already covered in <current translation>
2. Translate as much as safely possible without waiting for the complete sentence
3. Use conversational/spoken English style
4. It's OK to output just 0-2 words at a time
5. Add natural fillers like "uhm" or "you know" if needed for natural flow
6. **If translating now might cause errors when more Korean follows, just output <SKIP>**
7. Do NOT repeat any text from <current translation> in your output. You output should be english only.
8. All information must be preserved - no loss or mistranslation allowed
9. If the combination of <current translation> and your current output covers the full meaning of the Korean input, include <END> at the end of your output.
10. <previous utterances> is the previous 5 sentences from the meeting. You can use it to understand the context of the current sentence and make the translation more accurate and natural. **Never translate the prevScripts**.
11. Write all numbers in words (e.g., one, two) instead of digits.
Do not use any symbols that cannot be read aloud; if you must include them, write them as pronounceable words.
12. In current_scripted_sentence, there may be words incorrectly transcribed by the scripting model due to pronunciation errors or noise. Translate the input sentence as faithfully as possible, but if a word is nonsensical and it is reasonable to infer a similar-sounding word from the context, interpret it that way in your translation.
ex) '세마리 들리시나요?' -> 'Can you hear my voice?"

13. Preserve the original meaning, tone, and nuance.
14. Avoid overly literal translations — adapt expressions to sound natural to a native speaker.
15. Use colloquial vocabulary and sentence structure common in spoken conversation.
16. If a direct translation sounds awkward, rephrase it while keeping the intent.
17. Output only the translation, without additional commentary.
19. If you detect that <current translation> contains an error or mistranslation compared to the meaning of the current <transcript>, 
    you may output a short conversational correction in English before continuing, such as 
    "I mean", "Actually", "Wait, no", "No, I meant", etc to naturally fix the meaning.
    Keep the correction short and immediately follow it with the corrected translation.
    The correction should feel like natural spoken conversation, not a formal retraction.

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

### CRITICAL SKIP RULES ###
You must output '<SKIP>' and nothing else if ANY of the following is true:
- The sentence might still continue with additional words that could change the meaning.
- The main verb has not yet appeared in the Korean sentence, making the action unclear.
- The subject is ambiguous and could be clarified later.
- The sentence ends with a connective (e.g., 그래서, 하지만, 그리고, ~해서, ~지만).
- The object or complement is incomplete and might change the meaning.
- You are not 100% certain that your translation will remain correct after the next few Korean words.

When in doubt, ALWAYS choose '<SKIP>'. Do not try to guess the next sentence.
NEVER translate based only on the current_scripted_sentence without considering that more words may follow.

EXAMPLE:
<transcript>: 오늘 오후에 회의가 잡혀 있어서 그 전에 자료를 정리하고
<current translation>: I have a meeting
Output: scheduled this afternoon

<transcript>: 디자인 시안 수정본은 오늘 중으로 전달드릴 예정이고, 개발 쪽에도 공유해둘게요.
<current translation>: The revised design draft will be sent over today,
Output: and I'll also share it with the dev team.<END>

<transcript>: 나는 그사람이 너무너무
<current translation>: I
Output: <SKIP>

<transcript>: 그냥 큐가 잘 쌓이는지 만이라도
<current translation>: 
Output: Just
# Incomplete clause, needs following context

Only output the new translation. No explanations or additional text.

-- INPUT --
<previous utterances>: {hist}
<transcript> : {current_scripted_sentence}
<current translation> : {current_translated}
"""}
        ],
        temperature=0.1,
        user="k2e-translator-v1-hojinkhj6051230808",
        prompt_cache_key="k2e-translator-v1-hojinkhj6051230808",
        stream=True,
        stream_options={"include_usage": True},
    )

    sent = ''

    pt = 0
    pt_cached = 0
    ct = 0

    for chunk in response:
        if chunk.usage and chunk.usage is not None:
            u = chunk.usage;
            pt += u.prompt_tokens
            pt_cached += u.prompt_tokens_details.cached_tokens
            ct += u.completion_tokens
        else:
            if chunk.choices[0].delta.content != '' and chunk.choices[0].delta.content is not None:
                onToken(chunk.choices[0].delta.content)
                sent += chunk.choices[0].delta.content

    return {
        "text": sent,
        "prompt_tokens": pt,
        "prompt_tokens_cached": pt_cached,
        "completion_tokens": ct,
    }


def translate_speaker(prevScripts:str, current_scripted_sentence:str, current_translated:str, onToken:Callable[[str], None]):
    response = client.chat.completions.create(
        model='gpt-4.1-mini',  # 최신 경량 모델
        messages=[
            {"role": "system", "content": "You are a professional translator specializing in [English] → [Korean] translation. Your job is to incrementally translate English speech as it comes in."},
            {"role": "user", "content": f"""
# INPUT FORMAT:
- <previous utterances>: Previous 5 sentences What user said. You can use it to understand the context of the current sentence and make the translation more accurate and natural, mitigating scripting errors. **Never translate the prevScripts**.
- <transcript>: Current English sentence from speech recognition (may contain pronunciation errors)
- <current translation>: Parts of the current sentence that were already translated to Korean previously

# YOUR TASK:
Translate only the NEW parts of <transcript> that haven't been covered in <current translation>. Output the translation incrementally as more English text becomes available.

# RULES:
1. Only translate the portion NOT already covered in <current translation>.
2. But if you think <current translation> is wrong, return full new corrected translation and Start output with <CORRECTED>
3. Translate as much as safely possible without waiting for the complete sentence.
4. But **If translating now might cause errors when more English follows, just output <SKIP>**.
5. Use polite and formal Korean (존댓말).
6. It's OK to output just 0-2 words at a time.
7. All information must be preserved — no loss or mistranslation allowed.
8. If the combination of <current translation> and your current output covers the full meaning of the English input and you think that <transcript> is complete as a sentence, include <END> at the end of your output.
9. <previous utterances> is the previous 5 sentences from the meeting. You can use it to understand the context of the current sentence and make the translation more accurate and natural. **Never translate the them**.
10. In <transcript>, there may be words incorrectly transcribed due to pronunciation errors or noise. Translate as faithfully as possible, but if a word is nonsensical and it is reasonable to infer a similar-sounding word from the context, interpret it that way in your translation.
11. Preserve the original meaning, tone, and nuance.
12. Avoid overly literal translations — adapt expressions to sound natural in Korean.
13. If a direct translation sounds awkward, rephrase it while keeping the intent.
14. Output only the translation, without additional commentary.
15. If you detect that <current translation> contains an error or mistranslation compared to the meaning of the current <transcript>, you can output full corrected translation and start with <CORRECTED>
16. When in doubt, always choose <SKIP> rather than guessing.
17. NEVER translate based only on the current_scripted_sentence without considering that more words may follow.
18. If, when looking at <transcript>, it seems likely that more of the sentence will follow (because it is being received in real-time), you must append "..." at the end of your output.

# CRITICAL SKIP RULES:
If, when looking at <transcript>, it seems likely that more of the sentence will follow (because it is being received in real-time), you must append "..." at the end of your output.

You must output "<SKIP>" if ANY of the following is true:
- The sentence might still continue with additional words that could make current translation wrong.
- The main verb has not yet appeared in the English sentence, making the action unclear.
- The object or complement is incomplete and might change the meaning.
- You are not 100% certain that your translation will remain correct after the next few English words.

<previous utterances>를 보고 이전에 어떤 식의 얘기를 하고있었는지를 파악하고 현재 문장이 어떻게 번역되어야 자연스러운지를 반영해서 한글로 번역해줘.

# EXAMPLES:
<transcript>: I have a meeting scheduled this afternoon, so before that I need to organize the materials
<current translation>: 회의가 오늘 오후에 예정되어 있습니다.
Output: 그래서 그 전에 자료를 정리...

<transcript>: The revised design draft will be sent over today, and I'll also share it with the dev team.
<current translation>: 새로운 디자인 시안은 전달될 예정이고,
Output: <CORRECTED>수정된 디자인 시안은 오늘 중으로 전달될 예정이고, 개발팀에도 공유하겠습니다.<END>

<transcript>: Just want to make sure the queue is building up properly
<current translation>:
Output: 큐가 제대로 쌓이고 있는지만이라도...

<transcript>: I'm not telling you you got to be straight A's. I'm not telling you you got to win this or be the best. I need 100% effort, man.
<current translation>: A학점을 받아야 한다거나 이겨야 한다고 말하는 것이 아닙니다.
Output: 단지 100%의 노력을 쏟아야 한다는 거죠.<END>

<transcript>: Let's table this discussion until we have the latest numbers from finance team so
<current translation>: 이 논의를 테이블에 올려서 바로 진행합시다.
Output: <CORRECTED>재무팀의 최신 수치를 받기 전까지 이 논의는 보류하겠습니다...

-- INPUT --
<previous utterances>: {prevScripts}
<transcript> : {current_scripted_sentence}
<current translation> : {current_translated}
"""}
        ],
        temperature=0.5,
        user="k2e-translator-v1-hojinkhj6051230808-speaker",
        prompt_cache_key="k2e-translator-v1-hojinkhj6051230808-speaker",
        stream=True,
        stream_options={"include_usage": True},
    )

    sent = ''

    pt = 0
    pt_cached = 0
    ct = 0

    for chunk in response:
        if chunk.usage and chunk.usage is not None:
            u = chunk.usage;
            pt += u.prompt_tokens
            pt_cached += u.prompt_tokens_details.cached_tokens
            ct += u.completion_tokens
        else:
            if chunk.choices[0].delta.content != '' and chunk.choices[0].delta.content is not None:
                onToken(chunk.choices[0].delta.content)
                sent += chunk.choices[0].delta.content

    return {
        "text": sent,
        "prompt_tokens": pt,
        "prompt_tokens_cached": pt_cached,
        "completion_tokens": ct,
    }

def translate_speaker_make_end(prevScripts:str, current_scripted_sentence:str, current_translated:str):
    response = client.chat.completions.create(
        model='gpt-4.1-mini',  # 최신 경량 모델
        messages=[
            {"role": "system", "content": "You are a professional translator specializing in [English] → [Korean] translation. Your job is to incrementally translate Korean speech as it comes in."},
            {"role": "user", "content": f"""
# INPUT FORMAT:
- <previous utterances>: Previous 5 sentences from the meeting. You can use it to understand the context of the current sentence and make the translation more accurate and natural, mitigating recognition errors. **Never translate the prevScripts**.
- <transcript>: Current English sentence from speech recognition (may contain pronunciation errors)
- <current translation>: Parts of the current sentence that were already translated to Korean previously

# YOUR TASK:
Translate only the NEW parts of <transcript> that haven't been covered in <current translation>. Output the translation incrementally as more English text becomes available.

# RULES:
1. Only translate the portion NOT already covered in <current translation>.
2. But if you think <current translation> is wrong, return full new corrected translation and Start output with <CORRECTED>
3. Translate as much as safely possible without waiting for the complete sentence.
5. Use polite and formal Korean (존댓말).
6. It's OK to output just zero to two words.
7. All information must be preserved — no loss or mistranslation allowed.
8. If the combination of <current translation> and your current output covers the full meaning of the English input and you think that <transcript> is complete as a sentence, include <END> at the end of your output.
9. <previous utterances> is the previous 5 sentences from the meeting. You can use it to understand the context of the current sentence and make the translation more accurate and natural. **Never translate the prevScripts**. 이전에 어떤 식의 얘기를 하고있었는지를 파악하고 현재 문장이 어떻게 번역되어야 자연스러운지를 반영해서 한글로 번역해줘.
10. In <transcript>, there may be words incorrectly transcribed due to pronunciation errors or noise. Translate as faithfully as possible, but if a word is nonsensical and it is reasonable to infer a similar-sounding word from the context, interpret it that way in your translation.
11. Preserve the original meaning, tone, and nuance.
12. Avoid overly literal translations — adapt expressions to sound natural in Korean.
13. If a direct translation sounds awkward, rephrase it while keeping the intent.
14. Output only the translation, without additional commentary.
15. If you detect that <current translation> contains an error or mistranslation compared to the meaning of the current <transcript>, you can output full corrected translation and start with <CORRECTED>
17. NEVER translate based only on the current_scripted_sentence without considering that more words may follow.
18. If, when looking at <transcript>, it seems likely that the sentence is not end yet(because it is being received in real-time), 할 수 있는 만큼만 번역하고 절대 지나치게 많이 번역하려고 하지마. 그리고 you must append "..." at the end of your output.

! REMIND:
If you think <current translation> is wrong, return full new corrected translation and Start output with <CORRECTED>

# EXAMPLES:
<transcript>: I have a meeting scheduled this afternoon, so before that I need to organize the materials.
<current translation>: 회의가 오늘 오후에 예정되어 있습니다.
Output: 그래서 저는 그 전에 자료를 정리해야 합니다.<END>

<transcript>: The revised design draft will be sent over today, and I'll also share it with the dev team.
<current translation>: 새로운 디자인 시안은 전달될 예정이고,
Output: <CORRECTED>수정된 디자인 시안은 오늘 중으로 전달될 예정이고, 개발팀에도 공유하겠습니다.<END>

<transcript>: Just want to make sure the queue is building up properly.
<current translation>:
Output: 큐가 제대로 쌓이고 있는지만이라도 알고싶네요.<END>

<transcript>: I'm not telling you you got to be straight A's. I'm not telling you you got to win this or be the best. I need 100% effort, man.
<current translation>: A학점을 받아야 한다거나 이겨야 한다고 말하는 것이 아닙니다.
Output: 단지 100%의 노력을 쏟아야 한다는 거죠.<END>

<transcript>: Let's table this discussion until we have the latest numbers from finance team
<current translation>: 이 논의를 테이블에 올려서 바로 진행합시다.
Output: <CORRECTED>재무팀의 최신 수치를 받기 전까지 이 논의는 보류하겠습니다.<END>

-- INPUT --
<previous utterances>: {prevScripts}
<transcript> : {current_scripted_sentence}
<current translation> : {current_translated}
"""}
        ],
        temperature=0.6,
        user="k2e-translator-v1-hojinkhj6051230808-speaker",
        prompt_cache_key="k2e-translator-v1-hojinkhj6051230808-speaker",
        stream=True,
        stream_options={"include_usage": True},
    )

    sent = ''

    pt = 0
    pt_cached = 0
    ct = 0

    for chunk in response:
        if chunk.usage and chunk.usage is not None:
            u = chunk.usage;
            pt += u.prompt_tokens
            pt_cached += u.prompt_tokens_details.cached_tokens
            ct += u.completion_tokens
        else:
            if chunk.choices[0].delta.content != '' and chunk.choices[0].delta.content is not None:
                sent += chunk.choices[0].delta.content

    return {
        "text": sent,
        "prompt_tokens": pt,
        "prompt_tokens_cached": pt_cached,
        "completion_tokens": ct,
    }






