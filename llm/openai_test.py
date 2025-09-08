import os
from typing import Callable
from openai import OpenAI
import time

OPENAI_KEY = os.environ.get("OPENAI_KEY")

LANGUAGE_CODE = {
    "Arabic": "ar",
    "Danish": "da",
    "German": "de",
    "Greek": "el",
    "English": "en",
    "Spanish": "es",
    "Finnish": "fi",
    "French": "fr",
    "Hebrew": "he",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Malay": "ms",
    "Dutch": "nl",
    "Norwegian": "no",
    "Polish": "pl",
    "Portuguese": "pt",
    "Russian": "ru",
    "Swedish": "sv",
    "Swahili": "sw",
    "Turkish": "tr",
    "Chinese": "zh",
}

# 키와 값을 반대로 뒤집은 dict
LANGUAGE_CODE_REVERSED = {v: k for k, v in LANGUAGE_CODE.items()}


client = OpenAI(api_key=OPENAI_KEY)

def translate_simple(prevScripts:str, current_scripted_sentence:str, current_translated:str, onToken):
    hist = "\n".join([f" me:{x}," for x in prevScripts])

    response = client.chat.completions.create(
        model='gpt-4.1-mini',  # 최신 경량 모델
        messages=[
            {"role": "system", "content": "You are a professional translator specializing in [Korean] → [English] translation."},
            {"role": "user", "content": f"""
지금 계속 한글로 말하는걸 영어로 번역하고 있어.
<previous utterances>는 현재 문장 이전에 이야기하던 문장이야. 번역을 위한 맥락 파악에 사용할 수 있어.
<speaking english>은 번역해야하는 현재 발화야.

말을 하는걸 script로 만든 input이기 때문에, 발음 문제로 인해서 텍스트가 잘못 들어왔을 수 있어. 그걸 감안해서 번역해줘.

출력 english를 일반 글 문장보다는 실제로 사람이 말하는 것 같은 구어체로 적어줘. 예를 들어, 같은 단어를 두번 쓰거나 뭐 ...을 쓰거나 느낌표 이런 것들 있잖아?
Translate into casual spoken English. 근데 너무 심하게 하진 말고, 없는 말을 만들거나 들어온 input을 왜곡하면 안돼.
Do not start with word like Oh, So, Uhm, Huh, etc.

-- INPUT --
<previous utterances>{hist}
<speaking korean> : {current_scripted_sentence}
<english> : 
"""}
        ],
        temperature=0.1,
        user="k2e-translator-v1-hojinkhj6051230808",
        prompt_cache_key="k2e-translator-v1-hojinkhj6051230808",
        stream=True,
        stream_options={"include_usage": True},
    )

    sent = ''
    first = 0
    st = time.time()

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
                if first == 0:
                    first = time.time() - st
                onToken(chunk.choices[0].delta.content)
                sent += chunk.choices[0].delta.content

    return {
        "text": sent,
        "prompt_tokens": pt,
        "prompt_tokens_cached": pt_cached,
        "completion_tokens": ct,
        "ttft": first
    }

def translate(prevScripts:str, current_scripted_sentence:str, current_translated:str, onToken, input_language:str = 'Korean', output_language:str = 'English'):
    hist = "\n".join([f" me:{x}," for x in prevScripts])
    input_language = LANGUAGE_CODE_REVERSED[input_language]
    output_language = LANGUAGE_CODE_REVERSED[output_language]
    
    response = client.chat.completions.create(
        model='gpt-4.1-mini',  # 최신 경량 모델
        messages=[
            {"role": "system", "content": f"You are a professional translator specializing in [{input_language}] → [{output_language}] translation."},
            {"role": "user", "content": f"""
You are translating {input_language} speech into {output_language}.

<previous utterances> are the sentences spoken before the current one. Use them for context.  
<speaking {input_language}> is the current spoken input that needs to be translated.  
<{output_language}> is the translation generated so far.  

The input comes from speech-to-text, so there may be transcription errors due to pronunciation. Please take this into account when translating.  

Output the translation in casual spoken {output_language} — like how people actually talk, with natural pauses, repetitions, or fillers such as “...” or “!” — but don’t overdo it. Do not invent new words or distort the original meaning.  
Do not start with words like “Oh”, “So”, “Uhm”, or “Huh”.  

If the {input_language} input seems incomplete (cut off mid-sentence), output an unfinished {output_language} sentence too, so it can be naturally continued. You don’t need to force a full translation of every fragment if it isn’t complete yet.
한글 종결 어미의 특징 : ~~요. ~~니다. ~~어.

If there is already some translated {output_language}, continue from it. !Do not include the previous translation in the output never!
The earlier translation may not be perfect — refine it naturally into spoken {output_language}. Only output the additional translated part.  

If the current {input_language} input is fully translated and nothing else needs to be added, end with `<END>`.  
If more input is likely to follow, end with `...`.  

# Real-Time Translation Tips
1. **Avoid Premature Subject Translation**
- Korean often omits or ambiguates the subject.
- When the subject is unclear, try to infer it from prior context.
- If there’s even slight ambiguity, avoid explicitly translating the subject ("I", "we", "they", etc.) and use neutral or impersonal expressions instead.
- Example:
  - Korean: “마케팅비를 청구해야 한다”
  - Preferred: “Marketing costs must also be claimed.<END>”
  - Not: “I should claim the marketing costs.<END>”
- Example:
  - Korean: "내일 집에"
  - Preferred: "Tomorrow,"
  - Not: "Tomorrow at home, I'll "

2. Do not include the verb in the translation if no verb is spoken.
- Korean places verbs at the end. Don't translate prematurely if the action is unknown.
- Example: “운동장에가서 축구를…”
  - Preferred: "I'll go to the sports field and..." # Still don't know whether they will play soccer or watch. Just skip the sentence, because there will be more input to come next.
  - Not: "I'll go to the sports field and play soccer..." # This is not correct, because next input can be "축구를 봤어."

EXAMPLE:
<speaking {input_language}>: 오늘 오후에 회의가 잡혀 있어서 그 전에 자료를 정리하고
<{output_language}>: I have a meeting
Output: scheduled this afternoon...

<speaking {input_language}>: 디자인 시안 수정본은 오늘 중으로 전달드릴 예정이고, 개발 쪽에도 공유해둘게요.
<{output_language}>: The revised design draft will be sent over today,
Output: and I'll also share it with the dev team.<END>

-- INPUT --  
<previous utterances> {hist}  
<speaking {input_language}> : {current_scripted_sentence}  
<{output_language}> : {current_translated}  
"""}
        ],
        temperature=0.1,
        user="k2e-translator-v1-hojinkhj6051230808",
        prompt_cache_key="k2e-translator-v1-hojinkhj6051230808",
        stream=True,
        stream_options={"include_usage": True},
    )

    sent = ''
    first = 0
    st = time.time()

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
        "completion_tokens": ct
    }