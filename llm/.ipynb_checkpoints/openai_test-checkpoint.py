import os
from typing import Callable
from openai import OpenAI

OPENAI_KEY = os.environ.get("OPENAI_KEY")

client = OpenAI(api_key=OPENAI_KEY)


def translate_simple(prevScripts:str, current_scripted_sentence:str, current_translated:str, prevTranslations, onToken):
    hist = "\n".join([f" me:{x}," for x in prevScripts])
    
    response = client.chat.completions.create(
        model='gpt-4.1-mini',  # 최신 경량 모델
        messages=[
            {"role": "system", "content": "You are a professional translator specializing in [English] → [Korean] translation."},
            {"role": "user", "content": f"""
지금 계속 영어로 말하는걸 한글로 번역하고 있어.
<previous utterances>는 현재 문장 이전에 이야기하던 문장이야. 번역을 위한 맥락 파악에 사용할 수 있어.
<speaking english>은 번역해야하는 현재 발화야.
<korean>은 <speaking english>을 번역 중인 상태의 문장이야.

최대한 빨리 번역을 하고 싶기 때문에 말을 하는 도중의 문장이 <speaking english>에 들어왔을 수도 있어.


-- INPUT --
<previous utterances>{hist}
<speaking english> : {current_scripted_sentence}
<korean> : {current_scripted_sentence}
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