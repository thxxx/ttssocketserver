import os
from typing import Callable
from openai import OpenAI
import time

OPENAI_KEY = os.environ.get("OPENAI_KEY")

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