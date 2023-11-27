import openai
import pandas as pd
import os
import re
import tqdm
import time

def remove_numbers(input_string):
    return re.sub(r'\d+', '', input_string)

openai.api_key = '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'
print(os.getcwd())

df = pd.read_csv(f"Video-ChatGPT/scripts/50salads_context.csv")
# print(df)
os.makedirs("Video-ChatGPT/Fawad_saladsGPT",exist_ok=True)
for i in tqdm.tqdm(range(df.shape[0])):
    file =df.iloc[i,0].lower()
    if os.path.exists(f"Video-ChatGPT/Fawad_saladsGPT/{file}"):
        print("skipped",file)
        continue
    context=df.iloc[i,1].lower()

    messages = []
    actionx = action.split('_')[0]
    camera = remove_numbers(cam)
    message = f"""
Dataset: Breakfast 
Visual: {context}

Camera: {camera}

Action: {actionx}

Frames and Segments:
{lines}

SIL = "Static"

Fill the below 2 formats. 
keeping in mind the context I provided, You must write each segment and its frames with a detailed caption for the first question. 
you must rewrite each question (Q) in  CHAT-STYLE.
DON'T SKIP ANY QUESTIONS or ANY FORMAT.

Format 1:

Q: <Question about performed action>
A: <Answer which action is performed>

Q: <Question about segments in the video?> 
A: <frame start> - <frame end> <action name>
<action description>
Write all segments and their frames.

Format 2:

<ask 2-10 questions>
<For each of the random moments (not frame numbers) in the video>
Q: <Question about <moment like start/middle/end/etc>?> 
A: <frame start> - <frame end> <action name>
<action description>

<ask 2-10 questions>
Q: <Question about the visuals in video based on the context provided>
A: <Your responses>
"""
    messages.append(
    {"role": "user", "content": message},
)   
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
        )
    reply = chat.choices[0].message.content

    if not os.path.exists(f'Breakfast/{person}'): os.mkdir(f'Breakfast/{person}')
    if not os.path.exists(f'Breakfast/{person}/{cam}'): os.mkdir(f'Breakfast/{person}/{cam}')
    with open(f'Breakfast/{person}/{cam}/{ap}.avi.labels','w') as f:
        f.writelines(reply)
    f.close()
    if i > 0 and i % 100 == 0:
        time.sleep(10)