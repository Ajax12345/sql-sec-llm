import openai, typing
import json, re, ast
import datetime, time
import tiktoken, random
import os

#https://platform.openai.com/docs/overview
#https://github.com/intellectronica/text-clustering-embedding-vs-prompting/blob/main/text-clustering-embedding-vs-prompting.ipynb
#https://github.com/fanchongjion/LATuner/blob/main/mab.py#L42
#https://platform.openai.com/docs/guides/function-calling


CLIENT = openai.OpenAI(
  api_key=""
)

CONTEXT_LIMIT = 128000

def get_embedding(text:str) -> typing.List[float]:
    return CLIENT.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float"
    ).data[0].embedding

def query_gpt(system:str, user:str, 
            temperature:float = 1, model:str = 'gpt-4o-mini') -> str:
    completion = CLIENT.chat.completions.create(
        model=model,
        store=True,
        temperature = temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )


    return completion.choices[0].message.content


class LLM:
    def __init__(self, name:str) -> None:
        self.name = name

    def parse_json_response(self, response:str) -> typing.List:
        #return ast.literal_eval(re.findall('```json[\w\W]+', response)[0].lstrip('```json\n').rstrip('\n```'))
        return {'query': re.findall('(?<=query"\:\s)"(.*?)"', response)[0]}
    
    def __call__(self, config:dict, display_prompts:bool = False) -> typing.Any:
        '''
        config:
            - retries
            - temperature
            - model (optional, defaults to gpt-4o-mini)
            - params
                - system
                - user
        '''
        with open(os.path.join('prompts', 'system', f'{self.name}.txt')) as f:
            system = f.read().format(**config['params']['system'])

        with open(os.path.join('prompts', 'user', f'{self.name}.txt')) as f:
            user = f.read().format(**config['params']['user'])
        
        for _ in range(config['retries']):
            resp = query_gpt(system, user, config['temperature'], config.get('model', 'gpt-4o-mini'))
            if display_prompts:
                print('user prompt')
                print(user)
                print('-'*50)
                print(resp)
            try:
                return self.parse_json_response(resp)

            except Exception as e:
                
                print('got openai error')
                print()
                print(e)
                print('-'*40)
                

if __name__ == '__main__':
    pass