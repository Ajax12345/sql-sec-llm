import llm, model
import torch

QUERY_GEN_LLM = llm.LLM('query_gen')
CRITIC_LLM = llm.LLM('critic')

def run_process(query:str, depth = 10) -> str:
    buffer = []
    openai_model = model.get_model('openai')
    with open('prompts/user/bandit.txt') as f:
        bandit_f = f.read()

    for _ in range(depth):
        bandit = ''

        if buffer:
            bandit = bandit_f.format(history = '\n\n'.join(f'query: {a}\n recommendation: {b}' for a, b in buffer))


        query = QUERY_GEN_LLM({
            'retries': 4,
            'temperature': 0.8,
            'params': {
                'system': {},
                'user': {
                    'query': query,
                    'bandit': bandit,
                }
            }
        }, display_prompts=True)['query']

        emb = llm.get_embedding(query)
        t = torch.tensor([emb])
        if openai_model(t).item() < 0.5:
            return query
        
        recommendation = CRITIC_LLM({
            'retries': 4,
            'temperature': 0.8,
            'params': {
                'system': {},
                'user': {
                    'query': query,
                }
            }
        })['recommendation']
        buffer.append((query, recommendation))
    

if __name__ == '__main__':
    
    print(run_process('''SELECT * FROM products WHERE category = 'Gifts' AND released = 1'''))
    #SELECT * FROM products WHERE category = 'Gifts' AND released = 1
    #https://portswigger.net/web-security/sql-injection#what-is-sql-injection-sqli