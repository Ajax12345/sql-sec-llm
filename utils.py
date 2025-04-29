import openai, typing
import csv, os
import json, ollama

CLIENT = openai.OpenAI(
  api_key=os.environ['OPENAI_API']
)

CONTEXT_LIMIT = 128000

def get_embedding(text:str) -> typing.List[float]:
    return CLIENT.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float"
    ).data[0].embedding

def get_mxbai_embedding(text:str) -> typing.List[float]:
    return ollama.embed(
        model='mxbai-embed-large',
        input=text,
    ).embeddings[0]


def read_dataset() -> typing.List:
    with open('datasets/Modified_SQL_Dataset.csv') as f:
        _, *d = csv.reader(f)
        return d
    
def read_openai_embeddings() -> typing.List:
    with open('/Users/jamespetullo/sql-sec-data/openai_embeddings.json') as f:
        return json.load(f)
    
def read_mxbai_embeddings() -> typing.List:
    with open('/Users/jamespetullo/sql-sec-data/mxbai_embeddings.json') as f:
        return json.load(f)

def run_openai_embeddings() -> None:
    with open('/Users/jamespetullo/sql-sec-data/openai_embeddings.json') as f:
        data = json.load(f)
        max_id = -1 if not data else max(a for a, _ in data)


    for i, (query, _) in enumerate(read_dataset()):
        if i > max_id:
            data.append([i, get_embedding(query)])
            '''
            if not i%300:
                with open('datasets/openai_embeddings.json', 'w') as f:
                    json.dump(data, f)

                print(i)
            '''

    with open('/Users/jamespetullo/sql-sec-data/openai_embeddings.json', 'w') as f:
        json.dump(data, f)

def run_mxbai_embeddings() -> None:
    with open('/Users/jamespetullo/sql-sec-data/mxbai_embeddings.json') as f:
        data = json.load(f)
        max_id = -1 if not data else max(a for a, _ in data)

    for i, (query, _) in enumerate(read_dataset()):
        if i > max_id:
            data.append([i, get_mxbai_embedding(query)])
            
            if not i%300:
                with open('/Users/jamespetullo/sql-sec-data/mxbai_embeddings.json', 'w') as f:
                    json.dump(data, f)

                print(i)

    with open('/Users/jamespetullo/sql-sec-data/mxbai_embeddings.json', 'w') as f:
        json.dump(data, f)

def repair():
    s = ''
    last_seen = []
    with open('/Users/jamespetullo/sql-sec-data/openai_embeddings.json') as f:
        d = f.read()
        v = json.loads(d+']]]')
        v = [i for i in v if len(i[1]) == 1536]
    
    print(max(a for a, _  in v))
    with open('/Users/jamespetullo/sql-sec-data/openai_embeddings.json', 'w') as f:
        json.dump(v, f)
            

    #print(max(map(len, last_seen)))

def validate() -> None:
    with open('/Users/jamespetullo/sql-sec-data/openai_embeddings.json') as f:
        data = json.load(f)
        assert len(data) == len(read_dataset())

if __name__ == '__main__':
    #run_embeddings()
    #repair()
    run_mxbai_embeddings()
