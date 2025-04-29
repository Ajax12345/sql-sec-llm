import json, typing
import torch, torch.nn as nn
import torch.optim as optim
import utils, collections
import random

class Classifier(nn.Module):
    def __init__(self, in_dim:int) -> None:
        self.in_dim = in_dim
        super().__init__()
        self.init_model()

    def init_model(self) -> None:
        self.network = nn.Sequential(
            nn.Linear(self.in_dim, 800),
            nn.ReLU(),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Linear(400, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x) -> torch.tensor:
        return self.network(x)
    
def run_training(config:dict) -> None:
    emb_src = config['emb_src']
    batch_size = config['batch_size']
    #https://machinelearningmastery.com/building-a-binary-classification-model-in-pytorch/
    model = Classifier({'openai': 1536, 'mxbai': 1024}[emb_src])
    queries, labels = zip(*utils.read_dataset())
    with open('datasets/splits.json') as f:
        partitions = json.load(f)
    

    embeddings = dict(getattr(utils, f'read_{emb_src}_embeddings')())
    
    data_train = torch.tensor([embeddings[i] for i in partitions['train']])
    labels_train = torch.tensor([[float(labels[i])] for i in partitions['train']])
    
    data_test = torch.tensor([embeddings[i] for i in partitions['test']])
    labels_test = torch.tensor([[float(labels[i])] for i in partitions['test']])
    
    loss_fn = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(config['epochs']):
        for i in range(0, len(data_train), batch_size):
            d_y = model(data_train[i:i+batch_size])
            loss = loss_fn(d_y, labels_train[i:i+batch_size])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        print(f'after epoch {epoch + 1}')
        with torch.no_grad():
            d_y = model(data_test)
            acc = (d_y.round() == labels_test).float().mean()
            acc = float(acc)
            print('accuracy', acc)

    return model



def perform_split() -> None:
    d = collections.defaultdict(list)
    for i, (_, b) in enumerate(utils.read_dataset()):
        d[b].append(i)

    m = min(map(len, d.values()))
    train, test = int(m*0.9), int(m*0.1)
    full_result = {'train':[], 'test': []}
    for a, b in d.items():
        random.shuffle(b)
        v = random.sample(b, m)
        full_result['train'].extend(v[:train])
        full_result['test'].extend(v[train:])

    with open('datasets/splits.json', 'w') as f:
        json.dump(full_result, f)

def full_training() -> None:
    openai_model = run_training({
        'emb_src': 'openai',
        'epochs': 10,
        'batch_size': 100
    })
    mxbai_model = run_training({
        'emb_src': 'mxbai',
        'epochs': 10,
        'batch_size': 100
    })

    torch.save(openai_model.state_dict(), 'openai_weights.pt')
    torch.save(mxbai_model.state_dict(), 'mxbai_weights.pt')

def get_model(name:str) -> Classifier:
    model = Classifier({'openai': 1536, 'mxbai': 1024}[name])
    model.load_state_dict(torch.load(f'{name}_weights.pt', weights_only=True))
    model.eval()
    return model

if __name__ == '__main__':
    

    model = get_model('openai')
    #mxbai accuracy: 0.9644424915313721
    #openai accuracy: 0.9951711893081665

    '''
    Use LLM to make predictions
    '''


    

