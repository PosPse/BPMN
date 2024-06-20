import sys
sys.path.append('/home/btr/bpmn/llmsage/src')

from data_utils import DataCenter
from model import GCN, GraphSage
import torch
import Parser


args = Parser.args
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_center = DataCenter(datasets_json=args.datasets_json, vocab_dir=args.vocab_dir, vocab_len=args.vocab_len, embedding_size=args.embedding_size)
tarin_dataloader = data_center.get_train_dataloader(args.batch_size, args.shuffle)
test_dataloader = data_center.get_test_dataloader(args.batch_size, args.shuffle)
# gcn_model = GCN(embedding_size=args.embedding_size, hidden_size=args.hidden_size, num_classes=args.num_classes).to(device)
gcn_model = GraphSage(embedding_size=args.embedding_size, hidden_size=args.hidden_size, num_classes=args.num_classes, aggr=args.aggr).to(device)
optimizer = torch.optim.SGD(gcn_model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss().to(device)

def train():
    for epoch in range(args.epochs):
        gcn_model.train()
        LOSS = 0
        for batch_data in tarin_dataloader:
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            output = gcn_model(batch_data)
            loss = criterion(output, batch_data.y)
            LOSS += loss.item()
            loss.backward()
            optimizer.step()
        print(f'device: {device}, Epoch {epoch+1}/{args.epochs}, Train Loss: {(LOSS/len(tarin_dataloader))}')
        test()

def test():
    gcn_model.eval()
    with torch.no_grad():
        correct_pred_num = 0
        node_num = 0
        for batch_data in test_dataloader:
            batch_data = batch_data.to(device)
            output = gcn_model(batch_data)
            pred = output.argmax(dim=1)
            node_num += len(batch_data.y)
            correct_pred_num += torch.sum(pred == batch_data.y)
        accurcy = correct_pred_num / node_num
        print(f'device: {device}, Test Accuracy: {accurcy}')

if __name__ == '__main__':
    print(len(tarin_dataloader))
    print(len(test_dataloader))
    train()