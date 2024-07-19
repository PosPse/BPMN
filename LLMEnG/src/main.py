import sys
sys.path.append('/home/btr/bpmn/LLMEnG/src')

from data_utils import DataCenter
from model import GCN, GraphSage, EdgeClassification, NodeFusion, GAT
import torch
import Parser


args = Parser.args
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_center = DataCenter(datasets_json=args.datasets_json, vocab_dir=args.vocab_dir, vocab_len=args.vocab_len, embedding_size=args.embedding_size)
tarin_dataloader = data_center.get_train_dataloader(args.batch_size, args.shuffle)
test_dataloader = data_center.get_test_dataloader(args.batch_size, args.shuffle)
# node_model = GCN(embedding_size=args.embedding_size, hidden_size=args.hidden_size, node_num_classes=args.node_num_classes).to(device)
# node_model = GraphSage(embedding_size=args.embedding_size, hidden_size=args.hidden_size, node_num_classes=args.node_num_classes, aggr=args.aggr).to(device)
node_model = GAT(embedding_size=args.embedding_size, hidden_size=args.hidden_size, node_num_classes=args.node_num_classes).to(device)
node_optimizer = torch.optim.SGD(node_model.parameters(), lr=args.lr)
node_criterion = torch.nn.CrossEntropyLoss().to(device)

# node_model = torch.load('/home/btr/bpmn/LLMEnG/src/node_model.pth', map_location=device)
# node_model.eval()

node_fusion_model = NodeFusion(fusion_method=args.fusion_method).to(device)
edge_model = EdgeClassification(node_fusion = node_fusion_model).to(device)
edge_optimizer = torch.optim.SGD(edge_model.parameters(), lr=args.lr)
edge_criterion = torch.nn.CrossEntropyLoss().to(device)

def node_train():
    for epoch in range(args.epochs):
        node_model.train()
        LOSS = 0
        for batch_data in tarin_dataloader:
            node_optimizer.zero_grad()
            batch_data = batch_data.to(device)
            output = node_model(batch_data)
            loss = node_criterion(output, batch_data.y)
            LOSS += loss.item()
            loss.backward()
            node_optimizer.step()
        print(f'device: {device}, Epoch {epoch+1}/{args.epochs}, Train Loss: {(LOSS/len(tarin_dataloader))}')
        node_test()

def node_test():
    node_model.eval()
    with torch.no_grad():
        correct_pred_num = 0
        node_num = 0
        for batch_data in test_dataloader:
            batch_data = batch_data.to(device)
            output = node_model(batch_data)
            pred = output.argmax(dim=1)
            node_num += len(batch_data.y)
            correct_pred_num += torch.sum(pred == batch_data.y)
        accurcy = correct_pred_num / node_num
        print(f'device: {device}, Test Accuracy: {accurcy}')
# 冻结模型的参数
def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
def edge_train():
    freeze_model_parameters(node_model)
    for epoch in range(args.epochs):
        edge_model.train()
        LOSS = 0
        for batch_data in tarin_dataloader:
            edge_optimizer.zero_grad()
            batch_data = batch_data.to(device)
            unique_batch_indices = torch.unique(batch_data.batch)
            for batch_index in unique_batch_indices:
                subgraph = batch_data.get_example(batch_index)
                node_embedding = node_model(subgraph, use_last_layer=False)
                # node_embedding = subgraph.x.to(device)
                output = edge_model(node_embedding, subgraph)
                loss = edge_criterion(output, subgraph.edge_y)
                LOSS += loss.item()
                loss.backward()
                edge_optimizer.step()
        print(f'device: {device}, Epoch {epoch+1}/{args.epochs}, Train Loss: {(LOSS/len(tarin_dataloader))}')
        edge_test()

def edge_test():
    freeze_model_parameters(node_model)
    edge_model.eval()
    with torch.no_grad():
        correct_pred_num = 0
        edge_num = 0
        for batch_data in test_dataloader:
            batch_data = batch_data.to(device)
            unique_batch_indices = torch.unique(batch_data.batch)
            for batch_index in unique_batch_indices:
                subgraph = batch_data.get_example(batch_index)
                node_embedding = node_model(subgraph, use_last_layer=False)
                # node_embedding = subgraph.x.to(device)
                output = edge_model(node_embedding, subgraph)
                pred = output.argmax(dim=1)
                edge_num += subgraph.edge_y.size(0)
                correct_pred_num += torch.sum(pred == subgraph.edge_y)
        accurcy = correct_pred_num / edge_num
        print(f'device: {device}, Test Accuracy: {accurcy}')

if __name__ == '__main__':
    node_train()
    # torch.save(node_model, '/home/btr/bpmn/LLMEnG/src/node_model.pth')
    # edge_train()