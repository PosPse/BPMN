from data_utils import DataCenter
from model import GCN, GraphSage, EdgeClassification, EdgeFusion, GAT
from get_embs import Tokenizer
from focal_loss import FocalLossWithBinaryCrossEntropy, FocalLossWithCrossEntropy
import torch
import Parser as Parser
import math


args = Parser.args
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
tokenizer = Tokenizer(llm_model=args.llm_model, device=device)
data_center = DataCenter(datasets_json=args.datasets_json, tokenizer=tokenizer, method=args.method, add_scale=args.add_scale, del_scale=args.del_scale)

# node_model = GCN(hidden_size=args.hidden_size, node_num_classes=args.node_num_classes).to(device)
node_model = GraphSage(hidden_size=args.hidden_size, node_num_classes=args.node_num_classes, aggr=args.aggr).to(device)
# node_model = GAT(hidden_size=args.hidden_size, node_num_classes=args.node_num_classes).to(device)
node_optimizer = torch.optim.SGD(node_model.parameters(), lr=args.lr)
node_criterion = torch.nn.CrossEntropyLoss().to(device)

# node_model = torch.load('/home/btr/bpmn/LLMEnG/src/node_model.pth', map_location=device)
# node_model.eval()
edge_fusion_model = EdgeFusion(hidden_size=args.hidden_size, fusion_method=args.fusion_method).to(device)
edge_model = EdgeClassification(device=device,hidden_size=args.hidden_size, edge_fusion = edge_fusion_model).to(device)
edge_optimizer = torch.optim.SGD(edge_model.parameters(), lr=args.lr)
# edge_scheduler = torch.optim.lr_scheduler.StepLR(edge_optimizer, step_size=50, gamma=0.98)
weight = [11 for _ in range(11)]
weight[0] = 1
weight = torch.tensor(weight, dtype=torch.float32).to(device)
# edge_criterion = torch.nn.CrossEntropyLoss(weight=weight).to(device)
# edge_criterion = FocalLossWithBinaryCrossEntropy(device=device, alpha=weight, gamma=2, reduction='mean')
edge_criterion = FocalLossWithCrossEntropy(device=device, alpha=weight, gamma=2, reduction='mean')


def node_train_and_test():
    for epoch in range(args.epochs):
        data_center.random_split()
        tarin_dataloader = data_center.get_train_dataloader(args.batch_size, args.shuffle)
        test_dataloader = data_center.get_test_dataloader(args.batch_size, args.shuffle)
        node_train(tarin_dataloader, epoch)
        node_test(test_dataloader)
def node_train(tarin_dataloader, epoch):
    node_model.train()
    LOSS = 0
    for batch_data in tarin_dataloader:
        node_optimizer.zero_grad()
        batch_data = batch_data.to(device)
        _, output = node_model(batch_data)
        loss = node_criterion(output, batch_data.y)
        LOSS += loss.item()
        loss.backward()
        node_optimizer.step()
    print(f'device: {device}, Epoch {epoch+1}/{args.epochs}, Train Loss: {(LOSS/len(tarin_dataloader))}')

def node_test(test_dataloader):
    node_model.eval()
    with torch.no_grad():
        correct_pred_num = 0
        node_num = 0
        for batch_data in test_dataloader:
            batch_data = batch_data.to(device)
            _, output = node_model(batch_data)
            pred = output.argmax(dim=1)
            node_num += len(batch_data.y)
            correct_pred_num += torch.sum(pred == batch_data.y)
        accurcy = correct_pred_num / node_num
        print(f'device: {device}, Test Accuracy: {accurcy}')
# 冻结模型的参数
def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def edge_train_and_test():
    freeze_model_parameters(node_model)
    for epoch in range(args.epochs):
        data_center.random_split()
        tarin_dataloader = data_center.get_train_dataloader(args.batch_size, args.shuffle)
        test_dataloader = data_center.get_test_dataloader(args.batch_size, args.shuffle)
        edge_train(tarin_dataloader, epoch)
        edge_test(test_dataloader)

def edge_train(tarin_dataloader, epoch):
    edge_model.train()
    LOSS = 0
    for batch_data in tarin_dataloader:
        edge_optimizer.zero_grad()
        batch_data = batch_data.to(device)
        unique_batch_indices = torch.unique(batch_data.batch)
        for batch_index in unique_batch_indices:
            subgraph = batch_data.get_example(batch_index)
            node_embedding, node_output = node_model(subgraph)
            # node_embedding = subgraph.x.to(device)
            node_pred = node_output.argmax(dim=1)
            node_type_pred_emb = tokenizer.node_type_emb[node_pred]
            edge_output = edge_model(node_embedding, subgraph, node_type_pred_emb)
            # edge_y = subgraph.raw_data.edge_y.to_dense().view(-1).to(device)
            edge_y = torch.tensor(subgraph.edge_y.toarray(), dtype=torch.long).view(-1).to(device)
            loss = edge_criterion(edge_output, edge_y)
            LOSS += loss.item()
            loss.backward()
            edge_optimizer.step()
            # edge_scheduler.step()
    print(f'device: {device}, Epoch {epoch+1}/{args.epochs}, Train Loss: {(LOSS/len(tarin_dataloader))}')

def edge_test(test_dataloader):
    edge_model.eval()
    with torch.no_grad():
        edge_correct_pred_num = 0
        positive_edge_correct_pred_num = 0
        negative_edge_correct_pred_num = 0
        edge_num = 0
        positive_edge_num = 0
        negative_edge_num = 0
        for batch_data in test_dataloader:
            batch_data = batch_data.to(device)
            unique_batch_indices = torch.unique(batch_data.batch)
            for batch_index in unique_batch_indices:
                subgraph = batch_data.get_example(batch_index)
                edge_y = torch.tensor(subgraph.edge_y.toarray(), dtype=torch.long).view(-1).to(device)

                node_embedding, node_output = node_model(subgraph)
                node_pred = node_output.argmax(dim=1)
                node_type_pred_emb = tokenizer.node_type_emb[node_pred]

                edge_output = edge_model(node_embedding, subgraph, node_type_pred_emb)
                edge_pred = edge_output.argmax(dim=1)
                edge_num += len(edge_y)
                edge_correct_pred_num += torch.sum(edge_pred == edge_y)

                non_zero_indices = edge_y.nonzero().view(-1)
                positive_edge_num += len(non_zero_indices)
                positive_edge_correct_pred_num += torch.sum(edge_pred[non_zero_indices] == edge_y[non_zero_indices])

                zero_indices = (edge_y == 0).nonzero().view(-1)
                negative_edge_num += len(zero_indices)
                negative_edge_correct_pred_num += torch.sum(edge_pred[zero_indices] == edge_y[zero_indices])

        edge_accurcy = edge_correct_pred_num / edge_num
        positive_edge_accurcy = positive_edge_correct_pred_num / positive_edge_num
        negative_edge_accurcy = negative_edge_correct_pred_num / negative_edge_num
        print(f'device: {device}, edge classification accuracy: {edge_accurcy}, positive edge classification accuracy: {positive_edge_accurcy}, negative edge classification accuracy: {negative_edge_accurcy}')

if __name__ == '__main__':
    node_train_and_test()
    # torch.save(node_model, '/home/btr/bpmn/LLMEnG/src/node_model.pth')
    edge_train_and_test()
    # ModelConfig.set_current_model("bert-large-uncased")
    # pass