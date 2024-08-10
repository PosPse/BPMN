from data_utils import DataCenter
from model import GCN, GraphSage, EdgeClassification, EdgeFusion, GAT
from get_embs import Tokenizer
import torch
import Parser as Parser
from scipy.linalg import block_diag

args = Parser.args
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
tokenizer = Tokenizer(llm_model=args.llm_model, device=device)
data_center = DataCenter(datasets_json=args.datasets_json, tokenizer=tokenizer)
tarin_dataloader = data_center.get_train_dataloader(args.batch_size, args.shuffle)
test_dataloader = data_center.get_test_dataloader(args.batch_size, args.shuffle)
# node_model = GCN(hidden_size=args.hidden_size, node_num_classes=args.node_num_classes).to(device)
node_model = GraphSage(hidden_size=args.hidden_size, node_num_classes=args.node_num_classes, aggr=args.aggr).to(device)
# node_model = GAT(hidden_size=args.hidden_size, node_num_classes=args.node_num_classes).to(device)
node_optimizer = torch.optim.SGD(node_model.parameters(), lr=args.lr)
node_criterion = torch.nn.CrossEntropyLoss().to(device)

edge_fusion_model = EdgeFusion(hidden_size=args.hidden_size, fusion_method=args.fusion_method).to(device)
edge_model = EdgeClassification(hidden_size=args.hidden_size, edge_fusion = edge_fusion_model).to(device)
edge_optimizer = torch.optim.SGD(edge_model.parameters(), lr=args.lr)
weight = [5 for _ in range(21)]
weight[0] = 1
weight = torch.tensor(weight, dtype=torch.float32)
edge_criterion = torch.nn.CrossEntropyLoss(weight=weight).to(device)
alpha = 0.5
def train():
    for epoch in range(args.epochs):
        node_model.train()
        edge_model.train()
        LOSS = 0
        for batch_data in tarin_dataloader:
            node_optimizer.zero_grad()
            edge_optimizer.zero_grad()
            batch_data = batch_data.to(device)
            # 获取批量数据里的edge_y, 从稀疏矩阵中恢复原邻接矩阵, 组合为对角矩阵
            edge_y = block_diag(*[v.toarray() for v in batch_data.edge_y])
            # 转换为tensor, 并展平
            edge_y = torch.tensor(edge_y, dtype=torch.long).view(-1).to(device)
            node_embedding, node_output = node_model(batch_data)
            node_loss = node_criterion(node_output, batch_data.y)
            edge_output = edge_model(node_embedding, batch_data)
            # print(edge_output.shape, edge_y.shape)
            edge_loss = edge_criterion(edge_output, edge_y)
            loss = alpha * node_loss + (1-alpha)*edge_loss
            LOSS += loss.item()
            loss.backward()
            node_optimizer.step()
            edge_optimizer.step()
        print(f'device: {device}, Epoch {epoch+1}/{args.epochs}, Train Loss: {(LOSS/len(tarin_dataloader))}')
        test()

def test():
    node_model.eval()
    edge_model.eval()
    with torch.no_grad():
        node_correct_pred_num = 0
        edge_correct_pred_num = 0
        positive_edge_correct_pred_num = 0
        negative_edge_correct_pred_num = 0
        node_num = 0
        edge_num = 0
        positive_edge_num = 0
        negative_edge_num = 0
        for batch_data in tarin_dataloader:
            batch_data = batch_data.to(device)
            # 获取批量数据里的edge_y, 从稀疏矩阵中恢复原邻接矩阵, 组合为对角矩阵
            edge_y = block_diag(*[v.toarray() for v in batch_data.edge_y])
            # 转换为tensor, 并展平
            edge_y = torch.tensor(edge_y, dtype=torch.long).view(-1).to(device)
            
            node_embedding, node_output = node_model(batch_data)
            node_pred = node_output.argmax(dim=1)
            node_num += len(batch_data.y)
            node_correct_pred_num += torch.sum(node_pred == batch_data.y)
            
            edge_output = edge_model(node_embedding, batch_data)
            edge_pred = edge_output.argmax(dim=1)
            edge_num += len(edge_y)
            edge_correct_pred_num += torch.sum(edge_pred == edge_y)
           
            non_zero_indices = edge_y.nonzero().view(-1)
            positive_edge_num += len(non_zero_indices)
            positive_edge_correct_pred_num += torch.sum(edge_pred[non_zero_indices] == edge_y[non_zero_indices])

            zero_indices = (edge_y == 0).nonzero().view(-1)
            negative_edge_num += len(zero_indices)
            negative_edge_correct_pred_num += torch.sum(edge_pred[zero_indices] == edge_y[zero_indices])
        node_accurcy = node_correct_pred_num / node_num
        edge_accurcy = edge_correct_pred_num / edge_num
        positive_edge_accurcy = positive_edge_correct_pred_num / positive_edge_num
        negative_edge_accurcy = negative_edge_correct_pred_num / negative_edge_num
        print(f'device: {device}, node classification accuracy: {node_accurcy}, edge classification accuracy: {edge_accurcy}, positive edge classification accuracy: {positive_edge_accurcy}, negative edge classification accuracy: {negative_edge_accurcy}')

if __name__ == '__main__':
    train()
