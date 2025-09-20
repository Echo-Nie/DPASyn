import numpy as np
from torch_geometric.data import Data, Batch
import torch
from torch.utils.data import Dataset
import pickle


class DemoDataset(Dataset):
    def __init__(self, data_path=None, data=None):
        if data is None:
            if data_path is None:
                data_path = '../data/data.pt'
            try:
                with open(data_path, 'rb') as f:
                    self.data = pickle.load(f)
            except FileNotFoundError:
                print(f"Data file {data_path} not found, creating mock data...")
                self.data = self._create_demo_data()
        else:
            self.data = data

    def _create_demo_data(self):
        demo_data = []
        for i in range(100):
            num_nodes = np.random.randint(5, 15)
            
            x1 = torch.randn(num_nodes, 64)
            x2 = torch.randn(num_nodes, 64)
            
            edge_index1 = torch.randint(0, num_nodes, (2, num_nodes * 2))
            edge_index2 = torch.randint(0, num_nodes, (2, num_nodes * 2))
            
            graph1 = Data(x=x1, edge_index=edge_index1)
            graph2 = Data(x=x2, edge_index=edge_index2)
            
            label = torch.randint(0, 2, (1,)).item()
            
            demo_data.append((graph1, graph2, label))
        
        return demo_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def get_subset(self, indices):
        subset_data = [self.data[i] for i in indices]
        return DemoDataset(data=subset_data)


def demo_collate(batch):
    graphs1, graphs2, labels = [], [], []
    
    for item in batch:
        if len(item) == 4:
            # Handle format: (graph1, graph2, cell, label)
            graph1, graph2, cell, label = item
            # Attach cell feature to graph1 if needed
            if hasattr(graph1, 'cell'):
                graph1.cell = cell
            else:
                graph1.cell = cell
        elif len(item) == 3:
            # Handle format: (graph1, graph2, label)
            graph1, graph2, label = item
        else:
            print(f"Unexpected data format with {len(item)} elements: {item}")
            continue
            
        graphs1.append(graph1)
        graphs2.append(graph2)
        labels.append(label)

    return Batch.from_data_list(graphs1), Batch.from_data_list(graphs2), torch.tensor(labels)


def save_demo_metrics(metrics, filename):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(','.join(map(str, metrics)) + '\n')


def create_demo_graph(num_nodes=10, feature_dim=64):
    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    
    return Data(x=x, edge_index=edge_index)


if __name__ == '__main__':
    print("Testing Demo Dataset...")
    
    demo_dataset = DemoDataset()
    print(f"Dataset size: {len(demo_dataset)}")
    
    sample = demo_dataset[0]
    print(f"Sample type: {type(sample)}")
    print(f"Sample content: {len(sample)} elements")
    print(f"Sample structure: {[type(item) for item in sample]}")
    
    # Test with first few samples
    batch_data = [demo_dataset[i] for i in range(3)]
    try:
        batched_graphs1, batched_graphs2, labels = demo_collate(batch_data)
        print(f"Batched graphs1: {batched_graphs1}")
        print(f"Batched graphs2: {batched_graphs2}")
        print(f"Labels: {labels}")
        print("Demo dataset test completed!")
    except Exception as e:
        print(f"Error in collate function: {e}")
        print("First sample details:")
        for i, item in enumerate(batch_data[0]):
            print(f"  Item {i}: {type(item)} - {item}")
