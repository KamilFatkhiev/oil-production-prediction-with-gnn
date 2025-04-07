import matplotlib.pyplot as plt
import torch
import torch_geometric.transforms as T
from torch_geometric_temporal.signal import DynamicHeteroGraphTemporalSignal

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_sequences_new_idea(dataset, history_length, prediction_length):
    sequences = []
    targets = []
    target_sequences = []

    # Iterate over the dataset with a sliding window
    for i in range(len(dataset.feature_dicts) - history_length - prediction_length + 1):
        # Extract historical data (30 days)
        historical_data = dataset[i:i + history_length]

        # Extract target data (next 30 days)
        target_data = dataset[i + prediction_length:i + history_length + prediction_length]

        # # Extract target values from the target data
        t_v = [snapshot.y_dict for snapshot in target_data]
        target_values = torch.cat([t["prod"] for t in t_v], dim=1)
        
        # # Append to sequences and targets
        sequences.append(historical_data)
        targets.append(target_values)
        target_sequences.append(target_data)

    return sequences, target_sequences, targets


def plot_loss(history, history_test):
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Training Loss')
    plt.plot(history_test, label='Test Loss')
    plt.title('Loss function over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def train_test_split(sequences, targets, target_sequences=None, train_ratio=0.8):
    # Calculate the split index
    split_idx = int(len(sequences) * train_ratio)

    # Split sequences
    train_sequences = sequences[:split_idx]
    test_sequences = sequences[split_idx:]

    # Split targets
    train_targets = targets[:split_idx]
    test_targets = targets[split_idx:]

    if target_sequences is not None:
        train_target_sequences = target_sequences[:split_idx]
        test_target_sequences = target_sequences[split_idx:]
        return train_sequences, test_sequences, train_targets, test_targets, train_target_sequences, test_target_sequences

    return train_sequences, test_sequences, train_targets, test_targets


def predict_future_with_water(model, sequence, target_seq, steps=30):
    """
    Рекуррентное предсказание на `steps` шагов вперед для гетерогенной модели.
    
    :param model: Обученная модель HeteroGCLSTMSeq2Seq_new_GAT_one_step.
    :param initial_history: Начальная история (список словарей x_dict_seq).
    :param edge_index_dict_seq: Последовательность словарей edge_index_dict.
    :param feature: Дополнительный признак (список словарей x_dict_seq_target).
    :param steps: Количество шагов предсказания.
    :return: Предсказанные значения (список тензоров).
    """
    model.eval()
    predictions = []
    
    sequence = [T.ToUndirected()(snapshot).to(device) for snapshot in sequence]
    target_seq = [snapshot.to(device) for snapshot in target_seq]
    
    x_dict_sequence = [snapshot.x_dict for snapshot in sequence]
    edge_index_dict_sequence = [snapshot.edge_index_dict for snapshot in sequence]
    x_dict_sequence_target = [snapshot.x_dict for snapshot in target_seq]

    with torch.no_grad():
        for step in range(steps):
            # Предсказание на один шаг вперед
            next_step = model(x_dict_sequence, edge_index_dict_sequence, x_dict_sequence_target, 1)
            predictions.append(next_step)
            
            # Сдвигаем историю на один шаг вперед
            x_dict_sequence_target = x_dict_sequence_target[1:]
            if step < (steps - 1):
                tmp = {"prod": next_step, "inj": x_dict_sequence_target[0]["inj"]}
                x_dict_sequence = x_dict_sequence[1:] + [tmp]
                edge_index_dict_sequence = edge_index_dict_sequence[1:] + [edge_index_dict_sequence[-1]]
            
    return torch.stack(predictions, dim=1).squeeze()


def predict_future_no_water(model, sequence, target_seq, steps=30):
    """
    Рекуррентное предсказание на `steps` шагов вперед для гетерогенной модели.
    
    :param model: Обученная модель HeteroGCLSTMSeq2Seq_new_GAT_one_step.
    :param initial_history: Начальная история (список словарей x_dict_seq).
    :param edge_index_dict_seq: Последовательность словарей edge_index_dict.
    :param feature: Дополнительный признак (список словарей x_dict_seq_target).
    :param steps: Количество шагов предсказания.
    :return: Предсказанные значения (список тензоров).
    """
    model.eval()
    predictions = []
    
    sequence = [T.ToUndirected()(snapshot).to(device) for snapshot in sequence]
    target_seq = [snapshot.to(device) for snapshot in target_seq]
    
    x_dict_sequence = [snapshot.x_dict for snapshot in sequence]
    edge_index_dict_sequence = [snapshot.edge_index_dict for snapshot in sequence]

    with torch.no_grad():
        for step in range(steps):
            # Предсказание на один шаг вперед
            next_step, next_water = model(x_dict_sequence, edge_index_dict_sequence, 1)
            predictions.append(next_step)
            
            # Сдвигаем историю на один шаг вперед
            if step < (steps - 1):
                tmp = {"prod": next_step, "inj": next_water}
                x_dict_sequence = x_dict_sequence[1:] + [tmp]
                edge_index_dict_sequence = edge_index_dict_sequence[1:] + [edge_index_dict_sequence[-1]]
            
    return torch.stack(predictions, dim=1).squeeze()