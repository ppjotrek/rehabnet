import torch
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
from utils import emg_dataset, train
from model import SERes1d
from sklearn import metrics
import json
import logging
import datetime
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_path = "C:\\Users\\piotr\\Python\\rehabnet\\main\\config.json"

if __name__ == '__main__':
    
    config = json.load(open(config_path))
    root_dir = config['root_dir']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    state_dicts_folder = config['state_dicts_folder']
    logs_folder = config['logs_folder']
    num_classes = config['num_classes']
    num_channels = config['num_channels']
    num_features = config['num_features']
    training_path = config['training_path']
    eval_path = config['eval_path']
    
    try:
        os.makedirs(state_dicts_folder)
    except FileExistsError:
        pass
    try:
        os.makedirs(logs_folder)
    except FileExistsError:
        pass
    
    timestamp = datetime.datetime.now()
    logging_filename = str(timestamp.strftime("%Y")) + str(timestamp.strftime("%b")) + str(timestamp.strftime("%d")) + '_' + str(timestamp.strftime("%H")) + str(timestamp.strftime("%M")) + ".log"
    logging_path = logs_folder + logging_filename
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=logging_path, level=logging.INFO)
    logger.info("Config file loaded")
    
    training_dataset = emg_dataset(training_path)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=16, shuffle=True)

    eval_dataset = emg_dataset(eval_path)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=16, shuffle=True)
    class_to_index_map = config['class_to_id']
    logger.info("Data loaded")
    logger.info("Training dataset size: " + str(len(training_dataset)))
    logger.info("Eval dataset size: " + str(len(eval_dataset)))
    logger.info("Class to index map: " + str(class_to_index_map))
    
    model = SERes1d(num_channels, num_classes, num_features).to(device)
    criteria = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    logger.info("Model created")
    logger.info("Model summary: ")
    logger.info(summary(model, (num_channels, num_features)))
    
    acc_history = train(model, training_dataloader, criteria, optimizer, num_epochs, device)
    
    plt.plot(acc_history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('ResNet 1D' + num_features + ' features')
    plt.savefig(logs_folder + logging_filename + "_accuracy_plot.png")

    torch.save(model.state_dict(), state_dicts_folder + "model_state_dict_" + num_features + "_features.pth")
    torch.save(optimizer.state_dict(), state_dicts_folder + "optimizer_state_dict_" + num_features + "_features.pth")