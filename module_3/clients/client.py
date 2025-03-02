import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class Client:
    
    def __init__(self, seed, client_id, lr, weight_decay, batch_size, train_data, eval_data, model, device=None, num_workers=0, run=None):
        self.model = model
        self.id = client_id
        self.train_data = train_data
        self.eval_data = eval_data
        self.trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers) if self.train_data.__len__() != 0 else None
        self.testloader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers) if self.eval_data.__len__() != 0 else None
        self.seed = seed
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.run = run
        
    def train(self, rounds, writer, num_epochs=1, batch_size=10, patience=5):
        
        criterion = nn.BCELoss().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        losses = np.empty(num_epochs)
        val_losses = np.empty(num_epochs)
        best_auc = 0.5
        early_counter = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            losses[epoch] = self.run_epoch(optimizer, criterion)
            writer.add_scalar(f"round {rounds}, site_{self.id} training_loss", losses[epoch], epoch)
            val_losses[epoch], auc = self.evaluate(criterion)
            writer.add_scalar(f"round {rounds}, site_{self.id} validation_loss", val_losses[epoch], epoch)
            update = self.model.state_dict()
            if patience:
                if auc - best_auc > 0.01:
                    best_auc = auc
                    early_counter = 0
                    update = self.model.state_dict()
                else:
                    early_counter += 1
                
                if early_counter == patience:
                    break
            else:
                update = self.model.state_dict()
                if auc - best_auc > 0.01:
                    best_auc = auc
                    early_counter = 0
                    
        writer.add_scalar(f"site_{self.id} best_auc", best_auc, rounds)
        writer.add_scalar(f"site_{self.id} val_loss", np.average(val_losses), rounds)
            
        self.losses = losses
        self.val_losses = val_losses
        
        writer.flush()
        return self.num_train_samples, update, best_auc, self.val_losses
    
    def run_epoch(self, optimizer, criterion):
        
        running_loss = 0
        i = 0
        for j, data in enumerate(tqdm(self.trainloader)):
            input_data_tensor, target_data_tensor = data["image"].to(self.device), data["labels"].to(self.device)
            optimizer.zero_grad()
            outputs = self.model(input_data_tensor)
            loss = criterion(outputs.float(), target_data_tensor.float())
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            i += 1
        if i == 0:
            print("Not running epoch", self.id)
            return 0
        return running_loss / i
    
    def evaluate(self, criterion):
        with torch.no_grad():
            running_loss = 0
            i = 0
            y_prob = []
            y_true = []
            for j, data in enumerate(tqdm(self.testloader)):
                input_data_tensor, target_data_tensor = data["image"].to(self.device), data["labels"].to(self.device)
                outputs = self.model(input_data_tensor)
                loss = criterion(outputs.float(), target_data_tensor.float())
                running_loss += loss.item()
                i += 1
                y_true += [t for t in target_data_tensor.tolist()]
                y_prob += [p for p in outputs.tolist()]
            if i == 0:
                print("Not validating", self.id)
                return 0
        return running_loss / i, roc_auc_score(y_true, y_prob)
    
    @property
    def num_test_samples(self):
        return self.eval_data.__len__()
    
    @property
    def num_train_samples(self):
        return self.train_data.__len__()
    
    @property
    def num_samples(self):
        return self.num_train_samples + self.num_test_samples
    
    def number_of_samples_per_class(self):
        if self.train_data.__len__() > 0:
            loader = self.trainloader
        else:
            loader = self.testloader
        samples_per_class = {}
        for data in loader:
            labels = data[1].tolist()
            for l in labels:
                if l in samples_per_class:
                    samples_per_class[l] += 1
                else:
                    samples_per_class[l] = 1
        return samples_per_class