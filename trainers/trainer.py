# -*- coding: utf-8 -*-
"""
Model Trainer for MS-SSA-HGCN
Author: QinYongKang
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import copy


class ModelTrainer:
    """
    Handles training and evaluation of MS-SSA-HGCN model

    Supports:
        - Leave-One-Seizure-Out (LOOCV) cross-validation
        - Early stopping based on validation AUC
        - Focal loss and Cross-entropy loss
    """

    def __init__(self, model, loss_fn, train_loader, test_loader, config):
        """
        Initialize trainer

        Args:
            model: Neural network model
            loss_fn: Loss function
            train_loader: Training data loader
            test_loader: Validation/test data loader
            config: Training configuration object
        """
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.1, last_epoch=-1
        )

        # Track best model
        self.best_state = copy.deepcopy(model.state_dict())
        self.best_auc = 0.0
        self.best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())

        # Early stopping
        self.early_stop_counter = 0

    def _create_optimizer(self):
        """
        Create Adam optimizer with weight decay
        """
        return optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

    def train_epoch(self):
        """
        Train for one epoch
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        samples_seen = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (data, targets) in enumerate(pbar):
            # Move to GPU if available
            if self.config.use_cuda:
                data = data.cuda()
                targets = targets.cuda()

            data = Variable(data)
            targets = Variable(targets)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.loss_fn(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate metrics
            epoch_loss += loss.data.item()

            # Extract predictions
            predictions = outputs[0] if isinstance(outputs, tuple) else outputs
            pred_labels = predictions.detach().cpu().numpy().argmax(axis=1)
            true_labels = targets.view(-1).detach().cpu().numpy()

            batch_size = data.shape[0]
            epoch_acc += (pred_labels == true_labels).sum()
            samples_seen += batch_size

            pbar.set_postfix({
                "loss": f"{epoch_loss / max(samples_seen, 1):.4f}",
                "acc": f"{epoch_acc / max(samples_seen, 1):.4f}"
            })

        return epoch_loss / len(self.train_loader.dataset), epoch_acc / len(self.train_loader.dataset)

    def evaluate(self):
        """
        Evaluate model on test set
        """
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluating", leave=False)
            for data, targets in pbar:
                if self.config.use_cuda:
                    data = data.cuda()
                    targets = targets.cuda()

                # Forward pass
                outputs = self.model(data)
                loss = self.loss_fn(outputs, targets)
                test_loss += loss.data.item()

                # Extract predictions
                predictions = outputs[0] if isinstance(outputs, tuple) else outputs
                pred_labels = predictions.detach().cpu().numpy().argmax(axis=1)
                true_labels = targets.view(-1).detach().cpu().numpy()

                test_acc += (pred_labels == true_labels).sum()

                # Store for AUC calculation
                probs = nn.functional.softmax(predictions, dim=1).detach().cpu().numpy()[:, 1]
                all_labels.extend(true_labels.tolist())
                all_probs.extend(probs.tolist())

        # Calculate AUC
        test_auc = roc_auc_score(all_labels, all_probs) if all_labels else 0.0

        return (
            test_acc / len(self.test_loader.dataset),
            test_loss / len(self.test_loader.dataset),
            test_auc
        )

    def train(self):
        """
        Complete training loop with early stopping

        Returns:
            model: Trained model
            train_losses: List of training losses per epoch
            train_accs: List of training accuracies per epoch
            val_aucs: List of validation AUCs per epoch
        """
        train_losses = []
        train_accs = []
        val_aucs = []

        for epoch in range(self.config.num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch()
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation
            if self.config.validate_during_training:
                test_acc, test_loss, test_auc = self.evaluate()
                val_aucs.append(test_auc)

                # Update best model
                if test_auc > self.best_auc:
                    self.best_auc = test_auc
                    self.best_state = copy.deepcopy(self.model.state_dict())
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1

                # Print progress
                print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"  Val   - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
                print(f"  Early stop count: {self.early_stop_counter}")

                # Early stopping check
                if self.early_stop_counter >= self.config.early_stop_patience:
                    print("Early stopping triggered based on validation AUC")
                    break
            else:
                print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

            # Update learning rate
            self.scheduler.step()

        # Restore best model
        self.model.load_state_dict(self.best_state)

        return self.model, train_losses, train_accs, val_aucs

    def test(self):
        """
        Final test after training
        """
        test_acc, test_loss, test_auc = self.evaluate()
        print(f"\nFinal Test Results:")
        print(f"  Accuracy: {test_acc:.4%}")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        return test_acc, test_loss, test_auc
