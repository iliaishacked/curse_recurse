import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

class Wrapper(pl.LightningModule):

    def __init__(self, model, learning_rate, epochs=200, optimizer=None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.model = model
        self.optimizer = 'adamw' if optimizer is None else optimizer
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses, self.train_perplexities = [], []
        self.val_losses, self.val_perplexities = [], []
        self.test_losses, self.test_perplexities = [], []
        self.saved = []
        self.tosave = False

    def forward(self, input_ids, attention_mask, labels):
        """
        output.last_hidden_state (batch_size, token_num, hidden_size): hidden representation for each token in each sequence of the batch. 
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # loss = self.criterion(output, labels)
        loss = output.loss
        output = output.logits
        return loss, output

    def training_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)

        self.train_losses.append(loss)
        perplexity = torch.exp(loss)
        self.train_perplexities.append(perplexity)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        # self.log("train_perp", perplexity, prog_bar=True, sync_dist=True)
        return {
            "loss": loss,
            "predictions": outputs,
            "perplexity": perplexity
        }
#
    def on_train_epoch_end(self):
        train_mean_loss = torch.mean(
            torch.tensor(self.train_losses, dtype=torch.float32))
        train_mean_perp = torch.exp(train_mean_loss)
        # torch.mean(torch.tensor(self.train_perplexities, dtype=torch.float32))
        self.train_losses = []
        self.train_perplexities = []
        self.log("train_mean_loss",
                 train_mean_loss,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        self.log("train_mean_perp",
                 train_mean_perp,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        return {
            "train_mean_loss": train_mean_loss,
            "train_mean_acc": train_mean_perp
        }

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.val_losses.append(loss)
        perplexity = torch.exp(loss)
        self.val_perplexities.append(perplexity)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        #self.log("val_perp", perplexity, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        mean_loss = torch.mean(
            torch.tensor(self.val_losses, dtype=torch.float32))
        mean_perplexity = torch.exp(mean_loss)
        # torch.mean(torch.tensor(self.val_perplexities, dtype=torch.float32))
        self.val_losses = []
        self.val_perplexities = []
        self.log("val_mean_loss",
                 mean_loss,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        self.log("val_mean_perp",
                 mean_perplexity,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        #self.log("perps",
        #         self.val_perplexities,
        #         prog_bar=True,
        #         logger=True,
        #         sync_dist=True)
        return {"val_mean_loss": mean_loss, "val_mean_perplexity": mean_perplexity, "individual_perp": self.val_perplexities}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        perplexity = torch.exp(loss)
        self.test_losses.append(loss)
        self.test_perplexities.append(perplexity)
        return loss

    def on_test_epoch_end(self):
        mean_loss = torch.mean(
            torch.tensor(self.test_losses, dtype=torch.float32))
        std_loss = torch.std(
            torch.tensor(self.test_losses, dtype=torch.float32))

        if self.tosave:
            for item in self.test_losses:
                self.saved.append(item.clone().cpu().detach())

        mean_perplexity = torch.exp(mean_loss)
        std_perplexity = torch.exp(std_loss)

        test_perps = [torch.exp(x) for x in self.test_losses]
        # mean_perplexity = torch.mean(
        #     torch.tensor(self.test_perplexities, dtype=torch.float32))
        self.test_losses = []
        self.test_perplexities = []
        self.log("test_mean_loss",
                 mean_loss,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        self.log("test_mean_perp",
                 mean_perplexity,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        ttp = torch.tensor(test_perps, dtype=torch.float32)
        mean_perps = torch.mean(ttp)
        std_perps = torch.std(ttp)
        self.log("mean_perps",
                 mean_perplexity,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        self.log("std_perps",
                 std_perplexity,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        return {"test_mean_loss": mean_loss, "test_mean_acc": mean_perplexity, "perps": test_perps}

    def configure_optimizers(self):
        if self.optimizer == 'adamw':
            opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
            scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-6)
        elif self.optimizer == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-6)
        elif self.optimizer in ['sgd_warmup', 'sgd']:
            opt = torch.optim.SGD(self.parameters(),
                                  lr=self.learning_rate,
                                  momentum=0.9,
                                  weight_decay=0.0005,
                                  nesterov=True)
            if self.optimizer == 'sgd':
                scheduler = CosineAnnealingLR(opt,
                                              T_max=self.epochs,
                                              eta_min=0.0)
        else:
            raise ValueError(f" {self.optimizer} Optimizer not supported.")
        return {"optimizer": opt, "lr_scheduler": scheduler}
