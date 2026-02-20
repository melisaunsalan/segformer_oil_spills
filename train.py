import os
import numpy as np
import torch
import evaluate
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import yaml

from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torch import nn
from tqdm import tqdm
from datasets import disable_caching
from torch.utils.tensorboard import SummaryWriter

from data.oil_spill_dataset import OilSpillDataset

from utils.dice_loss import dice_loss, FocalLoss

class SegFormerTrainer:
    """
    A class to encapsulate the training process of SegFormer model for oil spill segmentation.
    """

    def __init__(self, config_path: str) -> None:
        super().__init__()
        config = self.load_config(config_path)
        # create the current experiment's path
        weights_path = self.__create_experiment_folder(
            experiment_path=config["EXPERIMENT"]["exp_path"],
            experiment_name=config["EXPERIMENT"]["name"],
        )
        # Init TensorBoard logger
        self.writer = self.__create_logger(weights_path)
        # Create the image processor
        image_processor = SegformerImageProcessor(do_rescale=False, do_normalize=False)
        if image_processor is None:
            raise ValueError("Image processor initialization failed.")

        # initialize datasets and dataloaders
        train_dataset = OilSpillDataset(
            config["DATASET_PARAMS"]["db_path"],
            split="train",
            image_processor=image_processor,
        )
        valid_dataset = OilSpillDataset(
            config["DATASET_PARAMS"]["db_path"],
            split="test",
            image_processor=image_processor,
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["TRAIN_PARAMS"]["batch_size"],
            shuffle=True,
            num_workers=config["TRAIN_PARAMS"]["num_workers"],
        )
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

        # define the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_checkpoint = config["MODEL_PARAMS"]["checkpoint"]
        model_config = config["MODEL_PARAMS"]["model_config"]
        self.model = self.__define_model(
            model_checkpoint=model_checkpoint, model_config=model_config
        )

        # Dump the cfg file under the experiment path for better traceability
        cfg_save_path = os.path.join(config["EXPERIMENT"]["exp_path"], "config.yaml")
        with open(cfg_save_path, 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

        # define the optimizer
        self.optimizer = self.__define_optimizer(
            learning_rate=config["TRAIN_PARAMS"]["lr"]
        )

        # other initializations can go here
        self.epochs = config["TRAIN_PARAMS"]["num_epochs"]
        self.weights_path = weights_path

        # Initialize metric
        self.metric = evaluate.load("mean_iou")

        # Initialize mixed precision training
        self.scaler = GradScaler()
        # Initialize cosine annealing scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=0)

        self.loss_fn = config["TRAIN_PARAMS"]["loss_fn"]

    def __del__(self):
        """Destructor to close the TensorBoard writer."""
        print("Cleaning up trainer resources.")
        if hasattr(self, "writer"):
            self.writer.close()

        del self.model
        del self.train_dataloader
        del self.valid_dataloader

        self.model = None
        self.train_dataloader = None
        self.valid_dataloader = None

        print("Trainer resources cleaned.")

    def load_config(self, config_path: str) -> dict:
        """Load training configuration from a YAML file.

        Raises:
            e: _if the YAML file cannot be loaded.

        Returns:
            dict: configuration parameters.
        """
        cfg = None
        with open(config_path) as file:
            try:
                cfg = yaml.safe_load(file)
            except yaml.YAMLError as e:
                print(f"Error loading configuration file: {e}")
                raise e
        return cfg

    def __create_experiment_folder(
        self, experiment_path: str, experiment_name: str
    ) -> str:
        """Create directory for the current experiment.

        Args:
            experiment_path (str): path where experiments are stored.
            experiment_name (str): path to where the weights will be saved.

        Raises:
            e: cannot create the experiment directory.

        Returns:
            str: path to where the weights will be saved.
        """
        weights_path = None
        weights_path = os.path.join(experiment_path, experiment_name)
        if not os.path.isdir(weights_path):
            try:
                os.makedirs(weights_path, exist_ok=True)
            except Exception as e:
                print(f"Error creating experiment directory: {e}")
                raise e

        return weights_path

    def __create_logger(self, weights_path: str) -> SummaryWriter:
        """Create TensorBoard logger.

        Args:
            weights_path (str): path to the experiment weights directory.

        Returns:
            SummaryWriter: logging object for TensorBoard.
        """
        tb_logger_path = None
        tb_logger_path = os.path.join(weights_path, "tb_logger")
        try:
            os.makedirs(tb_logger_path, exist_ok=True)
        except Exception as e:
            print(f"Error creating TensorBoard logger directory: {e}")
            raise e

        writer = SummaryWriter(log_dir=tb_logger_path)
        return writer

    def __define_model(
        self, model_checkpoint: str, model_config: str
    ) -> torch.nn.Module:
        """Define segformer model.

        Args:
            model_checkpoint (str): path to preexisting checkpoint.
            model_config (str): model configuration.

        Returns:
            torch.nn.Module: model architecture object
        """
        # Label and id
        id2label = {
            "0": "sea",
            "1": "oil_spill",
            "2": "look_alike",
            "3": "ship",
            "4": "land",
        }
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}

        # Define the model
        if model_checkpoint and os.path.isfile(model_checkpoint):
            # if resuming from checkpoint
            print(f"Loading model from checkpoint: {model_checkpoint}")
            model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-" + model_config,
                num_labels=5,
                id2label=id2label,
                label2id=label2id,
            )
            model.load_state_dict(
                torch.load(model_checkpoint, map_location=self.device), strict=True
            )
        else:
            # if training from scratch
            print("No valid checkpoint provided, loading pretrained model.")
            model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-" + model_config,
                num_labels=5,
                id2label=id2label,
                label2id=label2id,
            )
        # Move model to device
        model.to(self.device)

        return model

    def __define_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        """Define the optimizer.

        Args:
            optimizer_params (float): learning rate.

        Returns:
            torch.optim.Optimizer: optimizer object.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        return optimizer

    def __train_step(self, batch: dict) -> tuple:
        """Perform a single training step with mixed precision training.

        Args:
            batch (dict): batch of input data.

        Returns:
            tuple: loss and logits from the model.
        """
        pixel_values = batch["pixel_values"].to(self.device)
        labels = batch["original_labels"].to(self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize with mixed precision
        with torch.autocast(device_type="cuda"):
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            if self.loss_fn  == 'dice':
                logits = outputs.logits
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                predicted = upsampled_logits.argmax(dim=1)
                loss = dice_loss(predicted, labels)
            elif self.loss_fn == 'focal':
                criterion = FocalLoss(gamma=2.0, ignore_index=255)
                logits = outputs.logits
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                predicted = upsampled_logits.argmax(dim=1)
                loss = criterion(predicted, labels)
            else:
                loss = outputs.loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss

    def __train_epoch(self, epoch: int) -> None:
        """Train in single epoch

        Args:
            epoch (int): one pass through the training dataset
        """
        self.model.train()
        total_loss = 0

        for idx, batch in tqdm(enumerate(self.train_dataloader), desc="Training epoch"):
            loss = self.__train_step(batch.to(self.device))
            # Log training loss
            self.writer.add_scalar(
                "Train/Loss", loss.item(), epoch * len(self.train_dataloader) + idx
            )
            total_loss += loss.item()

        # Compute and log average training loss
        avg_loss = total_loss / len(self.train_dataloader)
        self.writer.add_scalar("Train/Avg_Loss", avg_loss, epoch)
        print(f"Epoch {epoch} - Avg Train Loss: {avg_loss:.4f}")

    def __validate(self, epoch: int) -> None:
        """Validate model on validation dataset and log metrics.

        Args:
            epoch (int): current epoch number
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.valid_dataloader, desc="Validating"):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["original_labels"].to(self.device)

                outputs = self.model(pixel_values=pixel_values, labels=labels)
                if self.loss_fn  == 'dice':
                    logits = outputs.logits
                    upsampled_logits = nn.functional.interpolate(
                        logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                    )
                    predicted = upsampled_logits.argmax(dim=1)
                    loss = dice_loss(predicted, labels)
                elif self.loss_fn == 'focal':
                    criterion = FocalLoss(gamma=2.0, ignore_index=255)
                    logits = outputs.logits
                    upsampled_logits = nn.functional.interpolate(
                        logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                    )
                    predicted = upsampled_logits.argmax(dim=1)
                    loss = criterion(predicted, labels)
                else:
                    loss = outputs.loss
                    logits = outputs.logits

                total_loss += loss.item()

                # Compute metrics
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                predicted_labels = upsampled_logits.argmax(dim=1)

                # Add batch to metric
                self.metric.add_batch(
                    predictions=predicted_labels.detach().cpu().numpy(),
                    references=labels.detach().cpu().numpy(),
                )

        # Compute and log metrics
        avg_loss = total_loss / len(self.valid_dataloader)
        metrics = self.metric.compute(
            num_labels=5,
            ignore_index=255,
            reduce_labels=False,
        )

        self.writer.add_scalar("Validation/Loss", avg_loss, epoch)
        self.writer.add_scalar("Validation/Mean_IoU", metrics["mean_iou"], epoch)
        self.writer.add_scalar(
            "Validation/Mean_Accuracy", metrics["mean_accuracy"], epoch
        )

        # Save checkpoint
        checkpoint_path = os.path.join(self.weights_path, f"checkpoint_{epoch}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)

        print(
            f"Epoch {epoch} - Val Loss: {avg_loss:.4f}, Mean IoU: {metrics['mean_iou']:.4f}"
        )

    def train(self):
        """Train model for all epochs"""
        for epoch in tqdm(range(self.epochs)):
            print("Epoch:", epoch)
            self.__train_epoch(epoch)
            self.__validate(epoch)
            # Step the scheduler and log current learning rate
            try:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("Train/LR", current_lr, epoch)
            except Exception:
                pass


if __name__ == "__main__":

    disable_caching()
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    trainer = SegFormerTrainer(config_path=args.config)
    trainer.train()
