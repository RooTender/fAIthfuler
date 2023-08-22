"""Provides CNN utils made for training neural network."""
import os
import sys
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import wandb


class FaithfulNet(nn.Module):
    """Convolutional neural network for Faithful textures generation."""

    def __init__(self):
        super(FaithfulNet, self).__init__()
        # Upscaling layers (for example, using Transposed Convolution)
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(4, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, image):
        """Forward function of the neural network."""
        return self.upscale(image)


class ImageLoaderDataset(Dataset):
    def __init__(self, input_root_dir, output_root_dir, transform=None):
        self.input_root_dir = input_root_dir
        self.output_root_dir = output_root_dir
        self.transform = transform
        self.image_paths = [(os.path.join(input_root_dir, file),
                             os.path.join(output_root_dir, file))
                            for file in os.listdir(input_root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        input_image_path, output_image_path = self.image_paths[idx]
        input_image = Image.open(input_image_path)
        output_image = Image.open(output_image_path)

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image


class CNN:
    """Trains neural networks and evaluates them."""

    def __load_data(self, input_path: str, output_path: str, batch_size: int):
        print(f'Input dataset: {input_path}')
        print(f'Output dataset: {output_path}')
        print('Reading datasets...')

        transformator = transforms.Compose([
            transforms.ToTensor()  # Convert images to tensors
        ])

        input_data = ImageLoaderDataset(
            input_path, output_path, transform=transformator)

        return DataLoader(input_data, batch_size=batch_size, pin_memory=True)

    def __train(self, dir_for_models: str, data_loader: DataLoader,
                num_of_epochs: int, learning_rate: float, early_stopping_patience: int,
                optimizer_type: str = 'adam',
                fine_tune_model: str = ''):
        model = FaithfulNet().cuda()

        if optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Unsupported optimizer type:", optimizer_type)

        criterion_l1 = nn.L1Loss()
        criterion_mse = nn.MSELoss()

        if fine_tune_model != '':
            checkpoint = torch.load(fine_tune_model)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']

        else:
            start_epoch = 0
            best_loss = float('inf')

        no_improvement_count = 0

        for epoch in range(start_epoch, start_epoch + num_of_epochs):
            total_l1 = 0.0
            total_mse = 0.0

            total_loss = 0.0
            total_batches = 0

            for (input_batch, output_batch) in tqdm(
                    data_loader,
                    unit="batch",
                    desc=f'epoch {epoch}'):

                input_batch = input_batch.cuda()
                output_batch = output_batch.cuda()

                optimizer.zero_grad()
                prediction = model(input_batch)

                # Loss
                l1_loss = criterion_l1(prediction, output_batch)
                mse_loss = criterion_mse(prediction, output_batch)

                loss = 0.5 * l1_loss + 0.5 * mse_loss

                total_loss += loss.item()
                total_batches += 1

                # Metrics
                total_l1 += l1_loss.item()
                total_mse += mse_loss.item()

                # Step
                loss.backward()
                nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()

            avg_l1 = total_l1 / total_batches
            avg_mse = total_mse / total_batches
            avg_loss = total_loss / total_batches

            wandb.log({
                "L1": avg_l1,
                "mse": avg_mse,
                "loss": avg_loss})

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = model.state_dict()
                no_improvement_count = 0

                os.makedirs(os.path.join(
                    '..', 'models', dir_for_models), exist_ok=True)
                torch.save(best_model_state, os.path.join(
                    '..', 'models', dir_for_models, f'e{epoch}_l{avg_loss:.4f}.pth'))
            else:
                no_improvement_count += 1
                if no_improvement_count >= early_stopping_patience:
                    print("Early stopping: No improvement for",
                          early_stopping_patience, "epochs.")
                    break

    def run(self, input_path: str, output_path: str, fine_tune_model: str = ''):
        """Just a test..."""

        # SGD performs the best on 0.1
        # Adam on 0.001
        learning_rate = 0.01

        # SGD on 16
        # Adam on ???
        batch_size = 16
        epochs = 100
        patience = 20
        optimizer = 'sgd'

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="FAIthfuler",
            entity="rootender",
            name="FaithfulNet",
            # track hyperparameters and run metadata
            config={
                "architecture": "CNN",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "hidden_layer_sizes": [256, 128, 64, 4],
                "optimizer": optimizer
            }
        )

        if not torch.cuda.is_available():
            print("CUDA architecture is unavailable!")
            sys.exit(1)
        else:
            print("CUDA detected! Initializing...")

        loaded_data = self.__load_data(
            input_path=input_path,
            output_path=output_path,
            batch_size=batch_size
        )

        # simulate training
        self.__train(f'{os.path.dirname(input_path)}_b{batch_size}_lr{learning_rate}',
                     loaded_data,
                     epochs, learning_rate, patience,
                     optimizer_type=optimizer,
                     fine_tune_model=fine_tune_model)

        wandb.finish()

    def generate_image(self, input_image_path: str, model_path="faithfulnet.pth"):
        """Generates an image using a GAN generator model."""

        # Load the FaithfulNet model and its trained weights
        model = FaithfulNet()
        model.load_state_dict(torch.load(model_path))
        model.cuda()
        model.eval()

        # Load and preprocess the input image
        input_image = Image.open(input_image_path)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        input_tensor = transform(input_image).unsqueeze(  # type: ignore
            0).cuda()

        # Generate the image
        with torch.no_grad():
            generated_image = model(input_tensor)

        # Post-process the generated image
        # Rescale from [-1, 1] to [0, 1]
        generated_image = generated_image.squeeze().cpu().detach()
        generated_image = (generated_image + 1) / 2.0
        generated_image = transforms.ToPILImage()(generated_image)
        # Convert the tensor to a PIL image (assuming it's an RGB image)

        os.makedirs(os.path.join('..', 'out'), exist_ok=True)
        generated_image.save(os.path.join(
            '..', 'out', os.path.basename(input_image_path)))
        generated_image.close()
