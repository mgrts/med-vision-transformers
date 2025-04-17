from torch import nn


class MIMHead(nn.Module):
    def __init__(self, embed_dim, image_size, patch_size, dropout_rate=0.1):
        super(MIMHead, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches_per_dim = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_dim ** 2
        self.dropout_rate = dropout_rate

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(in_channels=embed_dim * 2, out_channels=embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(in_channels=embed_dim // 2, out_channels=3 * (patch_size ** 2), kernel_size=1),
            nn.PixelShuffle(patch_size)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.last_hidden_state[:, 1:, :]  # shape: [batch_size, num_patches, embed_dim]
        x = x.view(-1, self.num_patches_per_dim, self.num_patches_per_dim, self.embed_dim)
        x = x.permute(0, 3, 1, 2)
        x = self.decoder(x)
        x = self.sigmoid(x)
        return x


class MultiLabelClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_patches_per_dim, num_classes, dropout_rate=0.1):
        super(MultiLabelClassificationHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_patches_per_dim = num_patches_per_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Convolutional layer to process patch embeddings
        self.conv = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)

        # Global average pooling to aggregate patch features
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Outputs a single value per channel

        # Fully connected layer for classification with Dropout
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),  # Hidden layer
            # nn.ReLU(),  # Activation
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # Activation
            nn.Dropout(p=self.dropout_rate),  # Add dropout here
            nn.Linear(128, self.num_classes)  # Output layer based on num_classes
        )

    def forward(self, x):
        # Extract all patch tokens (excluding the CLS token)
        x = x.last_hidden_state[:, 1:, :]  # shape: [batch_size, num_patches, embed_dim]

        # Reshape to [batch_size, embed_dim, num_patches_per_dim, num_patches_per_dim]
        x = x.view(-1, self.num_patches_per_dim, self.num_patches_per_dim, self.embed_dim)

        # Permute to match Conv2D input format: [batch_size, embed_dim, num_patches_per_dim, num_patches_per_dim]
        x = x.permute(0, 3, 1, 2)  # Shape: [batch_size, embed_dim, num_patches_per_dim, num_patches_per_dim]

        # Apply convolution to capture spatial relationships
        x = self.conv(x)

        # Apply global average pooling to aggregate patch embeddings
        x = self.global_avg_pool(x)  # Shape: [batch_size, embed_dim, 1, 1]

        # Flatten the output
        x = x.view(x.size(0), -1)  # Shape: [batch_size, embed_dim]

        # Apply the fully connected layer to get class logits
        class_logits = self.fc(x)  # Shape: [batch_size, num_classes]

        return class_logits  # Return logits for the entire image


class MIMTransformer(nn.Module):
    def __init__(self, base_model, dropout_rate=0.1):
        super().__init__()
        self.base_model = base_model
        self.image_size = self.base_model.config.image_size
        self.patch_size = self.base_model.config.patch_size
        self.embed_dim = self.base_model.config.hidden_size
        self.dropout_rate = dropout_rate
        self.mim_head = MIMHead(self.embed_dim, self.image_size, self.patch_size, self.dropout_rate)

    def forward(self, x):
        base_output = self.base_model(x)
        mim_output = self.mim_head(base_output)
        return mim_output


class MultiLabelClassificationTransformer(nn.Module):
    def __init__(self, base_model, num_classes, dropout_rate=0.1):
        super().__init__()
        self.base_model = base_model
        self.embed_dim = self.base_model.config.hidden_size
        self.dropout_rate = dropout_rate
        self.classification_head = MultiLabelClassificationHead(
            embed_dim=self.embed_dim,
            num_patches_per_dim=self.base_model.config.image_size // self.base_model.config.patch_size,
            num_classes=num_classes,
            dropout_rate=self.dropout_rate
        )

    def forward(self, x):
        x_clf = self.base_model(x)
        class_output = self.classification_head(x_clf)
        return class_output


class MultiTaskTransformer(nn.Module):
    def __init__(self, base_model, image_size, num_classes, dropout_rate=0.1):
        super().__init__()
        self.base_model = base_model
        self.patch_size = self.base_model.config.patch_size
        self.embed_dim = self.base_model.config.hidden_size
        self.dropout_rate = dropout_rate
        self.classification_head = MultiLabelClassificationHead(
            embed_dim=self.embed_dim,
            num_patches_per_dim=image_size // self.patch_size,
            num_classes=num_classes,
            dropout_rate=self.dropout_rate
        )
        self.mim_head = MIMHead(self.embed_dim, image_size, self.patch_size, self.dropout_rate)

    def forward(self, x, x_masked):
        x_base = self.base_model(x)

        # Classification using raw pixel values
        class_output = self.classification_head(x_base)

        x_masked_base = self.base_model(x_masked)
        mim_output = self.mim_head(x_masked_base)

        return mim_output, class_output
