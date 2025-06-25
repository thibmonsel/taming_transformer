import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer_indices, use_input_norm=True):
        """
        Args:
            feature_layer_indices (list of int): Indices of VGG16 layers to extract features from.
            use_input_norm (bool): If True, normalize inputs with ImageNet mean and std.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm

        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        for param in vgg16.parameters():
            param.requires_grad = False

        self.features = nn.ModuleList()
        current_idx = 0
        for i, layer in enumerate(vgg16.children()):
            if current_idx > max(feature_layer_indices):
                break
            self.features.append(layer)
            if i in feature_layer_indices:
                pass
            current_idx += 1

        self.feature_layer_indices = feature_layer_indices

        if self.use_input_norm:
            # ImageNet normalization (values from PyTorch's VGG16 example)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        extracted_features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layer_indices:
                extracted_features.append(x)

        return extracted_features


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        feature_layer_indices=[3, 8, 15, 22],
        loss_fn=nn.L1Loss(),
        weights=None,
        normalize_inputs_to_01=True,
    ):
        """
        Args:
            feature_layer_indices (list of int): VGG16 layers to use.
            loss_fn (nn.Module): Loss function to compare features (e.g., nn.L1Loss, nn.MSELoss).
            weights (list of float, optional): Weights for each feature layer's loss.
                                               If None, equal weights are used.
            device (str): Device.
        """
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = VGGFeatureExtractor(
            feature_layer_indices, use_input_norm=True
        )
        self.loss_fn = loss_fn
        self.weights = (
            weights
            if weights is not None
            else [1.0 / len(feature_layer_indices)] * len(feature_layer_indices)
        )
        self.normalize_inputs_to_01 = normalize_inputs_to_01
        if len(self.weights) != len(feature_layer_indices):
            raise ValueError("Number of weights must match number of feature layers.")

    def _normalize_batch_to_01(self, tensor_bchw: torch.Tensor) -> torch.Tensor:
        """
        Normalizes each image (C,H,W) in a batch (B,C,H,W) to the range [0,1]
        based on its own min and max values across C, H, W.
        """
        if not torch.is_floating_point(tensor_bchw):
            tensor_bchw = tensor_bchw.float()

        # Reshape to (B, -1) to find min/max per image in batch
        original_shape = tensor_bchw.shape
        tensor_b_chw_flat = tensor_bchw.view(original_shape[0], -1)

        min_vals = tensor_b_chw_flat.min(dim=1, keepdim=True)[0]  # Shape (B, 1)
        max_vals = tensor_b_chw_flat.max(dim=1, keepdim=True)[0]  # Shape (B, 1)

        # Expand min_vals and max_vals to be broadcastable with the original tensor shape
        # From (B, 1) to (B, 1, 1, 1) for broadcasting with (B, C, H, W)
        min_vals_expanded = min_vals.view(original_shape[0], 1, 1, 1)
        max_vals_expanded = max_vals.view(original_shape[0], 1, 1, 1)

        range_vals = max_vals_expanded - min_vals_expanded
        epsilon = torch.finfo(range_vals.dtype).eps
        normalized_tensor = (tensor_bchw - min_vals_expanded) / (range_vals + epsilon)
        return torch.clamp(normalized_tensor, 0.0, 1.0)

    def forward(self, generated_img, target_img):
        if self.normalize_inputs_to_01:
            # This normalizes each image in the batch (across its C,H,W) to [0,1]
            # CAUTION: This might not be standard for VGG if images already have a global intensity meaning.
            # It's better if your input images are *already* in [0,1] (e.g. output of sigmoid, or /255)
            # or [-1,1] then scaled by (x+1)/2.
            generated_img = self._normalize_batch_to_01(generated_img)
            target_img = self._normalize_batch_to_01(target_img)
        else:
            # If not normalizing here, we assume they are already in a suitable range, typically [0,1].
            # Let's add a clamp for safety, as VGG expects [0,1] before its internal normalization.
            # This is a soft guarantee. Best to ensure upstream.
            generated_img = torch.clamp(generated_img, 0.0, 1.0)
            target_img = torch.clamp(target_img, 0.0, 1.0)

        features_generated = self.feature_extractor(generated_img)
        features_target = self.feature_extractor(target_img)

        total_loss = torch.tensor(
            0.0, device=generated_img.device, requires_grad=True
        )  # Ensure it's a leaf tensor that requires grad if needed

        for i in range(len(features_generated)):
            loss_layer = self.loss_fn(features_generated[i], features_target[i])
            total_loss = total_loss + self.weights[i] * loss_layer

        return total_loss


# # --- 3. Example Usage ---
# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Define layer indices for VGG16 features
#     # Common choices for VGG16:
#     # relu1_2 (idx 3), relu2_2 (idx 8), relu3_3 (idx 15), relu4_3 (idx 22)
#     # Note: VGG16 layers are:
#     # 0: Conv2d, 1: ReLU, 2: Conv2d, 3: ReLU (relu1_2), 4: MaxPool2d
#     # 5: Conv2d, 6: ReLU, 7: Conv2d, 8: ReLU (relu2_2), 9: MaxPool2d
#     # 10: Conv2d, 11: ReLU, 12: Conv2d, 13: ReLU, 14: Conv2d, 15: ReLU (relu3_3), 16: MaxPool2d
#     # 17: Conv2d, 18: ReLU, 19: Conv2d, 20: ReLU, 21: Conv2d, 22: ReLU (relu4_3), 23: MaxPool2d
#     # ... and so on. We use the ReLU activations.
#     feature_layers = [3, 8, 15, 22]
#     layer_weights = [0.25, 0.25, 0.25, 0.25] # Equal weights

#     # Initialize perceptual loss
#     # You can choose L1Loss or MSELoss (L2Loss)
#     perceptual_criterion = PerceptualLoss(
#         feature_layer_indices=feature_layers,
#         loss_fn=nn.MSELoss(),
#         weights=layer_weights,
#         device=device
#     )
#     print("PerceptualLoss module initialized.")

#     # Create dummy images (Batch size 1, 3 Channels, 256x256 Height/Width)
#     # Ensure images are in the range [0, 1]
#     batch_size = 2
#     img_size = 224 # VGG was trained on 224x224, but can handle other sizes

#     # A "generated" image that we want to optimize to match the target
#     # Let's start with random noise
#     generated_image = torch.rand(batch_size, 3, img_size, img_size, device=device, requires_grad=True)

#     # A "target" image (e.g., ground truth)
#     target_image = torch.rand(batch_size, 3, img_size, img_size, device=device)

#     # --- Simple Optimization Example ---
#     print("\n--- Optimizing a random image to match another random image using perceptual loss ---")

#     optimizer = torch.optim.Adam([generated_image], lr=0.01)
#     num_epochs = 50

#     for epoch in range(num_epochs):
#         optimizer.zero_grad()

#         # Calculate perceptual loss
#         loss = perceptual_criterion(generated_image, target_image)

#         loss.backward()
#         optimizer.step()

#         # Optional: clamp image to [0, 1] range after optimizer step
#         with torch.no_grad():
#             generated_image.data.clamp_(0.0, 1.0)

#         if (epoch + 1) % 10 == 0:
#             print(f"Epoch [{epoch+1}/{num_epochs}], Perceptual Loss: {loss.item():.4f}")

#     print("Optimization finished.")

#     # --- Test with identical images (loss should be close to 0) ---
#     print("\n--- Testing with identical images ---")
#     identical_image = torch.rand(batch_size, 3, img_size, img_size, device=device)
#     loss_identical = perceptual_criterion(identical_image, identical_image.clone()) # Use clone for safety
#     print(f"Perceptual Loss for identical images: {loss_identical.item():.6f}")

#     # --- Test with very different images (loss should be higher) ---
#     print("\n--- Testing with very different images ---")
#     img1 = torch.zeros(batch_size, 3, img_size, img_size, device=device) # All black
#     img2 = torch.ones(batch_size, 3, img_size, img_size, device=device)  # All white
#     loss_different = perceptual_criterion(img1, img2)
#     print(f"Perceptual Loss for black vs white images: {loss_different.item():.4f}")
