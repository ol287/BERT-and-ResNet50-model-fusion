#The MultimodalFusionModel combines GatedFusion and BilinearFusion. 
#It first scales the contributions of text and image features using gates  
#Then projects them into a shared space where relationships are captured through bilinear pooling.

class MultimodalFusionModel(nn.Module):
    def __init__(self, text_dim=768, image_dim=2048, fusion_dim=512):
        super(MultimodalFusionModel, self).__init__()
        # Gated Fusion: Compute gating scores for text and image features
        self.text_gate = nn.Linear(text_dim, 1)  # Gating score for text
        self.image_gate = nn.Linear(image_dim, 1)  # Gating score for image
        
        # Bilinear Pooling: Project text and image features into a shared space
        self.text_proj = nn.Linear(text_dim, fusion_dim)  # Text projection
        self.image_proj = nn.Linear(image_dim, fusion_dim)  # Image projection
        
        # Classifier: Predict whether news is real or fake based on fused features
        self.classifier = nn.Sequential(
            nn.ReLU(),              # Non-linear activation to introduce non-linearity
            nn.Dropout(0.3),        # Dropout to prevent overfitting
            nn.Linear(fusion_dim, 1),  # Linear layer for binary classification
            nn.Sigmoid()            # Sigmoid activation to output probability (0 to 1)
        )
    
    def forward(self, text_features, image_features):
        # Compute gating scores for text and image features
        text_score = torch.sigmoid(self.text_gate(text_features))
        image_score = torch.sigmoid(self.image_gate(image_features))
        
        # Multiply gating scores with corresponding features
        gated_text = text_score * text_features
        gated_image = image_score * image_features
        
        # Project gated features into a shared space
        text_proj = self.text_proj(gated_text)
        image_proj = self.image_proj(gated_image)
        
        # Perform element-wise multiplication to capture relationships between modalities
        bilinear_output = text_proj * image_proj
        
        # Pass the fused features through the classifier
        return self.classifier(bilinear_output)



