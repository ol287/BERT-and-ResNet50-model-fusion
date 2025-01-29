#GatedFusion assigns learnable weights to text and image features based on their importance for the current task. 
#The gates dynamically scale each modality, allowing the model to emphasize one over the other when appropriate.

class GatedFusion(nn.Module):
    def __init__(self, text_dim, image_dim):
        super(GatedFusion, self).__init__()
        # Linear layers to compute gating scores for text and image features
        self.text_gate = nn.Linear(text_dim, 1)  # Outputs a gating score for text
        self.image_gate = nn.Linear(image_dim, 1)  # Outputs a gating score for image
        
    def forward(self, text_features, image_features):
        # Compute the gating score for text features (value between 0 and 1)
        text_score = torch.sigmoid(self.text_gate(text_features))
        
        # Compute the gating score for image features (value between 0 and 1)
        image_score = torch.sigmoid(self.image_gate(image_features))
        
        # Multiply the gating score with the corresponding features
        gated_text = text_score * text_features
        gated_image = image_score * image_features
        
        # Concatenate the gated text and image features into one vector
        return torch.cat((gated_text, gated_image), dim=1)
