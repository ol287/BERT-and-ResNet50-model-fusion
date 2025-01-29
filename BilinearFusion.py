#BilinearFusion captures multiplicative (pairwise) interactions between features from text and image modalities. 
# This fusion is useful for modeling higher-order relationships, such as how specific words relate to parts of an image.

class BilinearFusion(nn.Module):
    def __init__(self, text_dim, image_dim, output_dim):
        super(BilinearFusion, self).__init__()
        # Linear layers to project text and image features into a shared space
        self.proj_text = nn.Linear(text_dim, output_dim)  # Projects text features
        self.proj_image = nn.Linear(image_dim, output_dim)  # Projects image features
        
    def forward(self, text_features, image_features):
        # Project text features into the shared space
        text_proj = self.proj_text(text_features)
        
        # Project image features into the shared space
        image_proj = self.proj_image(image_features)
        
        # Perform element-wise multiplication to capture relationships between the two modalities
        bilinear_output = text_proj * image_proj
        return bilinear_output
