#TransformerFusion treats text and image features as a sequence and models their interactions using transformer encoder layers. 
#Positional encoding is added to distinguish between text and image inputs. 
#This method captures cross-modal dependencies over multiple layers.

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(TransformerFusion, self).__init__()
        # Initialize positional encoding to differentiate text and image inputs
        self.positional_encoding = nn.Parameter(torch.zeros(1, 2, embed_dim))
        
        # Transformer encoder layer with self-attention and feedforward components
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads)
        
        # Stack multiple transformer encoder layers
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

    def forward(self, text_features, image_features):
        # Stack text and image features along a new dimension (sequence of length 2)
        combined = torch.stack((text_features, image_features), dim=1)
        
        # Add positional encoding to the combined sequence
        combined += self.positional_encoding
        
        # Pass the sequence through the transformer encoder
        return self.transformer(combined)
