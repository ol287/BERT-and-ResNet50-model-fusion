#CrossAttention allows one modality (e.g., text) to selectively focus on parts of the other modality (e.g., image) by computing attention weights. 
# This helps capture relationships where one modality's context is relevant to interpreting the other

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        # Initialize layers to compute query, key, and value matrices
        self.query = nn.Linear(embed_dim, embed_dim)  # Projects input to query space
        self.key = nn.Linear(embed_dim, embed_dim)    # Projects input to key space
        self.value = nn.Linear(embed_dim, embed_dim)  # Projects input to value space
        self.scale = embed_dim ** 0.5  # Scaling factor to normalize attention scores

    def forward(self, query, key, value):
        # Transform the input into query, key, and value representations
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)

        # Compute scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply softmax to get attention weights (importance of each key to the query)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        # Multiply attention weights with values to get the final attended representation
        attended_output = torch.matmul(attention_weights, v)
        return attended_output

