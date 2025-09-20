"""
Australian Tourism Sentiment Model for Production Deployment Demo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

class AustralianTourismSentimentAnalyzer(nn.Module):
    """
    Production-ready sentiment analysis model for Australian tourism reviews.
    
    Supports multilingual analysis (English + Vietnamese) for:
    - Hotel and restaurant reviews
    - Tourist attraction feedback
    - Travel experience sentiment
    
    Args:
        vocab_size (int): Size of vocabulary (combined English + Vietnamese)
        embed_dim (int): Dimension of embedding layer
        hidden_dim (int): Dimension of LSTM hidden states
        num_classes (int): Number of sentiment classes (3: positive/neutral/negative)
        dropout_rate (float): Dropout rate for regularization
    
    TensorFlow Comparison:
        TF equivalent would use tf.keras.Sequential with:
        - tf.keras.layers.Embedding
        - tf.keras.layers.LSTM 
        - tf.keras.layers.Dense
        But PyTorch gives more explicit control over forward pass.
    """
    
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256, 
                 num_classes=3, dropout_rate=0.3):
        super(AustralianTourismSentimentAnalyzer, self).__init__()
        
        # Store hyperparameters for TorchScript compatibility
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Model layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, 
                           dropout=dropout_rate if dropout_rate > 0 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, input_ids):
        """
        Forward pass for sentiment analysis.
        
        Args:
            input_ids (torch.Tensor): Tokenized input sequences [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Sentiment logits [batch_size, num_classes]
        """
        # Embedding lookup
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state for classification
        # In production, we could use attention pooling instead
        last_hidden = hidden[-1]  # [batch_size, hidden_dim]
        
        # Apply dropout and classification
        dropped = self.dropout(last_hidden)
        logits = self.classifier(dropped)
        
        return logits
    
    def predict_sentiment(self, input_ids):
        """
        Production-ready prediction method with human-readable outputs.
        
        Returns:
            dict: Prediction results with sentiment labels and confidence
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]
            
            # Map predictions to human-readable labels
            sentiment_labels = ['negative', 'neutral', 'positive']
            
            results = []
            for i in range(len(predictions)):
                results.append({
                    'sentiment': sentiment_labels[predictions[i].item()],
                    'confidence': confidence[i].item(),
                    'probabilities': {
                        'negative': probabilities[i][0].item(),
                        'neutral': probabilities[i][1].item(),
                        'positive': probabilities[i][2].item()
                    }
                })
            
            return results

class SimpleTokenizer:
    """Simple tokenizer for Australian tourism text."""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {'<pad>': 0, '<unk>': 1}
        self.idx_to_word = {0: '<pad>', 1: '<unk>'}
        self.next_idx = 2
    
    def fit(self, texts):
        """Build vocabulary from texts."""
        for text in texts:
            words = text.lower().split()
            for word in words:
                if word not in self.word_to_idx and self.next_idx < self.vocab_size:
                    self.word_to_idx[word] = self.next_idx
                    self.idx_to_word[self.next_idx] = word
                    self.next_idx += 1
    
    def encode(self, text, max_length=128):
        """Convert text to token indices."""
        words = text.lower().split()[:max_length]
        indices = [self.word_to_idx.get(word, 1) for word in words]  # 1 = <unk>
        
        # Pad to max_length
        if len(indices) < max_length:
            indices.extend([0] * (max_length - len(indices)))  # 0 = <pad>
        
        return indices
    
    def save(self, path):
        """Save tokenizer to file."""
        vocab_data = {
            'vocab_size': self.vocab_size,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'next_idx': self.next_idx
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    def load(self, path):
        """Load tokenizer from file."""
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        self.vocab_size = vocab_data['vocab_size']
        self.word_to_idx = vocab_data['word_to_idx']
        self.idx_to_word = {int(k): v for k, v in vocab_data['idx_to_word'].items()}
        self.next_idx = vocab_data['next_idx']

# Sample Australian tourism data
AUSTRALIAN_TOURISM_DATA = {
    'reviews': [
        # Positive reviews (label: 2)
        "The Sydney Opera House tour was absolutely breathtaking! Best experience in Australia.",
        "Nhà hát Opera Sydney thật tuyệt vời! Trải nghiệm tốt nhất ở Úc.",
        "Melbourne's coffee culture exceeded all expectations. Amazing laneways and street art!",
        "Văn hóa cà phê Melbourne vượt quá mong đợi. Nghệ thuật đường phố tuyệt vời!",
        "Bondi Beach is perfect for surfing. Crystal clear water and great waves all day.",
        "Bãi biển Bondi hoàn hảo cho lướt sóng. Nước trong veo và sóng tuyệt vời.",
        "The Great Barrier Reef snorkeling was a once-in-a-lifetime experience.",
        "Lặn ngắm san hô Great Barrier Reef là trải nghiệm một lần trong đời.",
        "Perth's beaches are pristine and less crowded than other Australian cities.",
        "Bãi biển Perth sạch sẽ và ít đông đúc hơn các thành phố Úc khác.",
        
        # Neutral reviews (label: 1)
        "The hotel in Brisbane was decent, nothing special but clean and comfortable.",
        "Khách sạn ở Brisbane tạm được, không có gì đặc biệt nhưng sạch sẽ và thoải mái.",
        "Adelaide zoo has some interesting animals, worth a visit if you're in the area.",
        "Sở thú Adelaide có một số động vật thú vị, đáng ghé thăm nếu bạn ở gần đó.",
        "The food at the restaurant was okay, average quality for the price.",
        "Đồ ăn ở nhà hàng tạm được, chất lượng trung bình so với giá cả.",
        "Darwin's weather is quite hot and humid, take that into consideration.",
        "Thời tiết Darwin khá nóng và ẩm, cần cân nhắc điều này.",
        
        # Negative reviews (label: 0)
        "The Sydney harbor cruise was overpriced and disappointing. Not worth the money.",
        "Du thuyền cảng Sydney đắt quá và thất vọng. Không xứng đáng với số tiền bỏ ra.",
        "Melbourne weather ruined our entire vacation. Constant rain for five days straight.",
        "Thời tiết Melbourne làm hỏng cả kỳ nghỉ. Mưa liên tục suốt 5 ngày.",
        "The hotel staff in Gold Coast were rude and unhelpful throughout our stay.",
        "Nhân viên khách sạn ở Gold Coast thô lỗ và không hữu ích suốt thời gian lưu trú.",
        "Cairns tour guide was unprofessional and the activities were poorly organized.",
        "Hướng dẫn viên ở Cairns không chuyên nghiệp và các hoạt động được tổ chức kém.",
        "The restaurant in Hobart served terrible food and the service was extremely slow.",
        "Nhà hàng ở Hobart phục vụ đồ ăn tệ và dịch vụ cực kỳ chậm."
    ],
    'labels': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  # Positive
              1, 1, 1, 1, 1, 1, 1, 1,           # Neutral  
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     # Negative
}

def create_sample_model_and_data():
    """Create a sample trained model and tokenizer for deployment demos."""
    
    # Create tokenizer and fit on data
    tokenizer = SimpleTokenizer(vocab_size=10000)
    tokenizer.fit(AUSTRALIAN_TOURISM_DATA['reviews'])
    
    # Create model
    model = AustralianTourismSentimentAnalyzer(
        vocab_size=len(tokenizer.word_to_idx),
        embed_dim=128,
        hidden_dim=256,
        num_classes=3,
        dropout_rate=0.3
    )
    
    # Quick training simulation (just for demo weights)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare training data
    encoded_reviews = []
    for review in AUSTRALIAN_TOURISM_DATA['reviews']:
        encoded = tokenizer.encode(review, max_length=128)
        encoded_reviews.append(encoded)
    
    input_ids = torch.tensor(encoded_reviews, dtype=torch.long)
    labels_tensor = torch.tensor(AUSTRALIAN_TOURISM_DATA['labels'], dtype=torch.long)
    
    # Quick training for demo
    for epoch in range(3):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model, tokenizer

if __name__ == "__main__":
    # Test the model creation
    model, tokenizer = create_sample_model_and_data()
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"✅ Tokenizer created with vocabulary size: {len(tokenizer.word_to_idx)}")
    
    # Test prediction
    test_text = "Sydney Opera House is absolutely amazing!"
    encoded = torch.tensor([tokenizer.encode(test_text)], dtype=torch.long)
    result = model.predict_sentiment(encoded)
    print(f"✅ Test prediction: '{test_text}' -> {result[0]['sentiment']} ({result[0]['confidence']:.3f})")