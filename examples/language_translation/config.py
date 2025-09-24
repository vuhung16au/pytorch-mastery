"""
Configuration settings for PyTorch Language Translation models.

This module contains all hyperparameters, paths, and settings used across
the translation implementation, following repository standards for 
Australian context and English-Vietnamese translation.
"""

import torch
from pathlib import Path
from typing import Dict, Any

class TranslationConfig:
    """
    Configuration class for translation models with Australian context.
    
    All hyperparameters are optimized for English-Vietnamese translation
    with Australian tourism and culture examples.
    """
    
    # Model Architecture Settings
    EMBED_DIM = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.1
    
    # Transformer Settings  
    D_MODEL = 512
    N_HEADS = 8
    N_ENCODER_LAYERS = 6
    N_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 2048
    
    # Vocabulary Settings
    SRC_VOCAB_SIZE = 10000  # English vocabulary
    TGT_VOCAB_SIZE = 8000   # Vietnamese vocabulary
    MAX_SEQ_LENGTH = 100    # Maximum sentence length
    
    # Special Tokens
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    SOS_TOKEN = '<sos>'  # Start of sentence
    EOS_TOKEN = '<eos>'  # End of sentence
    
    PAD_IDX = 0
    UNK_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3
    
    # Training Settings
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    GRADIENT_CLIP_NORM = 1.0
    
    # Device-specific batch sizes (following repository standards)
    DEVICE_BATCH_SIZES = {
        'cuda': 64,    # GPU can handle larger batches
        'mps': 48,     # Apple Silicon moderate batch size
        'cpu': 16      # CPU smaller batch size
    }
    
    # Device-specific learning rates
    DEVICE_LEARNING_RATES = {
        'cuda': 0.001,
        'mps': 0.0008,
        'cpu': 0.0005
    }
    
    # Data Paths
    DATA_DIR = Path("data")
    MODEL_DIR = Path("models") 
    TENSORBOARD_DIR = Path("tensorboard_logs")
    
    # Australian Context Dataset
    AUSTRALIAN_TRANSLATION_PAIRS = [
        # Tourism and landmarks
        ("The Sydney Opera House attracts millions of visitors each year", 
         "Nhà hát Opera Sydney thu hút hàng triệu du khách mỗi năm"),
        ("Melbourne is famous for its coffee culture and street art", 
         "Melbourne nổi tiếng với văn hóa cà phê và nghệ thuật đường phố"),
        ("Bondi Beach is perfect for surfing and swimming", 
         "Bãi biển Bondi hoàn hảo cho lướt sóng và bơi lội"),
        ("The Great Barrier Reef is a UNESCO World Heritage site", 
         "Rạn san hô Great Barrier Reef là di sản thế giới UNESCO"),
        ("Perth has beautiful beaches and a Mediterranean climate", 
         "Perth có những bãi biển đẹp và khí hậu Địa Trung Hải"),
        ("Uluru is sacred to the Aboriginal people of Australia", 
         "Uluru là nơi thiêng liêng đối với người thổ dân Australia"),
        ("The Sydney Harbour Bridge offers spectacular views", 
         "Cầu Cảng Sydney mang đến tầm nhìn ngoạn mục"),
        ("Brisbane is the gateway to the Gold Coast", 
         "Brisbane là cửa ngõ đến Gold Coast"),
        ("Adelaide is known for its wine regions and festivals", 
         "Adelaide nổi tiếng với vùng rượu vang và lễ hội"),
        ("Darwin is the gateway to Kakadu National Park", 
         "Darwin là cửa ngõ đến Công viên Quốc gia Kakadu"),
        
        # Culture and lifestyle
        ("Australian coffee is among the best in the world", 
         "Cà phê Úc là một trong những loại tốt nhất thế giới"),
        ("The Aboriginal culture has a history of over 60,000 years", 
         "Văn hóa thổ dân có lịch sử hơn 60,000 năm"),
        ("AFL is the most popular sport in Melbourne", 
         "AFL là môn thể thao phổ biến nhất ở Melbourne"),
        ("Cricket is Australia's national summer sport", 
         "Cricket là môn thể thao mùa hè quốc gia của Australia"),
        ("The didgeridoo is a traditional Aboriginal instrument", 
         "Didgeridoo là nhạc cụ truyền thống của người thổ dân"),
        ("Anzac Day commemorates Australian and New Zealand soldiers", 
         "Ngày Anzac tưởng niệm binh sĩ Australia và New Zealand"),
        ("Fair dinkum means genuine or authentic in Australian slang", 
         "Fair dinkum có nghĩa là chính hiệu hoặc xác thực trong tiếng lóng Úc"),
        ("G'day mate is a typical Australian greeting", 
         "G'day mate là lời chào đặc trưng của người Úc"),
        ("The outback refers to remote areas of Australia", 
         "Outback chỉ những vùng xa xôi của Australia"),
        ("Bush telegraph means informal communication network", 
         "Bush telegraph có nghĩa là mạng lưới giao tiếp không chính thức"),
        
        # Food and dining
        ("Try the famous Australian meat pie with tomato sauce", 
         "Hãy thử bánh thịt Úc nổi tiếng với tương cà"),
        ("Barramundi is a popular fish in Australian cuisine", 
         "Cá barramundi là loại cá phổ biến trong ẩm thực Úc"),
        ("Lamington cake is a traditional Australian dessert", 
         "Bánh Lamington là món tráng miệng truyền thống của Úc"),
        ("Pavlova is a meringue-based dessert named after a ballerina", 
         "Pavlova là món tráng miệng làm từ meringue được đặt tên theo nữ diễn viên ba lê"),
        ("Vegemite is a dark brown Australian food paste", 
         "Vegemite là loại tương thực phẩm màu nâu đậm của Úc"),
        ("Tim Tam biscuits are perfect with coffee", 
         "Bánh quy Tim Tam hoàn hảo khi dùng với cà phê"),
        ("Anzac biscuits were originally sent to soldiers", 
         "Bánh quy Anzac ban đầu được gửi cho binh lính"),
        ("Damper bread is traditional Australian bush food", 
         "Bánh mì Damper là thực phẩm bush truyền thống của Úc"),
        ("Kangaroo meat is lean and high in protein", 
         "Thịt kangaroo nạc và giàu protein"),
        ("Macadamia nuts originated in Australia", 
         "Hạt macadamia có nguồn gốc từ Australia"),
        
        # Wildlife and nature
        ("Kangaroos are iconic Australian marsupials", 
         "Kangaroo là loài thú có túi biểu tượng của Australia"),
        ("Koalas sleep up to 20 hours per day", 
         "Koala ngủ tới 20 tiếng mỗi ngày"),
        ("Platypus is one of the few venomous mammals", 
         "Platypus là một trong số ít động vật có vú có nọc độc"),
        ("Echidnas are egg-laying mammals found in Australia", 
         "Echidna là động vật có vú đẻ trứng được tìm thấy ở Australia"),
        ("Wombats have cube-shaped droppings", 
         "Wombat có phân hình khối vuông"),
        ("Tasmanian devils are the largest carnivorous marsupials", 
         "Tasmanian devil là loài thú có túi ăn thịt lớn nhất"),
        ("Dingoes are Australia's largest land predator", 
         "Dingo là loài động vật săn mồi trên cạn lớn nhất của Australia"),
        ("Kookaburras are known for their distinctive laughing call", 
         "Kookaburra nổi tiếng với tiếng kêu cười đặc trưng"),
        ("Crocodiles are found in northern Australian waters", 
         "Cá sấu được tìm thấy ở vùng nước phía bắc Australia"),
        ("Blue-ringed octopus is one of the world's most venomous creatures", 
         "Bạch tuộc vòng xanh là một trong những sinh vật độc nhất thế giới"),
        
        # Weather and geography
        ("Australia is the world's sixth-largest country by area", 
         "Australia là quốc gia lớn thứ sáu thế giới về diện tích"),
        ("The continent is surrounded by the Indian and Pacific Oceans", 
         "Lục địa được bao quanh bởi Ấn Độ Dương và Thái Bình Dương"),
        ("Queensland has a tropical climate with wet and dry seasons", 
         "Queensland có khí hậu nhiệt đới với mùa mưa và mùa khô"),
        ("Tasmania has pristine wilderness and clean air", 
         "Tasmania có thiên nhiên hoang sơ và không khí trong lành"),
        ("The Red Centre refers to the arid interior of Australia", 
         "Red Centre chỉ vùng nội địa khô cằn của Australia"),
        ("The Wet season brings monsoonal rains to northern Australia", 
         "Mùa mưa mang mưa gió mùa đến phía bắc Australia"),
        ("Bushfires are a natural part of Australia's ecosystem", 
         "Cháy rừng là một phần tự nhiên của hệ sinh thái Australia"),
        ("The Murray River is Australia's longest river", 
         "Sông Murray là con sông dài nhất của Australia"),
        ("The Blue Mountains are famous for their eucalyptus forests", 
         "Blue Mountains nổi tiếng với rừng bạch đàn"),
        ("Cyclones affect northern Australia during summer months", 
         "Bão xoáy ảnh hưởng đến phía bắc Australia trong những tháng mùa hè")
    ]
    
    # Vietnamese-English pairs (for back-translation evaluation)
    VIETNAMESE_ENGLISH_PAIRS = [
        ("Nhà hát Opera Sydney là biểu tượng của Australia", 
         "Sydney Opera House is an icon of Australia"),
        ("Melbourne có văn hóa cà phê tuyệt vời", 
         "Melbourne has wonderful coffee culture"),
        ("Bãi biển Bondi rất nổi tiếng với du khách", 
         "Bondi Beach is very famous with tourists"),
        ("Great Barrier Reef cần được bảo vệ", 
         "The Great Barrier Reef needs to be protected"),
        ("Kangaroo là động vật đặc trưng của Úc", 
         "Kangaroos are characteristic animals of Australia")
    ]
    
    # Evaluation Settings
    BLEU_SMOOTHING = True
    METEOR_ALPHA = 0.9
    BEAM_WIDTH = 5
    MAX_DECODE_LENGTH = 150
    
    # TensorBoard Settings (following repository standards)
    LOG_INTERVAL = 100  # Log every 100 batches
    SAVE_INTERVAL = 5   # Save model every 5 epochs
    
    @classmethod
    def get_device_config(cls, device: torch.device) -> Dict[str, Any]:
        """Get device-specific configuration settings."""
        device_type = device.type
        
        return {
            'batch_size': cls.DEVICE_BATCH_SIZES.get(device_type, cls.BATCH_SIZE),
            'learning_rate': cls.DEVICE_LEARNING_RATES.get(device_type, cls.LEARNING_RATE),
            'num_workers': 4 if device_type == 'cuda' else 2
        }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories for the project."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.MODEL_DIR.mkdir(exist_ok=True) 
        cls.TENSORBOARD_DIR.mkdir(exist_ok=True)
        
        print(f"✅ Created directories:")
        print(f"   📁 Data: {cls.DATA_DIR}")
        print(f"   📁 Models: {cls.MODEL_DIR}")
        print(f"   📁 TensorBoard: {cls.TENSORBOARD_DIR}")

# Export configuration instance
config = TranslationConfig()