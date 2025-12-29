import torch
import torch.nn as nn

class NeuralNavigator(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=16, hidden_dim=64):
        super(NeuralNavigator, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.vision_fc = nn.Linear(64*16*16, hidden_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.decoder_lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, images, text_indices):
        img_feats = self.vision_fc(self.cnn(images))
        
        _, (text_hidden, _) = self.text_rnn(self.embedding(text_indices))
        text_feats = text_hidden.squeeze(0)

        fused = torch.cat((img_feats, text_feats), dim=1)
        
        decoder_input = fused.unsqueeze(1).repeat(1, 10, 1)
        lstm_out, _ = self.decoder_lstm(decoder_input)
        

        return self.output_layer(lstm_out)
