# NN libraries
import torch
from torch import nn
from TorchCRF import CRF

# Local library
from submodules import PositionalEncoder, MDCSA


class MDCSACRF(nn.Module):
    
    def __init__(
        self, rssi_size: int, hidden_size: int, T: int, loc_size: int, accl_size: int = None, 
        dropout: float = 0.15, with_binary: bool = True
    ):
        """
        rssi_size: Number of features for RSSI (#AccessPoint)
        accl_size: Number of features for Accel (#channel)
        T: Number of time steps
        hidden_size: Dimension of the embedding
        loc_size: Location classes
        dropout: Dropout probability
        """
        super(MDCSACRF, self).__init__()
                
        self.T = T
        self.loc_size = loc_size
        self.rssi_size = rssi_size
        self.accl_size = accl_size
        self.hidden_size = hidden_size
        self.with_binary = with_binary
        
        # RSSI selection and embedding layer
        self.rssi_pe = PositionalEncoder(hidden_size)
        self.rssi_encoder = nn.Linear(in_features=rssi_size, out_features=hidden_size)
        # ACCL selection and embedding layer
        if self.accl_size is not None:
            self.accl_pe = PositionalEncoder(hidden_size)
            self.accl_encoder = nn.Linear(in_features=accl_size, out_features=hidden_size)
        # Multihead Dual Self-Attention Block
        self.mdsca = MDCSA(hidden_size, 4*hidden_size, dropout)
        self.fc_dropout = nn.Dropout(dropout*2)  
        # Sequential classification constraining layer
        self.hidden2loc = nn.Linear(in_features=hidden_size, out_features=loc_size)
        if self.with_binary:
            self.hidden2bin = nn.Linear(in_features=hidden_size, out_features=1)
        self.crf = CRF(num_tags=loc_size, batch_first=True)
        
    def _get_location_embeddings(self, rssi: torch.Tensor, accl: torch.Tensor = None):
        """
        rssi: [Batch, T, rssi_size]
        accl: [Batch, T, accl_size]
        """
        # RSSI input selection, locality enhancement
        rssi_embeds = self.rssi_pe(self.rssi_encoder(rssi))
        # ACCL input selection, locality enhancement
        if accl is not None:
            accl_embeds = self.accl_pe(self.accl_encoder(accl))
        else:
            accl_embeds = None
        # RSSI enrichment with ACCL and processing past local enhancement
        embeds = self.mdsca(rssi_embeds, accl_embeds, mask=None)
        return self.fc_dropout(embeds)
    
    def forward(self, rssi: torch.Tensor, accl: torch.Tensor = None):
        """
        rssi: [Batch, T, rssi_size]
        accl: [Batch, T, accl_size]
        """
        embeds = self._get_location_embeddings(rssi, accl)
        # Sequential Classification
        location_feats = self.hidden2loc(embeds)
        locations = self.crf.decode(location_feats)
        return torch.tensor(locations, dtype=torch.int64, device=rssi.device)
    
    def calculate_loss(
        self, rssi: torch.Tensor, locations: torch.LongTensor, accl: torch.Tensor = None, 
        interested_value: int = None
    ):
        """
        rssi: [Batch, T, rssi_size]
        locations: [Batch, T, loc_size]
        accl: [Batch, T, accl_size]
        with_binary: True | False
        """
        embeds = self._get_location_embeddings(rssi, accl)
        # Sequential Classification
        location_feats = self.hidden2loc(embeds)
        if self.with_binary:
            binary_feats = self.hidden2bin(embeds)
            binary_loss = nn.BCEWithLogitsLoss(reduction='mean')
            binary_target = (locations == interested_value).type(torch.float)
            loss = -self.crf(location_feats, locations, reduction='mean') + binary_loss(binary_feats.squeeze(-1), binary_target)
        else:
            loss = -self.crf(location_feats, locations, reduction='mean')
        return loss