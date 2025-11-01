import torch
import attention_moudle
import einops.layers.torch  as elt

class VideoRec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = attention_moudle.Encoder()
        self.trans_layer = torch.nn.Sequential(
            elt.Rearrange("b l d -> b d l"),
            torch.nn.AdaptiveAvgPool1d((1)),
            elt.Rearrange("b d 1 -> b d")
        )
        self.last_linear = torch.nn.Linear(312,5)

    def forward(self,image):
        img = self.encoder(image)
        img = self.trans_layer(img)

        img = torch.nn.Dropout(0.1)(img)
        logits = self.last_linear(img)
        return logits


if __name__ == '__main__':
    image = torch.randn(size=(3,20,63))
    VideoRec()(image)
