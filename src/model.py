"""U-Net (implementación propia) con attention gates opcionales (Oktay et al., 2018)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class AttentionGate(nn.Module):
    """Pesa cada píxel del skip (x) con un valor 0–1 calculado a partir del decoder (g)."""

    def __init__(self, ch, inter):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(ch, inter, 1, bias=False), nn.BatchNorm2d(inter))
        self.w_x = nn.Sequential(nn.Conv2d(ch, inter, 1, bias=False), nn.BatchNorm2d(inter))
        self.psi = nn.Sequential(nn.Conv2d(inter, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, x, g):
        alpha = self.psi(F.relu(self.w_g(g) + self.w_x(x)))      # (B, 1, H, W): 0 = apagar, 1 = dejar pasar
        return x * alpha


class UNet(nn.Module):
    """Entrada (B, 2, H, W) con VV y VH normalizados; salida (B, 1, H, W) en logits. H y W múltiplos de 16."""

    def __init__(self, in_channels=2, out_channels=1, features=(32, 64, 128, 256), attention=False):
        super().__init__()
        self.down = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.down.append(double_conv(ch, f))
            ch = f
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(ch, ch * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for f in reversed(features):
            self.up.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.dec.append(double_conv(f * 2, f))
        self.gates = nn.ModuleList([AttentionGate(f, f // 2) for f in reversed(features)]) if attention else None
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i, (up, dec, skip) in enumerate(zip(self.up, self.dec, reversed(skips))):
            x = up(x)
            if self.gates is not None:
                skip = self.gates[i](skip, x)      # filtra el skip con el contexto del decoder
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        return self.head(x)
