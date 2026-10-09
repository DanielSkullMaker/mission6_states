"""Сравнение известных нейросетевых архитектур сегментации на той же
задаче и том же протоколе, что и `experiment_article4_unet_segmentation.py`
(Статья 4: «Анализ последовательной структуры смежных срезов медицинских
изображений методом свёрточных нейронных сетей»).

Зачем: базовая версия статьи использовала ОДНУ архитектуру (компактный
U-Net). Этот скрипт добавляет известные в литературе по (медицинской)
сегментации архитектуры — для ответа на вопрос "specific ли результат
(контекст соседних срезов помогает) для одной конкретной сети, или
воспроизводится на разных архитектурах":

  - Plain encoder-decoder (БЕЗ skip-connections) — ablation-baseline,
    показывающий ЦЕНУ отказа от skip-connections U-Net (Ronneberger et
    al., 2015) при прочих равных блоках.
  - Residual U-Net — заменяет обычные свёрточные блоки на остаточные
    (residual) блоки с BatchNorm (He et al., 2016 — остаточные связи;
    архитектурная комбинация с U-Net — Zhang et al., 2018, "Road
    Extraction by Deep Residual U-Net").
  - Attention U-Net — добавляет вентили внимания (attention gates) на
    skip-connections, подавляющие нерелевантные области перед
    конкатенацией с decoder-признаками (Oktay et al., 2018).
  - SegNet (Badrinarayanan et al., 2015) — ПРИНЦИПИАЛЬНО другой механизм
    восстановления разрешения: не skip-конкатенация и не transposed-conv,
    а unpooling по индексам, запомненным при max-pooling энкодера.
  - FCN-8s (Long et al., 2015) — исторический предшественник U-Net:
    грубая попиксельная карта на каждом масштабе декодируется через
    билинейный upsampling и СУММИРУЕТСЯ (не конкатенируется) с более
    точной картой следующего масштаба.
  - MultiResUNet (Ibtehaz & Rahman, 2020) — разработана специально для
    медицинской сегментации: MultiRes-блок аппроксимирует параллельные
    свёртки 3x3/5x5/7x7 цепочкой из трёх 3x3 (как в оригинальной
    статье) + остаточная связь; skip-пути проходят через ResPath —
    цепочку conv+shortcut блоков, сокращающую семантический разрыв
    энкодер/декодер перед конкатенацией.
  - DeepLabV3+-lite (Chen et al., 2018) — другая, не-U-Net парадигма:
    на последних уровнях энкодера pooling заменён на дилатированные
    (atrous) свёртки (сохраняют разрешение, расширяя рецептивное поле),
    контекст собирается модулем ASPP (параллельные atrous-свёртки
    разного масштаба расширения + global pooling), декодер использует
    ОДНО skip-соединение низкоуровневых признаков (не на каждом уровне,
    как в U-Net).

Методология: та же задача (бинарная сегментация опухоли), тот же честный
патиентский сплит (RANDOM_SEED=42, идентичная `split_patients`), те же
гиперпараметры (IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
MAX_TRAIN_PER_CLASS) и тот же state4_scripts-инспирированный блок
Conv-Conv (`conv_block`), что и в базовом U-Net — различаются ТОЛЬКО
архитектурные решения, перечисленные выше. Вычисления идут на `base.DEVICE`
(CUDA, если доступна — см. docstring базового модуля; на CPU ничего не
меняется, только устройство). Каждая архитектура обучается и оценивается
в ОДНОЙ конфигурации — "физические соседи" (K=N_CONTEXT, лучший результат
базового эксперимента), а не во всех трёх (K=1/meaningful/shuffled) —
вопрос "какая архитектура точнее сегментирует опухоль при наличии
контекста", а не повторная проверка H1/H2 (та уже сделана для базового
U-Net).

Запуск (после experiment_article4_unet_segmentation.py — использует тот
же датасет/маски):
    python experiments/experiment_article4_architectures.py
"""

import json
import time
from pathlib import Path
from typing import Tuple

import torch
from torch import nn

import experiment_article4_unet_segmentation as base

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "experiment_article4_architectures_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "experiment_article4_architectures_results.md"
BASE_UNET_REPORT_PATH = ARTIFACTS_DIR / "experiment_article4_unet_report.json"


# --------------------------------------------------------------------------
# Архитектуры
# --------------------------------------------------------------------------

class PlainEncoderDecoder(nn.Module):
    """U-Net БЕЗ skip-connections — encoder/decoder той же глубины и тех же
    размеров каналов, но decoder восстанавливает разрешение, не видя
    признаков энкодера напрямую. Ablation-baseline: показывает, какую
    часть качества даёт именно "пробрасывание" мелких деталей через
    skip-connections, а не глубина/ширина сети."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.enc1 = base.conv_block(in_channels, 32)
        self.enc2 = base.conv_block(32, 64)
        self.enc3 = base.conv_block(64, 128)
        self.bottleneck = base.conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = base.conv_block(128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = base.conv_block(64, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = base.conv_block(32, 32)

        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(self.up3(b))
        d2 = self.dec2(self.up2(d3))
        d1 = self.dec1(self.up1(d2))
        return self.out_conv(d1)


class ResConvBlock(nn.Module):
    """Остаточный блок (He et al., 2016): Conv-BN-ReLU-Conv-BN + проекция
    входа (1x1 свёртка, если меняется число каналов), сумма, ReLU."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResUNet(nn.Module):
    """U-Net с остаточными блоками вместо обычных Conv-Conv (Zhang et al.,
    2018) — та же энкодер/декодер-структура, что у базового U-Net, но
    каждый блок учится остатку (residual), не прямому отображению, и
    использует BatchNorm."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.enc1 = ResConvBlock(in_channels, 32)
        self.enc2 = ResConvBlock(32, 64)
        self.enc3 = ResConvBlock(64, 128)
        self.bottleneck = ResConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ResConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ResConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ResConvBlock(64, 32)

        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


class AttentionGate(nn.Module):
    """Вентиль внимания (Oktay et al., 2018): по сигналу от более грубого
    уровня decoder (g) вычисляет поканальную карту релевантности для
    skip-connection (x) и подавляет нерелевантные области ПЕРЕД
    конкатенацией — в отличие от обычного U-Net, где skip-connection
    передаётся без изменений."""

    def __init__(self, gate_c: int, skip_c: int, inter_c: int) -> None:
        super().__init__()
        self.Wg = nn.Conv2d(gate_c, inter_c, 1)
        self.bn_g = nn.BatchNorm2d(inter_c)
        self.Wx = nn.Conv2d(skip_c, inter_c, 1)
        self.bn_x = nn.BatchNorm2d(inter_c)
        self.psi = nn.Conv2d(inter_c, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        attn = self.sigmoid(self.psi(self.relu(self.bn_g(self.Wg(g)) + self.bn_x(self.Wx(x)))))
        return x * attn


class AttentionUNet(nn.Module):
    """U-Net с вентилями внимания на всех трёх skip-connections."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.enc1 = base.conv_block(in_channels, 32)
        self.enc2 = base.conv_block(32, 64)
        self.enc3 = base.conv_block(64, 128)
        self.bottleneck = base.conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = AttentionGate(gate_c=128, skip_c=128, inter_c=64)
        self.dec3 = base.conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att2 = AttentionGate(gate_c=64, skip_c=64, inter_c=32)
        self.dec2 = base.conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.att1 = AttentionGate(gate_c=32, skip_c=32, inter_c=16)
        self.dec1 = base.conv_block(64, 32)

        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        g3 = self.up3(b)
        d3 = self.dec3(torch.cat([g3, self.att3(g3, e3)], dim=1))
        g2 = self.up2(d3)
        d2 = self.dec2(torch.cat([g2, self.att2(g2, e2)], dim=1))
        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, self.att1(g1, e1)], dim=1))
        return self.out_conv(d1)


class SegNet(nn.Module):
    """SegNet (Badrinarayanan et al., 2015) — разрешение восстанавливается
    unpooling'ом по индексам, запомненным при max-pooling энкодера, а не
    skip-конкатенацией (U-Net) и не выученной transposed-свёрткой. Перед
    каждым unpooling число каналов проецируется (1x1 conv) до числа
    каналов, с которым был сделан соответствующий pooling — обязательное
    условие для `nn.MaxUnpool2d` (вход и indices должны быть одной формы)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.enc1 = base.conv_block(in_channels, 32)
        self.enc2 = base.conv_block(32, 64)
        self.enc3 = base.conv_block(64, 128)
        self.bottleneck = base.conv_block(128, 256)
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.reduce3 = nn.Conv2d(256, 128, 1)
        self.dec3 = base.conv_block(128, 128)
        self.reduce2 = nn.Conv2d(128, 64, 1)
        self.dec2 = base.conv_block(64, 64)
        self.reduce1 = nn.Conv2d(64, 32, 1)
        self.dec1 = base.conv_block(32, 32)

        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1, idx1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2, idx2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3, idx3 = self.pool(e3)
        b = self.bottleneck(p3)

        d3 = self.dec3(self.unpool(self.reduce3(b), idx3, output_size=e3.shape[-2:]))
        d2 = self.dec2(self.unpool(self.reduce2(d3), idx2, output_size=e2.shape[-2:]))
        d1 = self.dec1(self.unpool(self.reduce1(d2), idx1, output_size=e1.shape[-2:]))
        return self.out_conv(d1)


class FCN8s(nn.Module):
    """Упрощённая FCN-8s-подобная схема (Long et al., 2015) — исторический
    предшественник U-Net. В отличие от U-Net (конкатенация признаков на
    каждом уровне), здесь на каждом масштабе строится ГРУБАЯ попиксельная
    карта (1x1 conv в 1 канал), которая билинейно апсемплится до
    следующего, более точного масштаба и СУММИРУЕТСЯ с его картой — классическая
    схема "coarse-to-fine score fusion" оригинальной статьи, упрощённая
    (билинейный upsampling вместо обучаемых деконволюций с фиксированным
    шагом 2/4/8)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.enc1 = base.conv_block(in_channels, 32)
        self.enc2 = base.conv_block(32, 64)
        self.enc3 = base.conv_block(64, 128)
        self.bottleneck = base.conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.score_bottleneck = nn.Conv2d(256, 1, 1)
        self.score_e3 = nn.Conv2d(128, 1, 1)
        self.score_e2 = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        score = self.score_bottleneck(b)
        score = nn.functional.interpolate(score, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        score = score + self.score_e3(e3)
        score = nn.functional.interpolate(score, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        score = score + self.score_e2(e2)
        score = nn.functional.interpolate(score, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return score


class MultiResBlock(nn.Module):
    """MultiRes-блок (Ibtehaz & Rahman, 2020): три последовательные 3x3-свёртки
    аппроксимируют параллельные свёртки 3x3/5x5/7x7 (так же, как в
    оригинальной статье — ради экономии параметров), их выходы
    конкатенируются и складываются с остаточной проекцией входа (1x1 conv)."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        c1 = out_c // 3
        c2 = out_c // 3
        c3 = out_c - c1 - c2
        self.conv3 = nn.Conv2d(in_c, c1, 3, padding=1)
        self.conv5 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv7 = nn.Conv2d(c2, c3, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Conv2d(in_c, out_c, 1)
        self.bn_shortcut = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o1 = self.relu(self.conv3(x))
        o2 = self.relu(self.conv5(o1))
        o3 = self.relu(self.conv7(o2))
        out = self.bn(torch.cat([o1, o2, o3], dim=1))
        shortcut = self.bn_shortcut(self.shortcut(x))
        return self.relu(out + shortcut)


class ResPath(nn.Module):
    """ResPath (Ibtehaz & Rahman, 2020) — трансформирует skip-connection
    ПЕРЕД конкатенацией цепочкой из `length` conv+shortcut блоков, сокращая
    семантический разрыв между признаками энкодера и декодера. `length`
    уменьшается с глубиной уровня (как в оригинальной статье)."""

    def __init__(self, channels: int, length: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(channels, channels, 3, padding=1) for _ in range(length))
        self.shortcuts = nn.ModuleList(nn.Conv2d(channels, channels, 1) for _ in range(length))
        self.bns = nn.ModuleList(nn.BatchNorm2d(channels) for _ in range(length))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, shortcut, bn in zip(self.convs, self.shortcuts, self.bns):
            out = self.relu(conv(x))
            x = self.relu(bn(out + shortcut(x)))
        return x


class MultiResUNet(nn.Module):
    """U-Net со MultiRes-блоками вместо Conv-Conv и ResPath на всех
    skip-connections (Ibtehaz & Rahman, 2020) — архитектура, разработанная
    и показавшая результаты именно на медицинской (вкл. опухоли мозга)
    сегментации."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.enc1 = MultiResBlock(in_channels, 32)
        self.respath1 = ResPath(32, length=3)
        self.enc2 = MultiResBlock(32, 64)
        self.respath2 = ResPath(64, length=2)
        self.enc3 = MultiResBlock(64, 128)
        self.respath3 = ResPath(128, length=1)
        self.bottleneck = MultiResBlock(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = MultiResBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = MultiResBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = MultiResBlock(64, 32)

        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        s1 = self.respath1(e1)
        s2 = self.respath2(e2)
        s3 = self.respath3(e3)

        d3 = self.dec3(torch.cat([self.up3(b), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        return self.out_conv(d1)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (Chen et al., 2018) — параллельные
    atrous-свёртки с разным dilation (контекст на разных масштабах без
    потери разрешения через pooling) + global average pooling, результаты
    конкатенируются и проецируются обратно."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.atrous6 = nn.Conv2d(in_c, out_c, 3, padding=6, dilation=6)
        self.bn6 = nn.BatchNorm2d(out_c)
        self.atrous12 = nn.Conv2d(in_c, out_c, 3, padding=12, dilation=12)
        self.bn12 = nn.BatchNorm2d(out_c)
        self.atrous18 = nn.Conv2d(in_c, out_c, 3, padding=18, dilation=18)
        self.bn18 = nn.BatchNorm2d(out_c)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Conv2d(in_c, out_c, 1)
        self.project = nn.Conv2d(out_c * 5, out_c, 1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        global_feat = nn.functional.interpolate(
            self.global_conv(self.global_pool(x)), size=size, mode="bilinear", align_corners=False
        )
        feats = [
            self.relu(self.bn1(self.conv1(x))),
            self.relu(self.bn6(self.atrous6(x))),
            self.relu(self.bn12(self.atrous12(x))),
            self.relu(self.bn18(self.atrous18(x))),
            global_feat,
        ]
        return self.relu(self.bn(self.project(torch.cat(feats, dim=1))))


class DeepLabV3Plus(nn.Module):
    """Упрощённая DeepLabV3+ (Chen et al., 2018) — ДРУГАЯ, не-U-Net
    парадигма: на последних уровнях энкодера pooling заменён на
    дилатированные (atrous) свёртки (расширяют рецептивное поле БЕЗ потери
    разрешения — DeepLab-понятие "output stride"), контекст собирается
    модулем ASPP, декодер использует ОДНО skip-соединение низкоуровневых
    признаков (не на каждом уровне, как в U-Net)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.low = base.conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.mid = base.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.high1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=2, dilation=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=2, dilation=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.high2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=4, dilation=4), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=4, dilation=4), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.aspp = ASPP(256, 128)

        self.low_project = nn.Conv2d(32, 48, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(128 + 48, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.low(x)
        m = self.mid(self.pool1(low))
        h = self.high1(self.pool2(m))
        h = self.high2(h)
        h = self.aspp(h)

        h_up = nn.functional.interpolate(h, size=low.shape[-2:], mode="bilinear", align_corners=False)
        out = self.decoder(torch.cat([h_up, self.low_project(low)], dim=1))
        return self.out_conv(out)


class ResNet18UNetTransfer(nn.Module):
    """Transfer-learning бейзлайн (по требованию медицинского рецензента —
    medical_article_requirements.txt, п.15: "сравниться нужно с
    дообученной на нескольких сотнях снимков предобученной сетью, а не
    только с вариациями собственного метода"). Энкодер — ResNet18,
    предобученный на ImageNet (torchvision), декодер — U-Net-стиль со
    skip-connections на всех 4 масштабах ResNet (stem/layer1/layer2/layer3),
    КАК И остальные архитектуры в этом файле обучается с нуля ТОЛЬКО в
    decoder-части — encoder стартует с предобученных весов и дообучается
    (fine-tuning) на том же протоколе/бюджете эпох, что и все остальные
    модели, без специальной донастройки под максимальную точность (эта
    задача — "добавить честный literature-standard бейзлайн", НЕ "выжать
    максимум из transfer learning").

    Адаптация входных каналов: ImageNet-веса рассчитаны на 3-канальный
    RGB-вход, у нас 1 (K=1) или 5 (K=5) каналов — первая свёртка (conv1)
    пересоздана под фактическое число каналов, её веса инициализированы
    усреднением предобученных RGB-весов по каналу и тиражированием на
    нужное число каналов (стандартный приём адаптации ImageNet-весов к
    grayscale/multi-channel входу, не теряющий претрен для остальных слоёв).

    Вход НЕ нормализуется под ImageNet-статистику (mean/std по каналам) —
    намеренно, чтобы использовать ТОТ ЖЕ пайплайн предобработки ([0,1]
    по интенсивности), что и все остальные модели в сравнении; это
    методологический компромисс (чуть хуже откалиброванный transfer
    learning), сделанный ради честности сравнения "при прочих равных"."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        old_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            avg_weight = old_conv1.weight.mean(dim=1, keepdim=True)
            new_conv1.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1))

        self.stem = nn.Sequential(new_conv1, resnet.bn1, resnet.relu)  # 96 -> 48, 64ch
        self.pool = resnet.maxpool  # 48 -> 24
        self.layer1 = resnet.layer1  # 24, 64ch
        self.layer2 = resnet.layer2  # 12, 128ch
        self.layer3 = resnet.layer3  # 6, 256ch
        self.layer4 = resnet.layer4  # 3, 512ch

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = base.conv_block(256 + 256, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = base.conv_block(128 + 128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = base.conv_block(64 + 64, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = base.conv_block(64 + 64, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = base.conv_block(32, 32)

        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        p = self.pool(s0)
        s1 = self.layer1(p)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        b = self.layer4(s3)

        d4 = self.dec4(torch.cat([self.up4(b), s3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s0], dim=1))
        d0 = self.dec0(self.up0(d1))
        return self.out_conv(d0)


ARCHITECTURES = {
    "plain_no_skip": PlainEncoderDecoder,
    "res_unet": ResUNet,
    "attention_unet": AttentionUNet,
    "segnet": SegNet,
    "fcn8s": FCN8s,
    "multiresunet": MultiResUNet,
    "deeplabv3plus": DeepLabV3Plus,
    "resnet18_unet_transfer": ResNet18UNetTransfer,
}


def run_architecture(name: str, model_factory, train_groups, train_centers, test_groups, test_centers) -> dict:
    print(f"\n--- Архитектура: {name} ---")
    model, fit_seconds = base.train_model(
        train_groups, train_centers, n_context=base.N_CONTEXT, model_factory=model_factory,
    )
    result = base.evaluate_model(
        model, base.SegStackDataset(test_groups, test_centers, n_context=base.N_CONTEXT),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"   {name}: mean_dice={result['mean_dice']:.4f}  median_dice={result['median_dice']:.4f}  "
        f"fit={fit_seconds:.1f}s  n_params={n_params}"
    )
    result["fit_seconds"] = fit_seconds
    result["n_params"] = n_params
    return result


def main() -> None:
    start = time.perf_counter()
    print("=== Статья 4: сравнение архитектур сегментации (K=5, физические соседи) ===")

    rows = base.load_metadata()
    train_patient_ids, test_patient_ids = base.split_patients(rows)
    train_groups = base.group_by_patient(rows, train_patient_ids)
    test_groups = base.group_by_patient(rows, test_patient_ids)

    train_centers = base.build_center_list(train_groups, base.MAX_TRAIN_PER_CLASS, base.RANDOM_SEED)
    test_centers = base.all_centers(test_groups)
    print(f"n_train={len(train_centers)}, n_test={len(test_centers)} (идентично базовому эксперименту).")

    results = {}
    for name, factory in ARCHITECTURES.items():
        results[name] = run_architecture(name, factory, train_groups, train_centers, test_groups, test_centers)

    # Базовый U-Net — без переобучения, читаем уже полученный результат.
    base_report = json.loads(BASE_UNET_REPORT_PATH.read_text(encoding="utf-8"))
    unet_result = dict(base_report["results"]["meaningful_neighbors"])
    unet_result["fit_seconds"] = base_report["fit_seconds_kn"]
    results["unet_baseline"] = unet_result

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "random_seed": base.RANDOM_SEED,
        "n_context": base.N_CONTEXT,
        "img_size": base.IMG_SIZE,
        "epochs": base.EPOCHS,
        "n_train": len(train_centers),
        "results": results,
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    name_ru = {
        "unet_baseline": "U-Net (базовый)",
        "plain_no_skip": "Plain encoder-decoder (без skip)",
        "res_unet": "Residual U-Net",
        "attention_unet": "Attention U-Net",
        "segnet": "SegNet",
        "fcn8s": "FCN-8s",
        "multiresunet": "MultiResUNet",
        "deeplabv3plus": "DeepLabV3+ (lite)",
        "resnet18_unet_transfer": "ResNet18-UNet (transfer learning)",
    }
    order = [
        "unet_baseline", "plain_no_skip", "res_unet", "attention_unet",
        "segnet", "fcn8s", "multiresunet", "deeplabv3plus", "resnet18_unet_transfer",
    ]
    lines = [
        "# Статья 4 — сравнение архитектур сегментации",
        "",
        f"Конфигурация K=5 (физические соседи) для всех архитектур, "
        f"идентичные гиперпараметры и train/test сплит. Устройство "
        f"вычислений: {base.DEVICE}"
        + (f" ({torch.cuda.get_device_name(0)})." if base.DEVICE.type == "cuda" else "."),
        "",
        "mean_dice (с опухолью) — стратифицированная метрика на срезах с "
        "has_tumor=True ТОЛЬКО (см. project_article4_unet_segmentation: "
        "усреднение по всем срезам смешивает содержательную сегментацию с "
        "тривиальным распознаванием пустого среза).",
        "",
        "| Архитектура | Mean Dice (все) | Mean Dice (с опухолью) | Median Dice | Pixel F1 | Detection F1 | Detection recall CI95 | Параметров | Время обучения |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key in order:
        r = results[key]
        n_params = r.get("n_params", "—")
        if isinstance(n_params, int):
            n_params = f"{n_params:,}".replace(",", " ")
        dice_with_tumor = r.get("mean_dice_with_tumor")
        dice_with_tumor_str = f"{dice_with_tumor:.4f}" if dice_with_tumor is not None else "—"
        pixel_f1 = r.get("pixel_f1")
        pixel_f1_str = f"{pixel_f1:.4f}" if pixel_f1 is not None else "—"
        detection_f1 = r.get("slice_detection_f1")
        detection_f1_str = f"{detection_f1:.4f}" if detection_f1 is not None else "—"
        ci = r.get("slice_detection_recall_ci95")
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci is not None else "—"
        lines.append(
            f"| {name_ru[key]} | {r['mean_dice']:.4f} | {dice_with_tumor_str} | {r['median_dice']:.4f} | "
            f"{pixel_f1_str} | {detection_f1_str} | {ci_str} | {n_params} | {r['fit_seconds']:.0f}s |"
        )
    lines += [
        "",
        "Pixel F1 — Precision/Recall/F1, агрегированные по сумме TP/FP/FN "
        "ПИКСЕЛЕЙ всего тест-сета (не среднее Dice по срезам — другое число). "
        "Detection F1 — F1 бинарной классификации среза (предсказан ли хотя "
        "бы один пиксель опухоли) против истинного has_tumor. Detection "
        "recall CI95 — 95%-доверительный интервал Уилсона для detection "
        "recall (medical_article_requirements.txt, п.3).",
    ]
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown-отчёт: {REPORT_MD_PATH}")

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
