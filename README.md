# PaddleOCR-VL-For-Manga

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/spaces/jzhang533/paddleocr-vl-for-manga-demo" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DEMO-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/jzhang533/PaddleOCR-VL-For-Manga" target="_blank" style="margin: 2px;">
    <img alt="Github" src="https://img.shields.io/badge/GitHub-Repository-000?logo=github&color=0000FF" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## Model Description

PaddleOCR-VL-For-Manga is an OCR model enhanced for Japanese manga text recognition. It is fine-tuned from [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) and achieves much higher accuracy on manga speech bubbles and stylized fonts.

This model was fine-tuned on a combination of the [Manga109-s dataset](http://www.manga109.org/) and 1.5 million synthetic data samples. It showcases the potential of Supervised Fine-Tuning (SFT) to create highly accurate, domain-specific VLMs for OCR tasks from a powerful, general-purpose base like [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL), which supports 109 languages.

This project serves as a practical guide for developers looking to build their own custom OCR solutions. You can find the training code at the [Github Repository](https://github.com/jzhang533/PaddleOCR-VL-For-Manga), a tutorial is coming soon.

## Performance

The model achieves a **70% full-sentence accuracy** on a test set of Manga109-s crops (representing a 10% split of the dataset). For comparison, the original PaddleOCR-VL on the same test dataset achieves 27% full sentence accuracy.

Common errors involve discrepancies between visually similar characters that are often used interchangeably, such as:

- `！？` vs. `!?` (Full-width vs. half-width punctuation)
- `ＯＫ` vs. `ok` (Full-width vs. half-width letters)
- `１２０５` vs. `1205` (Full-width vs. half-width numbers)
- “人” (U+4EBA) vs. “⼈” (U+2F08) (Standard CJK Unified Ideograph vs. CJK Radical)

The prevalence of these character types highlights a limitation of standard metrics like Character Error Rate (CER). These metrics may not fully capture the model's practical accuracy, as they penalize semantically equivalent variations that are common in stylized text.

## Examples

| # | Image | Prediction |
|---|---|---|
| 1 | ![ex1](examples/01.png) | 心拍呼吸正常値<br>お人よし度過剰値...<br>間違いなく<br>パパッ...!<br>生存確認っ...! |
| 2 | ![ex2](examples/02.png) | あとは『メルニィ<br>宇宙鉄道』とか<br>『TipTap』とか<br>全部その人が<br>考えたらしい |
| 3 | ![ex3](examples/03.png) | ★コミックス20巻1月4日(土)発売〟TVアニメ1月11日(土)放送開始!! |
| 4 | ![ex4](examples/04.png) | 我々魔女協会が<br>長年追い続ける<br>最大の敵<br>ウロロが「王の魔法」なら<br>あれは世界を削り変える<br>「神の魔法」 |
| 5 | ![ex5](examples/05.png) | 天弓の動きについてくだけじゃ勝てねぇ…！ |

## How to Use

You can use this model with the [transformers](https://github.com/huggingface/transformers), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), or any library that supports [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) to perform OCR on manga images. The model architecture and weights layout are identical to the base model.

If your application involves documents with structured layouts, you can use your fine-tuned OCR model in conjunction with [PP-DocLayoutV2](https://huggingface.co/PaddlePaddle/PaddleOCR-VL/tree/main/PP-DocLayoutV2/) for layout analysis. However, for manga, the reading order and layout are quite different.

## Training Details

- **Base Model**: [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
- **Dataset**:
  - [Manga109-s](http://www.manga109.org/): 0.1 million randomly sampled text-region crops (not full pages) were used for training (90% split); the remaining 10% crops were used for testing.
  - Synthetic Data: 1.5 million generated samples.
- **Training Frameworks**:
  - [transformers](https://github.com/huggingface/transformers) and [trl](https://github.com/huggingface/trl)
- **Alternatives for SFT**:
  - [ERNIEKit](https://github.com/PaddlePaddle/ERNIE)
  - [ms-swift](https://github.com/modelscope/swift)

## Acknowledgements

- [Manga109-s](http://www.manga109.org/) dataset, which provided the manga text-region crops used for training and evaluation.
- [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL), the base Visual Language Model on which this model is fine-tuned.
- [manga-ocr](https://github.com/kha-white/manga-ocr), used in this project for data processing and synthetic data generation; it also inspired practical workflows and evaluation considerations for manga OCR.

## License

This model is licensed under the **Apache 2.0** license.
