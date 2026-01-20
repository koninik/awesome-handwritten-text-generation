# Handwritten Text Generation (HTG) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated collection of **methods**, **datasets**, and **evaluation tools** for *Handwritten Text Generation (HTG)*.

❗ Latest Update: 5 Dec 2025

🚧 Under Construction (updates might be needed)

This repo is a work based on the PhD monograph thesis [*Handwritten Text Generation with Diffusion Models: Beyond Visual Quality*](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A2006187&dswid=-3172) 

_Konstantina Nikolaidou_, 2025 - Advisors: _Marcus Liwicki_, [_George Retsinas_](https://github.com/georgeretsi), [_Giorgos Sfikas_](https://github.com/sfikas)  


---

# 📚 Contents
- [🧠 HTG Methods](#-htg-methods)
  - [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
  - [Diffusion Models](#diffusion-models)
  - [Autoregressive Models](#autoregressive--transformer-models-non-gan)
  - [Variational Autoencoders (VAEs)](#variational-autoencoders-vaes)
- [📝 Handwriting Datasets](#-handwriting-datasets)
  - [Modern Handwriting Datasets](#-modern-handwriting-datasets)
  - [Synthetic Handwriting Datasets](#-synthetic-handwriting-datasets)
- [📖 Surveys](#-surveys)
- [🧩 Evaluation](#-evaluation)

---
<details open>
<summary><h2>🧠 HTG Methods</h2></summary>


---

## Generative Adversarial Networks (GANs)


| Method / Model | Paper (full title) | Venue / Year | Granularity / Setting | Paper / Code Link |
|---|---|---|---|---|
| Alonso et al. | Adversarial Generation of Handwritten Text Images Conditioned on Sequences | ICDAR 2019 | offline, word-level | [📄 Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8977950) <br> [💻 Code (unofficial)](https://github.com/fractal2k/Handwriting-Synthesis)  |
| ScrabbleGAN | ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation | CVPR 2020 | offline, variable-length words | [📄 Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fogel_ScrabbleGAN_Semi-Supervised_Varying_Length_Handwritten_Text_Generation_CVPR_2020_paper.pdf) <br> [💻 Code](https://github.com/amzn/convolutional-handwriting-gan) |
| GANwriting | GANwriting: Content-Conditioned Generation of Styled Handwritten Word Images | ECCV 2020 | offline, word-level | [📄 Paper](https://link.springer.com/chapter/10.1007/978-3-030-58592-1_17) <br> [💻 Code](https://github.com/omni-us/research-GANwriting) |
| Davis et al. | Text and Style Conditioned GAN for the Generation of Offline-Handwriting Lines | BMVC 2020 | offline, line-level | [📄 Paper](https://www.bmva-archive.org.uk/bmvc/2020/assets/papers/0815.pdf) <br> [💻 Code](https://github.com/herobd/handwriting_line_generation) 
| GANwriting-Line | Content and Style Aware Generation of Text-Line Images for Handwriting Recognition | TPAMI 2021 | offline, line-level | [📄 Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9585646)  |
| SmartPatch | SmartPatch: Improving Handwritten Word Imitation with Patch Discriminators | ICDAR 2021 | offline, word-level | [📄 Paper](https://link.springer.com/chapter/10.1007/978-3-030-86549-8_18) <br> [💻 Code](https://github.com/MattAlexMiracle/SmartPatch) |
| HWT (Handwriting Transformers) | Handwriting Transformers | CVPR 2021 | offline, word/line-level | [📄 Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Bhunia_Handwriting_Transformers_ICCV_2021_paper.pdf) <br> [💻 Code](https://github.com/ankanbhunia/Handwriting-Transformers) 
| HiGAN | HiGAN: Handwriting Imitation Conditioned on Arbitrary-Length Texts and Disentangled Styles | AAAI 2021 | offline, arbitrary length | [📄 Paper](https://cdn.aaai.org/ojs/16917/16917-13-20411-1-2-20210518.pdf) <br> [💻 Code](https://github.com/ganji15/HiGAN) |
| JokerGAN | JokerGAN: Memory-Efficient Model for Handwritten Text Generation with Text Line Awareness | ACMM 2021 | offline, word-level | [📄 Paper](https://janzdenek.github.io/publication/jokergan/jokergan.pdf) |
| HiGAN+ | HiGAN+: Handwriting Imitation GAN with Disentangled Representations | ACM TOG 2022 | offline, word-level | [📄 Paper](https://dl.acm.org/doi/10.1145/3550070) <br> [💻 Code](https://github.com/ganji15/HiGANplus) |
| SLOGAN | SLOGAN: Handwriting Style Synthesis for Arbitrary-Length and Out-of-Vocabulary Text | IEEE Transactions on Neural Networks<br>and Learning Systems 2022 | offline, arbitrary-level | [📄 Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9722567)
| VATr | Handwritten Text Generation From Visual Archetypes | CVPR 2023 | offline, word-level | [📄 Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Pippi_Handwritten_Text_Generation_From_Visual_Archetypes_CVPR_2023_paper.pdf) <br> [💻 Code](https://github.com/aimagelab/VATr) |
| TextStyleBrush | TextStyleBrush: Transfer of Text Aesthetics from a Single Example | TPAMI 2023 | offline, in-the-wild, word-level | [📄 Paper](https://ai.meta.com/research/publications/textstylebrush-transfer-of-text-aesthetics-from-a-single-example/) |
| JokerGAN++ | Handwritten Text Generation with Character-specific Encoding for Style Imitation | ICDAR 2023 | offline, line-level | [📄 Paper](https://link.springer.com/chapter/10.1007/978-3-031-41679-8_18) |
| VATr++ | VATr++: Choose Your Words Wisely for Handwritten Text Generation | TPAMI 2024 | offline, word/line-level | [📄 Paper](https://ieeexplore.ieee.org/abstract/document/10716806) <br> [💻 Code](https://github.com/EDM-Research/VATr-pp)|
| — | — | — | — | — |

---

## Diffusion Models


| Method / Model | Paper (full title) | Venue / Year | Granularity / Setting | Paper / Code Link |
|---|---|---|---|---|
| CTIG-DM | Conditional Text Image Generation with Diffusion Models | CVPR 2023 | offline, scene-text, word-level, non-latent | [📄 Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Conditional_Text_Image_Generation_With_Diffusion_Models_CVPR_2023_paper.pdf) |
| WordStylist | WordStylist: Styled Verbatim Handwritten Text Generation with Latent Diffusion Models | ICDAR 2023 | offline, word-level, latent | [📄 Paper](https://link.springer.com/chapter/10.1007/978-3-031-41679-8_22) <br> [💻 Code](https://github.com/koninik/WordStylist)|
| DiffusionPen | DiffusionPen: Towards Controlling the Style of Handwritten Text Generation | ECCV 2024 | offline, in-the-wild, one-shot, few-shot, word (+line), latent | [📄 Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11492_ECCV_2024_paper.php) <br> [💻 Code](https://github.com/koninik/DiffusionPen) |
| One-DM | One-DM: One-Shot Diffusion Mimicker for Handwritten Text Generation | ECCV 2024 | offline, one-shot, non-latent, rendered content condition | [📄 Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07598.pdf) <br> [💻 Code](https://github.com/dailenson/One-DM)  |
| DiffWord | Word-Diffusion: Diffusion-Based Handwritten Text Word Image Generation | ICPR 2024 | offline, word-level | [📄 Paper](https://link.springer.com/chapter/10.1007/978-3-031-78495-8_4) |
| Brandenbusch | Semi-Supervised Adaptation of Diffusion Models for Handwritten Text Generation | arXiv 2024 | offline, word-level, few-shot, CFG | [📄 Paper](https://arxiv.org/abs/2412.15853) |
| Zero-Shot Paragraph DM | Zero-Shot Paragraph-level Handwriting Imitation with Latent Diffusion Models | IJCV 2025 | offline, paragraph-level | [📄 Paper](https://link.springer.com/article/10.1007/s11263-025-02525-0) <br> [💻 Code](https://github.com/M4rt1nM4yr/paragraph_handwriting_imitation_ldm)|
| DOG | Dual Orthogonal Guidance for Robust Diffusion-based Handwritten Text Generation |ICCV Workshops 2025 | diffusion guidance, offline, in-the-wild| [📄 Paper](https://openaccess.thecvf.com/content/ICCV2025W/P13N/html/Nikolaidou_Dual_Orthogonal_Guidance_for_Robust_Diffusion-based_Handwritten_Text_Generation_ICCVW_2025_paper.html) |

---

## Autoregressive Models


| Method / Model | Paper (full title) | Venue / Year | Granularity / Setting | Paper / Code Link |
|---|---|---|---|---|
| Emuru | Zero-Shot Styled Text Image Generation, but Make It Autoregressive | CVPR 2025 | font-based, synthetic, line-level, autoregressive | [📄 Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Pippi_Zero-Shot_Styled_Text_Image_Generation_but_Make_It_Autoregressive_CVPR_2025_paper.html) <br> [💻 Code](https://github.com/aimagelab/Emuru-autoregressive-text-img?tab=readme-ov-file) |
| Eruku | Zero-Shot Styled Text Image Generation, but Make It Autoregressive | WACV 2026 | font-based, synthetic, line-level, autoregressive | [📄 Paper](https://arxiv.org/pdf/2510.23240) <br> [💻 Code](https://aimagelab.github.io/Eruku/) |

---

## Variational Autoencoders (VAEs)

| Method / Model | Paper (full title) | Venue / Year | Granularity / Setting | Paper / Code Link |
|---|---|---|---|---|
| Cross-VAE | Modality Conversion of Handwritten Patterns by Cross Variational Autoencoders | ICDAR 2019 | online/offline conversion, character-level | [📄 Paper](https://ieeexplore.ieee.org/document/8978197) |
| DeepWriteSYN | DeepWriteSYN: On-line Handwriting Synthesis via Deep Short-Term Representations | AAAI 2021 | online, character/short-sequence | [📄 Paper](https://cdn.aaai.org/ojs/16139/16139-13-19633-1-2-20210518.pdf) |
| Emuru-VAE | Zero-Shot Styled Text Image Generation, but Make It Autoregressive | CVPR 2025 | font-based, writer/HTR VAE pre-training on synth lines| [📄 Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Pippi_Zero-Shot_Styled_Text_Image_Generation_but_Make_It_Autoregressive_CVPR_2025_paper.html) <br> [💻 Code](https://github.com/aimagelab/Emuru-autoregressive-text-img?tab=readme-ov-file)
</details>

<details open>
<summary><h2>📝 Handwriting Datasets</h2></summary>

### 📄 Modern Handwriting Datasets

| Dataset | Paper | Year | Description | Links |
|:--------|:------|:----:|:------------|:------|
| **IAM Handwriting Database** | *The IAM-database: an English sentence database for offline handwriting recognition* – Marti & Bunke | 2002 | 1.5K pages, 82K text lines, 115K words of unconstrained English handwriting from 657 writers. Standard benchmark for Latin-script HTR and HTG. | [📄 Paper](https://link.springer.com/article/10.1007/s100320200071)  [🗂️ Dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) |emuru
| **CVL Handwriting Database** | *CVL-DataBase: An Off-Line Database for Writer Retrieval, Writer Identification and Word Spotting* – Kleber et al. | 2013 | English/German handwritten texts from 310 writers, supporting writer identification, retrieval and word-level analysis. | [📄 Paper](https://ieeexplore.ieee.org/document/6628682)  [🗂️ Dataset](https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/) |
| **RIMES** | *RIMES evaluation campaign for handwritten mail processing* – Augustin et al. | 2006 | Large-scale French handwriting database collected from postal mail, with page- and line-level annotations for HTR. | [📄 Paper](https://ieeexplore.ieee.org/document/4032839)  [🗂️ Dataset](http://www.a2ialab.com/doku.php?id=rimes_database:start) |
| **GNHK** | *GNHK: a dataset for English handwriting in the wild* – Lee et al. | 2021 | 687 camera-captured pages with 39K text instances, 9K lines and 173K characters. Focused on in-the-wild handwritten text under real-world degradations. | [📄 Paper](https://link.springer.com/chapter/10.1007/978-3-030-86337-1_27)  [🗂️ Dataset](https://www.goodnotes.com/gnhk) |
| **Imgur5K** | *TextStyleBrush: Transfer of Text Aesthetics From a Single Example* – Krishnan et al. | 2023 | ~5K images and ~135K word instances of diverse “in-the-wild” handwriting sourced from Imgur, used to evaluate one-shot style transfer and robust HTG. | [📄 Paper](https://ai.meta.com/research/publications/textstylebrush-transfer-of-text-aesthetics-from-a-single-example/)  [🗂️ Dataset](https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset) |

---

### ✨ Synthetic Handwriting Datasets

| Dataset | Paper | Year | Description | Links |
|:--------|:------|:----:|:------------|:------|
| **IIIT-HWS** | *Generating Synthetic Data for Text Recognition* – Krishnan & Jawahar | 2016 | ~9M synthetic handwritten word images rendered with handwriting-style fonts and heavy augmentation, widely used for pretraining HTR/word-spotting models. | [📜 Paper](https://arxiv.org/pdf/1608.04224)  [📂 Dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-hws) |
| **Font²** | *Evaluating synthetic pre-training for handwriting processing tasks* – Pippi et al. | 2023 | Large-scale synthetic word corpus rendered with thousands of calligraphic/handwriting fonts, designed for supervised pretraining of style encoders and handwriting analysis. | [📜 Paper](https://doi.org/10.1016/j.patrec.2023.06.003)  [📂 Dataset](https://huggingface.co/datasets/blowing-up-groundhogs/font-square-v2) |

</details>

---

<details open>
<summary><h2>📖 Surveys</h2></summary>

- [Handwriting synthesis: classifications and techniques](https://link.springer.com/content/pdf/10.1007/s10032-014-0231-x.pdf)
- [Handwriting Generation and Synthesis: A Review](https://ieeexplore.ieee.org/document/9776932)
- [Data Augmentation for Offline Handwritten Text Recognition: A Systematic Literature Review](https://link.springer.com/article/10.1007/s42979-023-02583-6)
- [A survey of handwriting synthesis from 2019 to 2024: A comprehensive review](https://www.sciencedirect.com/science/article/pii/S0031320325000172)
</details>

---

<details open>
<summary><h2>🧩 Evaluation</h2></summary>

- _HWD: A Novel Evaluation Score for Styled Handwritten Text Generation_ - [arxiv](https://arxiv.org/abs/2310.20316) | [paper](https://papers.bmvc2023.org/0007.pdf) |  [code](https://github.com/aimagelab/HWD)
- _Rethinking HTG Evaluation: Bridging Generation and Recognition_ - [arxiv](https://arxiv.org/abs/2409.02683) | [paper](https://link.springer.com/chapter/10.1007/978-3-031-92089-9_12) |  [code](https://github.com/koninik/HTG_evaluation)
  
</details>

---

## 📄 Citation

If you find this work useful for your research, please cite:

```bibtex
@phdthesis{nikolaidou2025handwritten,
  title={Handwritten Text Generation with Diffusion Models: Beyond Visual Quality},
  author={Nikolaidou, Konstantina},
  year={2025},
  school={Lule{\aa} University of Technology}
}
```

---

## ⭐ StarGraph
[![Star History Chart](https://api.star-history.com/svg?repos=koninik/awesome-handwritten-text-generation&type=Timeline&theme=dark)](https://star-history.com/#koninik/awesome-handwritten-text-generation&Timeline&theme=dark)
