# DemAEndGAN --- Cross-modal hourly building energy demand prediction with Autoencoders and Generative Adversarial Networks

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Citation](#citation)
- [Special thanks](#thanks)
- [License](#license)

## Description
This repository contains the code and resources for demAEndGAN architecture, which focuses on two main areas:
- Disentanglement for early building design images using geometry autoencoder.
- Hourly demand generation based on multi-conditional GAN.

## Installation
1. Clone the repository.
2. Navigate to the project directory.
3. Create a virtual environment with Python >= 3.9.
4. Install dependencies:
    ``` pip install -r .\pytorch_requirements.txt. ```
    ``` pip install -r .\others_requirements.txt. ```

## Data
Data may be granted upon reasonable requests.

## Usage    
- Geometry autoencoder [geometry_autoencoder](geometry_autoencoder)
    - The main python files to call start with prefix "_main...":
        - To run autoencoder training [_main_training](geometry_autoencoder/_main_training.py)
        - To run post-processing encoding latent representations [_main_encoding](geometry_autoencoder/_main_encoding.py)
        - To run post-processing traversing latent spaces [_main_traversing](geometry_autoencoder/_main_traversing.py)
        - To generate reconstructed images [_main_reconstructing](geometry_autoencoder/_main_reconstructing.py)
    - Examples of batch scripts can be found in:
        - Training batch scripts [_batch_training](geometry_autoencoder/_batch_training/)
        - Processing batch scripts [_batch_processing](geometry_autoencoder/_batch_processing/)
    - Data should be put in a folder, which by default, is defined as "_data". Custom folder name for data can be done, but with explicit declaration to the batch scripts.
    - Trained models as used in this study are saved under [trained_model_](geometry_autoencoder/trained_model_/)

- Time series generator [timeseries_generator](timeseries_generator)
    - The main python file to initiate is [main.py](timeseries_generator/gan/main.py)
        - Training and test mode are defined in the parser command.
    - It is recommended to use batch scripts to perform instructions with examples: [_batch_scripts](timeseries_generator/_batch_scripts/)
    - Data should be put in a folder, which by default, is defined as "_data". Custom folder name for data can be done, but with explicit declaration to the batch scripts.
    - Trained models as used in this study are saved under [trained_model_](timeseries_generator/trained_model_/)

## Citation
### IEEE: 
```text
B. Bernadino and C. Waibel, ‘DemAEndGAN: Cross-modal hourly building energy demand prediction with autoencoders and generative adversarial networks’, Energy and Buildings, vol. 360, p. 117334, Jun. 2026, doi: 10.1016/j.enbuild.2026.117334.
```

### BibTeX:
```bibtex
@article{BERNADINO_2026117334,
	title = {{DemAEndGAN}: {Cross}-modal hourly building energy demand prediction with autoencoders and generative adversarial networks},
	volume = {360},
	issn = {0378-7788},
	shorttitle = {{DemAEndGAN}},
	url = {https://www.sciencedirect.com/science/article/pii/S0378778826003944},
	doi = {https://doi.org/10.1016/j.enbuild.2026.117334},
	language = {en},
	urldate = {2026-03-24},
	journal = {Energy and Buildings},
	author = {Bernadino, Bernadino and Waibel, Christoph},
	year = {2026},
	keywords = {Surrogate model, Data-driven, Building energy modelling (BEM), Generative artificial intelligence (GenAI), Hourly demand prediction, Stochastic urban building energy modelling (UBEM)},
	pages = {117334},
}
```

## Thanks
Special thanks are given to the authors of these repositories, which demAEndGAN is adapted from:
- S. Huajie, et al., ControlVAE https://github.com/HuajieShao/ControlVAE-ICML2020
- L. Zinan, et al., DoppelGANger https://github.com/fjxmlzn/DoppelGANger

## License
MIT License. See the [LICENSE](LICENSE) file for details.