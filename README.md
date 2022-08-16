# DCFM
The official repo of the paper `Democracy Does Matter: Comprehensive Feature Mining for Co-Salient Object Detection`.

## Environment Requirement
create enviroment and intall as following:
`pip install -r requirements.txt`

## Data Format
  trainset: CoCo-SEG
  
  testset: CoCA, CoSOD3k, Cosal2015
  
  Put the [CoCo-SEG](https://drive.google.com/file/d/1GbA_WKvJm04Z1tR8pTSzBdYVQ75avg4f/view), [CoCA](http://zhaozhang.net/coca.html), [CoSOD3k](http://dpfan.net/CoSOD3K/) and [Cosal2015](https://drive.google.com/u/0/uc?id=1mmYpGx17t8WocdPcw2WKeuFpz6VHoZ6K&export=download) datasets to `DCFM/data` as the following structure:
  ```
  DCFM
     ├── other codes
     ├── ...
     │ 
     └── data
           
           ├── CoCo-SEG (CoCo-SEG's image files)
           ├── CoCA (CoCA's image files)
           ├── CoSOD3k (CoSOD3k's image files)
           └── Cosal2015 (Cosal2015's image files)
  ```
  
## Trained model

trained model can be downloaded from [papermodel](https://drive.google.com/file/d/1cfuq4eJoCwvFR9W1XOJX7Y0ttd8TGjlp/view?usp=sharing).

Run `test.py` for inference.

The evaluation tool please follow: https://github.com/zzhanghub/eval-co-sod


<!-- USAGE EXAMPLES -->
## Usage
Download pretrainde backbone model [VGG](https://drive.google.com/file/d/1Z1aAYXMyJ6txQ1Z9N7gtxLOIai4dxrXd/view?usp=sharing).

Run `train.py` for training.

## Prediction results
The co-saliency maps of DCFM can be found at [preds](https://drive.google.com/file/d/1wGeNHXFWVSyqvmL4NIUmEFdlHDovEtQR/view?usp=sharing).

## Reproduction
reproductions by myself on 2080Ti can be found at [reproduction1](https://drive.google.com/file/d/1vovii0RtYR_EC0Y2zxjY_cTWKWM3WaxP/view?usp=sharing) and [reproduction2](https://drive.google.com/file/d/1YPOKZ5kBtmZrCDhHpP3-w1GMVR5BfDoU/view?usp=sharing).

reprodution by myself on TITAN X can be found at [reproduction3](https://drive.google.com/file/d/1bnGFtRTYkVXqI2dcjeWFRDXnqqbUUBJr/view?usp=sharing).

## Others
The code is based on [GCoNet](https://github.com/fanq15/GCoNet).
I've added a validation part to help select the model for closer results. This validation part is based on [GCoNet_plus](https://github.com/ZhengPeng7/GCoNet_plus). You can try different evaluation metrics to select the model.
