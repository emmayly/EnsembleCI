# EnsembleCI

## Installation

To install the necessary dependencies, please follow the instructions at [AutoGluon Installation Guide](https://auto.gluon.ai/stable/install.html).

## Usage

### Running `carbonForecast.py`
To run the carbon forecasting script, use the following command in `src`:

```bash
python3 carbonForecast.py <region> <d/l> <model> <l/t>
```
Example: `python3 carbonForecast.py CISO d GBM t`

`<region>`: 

US:
* `CISO`: California ISO
* `PJM`: Pennsylvania-Jersey-Maryland Interconnection
* `ERCO`: Electric Reliability Council of Texas
* `ISNE`: ISO New England
* `EPE`: El Paso Electric
* `MISO`: Midcontinent Independent System Operator

Europe:
* `DE`: Germany
* `SE`: Sweden
* `PL`: Poland
* `NL`: Netherlands
* `ES`: Spain

`<d/l>`: 
* `d`: direct_emission
* `l`: lifecycle_emission

`<model>`: 
* `GBM`: LightGBM
* `FASTAI`: neural network with FastAI backend （https://auto.gluon.ai/stable/_modules/autogluon/tabular/models/fastainn/tabular_nn_fastai.html#NNFastAiTabularModel）
* `CAT`: CatBoost
* `XGB`: XGBoost
* `AUTO`: Autogluon weightedEnsamble

`<l/t>` (optional)
* `l`: load existing model
* `t`: train new model
