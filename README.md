# Repository for COMP550-F21 Project

## Datasets
Our final datasets for the polarity, articles and smokers task can be found in `/data/{rt-polaritydata,articles,smokers}/augmentation/`. Some preprocessing was done in `/scripts/prepare_text_data.py` and in `/scripts/prepare_medical_data.py`. All of our augmented datasets were run using the cluster and are thus hosted there. You can e-mail `cesare.spinoso-dipiano@mail.mcgill.ca` for further information about the augmented datasets.

## Data augmentation
You can find the data augmentation script in `/scripts/augment_data.py`. The augmentation script works per task so run `python augment_data.py -t polarity` for the polarity data augmentation.

## SVM classifier
You can find the script for the SVM classifier in `/scripts/train_models.py`. The script also works on a per task basis. Another thing to point out is that we initially ran our experiments with naive bayes, logistic regression and SVM but we only presented the SVM in our report for brevity and because we only needed the comparison of one BoW model. The logs for the training can be found under `/logs/training/{polarity,articles,smokers}_constant.log` (we use `constant` because we fiz the hyperparameters).

## RNN classifier
You can find the script for the RNN classifier in `/scripts/train_rnn_classif.py`. It also works on per task basis. The logs are found in `/logs/training_rnn_classif/{polarity,articles,smokers}.log`.

## Masker classifier Architecture
You can find the script for the masker classifier in `/scripts/train_highway_augmenter.py`. All logs can be found in `train_highway_augmenter.py`.

## Notebooks for results and discussions
- You can find the notebook for the results calculations in `/notebooks/ad_hoc_data_analytics.ipynb`
- You can find the notebook for the masking ratios and the graph in `/notebooks/log_plots.ipynb`
