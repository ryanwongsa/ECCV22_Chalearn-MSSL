

# Downloads


Download the pretrained models and preprocessed data: [download](https://drive.google.com/file/d/1FN0t3H5bAB6fL8lsjNPonSsr_fUanHR4/view?usp=sharing)

Extract to base repo directory


# Prepare docker

```bash
docker build -t chalearn .
docker run -it --gpus all -v <REPO_PATH>:/chalearn/ chalearn /bin/bash
cd /charlearn/
```

# (Optional) Prepare lmdb format of data

The test data lmdb data is provided in the above download `processed` folder, alternatively it can be recreated with the follow script:

```
python scripts/track1_lmdb_creator.py \
  --dest_dir processed/test/lmdb_videos/test \
  --dataset_dir <VIDEO_TEST_DIR> \
  --bbox_dir  chalearn_mssl/data/test_bboxes.json
```
This should be used also to create the train lmdb data and validation lmdb data replacing `test` with `train` and `valid` respectively.


# Inference

```bash
python main_results_creator_v2.py --lmdb_dir $(lmdb_dir) --checkpoint_dir $(checkpoint_dir) --save_dir $(save_dir)
```

Example:
```bash
python main_results_creator_v2.py --lmdb_dir chalearn_mssl/processed/test/lmdb_videos/test --checkpoint_dir chalearn_mssl/submissions/7fold_neg0/valinc_maxfix_neg0_fold0/ckpts/best_checkpoint_80_0.2793.pt --save_dir chalearn_results
```

Repeating for each checkpoint in the submissions folder. (NOTE: models trained with fold3 (`*_fold3`) were ignored during online test submissions)
This will create a `.pkl` file in the `save_dir` to be used for the ensemble.

Submssions trained with folder prefix `5fold_` were trained only on training dataset while `7fold_` were trained with a mixture of training and validation data.


# Full ensemble models:
```
checkpoint_dir = chalearn_mssl/submissions/5fold_neg01/maxfix_neg01_fold4/ckpts/best_checkpoint_155_0.4848.pt
checkpoint_dir = chalearn_mssl/submissions/5fold_neg05/maxfix_neg05_fold4/ckpts/best_checkpoint_180_0.4932.pt
checkpoint_dir = chalearn_mssl/submissions/5fold_neg0/maxfix_neg0_fold4/ckpts/best_checkpoint_120_0.3233.pt
checkpoint_dir = chalearn_mssl/submissions/5fold_neg0/maxfix_neg0_fold1/ckpts/best_checkpoint_145_0.3379.pt
checkpoint_dir = chalearn_mssl/submissions/5fold_neg01/maxfix_neg01_fold1/ckpts/best_checkpoint_125_0.4596.pt
checkpoint_dir = chalearn_mssl/submissions/5fold_neg05/maxfix_neg05_fold1/ckpts/best_checkpoint_140_0.4892.pt
checkpoint_dir = chalearn_mssl/submissions/5fold_neg0/maxfix_neg0_fold0/ckpts/best_checkpoint_125_0.2309.pt
checkpoint_dir = chalearn_mssl/submissions/5fold_neg01/maxfix_neg01_fold0/ckpts/best_checkpoint_190_0.3855.pt
checkpoint_dir = chalearn_mssl/submissions/5fold_neg05/maxfix_neg05_fold0/ckpts/best_checkpoint_160_0.4257.pt
checkpoint_dir = chalearn_mssl/submissions/5fold_neg0/maxfix_neg0_fold2/ckpts/best_checkpoint_110_0.3898.pt
checkpoint_dir = chalearn_mssl/submissions/5fold_neg01/maxfix_neg01_fold2/ckpts/best_checkpoint_120_0.5180.pt
checkpoint_dir = chalearn_mssl/submissions/5fold_neg05/maxfix_neg05_fold2/ckpts/best_checkpoint_65_0.5259.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg0/valinc_maxfix_neg0_fold0/ckpts/best_checkpoint_80_0.2793.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg01/valinc_maxfix_neg01_fold0/ckpts/best_checkpoint_165_0.4602.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg05/valinc_maxfix_neg05_fold0/ckpts/best_checkpoint_145_0.4727.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg05/valinc_maxfix_neg05_fold1/ckpts/best_checkpoint_55_0.5041.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg01/valinc_maxfix_neg01_fold1/ckpts/best_checkpoint_185_0.5075.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg0/valinc_maxfix_neg0_fold1/ckpts/best_checkpoint_165_0.3156.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg05/valinc_maxfix_neg05_fold2/ckpts/best_checkpoint_140_0.6750.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg01/valinc_maxfix_neg01_fold2/ckpts/best_checkpoint_140_0.6392.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg0/valinc_maxfix_neg0_fold2/ckpts/best_checkpoint_125_0.4336.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg05/valinc_maxfix_neg05_fold4/ckpts/best_checkpoint_160_0.5983.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg01/valinc_maxfix_neg01_fold4/ckpts/best_checkpoint_200_0.5447.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg0/valinc_maxfix_neg0_fold4/ckpts/best_checkpoint_100_0.3466.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg05/valinc_maxfix_neg05_fold5/ckpts/best_checkpoint_110_0.5096.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg01/valinc_maxfix_neg01_fold5/ckpts/best_checkpoint_95_0.4951.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg0/valinc_maxfix_neg0_fold5/ckpts/best_checkpoint_105_0.2947.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg0/valinc_maxfix_neg0_fold6/ckpts/best_checkpoint_90_0.3334.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg01/valinc_maxfix_neg01_fold6/ckpts/best_checkpoint_185_0.5421.pt
checkpoint_dir = chalearn_mssl/submissions/7fold_neg05/valinc_maxfix_neg05_fold6/ckpts/best_checkpoint_90_0.5404.pt

python main_results_creator_v2.py --lmdb_dir chalearn_mssl/processed/test/lmdb_videos/test --checkpoint_dir $(checkpoint_dir) --save_dir chalearn_results
```



# Ensemble

```python
python solution_test_creator_v2.py
```

Takes all the pickle files in `chalearn_results/test` and ensembles the results, saving in a `predictions.pkl` file.




# (Optional) Training

```
python main.py --config config/example_config.py
```

For the submission each of the folds (`original_split` [train data only] and `new_split` [train+validation data]) were trained 3 times each with a different `neg_prob` [0.0, 0.1, 0.5].
Therefore in the config file the following parameters were changed:
`csv_dir` (location of fold csv directory), `vol_path` (location of lmdb dataset directory), `neg_prob` (probability of selecting other parts of the video sequence)

NOTE: `chalearn_mssl/data/MSSL_TRAIN_SET_GT.pkl` needs to be added to the relevant directory provided by the challange dataset hosts.
