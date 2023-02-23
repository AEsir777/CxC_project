# CxC_project
## Data Exploration
- checking shape of the dataset (relatively large)
```{python}
(497166, 50)
```
- notice that the label is binary -> check if imbalanced
```{python}
False: 479912
True: 17254
```
- basic correlation graph drawn  
![image](https://user-images.githubusercontent.com/77596290/220818828-9311f9d1-19d0-405f-9d1d-2ea7bf29fefd.png)
```{python}
print(correlations['feat_PSI']['feat_DSSP_H'])    -0.6441074214031565
print(correlations['feat_DSSP_9']['feat_BBSASA'])  0.5859295090759643
print(correlations['feat_DSSP_7']['feat_DSSP_11'])  0.554012707812967
```
- find null values
```{python}
'annotation_atomrec'
```

## model selection (10 fold 3 repetition cross validaton)
- first under and oversample the dataset by a random amount
- then use some models to compare f1, roc_auc, pr_auc  

| model | logistic regression | XGBoost | *Random Forest* |  Neural Network(too slow) | LightGBM | Voting Classifier |
| -------- | ------- | ------- | ------- | ------- |  ------- |  ------- | 
| f1 | 0.68414 | 0.852539 | _0.8700565_ | NA |  0.772295  |0.860123 |
| roc_auc | 0.85543 |0.956948 | _0.9664487_ | 0.832424 | 0.92824222 |0.9581390 |
| pr_auc | NA | 0.893542 | _0.90847_ | NA | 0.834910 |0.90321|
- Random Forest performs the best so we select **Random Forest**  as the prediction model

## preprocessing
### NA values

### scaling

### encoding categorical data

### check outliers

### deal with imbalanced data

## model tunning
```{python}
imblanced = {false:1, true: 1.2}
```
| parameters | class_weight=imbalanced, bootstrap=False | class_weight="balanced", bootstrap=False | class_weight=imblanced, bootstrap=False, max_features="log2" |  class_weight=imbalanced, bootstrap=False, n_estimators=50 | class_weight=imbalanced, bootstrap=False, n_estimators=80, max_features="log2" | class_weight=imblanced, bootstrap=False, min_samples_split=4, max_features="log2" | class_weight=imblanced, bootstrap=False, min_samples_split=3, max_features="log2"| class_weight=imblanced, bootstrap=False, min_samples_split=3, max_features="log2", max_depth=20 | class_weight=imblanced, bootstrap=False, min_samples_split=3, max_features="log2", min_samples_leaf=5|
| -------- | ------- | ------- | ------- | ------- |  ------- |  ------- |  ------- |------- |------- |
| roc_auc | 0.9958320 | 0.995803 | 0.995837| 0.995396|  0.99573  |0.99576 |0.995837|0.99258643|0.995076|
| pr_auc | 0.985214 |0.985180 | 0.985201 | 0.98509 | 0.985180 |0.98520 |0.98531 |0.9768451 |0.984155|
| f1 | NA | NA | NA | NA | NA |NA| NA|0.965574|0.97235|
- final choice   
![image](https://user-images.githubusercontent.com/77596290/220822512-6a495b78-8a4f-4761-8e1f-bf2846140cda.png)

## feature importance
![image](https://user-images.githubusercontent.com/77596290/220822578-82f3919b-22b4-4b36-a81f-20117098fa09.png)
![image](https://user-images.githubusercontent.com/77596290/220822601-a2572de2-9ab5-4441-96ac-91009312c549.png)

## predictions
![image](https://user-images.githubusercontent.com/77596290/220822636-13904423-2136-40b3-a6d0-d5b6c1d0ddef.png)

## host using gradio space
- using pickle to export and import data
- host on gradio space
![Screenshot_20230222_104822](https://user-images.githubusercontent.com/77596290/220822740-9b7bd91c-5fb6-4d84-bf3e-c17a48d988b6.png)



<iframe
	src="https://aesir777-cxc-project.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>
