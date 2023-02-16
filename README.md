<!--
 * @Author: Jiaxin Zheng
 * @LastEditors: Jiaxin Zheng
 * @LastEditTime: 2022-12-13 22:20:26
 * @Description: 
-->

# 4mc

(简单介绍论文)

In the project:

- 4mc:  `__init__.py` in the folder is the main program of this visual interface, run this py file to display the interface
- data:  simple data example is provided, the file is in csv format, where two columns of data are required, `data` and `name`
  - Each row of the data column stores the data as a sequence, and the name column stores the species abbreviation, including `A`,`C`,`D`,`E`,`Gsub `
- models: models used to predict results in the visualization interface

## Install

```
conda create -n 4mc python==3.9.12
conda activate 4mc
pip install -r requirements.txt

```

## Usage

```
cd M-SDBA-master\Scripts\UI\ui\4mc
python __init__.py
```

First, in the top left module, click Import to load the csv file![1670940643759](image/README/1670940643759.png)

The next module on the right-hand side allows you to select a model as needed, providing a base model, and six species enhancement models

![1670940922644](image/README/1670940922644.png)

Clicking on Predict will make the prediction and the output is displayed in the table below, while the file can be exported in csv format as needed.

![1670941055917](image/README/1670941055917.png)

The table results contain three columns of data, the original sequence, the species name and whether or not the site is included, with 0 representing no inclusion and 1 representing inclusion.



## Citation

（写上现在的文章)
