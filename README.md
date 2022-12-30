<a name="965e8e5e"></a>
# Infusing Definiteness into Randomness: Rethinking Composition Styles for Deep Image Matting 
This repository includes the official implementation of _triplet-style composition & quadruplet-style composition_, presented in our paper:<br />**Infusing Definiteness into Randomness: Rethinking Composition Styles for Deep Image Matting** <br />Proceedings of AAAI Conference on Artificial Intelligence (AAAI 2023)<br />![image.png](https://github.com/coconuthust/composition_styles/blob/main/figures/summary.png)
<a name="SezeY"></a>
## 🏃Author

👤 **Zixuan Ye1,  Yutong Dai2, Chaoyi Hong1, Zhiguo Cao1, **[**Hao Lu**](https://sites.google.com/site/poppinace/)**1**<br />🏠**1 Huazhong University of Science and Technology, China**<br />🏠**2 Australian Institute for Machine Learning, The University of Adelaide, Australia**

<a name="yYOux"></a>
## 📑More Information
[[paper](https://arxiv.org/abs/2212.13517)][[video](https://youtu.be/wmvP3YcGcSQ)]
<a name="Install"></a>
## 
<a name="Usage"></a>
## 🔆Highlights
📘The first to delve into the data generation flow and demonstrate that careful treatment can improve the performance significantly.<br />📘Explain the problem in NCF and propose a Reasonable Combination of Foregrounds (RCF)<br />📘Introduce triplet-style composition which builds the relation of source foregrounds and combined foreground.<br />📘Reveal the property of twin foregrounds and introduce quadruplet-style composition.


<a name="j6GIA"></a>
## ✔️Instructions
Our composition styles can be used in any deep matting models. To use our composition styles in your project, you only need follow the steps below:<br />**1️⃣** We need to renew the sample set each epoch, therefore follow the [**generate_index.ipynb**](https://github.com/coconuthust/composition_styles/blob/main/composition_styles/generate_index.ipynb) to obtain the foreground list, background list for each epoch. Order list can be used to control whether relevant samples will appear in the same batch or just the same sample set.<br />**2️⃣**Use the [dataloader.py](https://github.com/coconuthust/composition_styles/blob/main/composition_styles/data_generator.py) to generate the dataset with the selected composition styles. The changes are made from [L478](https://github.com/coconuthust/composition_styles/blob/main/data_generator.py#L478) to L612.<br />**3️⃣**Modify your own training code to load the sample set with generated indexes each epoch.(The code below is an example)
```python
for epoch in range(start_epoch, cfg.TRAIN.num_epochs):
    if cfg.DATASET.composition_style is not None:
        back_list = np.load(file = 'backlist.npy')
        fore_list = np.load(file = 'forelist.npy')
        order_list = np.load(file = 'orderlist.npy') 
        backlist = back_list[epoch]
        forelist = fore_list[epoch]
        trainset = dataset(cfg, phase='train', test_scale='origin', crop_size=cfg.TRAIN.crop_size, back_list = backlist, fore_list = forelist, order_list=order_list)

```

<a name="7b43efb3"></a>
## 📚Reported Results
<a name="Sfsc6"></a>
### Effectiveness on four baselines
![image.png](https://cdn.nlark.com/yuque/0/2022/png/34812589/1672210878568-dd9da735-da58-46f2-b463-14cfc6410712.png#averageHue=%23f0efed&clientId=u0835b2f4-a47b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=130&id=u70c158eb&margin=%5Bobject%20Object%5D&name=image.png&originHeight=346&originWidth=1927&originalType=binary&ratio=1&rotation=0&showTitle=false&size=168876&status=done&style=none&taskId=ud5f0b385-f0b2-4a2e-9f2d-f38b9f7afae&title=&width=722)

---
## Citation
If you find this work or code useful for your research, please cite:
```
@inproceedings{ye2023infusing,
  title={Infusing Definiteness into Randomness: Rethinking Composition Styles for Deep Image Matting},
  author={Ye, Zixuan and Dai, Yutong and Hong, Chaoyi and Cao, Zhiguo and Lu, Hao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

## Permission
This code is for academic purposes only. Contact: Zixuan Ye (yezixuan@hust.edu.cn)


<a name="cARQw"></a>
## Reference
[_IndexNet Matting_](https://github.com/poppinace/indexnet_matting)<br />[_GCA Matting_](https://github.com/Yaoyi-Li/GCA-Matting)<br />[_A2U Matting_](https://github.com/dongdong93/a2u_matting)<br />[_Matteformer_](https://github.com/webtoon/matteformer)

