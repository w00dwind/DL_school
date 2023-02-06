# Физтех-Школа Прикладной математики и информатики (ФПМИ) МФТИ
# Journey to Springfield
![alt text](https://vignette.wikia.nocookie.net/simpsons/images/5/5a/Spider_fat_piglet.png/revision/latest/scale-to-width-down/640?cb=20111118140828)

Проект по классификации персонажей симпосонов. 
* Устранение дисбаланса классов, используя парсинг изображений, вычисление весов каждого класса и использования weightedrandomsampler для даталоадера.
* Используются две архитектуры - ResNext101 и Efficientnetb4. Сравниваются результаты обучения (loss, f1).
* В обеих архитектурах используются предобученные веса imagenetv2. 
* Используется техника fine tune - на первом этапе размораживается только классификатор, затем неколько слоёв перед классификатором. 

# Kaggle submit
<img src="https://github.com/w00dwind/DL_school/blob/main/20_Simpsons/data/kaggle_submit.png" alt="submit" width="60%" height="60%"></img>

* Метрика соревнования - f1. Наилучший результат показала архитектура ResNext101. 
 
 | Architectire | max train f1 | max val f1 |
 | ------------- | ------- | ------ | 
 | efficientnetb4 | 0.9327 | 0.8677 |
 | resnext101 | 0.9750 | 0.9172 | 

 
<img src="https://github.com/w00dwind/DL_school/blob/main/20_Simpsons/data/f1.png" alt= “f1” width="60%" height="60%"></img>
 
 | Architectire | min train loss | min val loss |
 | ------------- | ------- | ------ | 
 | efficientnetb4 | 0.2499 | 0.4939 |
 | resnext101 | 0.0908 | 0.3358 | 
 
 <img src="https://github.com/w00dwind/DL_school/blob/main/20_Simpsons/data/loss.png" alt="loss" width="60%" height="60%"></img>
