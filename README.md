To install all the required packages:

```
pip install -r requirements.txt
```

pip3 might also be needed depending on your system set up.

Setting a virtualenviroment is recommended.


###########################################################
###HOW TO USE CLASSIFY.PY###
###########################################################


If the model takes only one input then:
```
python3 classify.py {model_path} {image} 
```

If the model(combined CNN) takes 2 inputs:
```
python3 classify.py {model_path} {image_1} {image_2}
```
