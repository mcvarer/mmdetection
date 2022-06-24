## Installation

- pip install torch torchvision
- pip install -U openmim
- mim install mmcv-full
- pip install -v -e .
- pip install mmdet

### bug
```
python3.6/site-packages/mim/commands/search.py'.

#From

if collection_name:
            model_info['model'] = cast2lowercase(collection_name)
            for key, value in name2collection[collection_name].items():
                model_info.setdefault(key, value)
# To

if collection_name and collection_name in name2collection.keys():
            model_info['model'] = cast2lowercase(collection_name)
            for key, value in name2collection[collection_name].items():
                model_info.setdefault(key, value)```