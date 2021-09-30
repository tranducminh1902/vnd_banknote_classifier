## Banknote classifier for Vietnam money

Banknote detection using YOLOv3 with customized weight to detect VND banknote.
The weight and config files can be download at:
- [YOLOv3 customized weights](https://drive.google.com/file/d/1a-4l-HerYbBVYQ5tVrFMFvuYDp-_Yjcq/view?usp=sharing)
- [YOLOv3 config](https://drive.google.com/file/d/1a2zFFxuRGMD0o1v6fIXOlgm7JV7idD-v/view?usp=sharing)

After detection, banknote will be predicted using trained model (9 classes labels) with base model of mobilenetv2.

### Steps:
1. Create banknote dataset using builtin webcam with [createData.py](https://github.com/tranducminh1902/vnd_banknote_classifier/blob/main/createData.py).
- Follow the folder structure of 9 classes: ['1,000 VND', '10,000 VND', '100,000 VND ', '2,000', '20,000 VND', '200,000 VND', '5,000 VND', '50,000 VND', '500,000 VND']
2. Model trained using [VND_Banknotes_Classifier_Model_for_Image.ipynb](https://github.com/tranducminh1902/vnd_banknote_classifier/blob/main/VND_Banknotes_Classifier_Model_for_Image.ipynb).
3. Export best model as .h5 files for later use: [VND_Banknotes_Classifier_Model_for_Image.h5](https://github.com/tranducminh1902/vnd_banknote_classifier/blob/main/VND_Banknotes_Classifier_Model_for_Image.h5).
4. Labeling and create custom YOLOv3 weight files for Vietnamese banknote.
5. Deploy on Streamlit using [Streamlit_Banknote_Classifier.py](https://github.com/tranducminh1902/vnd_banknote_classifier/blob/main/Streamlit_Banknote_Classifier.py).

### About Streamlit deploy:
The interface has 2 options for prediction:
- Upload static image for prediction
- Using webcam for prediction

### Contributors and references:
Contributors:
- Minh Tran: https://github.com/tranducminh1902
- Long Nguyen: https://github.com/longnguyentruong0607

References:
- Creating dataset: https://github.com/murtazahassan/OpenCV-Python-Tutorials-and-Projects/blob/master/Intermediate/Custom%20Object%20Detection/createData.py
- Train custom YOLOv3 weight file: https://pysource.com/2020/04/02/train-yolo-to-detect-a-custom-object-online-with-free-gpu/
