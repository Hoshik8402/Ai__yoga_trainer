# Ai__yoga_trainer
- Use Mediapipe to extract keypoint to file csv then use LSTM model to train data. Next use this model to predict user's yoga posture.
- Find angle between 2 vectors which endpoints is 2 keypoint in skeleton.

You can find data of yoga posture by find image on Internet or use this link: https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset then run file yoga_pose_prediction_using_mediapipe.ipynb to extract keypoint to file csv and test model on image for fun....
If you find out that your data extract from image is not enought, you can run file make_data.py to get your own data.
Next, run file train_lstm.py to train your model.
After that, run file real_time_detection.py to predict your yoga pose.
File user_interfaces.py will give you feed back based on image you provide and compare it and your posture.
