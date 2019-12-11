# manazashi

## 0.データ
動画は以下  
https://teams.microsoft.com/_#/school/files/%E4%B8%80%E8%88%AC?threadId=19%3A53f497dfc6b240b9b0a4434cb00db9d2%40thread.skype&ctx=channel&context=DCON2020%252Fmanazashi

## 1.前処理 (preprocessingフォルダ)
ffmpegを使用して動画から静止画を切り出すプログラム (ffmpegのインストールが必要)
#### 1.preprocessing/split_video.py
```
--input -s 入力ファイル(.mp4 or .m4v)
--out -o   出力先フォルダ(画像の保存先)
--rate -r　 フレームレート(1sに何枚切り出すか)    
--size -s  画像の出力サイズ
```
