# manazashi

## 1.前処理 (preprocessingフォルダ)
ffmpegを使用して動画から静止画を切り出すプログラム (ffmpegのインストールが必要)
#### 1.preprocessing/split_video.py
```
--input -s 入力ファイル(.mp4 or .m4v)
--out -o   出力先フォルダ(画像の保存先)
--rate -r　 フレームレート(1sに何枚切り出すか)    
--size -s  画像の出力サイズ
```