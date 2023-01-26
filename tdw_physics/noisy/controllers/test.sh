# python3 noisy_dominoes.py --dir tmp_l --num 10 --height 512 --width 512 --framerate 60 --save_passes "_img" --save_movies --noise noise_low.json

# python3 noisy_dominoes.py --dir tmp_h --num 10 --height 512 --width 512 --framerate 60 --save_passes "_img" --save_movies --noise noise_high.json

# python3 noisy_dominoes.py --dir tmp_f --num 1 --height 512 --width 512 --framerate 60 --save_passes "_img" --save_movies --noise noise_float.json

python3 noisy_dominoes.py --dir tmp_h --num 100 --height 512 --width 512 --framerate 60 --noise noise_high.json --spacing_jitter 0.0 --lateral_jitter 0.0 --mrot "[0,0]"
