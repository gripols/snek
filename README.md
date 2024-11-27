# snek

snek takes an audio file, and seperates it using *nEuRaL nEtWoRkS oOoOoOo* 
and spits out vocals isolated and instrumental tracks. It has the potential to
advance past this point, but I don't feel like it.

## Getting Started

Download the latest release of this and open a
virtual environment for this.

After running `requirements.txt` you should
have a song prepared to separate.

Run the following command to get it going.

```
python inference.py --input ~/song.mp3 --gpu 0
```

If you do not want to run it with the GPU option,
get rid of the flag.