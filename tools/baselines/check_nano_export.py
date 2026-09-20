"""Check public NanoFlowNet H5 against its public float TFLite export."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from evaluate import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    model=tf.keras.models.load_model(str(args.models/'nanoflownet.h5'),compile=False)
    interpreter=tf.lite.Interpreter(model_path=str(args.models/'nanoflownet_unquantized.tflite'),num_threads=4)
    fn=interpreter.get_signature_runner('serving_default')
    rows=[]
    for scene,frame in [('alley_1',1),('ambush_4',5)]:
        images=[]
        for i in (frame,frame+1):
            image=cv2.imread(str(args.dataset/'training/final'/scene/f'frame_{i:04d}.png'))[10:426]
            gray=cv2.cvtColor(cv2.resize(image,(160,112)),cv2.COLOR_BGR2GRAY).astype('float32')
            images.append(((gray-128)/128)[None,:,:,None])
        a=model(np.concatenate(images,-1),training=False)[0].numpy()
        b=fn(input_1=images[0],input_2=images[1])['model']
        assert a.shape==b.shape==(1,28,40,2)
        rows.append(dict(scene=scene,frame=frame,max_abs_difference=float(np.max(np.abs(a-b)))))
    data=dict(comparisons=rows,weights={p.name:sha(p) for p in [args.models/'nanoflownet.h5',args.models/'nanoflownet_unquantized.tflite']},
              note='Checks exported prediction correspondence only, not physical flow units or paper evaluation protocol.')
    args.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__':
    main()
