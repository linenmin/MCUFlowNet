"""Trace actual S/L training ops on a cached real FC2 batch; not an accuracy experiment."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'EdgeFlowNAS'))
from efnas.engine.distill_or_not_trainer import _build_graph
from efnas.data.fc2_dataset import FC2BatchProvider
from efnas.data.transforms_180x240 import standardize_image_tensor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    tf.compat.v1.disable_eager_execution()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    paths=sorted(Path('/datasets/FlyingChairs2/val').glob('*-img_0.png'))[:2]
    provider=FC2BatchProvider([str(p) for p in paths],352,480,crop_mode='center',sampling_mode='sequential')
    batch,_,_,labels=provider.next_batch(2)
    result={'purpose':'Device verification on cached FC2 validation images; updates discarded, not a trained candidate', 'batch_size':2,'shape':[352,480], 'models':{}}
    for scope,code in [('v3_light',[0]*11),('v3_efn_fps',[2,0,0,2,2,1,0,0,0,0,0])]:
        graph=tf.Graph()
        with graph.as_default():
            x=tf.compat.v1.placeholder(tf.float32,[2,352,480,6])
            y=tf.compat.v1.placeholder(tf.float32,[2,352,480,2])
            model=_build_graph(scope,code,x,y,tf.constant(1e-4),tf.constant(1.),tf.constant(True),2,4,0.,200.)
            init=tf.compat.v1.global_variables_initializer()
        cfg=tf.compat.v1.ConfigProto();cfg.gpu_options.allow_growth=True
        with tf.compat.v1.Session(graph=graph,config=cfg) as sess:
            sess.run(init)
            feed={x:standardize_image_tensor(batch),y:labels}
            for _ in range(3):
                sess.run(model['zero_grad_op']);sess.run(model['accum_op'],feed);sess.run(model['train_op'])
            traces={}
            for phase,op in [('forward_backward',model['accum_op']),('optimizer',model['train_op'])]:
                metadata=tf.compat.v1.RunMetadata()
                sess.run(op,feed,options=tf.compat.v1.RunOptions(output_partition_graphs=True),run_metadata=metadata)
                traces[phase]={}
                for partition in metadata.partition_graphs:
                    for node in partition.node:
                        traces[phase].setdefault(node.device,[]).append(node.name+' ['+node.op+']')
            started=time.perf_counter()
            cached_steps=0
            while time.perf_counter()-started < 20:
                cached_steps+=1
                sess.run(model['zero_grad_op']);sess.run(model['accum_op'],feed);sess.run(model['train_op'])
            seconds=time.perf_counter()-started
        gpu_nodes=[n for devices in traces.values() for device,nodes in devices.items() if 'GPU' in device.upper() for n in nodes]
        assert gpu_nodes, 'No actual GPU nodes recorded'
        result['models'][scope]={'traces':traces,'cached_steps':cached_steps,'cached_steps_seconds':seconds,'gpu_node_entries':len(gpu_nodes)}
        print(scope, 'GPU node entries:',len(gpu_nodes),'cached steps:',cached_steps,'seconds:',seconds,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
