"""CPU-only checkpoint readability, model/Adam initialization and final restore."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from model import graph


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--end-step',type=int,choices=[5000,10000],default=5000)
    a=p.parse_args();assert not tf.config.list_physical_devices('GPU'), 'This read-only audit runs on CPU'
    result=[]
    for model in ('edge','S','L'):
        g=graph(model)
        for arm in ('fc2_only','mixture75_25'):
            d=a.run/'seed42'/arm/model/'replay'
            initial=tf.train.load_checkpoint(str(d/'step-000000/model'))
            source=tf.train.load_checkpoint(str(a.run/'source'/model/'fc2/step-010000/model'))
            assert all(np.array_equal(initial.get_tensor(v.op.name),source.get_tensor(v.op.name)) for v in g['weights'])
            assert int(initial.get_tensor('global_step'))==0
            assert all(np.all(initial.get_tensor(n)==0) for n in initial.get_variable_to_shape_map() if '/Adam' in n)
            np.testing.assert_allclose([initial.get_tensor('beta1_power'),initial.get_tensor('beta2_power')],[.9,.999],rtol=0,atol=1e-7)
            names={v.op.name:v.shape.as_list() for v in tf.compat.v1.global_variables()}
            for step in range(0,a.end_step+1,1000):
                reader=tf.train.load_checkpoint(str(d/f'step-{step:06d}/model'))
                assert reader.get_variable_to_shape_map()==names
                assert int(reader.get_tensor('global_step'))==step
                assert all(np.isfinite(reader.get_tensor(n)).all() for n in names)
                assert all(np.all(reader.get_tensor(n)>=0) for n in names if '/moving_variance' in n)
                del reader
            with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=1)) as sess:
                g['saver'].restore(sess,str(d/f'step-{a.end_step:06d}/model'))
                reader=tf.train.load_checkpoint(str(d/f'step-{a.end_step:06d}/model'))
                assert all(np.array_equal(sess.run(v),reader.get_tensor(v.op.name)) for v in tf.compat.v1.global_variables())
            result.append(dict(model=model,arm=arm,six_checkpoints_finite=True,source_model_bn_exact=True,
                               adam_initial_reset=True,final_all_variables_restored_exact=True,step=a.end_step))
            del initial,source,reader
    out=dict(passed=True,cpu_only=True,checkpoint_count=6*(a.end_step//1000+1),tensorflow=tf.__version__,runs=result)
    folder=a.run/'control' if a.end_step==5000 else a.run/'control/continue10k'
    (folder/'checkpoints-verified.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
